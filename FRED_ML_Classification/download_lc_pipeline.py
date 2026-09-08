#!/usr/bin/env python3
"""
download_lc_pipeline.py
-----------------------
Download Fermi GBM lightcurves for multiple GRBs and save them as PNG/JPEG
images for visual FRED (Fast Rise Exponential Decay) profile screening.

Pipeline steps for each GRB:
    1. Discover GRBs from the Fermi GBM burst catalog (or use a provided list)
    2. Download TTE + position-history data via heapyx/gbmRetrieve
    3. Extract time-integrated spectra and lightcurve products
    4. Bin raw TTE photon times into lightcurve arrays
    5. Save images:
        - one image per detector
        - one combined multi-detector image per GRB
        - optional single-page summary grid of all GRBs

Usage (inside Docker container):
    # Discover GRBs from the Fermi catalog
    python download_lc_pipeline.py --start 2019-01-01 --end 2019-01-31 \
        --output /workspace/data/lightcurve_images --fmt png

    # Use your own GRB list (CSV with columns: name, ra, dec, utc, sel_dets, t1, t2)
    python download_lc_pipeline.py --csv my_grbs.csv --output ./lc_images

The script relies on sibling modules in this directory:
    grb_discovery.py, grb_pipeline_extras.py, grb_utils.py, fred_scorer.py
and on heapyx/bayspec being installed in the container environment.
"""

import os
import sys
import argparse
import glob
import traceback
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(__file__))

import grb_utils as gu
from grb_discovery import query_gbm_catalog, build_grbs_df
from grb_utils import load_grb, extract_tintegrated_spectra
from grb_pipeline_extras import extract_arrays_for_current_grb
from fred_scorer import score_burst


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_fmt(fmt):
    fmt = fmt.lower().lstrip(".")
    if fmt not in {"png", "jpeg", "jpg"}:
        raise ValueError(f"Unsupported image format: {fmt}. Use png or jpeg.")
    return fmt


def _load_lc_array(path):
    arr = np.load(path)
    return arr[:, 0], arr[:, 1]


def _det_from_lc_path(grb_name, path):
    return os.path.basename(path).replace(f"{grb_name}_", "").replace("_lc.npy", "")


def arrays_dir_for_grb(grb_name, data_base=None):
    data_base = data_base or gu.DATA_BASE
    return os.path.join(data_base, grb_name, "data", "tintegrated", "arrays")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_detector_lightcurve(
    grb_name, det, t, rate, output_dir, fmt="png", figsize=(8, 3.5), dpi=150
):
    """Save a single-detector lightcurve image."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)
    ax.step(t, rate, where="mid", color="steelblue", lw=0.8)
    ax.set_xlabel("Time since trigger (s)")
    ax.set_ylabel("Rate (counts/s)")
    ax.set_title(f"{grb_name} — {det}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(output_dir, f"{grb_name}_{det}_lc.{fmt}")
    fig.savefig(out, dpi=dpi, format=fmt, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_combined_lightcurves(
    grb_name, arrays_dir, output_dir, fmt="png", figsize=(10, 8), dpi=150
):
    """
    Create a stacked multi-detector lightcurve figure for one GRB from the
    per-detector *_lc.npy arrays.
    """
    npy_paths = sorted(glob.glob(os.path.join(arrays_dir, f"{grb_name}_*_lc.npy")))
    if not npy_paths:
        return None

    dets, all_t, all_rate = [], [], []
    global_tmin, global_tmax = np.inf, -np.inf

    for path in npy_paths:
        det = _det_from_lc_path(grb_name, path)
        t, rate = _load_lc_array(path)
        dets.append(det)
        all_t.append(t)
        all_rate.append(rate)
        if len(t):
            global_tmin = min(global_tmin, t.min())
            global_tmax = max(global_tmax, t.max())

    n = len(dets)
    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True, squeeze=False)
    axes = axes[:, 0]

    for ax, det, t, rate in zip(axes, dets, all_t, all_rate):
        ax.step(t, rate, where="mid", color="steelblue", lw=0.7)
        ax.text(
            0.02,
            0.92,
            det,
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="wheat", alpha=0.8),
        )
        ax.set_ylabel("Rate (cts/s)")
        ax.grid(True, alpha=0.3)
        if np.isfinite(global_tmin) and np.isfinite(global_tmax):
            ax.set_xlim(global_tmin, global_tmax)

    axes[-1].set_xlabel("Time since trigger (s)")
    fig.suptitle(
        f"{grb_name} — multi-detector lightcurves", fontsize=12, fontweight="bold"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"{grb_name}_combined_lc.{fmt}")
    fig.savefig(out, dpi=dpi, format=fmt, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_summary_grid(grb_names, data_base, output_path, ncols=5, fmt="png", dpi=150):
    """
    Create one large grid image showing the brightest detector lightcurve for
    each GRB. Useful for a quick visual FRED scan across the whole sample.
    """
    rows = []
    for grb_name in grb_names:
        npy_paths = glob.glob(
            os.path.join(
                arrays_dir_for_grb(grb_name, data_base), f"{grb_name}_*_lc.npy"
            )
        )
        best = None
        best_peak = -np.inf
        for path in npy_paths:
            t, rate = _load_lc_array(path)
            if len(rate) == 0:
                continue
            peak = np.nanmax(rate)
            if peak > best_peak:
                best_peak = peak
                best = (grb_name, path, t, rate)
        if best:
            rows.append(best)

    if not rows:
        print("No lightcurves available for summary grid.")
        return None

    n = len(rows)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 2.2 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, (grb_name, path, t, rate) in zip(axes, rows):
        det = _det_from_lc_path(grb_name, path)
        ax.step(t, rate, where="mid", lw=0.6, color="steelblue")
        ax.set_title(f"{grb_name}\n{det}", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        "FRED Screening Grid (brightest detector per GRB)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, format=fmt, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Per-GRB processing
# ---------------------------------------------------------------------------


def process_grb(
    row,
    output_dir,
    fmt="png",
    binsize=0.064,
    pre=-20,
    post=50,
    combined=True,
    score=True,
):
    """
    Download data and produce image products for one GRB.

    Returns a dict with status, saved image paths and optional FRED scores.
    """
    fmt = _ensure_fmt(fmt)
    name = row["name"]

    load_grb(row)
    extract_tintegrated_spectra()
    extract_arrays_for_current_grb(binsize=binsize, pre=pre, post=post)

    arr_dir = arrays_dir_for_grb(name)
    grb_output = os.path.join(output_dir, name)
    os.makedirs(grb_output, exist_ok=True)

    saved = {"single": [], "combined": None, "status": "ok", "scores": []}

    for npy_path in sorted(glob.glob(os.path.join(arr_dir, f"{name}_*_lc.npy"))):
        t, rate = _load_lc_array(npy_path)
        det = _det_from_lc_path(name, npy_path)
        out = plot_detector_lightcurve(name, det, t, rate, grb_output, fmt=fmt)
        saved["single"].append(out)

        if score:
            metrics = score_burst(t, rate)
            if metrics:
                saved["scores"].append(
                    {
                        "grb_name": name,
                        "det": det,
                        **metrics,
                    }
                )

    if combined:
        saved["combined"] = plot_combined_lightcurves(
            name, arr_dir, grb_output, fmt=fmt
        )

    return saved


# ---------------------------------------------------------------------------
# Batch pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    grbs_df,
    output_dir,
    fmt="png",
    binsize=0.064,
    pre=-20,
    post=50,
    combined=True,
    summary=True,
    score=True,
    log_path="download_lc_errors.log",
):
    """
    Run the full image-generation pipeline over a grbs_df DataFrame.

    Returns (results_list, summary_grid_path).
    """
    fmt = _ensure_fmt(fmt)
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, log_path)

    results = []
    with open(log_path, "a") as logf:
        for _, row in tqdm(grbs_df.iterrows(), total=len(grbs_df), desc="GRBs"):
            name = row["name"]
            try:
                res = process_grb(
                    row,
                    output_dir,
                    fmt=fmt,
                    binsize=binsize,
                    pre=pre,
                    post=post,
                    combined=combined,
                    score=score,
                )
                res["grb_name"] = name
                results.append(res)
                if res["scores"]:
                    best = max(res["scores"], key=lambda x: x["fred_score"])
                    print(
                        f"  {name} best FRED score: {best['fred_score']:.3f} ({best['det']})"
                    )
            except Exception as e:
                msg = f"{name}: {type(e).__name__}: {e}"
                print(f"  FAILED — {msg}")
                logf.write(msg + "\n")
                logf.write(traceback.format_exc() + "\n")
                results.append({"grb_name": name, "status": "failed", "error": msg})

    summary_path = None
    if summary:
        ok_names = [r["grb_name"] for r in results if r.get("status") == "ok"]
        summary_path = os.path.join(output_dir, f"FRED_screening_grid.{fmt}")
        plot_summary_grid(ok_names, gu.DATA_BASE, summary_path, fmt=fmt)

    return results, summary_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Download GRB lightcurves and save as PNG/JPEG images for FRED screening."
    )
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--csv", help="Path to CSV with grbs_df columns")
    parser.add_argument(
        "--output",
        "-o",
        default="/workspace/data/lightcurve_images",
        help="Output directory for images",
    )
    parser.add_argument(
        "--fmt", default="png", choices=["png", "jpeg", "jpg"], help="Image format"
    )
    parser.add_argument(
        "--binsize", type=float, default=0.064, help="Lightcurve bin size in seconds"
    )
    parser.add_argument(
        "--pre", type=float, default=-20, help="Seconds before trigger to include"
    )
    parser.add_argument(
        "--post", type=float, default=50, help="Seconds after trigger to include"
    )
    parser.add_argument(
        "--no-combined", action="store_true", help="Skip combined multi-detector plots"
    )
    parser.add_argument(
        "--no-summary", action="store_true", help="Skip summary grid image"
    )
    parser.add_argument(
        "--no-score", action="store_true", help="Skip FRED-score printing"
    )

    args = parser.parse_args()

    if args.csv:
        grbs_df = pd.read_csv(args.csv)
        for col in ["sel_dets", "time_slices"]:
            if col in grbs_df.columns:
                grbs_df[col] = grbs_df[col].apply(eval)
    elif args.start and args.end:
        raw = query_gbm_catalog(args.start, args.end)
        grbs_df = build_grbs_df(raw)
    else:
        parser.error("Provide either --start/--end dates or --csv.")

    if grbs_df.empty:
        print("No GRBs to process.")
        return

    print(f"Processing {len(grbs_df)} GRBs -> {args.output}")
    results, summary = run_pipeline(
        grbs_df,
        output_dir=args.output,
        fmt=args.fmt,
        binsize=args.binsize,
        pre=args.pre,
        post=args.post,
        combined=not args.no_combined,
        summary=not args.no_summary,
        score=not args.no_score,
    )

    ok = sum(1 for r in results if r.get("status") == "ok")
    failed = sum(1 for r in results if r.get("status") == "failed")
    print(f"\nDone: {ok} ok, {failed} failed.")
    if summary:
        print(f"Summary grid: {summary}")


if __name__ == "__main__":
    main()
