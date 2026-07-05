#!/usr/bin/env python3
"""
Step 2: Extract lightcurve features + manually label FRED vs non-FRED.

Run:
    python step2_features_label.py

Outputs (in ./grb_data/):
    labels.json        – your manual FRED labels {trigger_name: 0/1}
    features.csv       – numeric features for every labeled burst

How labeling works:
    - Each lightcurve PNG is displayed.
    - Type 1 (FRED), 0 (not FRED), s (skip), q (quit and save).
    - Re-running the script picks up where you left off.
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
# Use interactive backend if a display is available, otherwise fall back to Agg
import os
if os.environ.get('DISPLAY') or os.environ.get('MPLBACKEND'):
    try:
        matplotlib.use('TkAgg')
    except Exception:
        matplotlib.use('Agg')
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis

OUTDIR = '/workspace/data/grb_data'

FEATURE_COLS = [
    'asymmetry',    # rise / (rise + decay)   — FRED → near 0
    'n_peaks',      # number of significant peaks  — FRED → 1
    'skewness',     # shape of burst region    — FRED → positive
    'kurtosis',     # peakedness
    'exp_tau',      # exponential decay timescale (s)
    'rise_time',    # seconds
    'decay_time',   # seconds
    't90_local',    # T90 computed from lightcurve (s)
    't90_catalog',  # T90 from GBM catalog (s)
]


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════
def background_subtract(t: np.ndarray, rate: np.ndarray,
                         pre: tuple = (-60, -5),
                         post: tuple = (120, 200)) -> np.ndarray:
    """
    Subtract a linear (degree-1) background fit using pre/post-burst windows.
    Falls back to median if there aren't enough background bins.
    """
    bg_mask = ((t >= pre[0])  & (t <= pre[1])) | \
              ((t >= post[0]) & (t <= post[1]))

    if bg_mask.sum() < 5:
        # Not enough background; use median of edges
        edge = np.concatenate([rate[:8], rate[-8:]])
        return rate - np.median(edge)

    coeffs = np.polyfit(t[bg_mask], rate[bg_mask], deg=1)
    return rate - np.polyval(coeffs, t)


def compute_features(t: np.ndarray, rate: np.ndarray,
                     t90_catalog: float = None) -> dict | None:
    """
    Compute FRED-relevant shape features from a background-subtracted
    lightcurve. Returns None if the burst is too faint or too short.
    """
    src = np.clip(background_subtract(t, rate), 0, None)

    # ── Noise estimate from pre-trigger bins ─────────────────────────────────
    pre_mask = t < -10
    noise = np.std(src[pre_mask]) if pre_mask.sum() > 5 else src.max() * 0.05
    noise = max(noise, 1e-6)

    if src.max() < 3 * noise:
        return None          # burst too faint

    # ── Cumulative fluence → T5, T95 ─────────────────────────────────────────
    dt       = np.diff(t, prepend=t[0])
    fluence  = np.cumsum(src * dt)
    if fluence[-1] <= 0:
        return None
    cum_norm = fluence / fluence[-1]

    i5  = np.searchsorted(cum_norm, 0.05)
    i95 = np.searchsorted(cum_norm, 0.95)
    t5, t95 = t[min(i5,  len(t)-1)], t[min(i95, len(t)-1)]
    t90_local = float(t95 - t5)

    # ── Peak location ─────────────────────────────────────────────────────────
    i_peak = int(np.argmax(src))
    t_peak = t[i_peak]

    rise_time  = max(float(t_peak - t5),  0.01)
    decay_time = max(float(t95 - t_peak), 0.01)
    asymmetry  = rise_time / (rise_time + decay_time)

    # ── Number of significant peaks ───────────────────────────────────────────
    smoothed   = gaussian_filter1d(src, sigma=3)
    peaks, _   = find_peaks(smoothed, height=3 * noise, distance=5,
                             prominence=noise)
    n_peaks = int(len(peaks))

    # ── Shape statistics in burst window ─────────────────────────────────────
    burst_mask = (t >= t5) & (t <= t95)
    if burst_mask.sum() < 4:
        return None
    burst_src  = src[burst_mask]
    skewness_v = float(skew(burst_src))
    kurtosis_v = float(kurtosis(burst_src))

    # ── Exponential decay timescale ───────────────────────────────────────────
    decay_mask = (t > t_peak) & (t < t95) & (src > noise)
    if decay_mask.sum() > 5:
        log_src = np.log(np.clip(src[decay_mask], 1e-3, None))
        coeffs  = np.polyfit(t[decay_mask], log_src, 1)
        exp_tau = float(-1.0 / coeffs[0]) if coeffs[0] < 0 else 999.0
    else:
        exp_tau = 999.0

    return {
        'asymmetry':    asymmetry,
        'n_peaks':      n_peaks,
        'skewness':     skewness_v,
        'kurtosis':     kurtosis_v,
        'exp_tau':      min(exp_tau, 999.0),
        'rise_time':    rise_time,
        'decay_time':   decay_time,
        't90_local':    t90_local,
        't90_catalog':  float(t90_catalog) if t90_catalog else t90_local,
    }


def load_ctime(fits_path: str):
    """Read CTIME FITS → (time_centers, rate)."""
    with fits.open(fits_path) as hdul:
        spec     = hdul['SPECTRUM'].data
        trigtime = hdul[0].header.get('TRIGTIME', 0.0)
        t        = 0.5 * (spec['TIME'] + spec['ENDTIME']) - trigtime
        counts   = spec['COUNTS']
        exposure = spec['EXPOSURE']
    rate = counts[:, 1:].sum(axis=1) / np.clip(exposure, 1e-4, None)
    return t, rate


# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE LABELER
# ═══════════════════════════════════════════════════════════════════════════════
def label_bursts(manifest: list, label_file: str) -> dict:
    """
    Show each lightcurve PNG and collect a FRED label from the user.
    Saves incrementally so you can quit and resume.

    Keys: 1 = FRED   0 = not FRED   s = skip   q = quit & save
    """
    # Load any existing labels
    labels = {}
    if os.path.exists(label_file):
        with open(label_file) as f:
            labels = json.load(f)
        print(f"Loaded {len(labels)} existing labels from {label_file}")

    to_do = [m for m in manifest if m['trigger_name'] not in labels]
    print(f"\nBursts to label: {len(to_do)}")
    print("Keys:  1=FRED   0=not-FRED   s=skip   q=quit+save\n")

    fig = plt.figure(figsize=(10, 3.5))

    for i, entry in enumerate(to_do):
        trig = entry['trigger_name']
        png  = entry.get('png', '')

        # Show PNG
        fig.clear()
        ax = fig.add_subplot(111)
        if png and os.path.exists(png):
            img = mpimg.imread(png)
            ax.imshow(img)
            ax.axis('off')
        ax.set_title(
            f"[{i+1}/{len(to_do)}]  {trig}  "
            f"T90={entry['t90']:.1f}s",
            fontsize=11, fontweight='bold'
        )
        plt.tight_layout()
        plt.pause(0.05)

        # Collect label
        while True:
            ans = input(f"  {trig}: [1/0/s/q] → ").strip().lower()
            if ans in ('1', '0'):
                labels[trig] = int(ans)
                tag = 'FRED ✓' if ans == '1' else 'not FRED ✗'
                print(f"    → {tag}")
                break
            elif ans == 's':
                print("    → skipped")
                break
            elif ans == 'q':
                plt.close()
                _save_labels(labels, label_file)
                return labels
            else:
                print("    Type 1, 0, s, or q")

    plt.close()
    _save_labels(labels, label_file)
    return labels


def _save_labels(labels: dict, path: str):
    with open(path, 'w') as f:
        json.dump(labels, f, indent=2)
    n_fred = sum(v == 1 for v in labels.values())
    print(f"\nSaved {len(labels)} labels ({n_fred} FRED, "
          f"{len(labels)-n_fred} non-FRED) → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# BULK FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════
def extract_all_features(manifest: list, labels: dict) -> pd.DataFrame:
    """
    Extract numeric features for every labeled burst.
    Returns a DataFrame and saves features.csv.
    """
    records = []
    n_fail  = 0

    for entry in manifest:
        trig = entry['trigger_name']
        if trig not in labels:
            continue

        try:
            t, rate = load_ctime(entry['fits'])
            mask    = (t > -60) & (t < 200)
            feats   = compute_features(t[mask], rate[mask],
                                       t90_catalog=entry.get('t90'))
        except Exception as e:
            n_fail += 1
            continue

        if feats is None:
            n_fail += 1
            continue

        feats['trigger_name'] = trig
        feats['label']        = labels[trig]
        feats['t90_catalog']  = entry.get('t90', feats['t90_local'])
        records.append(feats)

    df = pd.DataFrame(records)
    out = f'{OUTDIR}/features.csv'
    df.to_csv(out, index=False)

    print(f"\nFeatures extracted for {len(df)} bursts  ({n_fail} failed)")
    print(f"Saved → {out}")
    print(f"\nClass balance:")
    print(df['label'].value_counts().rename({0: 'non-FRED', 1: 'FRED'}))
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    manifest_path = f'{OUTDIR}/manifest.json'
    label_path    = f'{OUTDIR}/labels.json'

    if not os.path.exists(manifest_path):
        print("ERROR: manifest.json not found. Run step1_download.py first.")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)
    print(f"Manifest: {len(manifest)} bursts")

    # ── Label ─────────────────────────────────────────────────────────────────
    labels = label_bursts(manifest, label_path)

    if len(labels) < 5:
        print("Label at least 5 bursts before training. Re-run to continue.")
        return

    # ── Extract features ──────────────────────────────────────────────────────
    df = extract_all_features(manifest, labels)

    # Quick peek at the features
    print("\nSample features:")
    cols_to_show = ['trigger_name', 'label', 'asymmetry',
                    'n_peaks', 'skewness', 't90_local']
    print(df[cols_to_show].to_string(index=False))

    print("\nNext: run  python step3_train.py")


if __name__ == '__main__':
    main()
