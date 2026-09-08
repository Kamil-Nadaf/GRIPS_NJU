#!/usr/bin/env python3
"""
Plot lightcurves for a GRB — one subplot per detector, stacked vertically
with a shared x-axis.

Usage:
    # CLI (standalone):
    python plot_lightcurves.py GRB131011A

    # Notebook (after load_grb):
    from plot_lightcurves import plot_lightcurves
    plot_lightcurves(GRB_Name, data_base=DATA_BASE)

Lightcurve files are expected at:
    {DATA_BASE}/{GRB_NAME}/data/tintegrated/heapy/<det>/lc.html
"""

import os
import re
import json
import base64
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DATA_BASE = '/workspace/data'

# heapy trace names → plot style
_TRACE_STYLES = {
    'source lightcurve':      {'color': '#1f77b4', 'label': 'Source',      'alpha': 0.85, 'lw': 1.0},
    'background lightcurve':  {'color': '#d62728', 'label': 'Background',  'alpha': 0.70, 'lw': 0.8},
    'total lightcurve':       {'color': '#2ca02c', 'label': 'Total',       'alpha': 0.85, 'lw': 1.0},
}

_FALLBACK_STYLE = {'color': '#333333', 'label': 'Unknown', 'alpha': 0.8, 'lw': 0.8}

# ---------------------------------------------------------------------------
# Plotly typed-array decoding
# ---------------------------------------------------------------------------

def decode_typed_array(d):
    """Decode a Plotly typed-array dict ``{'dtype': str, 'bdata': base64}``.

    Returns a numpy ndarray.  Raises ``ValueError`` if required keys are
    missing or the dtype string is unknown.
    """
    if not isinstance(d, dict):
        raise TypeError(f"Expected a dict with 'dtype' and 'bdata', got {type(d).__name__}")
    bdata = d.get('bdata')
    dtype_str = d.get('dtype')
    if bdata is None or dtype_str is None:
        raise ValueError(f"Typed array dict missing 'bdata' or 'dtype' key: {list(d.keys())}")
    try:
        raw = base64.b64decode(bdata)
        return np.frombuffer(raw, dtype=dtype_str)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid dtype '{dtype_str}' or corrupted bdata") from exc


# ---------------------------------------------------------------------------
# Plotly HTML parser
# ---------------------------------------------------------------------------

# Matches Plotly.newPlot("id", [...data...], {...layout...})
_PLOTLY_RE = re.compile(
    r'Plotly\.newPlot\(\s*"[^"]+",\s*(\[.*?\]),\s*(\{.*?\})',
    re.DOTALL,
)


def parse_plotly_html(path):
    """Parse a heapy-generated Plotly HTML lightcurve file.

    Returns a list of trace dicts.  Each dict has keys: ``name``, ``type``,
    ``x`` (ndarray), ``y`` (ndarray), and optionally ``error_y`` (ndarray).
    """
    with open(path) as f:
        html = f.read()

    match = _PLOTLY_RE.search(html)
    if not match:
        raise ValueError(f"Could not parse Plotly data from {path}")

    data_raw = json.loads(match.group(1))
    traces = []

    for tr in data_raw:
        trace = {
            'name': tr.get('name', ''),
            'type': tr.get('type', 'scatter'),
            'x': decode_typed_array(tr['x']),
            'y': decode_typed_array(tr['y']),
        }

        # error_y (optional) — may be a typed-array dict or a plain list
        if 'error_y' in tr and tr['error_y'] is not None:
            ey = tr['error_y']
            if isinstance(ey, dict) and 'array' in ey:
                trace['error_y'] = decode_typed_array(ey['array'])
            elif isinstance(ey, (list, np.ndarray)):
                trace['error_y'] = np.asarray(ey, dtype=float)
            # else: omit error_y silently

        traces.append(trace)

    return traces


# ---------------------------------------------------------------------------
# Detector discovery
# ---------------------------------------------------------------------------

def find_detectors(data_base, grb_name):
    """Return sorted detector names that have ``lc.html`` files.

    Returns an empty list (no error) if the directory tree is missing.
    """
    base = os.path.join(data_base, grb_name, 'data', 'tintegrated', 'heapy')
    if not os.path.isdir(base):
        return []
    dets = []
    for entry in sorted(os.listdir(base)):
        lc_path = os.path.join(base, entry, 'lc.html')
        if os.path.isfile(lc_path):
            dets.append(entry)
    return dets


def load_detector_data(data_base, grb_name, det):
    """Load lightcurve traces for a single detector."""
    path = os.path.join(data_base, grb_name, 'data', 'tintegrated', 'heapy', det, 'lc.html')
    return parse_plotly_html(path)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_lightcurves(grb_name, data_base=DEFAULT_DATA_BASE, save_path=None,
                     figsize=(10, 8)):
    """Plot GRB lightcurves — one subplot per detector, stacked vertically.

    Parameters
    ----------
    grb_name : str
        GRB name, e.g. ``'GRB131011A'``.
    data_base : str
        Root data directory (default ``/workspace/data``).
    save_path : str or None
        If given, save figure to this path.  Otherwise ``plt.show()``.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    fig, axes
    """
    # --- discover detectors ---
    dets = find_detectors(data_base, grb_name)
    if not dets:
        print(f"No detector lightcurves found for {grb_name} under {data_base}")
        return None, None

    print(f"Found {len(dets)} detectors: {dets}")

    # --- load data & compute global x-range in one pass ---
    all_data = {}
    global_x_min, global_x_max = np.inf, -np.inf

    for det in dets:
        traces = load_detector_data(data_base, grb_name, det)
        # keep only plottable scatter traces (skip bars, empty)
        plottable = []
        for tr in traces:
            if tr.get('type') == 'bar':
                continue
            if len(tr['x']) == 0 or len(tr['y']) == 0:
                continue
            if len(tr['x']) != len(tr['y']):
                continue
            plottable.append(tr)
            global_x_min = min(global_x_min, tr['x'].min())
            global_x_max = max(global_x_max, tr['x'].max())
        all_data[det] = plottable

    # Guard against all-empty data
    if not np.isfinite(global_x_min) or not np.isfinite(global_x_max):
        print("No plottable data (all traces empty).")
        return None, None

    x_pad = max(1.0, (global_x_max - global_x_min) * 0.02)
    x_lim = (global_x_min - x_pad, global_x_max + x_pad)

    # --- figure ---
    fig, axes = plt.subplots(len(dets), 1, figsize=figsize, sharex=True, squeeze=False)
    axes = axes[:, 0]  # flatten to 1-D

    for idx, det in enumerate(dets):
        ax = axes[idx]
        traces = all_data[det]

        # Detector label
        ax.text(0.02, 0.92, det, transform=ax.transAxes, fontsize=9,
                fontweight='bold', va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='wheat', alpha=0.8))

        for tr in traces:
            style = _TRACE_STYLES.get(tr['name'], _FALLBACK_STYLE)

            # Main line
            ax.plot(tr['x'], tr['y'],
                    color=style['color'],
                    alpha=style['alpha'],
                    linewidth=style['lw'],
                    label=style['label'])

            # Error band
            if 'error_y' in tr:
                err = tr['error_y']
                ax.fill_between(tr['x'],
                                tr['y'] - err,
                                tr['y'] + err,
                                alpha=0.12,
                                color=style['color'])

        if idx == 0:
            ax.legend(fontsize=8, loc='upper right', ncol=len(_TRACE_STYLES))

        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(x_lim)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.tick_params(labelsize=8)

    axes[-1].set_xlabel('Time since trigger (s)', fontsize=10)
    fig.text(0.04, 0.5, 'Counts s$^{-1}$', va='center', ha='center',
             rotation='vertical', fontsize=10)
    fig.suptitle(f'{grb_name} — Time-Integrated Lightcurves', fontsize=12,
                 fontweight='bold')
    fig.tight_layout(rect=[0.06, 0, 1, 0.97])

    # --- output ---
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    return fig, axes


# ---------------------------------------------------------------------------
# GRB name resolution helpers
# ---------------------------------------------------------------------------

def _find_grb_dirs(data_base):
    """Return sorted list of ``GRB*`` directory names under *data_base*."""
    if not os.path.isdir(data_base):
        return []
    return sorted(
        d for d in os.listdir(data_base)
        if d.startswith('GRB') and os.path.isdir(os.path.join(data_base, d))
    )


def _resolve_grb_name(explicit=None, data_base=None):
    """Resolve GRB name from *explicit* arg, env, or directory auto-detect.

    Returns ``(grb_name, data_base)`` or ``(None, data_base)`` if unresolvable.
    """
    if data_base is None:
        data_base = os.environ.get('DATA_BASE', DEFAULT_DATA_BASE)

    if explicit and explicit.startswith('GRB'):
        return explicit, data_base

    # Notebook globals (set by load_grb)
    try:
        ipy = __import__('IPython').get_ipython()
        if ipy is not None:
            nb = ipy.user_ns
            nb_grb = nb.get('GRB_Name')
            if nb_grb:
                return nb_grb, nb.get('DATA_BASE') or data_base
    except (ImportError, AttributeError):
        pass

    # Environment variable
    env_grb = os.environ.get('GRB_NAME')
    if env_grb:
        return env_grb, data_base

    # Auto-detect single GRB directory
    candidates = _find_grb_dirs(data_base)
    if len(candidates) == 1:
        return candidates[0], data_base
    elif len(candidates) > 1:
        print(f"Multiple GRB directories found: {candidates}")

    return None, data_base


def _parse_cli_args(argv):
    """Extract positional (non-flag, non-kernel) arguments from *argv*."""
    return [
        a for a in argv[1:]
        if not a.startswith('-')
        and '/jupyter/' not in a
        and not a.endswith('.json')
    ]


# ---------------------------------------------------------------------------
# CLI / notebook entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    cli_args = _parse_cli_args(sys.argv)
    explicit = cli_args[0] if cli_args else None

    grb_name, data_base = _resolve_grb_name(explicit=explicit)

    if not grb_name:
        print("No GRB found.  Either:")
        print("  python plot_lightcurves.py <GRB_NAME>")
        print("  # or, in a notebook after load_grb():")
        print("  from plot_lightcurves import plot_lightcurves")
        print("  plot_lightcurves(GRB_Name, data_base=DATA_BASE)")
        try:
            if __import__('IPython').get_ipython() is None:
                sys.exit(1)
        except (ImportError, AttributeError):
            sys.exit(1)
    else:
        print(f"Plotting lightcurves for {grb_name} (DATA_BASE={data_base})")
        save_path = f'{grb_name}_lightcurves.png'
        plot_lightcurves(grb_name, data_base=data_base, save_path=save_path)


