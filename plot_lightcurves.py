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
    {DATA_BASE}/{GRB_NAME}/data/tintegrated/heapy/<det>/lc.html       (fixed bins)
    {DATA_BASE}/{GRB_NAME}/data/tintegrated/heapy/<det>/bb_lc.json  (heapy pgSignal BB)
    {DATA_BASE}/{GRB_NAME}/data/tintegrated/heapy/<det>/rebin_lc.html (heapy SNR rebin)
"""

import os
import re
import json
import base64
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

try:
    from pipeline.mpl_setup import silence_missing_fonts
    silence_missing_fonts()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DATA_BASE = '/workspace/data'

# heapy trace names → plot style (pastel UI palette, still readable on white)
_TRACE_STYLES = {
    'source lightcurve':      {
        'color': '#5c6aa8', 'label': 'Source', 'alpha': 0.95, 'lw': 1.2, 'ls': '-',
    },
    'net lightcurve':         {
        'color': '#4a4560', 'label': 'Net (src−bkg)', 'alpha': 0.95, 'lw': 1.15, 'ls': '-',
    },
    'background lightcurve':  {
        'color': '#9a94a8', 'label': 'Background', 'alpha': 0.9, 'lw': 0.9, 'ls': '--',
    },
    'total lightcurve':       {
        'color': '#6fafa0', 'label': 'Total', 'alpha': 0.9, 'lw': 1.0, 'ls': '-',
    },
}

_BB_STYLE = {
    'color': '#d4897c', 'label': 'Bayesian blocks', 'alpha': 0.95, 'lw': 2.0,
    'ls': '-', 'zorder': 5,
}
_FALLBACK_STYLE = {'color': '#5c5478', 'label': 'Unknown', 'alpha': 0.8, 'lw': 0.8, 'ls': '-'}
_TINT_SHADE = {'color': '#fab49b', 'alpha': 0.28}

_DEFAULT_WHICH = ('source', 'background')

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


_LC_FILES = {'fixed': 'lc_fixed.json', 'rebin': 'rebin_lc.json', 'bayesian_blocks': 'bb_lc.json'}
_LC_HTML_FALLBACK = {'rebin': 'rebin_lc.html', 'fixed': 'lc.html'}


def _heapy_lc_dir(data_base, grb_name, heapy_dir=None):
    if heapy_dir:
        return heapy_dir
    return os.path.join(data_base, grb_name, 'data', 'tintegrated', 'heapy')


def resolve_lc_path(data_base, grb_name, det, lc_kind='bayesian_blocks',
                    heapy_dir=None):
    """Return path to LC file (lc_kind: auto|fixed|rebin|bayesian_blocks)."""
    base = _heapy_lc_dir(data_base, grb_name, heapy_dir=heapy_dir)
    if lc_kind == 'auto':
        for kind in ('bayesian_blocks', 'rebin', 'fixed'):
            path = os.path.join(base, det, _LC_FILES[kind])
            if os.path.isfile(path):
                return path
            html_fb = _LC_HTML_FALLBACK.get(kind)
            if html_fb and os.path.isfile(os.path.join(base, det, html_fb)):
                return os.path.join(base, det, html_fb)
        return os.path.join(base, det, _LC_FILES['fixed'])
    if lc_kind not in _LC_FILES:
        raise ValueError(f"lc_kind must be one of {list(_LC_FILES)} or 'auto'")
    return os.path.join(base, det, _LC_FILES[lc_kind])


def _trace_role(name):
    n = (name or '').strip().lower()
    if 'background' in n:
        return 'background'
    if n.startswith('net'):
        return 'net'
    if n.startswith('total'):
        return 'total'
    if 'source' in n:
        return 'source'
    return 'other'


def tint_xlim(t1, t2):
    """X-range that frames a tint interval without a long empty/next-pulse tail."""
    t1, t2 = float(t1), float(t2)
    span = max(t2 - t1, 1.0)
    return t1 - max(6.0, 0.5 * span), t2 + max(8.0, 1.5 * span)


def visible_ylim(traces, x0, x1, pad=0.10):
    """Y-limits from samples inside ``[x0, x1]`` so a later pulse cannot squash the view."""
    ys = []
    for tr in traces or []:
        x = np.asarray(tr.get('x'), dtype=float)
        y = np.asarray(tr.get('y'), dtype=float)
        if x.size == 0 or y.size != x.size:
            continue
        m = (x >= x0) & (x <= x1)
        if not np.any(m):
            continue
        yy = y[m]
        err = tr.get('error_y')
        if err is not None and len(err) == len(y):
            e = np.asarray(err, dtype=float)[m]
            ys.append(yy - e)
            ys.append(yy + e)
        else:
            ys.append(yy)
    if not ys:
        return None
    lo = float(np.nanmin(np.concatenate([np.ravel(v) for v in ys])))
    hi = float(np.nanmax(np.concatenate([np.ravel(v) for v in ys])))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    span = hi - lo
    return lo - pad * span, hi + pad * span


def _error_band(ax, x, y, err, edges, color, alpha=0.18):
    """Stepped uncertainty band — no caps, no vertical hair."""
    y = np.asarray(y, dtype=float)
    err = np.asarray(err, dtype=float)
    if y.size == 0 or err.size != y.size:
        return
    lo, hi = y - err, y + err
    if edges is not None and len(edges) == len(y) + 1:
        ax.fill_between(
            edges[:-1], lo, hi, step='post', color=color, alpha=alpha,
            linewidth=0, zorder=1)
    else:
        ax.fill_between(x, lo, hi, color=color, alpha=alpha, linewidth=0, zorder=1)


def _plot_trace(ax, tr, style, plot_style='step', show_errors='band'):
    """Plot one lightcurve trace as step (binned) or connected line."""
    x, y = tr['x'], tr['y']
    fill = bool(style.get('fill'))
    kwargs = dict(
        color=style['color'],
        alpha=style['alpha'],
        linewidth=style['lw'],
        linestyle=style.get('ls', '-'),
        label=style['label'],
        zorder=style.get('zorder', 3),
    )
    edges = tr.get('edges')
    if edges is None and plot_style != 'line' and len(x) >= 1:
        from pipeline.lc_io import infer_bin_edges_from_centers
        edges = infer_bin_edges_from_centers(x)
    if plot_style == 'line':
        ax.plot(x, y, **kwargs)
    elif edges is not None and len(edges) == len(y) + 1:
        if fill:
            kwargs.pop('linestyle', None)
            ax.stairs(
                y, edges, baseline=0, fill=True,
                facecolor=style.get('facecolor', style['color']),
                edgecolor=style['color'], linewidth=style['lw'],
                alpha=0.75, label=style['label'], zorder=2)
        else:
            ax.stairs(y, edges, baseline=None, **kwargs)
    else:
        ax.step(x, y, where='mid', **kwargs)

    err = tr.get('error_y')
    if err is None or show_errors in (False, 'none') or fill:
        return
    if show_errors == 'band':
        _error_band(ax, x, y, err, edges, style['color'])
        return
    if show_errors == 'bars':
        ax.errorbar(
            x, y, yerr=err, fmt='none', ecolor=style['color'],
            alpha=0.28, capsize=0, elinewidth=0.6, zorder=2)


def find_detectors(data_base, grb_name, lc_kind='rebin', heapy_dir=None):
    """Return sorted detector names that have lightcurve HTML files."""
    base = _heapy_lc_dir(data_base, grb_name, heapy_dir=heapy_dir)
    if not os.path.isdir(base):
        return []
    out = []
    for entry in sorted(os.listdir(base)):
        if entry in ('versions',):
            continue
        det_dir = os.path.join(base, entry)
        if not os.path.isdir(det_dir):
            continue
        fixed = os.path.join(det_dir, 'lc.html')
        fixed_json = os.path.join(det_dir, 'lc_fixed.json')
        rebin = os.path.join(det_dir, 'rebin_lc.json')
        rebin_html = os.path.join(det_dir, 'rebin_lc.html')
        bb = os.path.join(det_dir, 'bb_lc.json')
        pg = os.path.join(det_dir, 'pgsignal')
        if lc_kind == 'rebin' and not (os.path.isfile(rebin) or os.path.isfile(rebin_html)):
            continue
        if lc_kind == 'fixed' and not (os.path.isfile(fixed) or os.path.isfile(fixed_json)):
            continue
        if lc_kind == 'bayesian_blocks' and not (os.path.isfile(bb) or os.path.isdir(pg)):
            continue
        if lc_kind == 'auto' and not (
                os.path.isfile(bb) or os.path.isdir(pg) or
                os.path.isfile(rebin) or os.path.isfile(rebin_html) or
                os.path.isfile(fixed) or os.path.isfile(fixed_json)):
            continue
        out.append(entry)
    return out


def load_detector_data(data_base, grb_name, det, lc_kind='rebin',
                       bayesian_blocks_kwargs=None, heapy_dir=None):
    """Load lightcurve traces for a single detector."""
    if lc_kind == 'bayesian_blocks':
        from pipeline.lc_io import load_heapy_bayesian_blocks_lc, load_heapy_fixed_lc
        lc_dir = os.path.join(_heapy_lc_dir(data_base, grb_name, heapy_dir=heapy_dir), det)
        traces = load_heapy_bayesian_blocks_lc(lc_dir)
        try:
            for tr in load_heapy_fixed_lc(lc_dir, parse_plotly_html=parse_plotly_html):
                if 'background' in tr.get('name', '').lower():
                    traces = [t for t in traces
                              if 'background' not in t.get('name', '').lower()]
                    traces.append(tr)
                    break
        except FileNotFoundError:
            pass
        return traces
    if lc_kind == 'rebin':
        from pipeline.lc_io import load_heapy_rebin_lc
        lc_dir = os.path.join(_heapy_lc_dir(data_base, grb_name, heapy_dir=heapy_dir), det)
        return load_heapy_rebin_lc(lc_dir, parse_plotly_html=parse_plotly_html)
    if lc_kind == 'fixed':
        from pipeline.lc_io import load_heapy_fixed_lc
        lc_dir = os.path.join(_heapy_lc_dir(data_base, grb_name, heapy_dir=heapy_dir), det)
        return load_heapy_fixed_lc(lc_dir, parse_plotly_html=parse_plotly_html)
    path = resolve_lc_path(
        data_base, grb_name, det, lc_kind=lc_kind, heapy_dir=heapy_dir)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return parse_plotly_html(path)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_lightcurves(grb_name, data_base=DEFAULT_DATA_BASE, save_path=None,
                     figsize=None, lc_kind='fixed', plot_style='step',
                     bayesian_blocks_kwargs=None, heapy_dir=None, dets=None,
                     t1=None, t2=None, xlim=None, overlay_bb=True,
                     show_errors='none', which=_DEFAULT_WHICH, show=True):
    """Plot GRB lightcurves — one subplot per detector, stacked vertically.

    Parameters
    ----------
    grb_name : str
        GRB name, e.g. ``'GRB131011A'``.
    data_base : str
        Root data directory (default ``/workspace/data``).
    save_path : str or None
        If given, save figure to this path.
    figsize : tuple or None
        Figure size in inches. Default scales with the number of detectors.
    lc_kind : str
        ``'fixed'`` (default) — uniform heapy bins; ``'rebin'`` — SNR-adaptive;
        ``'bayesian_blocks'``, or ``'auto'``.
    plot_style : str
        ``'step'`` for binned step plots (default), ``'line'`` for connected lines.
    heapy_dir : str or None
        Override the canonical ``.../tintegrated/heapy`` path.
    dets : list or None
        If given, plot only these detectors (intersection with those found on disk).
    t1, t2 : float or None
        Tint / GCN interval. Shaded on each panel; also sets a pulse-framed
        ``xlim`` unless ``xlim`` is given.
    xlim : tuple or None
        Explicit x-range. ``None`` uses ``tint_xlim(t1, t2)`` when those are
        set, otherwise the full lightcurve.
    overlay_bb : bool
        Draw heapy Bayesian blocks in red on top of fixed/rebin data.
    show_errors : {'band', 'bars', 'none'}
        ``'band'`` (default) is a stepped fill with no caps.
    which : sequence
        Trace roles to draw: ``source``, ``background``, ``net``, ``total``.
    show : bool
        Call ``plt.show()`` when ``save_path`` is omitted.

    Returns
    -------
    fig, axes
    """
    # --- discover detectors ---
    found = find_detectors(
        data_base, grb_name, lc_kind=lc_kind, heapy_dir=heapy_dir)
    if dets is not None:
        want = {str(d).lower() for d in dets}
        dets = [d for d in found if d.lower() in want]
    else:
        dets = found
    if not dets:
        if lc_kind == 'bayesian_blocks':
            fixed_dets = find_detectors(
                data_base, grb_name, lc_kind='fixed', heapy_dir=heapy_dir)
            if fixed_dets:
                print(
                    f"No bb_lc.json for {grb_name} ({fixed_dets} have lc.html only).\n"
                    "Re-run Stage 2a: extract_tintegrated_spectra(bs_p0=0.05)\n"
                    "Or plot fixed bins: lc_kind='fixed'")
        else:
            print(f"No detector lightcurves found for {grb_name} under {data_base}")
        return None, None

    print(f"Found {len(dets)} detectors: {dets}")
    wanted = set(which)

    # --- load data & compute global x-range in one pass ---
    all_data = {}
    bb_data = {}
    global_x_min, global_x_max = np.inf, -np.inf

    for det in dets:
        traces = load_detector_data(
            data_base, grb_name, det, lc_kind=lc_kind,
            bayesian_blocks_kwargs=bayesian_blocks_kwargs,
            heapy_dir=heapy_dir)
        plottable = []
        for tr in traces:
            if len(tr['x']) == 0 or len(tr['y']) == 0:
                continue
            if len(tr['x']) != len(tr['y']):
                continue
            if not tr.get('name', '').strip():
                continue
            if _trace_role(tr['name']) not in wanted:
                continue
            plottable.append(tr)
            global_x_min = min(global_x_min, tr['x'].min())
            global_x_max = max(global_x_max, tr['x'].max())
        all_data[det] = plottable
        if overlay_bb and lc_kind != 'bayesian_blocks':
            try:
                bb_traces = load_detector_data(
                    data_base, grb_name, det, lc_kind='bayesian_blocks',
                    heapy_dir=heapy_dir)
                bb_data[det] = [
                    tr for tr in bb_traces
                    if _trace_role(tr.get('name')) == 'source' and len(tr.get('y', []))
                ]
            except FileNotFoundError:
                bb_data[det] = []

    # Guard against all-empty data
    if not np.isfinite(global_x_min) or not np.isfinite(global_x_max):
        print("No plottable data (all traces empty).")
        return None, None

    if xlim is not None:
        x_lim = (float(xlim[0]), float(xlim[1]))
    elif t1 is not None and t2 is not None:
        x_lim = tint_xlim(t1, t2)
    else:
        x_pad = max(1.0, (global_x_max - global_x_min) * 0.02)
        x_lim = (global_x_min - x_pad, global_x_max + x_pad)

    if figsize is None:
        figsize = (9.2, max(3.2, 1.85 * len(dets)))

    # --- figure ---
    fig, axes = plt.subplots(len(dets), 1, figsize=figsize, sharex=True, squeeze=False)
    axes = axes[:, 0]  # flatten to 1-D

    for idx, det in enumerate(dets):
        ax = axes[idx]
        traces = all_data[det]

        if t1 is not None and t2 is not None:
            ax.axvspan(
                float(t1), float(t2), color=_TINT_SHADE['color'],
                alpha=_TINT_SHADE['alpha'], zorder=0,
                label='Tint' if idx == 0 else None)

        ax.text(
            1.01, 0.50, det, transform=ax.transAxes, fontsize=9,
            fontweight='bold', va='center', ha='left', color='#5c5478',
            clip_on=False)

        for tr in traces:
            style = dict(_TRACE_STYLES.get(tr['name'], _FALLBACK_STYLE))
            role = _trace_role(tr['name'])
            err_mode = 'none' if role == 'background' else show_errors
            _plot_trace(ax, tr, style, plot_style=plot_style, show_errors=err_mode)

        for tr in bb_data.get(det, []):
            _plot_trace(
                ax, tr, dict(_BB_STYLE), plot_style='step', show_errors='none')

        ax.grid(True, alpha=0.22, linestyle=':', linewidth=0.6)
        ax.set_xlim(x_lim)
        y_lim = visible_ylim(list(traces) + list(bb_data.get(det, [])), *x_lim)
        if y_lim is not None:
            ax.set_ylim(y_lim)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.tick_params(labelsize=8, length=3)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)

    # Legend below the stack so it never covers the pulse or title
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc='upper center',
            bbox_to_anchor=(0.55, 0.02), ncol=min(4, len(handles)),
            fontsize=8, frameon=False, columnspacing=1.2,
            handlelength=2.0)

    axes[-1].set_xlabel('Time since trigger (s)', fontsize=10)
    fig.text(0.04, 0.52, 'Counts s$^{-1}$', va='center', ha='center',
             rotation='vertical', fontsize=10)
    lc_label = {
        'auto': 'heapy adaptive',
        'rebin': 'heapy SNR rebin',
        'fixed': 'fixed bins',
        'bayesian_blocks': 'Bayesian blocks (heapy pgSignal)',
    }[lc_kind]
    extra = [lc_label]
    if overlay_bb and any(bb_data.values()):
        extra.append('BB overlay')
    if t1 is not None and t2 is not None:
        extra.append(f'tint {float(t1):g}–{float(t2):g} s')
    fig.suptitle(f'{grb_name} lightcurves ({", ".join(extra)})', fontsize=11,
                 fontweight='bold', y=0.995)
    bottom = 0.08 if handles else 0.04
    fig.tight_layout(rect=[0.06, bottom, 0.97, 0.97])

    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches='tight')
        print(f"Saved: {save_path}")
    elif show:
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


