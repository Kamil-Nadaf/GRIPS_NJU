"""Time-resolved spectral-parameter evolution plots."""

from __future__ import annotations

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.mpl_setup import silence_missing_fonts
silence_missing_fonts()


def _slice_midpoints(df):
    if {'t_start', 't_stop'}.issubset(df.columns):
        t0 = df['t_start'].astype(float)
        t1 = df['t_stop'].astype(float)
        return 0.5 * (t0 + t1), 0.5 * (t1 - t0)
    if 'slice' in df.columns:
        x = df['slice'].astype(float)
        return x, np.full(len(df), 0.4)
    return np.arange(1, len(df) + 1, dtype=float), np.full(len(df), 0.4)


def plot_tres_params(df, title=None, params=None, figsize=None):
    """Plot 1σ parameter evolution vs mid-slice time.

    Default panels: α, Ep (keV). Adds β / νFν when columns exist.
    Returns ``(fig, axes)``; fig is None if ``df`` is empty.
    """
    if df is None or len(df) == 0:
        return None, None
    work = df.copy().reset_index(drop=True)
    if 'slice' in work.columns:
        work = work.sort_values('slice').reset_index(drop=True)

    candidates = params or [
        ('alpha', 'alpha_low', 'alpha_high', r'$\alpha$', None),
        ('Ep_best', 'Ep_low', 'Ep_high', r'$E_{\mathrm{p}}$', 'keV'),
        ('beta', 'beta_low', 'beta_high', r'$\beta$', None),
        ('vFv_best', 'vFv_low', 'vFv_high', r'$\nu F_{\nu}$', None),
    ]
    panels = []
    for col, lo, hi, label, unit in candidates:
        if col not in work.columns:
            continue
        panels.append((col, lo, hi, label, unit))
    if not panels:
        return None, None

    tmid, half = _slice_midpoints(work)
    n = len(panels)
    figsize = figsize or (7.5, max(2.6 * n, 3.2))
    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)
    if n == 1:
        axes = [axes]

    constrained = None
    if 'ep_constrained' in work.columns:
        constrained = work['ep_constrained'].astype(int).to_numpy()

    for ax, (col, lo_k, hi_k, label, unit) in zip(axes, panels):
        y = work[col].astype(float).to_numpy()
        yerr_lo = work[lo_k].astype(float).to_numpy() if lo_k in work.columns else None
        yerr_hi = work[hi_k].astype(float).to_numpy() if hi_k in work.columns else None
        yerr = None
        if yerr_lo is not None and yerr_hi is not None:
            yerr = np.vstack([yerr_lo, yerr_hi])

        if constrained is not None and col.startswith('Ep'):
            ok = constrained.astype(bool)
            bad = ~ok
            if ok.any():
                ax.errorbar(
                    tmid[ok], y[ok],
                    xerr=half[ok] if half is not None else None,
                    yerr=None if yerr is None else yerr[:, ok],
                    fmt='o-', capsize=3, color='C0', label='constrained')
            if bad.any():
                ax.errorbar(
                    tmid[bad], y[bad],
                    xerr=half[bad] if half is not None else None,
                    yerr=None if yerr is None else yerr[:, bad],
                    fmt='s--', capsize=3, color='C3', label='unconstrained')
            if ok.any() and bad.any():
                ax.legend(loc='best', fontsize=8)
        else:
            ax.errorbar(
                tmid, y,
                xerr=half if half is not None else None,
                yerr=yerr, fmt='o-', capsize=3, color='C0')

        ylab = label if not unit else f'{label} ({unit})'
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time since trigger (s)')
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def save_tres_params_plot(df, out_path, title=None, **kwargs):
    """Write ``tres_params.png`` (or ``out_path``) and return the path or None."""
    fig, _ = plot_tres_params(df, title=title, **kwargs)
    if fig is None:
        return None
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ tres param plot → {out_path}')
    return out_path
