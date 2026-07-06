"""
fred_scorer.py
---------------
Quantifies how FRED-like each extracted light curve is, so you can rank
GRBs instead of eyeballing PNGs one by one.

FRED = Fast Rise, Exponential Decay. A single clean pulse:
    - sharp rise to peak
    - smooth, ~single-exponential decay after the peak
    - no strong secondary peaks

Score components (each 0-1, higher = more FRED-like):
    rise_score   - how much faster the rise is than the decay
    decay_score  - R^2 of an exponential fit to the post-peak decay
    single_score - penalizes additional prominent peaks besides the main one

fred_score = rise_score * decay_score * single_score  (product, so any
             one bad component tanks the overall score — a multi-peaked
             burst with a gorgeous decay tail is still not a FRED)

Usage:
    from fred_scorer import score_all_bursts, plot_top_candidates

    df = score_all_bursts('/workspace/data', pattern='*_lc.npy')
    df.head(15)   # top 15 FRED candidates across all GRBs/detectors

    plot_top_candidates(df, n=12)   # grid of the best light curves
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import linregress


def _load_lc(npy_path):
    arr = np.load(npy_path)
    return arr[:, 0], arr[:, 1]  # t, rate


def _smooth(sig, dt, smooth_seconds=1.0):
    """Boxcar-smooth a signal to suppress Poisson counting noise before peak
    detection / decay fitting. Window is smooth_seconds wide, converted to
    samples using the array's actual bin width dt."""
    n = max(3, int(round(smooth_seconds / dt)))
    if n % 2 == 0:
        n += 1  # odd window keeps things centered
    kernel = np.ones(n) / n
    return np.convolve(sig, kernel, mode='same')


def score_burst(t, rate, baseline_window=(-20, -2), smooth_seconds=1.0):
    """Score a single light curve array (t, rate) for FRED-likeness.

    Returns a dict of metrics + overall fred_score, or None if the burst
    doesn't have enough signal to evaluate (e.g. never rises above baseline).
    """
    dt = np.median(np.diff(t))

    # Baseline from pre-trigger quiet time
    baseline_mask = (t >= baseline_window[0]) & (t <= baseline_window[1])
    if baseline_mask.sum() < 5:
        baseline_mask = t < 0
    baseline = np.median(rate[baseline_mask]) if baseline_mask.sum() else np.median(rate)

    sig = rate - baseline  # background-subtracted signal

    # Smooth BEFORE any peak-finding or decay fitting — raw TTE-binned rate is
    # Poisson noise on top of the pulse shape, and treating every noise bump
    # as a "peak" is what previously drove n_prominent_peaks to 40-65 for
    # every burst regardless of actual shape.
    sig_smooth = _smooth(sig, dt, smooth_seconds=smooth_seconds)
    # Noise level estimated AFTER smoothing (averaging reduces it by ~sqrt(n))
    baseline_smooth = _smooth(rate, dt, smooth_seconds=smooth_seconds)[baseline_mask] - baseline \
        if baseline_mask.sum() else np.array([0.0])
    baseline_std = np.std(baseline_smooth) if len(baseline_smooth) > 1 else np.std(sig_smooth[t < 0])

    # Only consider t >= -5 (near/after trigger) for peak-finding — avoids
    # picking up a pre-trigger noise fluctuation as "the peak"
    search_mask = t >= -5
    if search_mask.sum() < 10:
        return None

    t_s, sig_s = t[search_mask], sig_smooth[search_mask]
    peak_idx = np.argmax(sig_s)
    peak_val = sig_s[peak_idx]
    peak_t = t_s[peak_idx]

    if peak_val < 5 * baseline_std or peak_val <= 0:
        return None  # no significant peak found

    # --- Rise time: from 10% to 90% of peak, before the peak (on smoothed sig) ---
    pre_peak_mask = t_s <= peak_t
    t_pre, sig_pre = t_s[pre_peak_mask], sig_s[pre_peak_mask]
    if len(t_pre) < 3:
        return None

    thresh_10 = 0.1 * peak_val
    thresh_90 = 0.9 * peak_val
    above_10 = np.where(sig_pre >= thresh_10)[0]
    above_90 = np.where(sig_pre >= thresh_90)[0]
    if len(above_10) == 0 or len(above_90) == 0:
        return None
    t_rise_start = t_pre[above_10[0]]
    t_rise_end = t_pre[above_90[0]]
    rise_time = max(t_rise_end - t_rise_start, 1e-3)

    # --- Decay fit: exponential decay after peak (on smoothed sig) ---
    post_peak_mask = t_s > peak_t
    t_post, sig_post = t_s[post_peak_mask], sig_s[post_peak_mask]
    decay_mask = sig_post > max(3 * baseline_std, 0.05 * peak_val)
    t_decay, sig_decay = t_post[decay_mask], sig_post[decay_mask]

    if len(t_decay) < 5:
        return None

    decay_time = t_decay[-1] - peak_t

    valid = sig_decay > 0
    if valid.sum() < 5:
        return None
    log_sig = np.log(sig_decay[valid])
    t_fit = t_decay[valid] - peak_t
    slope, intercept, r_value, _, _ = linregress(t_fit, log_sig)
    decay_r2 = r_value ** 2
    tau = -1.0 / slope if slope < 0 else np.nan

    # --- Peak multiplicity on the SMOOTHED trace, with a real prominence
    # floor (multiples of the post-smoothing noise level) and a minimum
    # separation so adjacent noise wiggles in the smoothing window can't
    # register as separate peaks ---
    prominence = max(5 * baseline_std, 0.15 * peak_val)
    min_distance_samples = max(1, int(round(smooth_seconds / dt)))
    peaks, props = find_peaks(sig_s, prominence=prominence, distance=min_distance_samples)
    n_prominent_peaks = max(len(peaks), 1)  # the main peak itself always counts as 1

    # --- Combine into scores (0-1 each) ---
    rise_score = np.clip(decay_time / (rise_time + decay_time), 0, 1) if decay_time > 0 else 0
    decay_score = np.clip(decay_r2, 0, 1)
    single_score = 1.0 / n_prominent_peaks

    fred_score = rise_score * decay_score * single_score

    return {
        'peak_time': peak_t,
        'peak_rate': peak_val + baseline,
        'rise_time_s': rise_time,
        'decay_time_s': decay_time,
        'tau_decay_s': tau,
        'decay_r2': decay_r2,
        'n_prominent_peaks': n_prominent_peaks,
        'rise_score': rise_score,
        'decay_score': decay_score,
        'single_score': single_score,
        'fred_score': fred_score,
    }


def score_all_bursts(data_base='/workspace/data', pattern='*_lc.npy'):
    """Walk data_base for all {GRB}_{det}_lc.npy files, score each, return a
    DataFrame sorted by fred_score (descending) — best FRED candidates first.
    """
    search_glob = os.path.join(data_base, '*', 'data', 'tintegrated', 'arrays', pattern)
    npy_paths = sorted(glob.glob(search_glob))
    print(f"Found {len(npy_paths)} light-curve arrays under {data_base}")

    rows = []
    for path in npy_paths:
        fname = os.path.basename(path)
        # {GRB_Name}_{det}_lc.npy
        parts = fname.replace('_lc.npy', '').split('_')
        det = parts[-1]
        grb_name = '_'.join(parts[:-1])

        try:
            t, rate = _load_lc(path)
            metrics = score_burst(t, rate)
        except Exception as e:
            print(f"  Skipping {fname}: {e}")
            continue

        if metrics is None:
            continue

        metrics.update({'grb_name': grb_name, 'det': det, 'path': path})
        rows.append(metrics)

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values('fred_score', ascending=False).reset_index(drop=True)
    print(f"Scored {len(df)} usable light curves (some skipped: no clear peak).")
    return df


def plot_top_candidates(df, n=12, ncols=4):
    """Grid plot of the top-n FRED candidates' light curves, for quick visual
    confirmation of the ranking."""
    n = min(n, len(df))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for i in range(n):
        row = df.iloc[i]
        t, rate = _load_lc(row['path'])
        ax = axes[i]
        ax.step(t, rate, where='mid', lw=0.7)
        ax.set_title(f"{row['grb_name']} {row['det']}\nscore={row['fred_score']:.2f}",
                     fontsize=8)
        ax.tick_params(labelsize=6)

    for j in range(n, len(axes)):
        axes[j].axis('off')

    fig.tight_layout()
    return fig


def plot_best_per_grb_by_tier(df, ncols=5):
    """Organize all GRBs by FRED-score tier (good/mid/poor), show one best
    detector per GRB in each tier. Creates a summary showing the range of
    FRED-like profiles across the entire catalog."""
    
    # Group by GRB, take best (highest fred_score) detector for each
    best_per_grb = df.loc[df.groupby('grb_name')['fred_score'].idxmax()]
    best_per_grb = best_per_grb.sort_values('fred_score', ascending=False).reset_index(drop=True)
    
    # Define tiers by percentile
    n_grbs = len(best_per_grb)
    good_thresh = best_per_grb['fred_score'].quantile(0.33)  # top 33%
    mid_thresh = best_per_grb['fred_score'].quantile(0.67)   # middle 34%
    
    good = best_per_grb[best_per_grb['fred_score'] >= good_thresh]
    mid = best_per_grb[(best_per_grb['fred_score'] >= mid_thresh) & (best_per_grb['fred_score'] < good_thresh)]
    poor = best_per_grb[best_per_grb['fred_score'] < mid_thresh]
    
    tiers = [
        ('Excellent FRED Candidates', good, 'green'),
        ('Moderate FRED-like', mid, 'orange'),
        ('Poor FRED Profile', poor, 'red'),
    ]
    
    figs = []
    for tier_name, tier_df, color in tiers:
        n = len(tier_df)
        if n == 0:
            continue
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.5 * nrows))
        axes = np.atleast_1d(axes).flatten()
        
        fig.suptitle(f'{tier_name} (best detector per GRB, n={n})', 
                     fontsize=14, fontweight='bold')
        
        for i, (_, row) in enumerate(tier_df.iterrows()):
            t, rate = _load_lc(row['path'])
            ax = axes[i]
            ax.step(t, rate, where='mid', lw=0.8, color=color, alpha=0.7)
            ax.set_title(f"{row['grb_name']} ({row['det']})\nscore={row['fred_score']:.2f}",
                         fontsize=9, fontweight='bold')
            ax.set_xlabel('Time (s)', fontsize=7)
            ax.set_ylabel('Rate (cts/s)', fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.2)
        
        for j in range(n, len(axes)):
            axes[j].axis('off')
        
        fig.tight_layout()
        figs.append((tier_name, fig))
    
    return figs, best_per_grb


def plot_all_detectors_by_grb(df, ncols=5):
    """For each GRB, show all its detectors in a separate row. Detector rows
    are colored by FRED-score (green=good, red=poor). Useful for seeing how
    detector selection affects the burst profile."""
    
    grbs = df['grb_name'].unique()
    grbs = sorted(grbs)
    
    fig, axes = plt.subplots(len(grbs), ncols, figsize=(4 * ncols, 2.5 * len(grbs)))
    axes = np.atleast_1d(axes)
    if axes.ndim == 1:
        axes = axes.reshape(-1, ncols)
    
    fig.suptitle('All Detectors per GRB (colored by FRED-score)', fontsize=14, fontweight='bold')
    
    for grb_idx, grb_name in enumerate(grbs):
        grb_rows = df[df['grb_name'] == grb_name].sort_values('fred_score', ascending=False)
        best_score = grb_rows['fred_score'].max()
        
        for det_idx, (_, row) in enumerate(grb_rows.iterrows()):
            if det_idx >= ncols:
                break
            
            t, rate = _load_lc(row['path'])
            ax = axes[grb_idx, det_idx]
            
            # Color code by score
            if row['fred_score'] >= 0.65:
                color = 'green'
            elif row['fred_score'] >= 0.35:
                color = 'orange'
            else:
                color = 'red'
            
            ax.step(t, rate, where='mid', lw=0.7, color=color, alpha=0.7)
            ax.set_title(f"{row['det']}: {row['fred_score']:.2f}", fontsize=8)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.2)
        
        # Label the GRB name on the left
        axes[grb_idx, 0].set_ylabel(grb_name, fontsize=9, fontweight='bold')
        
        # Turn off extra subplots
        for det_idx in range(len(grb_rows), ncols):
            axes[grb_idx, det_idx].axis('off')
    
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    df = score_all_bursts()
    print(df[['grb_name', 'det', 'fred_score', 'rise_score', 'decay_score',
              'n_prominent_peaks']].head(20))