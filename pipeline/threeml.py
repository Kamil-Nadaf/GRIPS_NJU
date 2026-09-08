"""3ML spectral pipeline (alternative to heapy + bayspec).

Same GRB data layout; products live in sibling ``3ML/`` dirs::

    {GRB}/data/tintegrated/{heapy,bayspec,3ML}/
    {GRB}/data/tresolved/{heapy,bayspec,3ML}/

Uses already-downloaded heapy TTE + poshist, BALROG DRMs from gbm_drm_gen,
TimeSeriesBuilder plugins, and MultiNest via threeML. Follows
https://threeml.readthedocs.io/en/stable/notebooks/grb080916C.html
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .mpl_setup import silence_missing_fonts
silence_missing_fonts()
import numpy as np
import pandas as pd

from .constants import LOG_EP_PRIOR
from .detectors import (
    detector_energy_range, fit_detectors, nai_dets, rank_detector_angles,
    resolve_detectors,
)
from .download import retrieve
from .fitting import compute_vFv, ep_constraint_flags, get_model_params
from .paths import (
    commit_meta, fingerprint, fit_fingerprint, read_meta, resolve_fit_dir,
    write_meta,
)

DOCS_URL = (
    'https://threeml.readthedocs.io/en/stable/notebooks/grb080916C.html'
)
BINS_DOCS_URL = (
    'https://threeml.readthedocs.io/en/stable/notebooks/'
    'Building_Plugins_from_TimeSeries.html'
)

# TimeSeriesBuilder.create_time_bins methods
BIN_METHODS = ('custom', 'constant', 'significance', 'bayesblocks')
DEFAULT_BIN_DT = 2.0
DEFAULT_BIN_SIGMA = 25.0
DEFAULT_BIN_P0 = 0.01
DEFAULT_BIN_MIN_WIDTH = 0.1

# UI / CLI names → astromodels class
MODEL_ALIAS = {
    'cpl': 'Cutoff_powerlaw',
    'cutoffpl': 'Cutoff_powerlaw',
    'cutoff_powerlaw': 'Cutoff_powerlaw',
    'band': 'Band',
    'pl': 'Powerlaw',
    'powerlaw': 'Powerlaw',
    'sbpl': 'SmoothlyBrokenPowerLaw',
    'smoothlybrokenpowerlaw': 'SmoothlyBrokenPowerLaw',
}


def threeml_available():
    try:
        import threeML  # noqa: F401
        return True, getattr(threeML, '__version__', 'unknown')
    except Exception as exc:
        return False, str(exc)


def describe_workflow():
    return [
        ('01', 'Reuse heapy TTE + poshist (same DATA_BASE / gbm_data)'),
        ('02', 'BALROG DRM at slice mean time (gbm_drm_gen)'),
        ('03', 'TimeSeriesBuilder background + SpectrumLike plugins'),
        ('04', 'Joint Likelihood + MultiNest (CPL / Band / PL)'),
        ('05', 'Plots + HDF5 params under data/{tint,tres}/3ML/'),
        ('06', 'Tres bins: BB/constant/significance on brightest NaI, reuse all dets'),
    ]


def _tte_path(ctx, det):
    p = ctx.gbm_rtv.rtv_res['tte'][det]
    if isinstance(p, (list, tuple)):
        p = p[0]
    return p


def _poshist_path(ctx):
    p = ctx.gbm_rtv.rtv_res['poshist']
    if isinstance(p, (list, tuple)):
        return p[0]
    return p


def _canon_model(name):
    key = str(name).strip()
    alias = MODEL_ALIAS.get(key.lower(), key)
    return key.lower() if key.lower() in MODEL_ALIAS else key, alias


def _layout_3ml_tint(ctx):
    return os.path.join(ctx.paths.tintegrated_path, '3ML')


def _layout_3ml_tres(ctx):
    return os.path.join(ctx.paths.tresolved_path, '3ML')


def _layout_3ml_h5(ctx):
    return os.path.join(_layout_3ml_tres(ctx), f'{ctx.name}_3ML_data.h5')


def _fit_root(ctx, model_name, mode):
    tag = _canon_model(model_name)[0]
    if mode in ('tresolved', 'tres'):
        return os.path.join(_layout_3ml_tres(ctx), tag)
    return os.path.join(_layout_3ml_tint(ctx), tag)


def _weights_path(prefix):
    return prefix + 'post_equal_weights.dat'


def _bkg_intervals(ctx, background_interval=None):
    """Heapy-style background: LC window minus the burst (``bs_ignore``).

    Heapy ``lc_window`` is ``[-lc_pad_pre, t2+lc_pad_post]`` (trigger-relative)
    with ``bs_ignore=[t1, t2]``. 3ML polynomial bkg uses the two remaining
    segments: ``[-lc_pad_pre, t1]`` and ``[t2, t2+lc_pad_post]``.
    """
    if background_interval:
        parts = [p.strip() for p in str(background_interval).split(',') if p.strip()]
        if parts:
            return parts
    span0, span1 = ctx.burst_span()
    t1 = float(span0 if span0 is not None else ctx.t1)
    t2 = float(span1 if span1 is not None else ctx.t2)
    pre = float(ctx.lc_pad_pre)
    post = float(ctx.resolved_lc_pad_post)
    lo0, lo1 = -pre, t1
    hi0, hi1 = t2, t2 + post
    if lo1 <= lo0:
        lo0 = lo1 - max(pre, 10.0)
    if hi1 <= hi0:
        hi1 = hi0 + max(post, 10.0)
    return [f'{lo0:.3f}-{lo1:.3f}', f'{hi0:.3f}-{hi1:.3f}']


def pick_brightest_nai(dets, ranked=None):
    """Smallest-angle NaI among ``dets``; fall back to first NaI then first det."""
    nai = nai_dets(dets)
    if ranked:
        for det, _ang in ranked:
            if det in nai:
                return det
    if nai:
        return nai[0]
    return list(dets)[0] if dets else None


def merge_short_bins(slices, min_width=DEFAULT_BIN_MIN_WIDTH, sigmas=None,
                     min_sigma=None):
    """Merge adjacent bins that are too short or (optionally) too low-σ.

    Coverage stays contiguous. A poor last bin merges into the previous one.
    """
    if not slices:
        return []
    out = [(float(a), float(b)) for a, b in slices]
    sig = list(sigmas) if sigmas is not None else [None] * len(out)
    if len(sig) != len(out):
        sig = [None] * len(out)
    min_width = float(min_width) if min_width is not None else 0.0
    min_sigma = float(min_sigma) if min_sigma is not None else None

    def poor(i):
        t0, t1 = out[i]
        if (t1 - t0) < min_width:
            return True
        if min_sigma is not None and sig[i] is not None and float(sig[i]) < min_sigma:
            return True
        return False

    changed = True
    while changed and len(out) > 1:
        changed = False
        for i in range(len(out)):
            if not poor(i):
                continue
            if i < len(out) - 1:
                j = i + 1
                lo, hi = out[i][0], out[j][1]
                s0, s1 = sig[i], sig[j]
                s = None if s0 is None or s1 is None else (float(s0) ** 2 + float(s1) ** 2) ** 0.5
                out = out[:i] + [(lo, hi)] + out[j + 1:]
                sig = sig[:i] + [s] + sig[j + 1:]
            else:
                lo, hi = out[i - 1][0], out[i][1]
                s0, s1 = sig[i - 1], sig[i]
                s = None if s0 is None or s1 is None else (float(s0) ** 2 + float(s1) ** 2) ** 0.5
                out = out[:i - 1] + [(lo, hi)]
                sig = sig[:i - 1] + [s]
            changed = True
            break
    return out


def _slices_from_ts_bins(ts):
    bins = getattr(ts, 'bins', None)
    if bins is None:
        raise RuntimeError('TimeSeriesBuilder has no bins after create_time_bins')
    if hasattr(bins, 'start_times') and hasattr(bins, 'stop_times'):
        return [(float(a), float(b)) for a, b in zip(bins.start_times, bins.stop_times)]
    if hasattr(bins, 'starts') and hasattr(bins, 'stops'):
        return [(float(a), float(b)) for a, b in zip(bins.starts, bins.stops)]
    out = []
    try:
        n = len(bins)
    except TypeError as exc:
        raise RuntimeError(f'Cannot read bins from {type(bins)}') from exc
    for i in range(n):
        b = bins[i]
        if hasattr(b, 'start_time'):
            out.append((float(b.start_time), float(b.stop_time)))
        elif hasattr(b, 'start'):
            out.append((float(b.start), float(b.stop)))
        else:
            out.append((float(b[0]), float(b[1])))
    return out


def _sigmas_from_ts_bins(ts):
    bins = getattr(ts, 'bins', None)
    if bins is None:
        return None
    sigmas = []
    try:
        n = len(bins)
    except TypeError:
        return None
    for i in range(n):
        b = bins[i]
        val = None
        for attr in ('significance', 'sigma', 'snr'):
            if hasattr(b, attr):
                try:
                    val = float(getattr(b, attr))
                except (TypeError, ValueError):
                    val = None
                break
        sigmas.append(val)
    if all(s is None for s in sigmas):
        return None
    return sigmas


def canon_bin_method(name):
    key = str(name or 'custom').strip().lower()
    if key in ('manual',):
        return 'custom'
    if key not in BIN_METHODS:
        raise ValueError(f'bin_method must be one of {BIN_METHODS}, got {name!r}')
    return key


def persist_3ml_bins(persist_dir, payload):
    os.makedirs(persist_dir, exist_ok=True)
    path = os.path.join(persist_dir, '3ml_bins.json')
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2, default=str)
    print(
        f'  wrote {path}  n_raw={len(payload.get("raw_slices") or [])} '
        f'n_merged={len(payload.get("merged_slices") or [])}')
    return path


def resolve_3ml_bins(ctx, dets, bin_method='custom', background_interval=None,
                     bin_kwargs=None, mat_type=2, persist_dir=None):
    """Build tres intervals via TimeSeriesBuilder.create_time_bins.

    ``custom`` uses ``ctx.time_slices`` (Table C1 / UI). Other methods run on
    the brightest (smallest-angle) NaI, then the same edges are reused for
    every detector. See ``BINS_DOCS_URL``.
    """
    kw = dict(bin_kwargs or {})
    method = canon_bin_method(bin_method)
    span0, span1 = ctx.burst_span()
    t1 = float(span0 if span0 is not None else ctx.t1)
    t2 = float(span1 if span1 is not None else ctx.t2)
    min_width = float(kw.pop('min_width', DEFAULT_BIN_MIN_WIDTH))
    min_sigma = kw.pop('min_sigma', None)
    if min_sigma is not None:
        min_sigma = float(min_sigma)

    ref_det = None
    ranked = None
    raw = []
    sigmas = None

    if method == 'custom':
        slices = list(ctx.time_slices or [(t1, t2)])
        raw = [[float(a), float(b)] for a, b in slices]
    else:
        try:
            ranked = rank_detector_angles(ctx, dets=list(dets))
        except Exception as exc:
            print(f'  WARNING: angle ranking failed ({exc}); using first NaI')
            ranked = None
        ref_det = pick_brightest_nai(dets, ranked=ranked)
        if ref_det is None:
            raise ValueError(f'{ctx.name}: no detector for 3ML binning')
        print(f'  3ML bins method={method} ref_det={ref_det} window=({t1:g},{t2:g})')
        t_mean = 0.5 * (t1 + t2)
        bkg = _bkg_intervals(ctx, background_interval)
        rsp = _make_drm(ctx, ref_det, t_mean, mat_type=mat_type)
        ts = _ts_builder(ctx, ref_det, rsp)
        ts.set_background_interval(*bkg)
        create_kw = {}
        if method == 'constant':
            create_kw['dt'] = float(kw.get('dt', DEFAULT_BIN_DT))
            ts.create_time_bins(start=t1, stop=t2, method='constant', **create_kw)
        elif method == 'significance':
            create_kw['sigma'] = float(kw.get('sigma', DEFAULT_BIN_SIGMA))
            ts.create_time_bins(start=t1, stop=t2, method='significance', **create_kw)
        elif method == 'bayesblocks':
            create_kw['p0'] = float(kw.get('p0', DEFAULT_BIN_P0))
            create_kw['use_background'] = bool(kw.get('use_background', True))
            ts.create_time_bins(start=t1, stop=t2, method='bayesblocks', **create_kw)
        slices = _slices_from_ts_bins(ts)
        raw = [[float(a), float(b)] for a, b in slices]
        sigmas = _sigmas_from_ts_bins(ts)
        print(f'  raw bins ({len(slices)}): {slices}')

    merged = merge_short_bins(
        [(a, b) for a, b in raw], min_width=min_width,
        sigmas=sigmas, min_sigma=min_sigma)
    if not merged:
        merged = [(t1, t2)]
    print(f'  merged bins ({len(merged)}, min_width={min_width}): {merged}')

    payload = {
        'method': method,
        'ref_det': ref_det,
        't1': t1,
        't2': t2,
        'min_width': min_width,
        'min_sigma': min_sigma,
        'ranked': ranked,
        'raw_slices': raw,
        'merged_slices': [[float(a), float(b)] for a, b in merged],
        'docs': BINS_DOCS_URL,
    }
    if persist_dir:
        persist_3ml_bins(persist_dir, payload)
    return [(float(a), float(b)) for a, b in merged], payload


def _3ml_fingerprint(ctx, tag, mode, nlive, dets, include_bgo=False,
                     background_interval=None, bin_method='custom',
                     slices=None):
    return fingerprint({
        'backend': '3ML',
        'mode': mode,
        'fit': fit_fingerprint(ctx, tag, nlive=nlive, fit_dets=dets),
        'include_bgo': bool(include_bgo),
        'bkg': _bkg_intervals(ctx, background_interval),
        'nai_energy': list(getattr(ctx, 'nai_energy', []) or []),
        'bgo_energy': list(getattr(ctx, 'bgo_energy', []) or []),
        'bin_method': canon_bin_method(bin_method) if mode in ('tresolved', 'tres') else 'tint',
        'slices': slices if slices is not None else ctx.time_slices,
    })


def _active_3ml_dir(model_root, fp, name, tag, mode):
    """Prefer ``versions/{fp}`` when that run wrote MultiNest weights."""
    versioned = os.path.join(model_root, 'versions', fp)
    if mode in ('tresolved', 'tres'):
        probe = os.path.join(versioned, 'slice_01')
    else:
        probe = versioned
    if os.path.isfile(_weights_path(os.path.join(probe, f'{name}-'))):
        return versioned
    return _resolve_3ml_dir(model_root, fp, tag, mode, force=False)


def _resolve_3ml_dir(model_root, fp, tag, mode, force=False):
    """Like ``resolve_fit_dir``, but also recognizes older 3ML meta keys.

    Older runs stored ``3ml_{mode}_{tag}`` / ``fp`` instead of
    ``canonical_fit_{tag}``. Without that, a fingerprint change would
    land on the canonical dir and skip the stale MultiNest weights.
    """
    if force:
        return model_root
    mode_key = 'tresolved' if mode in ('tresolved', 'tres') else 'tintegrated'
    keys = (
        f'canonical_fit_{tag}',
        f'3ml_{mode_key}_{tag}',
        f'3ml_{mode}_{tag}',
        'fp',
    )
    for root in (model_root, os.path.dirname(model_root)):
        if not root:
            continue
        meta = read_meta(os.path.join(root, 'pipeline_meta.json'))
        stored = next((meta[k] for k in keys if meta.get(k)), None)
        if stored is None:
            continue
        if stored == fp:
            return model_root
        versioned = os.path.join(model_root, 'versions', fp)
        os.makedirs(versioned, exist_ok=True)
        return versioned
    return resolve_fit_dir(model_root, fp, f'canonical_fit_{tag}', force=False)


def _lc_view_limits(ctx):
    span0, span1 = ctx.burst_span()
    t2 = float(span1 if span1 is not None else ctx.t2)
    return -float(ctx.lc_pad_pre), t2 + float(ctx.resolved_lc_pad_post)


def _energy_str(ctx, det):
    lo, hi = detector_energy_range(det, ctx=ctx)
    return f'{lo:g}-{hi:g}'


def _make_drm(ctx, det, t_mean, mat_type=2):
    from gbm_drm_gen import DRMGenTTE, BALROG_DRM
    drm_gen = DRMGenTTE(
        tte_file=_tte_path(ctx, det), det_name=det, time=float(t_mean),
        poshist=_poshist_path(ctx), T0=ctx.fermi_met, mat_type=mat_type)
    return BALROG_DRM(drm_gen, ctx.ra, ctx.dec)


def _ts_builder(ctx, det, rsp, bkg_file=None, verbose=False):
    from threeML import TimeSeriesBuilder
    kw = dict(
        tte_file=_tte_path(ctx, det), rsp_file=rsp,
        trigger_time=ctx.fermi_met, poly_order=-1, unbinned=False,
        verbose=verbose,
    )
    if bkg_file and os.path.isfile(bkg_file):
        kw['restore_background'] = bkg_file
    return TimeSeriesBuilder.from_gbm_tte(det, **kw)


def make_3ml_model(ctx, model_name='cpl', fixed_params=None):
    from astromodels import (
        Band, Cutoff_powerlaw, Log_uniform_prior, Model, PointSource,
        Powerlaw, SmoothlyBrokenPowerLaw, Uniform_prior,
    )
    fixed_params = fixed_params or {}
    _, astro = _canon_model(model_name)

    if astro == 'Cutoff_powerlaw':
        spec = Cutoff_powerlaw()
        spec.piv = 1.0
        spec.piv.fix = True
        if 'alpha' in fixed_params:
            spec.index = float(fixed_params['alpha'])
            spec.index.fix = True
        else:
            spec.index.bounds = (-3.0, 1.0)
            spec.index.prior = Uniform_prior(lower_bound=-3.0, upper_bound=1.0)
        spec.K.bounds = (1e-8, 1e4)
        spec.K.prior = Log_uniform_prior(lower_bound=1e-8, upper_bound=1e4)
        ep_lo, ep_hi = 10.0 ** LOG_EP_PRIOR[0], 10.0 ** LOG_EP_PRIOR[1]
        spec.xc.bounds = (ep_lo, ep_hi)
        spec.xc.prior = Log_uniform_prior(lower_bound=ep_lo, upper_bound=ep_hi)
    elif astro == 'Band':
        spec = Band()
        spec.K.bounds = (1e-8, 1e4)
        spec.K.prior = Log_uniform_prior(lower_bound=1e-8, upper_bound=1e4)
        spec.alpha.bounds = (-1.5, 1.0)
        spec.alpha.prior = Uniform_prior(lower_bound=-1.5, upper_bound=1.0)
        spec.beta.bounds = (-5.0, -1.6)
        spec.beta.prior = Uniform_prior(lower_bound=-5.0, upper_bound=-1.6)
        ep_lo, ep_hi = 10.0 ** LOG_EP_PRIOR[0], 10.0 ** LOG_EP_PRIOR[1]
        spec.xp.bounds = (ep_lo, ep_hi)
        spec.xp.prior = Log_uniform_prior(lower_bound=ep_lo, upper_bound=ep_hi)
    elif astro == 'Powerlaw':
        spec = Powerlaw()
        spec.K.bounds = (1e-8, 1e4)
        spec.K.prior = Log_uniform_prior(lower_bound=1e-8, upper_bound=1e4)
        spec.index.bounds = (-3.0, 1.0)
        spec.index.prior = Uniform_prior(lower_bound=-3.0, upper_bound=1.0)
    elif astro == 'SmoothlyBrokenPowerLaw':
        spec = SmoothlyBrokenPowerLaw()
        spec.K.bounds = (1e-8, 1e4)
        spec.K.prior = Log_uniform_prior(lower_bound=1e-8, upper_bound=1e4)
        spec.alpha.bounds = (-1.5, 2.0)
        spec.alpha.prior = Uniform_prior(lower_bound=-1.5, upper_bound=2.0)
        spec.beta.bounds = (-5.0, -1.6)
        spec.beta.prior = Uniform_prior(lower_bound=-5.0, upper_bound=-1.6)
        spec.break_energy.bounds = (10.0, None)
        spec.break_energy.prior = Log_uniform_prior(lower_bound=10.0, upper_bound=1e4)
    else:
        raise ValueError(f'Unsupported 3ML model {model_name!r} ({astro})')

    src = ctx.name.replace('-', '_')
    return Model(PointSource(src, ra=ctx.ra, dec=ctx.dec, spectral_shape=spec))


def _to_plugin(ts, ctx, det):
    spec = ts.to_spectrumlike()
    spec.set_active_measurements(_energy_str(ctx, det))
    min_cts = 4
    if getattr(ctx, 'spec_rebn', None):
        min_cts = max(1, int(ctx.spec_rebn.get('min_sigma', 2)) * 2)
    try:
        spec.rebin_on_background(min_number_of_counts=min_cts)
    except Exception as exc:
        print(f'    rebin skipped for {det}: {exc}')
    return spec


def _save_plots(ba, ts_map, out_dir, ctx):
    os.makedirs(out_dir, exist_ok=True)
    try:
        from threeML import display_spectrum_model_counts
        fig = display_spectrum_model_counts(ba, min_rate=10, step=False)
        if fig is not None:
            fig.savefig(os.path.join(out_dir, 'ctsspec.png'), dpi=120,
                        bbox_inches='tight')
        plt.close('all')
    except Exception as exc:
        print(f'  WARNING: count-spectrum plot failed: {exc}')
    try:
        t_lo, t_hi = _lc_view_limits(ctx)
        _det, ts = next(iter(ts_map.items()))
        fig = ts.view_lightcurve(t_lo, t_hi)
        if fig is not None:
            fig.savefig(os.path.join(out_dir, 'lc.png'), dpi=120,
                        bbox_inches='tight')
        plt.close('all')
    except Exception as exc:
        print(f'  WARNING: LC plot failed: {exc}')
    try:
        ba.results.corner_plot()
        plt.savefig(os.path.join(out_dir, 'corner.png'), dpi=100,
                    bbox_inches='tight')
        plt.close('all')
    except Exception:
        pass


def _fit_interval(ctx, dets, t_start, t_stop, out_dir, model_name, nlive,
                  background_interval, force, mat_type=2, fp=None):
    from threeML import BayesianAnalysis, DataList, silence_warnings
    silence_warnings()
    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.join(out_dir, f'{ctx.name}-')
    if os.path.isfile(_weights_path(prefix)) and not force:
        print(f'  skip existing 3ML fit: {out_dir}')
        return out_dir

    t_mean = 0.5 * (float(t_start) + float(t_stop))
    bkg = _bkg_intervals(ctx, background_interval)
    print(f'  3ML bkg (heapy LC pads − burst): {bkg}')
    plugins = []
    ts_map = {}
    for det in dets:
        print(f'  [3ML] {det} DRM @ t={t_mean:.2f}s  src={t_start:.3g}–{t_stop:.3g}')
        rsp = _make_drm(ctx, det, t_mean, mat_type=mat_type)
        ts = _ts_builder(ctx, det, rsp)
        ts.set_background_interval(*bkg)
        bkg_file = os.path.join(out_dir, f'{ctx.name}_{det}_bkg.h5')
        ts.save_background(bkg_file, overwrite=True)
        ts.set_active_time_interval(f'{t_start}-{t_stop}')
        ts_map[det] = ts
        plugins.append(_to_plugin(ts, ctx, det))

    ba = BayesianAnalysis(make_3ml_model(ctx, model_name), DataList(*plugins))
    ba.set_sampler('multinest', share_spectrum=True)
    ba.sampler.setup(n_live_points=int(nlive), chain_name=prefix)
    ba.sample()
    _save_plots(ba, ts_map, out_dir, ctx)
    write_meta(os.path.join(out_dir, 'pipeline_meta.json'), {
        'backend': '3ML',
        'model': model_name,
        'fp': fp,
        't_start': t_start,
        't_stop': t_stop,
        'dets': list(dets),
        'nlive': nlive,
        'bkg': bkg,
        'docs': DOCS_URL,
    })
    print(f'  Saved 3ML posterior: {_weights_path(prefix)}')
    return out_dir


def _quantile_pm(series):
    arr = np.asarray(series, dtype=float)
    arr = arr[np.isfinite(arr)]
    med = float(np.median(arr))
    lo = float(np.percentile(arr, 16))
    hi = float(np.percentile(arr, 84))
    return med, med - lo, hi - med


def _posterior_to_row(samples, model_name, t_start=None, t_stop=None, slice_idx=None):
    tag, astro = _canon_model(model_name)
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    n = samples.shape[1]
    n_par = max(n - 1, 1)
    if astro == 'Cutoff_powerlaw':
        names = ['A', 'alpha', 'xc']
    elif astro == 'Band':
        names = ['A', 'alpha', 'xp', 'beta']
    elif astro == 'Powerlaw':
        names = ['A', 'alpha']
    else:
        names = [f'p{i}' for i in range(n_par)]
    cols = list(names)
    if len(cols) < n_par:
        cols.extend(f'p{i}' for i in range(len(cols), n_par))
    cols = cols[:n_par] + ['log_likelihood']
    df = pd.DataFrame(samples, columns=cols)
    if astro == 'Cutoff_powerlaw' and {'xc', 'alpha'}.issubset(df.columns):
        df['Ep'] = df['xc'] * (2.0 + df['alpha'])
        df['log_Ep'] = np.log10(np.clip(df['Ep'], 1e-30, None))
    elif astro == 'Band' and 'xp' in df.columns:
        df['Ep'] = df['xp']
        df['log_Ep'] = np.log10(np.clip(df['Ep'], 1e-30, None))
    vFv_model = 'cpl' if astro == 'Cutoff_powerlaw' else (
        'band' if astro == 'Band' else tag)
    if {'alpha', 'Ep', 'A'}.issubset(df.columns):
        df['vFv'] = compute_vFv(df['alpha'], df['Ep'], df['A'], vFv_model)
    df_sorted = df.sort_values(
        'log_likelihood', ascending=False).reset_index(drop=True)
    n_1sigma = max(int(0.6827 * len(df_sorted)), 1)
    df_1s = df_sorted.iloc[:n_1sigma].copy()
    row = get_model_params(vFv_model, df_1s)
    if 'alpha' in row:
        row['alpha_ml'] = row['alpha']
    if 'Ep_best' in row:
        row['Ep_ml'] = row['Ep_best']
    if 'A' in row:
        row['A_ml'] = row['A']
    # Equal-weight 16/50/84 (3ML corner / GCN-style), not max-L
    if 'alpha' in df.columns:
        best, low, high = _quantile_pm(df['alpha'])
        row.update({'alpha': best, 'alpha_low': low, 'alpha_high': high})
    if 'A' in df.columns:
        row['A'] = float(np.median(df['A']))
    if 'Ep' in df.columns:
        best, low, high = _quantile_pm(df['Ep'])
        row.update({
            'Ep_best': best, 'Ep_low': low, 'Ep_high': high,
            'sigma_Ep': 0.5 * (low + high),
        })
    if 'vFv' in df.columns:
        best, low, high = _quantile_pm(df['vFv'])
        row.update({
            'vFv_best': best, 'vFv_low': low, 'vFv_high': high,
            'sigma_vFv': 0.5 * (low + high),
        })
    if 'log_Ep' in df.columns:
        flags = ep_constraint_flags(df_1s['log_Ep'], prior=LOG_EP_PRIOR)
        row.update(flags)
        for flag in ('ep_constrained', 'log_Ep_hits_prior_low', 'log_Ep_hits_prior_high'):
            if flag in row:
                row[flag] = int(bool(row[flag]))
    row['model'] = tag
    row['backend'] = '3ML'
    if slice_idx is not None:
        row['slice'] = int(slice_idx)
    if t_start is not None:
        row['t_start'] = float(t_start)
        row['t_stop'] = float(t_stop)
    return row


def extract_params_3ml(ctx, model_name='cpl', mode='tintegrated', nlive=1000,
                       include_bgo=False, background_interval=None, fp=None,
                       save_root=None):
    tag, _ = _canon_model(model_name)
    model_root = _fit_root(ctx, tag, mode)
    dets = fit_detectors(ctx.sel_dets, include_bgo=include_bgo) or list(ctx.sel_dets or [])
    if fp is None:
        fp = _3ml_fingerprint(
            ctx, tag, mode, nlive, dets, include_bgo, background_interval)
    if save_root is None:
        save_root = _active_3ml_dir(model_root, fp, ctx.name, tag, mode)
    rows = []
    if mode in ('tresolved', 'tres'):
        slices = ctx.time_slices or [(ctx.t1, ctx.t2)]
        for i, (t1, t2) in enumerate(slices, 1):
            prefix = os.path.join(save_root, f'slice_{i:02d}', f'{ctx.name}-')
            path = _weights_path(prefix)
            if not os.path.isfile(path):
                print(f'  skip slice {i}: no {path}')
                continue
            rows.append(_posterior_to_row(
                np.loadtxt(path), tag, t1, t2, slice_idx=i))
        key = (
            f'{tag}_3ML_{ctx.name}' if save_root == model_root
            else f'{tag}_3ML_{fp}_{ctx.name}')
    else:
        prefix = os.path.join(save_root, f'{ctx.name}-')
        path = _weights_path(prefix)
        if not os.path.isfile(path):
            print(f'  no 3ML tint weights: {path}')
            return None
        rows.append(_posterior_to_row(
            np.loadtxt(path), tag, ctx.t1, ctx.t2))
        key = (
            f'tint_{tag}_3ML_{ctx.name}' if save_root == model_root
            else f'tint_{tag}_3ML_{fp}_{ctx.name}')
    if not rows:
        return None
    for row in rows:
        row['include_bgo'] = int(bool(include_bgo))
        for flag in ('ep_constrained', 'log_Ep_hits_prior_low', 'log_Ep_hits_prior_high'):
            if flag in row:
                row[flag] = int(bool(row[flag]))
    df = pd.DataFrame(rows)
    h5 = _layout_3ml_h5(ctx)
    os.makedirs(os.path.dirname(h5), exist_ok=True)
    df.to_hdf(h5, key=key, mode='w' if not os.path.isfile(h5) else 'a')
    print(f'  ✓ 3ML params → {h5}  key={key}')
    if 'ep_constrained' in df.columns:
        print(f'  ep_constrained={list(df["ep_constrained"])}')
    if mode in ('tresolved', 'tres') and 'slice' in df.columns:
        from plotting.params import save_tres_params_plot
        plot_path = os.path.join(save_root, 'tres_params.png')
        save_tres_params_plot(
            df, plot_path,
            title=f'{ctx.name} 3ML {tag} tres')
    return df


def run_3ml(ctx, model_name='cpl', mode='tintegrated', nlive=1000,
            include_bgo=False, force=False, background_interval=None,
            mat_type=2, bin_method='custom', bin_kwargs=None):
    """Run 3ML tint or tres. Requires threeML + gbm_drm_gen in the image.

    Tres binning (``bin_method``) follows TimeSeriesBuilder.create_time_bins
    on the brightest NaI, then reuses the same intervals for all detectors.
    ``custom`` keeps ``ctx.time_slices`` (catalog / UI boundaries).
    """
    ok, info = threeml_available()
    if not ok:
        raise RuntimeError(
            f'threeML is not installed ({info}). Rebuild the gbm image.')
    if ctx.gbm_rtv is None or ctx.fermi_met is None:
        retrieve(ctx)
    resolve_detectors(ctx)
    dets = fit_detectors(ctx.sel_dets, include_bgo=include_bgo) or list(ctx.sel_dets)
    if not dets:
        raise ValueError(f'{ctx.name}: no detectors for 3ML fit')
    tag, _ = _canon_model(model_name)
    model_root = _fit_root(ctx, tag, mode)
    os.makedirs(model_root, exist_ok=True)
    bin_method = canon_bin_method(bin_method) if mode in ('tresolved', 'tres') else 'custom'
    bin_info = None
    slices = ctx.time_slices or [(ctx.t1, ctx.t2)]
    if mode in ('tresolved', 'tres'):
        slices, bin_info = resolve_3ml_bins(
            ctx, dets, bin_method=bin_method,
            background_interval=background_interval,
            bin_kwargs=bin_kwargs, mat_type=mat_type,
            persist_dir=_layout_3ml_tres(ctx))
        ctx.time_slices = list(slices)
        from .slices import slices_to_boundaries
        ctx.slice_boundaries = slices_to_boundaries(slices)
    fp = _3ml_fingerprint(
        ctx, tag, mode, nlive, dets, include_bgo, background_interval,
        bin_method=bin_method, slices=slices)
    save_root = _resolve_3ml_dir(
        model_root, fp, tag, mode, force=force)
    os.makedirs(save_root, exist_ok=True)
    print(
        f'\n=== 3ML {ctx.name} mode={mode} model={tag} dets={dets} '
        f'nlive={nlive} bin={bin_method} fp={fp} ===')
    print(f'  savepath={save_root}')
    if save_root != model_root:
        print(f'  Versioned 3ML dir: {save_root}')

    if mode in ('tresolved', 'tres'):
        for i, (t1, t2) in enumerate(slices, 1):
            out = os.path.join(save_root, f'slice_{i:02d}')
            _fit_interval(
                ctx, dets, t1, t2, out, tag, nlive,
                background_interval, force, mat_type=mat_type, fp=fp)
    else:
        _fit_interval(
            ctx, dets, ctx.t1, ctx.t2, save_root, tag, nlive,
            background_interval, force, mat_type=mat_type, fp=fp)

    commit_meta(model_root, f'canonical_fit_{tag}', fp, {
        'model': tag, 'mode': mode, 'nlive': nlive, 'dets': dets,
        'include_bgo': include_bgo,
        'bkg': _bkg_intervals(ctx, background_interval),
        'bin_method': bin_method,
        'slices': slices,
    })
    df = extract_params_3ml(
        ctx, model_name=tag, mode=mode, nlive=nlive,
        include_bgo=include_bgo, background_interval=background_interval,
        fp=fp, save_root=save_root)
    return {'status': 'ok', 'root': save_root, 'fp': fp, 'dets': dets,
            'bin_method': bin_method, 'slices': slices, 'bins': bin_info,
            'params': None if df is None else df.to_dict('records')}
