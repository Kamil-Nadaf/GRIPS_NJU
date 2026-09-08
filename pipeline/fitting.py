"""bayspec PGSTAT + MultiNest fitting (MySpecFit successor)."""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .mpl_setup import silence_missing_fonts
silence_missing_fonts()

from .constants import (
    ADDITIVE_SKIP,
    LATEX_TO_COL,
    LOG_EP_EDGE_TOL,
    LOG_EP_PRIOR,
    MODEL_COLS,
    PARAM_KEY_MAP,
)
from .detectors import fit_detectors
from .paths import (
    commit_meta,
    fit_fingerprint,
    active_fit_dir,
    resolve_fit_dir,
    spec_slice_name,
)
from .spectra import active_heapy_tres_dir, tint_spec_base

_MODELS = None


def _build_models_registry():
    import importlib
    add = importlib.import_module('bayspec.model.local.additive')
    models = {}
    for name in dir(add):
        if name.startswith('_') or name in ADDITIVE_SKIP:
            continue
        obj = getattr(add, name)
        if callable(obj):
            models[name] = obj
    return models


def get_models():
    global _MODELS
    if _MODELS is None:
        _MODELS = _build_models_registry()
    return _MODELS


class _ModelsProxy(dict):
    def _ensure(self):
        if not dict.__len__(self):
            dict.update(self, get_models())

    def __getitem__(self, key):
        self._ensure()
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        self._ensure()
        return dict.get(self, key, default)

    def __contains__(self, key):
        self._ensure()
        return dict.__contains__(self, key)

    def keys(self):
        self._ensure()
        return dict.keys(self)

    def __iter__(self):
        self._ensure()
        return dict.__iter__(self)

    def __len__(self):
        self._ensure()
        return dict.__len__(self)


MODELS = _ModelsProxy()


def available_models():
    """Sorted bayspec additive names. Requires bayspec (Docker)."""
    return sorted(MODELS)


def resolve_fit_dets(ctx, include_bgo=False, skip_dets=None, fit_dets=None):
    if fit_dets is not None:
        return list(fit_dets)
    return fit_detectors(ctx.sel_dets, include_bgo=include_bgo, skip_dets=skip_dets)


def latex_to_col(label):
    if label in LATEX_TO_COL:
        return LATEX_TO_COL[label]
    if label in PARAM_KEY_MAP:
        return label
    cleaned = (
        str(label)
        .replace('$', '')
        .replace('\\', '')
        .replace('{', '')
        .replace('}', '')
        .strip()
    )
    return LATEX_TO_COL.get(str(label), cleaned or str(label))


def parse_unif_prior(prior_str):
    import re
    if not prior_str:
        return None
    m = re.search(r'unif\(\s*([^,]+)\s*,\s*([^)]+)\)', str(prior_str))
    if not m:
        return None
    try:
        return float(m.group(1)), float(m.group(2))
    except (TypeError, ValueError):
        return None


def log_ep_prior_bounds(savepath=None):
    """Read log Ep prior from bayspec model_par.json, else default unif(0, 4)."""
    if savepath:
        for fname in ('model_par.json', 'infer_par.json'):
            path = os.path.join(savepath, fname)
            if not os.path.isfile(path):
                continue
            try:
                import json
                with open(path) as f:
                    pars = json.load(f)
            except (OSError, ValueError):
                continue
            for par in pars:
                name = latex_to_col(par.get('Parameter', ''))
                if name in ('log_Ep', 'Ep'):
                    bounds = parse_unif_prior(par.get('Prior'))
                    if bounds:
                        return bounds
    return LOG_EP_PRIOR


def ep_constraint_flags(log_ep_samples, prior=None, tol=LOG_EP_EDGE_TOL):
    """Whether the 1σ log-Ep interval hits the prior edge."""
    if log_ep_samples is None:
        return {}
    arr = np.asarray(log_ep_samples, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {}
    lo, hi = prior if prior is not None else LOG_EP_PRIOR
    hits_low = bool(np.min(arr) <= lo + tol)
    hits_high = bool(np.max(arr) >= hi - tol)
    return {
        'ep_constrained': not (hits_low or hits_high),
        'log_Ep_hits_prior_low': hits_low,
        'log_Ep_hits_prior_high': hits_high,
        'log_Ep_1sigma_min': float(np.min(arr)),
        'log_Ep_1sigma_max': float(np.max(arr)),
        'log_Ep_prior_lo': float(lo),
        'log_Ep_prior_hi': float(hi),
    }


def posterior_columns(savepath, model_name, fixed_params=None):
    """Column names for ``1-post_equal_weights.dat`` (free params + logZ)."""
    fixed_params = fixed_params or {}
    if model_name in MODEL_COLS:
        free = [p for p in MODEL_COLS[model_name] if p not in fixed_params]
        return free + ['log_likelihood']
    path = os.path.join(savepath, 'post_free_par.json')
    if os.path.isfile(path):
        import json
        with open(path) as f:
            pars = json.load(f)
        names = [latex_to_col(p.get('Parameter', f'p{i}')) for i, p in enumerate(pars)]
        names = [n for n in names if n not in fixed_params]
        return names + ['log_likelihood']
    return None


def compute_vFv(alpha, Ep, A, model_name):
    """nuFnu at peak. epiv = pivot_energy, bayspec default per model."""
    epiv = {'cpl': 1.0, 'band': 100.0, 'bpl': 100.0, 'sbpl': 100.0, 'grbm': 100.0}.get(
        model_name, 100.0)
    return 1.602e-9 * A * (Ep ** (alpha + 2)) * epiv ** (-alpha) * np.exp(-(2 + alpha))


def make_model(model_name, fixed_params=None):
    if fixed_params is None:
        fixed_params = {}
    factory = MODELS.get(model_name)
    if factory is None:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {available_models()}")
    model = factory()
    for pname, pval in fixed_params.items():
        key = PARAM_KEY_MAP.get(pname, pname)
        if key in model.params:
            model.params[key].frozen_at(pval)
        elif hasattr(model, pname):
            setattr(model, pname, pval)
        else:
            print(f"  WARNING: param '{pname}' not found in {model_name}")
    return model


def save_fit_plots(post, model, savepath):
    from bayspec import Plot

    Plot.infer(post, style='CE', ploter='matplotlib').fig.savefig(
        f'{savepath}/ctsspec.png', dpi=100, bbox_inches='tight')
    Plot.infer(post, style='NE', ploter='matplotlib').fig.savefig(
        f'{savepath}/phtspec.png', dpi=100, bbox_inches='tight')
    modelplot = Plot.model(ploter='matplotlib', style='vFv', post=True)
    modelplot.add_model(model, E=np.logspace(1, 3, 100))
    modelplot.fig.savefig(f'{savepath}/model.png', dpi=100, bbox_inches='tight')
    for style, fname in (('CE', 'ctsspec_infer.html'), ('NE', 'phtspec_infer.html')):
        try:
            fig = Plot.infer(post, style=style, ploter='plotly')
            fig.fig.write_html(f'{savepath}/{fname}')
        except Exception as e:
            print(f'  WARNING: could not save {fname}: {e}')
    plt.close('all')


def _dataunit(spec_base, det, spec_rebn, ctx=None):
    from bayspec import DataUnit
    if ctx is not None:
        notc = list(ctx.nai_energy if det[0] == 'n' else ctx.bgo_energy)
    else:
        notc = [8, 900] if det[0] == 'n' else [300, 38000]
    return DataUnit(
        src=f'{spec_base}.src', bkg=f'{spec_base}.bkg', rsp=f'{spec_base}.rsp',
        notc=notc, stat='pgstat', rebn=dict(spec_rebn))


def build_fit_data(ctx, mode='tintegrated', slice_index=None,
                  include_bgo=False, skip_dets=None, fit_dets=None):
    from bayspec import Data

    dets = resolve_fit_dets(
        ctx, include_bgo=include_bgo, skip_dets=skip_dets, fit_dets=fit_dets)

    if mode == 'tintegrated':
        data_list = []
        for det in dets:
            spec_base = tint_spec_base(ctx, det)
            data_list.append((det, _dataunit(spec_base, det, ctx.spec_rebn, ctx=ctx)))
        return Data(data_list)

    if mode == 'tresolved':
        if slice_index is None:
            raise ValueError("slice_index required for mode='tresolved'")
        heapy_root = active_heapy_tres_dir(ctx)
        t_start, t_stop = ctx.time_slices[slice_index - 1]
        spec_name = spec_slice_name(t_start, t_stop)
        slice_dir = os.path.join(heapy_root, f'slice_{slice_index:02d}')
        data_list = []
        for det in dets:
            spec_base = os.path.join(slice_dir, f'{spec_name}_{det}')
            data_list.append((det, _dataunit(spec_base, det, ctx.spec_rebn, ctx=ctx)))
        return Data(data_list)

    raise ValueError(f'Unknown mode: {mode}')


def get_fit_savepath(ctx, model_name, mode='tintegrated', slice_index=None,
                     fixed_params=None, nlive=1000, include_bgo=False,
                     skip_dets=None, fit_dets=None):
    if fixed_params is None:
        fixed_params = {}
    dets = resolve_fit_dets(
        ctx, include_bgo=include_bgo, skip_dets=skip_dets, fit_dets=fit_dets)
    fit_fp = fit_fingerprint(ctx, model_name, fixed_params, nlive, fit_dets=dets)
    if mode == 'tintegrated':
        model_root = os.path.join(ctx.paths.bayspec_tintegrated_path, model_name)
        return active_fit_dir(model_root, fit_fp, f'canonical_fit_{model_name}')
    if mode == 'tresolved':
        if slice_index is None:
            raise ValueError("slice_index required for mode='tresolved'")
        model_root = os.path.join(ctx.paths.bayspec_tresolved_path, model_name)
        tres_root = active_fit_dir(model_root, fit_fp, f'canonical_fit_{model_name}')
        return os.path.join(tres_root, f'slice_{slice_index:02d}')
    raise ValueError(f'Unknown mode: {mode}')


def reload_posterior(ctx, savepath, model_name, fixed_params=None, nlive=None,
                     include_bgo=False, skip_dets=None, fit_dets=None):
    import json
    from bayspec import BayesInfer

    if fixed_params is None:
        fixed_params = {}
    if nlive is None:
        nlive_path = os.path.join(savepath, '1-nlive.json')
        if os.path.isfile(nlive_path):
            with open(nlive_path) as f:
                nlive = json.load(f)
        else:
            nlive = 1000
    mode = 'tresolved' if 'slice_' in os.path.basename(savepath.rstrip('/')) else 'tintegrated'
    slice_index = None
    if mode == 'tresolved':
        slice_index = int(os.path.basename(savepath.rstrip('/')).split('_')[1])
    data = build_fit_data(
        ctx, mode=mode, slice_index=slice_index,
        include_bgo=include_bgo, skip_dets=skip_dets, fit_dets=fit_dets)
    model = make_model(model_name, fixed_params)
    infer = BayesInfer([(data, model)])
    post = infer.multinest(nlive=nlive, resume=True, verbose=False, savepath=savepath)
    return post, model


def plot_infer_fit(ctx, model_name='cpl', mode='tintegrated', slice_index=None,
                   style='CE', fixed_params=None, nlive=1000, savepath=None,
                   display=True, include_bgo=False, skip_dets=None, fit_dets=None):
    from bayspec import Plot

    savepath = savepath or get_fit_savepath(
        ctx, model_name, mode=mode, slice_index=slice_index,
        fixed_params=fixed_params, nlive=nlive, include_bgo=include_bgo,
        skip_dets=skip_dets, fit_dets=fit_dets)
    if not os.path.isdir(savepath):
        raise FileNotFoundError(f'Fit not found: {savepath}')
    post, model = reload_posterior(
        ctx, savepath, model_name, fixed_params=fixed_params, nlive=nlive,
        include_bgo=include_bgo, skip_dets=skip_dets, fit_dets=fit_dets)
    fig = Plot.infer(post, style=style, ploter='plotly')
    if display:
        try:
            from IPython.display import HTML, display as ipy_display
            ipy_display(HTML(fig.fig.to_html(include_plotlyjs='cdn')))
        except ImportError:
            fig.fig.show()
    return fig


def _run_multinest(data, model, savepath, nlive):
    from bayspec import BayesInfer

    data.save(savepath)
    model.save(savepath)
    infer = BayesInfer([(data, model)])
    infer.save(savepath)
    post = infer.multinest(nlive=nlive, resume=False, verbose=False, savepath=savepath)
    post.save(savepath)
    save_fit_plots(post, model, savepath)
    return post


def fit_tintegrated(ctx, model_name='cpl', fixed_params=None, nlive=1000,
                    skip_dets=None, force=False, include_bgo=False, fit_dets=None):
    from bayspec import Data

    if fixed_params is None:
        fixed_params = {}
    dets = resolve_fit_dets(
        ctx, include_bgo=include_bgo, skip_dets=skip_dets, fit_dets=fit_dets)
    if len(dets) < 2:
        raise ValueError(
            f'<2 detectors for tint fit ({dets}). '
            'Need at least two NaI, or pass include_bgo=True.')

    fit_fp = fit_fingerprint(ctx, model_name, fixed_params, nlive, fit_dets=dets)
    model_root = os.path.join(ctx.paths.bayspec_tintegrated_path, model_name)
    savepath = resolve_fit_dir(
        model_root, fit_fp, f'canonical_fit_{model_name}', force=force)
    os.makedirs(savepath, exist_ok=True)

    print(f'\nFitting {model_name} (tint) | {ctx.name} | fp={fit_fp}')
    print(f'  catalog dets={ctx.sel_dets}  fit dets={dets}  include_bgo={include_bgo}')
    print(f'  savepath={savepath}')
    if savepath != model_root:
        print(f'  Versioned fit dir: {savepath}')
    weights = os.path.join(savepath, '1-post_equal_weights.dat')
    if os.path.isfile(weights) and not force:
        print(f'  Skip (exists). Pass force=True to rerun.')
        return savepath

    data_list = []
    for det in dets:
        try:
            spec_base = tint_spec_base(ctx, det)
            data_list.append((det, _dataunit(spec_base, det, ctx.spec_rebn, ctx=ctx)))
        except Exception as e:
            print(f'  Skip {det}: {e}')

    if len(data_list) < 2:
        raise ValueError(f'<2 detectors for tint fit ({len(data_list)})')

    data = Data(data_list)
    model = make_model(model_name, fixed_params)
    _run_multinest(data, model, savepath, nlive)
    commit_meta(model_root, f'canonical_fit_{model_name}', fit_fp, {
        'model': model_name, 'fixed_params': fixed_params, 'nlive': nlive,
        'spec_rebn': ctx.spec_rebn, 'sel_dets': list(ctx.sel_dets),
        'fit_dets': dets, 'include_bgo': include_bgo,
    })
    print(f'✓ {model_name} -> {savepath}')
    return savepath


def _fit_one_slice(payload):
    from bayspec import Data
    from .context import GRBContext
    from .download import retrieve

    ctx = GRBContext.from_dict(payload['ctx'])
    retrieve(ctx)
    model_name = payload['model_name']
    fixed_params = payload['fixed_params']
    nlive = payload['nlive']
    slice_index = payload['slice_index']
    t_start, t_stop = payload['t_start'], payload['t_stop']
    savepath = payload['savepath']
    slice_dir = payload['slice_dir']
    spec_name = spec_slice_name(t_start, t_stop)
    force = payload.get('force', False)
    dets = payload.get('fit_dets') or ctx.sel_dets

    os.makedirs(savepath, exist_ok=True)
    weights = os.path.join(savepath, '1-post_equal_weights.dat')
    if os.path.isfile(weights) and not force:
        print(f'  Skip slice {slice_index} (exists)')
        return {'slice': slice_index, 'ok': True, 'skipped': True}

    print(f'\n{"=" * 50}')
    print(f'Slice {slice_index}: [{t_start}, {t_stop}] s')
    print(f'{"=" * 50}')
    try:
        data_list = []
        for det in dets:
            try:
                spec_base = os.path.join(slice_dir, f'{spec_name}_{det}')
                data_list.append((det, _dataunit(spec_base, det, ctx.spec_rebn, ctx=ctx)))
            except Exception as e:
                print(f'  Skip {det}: {e}')
        if len(data_list) < 2:
            print(f'  Skip slice {slice_index}: <2 detectors')
            return {'slice': slice_index, 'ok': False, 'error': '<2 dets'}
        data = Data(data_list)
        model = make_model(model_name, fixed_params)
        _run_multinest(data, model, savepath, nlive)
        print(f'  ✓ slice {slice_index}')
        return {'slice': slice_index, 'ok': True}
    except Exception as e:
        print(f'  Error slice {slice_index}: {e}')
        return {'slice': slice_index, 'ok': False, 'error': str(e)}


def fit_all_time_slices(ctx, model_name='cpl', fixed_params=None, nlive=1000,
                       force=False, n_workers=None, include_bgo=False,
                       skip_dets=None, fit_dets=None):
    from .parallel import map_parallel

    if fixed_params is None:
        fixed_params = {}
    if not ctx.time_slices:
        raise ValueError('time_slices not set')

    dets = resolve_fit_dets(
        ctx, include_bgo=include_bgo, skip_dets=skip_dets, fit_dets=fit_dets)
    if len(dets) < 2:
        raise ValueError(
            f'<2 detectors for tres fit ({dets}). '
            'Need at least two NaI, or pass include_bgo=True.')

    fit_fp = fit_fingerprint(ctx, model_name, fixed_params, nlive, fit_dets=dets)
    heapy_root = active_heapy_tres_dir(ctx)
    model_root = os.path.join(ctx.paths.bayspec_tresolved_path, model_name)
    tres_fit_root = resolve_fit_dir(
        model_root, fit_fp, f'canonical_fit_{model_name}', force=force)
    os.makedirs(tres_fit_root, exist_ok=True)

    print(f'\nFitting {model_name} (tresolved) | {ctx.name} | fp={fit_fp}')
    print(f'  catalog dets={ctx.sel_dets}  fit dets={dets}  include_bgo={include_bgo}')
    print(f'  savepath={tres_fit_root}')

    jobs = []
    for slice_index, (t_start, t_stop) in enumerate(ctx.time_slices, 1):
        jobs.append({
            'ctx': ctx.to_dict(),
            'model_name': model_name,
            'fixed_params': fixed_params,
            'nlive': nlive,
            'slice_index': slice_index,
            't_start': t_start,
            't_stop': t_stop,
            'savepath': os.path.join(tres_fit_root, f'slice_{slice_index:02d}'),
            'slice_dir': os.path.join(heapy_root, f'slice_{slice_index:02d}'),
            'force': force,
            'fit_dets': dets,
        })
    map_parallel(_fit_one_slice, jobs, n_workers=n_workers, desc='fit-slice')

    commit_meta(model_root, f'canonical_fit_{model_name}', fit_fp, {
        'model': model_name, 'fixed_params': fixed_params, 'nlive': nlive,
        'spec_rebn': ctx.spec_rebn, 'sel_dets': list(ctx.sel_dets),
        'fit_dets': dets, 'include_bgo': include_bgo,
    })
    print('\nAll slices fitted!')
    return tres_fit_root


def _interval_stats(series):
    best = float(series.iloc[0])
    lo, hi = float(np.min(series)), float(np.max(series))
    return best, best - lo, hi - best, lo, hi


def quantile_pm(series):
    """Median and 16/84 deviations (equal-weight posterior)."""
    arr = np.asarray(series, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float('nan'), float('nan'), float('nan')
    med = float(np.median(arr))
    lo = float(np.percentile(arr, 16))
    hi = float(np.percentile(arr, 84))
    return med, med - lo, hi - med


def apply_equal_weight_quantiles(row, df):
    """Overwrite α / Ep / A / vFv with 16/50/84; keep max-L as ``*_ml``."""
    if row is None:
        return row
    if 'alpha' in row:
        row['alpha_ml'] = row['alpha']
    if 'Ep_best' in row:
        row['Ep_ml'] = row['Ep_best']
    if 'A' in row:
        row['A_ml'] = row['A']
    if 'alpha' in df.columns:
        best, low, high = quantile_pm(df['alpha'])
        row.update({'alpha': best, 'alpha_low': low, 'alpha_high': high})
    if 'A' in df.columns:
        row['A'] = float(np.median(np.asarray(df['A'], dtype=float)))
    if 'Ep' in df.columns:
        best, low, high = quantile_pm(df['Ep'])
        row.update({
            'Ep_best': best, 'Ep_low': low, 'Ep_high': high,
            'sigma_Ep': 0.5 * (low + high),
        })
    if 'vFv' in df.columns:
        best, low, high = quantile_pm(df['vFv'])
        row.update({
            'vFv_best': best, 'vFv_low': low, 'vFv_high': high,
            'sigma_vFv': 0.5 * (low + high),
        })
    return row


def get_model_params(model_name, df_1sigma, prior=None):
    """Summarize the 1σ posterior. Known models keep historical column names."""
    if len(df_1sigma) == 0:
        raise ValueError(
            f"get_model_params('{model_name}'): empty 1-sigma sample (0 rows)")

    row = {}
    known = model_name in MODEL_COLS and 'alpha' in df_1sigma.columns

    if known:
        best, low, high, _, _ = _interval_stats(df_1sigma['alpha'])
        row.update({'alpha': best, 'alpha_low': low, 'alpha_high': high})

        if 'beta' in df_1sigma.columns:
            best, low, high, _, _ = _interval_stats(df_1sigma['beta'])
            row.update({'beta': best, 'beta_low': low, 'beta_high': high})

        if 'A' in df_1sigma.columns:
            row['A'] = float(df_1sigma.iloc[0]['A'])

        if 'Ep' in df_1sigma.columns:
            best, low, high, lo, hi = _interval_stats(df_1sigma['Ep'])
            row.update({
                'Ep_best': best, 'Ep_low': low, 'Ep_high': high,
                'sigma_Ep': (hi - lo) / 2,
            })
        if 'vFv' in df_1sigma.columns:
            best, low, high, lo, hi = _interval_stats(df_1sigma['vFv'])
            row.update({
                'vFv_best': best, 'vFv_low': low, 'vFv_high': high,
                'sigma_vFv': (hi - lo) / 2,
            })
    else:
        skip = {'log_likelihood'}
        for col in df_1sigma.columns:
            if col in skip or not np.issubdtype(df_1sigma[col].dtype, np.number):
                continue
            best, low, high, _, _ = _interval_stats(df_1sigma[col])
            row[col] = best
            row[f'{col}_low'] = low
            row[f'{col}_high'] = high
        if 'Ep' in df_1sigma.columns:
            best, low, high, _, _ = _interval_stats(df_1sigma['Ep'])
            row.update({'Ep_best': best, 'Ep_low': low, 'Ep_high': high})

    if 'log_Ep' in df_1sigma.columns:
        flags = ep_constraint_flags(df_1sigma['log_Ep'], prior=prior)
        row.update(flags)
        if flags and not flags.get('ep_constrained', True):
            print(
                f'  WARNING: log Ep 1σ hits prior '
                f'[{flags.get("log_Ep_prior_lo")}, {flags.get("log_Ep_prior_hi")}] '
                f'(min={flags["log_Ep_1sigma_min"]:.3f}, '
                f'max={flags["log_Ep_1sigma_max"]:.3f}) — unconstrained')
        elif flags:
            print(
                f'  log Ep constrained: 1σ '
                f'[{flags["log_Ep_1sigma_min"]:.3f}, {flags["log_Ep_1sigma_max"]:.3f}]')
    return row


def _load_posterior_row(savepath, model_name, fixed_params):
    fpath = os.path.join(savepath, '1-post_equal_weights.dat')
    if not os.path.exists(fpath):
        return None
    try:
        samples = np.loadtxt(fpath)
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        columns = posterior_columns(savepath, model_name, fixed_params)
        n_cols = samples.shape[1]
        if columns is None or len(columns) != n_cols:
            columns = [f'p{i}' for i in range(n_cols - 1)] + ['log_likelihood']
            inferred = posterior_columns(savepath, None, fixed_params)
            if inferred and len(inferred) == n_cols:
                columns = inferred
            elif model_name in MODEL_COLS:
                print(
                    f'  WARNING: column mismatch at {fpath}: '
                    f'got {n_cols} cols, expected {len(posterior_columns(savepath, model_name, fixed_params))}')
        df = pd.DataFrame(samples, columns=columns)
        for pname, pval in (fixed_params or {}).items():
            df[pname] = pval
        if 'log_likelihood' not in df.columns:
            df['log_likelihood'] = np.arange(len(df))[::-1]
        df_sorted = df.sort_values('log_likelihood', ascending=False).reset_index(drop=True)
        if 'log_Ep' in df_sorted.columns:
            df_sorted['Ep'] = 10 ** df_sorted['log_Ep']
        if 'log_A' in df_sorted.columns:
            df_sorted['A'] = 10 ** df_sorted['log_A']
        if {'alpha', 'Ep', 'A'}.issubset(df_sorted.columns):
            df_sorted['vFv'] = compute_vFv(
                df_sorted['alpha'], df_sorted['Ep'], df_sorted['A'], model_name)
        n_1sigma = max(int(0.6827 * len(df_sorted)), 1)
        prior = log_ep_prior_bounds(savepath)
        row = get_model_params(
            model_name, df_sorted.iloc[:n_1sigma].copy(), prior=prior)
        return apply_equal_weight_quantiles(row, df_sorted)
    except ValueError as e:
        print(f'  ValueError at {fpath}: {e}')
        return None


def extract_params(ctx, model_name='cpl', mode='tintegrated', fixed_params=None,
                   nlive=1000, include_bgo=False, skip_dets=None, fit_dets=None):
    if fixed_params is None:
        fixed_params = {}
    dets = resolve_fit_dets(
        ctx, include_bgo=include_bgo, skip_dets=skip_dets, fit_dets=fit_dets)
    fit_fp = fit_fingerprint(ctx, model_name, fixed_params, nlive, fit_dets=dets)

    os.makedirs(ctx.paths.bayspec_tresolved_path, exist_ok=True)
    bayspec_data = ctx.paths.bayspec_data

    if mode == 'tintegrated':
        print(f'\nExtracting {model_name} (tint) | {ctx.name} | fit_dets={dets}')
        model_root = os.path.join(ctx.paths.bayspec_tintegrated_path, model_name)
        savepath = active_fit_dir(model_root, fit_fp, f'canonical_fit_{model_name}')
        row = _load_posterior_row(savepath, model_name, fixed_params)
        if row is None:
            print(f'  Fit not found at {savepath}')
            return None
        row['include_bgo'] = int(bool(include_bgo))
        for flag in ('ep_constrained', 'log_Ep_hits_prior_low', 'log_Ep_hits_prior_high'):
            if flag in row:
                row[flag] = int(bool(row[flag]))
        if savepath == model_root:
            key = f'tint_{model_name}_{ctx.name}'
        else:
            key = f'tint_{model_name}_{fit_fp}_{ctx.name}'
        df_out = pd.DataFrame([row])
        df_out.to_hdf(bayspec_data, key=key, mode='a')
        print(f"  ✓ Saved HDF5 key '{key}'")
        if 'ep_constrained' in row:
            print(f"  ep_constrained={row['ep_constrained']}")
        return df_out

    if mode == 'tresolved':
        print(f'\nExtracting {model_name} (tresolved) | {ctx.name} | fit_dets={dets}')
        model_root = os.path.join(ctx.paths.bayspec_tresolved_path, model_name)
        tres_root = active_fit_dir(model_root, fit_fp, f'canonical_fit_{model_name}')
        rows = []
        for slice_idx in range(1, ctx.n_slices + 1):
            savepath = os.path.join(tres_root, f'slice_{slice_idx:02d}')
            row = _load_posterior_row(savepath, model_name, fixed_params)
            if row is None:
                print(f'  Skipping slice {slice_idx}: file not found')
                continue
            t_start, t_stop = ctx.time_slices[slice_idx - 1]
            for flag in ('ep_constrained', 'log_Ep_hits_prior_low', 'log_Ep_hits_prior_high'):
                if flag in row:
                    row[flag] = int(bool(row[flag]))
            rows.append({
                'slice': slice_idx, 'model': model_name,
                't_start': t_start, 't_stop': t_stop,
                'include_bgo': int(bool(include_bgo)),
                **row,
            })
        if not rows:
            print('  No slices extracted')
            return None
        df_out = pd.DataFrame(rows).sort_values('slice').reset_index(drop=True)
        from .paths import read_meta
        meta_fit = read_meta(os.path.join(ctx.paths.bayspec_tresolved_path, 'pipeline_meta.json'))
        if fit_fp == meta_fit.get(f'canonical_fit_{model_name}', fit_fp) and tres_root == model_root:
            key = model_name
        else:
            key = f'{model_name}_{fit_fp}'
        df_out.to_hdf(bayspec_data, key=key, mode='a')
        print(f"  ✓ Saved HDF5 key '{key}'")
        try:
            from plotting.params import save_tres_params_plot
            save_tres_params_plot(
                df_out, os.path.join(tres_root, 'tres_params.png'),
                title=f'{ctx.name} bayspec {model_name} tres')
        except Exception as exc:
            print(f'  WARNING: tres param plot failed: {exc}')
        return df_out

    raise ValueError(f'Unknown mode: {mode}')
