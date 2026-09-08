"""Streamlit UI — time-integrated Fermi GBM analysis (heapy + bayspec, optional 3ML).

    streamlit run ui/app.py --server.port=8501 --server.address=0.0.0.0
"""

from __future__ import annotations

import glob
import io
import os
import sys
import zipfile

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use('Agg')

import streamlit as st

from grb_config import gcn_ref
from pipeline.constants import (
    ALL_DETS,
    BGO_ENERGY,
    DATA_BASE,
    DEFAULT_STAGES,
    MODEL_COLS,
    NAI_ENERGY,
)
from pipeline.detectors import fit_detectors
from pipeline.paths import PathLayout
from pipeline.slices import boundaries_to_slices
from pipeline.threeml import DOCS_URL, run_3ml as run_threeml_pipeline, threeml_available
from pipeline.util import get_bayspec_path
from ui.theme import inject_theme

FIT_PNGS = (
    ('ctsspec.png', 'Count spectrum'),
    ('phtspec.png', 'Photon spectrum'),
    ('model.png', 'Model (νFν)'),
)

PARAM_DISPLAY = (
    ('alpha', r'\alpha', ''),
    ('beta', r'\beta', ''),
    ('Ep', r'E_{\mathrm{p}}', 'keV'),
    ('Ep_best', r'E_{\mathrm{p}}', 'keV'),
    ('vFv', r'\nu F_{\nu}', ''),
    ('vFv_best', r'\nu F_{\nu}', ''),
    ('A', r'A', ''),
    ('kT', r'kT', 'keV'),
)

SKIP_RAW_COLS = {
    'alpha_low', 'alpha_high', 'beta_low', 'beta_high',
    'Ep_low', 'Ep_high', 'Ep_best',
    'vFv_low', 'vFv_high', 'vFv_best',
    'sigma_Ep', 'sigma_vFv',
    'log_Ep_1sigma_min', 'log_Ep_1sigma_max',
    'log_Ep_prior_lo', 'log_Ep_prior_hi',
    'log_Ep_hits_prior_low', 'log_Ep_hits_prior_high',
    'A_low', 'A_high', 'kT_low', 'kT_high',
}

TINT_STAGES = list(DEFAULT_STAGES)
PREFERRED_GRBS = ('GRB150514A', 'GRB140606B', 'GRB190829A')


def _pm_text(v, lo, hi, digits=3):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return '—'
    try:
        lo, hi = float(lo), float(hi)
    except (TypeError, ValueError):
        return f'{v:.{digits}g}'
    if not (np.isfinite(v) and np.isfinite(lo) and np.isfinite(hi)):
        return '—'
    return f'{v:.{digits}g} −{lo:.{digits}g}/+{hi:.{digits}g}'


def _fit_row_to_comparison(label, row):
    if row is None:
        return {'source': label, 'α': '—', 'Ep (keV)': '—', 'ep_constrained': '—'}
    if hasattr(row, 'index'):
        get = lambda k, default=None: row[k] if k in row.index else default
    else:
        get = row.get
    constrained = get('ep_constrained')
    return {
        'source': label,
        'α': _pm_text(get('alpha'), get('alpha_low'), get('alpha_high'), 3),
        'Ep (keV)': _pm_text(
            get('Ep_best', get('Ep')), get('Ep_low'), get('Ep_high'), 4),
        'ep_constrained': (
            '—' if constrained is None else ('yes' if int(constrained) else 'no')),
    }


def _read_hdf_row(path, key):
    if not path or not os.path.isfile(path) or not key:
        return None
    try:
        import pandas as pd
        return pd.read_hdf(path, key=key).iloc[0]
    except Exception:
        return None


def _hdf_keys(path):
    if not path or not os.path.isfile(path):
        return []
    try:
        import h5py
        with h5py.File(path, 'r') as f:
            return list(f.keys())
    except Exception:
        return []


def _canonical_tint_key(keys, model_name, grb, backend='bayspec'):
    tag = str(model_name).lower()
    if backend == '3ML':
        prefer = [f'tint_{tag}_3ML_{grb}', f'tint_{tag}_3ML']
        pool = [k for k in keys if '_3ML' in k and k.startswith('tint_')]
    else:
        prefer = [f'tint_{tag}_{grb}']
        pool = [
            k for k in keys
            if k.startswith(f'tint_{tag}') and '_3ML' not in k
        ]
    for k in prefer:
        if k in keys:
            return k
    return sorted(pool)[0] if pool else None


def _show_comparison_table(grb, data_base, model_name):
    import pandas as pd

    paths = PathLayout(grb, data_base=data_base)
    bay_path = get_bayspec_path(grb, data_base=data_base)
    ml_path = getattr(
        paths, 'threeml_data',
        os.path.join(paths.tresolved_path, '3ML', f'{grb}_3ML_data.h5'))

    bay_keys = _hdf_keys(bay_path)
    ml_keys = _hdf_keys(ml_path)
    key_bay = _canonical_tint_key(bay_keys, model_name, grb, 'bayspec')
    key_3ml = _canonical_tint_key(ml_keys, model_name, grb, '3ML')
    bay_row = _read_hdf_row(bay_path, key_bay)
    ml_row = _read_hdf_row(ml_path, key_3ml)

    rows = []
    ref = gcn_ref(grb)
    if ref:
        rows.append({
            'source': ref['source'],
            'α': _pm_text(ref['alpha'], ref['alpha_err'], ref['alpha_err'], 3),
            'Ep (keV)': _pm_text(ref['Ep'], ref['Ep_err'], ref['Ep_err'], 4),
            'ep_constrained': '—',
        })
    rows.append(_fit_row_to_comparison(f'bayspec {model_name}', bay_row))
    rows.append(_fit_row_to_comparison(f'3ML {model_name}', ml_row))

    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(
        f'bayspec `{bay_path}` · key=`{key_bay or "—"}`  ·  '
        f'3ML `{ml_path}` · key=`{key_3ml or "—"}`'
    )
    if ref and ref.get('note'):
        st.caption(ref['note'])


def _model_choices():
    try:
        from pipeline.fitting import available_models
        return available_models()
    except Exception:
        return sorted(MODEL_COLS)


def _grb_choices(data_base):
    names = []
    try:
        from grb_config import grbs_df_all
        names.extend(list(grbs_df_all['name']))
    except Exception:
        pass
    if os.path.isdir(data_base):
        for d in sorted(os.listdir(data_base)):
            if d.startswith('GRB') and os.path.isdir(os.path.join(data_base, d)):
                if d not in names:
                    names.append(d)
    return names or ['GRB140606B']


def _default_grb_index(grbs):
    for name in PREFERRED_GRBS:
        if name in grbs:
            return grbs.index(name)
    return 0


def _catalog_row(grb):
    try:
        from grb_config import grbs_df_all
        hits = grbs_df_all[grbs_df_all['name'] == grb]
        if not hits.empty:
            return hits.iloc[0]
    except Exception:
        pass
    return None


def _fmt_num(x):
    x = float(x)
    ax = abs(x)
    if ax == 0:
        return '0'
    if 0.01 <= ax < 10000:
        return f'{x:.4g}'
    exp = int(np.floor(np.log10(ax)))
    mant = x / (10 ** exp)
    return '{' + rf'{mant:.3g}\times 10^{{{exp}}}' + '}'


def _fmt_pm(value, low, high, symbol, unit=''):
    try:
        v, lo, hi = float(value), float(low), float(high)
    except (TypeError, ValueError):
        return None
    unit_tex = f'\\,\\mathrm{{{unit}}}' if unit else ''
    return (
        rf'{symbol} = {_fmt_num(v)}'
        rf'_{{-{_fmt_num(lo)}}}^{{+{_fmt_num(hi)}}}{unit_tex}'
    )


def _param_latex_lines(row):
    lines = []
    used = set()
    for base, symbol, unit in PARAM_DISPLAY:
        if base in used:
            continue
        if base == 'Ep' and 'Ep_best' in row.index:
            continue
        if base == 'vFv' and 'vFv_best' in row.index:
            continue
        if base not in row.index:
            continue
        low_key = 'Ep_low' if base == 'Ep_best' else (
            'vFv_low' if base == 'vFv_best' else f'{base}_low')
        high_key = 'Ep_high' if base == 'Ep_best' else (
            'vFv_high' if base == 'vFv_best' else f'{base}_high')
        if low_key in row.index and high_key in row.index:
            tex = _fmt_pm(row[base], row[low_key], row[high_key], symbol, unit)
            if tex:
                lines.append(tex)
                used.add(base)
                continue
        try:
            v = float(row[base])
        except (TypeError, ValueError):
            continue
        unit_tex = f'\\,\\mathrm{{{unit}}}' if unit else ''
        lines.append(rf'{symbol} = {_fmt_num(v)}{unit_tex}')
        used.add(base)
    if 'ep_constrained' in row.index:
        flag = bool(row['ep_constrained'])
        status = r'\mathrm{constrained}' if flag else r'\mathrm{unconstrained}'
        lines.append(rf'E_{{\mathrm{{p}}}}:\ {status}')
    return lines


def _resolve_fit_dir(grb, data_base, model_name, nlive, include_bgo):
    try:
        from pipeline.context import GRBContext
        from pipeline.fitting import get_fit_savepath
        ctx = GRBContext.from_name(grb, data_base=data_base)
        path = get_fit_savepath(
            ctx, model_name, mode='tintegrated', nlive=nlive,
            include_bgo=include_bgo)
        if os.path.isdir(path) and os.path.isfile(
                os.path.join(path, '1-post_equal_weights.dat')):
            return path
    except Exception:
        pass
    model_root = os.path.join(
        data_base, grb, 'data/tintegrated/bayspec', model_name)
    return _fallback_fit_dir(model_root, include_bgo)


def _fallback_fit_dir(model_root, include_bgo):
    candidates = []
    if os.path.isdir(model_root):
        candidates.append(model_root)
    versions = os.path.join(model_root, 'versions')
    if os.path.isdir(versions):
        for fp in sorted(os.listdir(versions)):
            candidates.append(os.path.join(versions, fp))

    def _dets_in(fit_dir):
        import json
        path = os.path.join(fit_dir, 'data.json')
        if not os.path.isfile(path):
            return None
        try:
            data = json.load(open(path))
            return [d.get('Name') for d in data if d.get('Name')]
        except Exception:
            return None

    matched = []
    for d in candidates:
        if not os.path.isfile(os.path.join(d, '1-post_equal_weights.dat')):
            continue
        dets = _dets_in(d)
        if dets is None:
            continue
        has_bgo = any(str(x)[:1].lower() == 'b' for x in dets)
        if bool(include_bgo) == has_bgo:
            matched.append(d)
    if matched:
        return matched[-1]
    for d in reversed(candidates):
        if os.path.isfile(os.path.join(d, '1-post_equal_weights.dat')):
            return d
    return model_root


def _fig_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=160, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.getvalue()


def _zip_files(paths):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for path in paths:
            if os.path.isfile(path):
                zf.write(path, arcname=os.path.basename(path))
    buf.seek(0)
    return buf.getvalue()


def _show_lightcurve(grb, data_base, heapy_dir, dets=None, t1=None, t2=None):
    from plot_lightcurves import plot_lightcurves
    if not os.path.isdir(heapy_dir):
        st.info(f'No heapy directory yet: `{heapy_dir}`')
        return
    fig, _axes = plot_lightcurves(
        grb, data_base=data_base, lc_kind='fixed', plot_style='step',
        heapy_dir=heapy_dir, dets=dets, overlay_bb=True,
        show_errors='none', t1=t1, t2=t2, show=False)
    if fig is None:
        st.info('No lightcurve traces for the selected detectors.')
        return
    st.pyplot(fig, clear_figure=True)
    st.download_button(
        'Download lightcurve PNG',
        data=_fig_png_bytes(fig),
        file_name=f'{grb}_tint_lightcurve.png',
        mime='image/png',
        key='dl_tint_lc',
        use_container_width=False,
    )


def _show_fit_plots(fit_dir, key_prefix='fit'):
    if not os.path.isdir(fit_dir):
        st.info(f'No fit directory: `{fit_dir}`')
        return
    shown = []
    labels = []
    for name, label in FIT_PNGS:
        path = os.path.join(fit_dir, name)
        if os.path.isfile(path):
            shown.append(path)
            labels.append(label)
    if not shown:
        st.info(f'No PNG plots in `{fit_dir}` — run the fit first.')
        return

    cols = st.columns(len(shown))
    for col, path, label in zip(cols, shown, labels):
        col.image(path, caption=label, use_container_width=True)

    htmls = sorted(glob.glob(os.path.join(fit_dir, '*infer.html')))
    with st.expander('Downloads', expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                'All fit plots (ZIP)',
                data=_zip_files(shown),
                file_name=f'{key_prefix}_plots.zip',
                mime='application/zip',
                key=f'dl_{key_prefix}_zip',
                use_container_width=True,
            )
        with c2:
            for path in shown:
                name = os.path.basename(path)
                with open(path, 'rb') as fh:
                    st.download_button(
                        name, data=fh.read(), file_name=name, mime='image/png',
                        key=f'dl_{key_prefix}_{name}',
                        use_container_width=True,
                    )
        for html in htmls:
            with open(html, 'rb') as fh:
                st.download_button(
                    os.path.basename(html),
                    data=fh.read(),
                    file_name=os.path.basename(html),
                    mime='text/html',
                    key=f'dl_{key_prefix}_{os.path.basename(html)}',
                    use_container_width=True,
                )


def _show_3ml_plots(fit_dir, key_prefix='3ml'):
    if not os.path.isdir(fit_dir):
        st.info(f'No 3ML directory yet: `{fit_dir}`')
        return
    names = ('lc.png', 'ctsspec.png', 'corner.png', 'model.png', 'phtspec.png')
    pngs = [(n, os.path.join(fit_dir, n)) for n in names if os.path.isfile(os.path.join(fit_dir, n))]
    if not pngs:
        pngs = [(os.path.basename(p), p) for p in sorted(glob.glob(os.path.join(fit_dir, '*.png')))]
    if not pngs:
        st.info(f'No 3ML plots in `{fit_dir}` — run 3ML first.')
        return
    n = min(3, len(pngs))
    for row_start in range(0, len(pngs), n):
        chunk = pngs[row_start:row_start + n]
        cols = st.columns(len(chunk))
        for col, (name, path) in zip(cols, chunk):
            col.image(path, caption=name, use_container_width=True)
    with st.expander('Downloads', expanded=False):
        st.download_button(
            'All 3ML plots (ZIP)',
            data=_zip_files([p for _, p in pngs]),
            file_name=f'{key_prefix}_plots.zip',
            mime='application/zip',
            key=f'dl_{key_prefix}_zip',
            use_container_width=True,
        )


def _show_parameters(grb, data_base, model_name):
    path = get_bayspec_path(grb, data_base=data_base)
    if not os.path.isfile(path):
        st.info(f'No HDF5 yet: `{path}`')
        return
    import pandas as pd

    keys = [
        k for k in _hdf_keys(path)
        if k.startswith(f'tint_{model_name}') and '_3ML' not in k
    ]
    if not keys:
        st.info(f'No time-integrated HDF5 keys for model `{model_name}`.')
        return

    key = st.selectbox('HDF5 key', keys, index=0, key='hdf5_tint')
    df = pd.read_hdf(path, key=key)
    row = df.iloc[0]
    lines = _param_latex_lines(row)
    if lines:
        cols = st.columns(min(3, len(lines)))
        for col, line in zip(cols, lines):
            col.latex(line)
        if len(lines) > 3:
            for line in lines[3:]:
                st.latex(line)
    if 'ep_constrained' in row.index:
        if bool(row['ep_constrained']):
            st.success(r'$E_{\mathrm{p}}$ constrained')
        else:
            st.warning(r'$E_{\mathrm{p}}$ unconstrained (1σ hits prior edge)')
    with st.expander('Raw HDF5 row'):
        show_cols = [c for c in df.columns if c not in SKIP_RAW_COLS]
        st.dataframe(df[show_cols] if show_cols else df, use_container_width=True)
    st.caption(f'`{path}` · key=`{key}`')


def _show_3ml_parameters(grb, data_base, model_name):
    paths = PathLayout(grb, data_base=data_base)
    path = getattr(
        paths, 'threeml_data',
        os.path.join(paths.tresolved_path, '3ML', f'{grb}_3ML_data.h5'))
    if not os.path.isfile(path):
        st.info(f'No 3ML HDF5 yet: `{path}`')
        return
    import pandas as pd

    tag = str(model_name).lower()
    keys = [k for k in _hdf_keys(path) if k.startswith(f'tint_{tag}_3ML')]
    if not keys:
        keys = [k for k in _hdf_keys(path) if k.startswith('tint_')]
    if not keys:
        st.info(f'No 3ML tint keys for `{model_name}`.')
        return
    key = st.selectbox('3ML HDF5 key', keys, index=0, key='hdf5_3ml_tint')
    df = pd.read_hdf(path, key=key)
    row = df.iloc[0]
    lines = _param_latex_lines(row)
    if lines:
        cols = st.columns(min(3, len(lines)))
        for col, line in zip(cols, lines):
            col.latex(line)
        if len(lines) > 3:
            for line in lines[3:]:
                st.latex(line)
    with st.expander('Raw 3ML HDF5 row'):
        st.dataframe(df, use_container_width=True)
    st.caption(f'`{path}` · key=`{key}`')


def _build_configured_ctx(grb, data_base, cfg):
    from pipeline.context import GRBContext
    ctx = GRBContext.from_name(grb, data_base=data_base)
    ctx.det_mode = cfg['det_mode']
    if cfg['det_mode'] == 'manual':
        ctx.sel_dets = list(cfg['sel_dets'])
    ctx.t1 = float(cfg['t1'])
    ctx.t2 = float(cfg['t2'])
    ctx.slice_mode = 'manual'
    ctx.slice_boundaries = list(cfg['slice_boundaries'])
    ctx.time_slices = boundaries_to_slices(ctx.slice_boundaries)
    ctx.lc_pad_pre = float(cfg['lc_pad_pre'])
    ctx.lc_pad_post = float(cfg['lc_pad_post'])
    ctx.lc_binsize = float(cfg['lc_binsize'])
    ctx.bs_p0 = float(cfg['bs_p0'])
    ctx.spec_filter_pad = float(cfg['spec_filter_pad'])
    ctx.lc_rebn = {
        'min_sigma': int(cfg['lc_min_sigma']),
        'max_bin': int(cfg['lc_max_bin']),
    }
    ctx.spec_rebn = {
        'min_sigma': int(cfg['spec_min_sigma']),
        'max_bin': int(cfg['spec_max_bin']),
    }
    ctx.nai_energy = list(cfg['nai_energy'])
    ctx.bgo_energy = list(cfg['bgo_energy'])
    if cfg.get('lc_combined_band'):
        ctx.lc_combined_band = list(cfg['lc_combined_band'])
    return ctx


def _run_bayspec(grb, data_base, cfg, model, nlive, include_bgo, force, workers):
    with st.status(f'bayspec tint: {grb} / {model} …', expanded=True) as status:
        try:
            from pipeline.runner import GRBPipelineRunner
            ctx = _build_configured_ctx(grb, data_base, cfg)
            runner = GRBPipelineRunner(
                n_workers=workers, model_name=model, nlive=nlive,
                force=force, include_bgo=include_bgo)
            ctx = runner.run(ctx, stages=TINT_STAGES)
            status.update(
                label=f'Done: {ctx.name}  dets={ctx.sel_dets}',
                state='complete')
            return ctx
        except Exception as exc:
            status.update(label='Pipeline failed', state='error')
            st.exception(exc)
            return None


def _run_3ml(grb, data_base, cfg, model, nlive, include_bgo, force, background_interval):
    with st.status(f'3ML tint: {grb} / {model} …', expanded=True) as status:
        try:
            ctx = _build_configured_ctx(grb, data_base, cfg)
            result = run_threeml_pipeline(
                ctx, model_name=model, mode='tintegrated',
                nlive=nlive, include_bgo=include_bgo, force=force,
                background_interval=background_interval or None)
            status.update(
                label=f'3ML done: {ctx.name}  dets={result.get("dets")}',
                state='complete')
            return result
        except Exception as exc:
            status.update(label='3ML pipeline failed', state='error')
            st.exception(exc)
            return None


def _threeml_tint_dir(paths, model_tag):
    tint_root = getattr(
        paths, 'threeml_tintegrated_path',
        os.path.join(paths.tintegrated_path, '3ML'))
    return os.path.join(tint_root, str(model_tag).lower())


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def _active_theme_name() -> str:
    """Follow Streamlit's built-in Light/Dark setting (☰ → Settings)."""
    try:
        t = st.context.theme
        name = t.get('type') if hasattr(t, 'get') else getattr(t, 'type', None)
        if name in ('light', 'dark'):
            return name
    except Exception:
        pass
    return 'light'


st.set_page_config(
    page_title='Fermi GBM · Time integrated',
    layout='wide',
    initial_sidebar_state='expanded',
)

if 'ui_compact' not in st.session_state:
    st.session_state.ui_compact = False

default_data = os.environ.get('DATA_BASE', DATA_BASE)
if 'data_base' not in st.session_state:
    st.session_state.data_base = default_data

ui_theme = _active_theme_name()
st.markdown(
    f'<style>{inject_theme(ui_theme, st.session_state.ui_compact)}</style>',
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="gbm-hero">Time-integrated spectral fit</p>'
    '<p class="gbm-sub">Fermi GBM · heapy + bayspec'
    ' · optional 3ML cross-check · GCN comparison</p>',
    unsafe_allow_html=True,
)

# ----- Sidebar -----
with st.sidebar:
    st.markdown('### Fermi GBM')
    st.caption('Time-integrated only')

    with st.expander('Customize', expanded=False):
        st.caption(
            'App theme: use **☰ → Settings → Theme** '
            f'(Light / Dark / System). Active: **{ui_theme}**.')
        compact = st.toggle('Compact spacing', value=st.session_state.ui_compact)
        if compact != st.session_state.ui_compact:
            st.session_state.ui_compact = compact
            st.rerun()

    st.markdown('---')
    data_base = st.text_input('DATA_BASE', key='data_base').strip() or default_data
    os.environ['DATA_BASE'] = data_base
    if not os.path.isdir(data_base):
        st.error(f'Missing path: `{data_base}`')

    grbs = _grb_choices(data_base)
    grb = st.selectbox('GRB', grbs, index=_default_grb_index(grbs))
    row = _catalog_row(grb)
    catalog_dets = list(row['sel_dets']) if row is not None else []
    default_t1 = float(row['t1']) if row is not None else -0.83
    default_t2 = float(row['t2']) if row is not None else 5.74
    default_bounds = list(row['slice_boundaries']) if row is not None else [
        -0.83, 1.345, 1.795, 2.34, 3.34, 4.43, 5.74]

    ref = gcn_ref(grb)
    if ref:
        st.success(f"{ref['source']}: α={ref['alpha']}±{ref['alpha_err']}, "
                   f"Ep={ref['Ep']}±{ref['Ep_err']} keV")
        st.caption(f"Tint window [{ref['t1']}, {ref['t2']}] s")

    st.markdown('##### Detectors')
    det_mode = st.selectbox(
        'Selection', ['manual', 'auto'], index=0,
        help='manual = pick detectors · auto = top NaI + BGO by angle')
    if det_mode == 'manual':
        sel_dets = st.multiselect(
            'Detectors', ALL_DETS,
            default=catalog_dets or ['n3', 'n4', 'n8'])
    else:
        sel_dets = catalog_dets
        st.caption('Auto resolves after download / geometry')
    include_bgo = st.checkbox('Include BGO in fit', value=False)

    st.markdown('##### Fit')
    models = _model_choices()
    model = st.selectbox(
        'Model', models,
        index=models.index('cpl') if 'cpl' in models else 0)
    nlive = st.selectbox('nlive', [200, 1000], index=1)
    force = st.checkbox('Force rerun', value=False)
    workers = st.slider('Workers', 1, 8, 1)

    st.markdown('##### Time window')
    c1, c2 = st.columns(2)
    with c1:
        t1 = st.number_input('t1 (s)', value=float(default_t1), format='%.3f')
    with c2:
        t2 = st.number_input('t2 (s)', value=float(default_t2), format='%.3f')

    nai_emin, nai_emax = float(NAI_ENERGY[0]), float(NAI_ENERGY[1])
    bgo_emin, bgo_emax = float(BGO_ENERGY[0]), float(BGO_ENERGY[1])
    lc_pad_pre, lc_pad_post = 20.0, 50.0
    spec_filter_pad, lc_binsize, bs_p0 = 380.0, 0.5, 0.05
    lc_min_sigma, lc_max_bin = 1, 8
    spec_min_sigma, spec_max_bin = 2, 20
    comb = '10, 1000'
    with st.expander('Advanced', expanded=False):
        nai_emin = st.number_input('NaI min (keV)', value=nai_emin)
        nai_emax = st.number_input('NaI max (keV)', value=nai_emax)
        bgo_emin = st.number_input('BGO min (keV)', value=bgo_emin)
        bgo_emax = st.number_input('BGO max (keV)', value=bgo_emax)
        lc_pad_pre = st.number_input('LC pad before (s)', value=lc_pad_pre, min_value=0.0)
        lc_pad_post = st.number_input('LC pad after (s)', value=lc_pad_post, min_value=0.0)
        spec_filter_pad = st.number_input(
            'Spectrum filter pad (s)', value=spec_filter_pad, min_value=10.0)
        lc_binsize = st.number_input('LC bin size (s)', value=lc_binsize, min_value=0.01)
        bs_p0 = st.number_input(
            'BB p0', value=bs_p0, min_value=1e-4, max_value=0.5, format='%.4f')
        lc_min_sigma = st.number_input('LC rebin σ', value=lc_min_sigma, min_value=1)
        lc_max_bin = st.number_input('LC max bin', value=lc_max_bin, min_value=1)
        spec_min_sigma = st.number_input('Spec rebin σ', value=spec_min_sigma, min_value=1)
        spec_max_bin = st.number_input('Spec max bin', value=spec_max_bin, min_value=1)
        comb = st.text_input('Combined LC band (keV)', value=comb)

    st.markdown('##### 3ML')
    ok_3ml, ver_3ml = threeml_available()
    if ok_3ml:
        st.caption(f'threeML {ver_3ml} available')
    else:
        st.caption('threeML not installed in this image')
    run_with_3ml = st.checkbox(
        'Also run 3ML after bayspec', value=False, disabled=not ok_3ml)
    background_interval = ''
    if run_with_3ml:
        default_bkg = (
            f'{-float(lc_pad_pre):.1f}-{float(t1):.2f}, '
            f'{float(t2):.2f}-{float(t2) + float(lc_pad_post):.1f}')
        background_interval = st.text_input(
            'Background intervals',
            value='',
            placeholder=default_bkg,
            help=f'Empty = LC pads minus burst: {default_bkg}')
        st.markdown(f'[3ML GRB tutorial]({DOCS_URL})')

    run_tint = st.button('Run time integrated', type='primary')
    refresh = st.button('Refresh results')

cfg = {
    'det_mode': det_mode,
    'sel_dets': sel_dets,
    't1': t1,
    't2': t2,
    'slice_boundaries': default_bounds if len(default_bounds) >= 2 else [t1, t2],
    'lc_pad_pre': lc_pad_pre,
    'lc_pad_post': lc_pad_post,
    'spec_filter_pad': spec_filter_pad,
    'lc_binsize': lc_binsize,
    'bs_p0': bs_p0,
    'lc_min_sigma': lc_min_sigma,
    'lc_max_bin': lc_max_bin,
    'spec_min_sigma': spec_min_sigma,
    'spec_max_bin': spec_max_bin,
    'nai_energy': [nai_emin, nai_emax],
    'bgo_energy': [bgo_emin, bgo_emax],
    'lc_combined_band': [10.0, 1000.0],
}
try:
    parts = [p.strip() for p in comb.replace(';', ',').split(',') if p.strip()]
    if len(parts) == 2:
        cfg['lc_combined_band'] = [float(parts[0]), float(parts[1])]
except Exception:
    st.sidebar.error('Combined LC band must be two numbers, e.g. 10, 1000')

plot_base = list(sel_dets) if det_mode == 'manual' else list(catalog_dets)
plot_dets = fit_detectors(plot_base, include_bgo=include_bgo) or plot_base

# ----- Actions -----
threeml_result = None
if run_tint:
    ctx = _run_bayspec(
        grb, data_base, cfg, model, nlive, include_bgo, force, workers)
    if ctx is not None and run_with_3ml:
        threeml_result = _run_3ml(
            grb, data_base, cfg, model, nlive, include_bgo, force,
            background_interval)

paths = PathLayout(grb, data_base=data_base)
tint_heapy = paths.heapy_tintegrated_path
tint_fit = _resolve_fit_dir(grb, data_base, model, nlive, include_bgo)
tint_3ml = _threeml_tint_dir(paths, model)
if threeml_result and threeml_result.get('root'):
    tint_3ml = threeml_result['root']

# ----- Results -----
st.markdown('---')
st.markdown(f'### Results · `{grb}`')
st.caption(
    f'tint [{t1:g}, {t2:g}] s · model `{model}` · '
    f'dets {", ".join(plot_dets) or "—"} · nlive={nlive}'
)

tab_lc, tab_fit, tab_params, tab_cmp, tab_3ml = st.tabs([
    'Lightcurve',
    'Bayspec fit',
    'Parameters',
    'GCN compare',
    '3ML',
])

with tab_lc:
    _show_lightcurve(
        grb, data_base, tint_heapy, dets=plot_dets, t1=t1, t2=t2)

with tab_fit:
    st.caption(tint_fit)
    _show_fit_plots(tint_fit, key_prefix='tint_fit')

with tab_params:
    _show_parameters(grb, data_base, model)

with tab_cmp:
    _show_comparison_table(grb, data_base, model)

with tab_3ml:
    st.caption(tint_3ml)
    _show_3ml_plots(tint_3ml, key_prefix='3ml_tint')
    _show_3ml_parameters(grb, data_base, model)

if refresh:
    st.rerun()
