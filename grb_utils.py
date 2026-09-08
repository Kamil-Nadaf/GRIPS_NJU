"""Thin shim: existing notebook ``from grb_utils import *`` keeps working.

Pipeline logic lives in ``pipeline/``. ``load_grb()`` still injects notebook
globals for backward compatibility; new code should use ``GRBContext`` /
``GRBPipelineRunner``.
"""

import inspect
import sys

from pipeline.constants import (
    ALL_DETS,
    DATA_BASE,
    DEFAULT_SPEC_REBN,
    MODEL_COLS,
    PARAM_KEY_MAP,
)
from pipeline.context import GRBContext, get_current, set_current
from pipeline.detectors import suggest_detectors as _suggest_detectors
from pipeline.download import load_grb_context
from pipeline.fitting import (
    MODELS,
    available_models,
    build_fit_data as _build_fit_data_impl,
    compute_vFv,
    ep_constraint_flags,
    extract_params as _extract_params,
    fit_all_time_slices as _fit_all_time_slices,
    fit_tintegrated as _fit_tintegrated,
    get_fit_savepath as _get_fit_savepath,
    get_model_params,
    make_model as _make_model,
    plot_infer_fit as _plot_infer_fit,
    reload_posterior as _reload_posterior,
    save_fit_plots as _save_fit_plots,
)
from pipeline.geometry import extract_geometry as _extract_geometry
from pipeline.lc_io import (
    bb_traces_from_pgsignal as _bb_traces_from_pgsignal,
    infer_bin_edges_from_centers as _infer_bin_edges_from_centers,
    load_heapy_bayesian_blocks_lc,
    load_heapy_rebin_lc,
    mirror_lc_dir as _mirror_lc_dir,
    persist_lc_files as _persist_lc_files,
    traces_from_lc_json as _traces_from_lc_json,
)
from pipeline.lightcurve import (
    extract_heapy_lightcurve as _extract_heapy_lightcurve,
    save_heapy_bayesian_blocks_lc,
    save_heapy_rebin_lc,
)
from pipeline.paths import (
    commit_meta as _commit_meta,
    extraction_fingerprint as _extraction_fingerprint_ctx,
    fingerprint as _fingerprint,
    fit_fingerprint as _fit_fingerprint_ctx,
    read_meta as _read_meta,
    resolve_versioned_dir as _resolve_versioned_dir,
    spec_slice_name as _spec_slice_name,
    write_meta as _write_meta,
)
from pipeline.runner import GRBPipelineRunner
from pipeline.spectra import (
    active_heapy_tint_dir as _active_heapy_tint_dir_ctx,
    active_heapy_tres_dir as _active_heapy_tres_dir_ctx,
    extract_pulse_lightcurve as _extract_pulse_lightcurve,
    extract_tintegrated_spectra as _extract_tintegrated_spectra,
    extract_tresolved_spectra as _extract_tresolved_spectra,
    tint_spec_base as _tint_spec_base_ctx,
)
from pipeline.util import (
    clean_dir,
    get_bayspec_path,
    list_grb_results as _list_grb_results,
    show_directory_tree,
    view_hdf5,
)
from pipeline.slices import boundaries_to_slices, resolve_time_slices


def _ctx():
    try:
        return get_current()
    except RuntimeError as exc:
        raise RuntimeError(
            'No GRB loaded. Call load_grb(grb) first, or use GRBPipelineRunner.') from exc


def load_grb(grb):
    ctx = load_grb_context(grb)
    to_push = ctx.as_globals()
    caller_globals = inspect.stack()[1].frame.f_globals
    caller_globals.update(to_push)
    sys.modules[__name__].__dict__.update(to_push)
    return ctx


def extraction_fingerprint():
    return _extraction_fingerprint_ctx(_ctx())


def fit_fingerprint(model_name, fixed_params=None, nlive=1000, fit_dets=None):
    return _fit_fingerprint_ctx(
        _ctx(), model_name, fixed_params, nlive, fit_dets=fit_dets)


def extract_geometry(dets=None):
    return _extract_geometry(_ctx(), dets=dets)


def suggest_detectors(ra=None, dec=None, gbm_rtv_obj=None, fermi_met_val=None,
                      n_nai=2, n_bgo=1, max_angle=60.0):
    ctx = _ctx()
    return _suggest_detectors(
        ctx, ra=ra, dec=dec, gbm_rtv_obj=gbm_rtv_obj,
        fermi_met_val=fermi_met_val, n_nai=n_nai, n_bgo=n_bgo,
        max_angle=max_angle)


def extract_tintegrated_spectra(lc_binsize=0.5, rebin=True,
                                rebin_min_sigma=1, rebin_max_bin=8, bs_p0=0.05,
                                force=False, n_workers=None):
    ctx = _ctx()
    ctx.lc_binsize = lc_binsize
    ctx.rebin = rebin
    ctx.lc_rebn = {'min_sigma': rebin_min_sigma, 'max_bin': rebin_max_bin}
    ctx.bs_p0 = bs_p0
    return _extract_tintegrated_spectra(ctx, force=force, n_workers=n_workers)


def extract_tresolved_spectra(lc_binsize=0.5, rebin=True,
                              rebin_min_sigma=1, rebin_max_bin=8, bs_p0=0.05,
                              force=False, n_workers=None):
    ctx = _ctx()
    ctx.lc_binsize = lc_binsize
    ctx.rebin = rebin
    ctx.lc_rebn = {'min_sigma': rebin_min_sigma, 'max_bin': rebin_max_bin}
    ctx.bs_p0 = bs_p0
    return _extract_tresolved_spectra(ctx, force=force, n_workers=n_workers)


def fit_tintegrated(model_name='cpl', fixed_params=None, nlive=1000, skip_dets=None,
                    force=False, include_bgo=False):
    return _fit_tintegrated(
        _ctx(), model_name=model_name, fixed_params=fixed_params,
        nlive=nlive, skip_dets=skip_dets, force=force, include_bgo=include_bgo)


def fit_all_time_slices(time_slices=None, model_name='cpl', fixed_params=None,
                        nlive=1000, force=False, n_workers=None, include_bgo=False):
    ctx = _ctx()
    if time_slices is not None:
        ctx.time_slices = [tuple(s) for s in time_slices]
    return _fit_all_time_slices(
        ctx, model_name=model_name, fixed_params=fixed_params,
        nlive=nlive, force=force, n_workers=n_workers, include_bgo=include_bgo)


def extract_params(model_name='cpl', mode='tintegrated', fixed_params=None, nlive=1000,
                   include_bgo=False):
    return _extract_params(
        _ctx(), model_name=model_name, mode=mode,
        fixed_params=fixed_params, nlive=nlive, include_bgo=include_bgo)


def extract_pulse_lightcurve(det=None, en_low=8, en_high=50, binsize=0.5,
                             pad_pre=10, pad_post=10):
    return _extract_pulse_lightcurve(
        _ctx(), det=det, en_low=en_low, en_high=en_high, binsize=binsize,
        pad_pre=pad_pre, pad_post=pad_post)


def get_fit_savepath(model_name, mode='tintegrated', slice_index=None,
                     fixed_params=None, nlive=1000, include_bgo=False):
    return _get_fit_savepath(
        _ctx(), model_name, mode=mode, slice_index=slice_index,
        fixed_params=fixed_params, nlive=nlive, include_bgo=include_bgo)


def _build_fit_data(mode='tintegrated', slice_index=None):
    return _build_fit_data_impl(_ctx(), mode=mode, slice_index=slice_index)


def reload_posterior(savepath, model_name, fixed_params=None, nlive=None):
    return _reload_posterior(
        _ctx(), savepath, model_name, fixed_params=fixed_params, nlive=nlive)


def plot_infer_fit(model_name='cpl', mode='tintegrated', slice_index=None,
                   style='CE', fixed_params=None, nlive=1000, savepath=None,
                   display=True, include_bgo=False):
    return _plot_infer_fit(
        _ctx(), model_name=model_name, mode=mode, slice_index=slice_index,
        style=style, fixed_params=fixed_params, nlive=nlive,
        savepath=savepath, display=display, include_bgo=include_bgo)


def list_grb_results(grb_name=None):
    name = grb_name or _ctx().name
    return _list_grb_results(name)


def _active_heapy_tint_dir():
    return _active_heapy_tint_dir_ctx(_ctx())


def _active_heapy_tres_dir():
    return _active_heapy_tres_dir_ctx(_ctx())


def _tint_spec_base(det):
    return _tint_spec_base_ctx(_ctx(), det)


    