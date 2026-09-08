"""Heapy lightcurve extraction (fixed bins, SNR rebin, pgSignal BB).

Background note
---------------
heapy ``pgSignal`` fits a polynomial background on the LC with optional
``bs_ignore``. This is **not** the one_fits_all "baseline algorithm"
(Zhang et al. 2016 / MySpecFit). We keep heapy + bayspec as the NJU stack.
"""

import os
import shutil

from .detectors import detector_energy_range
from .lc_io import (
    bb_payload_from_arrays,
    combine_fixed_bin_traces,
    persist_lc_files,
    rebin_payload_from_tte_arrays,
    traces_from_lc_json,
    write_lc_json,
)


def save_heapy_bayesian_blocks_lc(gbm_tte, savepath, time_offset=None):
    """Persist Bayesian-block LC from heapy ``pgSignal`` (after ``extract_curve``)."""
    bs = gbm_tte.lc_bs
    payload = bb_payload_from_arrays(
        bs.edges, bs.re_binsize, bs.re_cts, bs.re_bcts,
        p0=gbm_tte.bs_p0, time_offset=time_offset)
    write_lc_json(os.path.join(savepath, 'bb_lc.json'), payload)


def save_heapy_rebin_lc(gbm_tte, savepath, time_offset=None):
    """Persist SNR-rebinned source + background rates (after ``extract_rebin_curve``)."""
    payload = rebin_payload_from_tte_arrays(
        gbm_tte.lc_rebin_list, gbm_tte.lc_retime, gbm_tte.lc_reexps,
        gbm_tte.lc_src_rects, gbm_tte.lc_bkg_rebcts,
        gbm_tte.lc_src_rects_err, gbm_tte.lc_bkg_rebcts_err,
        time_offset=time_offset)
    write_lc_json(os.path.join(savepath, 'rebin_lc.json'), payload)


def extract_heapy_lightcurve(gbm_tte, temp_dir, lc_binsize=0.5, rebin=True,
                             rebin_min_sigma=1, rebin_max_bin=8, bs_p0=0.05,
                             time_offset=None):
    """Uniform-bin LC via extract_curve; heapy pgSignal BB saved as bb_lc.json.

    BB is a segmentation product, not the default display LC (use rebin for that).
    """
    gbm_tte.lc_binsize = lc_binsize
    gbm_tte.bs_p0 = bs_p0
    gbm_tte.extract_curve(savepath=temp_dir, show=False)
    save_fixed_bin_lc(gbm_tte, temp_dir, time_offset=time_offset)
    save_heapy_bayesian_blocks_lc(gbm_tte, temp_dir, time_offset=time_offset)
    if rebin:
        gbm_tte.extract_rebin_curve(
            min_sigma=rebin_min_sigma, max_bin=rebin_max_bin, step=True,
            savepath=temp_dir, show=False)
        save_heapy_rebin_lc(gbm_tte, temp_dir, time_offset=time_offset)
    gbm_tte.calculate_txx(xx=0.9, savepath=temp_dir)


def _open_tte(ctx, det):
    from heapy.pipe.event import gbmTTE

    gbm_tte = gbmTTE(ctx.gbm_rtv.rtv_res['tte'][det], ctx.gbm_rtv.rtv_res['poshist'])
    time_offset = ctx.fermi_met - gbm_tte.timezero
    return gbm_tte, time_offset


def _event_time_span(gbm_tte, inset=0.05):
    """Inclusive TIME min/max on current TTE events, inset from the edges."""
    import numpy as np

    ev = gbm_tte.event
    t = np.asarray(ev['TIME'], dtype=float)
    t = t[np.isfinite(t)]
    if t.size == 0:
        return None, None
    lo, hi = float(t.min()), float(t.max())
    pad = min(float(inset), 0.25 * max(hi - lo, 0.0))
    return lo + pad, hi - pad


def _clip_window(win, t_lo, t_hi):
    lo, hi = float(win[0]), float(win[1])
    if t_lo is not None:
        lo = max(lo, float(t_lo))
    if t_hi is not None:
        hi = min(hi, float(t_hi))
    if hi <= lo:
        hi = lo + 1.0
    return [lo, hi]


def prepare_tte_for_lc(ctx, det, energy_band=None):
    """Load TTE, apply LC time window + energy filter + burst-span ``bs_ignore``."""
    gbm_tte, time_offset = _open_tte(ctx, det)
    gbm_tte.event
    gbm_tte.filter_time(ctx.lc_window(time_offset))
    gbm_tte.filter_energy(detector_energy_range(det, energy_band, ctx=ctx))
    ignore = ctx.bs_ignore_interval(time_offset)
    if ignore is not None:
        gbm_tte.bs_ignore = [ignore]
        gbm_tte.bs_deg = 1
    return gbm_tte, time_offset


def save_fixed_bin_lc(gbm_tte, savepath, time_offset=None):
    """Persist uniform-bin source + background rates (after ``extract_curve``)."""
    import numpy as np
    t = np.asarray(gbm_tte.lc_time, dtype=float)
    src = np.asarray(gbm_tte.lc_src_rate, dtype=float)
    bkg = np.asarray(gbm_tte.lc_bkg_rate, dtype=float)
    src_err = np.asarray(gbm_tte.lc_src_rate_err, dtype=float)
    bkg_err = np.asarray(gbm_tte.lc_bkg_rate_err, dtype=float)
    bins = np.asarray(gbm_tte.lc_bin_list, dtype=float)
    edges = np.unique(bins.flatten())
    payload = {
        'method': 'heapy.extract_curve',
        'binsize': float(gbm_tte.lc_binsize),
        'edges': edges.tolist(),
        'traces': [
            {'name': 'source lightcurve', 'x': t.tolist(), 'y': src.tolist(),
             'error_y': src_err.tolist()},
            {'name': 'background lightcurve', 'x': t.tolist(), 'y': bkg.tolist(),
             'error_y': bkg_err.tolist()},
        ],
    }
    if time_offset is not None:
        payload['time_offset'] = float(time_offset)
        payload['edges_rel'] = (edges - float(time_offset)).tolist()
    write_lc_json(os.path.join(savepath, 'lc_fixed.json'), payload)


def extract_detector_lightcurve(ctx, det, dest_dir, energy_band=None,
                                products=('fixed', 'rebin', 'bb')):
    """Extract LC products for one detector into dest_dir."""
    gbm_tte, time_offset = prepare_tte_for_lc(ctx, det, energy_band=energy_band)
    temp_dir = dest_dir + '_temp'
    os.makedirs(temp_dir, exist_ok=True)
    lc_rebn = ctx.lc_rebn or {}
    gbm_tte.lc_binsize = ctx.lc_binsize
    gbm_tte.bs_p0 = ctx.bs_p0
    gbm_tte.extract_curve(savepath=temp_dir, show=False)
    save_fixed_bin_lc(gbm_tte, temp_dir, time_offset=time_offset)
    if 'bb' in products:
        save_heapy_bayesian_blocks_lc(gbm_tte, temp_dir, time_offset=time_offset)
    if ctx.rebin and 'rebin' in products:
        gbm_tte.extract_rebin_curve(
            min_sigma=lc_rebn.get('min_sigma', 1),
            max_bin=lc_rebn.get('max_bin', 8), step=True,
            savepath=temp_dir, show=False)
        save_heapy_rebin_lc(gbm_tte, temp_dir, time_offset=time_offset)
    if 'fixed' in products:
        gbm_tte.calculate_txx(xx=0.9, savepath=temp_dir)
    persist_lc_files(temp_dir, dest_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)
    return time_offset


def write_combined_lc(ctx, spec_dir, time_offset):
    """Optional 10–1000 keV combined LC (one_fits_all screening band).

    Re-extracts each selected detector in the clipped combined band, then sums
    source/background rates onto a common grid as ``lc_combined.json``.
    """
    from .lc_io import load_lc_json

    band = ctx.lc_combined_band
    if not band:
        return None
    traces_by_det = {}
    for det in ctx.sel_dets:
        dest = os.path.join(spec_dir, f'_combined_{det}')
        extract_detector_lightcurve(
            ctx, det, dest, energy_band=band, products=('fixed',))
        json_path = os.path.join(dest, 'lc_fixed.json')
        if not os.path.isfile(json_path):
            json_path = os.path.join(dest, 'rebin_lc.json')
        if not os.path.isfile(json_path):
            json_path = os.path.join(dest, 'bb_lc.json')
        if os.path.isfile(json_path):
            traces_by_det[det] = traces_from_lc_json(load_lc_json(json_path))
        shutil.rmtree(dest, ignore_errors=True)
    if not traces_by_det:
        print('  WARNING: could not build lc_combined.json')
        return None
    payload = combine_fixed_bin_traces(traces_by_det, time_offset=time_offset)
    payload['band_keV'] = list(band)
    out = os.path.join(spec_dir, 'lc_combined.json')
    write_lc_json(out, payload)
    print(f'  Combined LC ({band[0]}–{band[1]} keV) -> {out}')
    return out


def extract_combined_from_native(ctx, spec_dir, time_offset):
    """Sum already-extracted per-detector LCs into ``lc_combined.json``."""
    from .lc_io import load_lc_json

    if not ctx.lc_combined_band:
        return None
    traces_by_det = {}
    for det in ctx.sel_dets:
        for fname in ('rebin_lc.json', 'bb_lc.json'):
            path = os.path.join(spec_dir, det, fname)
            if os.path.isfile(path):
                traces_by_det[det] = traces_from_lc_json(load_lc_json(path))
                break
    if not traces_by_det:
        return None
    payload = combine_fixed_bin_traces(traces_by_det, time_offset=time_offset)
    payload['band_keV'] = list(ctx.lc_combined_band)
    payload['note'] = (
        'Sum of selected-detector LCs. Per-detector energy is native GBM band '
        'unless a dedicated combined-band extraction was run.')
    out = os.path.join(spec_dir, 'lc_combined.json')
    write_lc_json(out, payload)
    print(f'  Combined LC -> {out}')
    return out


def tint_lc_complete(spec_dir, dets):
    for det in dets:
        if not os.path.isfile(os.path.join(spec_dir, det, 'bb_lc.json')):
            return False
    return True
