"""Heapy spectrum extraction (time-integrated and time-resolved).

Background note
---------------
heapy ``pgSignal`` polynomial background (with ``bs_ignore`` masking the
burst) is **not** the one_fits_all baseline algorithm + ``gbm_drm_gen``.
We keep heapy spectra + bayspec PGSTAT as the current NJU stack.
"""

import os
import shutil

from .context import GRBContext
from .lc_io import persist_lc_files
from .lightcurve import (
    _clip_window,
    extract_combined_from_native,
    extract_heapy_lightcurve,
    prepare_tte_for_lc,
    write_combined_lc,
)
from .parallel import default_n_workers, map_parallel
from .paths import (
    commit_meta,
    extraction_fingerprint,
    resolve_versioned_dir,
    spec_slice_name,
)


def _flatten_spec_files(temp_dir, dest_dir, dest_stem):
    saved_name = None
    for f in os.listdir(temp_dir):
        if f.endswith('.src'):
            saved_name = f.replace('.src', '')
            break
    if not saved_name:
        return False
    os.makedirs(dest_dir, exist_ok=True)
    for ext in ('src', 'bkg', 'rsp'):
        src_file = os.path.join(temp_dir, f'{saved_name}.{ext}')
        dst_file = os.path.join(dest_dir, f'{dest_stem}.{ext}')
        if os.path.exists(src_file):
            shutil.move(src_file, dst_file)
    return True


def _apply_bs_ignore(gbm_tte, ctx, time_offset):
    ignore = ctx.bs_ignore_interval(time_offset)
    if ignore is not None:
        gbm_tte.bs_ignore = [ignore]
        gbm_tte.bs_deg = 1


def extract_spectrum_and_response(ctx, gbm_tte, time_offset, t_start, t_stop,
                                  temp_dir):
    """Extract .src/.bkg/.rsp on the LC window (not the 380 s spec pad).

    ``spec_filter_pad=380`` overruns TTE/DRM coverage (heapy then warns
    ``Extrapolation may be imprecise: -80 < -2.3`` and can kill a
    ProcessPool worker). Background already uses LC pads via ``bs_ignore``.
    """
    win = _clip_window(
        ctx.spec_window(time_offset), *ctx.lc_window(time_offset))
    print(f'    spec filter_time={win}')
    gbm_tte.filter_time(win)
    _apply_bs_ignore(gbm_tte, ctx, time_offset)
    gbm_tte.spec_slices = [[time_offset + t_start, time_offset + t_stop]]
    gbm_tte.extract_spectrum(savepath=temp_dir, show=False)
    gbm_tte.extract_response(ra=ctx.ra, dec=ctx.dec, savepath=temp_dir)


def _ctx_from_payload(payload):
    ctx = GRBContext.from_dict(payload['ctx'])
    from .download import retrieve
    retrieve(ctx)
    return ctx


def _extract_one_det_tint(payload):
    ctx = _ctx_from_payload(payload)
    det = payload['det']
    spec_dir = payload['spec_dir']
    spec_name = payload['spec_name']
    print(f'Processing {det}...')
    try:
        gbm_tte, time_offset = prepare_tte_for_lc(ctx, det)
        print(f'  {det} offset {time_offset:.2f}s  LC window={ctx.lc_window(time_offset)}')
        temp_dir = os.path.join(spec_dir, f'temp_{det}')
        os.makedirs(temp_dir, exist_ok=True)
        lc_rebn = ctx.lc_rebn or {}
        extract_heapy_lightcurve(
            gbm_tte, temp_dir, lc_binsize=ctx.lc_binsize, rebin=ctx.rebin,
            rebin_min_sigma=lc_rebn.get('min_sigma', 1),
            rebin_max_bin=lc_rebn.get('max_bin', 8),
            bs_p0=ctx.bs_p0, time_offset=time_offset)
        lc_dest = os.path.join(spec_dir, det)
        persist_lc_files(temp_dir, lc_dest)
        canonical = ctx.paths.heapy_tintegrated_path
        if spec_dir != canonical:
            from .lc_io import mirror_lc_dir
            mirror_lc_dir(lc_dest, os.path.join(canonical, det))

        extract_spectrum_and_response(
            ctx, gbm_tte, time_offset, ctx.t1, ctx.t2, temp_dir)
        if not _flatten_spec_files(temp_dir, spec_dir, f'{spec_name}_{det}'):
            print(f'  WARNING: no .src for {det}')
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f'  Saved {det}')
        return {'det': det, 'ok': True, 'time_offset': time_offset}
    except Exception as e:
        print(f'  Error processing {det}: {e}')
        return {'det': det, 'ok': False, 'error': str(e)}


def _extract_one_det_tres(payload):
    ctx = _ctx_from_payload(payload)
    det = payload['det']
    t_start, t_stop = payload['t_start'], payload['t_stop']
    slice_dir = payload['slice_dir']
    spec_name = payload['spec_name']
    print(f'  {det}...')
    try:
        gbm_tte, time_offset = prepare_tte_for_lc(ctx, det)
        temp_dir = os.path.join(slice_dir, f'temp_{det}')
        os.makedirs(temp_dir, exist_ok=True)
        lc_rebn = ctx.lc_rebn or {}
        extract_heapy_lightcurve(
            gbm_tte, temp_dir, lc_binsize=ctx.lc_binsize, rebin=ctx.rebin,
            rebin_min_sigma=lc_rebn.get('min_sigma', 1),
            rebin_max_bin=lc_rebn.get('max_bin', 8),
            bs_p0=ctx.bs_p0, time_offset=time_offset)
        lc_dest = os.path.join(slice_dir, det)
        persist_lc_files(temp_dir, lc_dest)
        extract_spectrum_and_response(
            ctx, gbm_tte, time_offset, t_start, t_stop, temp_dir)
        if not _flatten_spec_files(temp_dir, slice_dir, f'{spec_name}_{det}'):
            print(f'    WARNING: no .src for {det}')
        shutil.rmtree(temp_dir, ignore_errors=True)
        return {'det': det, 'ok': True}
    except Exception as e:
        print(f'    Error: {e}')
        return {'det': det, 'ok': False, 'error': str(e)}


def _tint_src_complete(spec_dir, dets, spec_name='tintegrated'):
    return all(os.path.isfile(os.path.join(spec_dir, f'{spec_name}_{det}.src'))
               for det in dets)


def extract_tintegrated_spectra(ctx, force=False, n_workers=None):
    """Extract time-integrated lightcurves + spectra for all detectors."""
    if ctx.t1 is None or ctx.t2 is None:
        raise ValueError(f'{ctx.name}: t1/t2 not set')

    spec_name = 'tintegrated'
    ext_fp = extraction_fingerprint(ctx)
    spec_dir = resolve_versioned_dir(
        ctx.paths.heapy_tintegrated_path, ext_fp, 'canonical_extraction')
    os.makedirs(spec_dir, exist_ok=True)

    if spec_dir == ctx.paths.heapy_tintegrated_path:
        print(f'\nProcessing time-integrated spectra for {ctx.name} (canonical, fp={ext_fp})')
    else:
        print(f'\nProcessing time-integrated spectra for {ctx.name} (versioned fp={ext_fp})')

    if not force and _tint_src_complete(spec_dir, ctx.sel_dets, spec_name):
        print(f'  Skip (exists). Pass force=True to rerun.')
    else:
        jobs = [{
            'ctx': ctx.to_dict(),
            'det': det,
            'spec_dir': spec_dir,
            'spec_name': spec_name,
        } for det in ctx.sel_dets]
        results = map_parallel(
            _extract_one_det_tint, jobs, n_workers=n_workers, desc='tint')
        failed = [r for r in results if not r or not r.get('ok')]
        if failed:
            print('  WARNING: some detectors failed:')
            for r in failed:
                print(f"    {r.get('det')}: {r.get('error')}")
        offsets = [r.get('time_offset') for r in results if r and r.get('ok')]
        time_offset = offsets[0] if offsets else None
        if ctx.lc_combined_band:
            try:
                write_combined_lc(ctx, spec_dir, time_offset)
            except Exception as e:
                print(f'  Combined LC failed ({e}); summing native-band LCs')
                extract_combined_from_native(ctx, spec_dir, time_offset)
            combined = os.path.join(spec_dir, 'lc_combined.json')
            canonical_combined = os.path.join(
                ctx.paths.heapy_tintegrated_path, 'lc_combined.json')
            if os.path.isfile(combined) and spec_dir != ctx.paths.heapy_tintegrated_path:
                shutil.copy2(combined, canonical_combined)

    commit_meta(ctx.paths.heapy_tintegrated_path, 'canonical_extraction', ext_fp, {
        't1': ctx.t1, 't2': ctx.t2, 'sel_dets': ctx.sel_dets,
        'slice_mode': ctx.slice_mode, 'time_slices': ctx.time_slices,
        'spec_rebn': ctx.spec_rebn,
        'lc_pad_pre': ctx.lc_pad_pre, 'lc_pad_post': ctx.resolved_lc_pad_post,
        'note': 'heapy pgSignal background != one_fits_all baseline algorithm',
    })
    # BB-assisted slice proposal after LC exists
    if ctx.slice_mode in ('bb', 'bb_manual'):
        from .slices import resolve_time_slices
        resolve_time_slices(ctx, heapy_dir=spec_dir, persist=True)
    print(f'Time-integrated spectra saved to: {spec_dir}')
    return spec_dir


def extract_tresolved_spectra(ctx, force=False, n_workers=None):
    """Extract time-resolved spectra for current GRB (all slices, all detectors)."""
    from .slices import resolve_time_slices

    if ctx.slice_mode in ('bb', 'bb_manual'):
        resolve_time_slices(ctx, persist=True)
    if not ctx.time_slices:
        raise ValueError(f'{ctx.name}: time_slices not set')

    ext_fp = extraction_fingerprint(ctx)
    heapy_root = resolve_versioned_dir(
        ctx.paths.heapy_tresolved_path, ext_fp, 'canonical_extraction')
    os.makedirs(heapy_root, exist_ok=True)

    print(f'#--- {ctx.name} time-resolved extraction (fp={ext_fp}) ---#')
    if heapy_root != ctx.paths.heapy_tresolved_path:
        print(f'  Versioned dir: {heapy_root}')

    workers = default_n_workers(n_workers)
    for slice_index, (t_start, t_stop) in enumerate(ctx.time_slices, 1):
        spec_name = spec_slice_name(t_start, t_stop)
        print(f'\nSlice {slice_index}: {spec_name} ({t_start}, {t_stop})')
        slice_dir = os.path.join(heapy_root, f'slice_{slice_index:02d}')
        os.makedirs(slice_dir, exist_ok=True)
        if not force and all(
                os.path.isfile(os.path.join(slice_dir, f'{spec_name}_{det}.src'))
                for det in ctx.sel_dets):
            print('  Skip (exists). Pass force=True to rerun.')
            continue
        jobs = [{
            'ctx': ctx.to_dict(),
            'det': det,
            't_start': t_start,
            't_stop': t_stop,
            'slice_dir': slice_dir,
            'spec_name': spec_name,
        } for det in ctx.sel_dets]
        map_parallel(_extract_one_det_tres, jobs, n_workers=workers, desc=f'slice{slice_index:02d}')

    commit_meta(ctx.paths.heapy_tresolved_path, 'canonical_extraction', ext_fp, {
        'time_slices': ctx.time_slices, 'sel_dets': ctx.sel_dets,
        'slice_mode': ctx.slice_mode,
    })
    print(f'\nTime-resolved spectra saved in: {heapy_root}')
    return heapy_root


def extract_pulse_lightcurve(ctx, det=None, en_low=8, en_high=50, binsize=0.5,
                             pad_pre=10, pad_post=10):
    """Full-burst low-energy lightcurve for Norris-pulse onset fits."""
    import base64
    import json
    import struct

    import numpy as np
    import pandas as pd
    from heapy.pipe.event import gbmTTE

    if det is None:
        nai_dets = [d for d in ctx.sel_dets if d[0] == 'n']
        if not nai_dets:
            raise ValueError("No NaI ('n'-prefix) detector in sel_dets; pass det= explicitly")
        det = nai_dets[0]

    gbm_tte = gbmTTE(ctx.gbm_rtv.rtv_res['tte'][det], ctx.gbm_rtv.rtv_res['poshist'])
    time_offset = ctx.fermi_met - gbm_tte.timezero
    print(f'[lc_lowE] det={det}, offset={time_offset:.2f}s, energy=[{en_low},{en_high}] keV')

    gbm_tte.event
    span0, span1 = ctx.burst_span()
    t0 = span0 - pad_pre
    t1 = span1 + pad_post
    gbm_tte.filter_time([time_offset + t0, time_offset + t1])
    gbm_tte.filter_energy([en_low, en_high])

    temp_dir = os.path.join(ctx.paths.tresolved_path, f'temp_lc_{det}')
    os.makedirs(temp_dir, exist_ok=True)

    gbm_tte.lc_binsize = binsize
    ignore = ctx.bs_ignore_interval(time_offset)
    if ignore is not None:
        gbm_tte.bs_ignore = [ignore]
    gbm_tte.bs_deg = 1
    gbm_tte.extract_curve(savepath=temp_dir, show=False)

    with open(os.path.join(temp_dir, 'lc.json')) as f:
        fig = json.load(f)

    def _decode(f8_field):
        raw = base64.b64decode(f8_field['bdata'])
        n = len(raw) // 8
        return np.array(struct.unpack(f'<{n}d', raw[:n * 8]))

    src = fig['data'][0]
    t_abs = _decode(src['x'])
    rate = _decode(src['y'])
    rate_err = _decode(src['error_y']['array'])
    t_rel = t_abs - time_offset

    df = pd.DataFrame({'t': t_rel, 'rate': rate, 'rate_err': rate_err})
    out_path = os.path.join(ctx.paths.tresolved_path, f'{ctx.name}_lc_lowE.h5')
    df.to_hdf(out_path, key=f'lc_{det}', mode='a')
    print(f"[lc_lowE] Saved {len(df)} bins -> {out_path} (key='lc_{det}')")
    shutil.rmtree(temp_dir)
    return df


def _dir_with_src(root, fp, meta_key):
    resolved = resolve_versioned_dir(root, fp, meta_key)
    versioned = os.path.join(root, 'versions', fp)

    def has_src(path):
        if not os.path.isdir(path):
            return False
        for dirpath, _, filenames in os.walk(path):
            if any(name.endswith('.src') for name in filenames):
                return True
        return False

    if has_src(versioned):
        return versioned
    return resolved


def active_heapy_tint_dir(ctx):
    return _dir_with_src(
        ctx.paths.heapy_tintegrated_path, extraction_fingerprint(ctx),
        'canonical_extraction')


def active_heapy_tres_dir(ctx):
    return _dir_with_src(
        ctx.paths.heapy_tresolved_path, extraction_fingerprint(ctx),
        'canonical_extraction')


def tint_spec_base(ctx, det):
    heapy_dir = active_heapy_tint_dir(ctx)
    versioned = os.path.join(heapy_dir, f'tintegrated_{det}')
    if os.path.isfile(f'{versioned}.src'):
        return versioned
    canonical = os.path.join(ctx.paths.heapy_tintegrated_path, f'tintegrated_{det}')
    if os.path.isfile(f'{canonical}.src'):
        return canonical
    return versioned
