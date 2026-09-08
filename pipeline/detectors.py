"""Detector selection: manual, auto (top-2 NaI + 1 BGO), one_fits_all."""

from .constants import (
    ALL_DETS,
    DEFAULT_MAX_DETS,
    DEFAULT_N_BGO,
    DEFAULT_N_NAI,
    DEFAULT_NAI_MAX_ANGLE,
)


def detector_energy_range(det, band=None, ctx=None):
    if ctx is not None and band is None:
        native = list(ctx.nai_energy if str(det)[:1].lower() == 'n' else ctx.bgo_energy)
    else:
        native = [8, 900] if det[0] == 'n' else [300, 38000]
    if not band:
        return native
    lo, hi = float(band[0]), float(band[1])
    return [max(native[0], lo), min(native[1], hi)]


def rank_detector_angles(ctx, dets=None):
    from heapy.geos.geometry import gbmGeometry

    dets = dets or ALL_DETS
    gbm_geo = gbmGeometry(file=ctx.gbm_rtv.rtv_res['poshist'])
    ranked = []
    for det in dets:
        angles = gbm_geo.get_detector_angle(
            ra=ctx.ra, dec=ctx.dec, det=det,
            met=[ctx.fermi_met - 100, ctx.fermi_met, ctx.fermi_met + 100])
        ranked.append((det, float(angles[1])))
    ranked.sort(key=lambda x: x[1])
    return ranked


def suggest_detectors(ctx=None, ra=None, dec=None, gbm_rtv_obj=None,
                      fermi_met_val=None, n_nai=DEFAULT_N_NAI, n_bgo=DEFAULT_N_BGO,
                      max_angle=DEFAULT_NAI_MAX_ANGLE, max_dets=DEFAULT_MAX_DETS):
    """Auto-select detectors by smallest angle to source.

    Default policy (production): top 2 NaI with angle < 60° plus the closest
    BGO (no 60° cut on BGO — BGO angles are often larger), max 4 detectors.
    """
    if ctx is not None:
        ra = ctx.ra if ra is None else ra
        dec = ctx.dec if dec is None else dec
        gbm_rtv_obj = ctx.gbm_rtv if gbm_rtv_obj is None else gbm_rtv_obj
        fermi_met_val = ctx.fermi_met if fermi_met_val is None else fermi_met_val

    from heapy.geos.geometry import gbmGeometry
    import numpy as np

    gbm_geo = gbmGeometry(file=gbm_rtv_obj.rtv_res['poshist'])
    visible = gbm_geo.get_location_visible(
        ra=ra, dec=dec, met=[fermi_met_val - 500, fermi_met_val, fermi_met_val + 500])
    if not np.all(visible):
        print(f'  WARNING: source not fully visible at trigger: {visible}')

    ranked = []
    for det in ALL_DETS:
        angles = gbm_geo.get_detector_angle(
            ra=ra, dec=dec, det=det,
            met=[fermi_met_val - 100, fermi_met_val, fermi_met_val + 100])
        ranked.append((det, float(angles[1])))
    ranked.sort(key=lambda x: x[1])

    print('\nDetector angles at trigger (sorted):')
    for det, ang in ranked:
        print(f'  {det}: {ang:.1f}°')

    nai = [d for d, a in ranked if d[0] == 'n' and a <= max_angle][:n_nai]
    # BGO: closest, no 60° cut (Table C1 often has BGO at >60°)
    bgo = [d for d, a in ranked if d[0] == 'b'][:n_bgo]
    chosen = []
    for d in nai + bgo:
        if d not in chosen:
            chosen.append(d)
        if len(chosen) >= max_dets:
            break
    chosen = sorted(chosen)
    if len(chosen) < 2:
        raise ValueError(f'Auto-selection found <2 detectors: {chosen}')
    print(f'  Auto-selected (n_nai={n_nai}, n_bgo={n_bgo}, max={max_dets}): {chosen}')
    return chosen


def is_nai(det):
    return str(det)[:1].lower() == 'n'


def is_bgo(det):
    return str(det)[:1].lower() == 'b'


def nai_dets(dets):
    return [d for d in (dets or []) if is_nai(d)]


def bgo_dets(dets):
    return [d for d in (dets or []) if is_bgo(d)]


def fit_detectors(sel_dets, include_bgo=False, skip_dets=None):
    """Detectors used in a spectral fit.

    Geometry / LC keep the catalog set (often including BGO). Tint fits
    default to NaI only; pass ``include_bgo=True`` to add BGO.
    """
    skip = set(skip_dets or [])
    out = []
    for det in sel_dets or []:
        if det in skip:
            continue
        if not include_bgo and is_bgo(det):
            continue
        out.append(det)
    return out


def resolve_detectors(ctx):
    """Set ``ctx.sel_dets`` from ``det_mode``."""
    mode = (ctx.det_mode or 'manual').lower()
    if mode in ('one_fits_all', 'ofa', 'catalog'):
        if not ctx.sel_dets:
            raise ValueError(f"{ctx.name}: det_mode={mode!r} but sel_dets is empty")
        ctx.sel_dets = list(ctx.sel_dets)
        print(f'  one_fits_all detectors: {ctx.sel_dets}')
        return ctx.sel_dets
    if mode == 'auto':
        ctx.sel_dets = suggest_detectors(ctx)
        return ctx.sel_dets
    if not ctx.sel_dets:
        raise ValueError(f"{ctx.name}: det_mode='manual' but sel_dets is empty")
    ctx.sel_dets = list(ctx.sel_dets)
    return ctx.sel_dets
