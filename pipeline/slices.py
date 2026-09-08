"""Time-slice helpers: Table C1 boundaries, manual / BB / hybrid modes."""

import json
import os

VALID_SLICE_MODES = ('manual', 'bb', 'bb_manual')


def boundaries_to_slices(bounds):
    """Convert a sorted list of edges ``[t0, t1, ..., tn]`` to ``[(t0,t1), ...].``"""
    bounds = [float(b) for b in bounds]
    if len(bounds) < 2:
        raise ValueError(f'Need >=2 slice boundaries, got {bounds}')
    for i in range(1, len(bounds)):
        if bounds[i] <= bounds[i - 1]:
            raise ValueError(f'slice_boundaries must be strictly increasing: {bounds}')
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def slices_to_boundaries(slices):
    """Inverse of ``boundaries_to_slices``."""
    if not slices:
        return []
    bounds = [float(slices[0][0])]
    for t0, t1 in slices:
        if bounds[-1] != float(t0):
            raise ValueError(f'Non-contiguous slices: {slices}')
        bounds.append(float(t1))
    return bounds


def burst_span(t1=None, t2=None, time_slices=None, slice_boundaries=None):
    """Full burst interval used for tint window and ``bs_ignore``."""
    starts, stops = [], []
    if t1 is not None:
        starts.append(float(t1))
    if t2 is not None:
        stops.append(float(t2))
    if time_slices:
        starts.append(float(time_slices[0][0]))
        stops.append(float(time_slices[-1][1]))
    if slice_boundaries:
        starts.append(float(slice_boundaries[0]))
        stops.append(float(slice_boundaries[-1]))
    if not starts or not stops:
        return None, None
    return min(starts), max(stops)


def _edges_rel_from_bb_payload(data, time_offset=None):
    """Trigger-relative Bayesian-block edges from ``bb_lc.json``."""
    if data.get('edges_rel'):
        return [float(e) for e in data['edges_rel']]
    edges = [float(e) for e in data.get('edges', [])]
    offset = data.get('time_offset', time_offset)
    if offset is None:
        # Heuristic: heapy LC is in timezero frame (~hundreds of seconds).
        if edges and abs(edges[0]) > 100:
            raise ValueError(
                'bb_lc.json has no time_offset; re-run LC extraction or pass offset')
        return edges
    return [e - float(offset) for e in edges]


def slices_from_bb_lc(bb_path, t1, t2, time_offset=None, min_width=0.0):
    """Build slices from heapy pgSignal BB edges, clipped to ``[t1, t2]``."""
    with open(bb_path) as f:
        data = json.load(f)
    edges_rel = _edges_rel_from_bb_payload(data, time_offset=time_offset)
    interior = [e for e in edges_rel if t1 < e < t2]
    bounds = [float(t1)]
    for e in interior:
        if e - bounds[-1] > min_width:
            bounds.append(e)
    if float(t2) - bounds[-1] > min_width:
        bounds.append(float(t2))
    elif bounds[-1] != float(t2):
        bounds[-1] = float(t2)
    if len(bounds) < 2:
        bounds = [float(t1), float(t2)]
    return boundaries_to_slices(bounds), bounds


def _pick_bb_reference_det(heapy_dir, sel_dets):
    """Prefer first NaI with bb_lc.json, else any detector that has it."""
    nai = [d for d in sel_dets if d[0] == 'n']
    for det in nai + list(sel_dets):
        path = os.path.join(heapy_dir, det, 'bb_lc.json')
        if os.path.isfile(path):
            return det, path
    return None, None


def persist_resolved_slices(meta_root, payload):
    """Write resolved slices into ``pipeline_meta.json`` (and ``bb_slices.json``)."""
    from .paths import read_meta, write_meta

    os.makedirs(meta_root, exist_ok=True)
    meta_path = os.path.join(meta_root, 'pipeline_meta.json')
    meta = read_meta(meta_path)
    meta['slices'] = payload
    write_meta(meta_path, meta)
    if payload.get('bb_boundaries') is not None:
        bb_path = os.path.join(meta_root, 'bb_slices.json')
        with open(bb_path, 'w') as f:
            json.dump(payload, f, indent=2, default=str)
    return meta_path


def resolve_time_slices(ctx, heapy_dir=None, time_offset=None, persist=True):
    """Resolve ``ctx.time_slices`` from ``slice_mode``.

    Modes
    -----
    manual
        Use ``slice_boundaries`` or existing ``time_slices``. Default for
        one_fits_all Table C1 reproduction.
    bb
        Replace slices with heapy pgSignal Bayesian-block edges (opt-in).
        Requires ``bb_lc.json`` from a prior LC extraction.
    bb_manual
        Compute BB proposal and write ``bb_slices.json``, but keep manual
        slices for spectral extraction/fitting.
    """
    mode = getattr(ctx, 'slice_mode', 'manual') or 'manual'
    if mode not in VALID_SLICE_MODES:
        raise ValueError(f"slice_mode must be one of {VALID_SLICE_MODES}, got {mode!r}")

    manual_bounds = list(ctx.slice_boundaries) if ctx.slice_boundaries else None
    if manual_bounds is None and ctx.time_slices:
        manual_bounds = slices_to_boundaries(ctx.time_slices)

    manual_slices = (
        boundaries_to_slices(manual_bounds) if manual_bounds else ctx.time_slices)

    t1, t2 = burst_span(ctx.t1, ctx.t2, manual_slices, manual_bounds)
    if ctx.t1 is None:
        ctx.t1 = t1
    if ctx.t2 is None:
        ctx.t2 = t2

    bb_slices, bb_bounds = None, None
    bb_det = None
    if mode in ('bb', 'bb_manual'):
        heapy_dir = heapy_dir or ctx.paths.heapy_tintegrated_path
        bb_params = ctx.bb_slice_params or {}
        min_width = float(bb_params.get('min_width', 0.0))
        bb_det, bb_path = _pick_bb_reference_det(heapy_dir, ctx.sel_dets or [])
        if bb_path is None:
            raise FileNotFoundError(
                f'No bb_lc.json under {heapy_dir} for dets={ctx.sel_dets}. '
                'Run spectra_tint first, then slice_mode bb / bb_manual.')
        if t1 is None or t2 is None:
            raise ValueError('t1/t2 required to clip BB slices')
        bb_slices, bb_bounds = slices_from_bb_lc(
            bb_path, t1, t2, time_offset=time_offset, min_width=min_width)

    if mode == 'bb':
        ctx.time_slices = bb_slices
        ctx.slice_boundaries = bb_bounds
    else:
        if not manual_slices:
            raise ValueError(
                f"{ctx.name}: slice_mode={mode!r} needs slice_boundaries or time_slices")
        ctx.time_slices = [tuple(s) for s in manual_slices]
        ctx.slice_boundaries = manual_bounds or slices_to_boundaries(ctx.time_slices)

    payload = {
        'slice_mode': mode,
        'time_slices': [list(s) for s in ctx.time_slices],
        'slice_boundaries': ctx.slice_boundaries,
        'bb_boundaries': bb_bounds,
        'bb_slices': [list(s) for s in bb_slices] if bb_slices else None,
        'bb_det': bb_det,
        't1': ctx.t1,
        't2': ctx.t2,
    }
    if persist and ctx.paths is not None:
        persist_resolved_slices(ctx.paths.heapy_tintegrated_path, payload)
    return ctx.time_slices
