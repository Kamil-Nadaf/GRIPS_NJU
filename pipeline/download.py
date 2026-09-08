"""GBM TTE + poshist download via heapy ``gbmRetrieve``."""

from .context import GRBContext, set_current


def retrieve(ctx: GRBContext):
    """Download (or reuse cached) TTE/poshist and fill ``fermi_met`` / ``gbm_rtv``."""
    from heapy.util.time import fermi_utc_to_met
    from heapy.data.retrieve import gbmRetrieve

    ctx.paths.ensure_dirs()
    ctx.fermi_met = fermi_utc_to_met(ctx.utc)
    ctx.gbm_rtv = gbmRetrieve.from_utc(
        utc=ctx.utc, t1=ctx.retrieve_t1, t2=ctx.retrieve_t2,
        datapath=ctx.paths.gbm_data)
    return ctx


def load_grb_context(grb, data_base=None):
    """Build a fully retrieved ``GRBContext`` from a catalog row."""
    ctx = GRBContext.from_row(grb, data_base=data_base)
    retrieve(ctx)
    from .detectors import resolve_detectors
    resolve_detectors(ctx)
    if ctx.sel_dets is None or len(ctx.sel_dets) < 2:
        raise ValueError(f'{ctx.name}: need >=2 detectors, got {ctx.sel_dets}')
    if ctx.t1 is None or ctx.t2 is None:
        print(f'  WARNING: {ctx.name} t1/t2 not set — required for tint spectra')
    from .slices import resolve_time_slices
    try:
        resolve_time_slices(ctx, persist=True)
    except FileNotFoundError:
        # bb / bb_manual need LC products; resolve manual slices now, BB later
        if ctx.slice_mode in ('bb', 'bb_manual') and ctx.time_slices:
            print(f'  {ctx.name}: BB slices deferred until after LC extraction')
        elif ctx.slice_mode == 'manual':
            raise
    from .paths import extraction_fingerprint
    ext_fp = extraction_fingerprint(ctx)
    print(
        f'Loaded: {ctx.name} | RA={ctx.ra} | DEC={ctx.dec} | {ctx.n_slices} slices | '
        f'tint=({ctx.t1},{ctx.t2}) | slice_mode={ctx.slice_mode} | '
        f'det_mode={ctx.det_mode} | dets={ctx.sel_dets} | ext_fp={ext_fp}')
    set_current(ctx)
    return ctx
