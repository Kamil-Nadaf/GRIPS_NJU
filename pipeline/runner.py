"""Stage runner: geometry, spectra, fit, params — sequential or parallel GRBs."""

from .constants import DEFAULT_STAGES
from .context import GRBContext, set_current
from .download import load_grb_context, retrieve
from .detectors import resolve_detectors
from .geometry import extract_geometry
from .parallel import default_n_workers, map_parallel
from .slices import resolve_time_slices
from .spectra import extract_tintegrated_spectra, extract_tresolved_spectra
from .fitting import extract_params, fit_all_time_slices, fit_tintegrated

STAGE_ALIASES = {
    'download': ('download',),
    'geometry': ('geometry',),
    'spectra_tint': ('spectra_tint',),
    'spectra_tres': ('spectra_tres',),
    'spectra': ('spectra_tint', 'spectra_tres'),
    'fit_tint': ('fit_tint',),
    'fit_tres': ('fit_tres',),
    'fit': ('fit_tint', 'fit_tres'),
    'params': ('params',),
    # ``all`` is tint-first. Request spectra_tres / fit_tres explicitly.
    'all': tuple(DEFAULT_STAGES),
}


def expand_stages(stages):
    if stages is None:
        stages = list(DEFAULT_STAGES)
    if isinstance(stages, str):
        stages = [s.strip() for s in stages.split(',') if s.strip()]
    out = []
    for s in stages:
        if s not in STAGE_ALIASES:
            raise ValueError(f'Unknown stage {s!r}. Known: {sorted(STAGE_ALIASES)}')
        for item in STAGE_ALIASES[s]:
            if item not in out:
                out.append(item)
    return out


def _ensure_loaded(ctx):
    if ctx.gbm_rtv is None or ctx.fermi_met is None:
        retrieve(ctx)
    resolve_detectors(ctx)
    if ctx.time_slices is None and ctx.slice_boundaries:
        resolve_time_slices(ctx, persist=False)
    set_current(ctx)
    return ctx


class GRBPipelineRunner:
    def __init__(self, n_workers=None, model_name='cpl', nlive=1000,
                 fixed_params=None, force=False, include_bgo=False):
        self.n_workers = default_n_workers(n_workers)
        self.model_name = model_name
        self.nlive = nlive
        self.fixed_params = fixed_params or {}
        self.force = force
        self.include_bgo = include_bgo

    def load(self, grb, data_base=None):
        if isinstance(grb, GRBContext):
            ctx = grb
            _ensure_loaded(ctx)
            return ctx
        if isinstance(grb, str):
            ctx = GRBContext.from_name(grb, data_base=data_base)
            retrieve(ctx)
            resolve_detectors(ctx)
            try:
                resolve_time_slices(ctx, persist=True)
            except FileNotFoundError:
                pass
            set_current(ctx)
            from .paths import extraction_fingerprint
            print(
                f'Loaded: {ctx.name} | {ctx.n_slices} slices | '
                f'tint=({ctx.t1},{ctx.t2}) | dets={ctx.sel_dets} | '
                f'ext_fp={extraction_fingerprint(ctx)}')
            return ctx
        return load_grb_context(grb, data_base=data_base)

    def run(self, ctx, stages=None, model_name=None, nlive=None,
            fixed_params=None, force=None, n_workers=None, include_bgo=None):
        """Run selected stages for one GRB.

        Default stages are tint-only: geometry, spectra_tint, fit_tint, params.
        Time-resolved stages run only if requested.

        Parameters
        ----------
        ctx : GRBContext, catalog row, or GRB name
        stages : list or comma-separated str
        include_bgo : bool
            Tint/tres fits default to NaI only. True adds BGO.
        """
        model_name = model_name or self.model_name
        nlive = nlive if nlive is not None else self.nlive
        fixed_params = self.fixed_params if fixed_params is None else fixed_params
        force = self.force if force is None else force
        n_workers = self.n_workers if n_workers is None else n_workers
        include_bgo = self.include_bgo if include_bgo is None else include_bgo
        stage_list = expand_stages(stages)
        tres_requested = any(s in stage_list for s in ('spectra_tres', 'fit_tres'))

        if not isinstance(ctx, GRBContext) or ctx.gbm_rtv is None:
            ctx = self.load(ctx)
        else:
            set_current(ctx)

        print(
            f'\n=== {ctx.name} stages={stage_list} workers={n_workers} '
            f'model={model_name} include_bgo={include_bgo} ===')
        for stage in stage_list:
            if stage == 'download':
                retrieve(ctx)
                resolve_detectors(ctx)
            elif stage == 'geometry':
                extract_geometry(ctx)
            elif stage == 'spectra_tint':
                extract_tintegrated_spectra(ctx, force=force, n_workers=n_workers)
            elif stage == 'spectra_tres':
                extract_tresolved_spectra(ctx, force=force, n_workers=n_workers)
            elif stage == 'fit_tint':
                fit_tintegrated(
                    ctx, model_name=model_name, fixed_params=fixed_params,
                    nlive=nlive, force=force, include_bgo=include_bgo)
            elif stage == 'fit_tres':
                fit_all_time_slices(
                    ctx, model_name=model_name, fixed_params=fixed_params,
                    nlive=nlive, force=force, n_workers=n_workers,
                    include_bgo=include_bgo)
            elif stage == 'params':
                extract_params(
                    ctx, model_name=model_name, mode='tintegrated',
                    fixed_params=fixed_params, nlive=nlive,
                    include_bgo=include_bgo)
                if tres_requested:
                    extract_params(
                        ctx, model_name=model_name, mode='tresolved',
                        fixed_params=fixed_params, nlive=nlive,
                        include_bgo=include_bgo)
            else:
                raise ValueError(f'unhandled stage {stage}')
        return ctx

    def run_batch(self, grb_names, stages=None, parallel_grbs=False, **kwargs):
        """Run the pipeline for many GRBs.

        ``parallel_grbs=True`` farms whole GRBs out to ProcessPoolExecutor.
        Detector/slice parallelism still applies inside each GRB via ``n_workers``.
        """
        if isinstance(grb_names, str):
            grb_names = [grb_names]
        names = list(grb_names)
        if not parallel_grbs or len(names) == 1:
            return [self.run(name, stages=stages, **kwargs) for name in names]

        payload = {
            'stages': stages,
            'model_name': kwargs.get('model_name', self.model_name),
            'nlive': kwargs.get('nlive', self.nlive),
            'fixed_params': kwargs.get('fixed_params', self.fixed_params),
            'force': kwargs.get('force', self.force),
            'include_bgo': kwargs.get('include_bgo', self.include_bgo),
            'n_workers': 1 if parallel_grbs else kwargs.get('n_workers', self.n_workers),
        }
        jobs = [{'name': n, **payload} for n in names]
        n_grb_workers = default_n_workers(kwargs.get('n_workers', self.n_workers))
        return map_parallel(_run_one_grb_job, jobs, n_workers=n_grb_workers, desc='grb')


def _run_one_grb_job(job):
    runner = GRBPipelineRunner(
        n_workers=job.get('n_workers', 1),
        model_name=job.get('model_name', 'cpl'),
        nlive=job.get('nlive', 1000),
        fixed_params=job.get('fixed_params') or {},
        force=job.get('force', False),
        include_bgo=job.get('include_bgo', False),
    )
    ctx = runner.run(job['name'], stages=job.get('stages'))
    return ctx.name
