"""Path layout, extraction/fit fingerprints, and versioned output dirs."""

import hashlib
import json
import os

from .constants import DATA_BASE


def fingerprint(data):
    payload = json.dumps(data, sort_keys=True, default=str)
    return hashlib.md5(payload.encode()).hexdigest()[:8]


def spec_slice_name(t_start, t_stop):
    if t_start < 0:
        return f'm{abs(t_start):.2f}'.replace('.', 'd') + f'_p{t_stop:.2f}'.replace('.', 'd')
    return f'p{t_start:.2f}'.replace('.', 'd') + f'_p{t_stop:.2f}'.replace('.', 'd')


def read_meta(meta_path):
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {}


def write_meta(meta_path, meta):
    os.makedirs(os.path.dirname(meta_path) or '.', exist_ok=True)
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)


def resolve_versioned_dir(root, fp, meta_key):
    """Return canonical root if fingerprint matches (or first run); else versions/{fp}/."""
    meta_path = os.path.join(root, 'pipeline_meta.json')
    meta = read_meta(meta_path)
    canonical = meta.get(meta_key)
    if canonical is None or canonical == fp:
        return root
    versioned = os.path.join(root, 'versions', fp)
    os.makedirs(versioned, exist_ok=True)
    return versioned


def fit_meta_roots(model_root):
    """Dirs that may hold ``pipeline_meta.json`` for a fit.

    Fits write products under ``.../bayspec/{model}`` but older runs stored
    meta one level up in ``.../bayspec/``. Check both so a fingerprint
    change actually version-directs instead of skipping stale weights.
    """
    parent = os.path.dirname(model_root)
    roots = [model_root]
    if parent and parent not in roots:
        roots.append(parent)
    return roots


def resolve_fit_dir(model_root, fp, meta_key, force=False):
    """Resolve where a fit should write.

    ``force=True`` overwrites the canonical model directory (the usual
    "rerun this GRB" path). Otherwise a fingerprint mismatch goes to
    ``{model_root}/versions/{fp}/``.
    """
    if force:
        return model_root
    for root in fit_meta_roots(model_root):
        canonical = read_meta(os.path.join(root, 'pipeline_meta.json')).get(meta_key)
        if canonical is None:
            continue
        if canonical == fp:
            return model_root
        versioned = os.path.join(model_root, 'versions', fp)
        os.makedirs(versioned, exist_ok=True)
        return versioned
    return model_root


def active_fit_dir(model_root, fp, meta_key):
    """Prefer ``versions/{fp}`` if that run actually wrote weights."""
    versioned = os.path.join(model_root, 'versions', fp)
    if os.path.isfile(os.path.join(versioned, '1-post_equal_weights.dat')):
        return versioned
    return resolve_fit_dir(model_root, fp, meta_key, force=False)


def commit_meta(root, meta_key, fp, params):
    meta_path = os.path.join(root, 'pipeline_meta.json')
    meta = read_meta(meta_path)
    meta[meta_key] = fp
    meta.setdefault('runs', {})[fp] = params
    write_meta(meta_path, meta)


class PathLayout:
    """On-disk layout under ``DATA_BASE/{GRB_NAME}/``."""

    def __init__(self, grb_name, data_base=None):
        self.grb_name = grb_name
        self.data_base = data_base or DATA_BASE

    @property
    def base(self):
        return os.path.join(self.data_base, self.grb_name)

    @property
    def gbm_data(self):
        return os.path.join(self.data_base, 'gbm_data')

    @property
    def geometry(self):
        return os.path.join(self.base, 'geometry')

    @property
    def tintegrated_path(self):
        return os.path.join(self.base, 'data/tintegrated')

    @property
    def heapy_tintegrated_path(self):
        return os.path.join(self.base, 'data/tintegrated/heapy')

    @property
    def bayspec_tintegrated_path(self):
        return os.path.join(self.base, 'data/tintegrated/bayspec')

    @property
    def tresolved_path(self):
        return os.path.join(self.base, 'data/tresolved')

    @property
    def heapy_tresolved_path(self):
        return os.path.join(self.base, 'data/tresolved/heapy')

    @property
    def bayspec_tresolved_path(self):
        return os.path.join(self.base, 'data/tresolved/bayspec')

    @property
    def threeml_tintegrated_path(self):
        return os.path.join(self.base, 'data/tintegrated/3ML')

    @property
    def threeml_tresolved_path(self):
        return os.path.join(self.base, 'data/tresolved/3ML')

    @property
    def threeml_data(self):
        return os.path.join(
            self.threeml_tresolved_path, f'{self.grb_name}_3ML_data.h5')

    @property
    def bayspec_data(self):
        return os.path.join(
            self.bayspec_tresolved_path, f'{self.grb_name}_bayspec_data.h5')

    @property
    def sampler_output_dir(self):
        return os.path.join(self.base, 'data/Sampler_Output')

    @property
    def posterior_stat_file(self):
        return os.path.join(
            self.sampler_output_dir, f'{self.grb_name}_posterior_stat.h5')

    def ensure_dirs(self):
        os.makedirs(self.bayspec_tresolved_path, exist_ok=True)
        os.makedirs(self.heapy_tintegrated_path, exist_ok=True)
        os.makedirs(self.heapy_tresolved_path, exist_ok=True)
        os.makedirs(self.bayspec_tintegrated_path, exist_ok=True)
        os.makedirs(self.threeml_tintegrated_path, exist_ok=True)
        os.makedirs(self.threeml_tresolved_path, exist_ok=True)

    def as_globals(self):
        return {
            'DATA_BASE': self.data_base,
            'base': self.base,
            'gbm_data': self.gbm_data,
            'tintegrated_path': self.tintegrated_path,
            'heapy_tintegrated_path': self.heapy_tintegrated_path,
            'bayspec_tintegrated_path': self.bayspec_tintegrated_path,
            'tresolved_path': self.tresolved_path,
            'heapy_tresolved_path': self.heapy_tresolved_path,
            'bayspec_tresolved_path': self.bayspec_tresolved_path,
            'threeml_tintegrated_path': self.threeml_tintegrated_path,
            'threeml_tresolved_path': self.threeml_tresolved_path,
            'threeml_data': self.threeml_data,
            'bayspec_data': self.bayspec_data,
            'Sampler_Output_DIR': self.sampler_output_dir,
            'Posterior_Stat_File': self.posterior_stat_file,
        }


def extraction_fingerprint(ctx):
    """Hash of extraction-relevant config (window, slices, dets, rebin)."""
    return fingerprint({
        't1': ctx.t1,
        't2': ctx.t2,
        'utc': getattr(ctx, 'utc', None),
        'time_slices': ctx.time_slices,
        'sel_dets': sorted(ctx.sel_dets) if ctx.sel_dets else [],
        'retrieve_t1': ctx.retrieve_t1,
        'retrieve_t2': ctx.retrieve_t2,
        'slice_mode': ctx.slice_mode,
        'spec_rebn': ctx.spec_rebn,
        'lc_pad_pre': ctx.lc_pad_pre,
        'lc_pad_post': ctx.resolved_lc_pad_post,
        'bs_ignore_tint': True,
    })


def fit_fingerprint(ctx, model_name, fixed_params=None, nlive=1000, fit_dets=None):
    if fixed_params is None:
        fixed_params = {}
    dets = list(fit_dets) if fit_dets is not None else list(ctx.sel_dets or [])
    return fingerprint({
        'extraction': extraction_fingerprint(ctx),
        'model': model_name,
        'fixed_params': fixed_params,
        'nlive': nlive,
        'spec_rebn': ctx.spec_rebn,
        'fit_dets': sorted(dets),
    })
