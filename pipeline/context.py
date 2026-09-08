"""GRBContext — replaces ``load_grb()`` inspect.stack() global injection."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict

from .constants import (
    DATA_BASE,
    DEFAULT_BS_P0,
    DEFAULT_LC_BINSIZE,
    DEFAULT_LC_PAD_PRE,
    DEFAULT_LC_REBN,
    DEFAULT_SPEC_FILTER_PAD,
    DEFAULT_SPEC_REBN,
    BGO_ENERGY,
    NAI_ENERGY,
)
from .paths import PathLayout
from .slices import boundaries_to_slices, burst_span

_CURRENT = None
_RUNTIME_KEYS = ('gbm_rtv', 'paths')


def get_current():
    if _CURRENT is None:
        raise RuntimeError(
            'No GRBContext loaded. Call load_grb() or GRBPipelineRunner.load().')
    return _CURRENT


def set_current(ctx):
    global _CURRENT
    _CURRENT = ctx
    return ctx


def _series_get(row, key, default=None):
    if row is None:
        return default
    if hasattr(row, 'index'):
        if key not in row.index:
            return default
        val = row[key]
        try:
            if val is None or (hasattr(val, '__len__') and len(val) == 0 and not isinstance(val, str)):
                if key in ('sel_dets', 'time_slices', 'slice_boundaries') and val is not None:
                    return list(val) if len(val) == 0 else val
        except TypeError:
            pass
        try:
            import pandas as pd
            if pd.isna(val):
                return default
        except Exception:
            pass
        return val
    return row.get(key, default) if isinstance(row, dict) else default


@dataclass
class GRBContext:
    name: str
    ra: float
    dec: float
    utc: str
    det_mode: str = 'manual'
    sel_dets: list | None = None
    t1: float | None = None
    t2: float | None = None
    time_slices: list | None = None
    slice_boundaries: list | None = None
    slice_mode: str = 'manual'
    bb_slice_params: dict | None = None
    retrieve_t1: float = -400.0
    retrieve_t2: float = 400.0
    data_base: str = DATA_BASE
    z: float | None = None
    note: str | None = None

    lc_binsize: float = DEFAULT_LC_BINSIZE
    lc_pad_pre: float = DEFAULT_LC_PAD_PRE
    lc_pad_post: float | None = None
    lc_combined_band: list | None = None
    lc_rebn: dict = field(default_factory=lambda: dict(DEFAULT_LC_REBN))
    spec_rebn: dict = field(default_factory=lambda: dict(DEFAULT_SPEC_REBN))
    spec_filter_pad: float = DEFAULT_SPEC_FILTER_PAD
    bs_p0: float = DEFAULT_BS_P0
    rebin: bool = True
    nai_energy: list = field(default_factory=lambda: list(NAI_ENERGY))
    bgo_energy: list = field(default_factory=lambda: list(BGO_ENERGY))

    fermi_met: float | None = None
    gbm_rtv: object | None = field(default=None, repr=False)
    paths: PathLayout | None = field(default=None, repr=False)

    def __post_init__(self):
        if self.sel_dets is not None:
            self.sel_dets = list(self.sel_dets)
        if self.time_slices is not None:
            self.time_slices = [tuple(s) for s in self.time_slices]
        if self.slice_boundaries is not None:
            self.slice_boundaries = [float(b) for b in self.slice_boundaries]
            if not self.time_slices:
                self.time_slices = boundaries_to_slices(self.slice_boundaries)
        if self.time_slices and not self.slice_boundaries:
            from .slices import slices_to_boundaries
            self.slice_boundaries = slices_to_boundaries(self.time_slices)
        span0, span1 = burst_span(
            self.t1, self.t2, self.time_slices, self.slice_boundaries)
        if self.t1 is None:
            self.t1 = span0
        if self.t2 is None:
            self.t2 = span1
        if self.nai_energy is None:
            self.nai_energy = list(NAI_ENERGY)
        else:
            self.nai_energy = [float(x) for x in self.nai_energy]
        if self.bgo_energy is None:
            self.bgo_energy = list(BGO_ENERGY)
        else:
            self.bgo_energy = [float(x) for x in self.bgo_energy]
        if self.paths is None:
            self.paths = PathLayout(self.name, data_base=self.data_base)
        if self.lc_rebn is None:
            self.lc_rebn = dict(DEFAULT_LC_REBN)
        if self.spec_rebn is None:
            self.spec_rebn = dict(DEFAULT_SPEC_REBN)

    @property
    def GRB_Name(self):
        return self.name

    @property
    def RA(self):
        return self.ra

    @property
    def DEC(self):
        return self.dec

    @property
    def n_slices(self):
        return len(self.time_slices) if self.time_slices else 0

    @property
    def resolved_lc_pad_post(self):
        if self.lc_pad_post is not None:
            return float(self.lc_pad_post)
        duration = 0.0
        if self.t1 is not None and self.t2 is not None:
            duration = float(self.t2) - float(self.t1)
        return max(50.0, duration + 20.0)

    def burst_span(self):
        return burst_span(self.t1, self.t2, self.time_slices, self.slice_boundaries)

    def lc_window(self, time_offset):
        """``[t_lo, t_hi]`` in the heapy timezero frame.

        End is ``max(t2, last_slice_end) + pad_post``, not a hard +50 s clip
        from trigger (that clipped long bursts such as GRB131011A at 25 s).
        """
        span0, span1 = self.burst_span()
        last_end = span1 if span1 is not None else 0.0
        pad_post = self.resolved_lc_pad_post
        return [time_offset - self.lc_pad_pre, time_offset + last_end + pad_post]

    def spec_window(self, time_offset, t_lo=None, t_hi=None):
        """Heapy ``filter_time`` for spectrum/DRM extraction.

        Default pad is ``spec_filter_pad`` (380 s), but TTE/poshist often
        start only a few seconds before trigger. Clip to ``[t_lo, t_hi]``
        (event coverage) so heapy does not extrapolate and segfault in workers.
        """
        pad = float(self.spec_filter_pad)
        lo = float(time_offset) - pad
        hi = float(time_offset) + pad
        if t_lo is not None:
            lo = max(lo, float(t_lo))
        if t_hi is not None:
            hi = min(hi, float(t_hi))
        if hi <= lo:
            hi = lo + 1.0
        return [lo, hi]

    def bs_ignore_interval(self, time_offset):
        span0, span1 = self.burst_span()
        if span0 is None or span1 is None:
            return None
        return [time_offset + span0, time_offset + span1]

    def to_dict(self):
        d = asdict(self)
        d.pop('gbm_rtv', None)
        d.pop('paths', None)
        d['data_base'] = self.data_base
        return d

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data.pop('gbm_rtv', None)
        data.pop('paths', None)
        return cls(**data)

    @classmethod
    def from_row(cls, row, data_base=None):
        """Build from a ``grbs_df`` row (Series or dict)."""
        name = _series_get(row, 'name')
        bounds = _series_get(row, 'slice_boundaries')
        if bounds is not None:
            bounds = list(bounds)
        slices = _series_get(row, 'time_slices')
        if slices is not None:
            slices = [tuple(s) for s in slices]
        dets = _series_get(row, 'sel_dets')
        if dets is not None:
            dets = list(dets)
        bb_params = _series_get(row, 'bb_slice_params')
        if bb_params is not None and hasattr(bb_params, 'to_dict'):
            bb_params = dict(bb_params)
        combined = _series_get(row, 'lc_combined_band')
        if combined is not None:
            combined = list(combined)
        return cls(
            name=name,
            ra=float(_series_get(row, 'ra')),
            dec=float(_series_get(row, 'dec')),
            utc=str(_series_get(row, 'utc')),
            det_mode=_series_get(row, 'det_mode', 'manual') or 'manual',
            sel_dets=dets,
            t1=_series_get(row, 't1'),
            t2=_series_get(row, 't2'),
            time_slices=slices,
            slice_boundaries=bounds,
            slice_mode=_series_get(row, 'slice_mode', 'manual') or 'manual',
            bb_slice_params=bb_params if isinstance(bb_params, dict) else None,
            retrieve_t1=float(_series_get(row, 'retrieve_t1', -400)),
            retrieve_t2=float(_series_get(row, 'retrieve_t2', 400)),
            data_base=data_base or DATA_BASE,
            z=_series_get(row, 'z'),
            note=_series_get(row, 'note'),
            lc_combined_band=combined,
        )

    @classmethod
    def from_name(cls, grb_name, grbs_df=None, data_base=None):
        if grbs_df is None:
            import importlib
            cfg = None
            for modname in ('grb_config',):
                try:
                    cfg = importlib.import_module(modname)
                    break
                except ImportError:
                    continue
            if cfg is None:
                raise ImportError('grb_config not found')
            grbs_df = getattr(cfg, 'grbs_df_all', cfg.grbs_df)
        hits = grbs_df[grbs_df['name'] == grb_name]
        if hits.empty:
            raise KeyError(f'{grb_name} not in catalog')
        return cls.from_row(hits.iloc[0], data_base=data_base)

    def as_globals(self):
        """Flatten for notebook / ``grb_utils`` shim injection."""
        g = {
            'GRB_Name': self.name,
            'RA': self.ra,
            'DEC': self.dec,
            'utc': self.utc,
            'det_mode': self.det_mode,
            'sel_dets': self.sel_dets,
            'time_slices': self.time_slices,
            'slice_boundaries': self.slice_boundaries,
            'slice_mode': self.slice_mode,
            'bb_slice_params': self.bb_slice_params,
            'n_slices': self.n_slices,
            't1': self.t1,
            't2': self.t2,
            'retrieve_t1': self.retrieve_t1,
            'retrieve_t2': self.retrieve_t2,
            'fermi_met': self.fermi_met,
            'gbm_rtv': self.gbm_rtv,
            'spec_rebn': self.spec_rebn,
            'lc_rebn': self.lc_rebn,
            'lc_binsize': self.lc_binsize,
            'lc_pad_pre': self.lc_pad_pre,
            'lc_pad_post': self.resolved_lc_pad_post,
            'lc_combined_band': self.lc_combined_band,
            'bs_p0': self.bs_p0,
            'ctx': self,
        }
        if self.paths is not None:
            g.update(self.paths.as_globals())
        if self.fermi_met is not None and self.sel_dets is not None:
            from .paths import extraction_fingerprint
            g['ext_fp'] = extraction_fingerprint(self)
        return g

    def copy(self):
        return copy.deepcopy(self)
