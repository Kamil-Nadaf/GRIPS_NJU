import os

import pandas as pd


def _boundaries_to_slices(bounds):
    return [(float(bounds[i]), float(bounds[i + 1])) for i in range(len(bounds) - 1)]

# Yan et al. 2024, ApJ 962:85, Appendix C Table C1
# Detectors + slice_boundaries for the 8 FRED pulses.
ONE_FITS_ALL = [
    {
        'name': 'GRB120326A',
        'ra': 273.90471, 'dec': 69.259822,
        'utc': '2012-03-26T01:20:31.51',
        'z': 1.798,
        'sel_dets': ['n0', 'n1'],
        'slice_boundaries': [-2.51, 0.49, 1.49, 2.19, 2.89, 7.49],
    },
    {
        'name': 'GRB131011A',
        'ra': 32.526, 'dec': -4.411,
        'utc': '2013-10-11T17:47:34.99',
        'z': 1.874,
        'sel_dets': ['b1', 'n9', 'na', 'nb'],
        'slice_boundaries': [-2.99, 1.74, 3.56, 5.066, 8.61, 12.746, 25.01],
    },
    {
        'name': 'GRB140606B',
        'ra': 328.12501, 'dec': 32.01458,
        'utc': '2014-06-06T03:11:51.86',
        'z': 0.384,
        'sel_dets': ['b0', 'n3', 'n4', 'n8'],
        'slice_boundaries': [-0.83, 1.345, 1.795, 2.34, 3.34, 4.43, 5.74],
        # GCN 16363 time-averaged CPL (Table C1 slices stay for later tres)
        'tint_t1': -3.0, 'tint_t2': 12.3,
    },
    {
        'name': 'GRB150514A',
        'ra': 74.8750, 'dec': -60.9691,
        'utc': '2015-05-14T18:35:05.35',
        'z': 0.807,
        'sel_dets': ['b0', 'n3', 'n6', 'n7'],
        'slice_boundaries': [-0.46, 0.34, 0.60, 0.83, 1.05, 1.30, 1.55, 1.85, 2.25, 3.05, 6.05],
        # GCN 17819 Band window (Table C1 slices stay for later tres)
        'tint_t1': 0.0, 'tint_t2': 11.3,
        'note': 'single FRED; GCN 17819 Band window (pipeline fits CPL)',
    },
    {
        'name': 'GRB151027A',
        'ra': 272.48695, 'dec': 61.35344,
        'utc': '2015-10-27T03:58:24',
        'z': 0.81,
        'sel_dets': ['b0', 'n0', 'n1', 'n3'],
        'slice_boundaries': [-0.5, 0.58, 1.10, 1.60, 2.30, 3.50],
        'note': 'pulse 1 / first episode',
    },
    {
        'name': 'GRB170607A',
        'ra': 7.36591, 'dec': 9.24334,
        'utc': '2017-06-07T23:17:59.57',
        'z': 0.557,
        'sel_dets': ['b0', 'n2', 'n5'],
        'slice_boundaries': [-1.57, 0.39, 1.02, 2.03, 3.43, 7.23],
    },
    {
        'name': 'GRB190829A',
        'ra': 44.54402, 'dec': -8.95837,
        # GBM trigger (GCN 25575). Catalog previously used Swift T0
        # 19:56:44.60 (~T0+51 s), which selected the second pulse.
        'utc': '2019-08-29T19:55:53.13',
        'z': 0.0785,
        'sel_dets': ['n6', 'n7', 'n9'],
        'slice_boundaries': [-0.81, 0.33, 0.81, 1.26, 1.80, 2.43, 5.07],
        'tint_t1': 0.0, 'tint_t2': 4.0,
        'note': 'pulse 1 / first episode (GBM T0; GCN 25575 CPL window)',
    },
    {
        'name': 'GRB210204A',
        'ra': 117.08071, 'dec': 11.40951,
        'utc': '2021-02-04T06:29:25',
        'z': 0.876,
        'sel_dets': ['b1', 'n7', 'n8'],
        'slice_boundaries': [32.83, 34.565, 36.14, 37.885, 39.54, 44.34],
        'note': 'pulse 2 / second episode',
    },
]


def _catalog_frame(rows):
    df = pd.DataFrame(rows)
    df = df.sort_values('name').reset_index(drop=True)
    df['det_mode'] = 'one_fits_all'
    df['slice_mode'] = 'manual'
    df['time_slices'] = df['slice_boundaries'].apply(_boundaries_to_slices)
    slice_t1 = df['slice_boundaries'].apply(lambda b: float(b[0]))
    slice_t2 = df['slice_boundaries'].apply(lambda b: float(b[-1]))
    if 'tint_t1' not in df.columns:
        df['tint_t1'] = slice_t1
    else:
        df['tint_t1'] = [
            float(v) if pd.notna(v) else float(fb)
            for v, fb in zip(df['tint_t1'], slice_t1)]
    if 'tint_t2' not in df.columns:
        df['tint_t2'] = slice_t2
    else:
        df['tint_t2'] = [
            float(v) if pd.notna(v) else float(fb)
            for v, fb in zip(df['tint_t2'], slice_t2)]
    df['t1'] = df['tint_t1']
    df['t2'] = df['tint_t2']
    df['retrieve_t1'] = -400
    df['retrieve_t2'] = 400
    df['bb_slice_params'] = [{'min_width': 0.0} for _ in range(len(df))]
    df['lc_combined_band'] = [[10, 1000] for _ in range(len(df))]
    return df


# GCN circular references for tint comparison (interval + CPL params).
GCN_REFS = {
    'GRB140606B': {
        'source': 'GCN 16363',
        't1': -3.0,
        't2': 12.3,
        'alpha': -1.22,
        'alpha_err': 0.04,
        'Ep': 473.0,
        'Ep_err': 82.6,
        'model': 'cpl',
        'note': (
            'Burns, Fermi GBM; time-averaged CPL from T0-3.0 to T0+12.3 s; '
            'cutoff parameterized as Epeak'
        ),
    },
    'GRB190829A': {
        'source': 'GCN 25575',
        't1': 0.0,
        't2': 4.0,
        'alpha': -1.41,
        'alpha_err': 0.08,
        'Ep': 130.0,
        'Ep_err': 20.0,
        'model': 'cpl',
        'note': (
            'Lesage et al., Fermi GBM; first pulse T0 to T0+4.0 s '
            '(GBM T0=2019-08-29T19:55:53.13); cutoff parameterized as Epeak. '
            'Second pulse (T0+47.1 to T0+61.4) is Band with Ep=11 keV — not this tint.'
        ),
    },
    'GRB150514A': {
        'source': 'GCN 17819',
        't1': 0.0,
        't2': 11.3,
        # GBM published Band; pipeline fits CPL. Soft Ep, steep beta → CPL≈Band.
        'alpha': -1.34,
        'alpha_err': 0.07,
        'Ep': 73.0,
        'Ep_err': 6.0,
        'model': 'band',
        'note': (
            'Roberts, Zhang & Meegan, Fermi GBM; Band T0 to T0+11.3 s '
            '(alpha=-1.34+/-0.07, Ep=73+/-6, beta=-2.51+/-0.17). '
            'Konus-Wind GCN 17823 CPL on T0 to T0+8.448 s: '
            'alpha=-1.44(-0.29/+0.33), Ep=60(-14/+10) keV (90% CL).'
        ),
    },
}


def gcn_ref(name):
    """Return a copy of the GCN reference dict, or None."""
    ref = GCN_REFS.get(name)
    return dict(ref) if ref else None


grbs_df_all = _catalog_frame(ONE_FITS_ALL)

# Interactive / notebook default: one GRB. Override with GRB_ACTIVE=GRB140606B,GRB131011A
_ACTIVE = os.environ.get('GRB_ACTIVE', 'GRB150514A')
ACTIVE_GRBS = [s.strip() for s in _ACTIVE.split(',') if s.strip()]
grbs_df = grbs_df_all[grbs_df_all['name'].isin(ACTIVE_GRBS)].reset_index(drop=True)
if grbs_df.empty:
    raise ValueError(f'GRB_ACTIVE={_ACTIVE!r} matched no catalog rows')

sampler_df = pd.DataFrame([
    {
        'sampler': 'dynesty',
        'description': 'Nested sampling with dynamic live points.',
        'nlive': 2000,
        'sampler_config': {
            'sample': 'rwalk',
            'naccept': 60,
            'dlogz': 0.0005,
            'maxiter': 100000,
            'walks': 60,
        },
    },
    {
        'sampler': 'emcee',
        'description': 'Ensemble MCMC sampler.',
        'nlive': 1000,
        'sampler_config': {
            'nsteps': 5000,
            'nburn': 1250,
        },
    },
    {
        'sampler': 'multinest',
        'description': 'MultiNest nested sampler.',
        'nlive': 1000,
        'sampler_config': {
            'importance_nested': False,
            'sampling_efficiency': 0.8,
        },
    },
    {
        'sampler': 'pymultinest',
        'description': 'PyMultiNest wrapper for MultiNest.',
        'nlive': 1000,
        'sampler_config': {
            'importance_nested': False,
            'sampling_efficiency': 0.8,
        },
    },
])

sampler_df = sampler_df.sort_values('sampler').reset_index(drop=True)
