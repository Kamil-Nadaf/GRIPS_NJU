"""Shared constants for the GBM pipeline."""

import os

DATA_BASE = os.environ.get('DATA_BASE', '/workspace/data')

ALL_DETS = [
    'n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'na', 'nb',
    'b0', 'b1',
]

PARAM_KEY_MAP = {
    'alpha': '$\\alpha$',
    'beta': '$\\beta$',
    'Ep': 'log$E_p$',
    'log_Ep': 'log$E_p$',
    'A': 'log$A$',
    'log_A': 'log$A$',
}

# bayspec latex labels → posterior column names
LATEX_TO_COL = {
    '$\\alpha$': 'alpha',
    '$\\beta$': 'beta',
    'log$E_p$': 'log_Ep',
    'log$A$': 'log_A',
    'log$E_c$': 'log_Ec',
    'log$E_b$': 'log_Eb',
    '$kT$': 'kT',
}

MODEL_COLS = {
    'cpl': ['alpha', 'log_Ep', 'log_A'],
    'band': ['alpha', 'beta', 'log_Ep', 'log_A'],
    'bpl': ['alpha', 'beta', 'log_Ep', 'log_A'],
    'pl': ['alpha', 'log_A'],
    'sbpl': ['alpha', 'beta', 'log_Ep', 'log_A'],
    'cband': ['alpha', 'beta', 'log_Ep', 'log_A'],
    'dband': ['alpha', 'beta', 'log_Ep', 'log_A'],
    'hlecpl': ['alpha', 'log_Ep', 'log_A'],
    'hleband': ['alpha', 'beta', 'log_Ep', 'log_A'],
    'grbm': ['alpha', 'beta', 'log_Ep', 'log_A'],
    'cutoffpl': ['alpha', 'log_Ep', 'log_A'],
}

# bayspec default for CPL/Band log Ep
LOG_EP_PRIOR = (0.0, 4.0)
LOG_EP_EDGE_TOL = 0.05

# Default runner path: tint only. Tres is opt-in via spectra_tres / fit_tres.
DEFAULT_STAGES = ('geometry', 'spectra_tint', 'fit_tint', 'params')

ADDITIVE_SKIP = {
    'Cfg', 'Par', 'OrderedDict', 'unif', 'abspath', 'dirname',
    'cached_property', 'Additive', 'zxhsync', 'katu',
}

# LC SNR-rebin (display) vs spectral fit rebin (PGSTAT)
DEFAULT_LC_REBN = {'min_sigma': 1, 'max_bin': 8}
DEFAULT_SPEC_REBN = {'min_sigma': 2, 'max_bin': 20}

DEFAULT_LC_PAD_PRE = 20.0
DEFAULT_LC_PAD_POST_FLOOR = 50.0
DEFAULT_SPEC_FILTER_PAD = 380.0
DEFAULT_LC_BINSIZE = 0.5
DEFAULT_BS_P0 = 0.05
DEFAULT_NAI_MAX_ANGLE = 60.0
DEFAULT_N_NAI = 2
DEFAULT_N_BGO = 1
DEFAULT_MAX_DETS = 4

NAI_ENERGY = [8, 900]
BGO_ENERGY = [300, 38000]
COMBINED_LC_BAND = [10, 1000]

LC_ARTIFACTS = (
    'lc.html', 'lc.json', 'lc_fixed.json', 'bb_lc.json', 'rebin_lc.json', 'rebin_lc.html',
    'cum_lc.html', 'txx.pdf', 'txx_res.json', 'pulse_res.json',
)
