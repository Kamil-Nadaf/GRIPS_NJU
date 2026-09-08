"""NaI-default fits, Ep flags, generic posterior dump — no heapy/bayspec."""

import json
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.detectors import fit_detectors, nai_dets
from pipeline.fitting import (
    apply_equal_weight_quantiles,
    ep_constraint_flags,
    get_model_params,
    latex_to_col,
    parse_unif_prior,
    posterior_columns,
    quantile_pm,
)
from pipeline.paths import fit_fingerprint
from pipeline.slices import boundaries_to_slices


def _ctx(**kwargs):
    from types import SimpleNamespace
    bounds = kwargs.pop('slice_boundaries', [-0.83, 1.345, 5.74])
    slices = boundaries_to_slices(bounds)
    defaults = dict(
        t1=bounds[0], t2=bounds[-1], time_slices=slices,
        sel_dets=['b0', 'n3', 'n4', 'n8'],
        retrieve_t1=-400, retrieve_t2=400,
        slice_mode='manual',
        spec_rebn={'min_sigma': 2, 'max_bin': 20},
        lc_pad_pre=20.0, resolved_lc_pad_post=50.0,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class TestFitDetectors(unittest.TestCase):
    def test_nai_default(self):
        dets = ['b0', 'n3', 'n4', 'n8']
        self.assertEqual(nai_dets(dets), ['n3', 'n4', 'n8'])
        self.assertEqual(fit_detectors(dets, include_bgo=False), ['n3', 'n4', 'n8'])
        self.assertEqual(fit_detectors(dets, include_bgo=True), dets)

    def test_skip_dets(self):
        dets = ['b0', 'n3', 'n4', 'n8']
        self.assertEqual(
            fit_detectors(dets, include_bgo=False, skip_dets=['n8']),
            ['n3', 'n4'])


class TestEpFlags(unittest.TestCase):
    def test_constrained(self):
        flags = ep_constraint_flags([2.1, 2.3, 2.5], prior=(0.0, 4.0))
        self.assertTrue(flags['ep_constrained'])
        self.assertFalse(flags['log_Ep_hits_prior_high'])

    def test_hits_high_edge(self):
        flags = ep_constraint_flags([3.7, 3.9, 3.98], prior=(0.0, 4.0))
        self.assertFalse(flags['ep_constrained'])
        self.assertTrue(flags['log_Ep_hits_prior_high'])

    def test_parse_unif(self):
        self.assertEqual(parse_unif_prior('unif(0, 4)'), (0.0, 4.0))
        self.assertIsNone(parse_unif_prior('frozen'))


class TestGenericPosterior(unittest.TestCase):
    def test_latex_map(self):
        self.assertEqual(latex_to_col('$\\alpha$'), 'alpha')
        self.assertEqual(latex_to_col('log$E_p$'), 'log_Ep')

    def test_posterior_columns_from_json(self):
        with tempfile.TemporaryDirectory() as td:
            payload = [
                {'Parameter': '$\\alpha$'},
                {'Parameter': 'log$E_p$'},
                {'Parameter': 'log$A$'},
            ]
            with open(os.path.join(td, 'post_free_par.json'), 'w') as f:
                json.dump(payload, f)
            cols = posterior_columns(td, 'mystery', {})
            self.assertEqual(cols, ['alpha', 'log_Ep', 'log_A', 'log_likelihood'])

    def test_cpl_column_map(self):
        self.assertEqual(
            posterior_columns('/nope', 'cpl', {}),
            ['alpha', 'log_Ep', 'log_A', 'log_likelihood'])

    def test_get_model_params_flags(self):
        rng = np.random.default_rng(0)
        log_ep = rng.uniform(2.0, 2.4, size=100)
        df = pd.DataFrame({
            'alpha': rng.uniform(-1.3, -1.1, size=100),
            'log_Ep': log_ep,
            'Ep': 10 ** log_ep,
            'A': np.ones(100),
            'vFv': np.ones(100),
        })
        row = get_model_params('cpl', df, prior=(0.0, 4.0))
        self.assertIn('Ep_best', row)
        self.assertTrue(row['ep_constrained'])

    def test_generic_model_dump(self):
        df = pd.DataFrame({
            'kT': [1.0, 0.9, 1.1],
            'log_A': [0.0, -0.1, 0.1],
        })
        row = get_model_params('mbb', df)
        self.assertIn('kT', row)
        self.assertIn('kT_low', row)
        self.assertNotIn('ep_constrained', row)


class TestEqualWeightQuantiles(unittest.TestCase):
    def test_quantile_pm_symmetric(self):
        med, low, high = quantile_pm(np.linspace(0.0, 100.0, 101))
        self.assertAlmostEqual(med, 50.0, places=5)
        self.assertAlmostEqual(low, 34.0, places=0)
        self.assertAlmostEqual(high, 34.0, places=0)

    def test_apply_keeps_maxl_and_overwrites_median(self):
        rng = np.random.default_rng(1)
        n = 200
        alpha = rng.normal(-1.2, 0.05, size=n)
        ep = rng.normal(470.0, 40.0, size=n)
        logL = -np.abs(alpha + 1.2)
        alpha[0] = -0.5
        ep[0] = 900.0
        logL[0] = 10.0
        df = pd.DataFrame({
            'alpha': alpha,
            'Ep': ep,
            'log_Ep': np.log10(ep),
            'A': np.ones(n),
            'vFv': np.ones(n),
            'log_likelihood': logL,
        }).sort_values('log_likelihood', ascending=False).reset_index(drop=True)
        n_1s = max(int(0.6827 * len(df)), 1)
        row = get_model_params('cpl', df.iloc[:n_1s].copy(), prior=(0.0, 4.0))
        self.assertAlmostEqual(row['alpha'], -0.5, places=5)
        self.assertAlmostEqual(row['Ep_best'], 900.0, places=4)
        apply_equal_weight_quantiles(row, df)
        self.assertAlmostEqual(row['alpha_ml'], -0.5, places=5)
        self.assertAlmostEqual(row['Ep_ml'], 900.0, places=4)
        self.assertAlmostEqual(row['alpha'], float(np.median(df['alpha'])), places=5)
        self.assertAlmostEqual(row['Ep_best'], float(np.median(df['Ep'])), places=4)
        self.assertLess(abs(row['alpha'] - (-1.2)), 0.05)
        self.assertLess(abs(row['Ep_best'] - 470.0), 20.0)


class TestFitFingerprint(unittest.TestCase):
    def test_nai_vs_bgo_differs(self):
        ctx = _ctx()
        a = fit_fingerprint(ctx, 'cpl', nlive=1000, fit_dets=['n3', 'n4', 'n8'])
        b = fit_fingerprint(ctx, 'cpl', nlive=1000, fit_dets=['b0', 'n3', 'n4', 'n8'])
        self.assertNotEqual(a, b)


if __name__ == '__main__':
    unittest.main()
