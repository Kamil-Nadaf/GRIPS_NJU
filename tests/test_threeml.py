"""3ML helpers that do not require threeML at import time."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.context import GRBContext
from pipeline.threeml import (
    MODEL_ALIAS,
    _3ml_fingerprint,
    _bkg_intervals,
    _canon_model,
    _lc_view_limits,
    _posterior_to_row,
    _resolve_3ml_dir,
    canon_bin_method,
    merge_short_bins,
    pick_brightest_nai,
    resolve_3ml_bins,
)


def _ctx(**kwargs):
    defaults = dict(
        name='GRB140606B', ra=328.12501, dec=32.01458,
        utc='2014-06-06T03:11:51.86',
        sel_dets=['n3', 'n4', 'n8'],
        t1=-0.83, t2=5.74,
        lc_pad_pre=20.0, lc_pad_post=50.0,
        spec_rebn={'min_sigma': 2, 'max_bin': 20},
        data_base='/tmp',
    )
    defaults.update(kwargs)
    return GRBContext(**defaults)


class TestAliases(unittest.TestCase):
    def test_cpl(self):
        tag, astro = _canon_model('cpl')
        self.assertEqual(tag, 'cpl')
        self.assertEqual(astro, 'Cutoff_powerlaw')
        self.assertEqual(MODEL_ALIAS['band'], 'Band')

    def test_cpl_ep_from_xc(self):
        # K, index, xc, logL  —  Ep = xc*(2+alpha)
        samples = np.array([
            [1.0, -1.0, 200.0, 10.0],
            [1.1, -1.0, 210.0, 9.0],
            [0.9, -1.0, 190.0, 8.0],
        ])
        row = _posterior_to_row(samples, 'cpl', t_start=-0.8, t_stop=5.7)
        self.assertAlmostEqual(row['Ep_best'], 200.0, places=4)
        self.assertEqual(row['backend'], '3ML')
        self.assertAlmostEqual(row['t_start'], -0.8)
        self.assertEqual(row['ep_constrained'], 1)


class TestHeapyBackground(unittest.TestCase):
    def test_lc_pads_minus_burst(self):
        ctx = _ctx()
        bkg = _bkg_intervals(ctx)
        self.assertEqual(bkg, ['-20.000--0.830', '5.740-55.740'])
        t_lo, t_hi = _lc_view_limits(ctx)
        self.assertAlmostEqual(t_lo, -20.0)
        self.assertAlmostEqual(t_hi, 55.74)

    def test_override(self):
        ctx = _ctx()
        bkg = _bkg_intervals(ctx, '-40--10,50-100')
        self.assertEqual(bkg, ['-40--10', '50-100'])

    def test_fingerprint_includes_bkg(self):
        ctx = _ctx()
        a = _3ml_fingerprint(ctx, 'cpl', 'tintegrated', 1000, ['n3', 'n4', 'n8'])
        b = _3ml_fingerprint(
            ctx, 'cpl', 'tintegrated', 1000, ['n3', 'n4', 'n8'],
            background_interval='-40--10,50-100')
        self.assertNotEqual(a, b)
        c = _3ml_fingerprint(ctx, 'cpl', 'tintegrated', 400, ['n3', 'n4', 'n8'])
        self.assertNotEqual(a, c)


class TestPosteriorQuantiles(unittest.TestCase):
    def test_median_not_max_likelihood(self):
        # Max-L sits on the prior edge; equal-weight median stays near 200 keV.
        rng = np.random.default_rng(0)
        xc = np.concatenate([
            rng.normal(200.0, 8.0, 80),
            np.full(5, 1e4),
        ])
        alpha = np.full_like(xc, -1.0)
        A = np.ones_like(xc)
        logl = np.concatenate([np.linspace(10, 1, 80), np.full(5, 50.0)])
        samples = np.column_stack([A, alpha, xc, logl])
        row = _posterior_to_row(samples, 'cpl')
        self.assertLess(row['Ep_best'], 400.0)
        self.assertGreater(row['Ep_ml'], 5000.0)
        self.assertEqual(row['ep_constrained'], 0)
        self.assertEqual(row['log_Ep_hits_prior_high'], 1)


class TestResolveDir(unittest.TestCase):
    def test_old_meta_key_versions_on_mismatch(self):
        import json
        import tempfile
        ctx = _ctx()
        fp_new = _3ml_fingerprint(ctx, 'cpl', 'tintegrated', 1000, ['n3', 'n4', 'n8'])
        with tempfile.TemporaryDirectory() as td:
            model_root = os.path.join(td, 'cpl')
            os.makedirs(model_root)
            with open(os.path.join(model_root, 'pipeline_meta.json'), 'w') as f:
                json.dump({'3ml_tintegrated_cpl': 'oldold01', 'fp': 'oldold01'}, f)
            out = _resolve_3ml_dir(model_root, fp_new, 'cpl', 'tintegrated')
            self.assertEqual(out, os.path.join(model_root, 'versions', fp_new))
            forced = _resolve_3ml_dir(
                model_root, fp_new, 'cpl', 'tintegrated', force=True)
            self.assertEqual(forced, model_root)


class TestTresBins(unittest.TestCase):
    def test_pick_brightest_nai_from_ranking(self):
        dets = ['n8', 'n3', 'n4']
        ranked = [('n4', 12.0), ('n3', 20.0), ('n8', 45.0)]
        self.assertEqual(pick_brightest_nai(dets, ranked=ranked), 'n4')

    def test_pick_brightest_nai_fallback(self):
        self.assertEqual(pick_brightest_nai(['b0', 'n8', 'n3']), 'n8')
        self.assertEqual(pick_brightest_nai(['b0']), 'b0')
        self.assertIsNone(pick_brightest_nai([]))

    def test_merge_short_middle(self):
        slices = [(-0.83, 1.0), (1.0, 1.05), (1.05, 5.74)]
        merged = merge_short_bins(slices, min_width=0.1)
        self.assertEqual(merged, [(-0.83, 1.0), (1.0, 5.74)])

    def test_merge_short_last(self):
        slices = [(-0.83, 5.0), (5.0, 5.05)]
        merged = merge_short_bins(slices, min_width=0.1)
        self.assertEqual(merged, [(-0.83, 5.05)])

    def test_merge_low_sigma(self):
        slices = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
        merged = merge_short_bins(
            slices, min_width=0.0, sigmas=[30.0, 5.0, 40.0], min_sigma=10.0)
        self.assertEqual(merged, [(0.0, 1.0), (1.0, 3.0)])

    def test_canon_bin_method(self):
        self.assertEqual(canon_bin_method('manual'), 'custom')
        self.assertEqual(canon_bin_method('bayesblocks'), 'bayesblocks')
        with self.assertRaises(ValueError):
            canon_bin_method('bb')

    def test_custom_passthrough(self):
        ctx = _ctx(slice_boundaries=[-0.83, 1.345, 5.74])
        dets = ['n3', 'n4', 'n8']
        slices, payload = resolve_3ml_bins(
            ctx, dets, bin_method='custom', bin_kwargs={'min_width': 0.0})
        self.assertEqual(payload['method'], 'custom')
        self.assertIsNone(payload['ref_det'])
        self.assertEqual(len(slices), 2)
        self.assertAlmostEqual(slices[0][0], -0.83)
        self.assertAlmostEqual(slices[-1][1], 5.74)

    def test_fingerprint_includes_bin_method(self):
        ctx = _ctx()
        a = _3ml_fingerprint(
            ctx, 'cpl', 'tresolved', 1000, ['n3', 'n4'],
            bin_method='custom', slices=[(-0.83, 5.74)])
        b = _3ml_fingerprint(
            ctx, 'cpl', 'tresolved', 1000, ['n3', 'n4'],
            bin_method='bayesblocks', slices=[(-0.83, 5.74)])
        self.assertNotEqual(a, b)
        c = _3ml_fingerprint(
            ctx, 'cpl', 'tresolved', 1000, ['n3', 'n4'],
            bin_method='custom', slices=[(-0.83, 1.0), (1.0, 5.74)])
        self.assertNotEqual(a, c)


if __name__ == '__main__':
    unittest.main()
