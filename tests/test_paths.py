"""Path fingerprints and versioned dirs — no heapy required."""

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.paths import (
    PathLayout,
    commit_meta,
    extraction_fingerprint,
    fingerprint,
    resolve_fit_dir,
    resolve_versioned_dir,
    spec_slice_name,
)
from pipeline.slices import boundaries_to_slices


def _ctx(**kwargs):
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


class TestFingerprint(unittest.TestCase):
    def test_stable(self):
        self.assertEqual(fingerprint({'a': 1}), fingerprint({'a': 1}))
        self.assertNotEqual(fingerprint({'a': 1}), fingerprint({'a': 2}))

    def test_extraction_changes_with_window(self):
        a = extraction_fingerprint(_ctx())
        b = extraction_fingerprint(_ctx(t2=25.01, slice_boundaries=[-2.99, 25.01]))
        self.assertNotEqual(a, b)

    def test_extraction_changes_with_utc(self):
        a = extraction_fingerprint(_ctx(utc='2019-08-29T19:55:53.13'))
        b = extraction_fingerprint(_ctx(utc='2019-08-29T19:56:44.60'))
        self.assertNotEqual(a, b)

    def test_spec_slice_name(self):
        self.assertIn('m', spec_slice_name(-0.83, 1.345))
        self.assertTrue(spec_slice_name(1.345, 1.795).startswith('p'))


class TestVersionedDir(unittest.TestCase):
    def test_first_run_canonical(self):
        with tempfile.TemporaryDirectory() as td:
            root = os.path.join(td, 'heapy')
            os.makedirs(root)
            d = resolve_versioned_dir(root, 'aaaa1111', 'canonical_extraction')
            self.assertEqual(d, root)
            commit_meta(root, 'canonical_extraction', 'aaaa1111', {'t1': -1})
            d2 = resolve_versioned_dir(root, 'bbbb2222', 'canonical_extraction')
            self.assertEqual(d2, os.path.join(root, 'versions', 'bbbb2222'))

    def test_fit_dir_sees_parent_meta(self):
        with tempfile.TemporaryDirectory() as td:
            bayspec = os.path.join(td, 'bayspec')
            model_root = os.path.join(bayspec, 'cpl')
            os.makedirs(model_root)
            commit_meta(bayspec, 'canonical_fit_cpl', 'oldold01', {'nlive': 1000})
            # mismatch must not return the canonical dir (that would skip stale weights)
            d = resolve_fit_dir(model_root, 'newnew02', 'canonical_fit_cpl')
            self.assertEqual(d, os.path.join(model_root, 'versions', 'newnew02'))
            forced = resolve_fit_dir(
                model_root, 'newnew02', 'canonical_fit_cpl', force=True)
            self.assertEqual(forced, model_root)

    def test_path_layout(self):
        p = PathLayout('GRB140606B', data_base='/workspace/data')
        self.assertTrue(p.heapy_tintegrated_path.endswith('tintegrated/heapy'))
        self.assertTrue(p.threeml_tintegrated_path.endswith('tintegrated/3ML'))
        self.assertTrue(p.threeml_tresolved_path.endswith('tresolved/3ML'))
        self.assertIn('GRB140606B_3ML_data.h5', p.threeml_data)
        self.assertIn('GRB140606B_bayspec_data.h5', p.bayspec_data)

    def test_fit_fingerprint_includes_fit_dets(self):
        from pipeline.paths import fit_fingerprint
        a = fit_fingerprint(_ctx(), 'cpl', nlive=1000, fit_dets=['n3', 'n4'])
        b = fit_fingerprint(_ctx(), 'cpl', nlive=1000, fit_dets=['n3', 'n4', 'n8'])
        self.assertNotEqual(a, b)


class TestLcWindow(unittest.TestCase):
    def test_covers_long_burst(self):
        from pipeline.context import GRBContext
        ctx = GRBContext(
            name='GRB131011A', ra=32.5, dec=-4.4,
            utc='2013-10-11T17:47:34.99',
            sel_dets=['b1', 'n9'],
            slice_boundaries=[-2.99, 1.74, 25.01],
            t1=-2.99, t2=25.01,
        )
        offset = 100.0
        lo, hi = ctx.lc_window(offset)
        # Must extend past t2=25 s, not clip at offset+50
        self.assertLessEqual(lo, offset - 20)
        self.assertGreaterEqual(hi, offset + 25.01 + 50)
        self.assertGreater(hi, offset + 50)

    def test_spec_window_clips_to_tte_span(self):
        from pipeline.context import GRBContext
        ctx = GRBContext(
            name='GRB140606B', ra=328.1, dec=32.0,
            utc='2014-06-06T03:11:51.86',
            sel_dets=['b0', 'n3'],
            t1=-3.0, t2=12.3,
        )
        offset = 298.11
        raw = ctx.spec_window(offset)
        self.assertLess(raw[0], offset - 100)
        clipped = ctx.spec_window(offset, t_lo=offset - 2.3, t_hi=offset + 200.0)
        self.assertAlmostEqual(clipped[0], offset - 2.3, places=5)
        self.assertLess(clipped[1], offset + 380.0)

    def test_spec_window_does_not_widen_past_lc(self):
        from pipeline.context import GRBContext
        from pipeline.lightcurve import _clip_window
        ctx = GRBContext(
            name='GRB140606B', ra=328.1, dec=32.0,
            utc='2014-06-06T03:11:51.86',
            t1=-3.0, t2=12.3,
        )
        offset = 298.11
        win = _clip_window(ctx.spec_window(offset), *ctx.lc_window(offset))
        self.assertEqual(win, list(ctx.lc_window(offset)))
        self.assertGreater(win[0], 0)


if __name__ == '__main__':
    unittest.main()
