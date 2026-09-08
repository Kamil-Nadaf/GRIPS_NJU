"""Slice boundary helpers — no heapy required."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.slices import (
    boundaries_to_slices,
    burst_span,
    slices_from_bb_lc,
    slices_to_boundaries,
)
from pipeline.lc_io import write_lc_json


class TestBoundaries(unittest.TestCase):
    def test_table_c1_140606b(self):
        bounds = [-0.83, 1.345, 1.795, 2.34, 3.34, 4.43, 5.74]
        slices = boundaries_to_slices(bounds)
        self.assertEqual(len(slices), 6)
        self.assertEqual(slices[0], (-0.83, 1.345))
        self.assertEqual(slices[-1], (4.43, 5.74))
        self.assertEqual(slices_to_boundaries(slices), bounds)

    def test_rejects_unsorted(self):
        with self.assertRaises(ValueError):
            boundaries_to_slices([0.0, 1.0, 0.5])

    def test_burst_span_gcn_wider_than_slices(self):
        t0, t1 = burst_span(
            t1=-3.0, t2=12.3,
            time_slices=[(-0.83, 1.345), (4.43, 5.74)],
            slice_boundaries=[-0.83, 1.345, 5.74])
        self.assertEqual((t0, t1), (-3.0, 12.3))

    def test_table_c1_catalog(self):
        from grb_config import GCN_REFS, gcn_ref, grbs_df_all
        row = grbs_df_all[grbs_df_all['name'] == 'GRB140606B'].iloc[0]
        self.assertEqual(list(row['sel_dets']), ['b0', 'n3', 'n4', 'n8'])
        self.assertEqual(list(row['slice_boundaries'])[0], -0.83)
        self.assertEqual(list(row['slice_boundaries'])[-1], 5.74)
        self.assertEqual(row['slice_mode'], 'manual')
        self.assertEqual(len(row['time_slices']), 6)
        # Tint uses GCN 16363; Table C1 slices stay for later tres
        self.assertEqual(float(row['t1']), -3.0)
        self.assertEqual(float(row['t2']), 12.3)
        self.assertEqual(float(row['tint_t1']), -3.0)
        self.assertEqual(float(row['tint_t2']), 12.3)
        ref = gcn_ref('GRB140606B')
        self.assertEqual(ref['source'], 'GCN 16363')
        self.assertEqual(ref['alpha'], GCN_REFS['GRB140606B']['alpha'])
        self.assertEqual(ref['Ep'], 473.0)
        other = grbs_df_all[grbs_df_all['name'] == 'GRB131011A'].iloc[0]
        self.assertEqual(float(other['t1']), -2.99)
        self.assertEqual(float(other['t2']), 25.01)
        self.assertIsNone(gcn_ref('GRB131011A'))
        r190 = grbs_df_all[grbs_df_all['name'] == 'GRB190829A'].iloc[0]
        self.assertEqual(r190['utc'], '2019-08-29T19:55:53.13')
        self.assertEqual(float(r190['t1']), 0.0)
        self.assertEqual(float(r190['t2']), 4.0)
        self.assertEqual(list(r190['slice_boundaries'])[0], -0.81)
        self.assertEqual(list(r190['slice_boundaries'])[-1], 5.07)
        ref190 = gcn_ref('GRB190829A')
        self.assertEqual(ref190['source'], 'GCN 25575')
        self.assertEqual(ref190['Ep'], 130.0)
        r150 = grbs_df_all[grbs_df_all['name'] == 'GRB150514A'].iloc[0]
        self.assertEqual(r150['utc'], '2015-05-14T18:35:05.35')
        self.assertEqual(float(r150['t1']), 0.0)
        self.assertEqual(float(r150['t2']), 11.3)
        self.assertEqual(list(r150['slice_boundaries'])[0], -0.46)
        self.assertEqual(list(r150['slice_boundaries'])[-1], 6.05)
        ref150 = gcn_ref('GRB150514A')
        self.assertEqual(ref150['source'], 'GCN 17819')
        self.assertEqual(ref150['Ep'], 73.0)
        self.assertEqual(ref150['model'], 'band')

    def test_context_uses_gcn_tint(self):
        from pipeline.context import GRBContext
        ctx = GRBContext.from_name('GRB140606B', data_base='/tmp')
        self.assertEqual(ctx.t1, -3.0)
        self.assertEqual(ctx.t2, 12.3)
        self.assertEqual(ctx.burst_span(), (-3.0, 12.3))
        self.assertEqual(ctx.slice_boundaries[0], -0.83)
        self.assertEqual(ctx.slice_boundaries[-1], 5.74)

    def test_context_190829a_gcn_tint(self):
        from pipeline.context import GRBContext
        ctx = GRBContext.from_name('GRB190829A', data_base='/tmp')
        self.assertEqual(ctx.utc, '2019-08-29T19:55:53.13')
        self.assertEqual(ctx.t1, 0.0)
        self.assertEqual(ctx.t2, 4.0)
        # Union of GCN tint and Table C1 (C1 is wider here)
        self.assertEqual(ctx.burst_span(), (-0.81, 5.07))
        self.assertEqual(ctx.slice_boundaries[0], -0.81)
        self.assertEqual(ctx.slice_boundaries[-1], 5.07)

    def test_context_150514a_gcn_tint(self):
        from pipeline.context import GRBContext
        ctx = GRBContext.from_name('GRB150514A', data_base='/tmp')
        self.assertEqual(ctx.utc, '2015-05-14T18:35:05.35')
        self.assertEqual(ctx.t1, 0.0)
        self.assertEqual(ctx.t2, 11.3)
        # GCN tint wider than Table C1 on the right
        self.assertEqual(ctx.burst_span(), (-0.46, 11.3))
        self.assertEqual(ctx.slice_boundaries[0], -0.46)
        self.assertEqual(ctx.slice_boundaries[-1], 6.05)



class TestBbSlices(unittest.TestCase):
    def test_clip_to_tint(self):
        payload = {
            'method': 'heapy.pgSignal.bblock',
            'p0': 0.05,
            'time_offset': 298.11,
            'edges': [278.11, 298.11, 299.11, 300.11, 304.11, 348.11],
            'edges_rel': [-20.0, 0.0, 1.0, 2.0, 6.0, 50.0],
            'traces': [],
        }
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'bb_lc.json')
            write_lc_json(path, payload)
            slices, bounds = slices_from_bb_lc(path, t1=-0.83, t2=5.74)
        self.assertEqual(bounds[0], -0.83)
        self.assertEqual(bounds[-1], 5.74)
        self.assertTrue(all(a < b for a, b in slices))
        self.assertTrue(all(-0.83 <= a and b <= 5.74 for a, b in slices))


if __name__ == '__main__':
    unittest.main()
