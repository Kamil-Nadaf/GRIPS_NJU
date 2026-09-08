"""LC JSON round-trip — no heapy required."""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.lc_io import (
    bb_payload_from_arrays,
    combine_fixed_bin_traces,
    traces_from_lc_json,
    write_lc_json,
    load_lc_json,
    load_heapy_fixed_lc,
)


class TestLcJson(unittest.TestCase):
    def test_bb_roundtrip_trigger_relative(self):
        edges = np.array([278.11, 298.11, 308.11])
        re_binsize = np.array([20.0, 10.0])
        re_cts = np.array([100.0, 200.0])
        re_bcts = np.array([80.0, 90.0])
        payload = bb_payload_from_arrays(
            edges, re_binsize, re_cts, re_bcts, p0=0.05, time_offset=298.11)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'bb_lc.json')
            write_lc_json(path, payload)
            loaded = load_lc_json(path)
        self.assertAlmostEqual(loaded['time_offset'], 298.11)
        self.assertAlmostEqual(loaded['edges_rel'][1], 0.0, places=5)
        traces = traces_from_lc_json(loaded)
        src = traces[0]
        self.assertAlmostEqual(src['x'][1], 5.0, places=5)

    def test_combine_dets(self):
        t = np.array([0.0, 0.5, 1.0])
        traces_a = [
            {'name': 'source lightcurve', 'x': t, 'y': np.array([1.0, 2.0, 3.0])},
            {'name': 'background lightcurve', 'x': t, 'y': np.array([0.1, 0.1, 0.1])},
        ]
        traces_b = [
            {'name': 'source lightcurve', 'x': t, 'y': np.array([1.0, 1.0, 1.0])},
            {'name': 'background lightcurve', 'x': t, 'y': np.array([0.2, 0.2, 0.2])},
        ]
        payload = combine_fixed_bin_traces({'n3': traces_a, 'n4': traces_b})
        src = payload['traces'][0]
        self.assertEqual(src['y'], [2.0, 3.0, 4.0])

    def test_fixed_json_is_trigger_relative(self):
        offset = 2598.91
        payload = {
            'time_offset': offset,
            'edges': [offset - 1.0, offset, offset + 1.0],
            'traces': [{
                'name': 'source lightcurve',
                'x': [offset - 0.5, offset + 0.5],
                'y': [10.0, 20.0],
                'error_y': [1.0, 1.0],
            }],
        }
        with tempfile.TemporaryDirectory() as td:
            write_lc_json(os.path.join(td, 'lc_fixed.json'), payload)
            traces = load_heapy_fixed_lc(td)
        self.assertAlmostEqual(traces[0]['x'][0], -0.5, places=5)
        self.assertAlmostEqual(traces[0]['x'][1], 0.5, places=5)


if __name__ == '__main__':
    unittest.main()
