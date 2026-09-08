"""Lightcurve plot styling — no heapy required."""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from plot_lightcurves import _error_band, _plot_trace, _trace_role, tint_xlim


class TestTintXlim(unittest.TestCase):
    def test_frames_short_pulse(self):
        lo, hi = tint_xlim(0.0, 4.0)
        self.assertLess(lo, 0.0)
        self.assertGreater(hi, 4.0)
        self.assertLess(hi, 40.0)  # do not include GRB190829A's second pulse

    def test_visible_ylim_ignores_later_pulse(self):
        from plot_lightcurves import visible_ylim
        traces = [{
            'x': np.array([-2.0, 2.0, 50.0]),
            'y': np.array([100.0, 200.0, 4000.0]),
        }]
        lo, hi = visible_ylim(traces, -6.0, 12.0)
        self.assertLess(hi, 500.0)
        self.assertGreater(lo, 0.0)


class TestTraceRole(unittest.TestCase):
    def test_names(self):
        self.assertEqual(_trace_role('source lightcurve'), 'source')
        self.assertEqual(_trace_role('background lightcurve'), 'background')
        self.assertEqual(_trace_role('net lightcurve'), 'net')


class TestErrorBand(unittest.TestCase):
    def test_band_not_errorbar(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        x = np.array([0.5, 1.5, 2.5])
        y = np.array([10.0, 12.0, 9.0])
        err = np.array([1.0, 1.0, 1.0])
        edges = np.array([0.0, 1.0, 2.0, 3.0])
        _error_band(ax, x, y, err, edges, '#111111')
        self.assertEqual(len(ax.collections), 1)
        self.assertEqual(len(ax.lines), 0)
        plt.close(fig)

    def test_plot_trace_band_has_no_errorbar_caps(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        tr = {
            'name': 'source lightcurve',
            'x': np.array([0.5, 1.5]),
            'y': np.array([3.0, 5.0]),
            'error_y': np.array([0.4, 0.5]),
            'edges': np.array([0.0, 1.0, 2.0]),
        }
        style = {
            'color': '#111', 'label': 'Source', 'alpha': 1.0, 'lw': 1.0, 'ls': '-',
        }
        _plot_trace(ax, tr, style, show_errors='band')
        self.assertFalse(
            any(type(c).__name__ == 'ErrorbarContainer' for c in ax.containers))
        plt.close(fig)


class TestPlotLightcurvesFile(unittest.TestCase):
    def test_saves_png(self):
        import matplotlib
        matplotlib.use('Agg')
        from pipeline.lc_io import write_lc_json
        from plot_lightcurves import plot_lightcurves

        t = np.array([0.25, 0.75, 1.25, 1.75])
        y = np.array([10.0, 40.0, 25.0, 12.0])
        payload = {
            'method': 'fixed',
            'edges': [0.0, 0.5, 1.0, 1.5, 2.0],
            'traces': [
                {'name': 'source lightcurve', 'x': t.tolist(), 'y': y.tolist(),
                 'error_y': [1, 2, 1.5, 1]},
                {'name': 'background lightcurve', 'x': t.tolist(),
                 'y': [8, 8, 8, 8], 'error_y': [0.5, 0.5, 0.5, 0.5]},
            ],
        }
        with tempfile.TemporaryDirectory() as td:
            det_dir = os.path.join(td, 'GRBX', 'data', 'tintegrated', 'heapy', 'n6')
            os.makedirs(det_dir)
            write_lc_json(os.path.join(det_dir, 'rebin_lc.json'), payload)
            # plot_lightcurves default is fixed (lc.html); use rebin for this fixture
            path = os.path.join(td, 'lc.png')
            fig, axes = plot_lightcurves(
                'GRBX', data_base=td, lc_kind='rebin', overlay_bb=False,
                t1=0.0, t2=2.0, save_path=path, show=False)
            self.assertIsNotNone(fig)
            self.assertTrue(os.path.isfile(path))
            self.assertGreater(os.path.getsize(path), 1000)
            lo, hi = tint_xlim(0.0, 2.0)
            xlim = axes[0].get_xlim()
            self.assertAlmostEqual(xlim[0], lo, places=2)
            self.assertAlmostEqual(xlim[1], hi, places=2)
            import matplotlib.pyplot as plt
            plt.close(fig)


if __name__ == '__main__':
    unittest.main()
