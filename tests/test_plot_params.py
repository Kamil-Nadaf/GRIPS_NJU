"""Tres parameter evolution plots — no heapy/threeML required."""

import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from plotting.params import plot_tres_params, save_tres_params_plot


def _df():
    return pd.DataFrame([
        {'slice': 1, 't_start': -0.83, 't_stop': 1.0,
         'alpha': -1.1, 'alpha_low': 0.1, 'alpha_high': 0.1,
         'Ep_best': 400.0, 'Ep_low': 50.0, 'Ep_high': 80.0,
         'ep_constrained': 1},
        {'slice': 2, 't_start': 1.0, 't_stop': 3.0,
         'alpha': -1.3, 'alpha_low': 0.15, 'alpha_high': 0.12,
         'Ep_best': 300.0, 'Ep_low': 40.0, 'Ep_high': 60.0,
         'ep_constrained': 1},
        {'slice': 3, 't_start': 3.0, 't_stop': 5.74,
         'alpha': -1.5, 'alpha_low': 0.2, 'alpha_high': 0.2,
         'Ep_best': 9000.0, 'Ep_low': 1000.0, 'Ep_high': 500.0,
         'ep_constrained': 0},
    ])


class TestTresParamPlot(unittest.TestCase):
    def test_plot_returns_fig(self):
        fig, axes = plot_tres_params(_df(), title='test')
        self.assertIsNotNone(fig)
        self.assertEqual(len(axes), 2)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_empty(self):
        fig, axes = plot_tres_params(pd.DataFrame())
        self.assertIsNone(fig)

    def test_save(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'tres_params.png')
            out = save_tres_params_plot(_df(), path, title='save')
            self.assertEqual(out, path)
            self.assertTrue(os.path.isfile(path))
            self.assertGreater(os.path.getsize(path), 1000)


if __name__ == '__main__':
    unittest.main()
