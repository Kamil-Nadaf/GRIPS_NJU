"""Tint-first stage defaults — no heapy required."""

import logging
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.constants import DEFAULT_STAGES
from pipeline.mpl_setup import silence_missing_fonts
from pipeline.runner import STAGE_ALIASES, expand_stages


class TestStages(unittest.TestCase):
    def test_default_is_tint(self):
        self.assertEqual(expand_stages(None), list(DEFAULT_STAGES))
        self.assertEqual(
            list(DEFAULT_STAGES),
            ['geometry', 'spectra_tint', 'fit_tint', 'params'])
        self.assertNotIn('spectra_tres', expand_stages(None))
        self.assertNotIn('fit_tres', expand_stages(None))

    def test_all_is_tint_first(self):
        stages = expand_stages('all')
        self.assertEqual(stages, list(DEFAULT_STAGES))
        self.assertNotIn('spectra_tres', stages)
        self.assertNotIn('fit_tres', stages)

    def test_explicit_tres(self):
        stages = expand_stages('geometry,spectra,fit')
        self.assertIn('spectra_tres', stages)
        self.assertIn('fit_tres', stages)
        self.assertIn('spectra_tint', stages)
        self.assertIn('fit_tint', stages)

    def test_unknown_stage(self):
        with self.assertRaises(ValueError):
            expand_stages('not_a_stage')

    def test_aliases_cover_default(self):
        for s in DEFAULT_STAGES:
            self.assertIn(s, STAGE_ALIASES)


class TestFontSilence(unittest.TestCase):
    def test_findfont_logger_is_error(self):
        silence_missing_fonts()
        log = logging.getLogger('matplotlib.font_manager')
        self.assertGreaterEqual(log.level, logging.ERROR)
        self.assertFalse(log.propagate)


class TestMapParallel(unittest.TestCase):
    def test_sequential_one_worker(self):
        from pipeline.parallel import map_parallel
        out = map_parallel(lambda x: x['v'] * 2, [{'v': 1}, {'v': 2}], n_workers=1)
        self.assertEqual(out, [2, 4])

    def test_retries_none_results(self):
        from pipeline.parallel import map_parallel

        def fn(job):
            return job['v']

        out = map_parallel(fn, [{'v': 'a'}, {'v': 'b'}], n_workers=1)
        self.assertEqual(out, ['a', 'b'])


if __name__ == '__main__':
    unittest.main()
