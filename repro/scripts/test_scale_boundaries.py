import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scale_boundaries import affine_scale_boundaries, boundary_summary


class ScaleBoundaryTests(unittest.TestCase):
    def test_multiclass_shrink_first_differs_from_bias_exit(self):
        out = affine_scale_boundaries([1, 0, -1], [[0, 10, 12]])
        self.assertAlmostEqual(out['bias_exit'][0], 0.1)
        self.assertAlmostEqual(out['shrink_first'][0], 0.5)
        self.assertEqual(int(out['healthy_class'][0]), 2)

    def test_no_crossing_and_conditional_median(self):
        out = affine_scale_boundaries([2, 0], [[1, 0], [0, 8]])
        self.assertTrue(np.isinf(out['bias_exit'][0]))
        self.assertEqual(out['shrink_first'][0], 0)
        summary = boundary_summary([0.1, 0.2, np.inf])
        self.assertAlmostEqual(summary['finite_median'], 0.15)
        self.assertAlmostEqual(summary['all_median'], 0.2)

    def test_dense_grid_matches_first_departure(self):
        rng = np.random.default_rng(9)
        b = rng.normal(size=5)
        s = rng.normal(size=(40, 5)) * 3
        out = affine_scale_boundaries(b, s)
        for i, threshold in enumerate(out['shrink_first']):
            if threshold > 0:
                just_above = min(1, threshold + 1e-7)
                just_below = max(0, threshold - 1e-7)
                self.assertEqual(int((b + just_above * s[i]).argmax()),
                                 int(out['healthy_class'][i]))
                self.assertNotEqual(int((b + just_below * s[i]).argmax()),
                                    int(out['healthy_class'][i]))

    def test_refitted_statistics_cancel_positive_scale(self):
        rng = np.random.default_rng(7)
        train = rng.normal(size=(100, 8)) + 3
        test = rng.normal(size=(20, 8)) + 3
        expected = (test - train.mean(0)) / train.std(0)
        for c in [1e-3, 0.05, 2.0]:
            actual = (c * test - (c * train).mean(0)) / (c * train).std(0)
            np.testing.assert_allclose(actual, expected, atol=1e-11)

    def test_ties_require_explicit_handling(self):
        with self.assertRaises(ValueError):
            affine_scale_boundaries([0, 0], [[1, 2]])


if __name__ == '__main__':
    unittest.main()
