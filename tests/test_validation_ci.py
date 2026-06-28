"""Deterministic CI/selection tests for cross-sectional validation."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from validate_cross_sectional import (
    ACTIVE_IR_EDGE_MARGIN,
    VERDICT_EDGE,
    VERDICT_FAILS,
    VERDICT_MATCHES,
    expected_max_ir_under_null,
    information_ratio_ci,
    validation_verdict,
)


def _series(values: np.ndarray) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range('2024-01-01', periods=len(values)))


def _target_ir_series(target_ir: float, n: int = 126, seed: int = 41) -> pd.Series:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 0.010, size=n)
    noise = noise - noise.mean()
    shifted = noise + target_ir * noise.std(ddof=1) / np.sqrt(252.0)
    return _series(shifted)


class TestValidationCI(unittest.TestCase):
    def test_bootstrap_ci_is_deterministic_for_same_seed(self):
        rng = np.random.default_rng(7)
        active = _series(rng.normal(0.0004, 0.006, size=252))

        first = information_ratio_ci(active, n_boot=400, seed=123)
        second = information_ratio_ci(active, n_boot=400, seed=123)

        self.assertEqual(first, second)

    def test_high_signal_series_resolves_to_edge(self):
        rng = np.random.default_rng(11)
        active = _series(rng.normal(0.0012, 0.00025, size=252))
        point_ir, ci_lower, ci_upper, se = information_ratio_ci(active, n_boot=400, seed=1)
        threshold = expected_max_ir_under_null(20, se)

        self.assertGreater(ci_lower, ACTIVE_IR_EDGE_MARGIN)
        self.assertEqual(
            validation_verdict(
                1.0,
                point_ir,
                folds=8,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                selection_threshold=threshold,
            ),
            VERDICT_EDGE,
        )

    def test_noise_series_ci_straddles_zero_and_matches(self):
        rng = np.random.default_rng(13)
        active = _series(rng.normal(0.0, 0.010, size=252))
        point_ir, ci_lower, ci_upper, se = information_ratio_ci(active, n_boot=400, seed=2)
        threshold = expected_max_ir_under_null(20, se)

        self.assertLess(ci_lower, 0.0)
        self.assertGreater(ci_upper, 0.0)
        self.assertEqual(
            validation_verdict(
                1.0,
                point_ir,
                folds=8,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                selection_threshold=threshold,
            ),
            VERDICT_MATCHES,
        )

    def test_confidently_negative_series_fails(self):
        rng = np.random.default_rng(17)
        active = _series(rng.normal(-0.0012, 0.00025, size=252))
        point_ir, ci_lower, ci_upper, se = information_ratio_ci(active, n_boot=400, seed=3)
        threshold = expected_max_ir_under_null(20, se)

        self.assertLess(ci_upper, 0.0)
        self.assertEqual(
            validation_verdict(
                1.0,
                point_ir,
                folds=8,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                selection_threshold=threshold,
            ),
            VERDICT_FAILS,
        )

    def test_expected_max_ir_under_null_is_monotonic(self):
        low_trials = expected_max_ir_under_null(5, 0.20)
        high_trials = expected_max_ir_under_null(30, 0.20)
        low_se = expected_max_ir_under_null(20, 0.10)
        high_se = expected_max_ir_under_null(20, 0.30)

        self.assertGreater(high_trials, low_trials)
        self.assertGreater(high_se, low_se)

    def test_current_data_like_point_ir_is_not_edge_when_ci_is_wide(self):
        active = _target_ir_series(0.28)
        point_ir, ci_lower, ci_upper, se = information_ratio_ci(active, n_boot=500, seed=4)
        threshold = expected_max_ir_under_null(20, se)

        self.assertAlmostEqual(point_ir, 0.28, places=12)
        self.assertGreater(point_ir, ACTIVE_IR_EDGE_MARGIN)
        self.assertLess(ci_lower, ACTIVE_IR_EDGE_MARGIN)
        self.assertEqual(
            validation_verdict(
                1.0,
                point_ir,
                folds=8,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                selection_threshold=threshold,
            ),
            VERDICT_MATCHES,
        )


if __name__ == '__main__':
    unittest.main()
