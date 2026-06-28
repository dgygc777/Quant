"""Unit tests for cross-sectional validation helpers."""

from __future__ import annotations

import contextlib
import io
import unittest

import numpy as np
import pandas as pd

from validate_cross_sectional import (
    VERDICT_EDGE,
    VERDICT_FAILS,
    VERDICT_MATCHES,
    filter_panel_by_coverage,
    information_ratio,
    report_panel_validation,
    validation_verdict,
)


class TestCrossSectionalValidation(unittest.TestCase):
    def test_filter_panel_by_coverage_drops_sparse_ticker(self):
        idx = pd.bdate_range('2024-01-01', periods=10)
        panel = pd.DataFrame({
            'FULL': np.arange(10, dtype=float),
            'SPARSE': [np.nan] * 8 + [1.0, 2.0],
            'ENOUGH': [np.nan] * 3 + list(range(7)),
        }, index=idx)

        filtered, coverage, dropped = filter_panel_by_coverage(panel, min_coverage=0.60)

        self.assertEqual(set(filtered.columns), {'FULL', 'ENOUGH'})
        self.assertAlmostEqual(float(coverage['SPARSE']), 0.20)
        self.assertEqual(dropped.index.tolist(), ['SPARSE'])

    def test_filter_panel_by_coverage_keeps_sparse_ticker_at_low_threshold(self):
        idx = pd.bdate_range('2024-01-01', periods=10)
        panel = pd.DataFrame({
            'FULL': np.arange(10, dtype=float),
            'SPARSE': [np.nan] * 8 + [1.0, 2.0],
        }, index=idx)

        filtered, _, dropped = filter_panel_by_coverage(panel, min_coverage=0.10)

        self.assertEqual(set(filtered.columns), {'FULL', 'SPARSE'})
        self.assertTrue(dropped.empty)

    def test_benchmark_relative_verdict_requires_active_edge(self):
        self.assertEqual(validation_verdict(1.60, 0.40, folds=8), VERDICT_EDGE)
        self.assertEqual(validation_verdict(1.00, 0.02, folds=8), VERDICT_MATCHES)
        self.assertEqual(validation_verdict(1.00, -0.40, folds=8), VERDICT_FAILS)
        self.assertNotEqual(validation_verdict(1.00, -0.40, folds=8), VERDICT_EDGE)
        self.assertEqual(validation_verdict(-0.10, 1.00, folds=12), VERDICT_FAILS)

    def test_information_ratio_handles_zero_variance_spread(self):
        idx = pd.bdate_range('2025-01-01', periods=20)
        self.assertEqual(information_ratio(pd.Series(0.0, index=idx)), 0.0)
        self.assertEqual(information_ratio(pd.Series(0.001, index=idx)), float('inf'))
        self.assertEqual(information_ratio(pd.Series(-0.001, index=idx)), float('-inf'))

    def test_validation_report_prints_benchmark_and_verdict(self):
        rng = np.random.default_rng(5)
        idx = pd.bdate_range('2024-01-01', periods=90)
        rets = rng.normal(0.0006, 0.012, size=(len(idx), 5))
        panel = pd.DataFrame(
            100.0 * np.exp(np.cumsum(rets, axis=0)),
            index=idx,
            columns=['A', 'B', 'C', 'D', 'E'],
        )

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            wf = report_panel_validation(
                'synthetic validation',
                panel,
                {'lookback': [5], 'skip': [0], 'top_frac': [0.4], 'rebalance': [5]},
                train=30,
                test=10,
                warmup=5,
            )
        out = buf.getvalue()

        self.assertIn('Walk-forward, OUT-OF-SAMPLE', out)
        self.assertIn('Equal-weight benchmark OOS', out)
        self.assertIn('Active OOS Sharpe (strategy - benchmark)', out)
        self.assertIn('Information ratio (active-return OOS)', out)
        self.assertIn('Validation verdict:', out)
        self.assertIn('benchmark_oos_metrics', wf)
        self.assertIn('active_oos_sharpe', wf)
        self.assertIn('information_ratio', wf)
        self.assertIn('validation_verdict', wf)


if __name__ == '__main__':
    unittest.main()
