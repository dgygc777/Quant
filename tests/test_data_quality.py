"""Unit tests for shared panel data-quality helpers."""

from __future__ import annotations

import unittest

import pandas as pd

from quant.data_quality import (
    EXTREME_DAILY_RETURN,
    coverage_by_ticker,
    filter_panel_by_coverage,
    winsorize_extreme_returns,
)


class TestDataQuality(unittest.TestCase):
    def test_sparse_name_has_shared_coverage_gate(self):
        idx = pd.bdate_range('2025-01-01', periods=100)
        panel = pd.DataFrame({
            'FULL': range(100),
            'SPARSE': [float('nan')] * 86 + list(range(14)),
        }, index=idx)

        filtered, coverage, dropped = filter_panel_by_coverage(panel)

        self.assertAlmostEqual(float(coverage_by_ticker(panel)['SPARSE']), 0.14)
        self.assertAlmostEqual(float(coverage['SPARSE']), 0.14)
        self.assertEqual(filtered.columns.tolist(), ['FULL'])
        self.assertEqual(dropped.index.tolist(), ['SPARSE'])

    def test_winsorize_extreme_return_clips_and_rebuilds_path(self):
        idx = pd.bdate_range('2025-01-01', periods=3)
        prices = pd.DataFrame({'GLITCH': [100.0, 200.0, 210.0]}, index=idx)

        adjusted, clipped = winsorize_extreme_returns(prices)

        self.assertEqual(len(clipped), 1)
        self.assertEqual(clipped.loc[0, 'ticker'], 'GLITCH')
        self.assertAlmostEqual(float(clipped.loc[0, 'original_return']), 1.0)
        self.assertAlmostEqual(float(clipped.loc[0, 'clipped_return']), EXTREME_DAILY_RETURN)
        self.assertAlmostEqual(float(adjusted.loc[idx[1], 'GLITCH']), 135.0)
        self.assertAlmostEqual(float(adjusted.loc[idx[2], 'GLITCH']), 141.75)


if __name__ == '__main__':
    unittest.main()
