"""Phase 3 tests: walk-forward validation of the combined long-only book.

Offline, synthetic panels only. Verifies the precompute-once combined backtest
matches the original per-call path, that combined walk-forward selection has no
look-ahead, and that the combined book is reported against the benchmark.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from quant.combined_signal import (
    CombinedParams,
    backtest_combined_long_only,
    combined_long_only_weights,
    precompute_single_stock_signals,
)
from quant.models.cross_sectional import build_weights, compute_scores
from validate_cross_sectional import (
    COMBINED_GRID,
    _combined_combo_returns,
    _precompute_combined_panels,
    report_combined_validation,
    walk_forward_combined_long_only,
)


def _panel(n_days: int = 760, n_assets: int = 8, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0004, 0.009, size=n_days)
    cols = {}
    t = np.arange(n_days)
    for i in range(n_assets):
        phase = 2.0 * np.pi * i / n_assets
        rotate = 0.0007 * np.sin(2.0 * np.pi * t / 90.0 + phase)
        noise = rng.normal(0.0, 0.013, size=n_days)
        returns = market * (0.9 + 0.04 * i) + rotate + noise
        cols[f'T{i + 1}'] = 100.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame(cols, index=pd.bdate_range('2021-01-01', periods=n_days))


_XS_PARAMS = {'lookback': 63, 'skip': 0, 'top_frac': 0.33, 'rebalance': 5}


class TestCombinedBacktestRefactor(unittest.TestCase):
    def test_precomputed_matches_inline(self):
        """Passing precomputed panels/xs weights must equal the inline compute."""
        panel = _panel()
        combined = CombinedParams(z_overextended=1.5, z_oversold=-1.0,
                                  require_momentum_buy=True, long_only_mode=True)
        inline, _ = backtest_combined_long_only(panel, _XS_PARAMS, combined, cost=0.0005)

        scores = compute_scores(panel, mode='momentum', lookback=63, skip=0,
                                score_mode='raw_momentum')
        xs_w = build_weights(panel, scores, top_frac=0.33, rebalance=5,
                             market_neutral=True)
        pre = precompute_single_stock_signals(panel, mom_params={'lookback': 63, 'skip': 0})
        fast, _ = backtest_combined_long_only(
            panel, _XS_PARAMS, combined, cost=0.0005,
            precomputed=pre, xs_weights=xs_w,
        )
        pd.testing.assert_frame_equal(inline, fast)

    def test_combined_weights_normalized_long_only(self):
        panel = _panel()
        combined = CombinedParams(long_only_mode=True)
        scores = compute_scores(panel, mode='momentum', lookback=63, skip=0)
        xs_w = build_weights(panel, scores, top_frac=0.33, rebalance=5,
                             market_neutral=True)
        z, mr, mom = precompute_single_stock_signals(panel, mom_params={'lookback': 63, 'skip': 0})
        weights, _ = combined_long_only_weights(panel, xs_w, z, mr, mom, combined, 5)
        row_sums = weights.sum(axis=1)
        invested = row_sums[row_sums > 1e-9]
        np.testing.assert_allclose(invested.to_numpy(), 1.0, atol=1e-9)
        self.assertTrue((weights.to_numpy() >= -1e-12).all())


class TestCombinedWalkForward(unittest.TestCase):
    def test_runs_and_reports_structure(self):
        panel = _panel()
        wf = walk_forward_combined_long_only(
            panel, _XS_PARAMS, cost=0.0005, train=252, test=63, select='active_ir',
        )
        self.assertGreater(len(wf['folds']), 0)
        self.assertIn('oos_returns', wf)
        self.assertTrue(np.isfinite(wf['oos_metrics']['sharpe']))
        n_expected = (3 * 3 * 2 * 3)
        self.assertEqual(wf['n_combos'], n_expected)
        for fold in wf['folds']:
            self.assertIn('best_params', fold)
            self.assertIn(fold['best_params']['score_mode'], COMBINED_GRID['score_mode'])

    def test_no_lookahead_future_price_perturbation(self):
        """Perturbing a price strictly after a fold's test window cannot change
        that fold's selected params or its OOS returns."""
        panel = _panel()
        train, test = 252, 63
        base = walk_forward_combined_long_only(
            panel, _XS_PARAMS, cost=0.0005, train=train, test=test, select='active_ir',
        )
        # Perturb the very last 20 rows (strictly after the first fold's test end).
        bumped = panel.copy()
        bumped.iloc[-20:] *= 1.10
        bumped_wf = walk_forward_combined_long_only(
            bumped, _XS_PARAMS, cost=0.0005, train=train, test=test, select='active_ir',
        )
        first_test_end = panel.index[train + test - 1]
        base_first = base['oos_returns'].loc[:first_test_end]
        bumped_first = bumped_wf['oos_returns'].loc[:first_test_end]
        common = base_first.index.intersection(bumped_first.index)
        self.assertGreater(len(common), 0)
        pd.testing.assert_series_equal(
            base_first.loc[common], bumped_first.loc[common], check_names=False,
        )
        self.assertEqual(base['folds'][0]['best_params'], bumped_wf['folds'][0]['best_params'])

    def test_combo_returns_no_future_data(self):
        """A single combo's return at date t cannot move when a strictly-later
        price is perturbed."""
        panel = _panel()
        score_modes = list(dict.fromkeys(COMBINED_GRID['score_mode']))
        combo = {'z_overextended': 1.5, 'z_oversold': -1.0,
                 'require_momentum_buy': True, 'score_mode': 'raw_momentum'}

        pre, xs_w_by_mode = _precompute_combined_panels(panel, _XS_PARAMS, score_modes, 126)
        base = _combined_combo_returns(panel, combo, xs_w_by_mode, pre, 5, 0.0005)

        bumped = panel.copy()
        bumped.iloc[-5:] *= 1.2
        pre_b, xs_w_b = _precompute_combined_panels(bumped, _XS_PARAMS, score_modes, 126)
        bumped_ret = _combined_combo_returns(bumped, combo, xs_w_b, pre_b, 5, 0.0005)

        cutoff = panel.index[-6]
        pd.testing.assert_series_equal(
            base.loc[:cutoff], bumped_ret.loc[:cutoff], check_names=False,
        )

    def test_report_runs_and_returns_summaries(self):
        panel = _panel()
        out = report_combined_validation(
            panel, _XS_PARAMS, cost=0.0005, train=252, test=63, select='active_ir',
        )
        self.assertIn('combined_summary', out)
        self.assertIn('xs_summary', out)
        self.assertIn('verdict', out['combined_summary'])
        self.assertEqual(out['n_combos'], 3 * 3 * 2 * 3)


if __name__ == '__main__':
    unittest.main()
