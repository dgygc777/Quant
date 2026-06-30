"""Phase 1 tests: benchmark-relative score modes + active-IR selection.

All synthetic, offline, no network. Focus is correctness, alignment, and
absence of look-ahead, plus backward compatibility of the default path.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from quant.models.cross_sectional import (
    DEFAULT_BETA_WINDOW,
    DEFAULT_SCORE_MODE,
    SCORE_MODES,
    CrossSectionalModel,
    backtest_xs,
    compute_scores,
)
from quant.validation import (
    active_information_ratio,
    selection_objective,
    walk_forward,
)
from validate_cross_sectional import make_xs_strategy


def _panel(n_days: int = 400, n_assets: int = 6, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0004, 0.009, size=n_days)
    cols = {}
    for i in range(n_assets):
        drift = (i - n_assets / 2) * 0.00010
        noise = rng.normal(0.0, 0.011 + i * 0.001, size=n_days)
        returns = market * (0.8 + 0.05 * i) + drift + noise
        cols[f'T{i + 1}'] = 100.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame(cols, index=pd.bdate_range('2022-01-01', periods=n_days))


class TestScoreModes(unittest.TestCase):
    def test_default_is_raw_momentum_unchanged(self):
        panel = _panel()
        default = compute_scores(panel, mode='momentum', lookback=63, skip=0)
        raw = compute_scores(panel, mode='momentum', lookback=63, skip=0,
                             score_mode='raw_momentum')
        explicit_raw = panel.shift(0) / panel.shift(63) - 1.0
        pd.testing.assert_frame_equal(default, raw)
        pd.testing.assert_frame_equal(default, explicit_raw)

    def test_all_modes_listed_and_runnable(self):
        panel = _panel()
        for mode in SCORE_MODES:
            with self.subTest(score_mode=mode):
                scores = compute_scores(panel, mode='momentum', lookback=63, skip=0,
                                        score_mode=mode)
                self.assertEqual(scores.shape, panel.shape)
                # Late rows should have finite scores for most names.
                self.assertTrue(np.isfinite(scores.iloc[-1].dropna()).all())

    def test_relative_momentum_is_raw_minus_benchmark(self):
        panel = _panel()
        lookback, skip = 63, 0
        rel = compute_scores(panel, mode='momentum', lookback=lookback, skip=skip,
                             score_mode='relative_momentum')
        raw = panel.shift(skip) / panel.shift(skip + lookback) - 1.0
        bench_ret = panel.pct_change(fill_method=None).mean(axis=1)
        bench_px = (1.0 + bench_ret.fillna(0.0)).cumprod()
        bench_mom = bench_px.shift(skip) / bench_px.shift(skip + lookback) - 1.0
        expected = raw.sub(bench_mom, axis=0)
        pd.testing.assert_frame_equal(rel, expected)

    def test_relative_momentum_cross_sectionally_demeaned_ranks_match_raw(self):
        # Subtracting a per-day scalar benchmark cannot change the cross-sectional
        # ordering on any given day (it shifts every name equally).
        panel = _panel()
        raw = compute_scores(panel, mode='momentum', lookback=63, skip=0,
                             score_mode='raw_momentum')
        rel = compute_scores(panel, mode='momentum', lookback=63, skip=0,
                             score_mode='relative_momentum')
        last_raw = raw.iloc[-1].rank()
        last_rel = rel.iloc[-1].rank()
        pd.testing.assert_series_equal(last_raw, last_rel)

    def test_residual_momentum_differs_from_relative_and_no_future_data(self):
        panel = _panel()
        resid = compute_scores(panel, mode='momentum', lookback=63, skip=0,
                               score_mode='residual_momentum', beta_window=126)
        rel = compute_scores(panel, mode='momentum', lookback=63, skip=0,
                             score_mode='relative_momentum')
        # With heterogeneous betas, residual should not be identical to relative.
        diff = (resid - rel).abs().to_numpy()
        self.assertTrue(np.nanmax(diff) > 1e-9)

    def test_residual_momentum_no_lookahead(self):
        # Perturbing a future price must not change earlier residual scores.
        panel = _panel()
        perturb_pos = 300
        base = compute_scores(panel, mode='momentum', lookback=63, skip=0,
                              score_mode='residual_momentum', beta_window=126)
        bumped = panel.copy()
        bumped.iloc[perturb_pos, 0] *= 1.20
        alt = compute_scores(bumped, mode='momentum', lookback=63, skip=0,
                             score_mode='residual_momentum', beta_window=126)
        before = panel.index[:perturb_pos]
        np.testing.assert_allclose(
            base.loc[before].fillna(0.0).to_numpy(),
            alt.loc[before].fillna(0.0).to_numpy(),
            rtol=0.0,
            atol=1e-12,
        )

    def test_relative_momentum_no_lookahead(self):
        panel = _panel()
        perturb_pos = 250
        base = compute_scores(panel, mode='momentum', lookback=63, skip=0,
                              score_mode='relative_momentum')
        bumped = panel.copy()
        bumped.iloc[perturb_pos, 1] *= 1.15
        alt = compute_scores(bumped, mode='momentum', lookback=63, skip=0,
                             score_mode='relative_momentum')
        before = panel.index[:perturb_pos]
        np.testing.assert_allclose(
            base.loc[before].fillna(0.0).to_numpy(),
            alt.loc[before].fillna(0.0).to_numpy(),
            rtol=0.0,
            atol=1e-12,
        )

    def test_invalid_beta_window_raises(self):
        panel = _panel()
        with self.assertRaises(ValueError):
            compute_scores(panel, mode='momentum', lookback=63, skip=0,
                           score_mode='residual_momentum', beta_window=1)

    def test_model_default_params_include_score_mode(self):
        params = CrossSectionalModel().default_params()
        self.assertEqual(params['score_mode'], DEFAULT_SCORE_MODE)
        self.assertEqual(params['beta_window'], DEFAULT_BETA_WINDOW)

    def test_backtest_xs_threads_score_mode(self):
        panel = _panel()
        df, _ = backtest_xs(panel, mode='momentum', lookback=63, skip=0,
                            top_frac=0.33, rebalance=5, market_neutral=False,
                            score_mode='relative_momentum')
        self.assertIn('strat_net', df.columns)
        self.assertFalse(df['strat_net'].isna().all())


class TestActiveIRSelection(unittest.TestCase):
    def test_active_information_ratio_matches_manual(self):
        idx = pd.bdate_range('2024-01-01', periods=60)
        strat = pd.Series(np.linspace(0.001, 0.002, 60), index=idx)
        bench = pd.Series(0.0005, index=idx)
        active = strat - bench
        expected = active.mean() / active.std() * np.sqrt(252.0)
        self.assertAlmostEqual(active_information_ratio(strat, bench), expected, places=10)

    def test_selection_objective_sharpe_backward_compatible(self):
        idx = pd.bdate_range('2024-01-01', periods=60)
        returns = pd.Series(np.random.default_rng(1).normal(0.0005, 0.01, 60), index=idx)
        from quant.metrics import metrics
        self.assertAlmostEqual(
            selection_objective(returns, None, select='sharpe'),
            metrics(returns)['sharpe'],
            places=12,
        )

    def test_walk_forward_active_ir_no_lookahead_in_selection(self):
        # Selecting by active_ir must give identical results whether or not
        # future data exists beyond the evaluated window, since each train
        # window builds its own benchmark.
        panel = _panel(n_days=320)
        grid = {'lookback': [20, 63], 'skip': [0], 'top_frac': [0.33], 'rebalance': [5]}
        strat = make_xs_strategy('long_only', cost=0.0)
        wf_short = walk_forward(strat, panel.iloc[:260], grid, train=120, test=40,
                                warmup=30, select='active_ir')
        wf_long = walk_forward(strat, panel, grid, train=120, test=40,
                               warmup=30, select='active_ir')
        # The first fold's selected params depend only on the first train slice,
        # so they must match regardless of how much later data exists.
        self.assertEqual(
            wf_short['folds'][0]['best_params'],
            wf_long['folds'][0]['best_params'],
        )

    def test_walk_forward_select_modes_both_run(self):
        panel = _panel(n_days=300)
        grid = {'lookback': [20, 63], 'skip': [0], 'top_frac': [0.33], 'rebalance': [5]}
        strat = make_xs_strategy('long_only', cost=0.0)
        for select in ('sharpe', 'active_ir'):
            with self.subTest(select=select):
                wf = walk_forward(strat, panel, grid, train=120, test=40,
                                  warmup=30, select=select)
                self.assertTrue(len(wf['folds']) >= 1)
                self.assertEqual(wf['folds'][0]['select'], select)


if __name__ == '__main__':
    unittest.main()
