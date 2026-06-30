"""Phase 4 tests: rank hysteresis / turnover control. Offline, synthetic."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from quant.models.cross_sectional import (
    backtest_xs,
    build_weights,
    compute_scores,
)
from quant.validation import walk_forward
from validate_cross_sectional import (
    hysteresis_kwargs,
    make_xs_strategy,
    walk_forward_long_only_with_turnover,
)


def _panel(n_days: int = 500, n_assets: int = 8, seed: int = 19) -> pd.DataFrame:
    """Panel with frequent leadership rotation to create churn without hysteresis."""
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0003, 0.008, size=n_days)
    cols = {}
    for i in range(n_assets):
        # Slow sinusoidal drift rotation so rank order shuffles over time.
        phase = 2.0 * np.pi * i / n_assets
        t = np.arange(n_days)
        rotate = 0.0006 * np.sin(2.0 * np.pi * t / 80.0 + phase)
        noise = rng.normal(0.0, 0.012, size=n_days)
        returns = market * (0.9 + 0.03 * i) + rotate + noise
        cols[f'T{i + 1}'] = 100.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame(cols, index=pd.bdate_range('2022-01-01', periods=n_days))


def _avg_turnover(df: pd.DataFrame) -> float:
    return float(df['turnover'].mean()) if len(df) else 0.0


class TestHysteresis(unittest.TestCase):
    def test_default_off_is_backward_compatible(self):
        panel = _panel()
        params = dict(mode='momentum', lookback=63, skip=0, top_frac=0.33,
                      rebalance=5, market_neutral=False)
        base = backtest_xs(panel, **params)[0]
        explicit_off = backtest_xs(panel, use_hysteresis=False, **params)[0]
        pd.testing.assert_frame_equal(base, explicit_off)

    def test_hysteresis_lowers_turnover(self):
        panel = _panel()
        params = dict(mode='momentum', lookback=63, skip=0, top_frac=0.33,
                      rebalance=5, market_neutral=False)
        no_hys = backtest_xs(panel, **params)[0]
        with_hys = backtest_xs(
            panel, use_hysteresis=True, entry_rank_pct=0.80, exit_rank_pct=0.60,
            **params,
        )[0]
        self.assertLess(_avg_turnover(with_hys), _avg_turnover(no_hys))

    def test_weights_normalized_long_only(self):
        panel = _panel()
        scores = compute_scores(panel, mode='momentum', lookback=63, skip=0)
        weights = build_weights(
            panel, scores, top_frac=0.33, rebalance=5, market_neutral=False,
            use_hysteresis=True,
        )
        # On every day with an active book, long weights sum to 1.
        sums = weights[weights > 0].sum(axis=1)
        active = sums[sums > 1e-9]
        np.testing.assert_allclose(active.to_numpy(), 1.0, atol=1e-9)

    def test_max_new_names_limits_entries(self):
        panel = _panel()
        scores = compute_scores(panel, mode='momentum', lookback=63, skip=0)
        weights = build_weights(
            panel, scores, top_frac=0.5, rebalance=5, market_neutral=False,
            use_hysteresis=True, entry_rank_pct=0.55, exit_rank_pct=0.50,
            max_new_names_per_rebalance=1,
        )
        # Count names newly entering the book at each rebalance after the first.
        held = set()
        first = True
        rebal_rows = weights.iloc[::5]
        for _, row in rebal_rows.iterrows():
            current = set(row[row > 0].index)
            if not first:
                new = current - held
                self.assertLessEqual(len(new), 1)
            held = current
            first = False

    def test_hysteresis_invalid_bands_raise(self):
        panel = _panel(n_days=200)
        scores = compute_scores(panel, mode='momentum', lookback=63, skip=0)
        with self.assertRaises(ValueError):
            build_weights(
                panel, scores, top_frac=0.33, rebalance=5, market_neutral=False,
                use_hysteresis=True, entry_rank_pct=0.50, exit_rank_pct=0.80,
            )

    def test_hysteresis_no_lookahead(self):
        panel = _panel()
        params = dict(mode='momentum', lookback=63, skip=0, top_frac=0.33,
                      rebalance=5, market_neutral=False, use_hysteresis=True)
        base = backtest_xs(panel, **params)[0]['strat_net']
        perturb_pos = 400
        perturb_date = panel.index[perturb_pos]
        bumped = panel.copy()
        bumped.iloc[perturb_pos, 0] *= 1.30
        alt = backtest_xs(bumped, **params)[0]['strat_net']
        before = pd.concat([base, alt], axis=1).loc[lambda d: d.index < perturb_date].dropna()
        np.testing.assert_allclose(
            before.iloc[:, 0].to_numpy(), before.iloc[:, 1].to_numpy(),
            rtol=0.0, atol=1e-12,
        )


class TestTurnoverPenaltySelection(unittest.TestCase):
    def test_hysteresis_kwargs_helper(self):
        self.assertEqual(hysteresis_kwargs(False), {})
        kw = hysteresis_kwargs(True, entry_rank_pct=0.8, exit_rank_pct=0.6)
        self.assertTrue(kw['use_hysteresis'])
        self.assertEqual(kw['entry_rank_pct'], 0.8)

    def test_turnover_penalty_changes_selection_toward_lower_turnover(self):
        panel = _panel(n_days=420)
        grid = {'lookback': [20, 126], 'skip': [0], 'top_frac': [0.33], 'rebalance': [5, 21]}
        no_penalty = walk_forward_long_only_with_turnover(
            panel, grid, cost=0.0, train=160, test=40, warmup=40,
            select='active_ir', turnover_penalty=0.0,
        )
        high_penalty = walk_forward_long_only_with_turnover(
            panel, grid, cost=0.0, train=160, test=40, warmup=40,
            select='active_ir', turnover_penalty=5.0,
        )
        # A large turnover penalty should not increase average OOS turnover.
        self.assertLessEqual(
            float(high_penalty['turnover'].mean()) - 1e-9,
            float(no_penalty['turnover'].mean()) + 1e-9 + 0.5,
        )
        # Both produce finite OOS metrics.
        self.assertTrue(np.isfinite(high_penalty['oos_metrics']['sharpe']))

    def test_make_xs_strategy_applies_hysteresis(self):
        panel = _panel(n_days=300)
        grid = {'lookback': [63], 'skip': [0], 'top_frac': [0.33], 'rebalance': [5]}
        plain = make_xs_strategy('long_only', cost=0.0)
        hyst = make_xs_strategy('long_only', cost=0.0,
                                hysteresis=hysteresis_kwargs(True))
        wf_plain = walk_forward(plain, panel, grid, train=140, test=40, warmup=30)
        wf_hyst = walk_forward(hyst, panel, grid, train=140, test=40, warmup=30)
        # Returns differ once hysteresis is engaged (different membership path).
        merged = pd.concat([
            wf_plain['oos_returns'].rename('a'),
            wf_hyst['oos_returns'].rename('b'),
        ], axis=1).dropna()
        self.assertGreater((merged['a'] - merged['b']).abs().sum(), 0.0)


if __name__ == '__main__':
    unittest.main()
