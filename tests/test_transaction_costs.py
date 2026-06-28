"""Transaction-cost tests for cross-sectional validation."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from quant.models.cross_sectional import backtest_xs
from quant.validation import walk_forward
from validate_cross_sectional import (
    active_ir_increase_warnings,
    break_even_cost_bps,
    compare_weighting_validation,
    make_xs_strategy,
    transaction_cost_sensitivity,
)


def synthetic_cost_panel(n_days: int = 180, n_assets: int = 6, seed: int = 101) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0003, 0.008, size=n_days)
    data = {}
    for i in range(n_assets):
        drift = (i - n_assets / 2) * 0.00007
        noise = rng.normal(0.0, 0.010 + i * 0.001, size=n_days)
        returns = market + drift + noise
        data[f'T{i + 1}'] = 100.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame(data, index=pd.bdate_range('2024-01-01', periods=n_days))


class TestTransactionCosts(unittest.TestCase):
    def test_scalar_cost_matches_equivalent_constant_series(self):
        panel = synthetic_cost_panel()
        params = {
            'mode': 'momentum',
            'lookback': 20,
            'skip': 0,
            'top_frac': 0.33,
            'rebalance': 5,
            'market_neutral': False,
        }
        scalar, _ = backtest_xs(panel, cost=0.001, **params)
        series_cost = pd.Series(0.001, index=panel.columns)
        series, _ = backtest_xs(panel, cost=series_cost, **params)

        np.testing.assert_allclose(
            scalar['strat_net'].to_numpy(),
            series['strat_net'].to_numpy(),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            scalar['cost'].to_numpy(),
            series['cost'].to_numpy(),
            rtol=0.0,
            atol=1e-12,
        )

    def test_higher_per_name_cost_lowers_net_return(self):
        panel = synthetic_cost_panel()
        params = {
            'mode': 'momentum',
            'lookback': 20,
            'skip': 0,
            'top_frac': 0.33,
            'rebalance': 5,
            'market_neutral': False,
        }
        low_cost = pd.Series(0.0001, index=panel.columns)
        high_cost = pd.Series(0.0020, index=panel.columns)
        low, _ = backtest_xs(panel, cost=low_cost, **params)
        high, _ = backtest_xs(panel, cost=high_cost, **params)

        self.assertLessEqual(float(high['strat_net'].sum()), float(low['strat_net'].sum()))
        self.assertGreaterEqual(float(high['cost'].sum()), float(low['cost'].sum()))

    def test_per_name_cost_introduces_no_nans(self):
        panel = synthetic_cost_panel()
        sparse_cost = pd.Series({'T1': 0.002, 'T3': 0.001})
        result, _ = backtest_xs(
            panel,
            mode='momentum',
            lookback=20,
            skip=0,
            top_frac=0.33,
            rebalance=5,
            market_neutral=False,
            cost=sparse_cost,
        )

        self.assertFalse(result[['strat_net', 'ret', 'turnover', 'cost', 'gross']].isna().any().any())

    def test_per_name_cost_does_not_look_ahead(self):
        panel = synthetic_cost_panel(n_days=220)
        perturb_pos = 120
        perturb_date = panel.index[perturb_pos]
        cost = pd.Series(np.linspace(0.0002, 0.0012, panel.shape[1]), index=panel.columns)
        params = {
            'mode': 'momentum',
            'lookback': 20,
            'skip': 0,
            'top_frac': 0.33,
            'rebalance': 5,
            'market_neutral': False,
            'cost': cost,
        }

        base, _ = backtest_xs(panel, **params)
        perturbed = panel.copy()
        perturbed.iloc[perturb_pos, 0] *= 1.25
        alt, _ = backtest_xs(perturbed, **params)

        before = pd.concat([base['strat_net'], alt['strat_net']], axis=1)
        before = before.loc[before.index < perturb_date].dropna()
        np.testing.assert_allclose(
            before.iloc[:, 0].to_numpy(),
            before.iloc[:, 1].to_numpy(),
            rtol=0.0,
            atol=1e-14,
        )

    def test_cost_sweep_cost_paid_is_non_decreasing(self):
        panel = synthetic_cost_panel(n_days=240, seed=202)
        grid = {'lookback': [20], 'skip': [0], 'top_frac': [0.33], 'rebalance': [5]}
        summary = transaction_cost_sensitivity(
            panel,
            grid,
            cost_bps_levels=[0, 10, 20, 40],
            ci_n_boot=200,
            train=80,
            test=20,
            warmup=30,
        )
        cost_paid = [row['strategy_cost_mean'] for row in summary['rows']]

        for left, right in zip(cost_paid, cost_paid[1:]):
            self.assertLessEqual(left, right + 1e-12)

    def test_cost_sweep_zero_bps_matches_canonical_walk_forward(self):
        panel = synthetic_cost_panel(n_days=240, seed=303)
        grid = {'lookback': [20], 'skip': [0], 'top_frac': [0.33], 'rebalance': [5]}
        wf_kwargs = {'train': 80, 'test': 20, 'warmup': 30}
        canonical = walk_forward(
            make_xs_strategy('long_only', cost=0.0),
            panel,
            grid,
            **wf_kwargs,
        )
        summary = transaction_cost_sensitivity(
            panel,
            grid,
            cost_bps_levels=[0, 10],
            ci_n_boot=200,
            **wf_kwargs,
        )
        row0 = summary['rows'][0]

        self.assertAlmostEqual(row0['sharpe'], canonical['oos_metrics']['sharpe'], places=6)
        pd.testing.assert_series_equal(
            row0['wf']['oos_returns'],
            canonical['oos_returns'],
            check_names=False,
        )

    def test_cost_sweep_headline_bps_matches_canonical_walk_forward(self):
        panel = synthetic_cost_panel(n_days=240, seed=313)
        grid = {'lookback': [20], 'skip': [0], 'top_frac': [0.33], 'rebalance': [5]}
        wf_kwargs = {'train': 80, 'test': 20, 'warmup': 30}
        headline_cost = 0.001
        canonical = walk_forward(
            make_xs_strategy('long_only', cost=headline_cost),
            panel,
            grid,
            **wf_kwargs,
        )
        summary = transaction_cost_sensitivity(
            panel,
            grid,
            cost_bps_levels=[0, 10, 20],
            selection_cost=headline_cost,
            ci_n_boot=200,
            **wf_kwargs,
        )
        row10 = next(row for row in summary['rows'] if row['cost_bps'] == 10.0)

        self.assertAlmostEqual(row10['sharpe'], canonical['oos_metrics']['sharpe'], places=6)
        pd.testing.assert_series_equal(
            row10['wf']['oos_returns'],
            canonical['oos_returns'],
            check_names=False,
        )

    def test_active_ir_increase_is_warning_not_assertion(self):
        rows = [
            {'cost_bps': 0.0, 'active_ir': 0.191702738435},
            {'cost_bps': 5.0, 'active_ir': 0.255967426701},
        ]

        warnings = active_ir_increase_warnings(rows)

        self.assertEqual(len(warnings), 1)
        self.assertIn('Active IR increased', warnings[0])

    def test_equal_sizing_validation_matches_canonical_long_only_at_cost(self):
        panel = synthetic_cost_panel(n_days=240, seed=404)
        grid = {'lookback': [20], 'skip': [0], 'top_frac': [0.33], 'rebalance': [5]}
        wf_kwargs = {'train': 80, 'test': 20, 'warmup': 30}
        cost = 0.001
        canonical = walk_forward(
            make_xs_strategy('long_only', cost=cost),
            panel,
            grid,
            **wf_kwargs,
        )
        sizing = compare_weighting_validation(panel, grid, cost=cost, **wf_kwargs)

        self.assertAlmostEqual(
            sizing['rows']['equal']['sharpe'],
            canonical['oos_metrics']['sharpe'],
            places=6,
        )

    def test_break_even_interpolation_between_bracketing_levels(self):
        breakeven = break_even_cost_bps([0, 10, 20], [0.50, 0.10, -0.30])

        self.assertIsNotNone(breakeven)
        self.assertGreater(float(breakeven), 10.0)
        self.assertLess(float(breakeven), 20.0)
        self.assertAlmostEqual(float(breakeven), 12.5)


if __name__ == '__main__':
    unittest.main()
