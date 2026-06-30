"""Tests for the trend / vol-timing overlay. Offline, synthetic panels."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from quant.metrics import metrics
from quant.trend_overlay import (
    backtest_trend_overlay,
    basket_price_and_returns,
    make_trend_overlay_strategy,
    overlay_verdict,
    report_trend_overlay_validation,
    trend_exposure,
    vol_target_scale,
    VERDICT_DEFENSIVE,
    VERDICT_INSUFFICIENT,
)


def _trending_panel(n_days=900, n_assets=6, seed=3) -> pd.DataFrame:
    """Up-trend then a sharp crash then recovery — so a trend gate can help."""
    rng = np.random.default_rng(seed)
    cols = {}
    drift = np.concatenate([
        np.full(n_days // 3, 0.0009),     # bull
        np.full(n_days // 3, -0.0016),    # bear / crash
        np.full(n_days - 2 * (n_days // 3), 0.0008),  # recovery
    ])
    for i in range(n_assets):
        noise = rng.normal(0.0, 0.013, size=n_days)
        r = drift + noise
        cols[f'T{i+1}'] = 100.0 * np.exp(np.cumsum(r))
    return pd.DataFrame(cols, index=pd.bdate_range('2019-01-01', periods=n_days))


class TestExposure(unittest.TestCase):
    def test_exposure_is_binary_and_bounded(self):
        panel = _trending_panel()
        price, _ = basket_price_and_returns(panel)
        expo = trend_exposure(price, mode='sma', window=100).dropna()
        self.assertTrue(set(np.unique(expo.to_numpy())).issubset({0.0, 1.0}))

    def test_invalid_window_raises(self):
        panel = _trending_panel()
        price, _ = basket_price_and_returns(panel)
        with self.assertRaises(ValueError):
            trend_exposure(price, window=1)

    def test_vol_target_scale_capped(self):
        panel = _trending_panel()
        _, rets = basket_price_and_returns(panel)
        scale = vol_target_scale(rets, target_vol=0.15, vol_window=63, max_leverage=1.0).dropna()
        self.assertTrue((scale <= 1.0 + 1e-12).all())
        self.assertTrue((scale >= 0.0).all())


class TestBacktest(unittest.TestCase):
    def test_no_lookahead_future_perturbation(self):
        panel = _trending_panel()
        base = backtest_trend_overlay(panel, mode='sma', window=100)
        bumped = panel.copy()
        bumped.iloc[-15:] *= 1.3
        bump = backtest_trend_overlay(bumped, mode='sma', window=100)
        cutoff = panel.index[-16]
        pd.testing.assert_series_equal(
            base['overlay_net'].loc[:cutoff],
            bump['overlay_net'].loc[:cutoff],
            check_names=False,
        )

    def test_cash_earned_when_flat(self):
        """With a positive cash rate, flat days earn the cash return, not zero."""
        panel = _trending_panel()
        df = backtest_trend_overlay(panel, mode='sma', window=100, cash_rate=0.05)
        flat_days = df[df['exposure'].shift(1).fillna(0.0) < 1e-9]
        # On fully-flat days with no turnover, net return equals the daily cash rate.
        no_turn = flat_days[flat_days['turnover'] < 1e-12]
        if len(no_turn):
            daily_cash = (1.05) ** (1.0 / 252.0) - 1.0
            np.testing.assert_allclose(
                no_turn['overlay_net'].to_numpy(), daily_cash, atol=1e-12,
            )

    def test_reduces_drawdown_vs_buy_and_hold(self):
        """On a panel with a crash, the trend overlay should cut drawdown."""
        panel = _trending_panel()
        df = backtest_trend_overlay(panel, mode='sma', window=100, cost=0.0005)
        overlay_dd = metrics(df['overlay_net'])['max_dd']
        basket_dd = metrics(df['basket_ret'])['max_dd']
        self.assertLess(abs(overlay_dd), abs(basket_dd))

    def test_zero_window_change_no_nans(self):
        panel = _trending_panel()
        df = backtest_trend_overlay(panel, mode='tsmom', window=120)
        self.assertFalse(df['overlay_net'].isna().any())


class TestVerdict(unittest.TestCase):
    def test_insufficient_folds(self):
        v = overlay_verdict(1.0, 0.5, -0.10, -0.30, folds=2)
        self.assertEqual(v, VERDICT_INSUFFICIENT)

    def test_defensive_when_dd_better_sharpe_flat(self):
        v = overlay_verdict(0.80, 0.82, -0.18, -0.40, folds=10)
        self.assertEqual(v, VERDICT_DEFENSIVE)


class TestWalkForward(unittest.TestCase):
    def test_report_runs_and_returns_stats(self):
        panel = _trending_panel()
        wf = report_trend_overlay_validation(
            panel, train=252, test=63, windows=[50, 100, 150], mode='sma',
        )
        self.assertIn('verdict', wf)
        self.assertIn('basket_oos_metrics', wf)
        self.assertGreater(len(wf['folds']), 0)
        self.assertTrue(0.0 <= wf['pct_invested'] <= 1.0)

    def test_strategy_adapter_matches_backtest(self):
        panel = _trending_panel()
        strat = make_trend_overlay_strategy(mode='sma')
        direct = backtest_trend_overlay(panel, mode='sma', window=150)['overlay_net']
        via = strat(panel, window=150)
        pd.testing.assert_series_equal(direct, via, check_names=False)


if __name__ == '__main__':
    unittest.main()
