"""Synthetic stress tests for the cross-sectional validation pipeline."""

from __future__ import annotations

import contextlib
import io
import unittest

import numpy as np
import pandas as pd

from cli import _buy_strategy_conclusion, _print_ranked_candidate_risk, _print_wf_validation
from quant.data_quality import coverage_by_ticker, filter_panel_by_coverage
from quant.models.cross_sectional import backtest_xs
from quant.risk_model import WEIGHTING_METHODS
from quant.validation import walk_forward
from validate_cross_sectional import (
    VERDICT_EDGE,
    VERDICT_FAILS,
    VERDICT_MATCHES,
    compare_weighting_validation,
    information_ratio,
    make_xs_strategy,
    validation_verdict,
)


def synthetic_panel(n_days: int = 260, n_assets: int = 6, seed: int = 23) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0004, 0.008, size=n_days)
    cols = {}
    for i in range(n_assets):
        drift = (i - n_assets / 2) * 0.00008
        noise = rng.normal(0.0, 0.010 + i * 0.001, size=n_days)
        returns = market * (0.8 + i * 0.05) + drift + noise
        cols[f'T{i + 1}'] = 100.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame(cols, index=pd.bdate_range('2024-01-01', periods=n_days))


class TestPipelineStress(unittest.TestCase):
    def test_weighting_schemes_do_not_look_ahead(self):
        panel = synthetic_panel(n_days=280)
        perturb_pos = 170
        perturb_date = panel.index[perturb_pos]
        params = {
            'mode': 'momentum',
            'lookback': 20,
            'skip': 0,
            'top_frac': 0.33,
            'rebalance': 5,
            'market_neutral': False,
        }

        for weighting in WEIGHTING_METHODS:
            with self.subTest(weighting=weighting):
                base = backtest_xs(panel, weighting=weighting, **params)[0]['strat_net']
                perturbed = panel.copy()
                perturbed.iloc[perturb_pos, 0] *= 1.35
                alt = backtest_xs(perturbed, weighting=weighting, **params)[0]['strat_net']

                before = pd.concat([base, alt], axis=1).loc[lambda df: df.index < perturb_date].dropna()
                np.testing.assert_allclose(
                    before.iloc[:, 0].to_numpy(),
                    before.iloc[:, 1].to_numpy(),
                    rtol=0.0,
                    atol=1e-14,
                )

    def test_information_ratio_verdicts_on_hand_built_cases(self):
        idx = pd.bdate_range('2025-01-01', periods=40)
        matches_ir = information_ratio(pd.Series([0.001, -0.001] * 20, index=idx))
        edge_ir = information_ratio(pd.Series(0.0005, index=idx))
        fail_ir = information_ratio(pd.Series(-0.0005, index=idx))

        self.assertEqual(validation_verdict(1.0, matches_ir, folds=8), VERDICT_MATCHES)
        self.assertEqual(validation_verdict(1.0, edge_ir, folds=8), VERDICT_EDGE)
        self.assertEqual(validation_verdict(1.0, fail_ir, folds=8), VERDICT_FAILS)
        self.assertEqual(validation_verdict(-0.1, edge_ir, folds=8), VERDICT_FAILS)

    def test_coverage_propagates_to_validation_risk_and_actionable(self):
        panel = synthetic_panel(n_days=100, n_assets=6)
        sparse = pd.Series(np.nan, index=panel.index, name='SNDK')
        sparse.iloc[-14:] = np.linspace(100.0, 115.0, 14)
        panel['SNDK'] = sparse
        coverage = coverage_by_ticker(panel)
        filtered, _, dropped = filter_panel_by_coverage(panel, coverage=coverage)
        self.assertNotIn('SNDK', filtered.columns)
        self.assertIn('SNDK', dropped.index)

        scores = pd.Series({
            'SNDK': 0.50,
            'T1': 0.42,
            'T2': 0.36,
            'T3': 0.31,
            'T4': 0.28,
            'T5': 0.22,
            'T6': 0.18,
        })
        weights = pd.Series(0.0, index=scores.index)
        weights[['SNDK', 'T1']] = 0.5
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            sizing = _print_ranked_candidate_risk(
                panel,
                weights,
                scores,
                {'momentum_preset': 'mom_10d', 'lookback': 10, 'skip': 0},
                years=1,
                coverage=coverage,
            )

        self.assertIn('SNDK', sizing['speculative'])
        self.assertNotIn('SNDK', sizing['risk_sized_pool'])
        self.assertIn('INSUFFICIENT HISTORY - speculative only', buf.getvalue())

        tenk_df = pd.DataFrame([
            {'ticker': ticker, 'final_action': 'BUY', 'risk_flag': False, 'tenk_ok': True}
            for ticker in scores.index
        ])
        text = _buy_strategy_conclusion(
            'custom',
            {'momentum_preset': 'mom_10d', 'lookback': 10, 'skip': 0},
            weights,
            scores,
            {
                'folds': 8,
                'oos_sharpe': 1.0,
                'oos_ann_return': 0.10,
                'oos_max_dd': -0.10,
                'benchmark_oos_sharpe': 0.40,
                'active_oos_sharpe': 0.60,
                'information_ratio': 0.50,
                'validation_verdict': VERDICT_EDGE,
                'winning_weighting': 'risk_parity',
                'risk_parity_beats_equal': True,
            },
            tenk_df,
            sizing,
            coverage=coverage,
        )
        self.assertIn('insufficient-history speculative bucket: SNDK 14%', text)
        self.assertIn('Actionable candidate list after filing and validation gates: T1', text)
        self.assertNotIn('Actionable candidate list after filing and validation gates: SNDK', text)

    def test_weighting_validation_table_is_finite_and_equal_matches_plain_walk_forward(self):
        panel = synthetic_panel(n_days=220, n_assets=6)
        grid = {'lookback': [20], 'skip': [0], 'top_frac': [0.33], 'rebalance': [5]}
        wf_kwargs = {'train': 80, 'test': 20, 'warmup': 30}
        plain = walk_forward(make_xs_strategy('long_only'), panel, grid, **wf_kwargs)
        summary = compare_weighting_validation(panel, grid, **wf_kwargs)

        for weighting in WEIGHTING_METHODS:
            with self.subTest(weighting=weighting):
                row = summary['rows'][weighting]
                self.assertIsNone(row['error'])
                self.assertTrue(np.isfinite(row['sharpe']))
                self.assertTrue(np.isfinite(row['ann_return']))
                self.assertTrue(np.isfinite(row['ann_vol']))
                self.assertTrue(np.isfinite(row['max_dd']))

        self.assertAlmostEqual(
            float(summary['rows']['equal']['sharpe']),
            float(plain['oos_metrics']['sharpe']),
            places=12,
        )
        self.assertIn(summary['best_weighting'], WEIGHTING_METHODS)
        self.assertTrue(np.isfinite(summary['rows']['benchmark']['sharpe']))

    def test_validate_path_prints_sizing_table_and_self_check(self):
        panel = synthetic_panel(n_days=180, n_assets=6)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            stats = _print_wf_validation(
                panel,
                train=80,
                test=20,
                coverage=coverage_by_ticker(panel),
            )
        out = buf.getvalue()

        self.assertIsNotNone(stats)
        self.assertIn('Information ratio (active-return OOS)', out)
        self.assertIn('=== Sizing-scheme OOS validation (long-only book) ===', out)
        self.assertIn('Self-check: OK', out)


if __name__ == '__main__':
    unittest.main()
