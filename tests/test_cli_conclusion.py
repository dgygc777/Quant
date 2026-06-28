"""Unit tests for CLI summary text."""

from __future__ import annotations

import contextlib
import io
import unittest

import numpy as np
import pandas as pd

from cli import _buy_strategy_conclusion, _print_ranked_candidate_risk, _ranked_risk_pool
from quant.data_quality import coverage_by_ticker


class TestCliConclusion(unittest.TestCase):
    def test_risk_pool_expands_beyond_two_name_long_leg(self):
        tickers = ['AMAT', 'ASX', 'COHR', 'INTC', 'LRCX', 'GFS', 'ARM', 'AMD', 'MRVL', 'ALAB']
        scores = pd.Series(
            [0.30, 0.25, 0.22, 0.19, 0.17, 0.14, 0.12, 0.08, 0.02, -0.04],
            index=tickers,
        )
        weights = pd.Series(0.0, index=tickers)
        weights[['AMAT', 'ASX']] = 0.5

        active_longs, risk_pool = _ranked_risk_pool(weights, scores)

        self.assertEqual(active_longs, ['AMAT', 'ASX'])
        self.assertEqual(risk_pool, ['AMAT', 'ASX', 'COHR', 'INTC', 'LRCX', 'GFS'])

    def test_positive_validation_conclusion_includes_gated_candidates(self):
        weights = pd.Series({
            'TER': 1 / 3,
            'MU': 1 / 3,
            'AMAT': 1 / 3,
            'INTC': -1 / 3,
        })
        scores = pd.Series({
            'TER': 0.12,
            'MU': 0.08,
            'AMAT': 0.05,
            'INTC': -0.04,
        })
        tenk_df = pd.DataFrame([
            {'ticker': 'TER', 'final_action': 'BUY', 'risk_flag': False, 'tenk_ok': True},
            {'ticker': 'MU', 'final_action': 'BUY', 'risk_flag': True, 'tenk_ok': True},
            {'ticker': 'AMAT', 'final_action': 'BUY', 'risk_flag': False, 'tenk_ok': False},
            {'ticker': 'INTC', 'final_action': 'AVOID', 'risk_flag': False, 'tenk_ok': True},
        ])
        validation_stats = {
            'folds': 9,
            'oos_sharpe': 1.23,
            'oos_ann_return': 0.147,
            'oos_max_dd': -0.182,
            'benchmark_oos_sharpe': 0.72,
            'active_oos_sharpe': 0.51,
            'validation_verdict': 'EDGE',
        }
        sizing_stats = {
            'max_rc_ticker': 'TER',
            'max_rc_pct': 0.55,
            'min_rc_ticker': 'AMAT',
            'min_rc_pct': 0.21,
            'equal_ann_vol': 0.31,
            'rc_pct_by_ticker': {'TER': 0.55, 'MU': 0.24, 'AMAT': 0.21},
        }

        text = _buy_strategy_conclusion(
            'semis',
            {'momentum_preset': 'mom_10d', 'lookback': 10, 'skip': 0},
            weights,
            scores,
            validation_stats,
            tenk_df,
            sizing_stats,
        )

        self.assertTrue(text.startswith('Validation verdict EDGE: strategy OOS Sharpe 1.23'))
        self.assertIn('TER +12.0%, MU +8.0%, AMAT +5.0%', text)
        self.assertIn(
            'MU flagged - verify filing before sizing',
            text,
        )
        self.assertIn('AMAT (no filing data)', text)
        self.assertIn('TER is a disproportionate risk contributor', text)
        self.assertIn('equal-dollar sizing is inappropriate - use risk-parity sizing instead', text)
        self.assertIn('Actionable candidate list after filing and validation gates: TER', text)
        self.assertTrue(text.endswith('Standing caveat: this is a curated-universe research overlay, not a validated edge.'))

    def test_negative_validation_outputs_research_only_without_actionable_buy_language(self):
        weights = pd.Series({
            'TER': 1 / 6,
            'MU': 1 / 6,
            'AMAT': 1 / 6,
            'SNDK': 1 / 6,
            'INTC': 1 / 6,
            'ASX': 1 / 6,
        })
        scores = pd.Series({
            'TER': 0.18,
            'MU': 0.16,
            'AMAT': 0.14,
            'SNDK': 0.12,
            'INTC': 0.10,
            'ASX': 0.08,
        })
        tenk_df = pd.DataFrame([
            {'ticker': 'TER', 'final_action': 'BUY', 'risk_flag': True, 'tenk_ok': True},
            {'ticker': 'MU', 'final_action': 'BUY', 'risk_flag': False, 'tenk_ok': True},
            {'ticker': 'AMAT', 'final_action': 'BUY', 'risk_flag': False, 'tenk_ok': True},
            {'ticker': 'SNDK', 'final_action': 'BUY', 'risk_flag': False, 'tenk_ok': True},
            {'ticker': 'INTC', 'final_action': 'BUY', 'risk_flag': True, 'tenk_ok': True},
            {'ticker': 'ASX', 'final_action': 'BUY', 'risk_flag': False, 'tenk_ok': True},
        ])
        validation_stats = {
            'folds': 10,
            'oos_sharpe': -0.35,
            'oos_ann_return': -0.021,
            'oos_max_dd': -0.284,
            'benchmark_oos_sharpe': 0.18,
            'active_oos_sharpe': -0.53,
            'validation_verdict': 'FAILS',
        }
        sizing_stats = {
            'max_rc_ticker': 'SNDK',
            'max_rc_pct': 0.257,
            'min_rc_ticker': 'ASX',
            'min_rc_pct': 0.11,
            'equal_ann_vol': 0.29,
            'rc_pct_by_ticker': {
                'TER': 0.14,
                'MU': 0.17,
                'AMAT': 0.16,
                'SNDK': 0.257,
                'INTC': 0.163,
                'ASX': 0.11,
            },
        }

        text = _buy_strategy_conclusion(
            'semis',
            {'momentum_preset': 'mom_10d', 'lookback': 10, 'skip': 0},
            weights,
            scores,
            validation_stats,
            tenk_df,
            sizing_stats,
        )
        lower = text.lower()

        self.assertTrue(text.startswith('Validation verdict FAILS: the ranking has no validated benchmark-relative edge'))
        self.assertIn('strategy OOS Sharpe -0.35 vs benchmark 0.18', text)
        self.assertIn('max DD -28.4%', text)
        self.assertIn('Names to research: TER, MU, AMAT, SNDK, INTC, ASX.', text)
        self.assertIn('SNDK is a disproportionate risk contributor (25.7% vs equal-dollar 16.7%)', text)
        self.assertIn('equal-dollar sizing is inappropriate - use risk-parity sizing instead', text)
        self.assertIn('TER flagged - verify filing before sizing', text)
        self.assertIn('INTC flagged - verify filing before sizing', text)
        self.assertIn('SNDK momentum-ranked but contradicted by negative validation regime, risk-concentration flag', text)
        self.assertNotIn('buy', lower)
        self.assertNotIn('buy list', lower)
        self.assertNotIn('actionable', lower)
        self.assertNotIn('practical buy', lower)

    def test_sparse_history_excluded_from_risk_sizing_and_actionable_list(self):
        rng = np.random.default_rng(11)
        idx = pd.bdate_range('2025-01-01', periods=100)
        full_names = ['AMAT', 'ASX', 'COHR', 'INTC', 'LRCX', 'GFS']
        full_rets = rng.normal(0.0005, 0.015, size=(len(idx), len(full_names)))
        panel = pd.DataFrame(
            100.0 * np.exp(np.cumsum(full_rets, axis=0)),
            index=idx,
            columns=full_names,
        )
        sndk = pd.Series(np.nan, index=idx, name='SNDK')
        sndk.iloc[-14:] = 100.0 * np.exp(np.cumsum(rng.normal(0.001, 0.02, size=14)))
        panel['SNDK'] = sndk
        coverage = coverage_by_ticker(panel)
        scores = pd.Series({
            'SNDK': 0.50,
            'AMAT': 0.42,
            'ASX': 0.36,
            'COHR': 0.31,
            'INTC': 0.28,
            'LRCX': 0.22,
            'GFS': 0.18,
        })
        weights = pd.Series(0.0, index=scores.index)
        weights[['SNDK', 'AMAT']] = 0.5

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            sizing_stats = _print_ranked_candidate_risk(
                panel,
                weights,
                scores,
                {'momentum_preset': 'mom_10d', 'lookback': 10, 'skip': 0},
                years=1,
                coverage=coverage,
            )

        self.assertIn('SNDK', sizing_stats['speculative'])
        self.assertNotIn('SNDK', sizing_stats['risk_sized_pool'])
        self.assertIn('INSUFFICIENT HISTORY - speculative only', buf.getvalue())

        tenk_df = pd.DataFrame([
            {'ticker': ticker, 'final_action': 'BUY', 'risk_flag': False, 'tenk_ok': True}
            for ticker in scores.index
        ])
        validation_stats = {
            'folds': 10,
            'oos_sharpe': 1.10,
            'oos_ann_return': 0.12,
            'oos_max_dd': -0.11,
            'benchmark_oos_sharpe': 0.70,
            'active_oos_sharpe': 0.40,
            'validation_verdict': 'EDGE',
        }
        text = _buy_strategy_conclusion(
            'custom',
            {'momentum_preset': 'mom_10d', 'lookback': 10, 'skip': 0},
            weights,
            scores,
            validation_stats,
            tenk_df,
            sizing_stats,
            coverage=coverage,
        )

        self.assertIn('insufficient-history speculative bucket: SNDK 14%', text)
        self.assertIn('Actionable candidate list after filing and validation gates: AMAT', text)
        self.assertNotIn('Actionable candidate list after filing and validation gates: SNDK', text)


if __name__ == '__main__':
    unittest.main()
