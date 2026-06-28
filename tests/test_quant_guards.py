"""Unit tests for parameter validation and tenk cache integration."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quant.combined_signal import attach_tenk_metadata, CombinedParams
from quant.params import validate_top_frac, validate_xs_params
from quant.tenk_cache import load_tenk_scores
from quant.universes import validate_universe_size


class TestQuantGuards(unittest.TestCase):
    def test_top_frac_bounds(self):
        with self.assertRaises(ValueError):
            validate_top_frac(0.0)
        with self.assertRaises(ValueError):
            validate_top_frac(-0.1)
        with self.assertRaises(ValueError):
            validate_top_frac(0.51)

    def test_xs_params(self):
        with self.assertRaises(ValueError):
            validate_xs_params(top_frac=0.25, rebalance=0)
        with self.assertRaises(ValueError):
            validate_xs_params(top_frac=0.25, rebalance=5, short_window=0)
        with self.assertRaises(ValueError):
            validate_xs_params(top_frac=0.25, rebalance=5, lookback=0)

    def test_universe_size_invalid_top_frac(self):
        with self.assertRaises(ValueError):
            validate_universe_size(['A', 'B', 'C', 'D'], 0.0)

    def test_load_tenk_scores_filters_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'tenk_cache.json'
            path.write_text(json.dumps({
                'NVDA:2026-02-25': {
                    'change_score': -0.35,
                    'ok': True,
                    'current_filing_date': '2026-02-25',
                },
                'AMD:2026-02-04': {
                    'change_score': None,
                    'ok': False,
                    'current_filing_date': '2026-02-04',
                },
            }), encoding='utf-8')
            scores = load_tenk_scores(str(path))
            self.assertIn('NVDA', scores)
            self.assertNotIn('AMD', scores)

    def test_risk_flag_excludes_clean_actionable(self):
        df = pd.DataFrame([
            {'ticker': 'INTC', 'final_action': 'BUY', 'xs_score': 0.1, 'xs_leg': 'LONG',
             'price': 1, 'z': 0, 'mr_signal': 'WAIT', 'momentum': 0.1, 'mom_signal': 'BUY',
             'momentum_preset': 'mom_10d', 'reason': 'ok'},
            {'ticker': 'MU', 'final_action': 'BUY', 'xs_score': 0.2, 'xs_leg': 'LONG',
             'price': 1, 'z': 0, 'mr_signal': 'WAIT', 'momentum': 0.2, 'mom_signal': 'BUY',
             'momentum_preset': 'mom_10d', 'reason': 'ok'},
        ])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'tenk_cache.json'
            path.write_text(json.dumps({
                'INTC:2026-01-23': {'change_score': -0.35, 'ok': True, 'current_filing_date': '2026-01-23'},
                'MU:2025-10-03': {'change_score': -0.15, 'ok': True, 'current_filing_date': '2025-10-03'},
            }), encoding='utf-8')
            out = attach_tenk_metadata(df, cache_path=str(path), risk_threshold=-0.30)
            clean = out[(out['final_action'] == 'BUY') & ~out['risk_flag']]['ticker'].tolist()
            conflict = out[out['risk_flag']]['ticker'].tolist()
            self.assertEqual(clean, ['MU'])
            self.assertEqual(conflict, ['INTC'])

    def test_quarterly_score_flags_risk_and_20f_only_is_labeled(self):
        df = pd.DataFrame([
            {'ticker': 'AMAT', 'final_action': 'BUY', 'xs_score': 0.2, 'xs_leg': 'LONG',
             'price': 1, 'z': 0, 'mr_signal': 'WAIT', 'momentum': 0.2, 'mom_signal': 'BUY',
             'momentum_preset': 'mom_10d', 'reason': 'ok'},
            {'ticker': 'ASX', 'final_action': 'BUY', 'xs_score': 0.1, 'xs_leg': 'LONG',
             'price': 1, 'z': 0, 'mr_signal': 'WAIT', 'momentum': 0.1, 'mom_signal': 'BUY',
             'momentum_preset': 'mom_10d', 'reason': 'ok'},
        ])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'tenk_cache.json'
            path.write_text(json.dumps({
                'AMAT:2025-12-12': {
                    'source': 'annual',
                    'change_score': 0.05,
                    'ok': True,
                    'current_filing_date': '2025-12-12',
                    'form': '10-K',
                },
                'AMAT:2026-05-21': {
                    'source': 'quarterly',
                    'change_score': -0.35,
                    'ok': True,
                    'current_filing_date': '2026-05-21',
                    'form': '10-Q',
                    'section_used': 'risk_factors',
                },
                'ASX:2026-04-01': {
                    'source': 'annual',
                    'change_score': 0.15,
                    'ok': True,
                    'current_filing_date': '2026-04-01',
                    'form': '20-F',
                },
            }), encoding='utf-8')

            out = attach_tenk_metadata(df, cache_path=str(path), risk_threshold=-0.30)

            amat = out.set_index('ticker').loc['AMAT']
            asx = out.set_index('ticker').loc['ASX']
            self.assertEqual(float(amat['tenq_score']), -0.35)
            self.assertTrue(bool(amat['risk_flag']))
            self.assertEqual(asx['filing_report'], '20-F only (no 10-Q)')
            self.assertFalse(bool(asx['risk_flag']))


if __name__ == '__main__':
    unittest.main()
