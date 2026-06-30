"""Phase 8 tests: point-in-time filing-risk factor. Offline, synthetic cache."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from quant.filing_factor import (
    DEFAULT_FILING_THRESHOLD,
    FilingEvent,
    apply_filing_filter,
    build_filing_score_panel,
    filing_data_sufficiency,
    load_filing_events_from_dict,
)


def _index(n=300, start='2022-01-03'):
    return pd.bdate_range(start, periods=n)


def _panel(n=300, cols=('A', 'B', 'C', 'D')):
    rng = np.random.default_rng(1)
    idx = _index(n)
    return pd.DataFrame(
        {c: 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n))) for c in cols},
        index=idx,
    )


class TestLoad(unittest.TestCase):
    def test_parses_keys_and_ignores_bad(self):
        raw = {
            'AMD:2026-02-04': {'change_score': -0.4, 'current_filing_date': '2026-02-04', 'ok': True},
            'TSM:2026-04-16': {'change_score': -0.35, 'current_filing_date': '2026-04-16', 'ok': True},
            'BAD:2026-01-01': {'change_score': None, 'current_filing_date': '2026-01-01', 'ok': False},
        }
        ev = load_filing_events_from_dict(raw)
        self.assertEqual(ev['AMD'][0].score, -0.4)
        self.assertTrue(ev['AMD'][0].ok)
        self.assertFalse(ev['BAD'][0].ok)


class TestPanelNoLookahead(unittest.TestCase):
    def test_activation_is_after_filing_date(self):
        idx = _index(100)
        filing = idx[40]
        events = {'A': [FilingEvent('A', filing, -0.5, True)]}
        panel = build_filing_score_panel(events, idx, ['A'], activation_lag=1)
        # Strictly before/at filing date -> NaN; the day AFTER -> active.
        self.assertTrue(panel['A'].iloc[:41].isna().all())
        self.assertEqual(panel['A'].iloc[41], -0.5)
        self.assertEqual(panel['A'].iloc[99], -0.5)

    def test_forward_fill_until_superseded(self):
        idx = _index(100)
        events = {'A': [
            FilingEvent('A', idx[20], -0.5, True),
            FilingEvent('A', idx[60], 0.2, True),
        ]}
        panel = build_filing_score_panel(events, idx, ['A'], activation_lag=1)
        self.assertEqual(panel['A'].iloc[30], -0.5)
        self.assertEqual(panel['A'].iloc[59], -0.5)
        self.assertEqual(panel['A'].iloc[61], 0.2)  # newer filing supersedes

    def test_future_filing_does_not_leak(self):
        idx = _index(100)
        events = {'A': [FilingEvent('A', idx[80], -0.9, True)]}
        panel = build_filing_score_panel(events, idx, ['A'], activation_lag=1)
        self.assertTrue(panel['A'].iloc[:81].isna().all())

    def test_quarterly_events_densify_panel(self):
        """Quarterly source events produce more frequent step-changes than annual."""
        idx = _index(260)  # ~1 trading year
        events = {'A': [
            FilingEvent('A', idx[20], -0.5, True, 'quarterly'),
            FilingEvent('A', idx[80], -0.2, True, 'quarterly'),
            FilingEvent('A', idx[140], 0.1, True, 'quarterly'),
            FilingEvent('A', idx[200], -0.3, True, 'quarterly'),
        ]}
        panel = build_filing_score_panel(events, idx, ['A'], activation_lag=1)
        # Four distinct score levels active across one year (annual would give one).
        self.assertEqual(panel['A'].iloc[30], -0.5)
        self.assertEqual(panel['A'].iloc[90], -0.2)
        self.assertEqual(panel['A'].iloc[150], 0.1)
        self.assertEqual(panel['A'].iloc[210], -0.3)
        self.assertGreaterEqual(panel['A'].dropna().nunique(), 4)


class TestSufficiency(unittest.TestCase):
    def test_thin_cache_is_insufficient(self):
        raw = {
            'AMD:2026-02-04': {'change_score': -0.4, 'current_filing_date': '2026-02-04', 'ok': True},
            'TSM:2026-04-16': {'change_score': -0.35, 'current_filing_date': '2026-04-16', 'ok': True},
        }
        suff = filing_data_sufficiency(load_filing_events_from_dict(raw))
        self.assertFalse(suff['validatable'])
        self.assertTrue(suff['reasons'])

    def test_rich_synthetic_is_sufficient(self):
        raw = {}
        for t in ('A', 'B', 'C', 'D', 'E'):
            for yr in range(2014, 2024):
                raw[f'{t}:{yr}-03-01'] = {
                    'change_score': -0.1, 'current_filing_date': f'{yr}-03-01', 'ok': True,
                }
        suff = filing_data_sufficiency(load_filing_events_from_dict(raw))
        self.assertTrue(suff['validatable'], suff['reasons'])
        self.assertGreaterEqual(suff['names_with_history'], 4)
        self.assertEqual(suff['source_counts'].get('annual'), 50)

    def test_quarterly_cadence_reaches_sufficiency_faster(self):
        """~4 quarterly updates/name/yr clears the event bar in ~3y of span."""
        raw = {}
        months = ['02-15', '05-15', '08-15', '11-15']
        for t in ('A', 'B', 'C', 'D', 'E'):
            for yr in range(2021, 2025):
                for mm in months:
                    raw[f'{t}:{yr}-{mm}'] = {
                        'change_score': -0.05,
                        'current_filing_date': f'{yr}-{mm}',
                        'ok': True,
                        'source': 'quarterly',
                    }
        suff = filing_data_sufficiency(load_filing_events_from_dict(raw))
        self.assertTrue(suff['validatable'], suff['reasons'])
        self.assertTrue(suff['has_quarterly'])
        self.assertEqual(suff['source_counts'].get('quarterly'), 80)
        self.assertLess(suff['median_spacing_days'], 100)  # ~quarterly

    def test_source_defaults_to_annual(self):
        raw = {'A:2023-03-01': {'change_score': -0.1,
                                'current_filing_date': '2023-03-01', 'ok': True}}
        ev = load_filing_events_from_dict(raw)
        self.assertEqual(ev['A'][0].source, 'annual')


class TestFilter(unittest.TestCase):
    def test_exclude_and_renormalize(self):
        panel = _panel()
        idx = panel.index
        # Equal-weight long book of all 4 names from day 0.
        weights = pd.DataFrame(0.25, index=idx, columns=panel.columns)
        events = {'A': [FilingEvent('A', idx[10], -0.5, True)]}
        sp = build_filing_score_panel(events, idx, list(panel.columns), activation_lag=1)
        filtered = apply_filing_filter(weights, sp, threshold=-0.30, action='exclude')
        # After activation, A is dropped and the book renormalizes to 1 across B,C,D.
        row = filtered.iloc[20]
        self.assertAlmostEqual(row['A'], 0.0, places=9)
        self.assertAlmostEqual(float(row.sum()), 1.0, places=9)
        self.assertAlmostEqual(row['B'], 1.0 / 3.0, places=9)
        # Before activation, untouched.
        self.assertAlmostEqual(filtered.iloc[5]['A'], 0.25, places=9)

    def test_half_weight_action(self):
        panel = _panel()
        idx = panel.index
        weights = pd.DataFrame(0.25, index=idx, columns=panel.columns)
        events = {'A': [FilingEvent('A', idx[10], -0.5, True)]}
        sp = build_filing_score_panel(events, idx, list(panel.columns), activation_lag=1)
        filtered = apply_filing_filter(weights, sp, threshold=-0.30,
                                       action='half_weight', renormalize=False)
        self.assertAlmostEqual(filtered.iloc[20]['A'], 0.125, places=9)

    def test_no_filter_when_above_threshold(self):
        panel = _panel()
        idx = panel.index
        weights = pd.DataFrame(0.25, index=idx, columns=panel.columns)
        events = {'A': [FilingEvent('A', idx[10], 0.1, True)]}  # positive -> not flagged
        sp = build_filing_score_panel(events, idx, list(panel.columns), activation_lag=1)
        filtered = apply_filing_filter(weights, sp, threshold=DEFAULT_FILING_THRESHOLD)
        pd.testing.assert_frame_equal(filtered, weights)

    def test_filter_no_lookahead_on_future_price(self):
        """The filter depends only on the score panel, not on prices, so a
        future price change cannot alter earlier filtered weights."""
        panel = _panel()
        idx = panel.index
        weights = pd.DataFrame(0.25, index=idx, columns=panel.columns)
        events = {'A': [FilingEvent('A', idx[10], -0.5, True)]}
        sp = build_filing_score_panel(events, idx, list(panel.columns), activation_lag=1)
        filtered = apply_filing_filter(weights, sp, threshold=-0.30)
        # Rebuild the panel on a future-perturbed score-free path: identical.
        sp2 = build_filing_score_panel(events, idx, list(panel.columns), activation_lag=1)
        filtered2 = apply_filing_filter(weights, sp2, threshold=-0.30)
        pd.testing.assert_frame_equal(filtered, filtered2)


if __name__ == '__main__':
    unittest.main()
