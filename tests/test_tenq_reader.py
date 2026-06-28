"""Synthetic 10-Q extraction, matching, and cache tests."""

from __future__ import annotations

import json
import tempfile
import unittest

from quant.risk_extraction import (
    _is_toc_slice,
    extract_quarterly_section_candidates,
    extract_risk_section_detail,
    select_quarterly_comparison_sections,
)
from quant.tenk_cache import load_tenk_scores, load_tenq_scores
from tenk_reader import _select_quarterly_pair, load_cache, save_cache


def _risk_words(mult: int = 120) -> str:
    return ' '.join([
        'risk', 'adversely', 'could', 'uncertain', 'no assurance',
        'fluctuat', 'harm', 'may not', 'subject to',
    ] * mult)


def _mda_words(mult: int = 70) -> str:
    return ' '.join([
        'revenue', 'demand', 'margin', 'liquidity', 'customer',
        'inventory', 'supply', 'cash', 'operations',
    ] * mult)


def _tenq_text(risk_text: str, *, mda_text: str | None = None) -> str:
    mda_text = mda_text if mda_text is not None else _mda_words()
    return f"""
PART I
Item 1. Financial Statements
Condensed statements.
Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations
{mda_text}
Item 3. Quantitative and Qualitative Disclosures About Market Risk
Market risk text.
Item 4. Controls and Procedures
Controls text.

PART II
Item 1. Legal Proceedings
Legal text.
Item 1A. Risk Factors
{risk_text}
Item 2. Unregistered Sales of Equity Securities
This Part II Item 2 text should not be used as MD&A.
    """


def _tenq_text_with_toc(risk_text: str, *, mda_text: str | None = None) -> str:
    mda_text = mda_text if mda_text is not None else _mda_words()
    return f"""
TABLE OF CONTENTS
PART I
Item 1:     Financial Statements            4
Item\xa02:     Management’s Discussion and Analysis of Financial Condition and Results of Operations            29
Item 3:     Quantitative and Qualitative Disclosures About Market Risk            44
Item 4:     Controls and Procedures            45
PART II
Item 1:     Legal Proceedings            48
Item 1A:     Risk Factors            39
Item 2:     Unregistered Sales of Equity Securities            51

PART I
Item 1. Financial Statements
Condensed statements.
Item 2. Management’s Discussion and Analysis of Financial Condition and Results of Operations
{mda_text}
Item 3. Quantitative and Qualitative Disclosures About Market Risk
Market risk text.

PART II
Item 1. Legal Proceedings
Legal text.
Item 1A. Risk Factors
{risk_text}
Item 2. Unregistered Sales of Equity Securities
This Part II Item 2 text should not be used as MD&A.
"""


def _tenq_text_with_toc_no_body_parts(risk_text: str, *, mda_text: str | None = None) -> str:
    mda_text = mda_text if mda_text is not None else _mda_words()
    return f"""
Table of Contents
Part I
Item 1. Financial Statements            4
Item 2. Management’s Discussion and Analysis of Financial Condition and Results of Operations            29
Item 3. Quantitative and Qualitative Disclosures About Market Risk            38
Part II
Item 1. Legal Proceedings            39
Item 1A. Risk Factors            39
Item 2. Unregistered Sales of Equity Securities            44

Item 1. Financial Statements
Condensed statements and notes.
Item 2. Management’s Discussion and Analysis of Financial Condition and Results of Operations
{mda_text}
Item 3. Quantitative and Qualitative Disclosures About Market Risk
Market risk text.

Item 1. Legal Proceedings
Legal text.
Item 1A. Risk Factors
{risk_text}
Item 2. Unregistered Sales of Equity Securities
Equity text.
"""


def _tenq_text_with_late_reference_index(risk_text: str, *, mda_text: str | None = None) -> str:
    mda_text = mda_text if mda_text is not None else _mda_words(150)
    return f"""
Forward-Looking Statements
General introduction.

Table of Contents

Management's Discussion and Analysis

Overview
This is the real operating discussion, not the reference index.
{mda_text}

Risk Factors and Other Key Information

Risk Factors
{risk_text}

Quantitative and Qualitative Disclosures About Market Risk
Market risk body.

Controls and Procedures
Controls body.

Form 10-Q Cross-Reference Index

Reference Index
Item Number                   Item
Part I - Financial
Information
Item 1.                       Financial Statements            Pages3-25
                                Management's Discussion and
Item 2.                       Analysis of Financial
                                Condition and Results of
                                Operations
                                Results of operations           Pages28-36
                                Liquidity and capital           Pages36-38
                                resources
                                Critical accounting             Not applicable
                                estimates
                                Quantitative and Qualitative
Item 3.                       Disclosures About Market        Page39
                                Risk
Item 4.                       Controls and Procedures         Page39
Part II - Other
Information
Item 1.                       Legal Proceedings               Pages20-23
Item 1A.                      Risk Factors                    Page39
Item 2.                       Securities and Use of           Page39
                                Proceeds
"""


class TestTenQReader(unittest.TestCase):
    def test_10q_risk_section_returns_part_ii_item_1a_when_full(self):
        text = _tenq_text(_risk_words())

        detail = extract_risk_section_detail(text, '10-Q')

        self.assertEqual(detail.section_used, 'risk_factors')
        self.assertIn('Item 1A. Risk Factors', detail.section)
        self.assertIn('adversely', detail.section)
        self.assertNotIn('Unregistered Sales', detail.section)

    def test_short_risk_stub_falls_back_to_mda_for_comparison(self):
        stub = 'There have been no material changes to our risk factors.'
        prior = _tenq_text(stub, mda_text='prior mda ' + _mda_words())
        current = _tenq_text(stub, mda_text='current mda ' + _mda_words())

        selection = select_quarterly_comparison_sections(prior, current)

        self.assertTrue(selection.ok)
        self.assertEqual(selection.section_used, 'mda_both')
        self.assertIn('prior mda', selection.prior_detail.section)
        self.assertIn('current mda', selection.current_detail.section)

    def test_coordination_uses_mda_when_only_one_item_1a_is_full(self):
        prior = _tenq_text('There have been no material changes to our risk factors.')
        current = _tenq_text(_risk_words())

        selection = select_quarterly_comparison_sections(prior, current)

        self.assertTrue(selection.ok)
        self.assertEqual(selection.section_used, 'mda_both')
        self.assertNotEqual(selection.current_detail.section_used, 'risk_factors')
        self.assertIn('Management', selection.current_detail.section)

    def test_part_aware_item_2_resolves_part_i_mda(self):
        text = _tenq_text('There have been no material changes to our risk factors.')

        candidates = extract_quarterly_section_candidates(text)
        mda = candidates['mda'].section

        self.assertIn('Management', mda)
        self.assertIn('revenue', mda)
        self.assertNotIn('Unregistered Sales', mda)

    def test_toc_not_matched_for_10q(self):
        text = _tenq_text_with_toc(_risk_words(), mda_text='actual mda body ' + _mda_words())
        toc_slice = (
            'Item 2:     Management’s Discussion and Analysis of Financial Condition '
            'and Results of Operations            29\n'
            'Item 3:     Quantitative and Qualitative Disclosures About Market Risk            44\n'
            'Item 1A:     Risk Factors            39\n'
        )

        candidates = extract_quarterly_section_candidates(text)
        selection = select_quarterly_comparison_sections(text, text)

        self.assertTrue(_is_toc_slice(toc_slice))
        self.assertGreater(candidates['mda'].candidate_lengths['mda'], 1000)
        self.assertGreater(candidates['risk_factors'].candidate_lengths['risk_factors'], 1500)
        self.assertIn('actual mda body', candidates['mda'].section)
        self.assertIn('adversely', candidates['risk_factors'].section)
        self.assertNotIn('            29', candidates['mda'].heading_preview)
        self.assertNotIn('            39', candidates['risk_factors'].heading_preview)
        self.assertTrue(selection.ok)
        self.assertEqual(selection.section_used, 'risk_factors')

    def test_10q_global_item_fallback_when_body_part_markers_missing(self):
        text = _tenq_text_with_toc_no_body_parts(
            _risk_words(),
            mda_text='fallback mda body ' + _mda_words(),
        )

        candidates = extract_quarterly_section_candidates(text)
        selection = select_quarterly_comparison_sections(text, text)

        self.assertGreater(candidates['mda'].candidate_lengths['mda'], 1000)
        self.assertGreater(candidates['risk_factors'].candidate_lengths['risk_factors'], 1500)
        self.assertIn('fallback mda body', candidates['mda'].section)
        self.assertIn('adversely', candidates['risk_factors'].section)
        self.assertTrue(selection.ok)
        self.assertEqual(selection.section_used, 'risk_factors')

    def test_late_reference_index_does_not_beat_standalone_mda_body(self):
        stub = 'The risks described in our Form 10-K could materially affect us.'
        text = _tenq_text_with_late_reference_index(stub, mda_text='real mda body ' + _mda_words(180))
        reference_slice = """
Reference Index
Item Number                   Item
Item 1.                       Financial Statements            Pages3-25
Item 3.                       Disclosures About Market        Page39
Item 4.                       Controls and Procedures         Page39
"""

        candidates = extract_quarterly_section_candidates(text)
        selection = select_quarterly_comparison_sections(text, text)

        self.assertTrue(_is_toc_slice(reference_slice))
        self.assertGreater(candidates['mda'].candidate_lengths['mda'], 2_000)
        self.assertIn('real mda body', candidates['mda'].section)
        self.assertNotIn('Reference Index', candidates['mda'].section[:300])
        self.assertTrue(selection.ok)
        self.assertEqual(selection.section_used, 'mda_both')

    def test_quarterly_matching_prefers_year_over_year(self):
        records = [
            {'filing_date': '2025-08-01', 'period': '2025-06-30', 'form': '10-Q', 'full_text': 'latest'},
            {'filing_date': '2025-05-02', 'period': '2025-03-31', 'form': '10-Q', 'full_text': 'prior quarter'},
            {'filing_date': '2024-08-02', 'period': '2024-06-30', 'form': '10-Q', 'full_text': 'year ago'},
        ]

        pair = _select_quarterly_pair(records)

        self.assertTrue(pair['ok'])
        self.assertEqual(pair['prior_filing_date'], '2024-08-02')
        self.assertEqual(pair['comparison_type'], 'year_over_year')
        self.assertEqual(pair['match_method'], 'fiscal_period')

    def test_quarterly_matching_day_window_then_sequential(self):
        no_period_records = [
            {'filing_date': '2025-08-01', 'form': '10-Q', 'full_text': 'latest'},
            {'filing_date': '2025-05-02', 'form': '10-Q', 'full_text': 'prior quarter'},
            {'filing_date': '2024-08-03', 'form': '10-Q', 'full_text': 'year ago'},
        ]
        pair = _select_quarterly_pair(no_period_records)
        self.assertEqual(pair['prior_filing_date'], '2024-08-03')
        self.assertEqual(pair['comparison_type'], 'year_over_year')
        self.assertEqual(pair['match_method'], 'day_window')

        sequential_records = [
            {'filing_date': '2025-08-01', 'form': '10-Q', 'full_text': 'latest'},
            {'filing_date': '2025-05-02', 'form': '10-Q', 'full_text': 'prior quarter'},
        ]
        pair = _select_quarterly_pair(sequential_records)
        self.assertEqual(pair['prior_filing_date'], '2025-05-02')
        self.assertEqual(pair['comparison_type'], 'sequential')
        self.assertEqual(pair['match_method'], 'sequential')

    def test_cache_roundtrip_source_fields_and_old_entries(self):
        with tempfile.NamedTemporaryFile('w+', suffix='.json') as f:
            cache = {
                'AAA:2025-02-01': {
                    'change_score': -0.2,
                    'summary': 'old annual schema',
                    'ok': True,
                },
                'BBB:2025-05-01': {
                    'source': 'quarterly',
                    'comparison_type': 'year_over_year',
                    'section_used': 'mda_both',
                    'match_method': 'day_window',
                    'change_score': -0.1,
                    'summary': 'quarterly schema',
                    'ok': True,
                },
            }
            save_cache(cache, f.name)
            raw = load_cache(f.name)
            self.assertEqual(raw['BBB:2025-05-01']['source'], 'quarterly')

            annual = load_tenk_scores(f.name)
            quarterly = load_tenq_scores(f.name)

            self.assertEqual(annual['AAA'].source, 'annual')
            self.assertEqual(annual['AAA'].change_score, -0.2)
            self.assertEqual(quarterly['BBB'].source, 'quarterly')
            self.assertEqual(quarterly['BBB'].section_used, 'mda_both')
            self.assertEqual(quarterly['BBB'].match_method, 'day_window')

    def test_source_filter_before_newest_reduction(self):
        with tempfile.NamedTemporaryFile('w+', suffix='.json', delete=True) as f:
            json.dump({
                'XYZ:2025-02-01': {
                    'change_score': -0.4,
                    'summary': 'annual score',
                    'ok': True,
                },
                'XYZ:2025-08-01': {
                    'source': 'quarterly',
                    'change_score': 0.3,
                    'summary': 'newer quarterly score',
                    'ok': True,
                },
            }, f)
            f.flush()

            annual = load_tenk_scores(f.name)

            self.assertIn('XYZ', annual)
            self.assertEqual(annual['XYZ'].filing_date.isoformat(), '2025-02-01')
            self.assertEqual(annual['XYZ'].source, 'annual')
            self.assertEqual(annual['XYZ'].change_score, -0.4)


if __name__ == '__main__':
    unittest.main()
