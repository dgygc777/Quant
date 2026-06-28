"""Unit tests for risk-section extraction."""

from __future__ import annotations

import unittest

from quant.risk_extraction import (
    extract_risk_section,
    extract_risk_section_detail,
    looks_like_risk_factors,
    risk_cue_count,
)
from tenk_reader import _is_valid_annual_form


def _risk_body(prefix: str = 'Item 1A. Risk Factors') -> str:
    body = (
        ' '.join([
            'risk', 'adversely', 'could', 'uncertain', 'no assurance',
            'fluctuat', 'harm', 'may not', 'subject to',
        ] * 80)
    )
    return f'{prefix}\n{body}\nItem 1B. Unresolved Staff Comments\nrest'


class TestRiskExtraction(unittest.TestCase):
    def test_10k_heading_slice(self):
        text = _risk_body('Item 1A. Risk Factors')
        detail = extract_risk_section_detail(text, '10-K')
        self.assertTrue(detail.ok)
        self.assertIn('adversely', detail.section)
        self.assertNotIn('Item 1B', detail.section)

    def test_20f_heading_slice(self):
        text = _risk_body('Item 3. Key Information - D. Risk Factors')
        detail = extract_risk_section_detail(text, '20-F')
        self.assertTrue(detail.ok)
        self.assertIn('uncertain', detail.section)

    def test_20f_inline_item4_cross_reference_does_not_end_section(self):
        risk_text = ' '.join([
            'risk', 'adversely', 'could', 'uncertain', 'no assurance',
            'fluctuat', 'harm', 'may not', 'subject to',
        ] * 80)
        text = (
            'Item 3. Key Information - D. Risk Factors\n'
            f'{risk_text}\n'
            'For more information, see Item 4. Information on the Company.\n'
            'This sentence is still part of the risk factor section and should remain.\n'
            'Item 4. Information on the Company\n'
            'Company description should not be included.'
        )

        detail = extract_risk_section_detail(text, '20-F')

        self.assertTrue(detail.ok)
        self.assertIn('This sentence is still part of the risk factor section', detail.section)
        self.assertNotIn('Company description should not be included', detail.section)

    def test_toc_rejected(self):
        text = 'Item 1A. Risk Factors                12    Item 1B. foo'
        detail = extract_risk_section_detail(text, '10-K')
        self.assertFalse(detail.ok)

    def test_forward_looking_rejected(self):
        text = (
            'Special Note Regarding Forward-Looking Statements. Risk factors titled '
            '"foo" see Item 1A. ' + 'risk ' * 500
        )
        detail = extract_risk_section_detail(text, '10-K')
        self.assertFalse(detail.ok)

    def test_cross_reference_rejected(self):
        text = 'Risk Factors titled "Regulatory Matters" ' + ('could adversely ' * 200)
        detail = extract_risk_section_detail(text, '10-K')
        self.assertFalse(detail.ok)
        self.assertFalse(looks_like_risk_factors(detail.section, detail.quality_score))

    def test_lowercase_inline_rejected(self):
        text = 'risk factors related to geography ' + ('uncertain harm ' * 200)
        detail = extract_risk_section_detail(text, '10-K')
        self.assertFalse(detail.ok)

    def test_compat_wrapper(self):
        s = '...Item 1A Risk Factors aaa could adversely affect ... Item 1B...'
        out = extract_risk_section(s, '10-K')
        self.assertIn('aaa could adversely', out)
        self.assertNotIn('item 1b', out.lower())

    def test_amendment_filter(self):
        self.assertFalse(_is_valid_annual_form('10-K/A', include_amendments=False))
        self.assertTrue(_is_valid_annual_form('10-K', include_amendments=False))
        self.assertTrue(_is_valid_annual_form('10-K/A', include_amendments=True))


if __name__ == '__main__':
    unittest.main()
