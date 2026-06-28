#!/usr/bin/env python3
"""
10-K / 20-F risk-factor change signal via SEC EDGAR + Anthropic.

Qualitative research overlay — not a statistically validated factor.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone

import anthropic
import edgar
import pandas as pd

from quant.risk_extraction import (
    RISK_CAP,
    RiskExtraction,
    extract_risk_section,
    extract_risk_section_detail,
    looks_like_risk_factors,
    risk_cue_count,
    select_quarterly_comparison_sections,
)

def _configure_edgar() -> None:
    try:
        import hishel
        if not hasattr(hishel, 'FileStorage'):
            edgar.httpclient.CACHE_ENABLED = False
            edgar.httpclient.close_clients()
    except ImportError:
        edgar.httpclient.CACHE_ENABLED = False
        edgar.httpclient.close_clients()


_configure_edgar()
edgar.set_identity('Dean Chen dean@example.com')

CACHE_PATH = 'tenk_cache.json'
DEFAULT_MODEL = 'claude-haiku-4-5-20251001'
BASE_ANNUAL_FORMS = frozenset({'10-K', '20-F'})
AMENDMENT_FORMS = frozenset({'10-K/A', '20-F/A'})
QUARTERLY_FORMS = frozenset({'10-Q'})

FALLBACK_TICKERS = [
    'NVDA', 'AMD', 'AVGO', 'QCOM', 'MU', 'INTC', 'MRVL', 'AMAT', 'LRCX', 'KLAC',
    'TXN', 'ADI', 'MCHP', 'ON', 'COHR', 'TSM', 'ASML',
]

EXTRACTION_FAIL_SUMMARY = (
    'extraction unreliable — risk section not found '
    '(likely 20-F structure or section mismatch)'
)


def _resolve_tickers(tickers: list[str], universe: str | None) -> list[str]:
    if tickers:
        return [t.upper() for t in tickers]
    if universe:
        from quant.universes import get_universe
        resolved = get_universe(universe)
        print(f"Universe '{universe}' ({len(resolved)}): {', '.join(resolved)}")
        return resolved
    return _default_tickers()


def _default_tickers() -> list[str]:
    try:
        from quant.universes import UNIVERSE_PRESETS
        return list(UNIVERSE_PRESETS['semis'])
    except Exception:
        pass
    try:
        from quant.universes import DEFAULT_UNIVERSE
        return list(DEFAULT_UNIVERSE)
    except Exception:
        return list(FALLBACK_TICKERS)


def _filing_text(filing) -> str:
    for method in ('text', 'markdown'):
        fn = getattr(filing, method, None)
        if callable(fn):
            try:
                out = fn()
                if out:
                    return str(out)
            except Exception:
                continue
    try:
        obj = filing.obj()
        if obj is not None:
            if hasattr(obj, 'text') and callable(obj.text):
                out = obj.text()
                if out:
                    return str(out)
            if hasattr(obj, 'markdown') and callable(obj.markdown):
                out = obj.markdown()
                if out:
                    return str(out)
    except Exception:
        pass
    return ''


def _filing_date_str(filing) -> str:
    fd = getattr(filing, 'filing_date', None)
    return str(fd) if fd is not None else 'unknown'


def _filing_form(filing) -> str:
    form = getattr(filing, 'form', None)
    return str(form).upper() if form else '10-K'


def _filing_period_str(filing) -> str | None:
    for attr in (
        'period_of_report',
        'period',
        'report_date',
        'period_end',
        'document_period_end_date',
    ):
        val = getattr(filing, attr, None)
        if callable(val):
            try:
                val = val()
            except Exception:
                val = None
        if val:
            return str(val)
    return None


def _parse_date(raw: str | None):
    if not raw or str(raw).lower() == 'unknown':
        return None
    try:
        return datetime.strptime(str(raw)[:10], '%Y-%m-%d').date()
    except ValueError:
        return None


def _date_quarter(dt) -> int:
    return (dt.month - 1) // 3 + 1


def _iter_recent_filings(filings, max_n: int = 12):
    if filings is None:
        return
    try:
        recent = filings.latest(max_n)
    except Exception:
        recent = filings
    if hasattr(recent, 'filing_date') and hasattr(recent, 'accession_no'):
        yield recent
        return
    try:
        n = len(recent)
    except TypeError:
        return
    for i in range(min(n, max_n)):
        yield recent[i]


def _is_valid_annual_form(form: str, include_amendments: bool) -> bool:
    form = str(form or '').upper().strip()
    if include_amendments:
        return form in BASE_ANNUAL_FORMS or form in AMENDMENT_FORMS
    return form in BASE_ANNUAL_FORMS


def _is_valid_quarterly_form(form: str) -> bool:
    return str(form or '').upper().strip() in QUARTERLY_FORMS


def _quarterly_record_from_filing(filing) -> dict:
    return {
        'filing_date': _filing_date_str(filing),
        'full_text': _filing_text(filing),
        'form': _filing_form(filing),
        'period': _filing_period_str(filing),
    }


def _select_quarterly_pair(records: list[dict]) -> dict:
    """Choose latest 10-Q and its seasonality-controlled comparison filing."""
    usable = [
        dict(record)
        for record in records
        if record.get('full_text') and _parse_date(record.get('filing_date')) is not None
    ]
    usable.sort(key=lambda record: _parse_date(record.get('filing_date')), reverse=True)
    if len(usable) < 2:
        return {
            'ok': False,
            'reason': 'insufficient filings',
            'records': usable,
        }

    latest = usable[0]
    latest_date = _parse_date(latest.get('filing_date'))
    latest_period = _parse_date(latest.get('period'))
    prior = None
    match_method = None
    comparison_type = None

    if latest_period is not None:
        fiscal_matches = []
        for record in usable[1:]:
            period = _parse_date(record.get('period'))
            if period is None:
                continue
            if period.year == latest_period.year - 1 and _date_quarter(period) == _date_quarter(latest_period):
                fiscal_matches.append(record)
        if fiscal_matches:
            target = latest_period.replace(year=latest_period.year - 1)
            prior = min(
                fiscal_matches,
                key=lambda record: abs((_parse_date(record.get('period')) - target).days),
            )
            match_method = 'fiscal_period'
            comparison_type = 'year_over_year'

    if prior is None and latest_date is not None:
        target = latest_date - timedelta(days=365)
        window_matches = []
        for record in usable[1:]:
            filing_date = _parse_date(record.get('filing_date'))
            if filing_date is None:
                continue
            delta = (latest_date - filing_date).days
            if 300 <= delta <= 430:
                window_matches.append(record)
        if window_matches:
            prior = min(
                window_matches,
                key=lambda record: abs((_parse_date(record.get('filing_date')) - target).days),
            )
            match_method = 'day_window'
            comparison_type = 'year_over_year'

    if prior is None:
        prior = usable[1]
        match_method = 'sequential'
        comparison_type = 'sequential'

    return {
        'ok': True,
        'current_filing_date': latest.get('filing_date'),
        'prior_filing_date': prior.get('filing_date'),
        'current_full_text': latest.get('full_text', ''),
        'prior_full_text': prior.get('full_text', ''),
        'current_form': latest.get('form', '10-Q'),
        'prior_form': prior.get('form', '10-Q'),
        'current_period': latest.get('period'),
        'prior_period': prior.get('period'),
        'comparison_type': comparison_type,
        'match_method': match_method,
    }


def fetch_two_annuals(
    ticker: str,
    *,
    include_amendments: bool = False,
) -> list[tuple[str, str, str]]:
    """Return up to two (filing_date, full_text, form) base annual filings, newest first."""
    try:
        company = edgar.Company(ticker)
        forms = ['10-K', '20-F', '10-K/A', '20-F/A'] if include_amendments else ['10-K', '20-F']
        filings = company.get_filings(form=forms)
        if filings is None:
            return []

        candidates: list[tuple[str, str, str, str]] = []
        for filing in _iter_recent_filings(filings, max_n=12):
            form = _filing_form(filing)
            if not _is_valid_annual_form(form, include_amendments):
                continue
            text = _filing_text(filing)
            if text:
                candidates.append((_filing_date_str(filing), text, form, form))

        candidates.sort(key=lambda x: x[0], reverse=True)
        out = [(d, t, f) for d, t, f, _ in candidates[:2]]
        time.sleep(0.3)
        return out
    except Exception:
        return []


def fetch_two_quarterlies(ticker: str) -> dict:
    """Return latest 10-Q and seasonality-controlled comparison metadata."""
    try:
        company = edgar.Company(ticker)
        filings = company.get_filings(form=['10-Q'])
        if filings is None:
            return {'ok': False, 'reason': 'insufficient filings', 'records': []}

        records: list[dict] = []
        for filing in _iter_recent_filings(filings, max_n=16):
            form = _filing_form(filing)
            if not _is_valid_quarterly_form(form):
                continue
            record = _quarterly_record_from_filing(filing)
            if record.get('full_text'):
                records.append(record)
        time.sleep(0.3)
        return _select_quarterly_pair(records)
    except Exception as exc:
        return {'ok': False, 'reason': f'fetch error: {exc}', 'records': []}


def debug_extract(
    ticker: str,
    *,
    include_amendments: bool = False,
    filing_type: str = 'annual',
) -> None:
    sym = ticker.upper()
    print(f'=== debug extract: {sym} ({filing_type}) ===\n')

    if filing_type in {'annual', 'both'}:
        annuals = fetch_two_annuals(sym, include_amendments=include_amendments)
    else:
        annuals = []
    if not annuals:
        if filing_type in {'annual', 'both'}:
            print('No annual filings fetched (check ticker or EDGAR connectivity).')
    else:
        if len(annuals) < 2:
            print(f'Only {len(annuals)} annual filing(s) found; need two for year-over-year compare.')

        labels = ['current (newest)', 'prior']
        details: list[RiskExtraction] = []
        for i, (filing_date, full_text, form) in enumerate(annuals[:2]):
            label = labels[i] if i < len(labels) else f'filing_{i}'
            detail = extract_risk_section_detail(full_text, form)
            details.append(detail)
            print(f'--- annual {label}: {filing_date} ({form}) ---')
            print(f'  full text length : {len(full_text):,}')
            print(f'  extracted length : {len(detail.section):,}')
            print(f'  start/end offset : {detail.start} / {detail.end}')
            print(f'  quality score    : {detail.quality_score:.1f}')
            print(f'  risk cue count   : {detail.cue_count}')
            print(f'  looks_like_risk  : {detail.ok}')
            if detail.reject_reason:
                print(f'  reject reason    : {detail.reject_reason}')
            print(f'  heading preview  : {detail.heading_preview!r}')
            print()

        if len(annuals) >= 2:
            f0, f1 = annuals[0][2], annuals[1][2]
            if f0 != f1:
                print(f'WARNING: form mismatch ({f1} vs {f0}) — YoY compare may be unreliable.')
            both_ok = details[0].ok and details[1].ok
            print(f'Comparable for scoring: {both_ok}')
            if not both_ok:
                for label, detail in zip(labels, details):
                    if not detail.ok:
                        print(f'  NOT comparable ({label}): {detail.reject_reason}')

    if filing_type not in {'quarterly', 'both'}:
        return

    pair = fetch_two_quarterlies(sym)
    if not pair.get('ok'):
        print(f'No comparable quarterly filings fetched: {pair.get("reason", "unknown")}.')
        return

    selection = select_quarterly_comparison_sections(
        pair['prior_full_text'],
        pair['current_full_text'],
    )
    print()
    print(
        f"Quarterly pair: {pair['current_filing_date']} vs {pair['prior_filing_date']} "
        f"({pair['comparison_type']}, {pair['match_method']})"
    )
    print(f'Coordinated section: {selection.section_used or "non-comparable sections"}')
    print(f'Candidate lengths: {selection.candidate_lengths}')
    for label, detail in [
        ('quarterly current', selection.current_detail),
        ('quarterly prior', selection.prior_detail),
    ]:
        print(f'--- {label}: {detail.form} ---')
        print(f'  extracted length : {len(detail.section):,}')
        print(f'  section used     : {detail.section_used}')
        print(f'  start/end offset : {detail.start} / {detail.end}')
        print(f'  quality score    : {detail.quality_score:.1f}')
        print(f'  risk cue count   : {detail.cue_count}')
        print(f'  comparable       : {detail.ok}')
        if detail.reject_reason:
            print(f'  reject reason    : {detail.reject_reason}')
        print(f'  heading preview  : {detail.heading_preview!r}')
        print()


def _is_failed_cache(entry: dict) -> bool:
    if not isinstance(entry, dict):
        return True
    if not entry.get('ok', True):
        return True
    summary = str(entry.get('summary', ''))
    if entry.get('change_score') is None and summary in {'parse error', ''}:
        return True
    return summary.startswith(('parse error', 'API error', 'error:'))


def _cache_source(entry: dict | None) -> str:
    if not isinstance(entry, dict):
        return 'annual'
    return str(entry.get('source') or 'annual').lower()


def _is_cache_usable(entry: dict | None, *, force: bool, source: str) -> bool:
    return (
        entry is not None
        and not force
        and _cache_source(entry) == source
        and not _is_failed_cache(entry)
        and entry.get('ok', True)
    )


def _response_text(msg) -> str:
    parts: list[str] = []
    for block in msg.content:
        text = getattr(block, 'text', None)
        if text:
            parts.append(str(text))
    return '\n'.join(parts).strip()


def _parse_json_object(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith('```'):
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
    start = raw.find('{')
    end = raw.rfind('}')
    if start < 0 or end <= start:
        raise ValueError('no JSON object in response')
    chunk = raw[start:end + 1]
    try:
        return json.loads(chunk)
    except json.JSONDecodeError:
        score_m = re.search(r'"change_score"\s*:\s*(-?\d+(?:\.\d+)?|null)', chunk)
        conf_m = re.search(r'"confidence"\s*:\s*(-?\d+(?:\.\d+)?|null)', chunk)
        summary_m = re.search(r'"summary"\s*:\s*"(.*?)"\s*[,}]', chunk, re.DOTALL)
        out: dict = {}
        if score_m and score_m.group(1) != 'null':
            out['change_score'] = float(score_m.group(1))
        else:
            out['change_score'] = None
        if conf_m and conf_m.group(1) != 'null':
            out['confidence'] = float(conf_m.group(1))
        out['summary'] = summary_m.group(1) if summary_m else ''
        return out


def load_cache(path: str = CACHE_PATH) -> dict:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError, TypeError):
        return {}


def save_cache(cache: dict, path: str = CACHE_PATH) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2)


def score_change(
    prior_detail: RiskExtraction,
    current_detail: RiskExtraction,
    model: str,
    *,
    source: str = 'annual',
    section_used: str | None = None,
) -> dict:
    if not prior_detail.ok or not current_detail.ok:
        return {
            'change_score': None,
            'summary': EXTRACTION_FAIL_SUMMARY,
            'confidence': None,
            'non_comparable_reason': 'extraction quality below threshold',
        }

    client = anthropic.Anthropic()
    if source == 'quarterly':
        section_label = 'MD&A' if section_used == 'mda_both' else 'Risk Factors'
        prompt = (
            f'Compare the PRIOR vs CURRENT quarterly 10-Q {section_label} excerpts below.\n'
            'Focus on material changes in risk posture, business pressure, liquidity, '
            'demand, regulation, supply chain, customer concentration, and management tone. '
            'Treat "no material changes" language as approximately unchanged, not as an error.\n'
            'Respond with ONLY a JSON object (no prose, no markdown fences) with keys:\n'
            '  "change_score": float from -1.0 to +1.0 '
            '(NEGATIVE = added/strengthened risk vs prior; POSITIVE = risk eased; 0 = unchanged)\n'
            '  "summary": one sentence on what changed\n'
            '  "confidence": float 0.0 to 1.0\n'
            '  "non_comparable_reason": null or short string if excerpts are not comparable\n'
            'If the excerpts are not the same kind of section, return '
            '{"change_score": null, "summary": "non-comparable sections", '
            '"confidence": 0.0, "non_comparable_reason": "..."}.\n\n'
            f'PRIOR QUARTERLY SECTION:\n{prior_detail.section[:RISK_CAP]}\n\n'
            f'CURRENT QUARTERLY SECTION:\n{current_detail.section[:RISK_CAP]}'
        )
    else:
        prompt = (
            'Compare the PRIOR-year vs CURRENT-year risk-factor excerpts below.\n'
            'Respond with ONLY a JSON object (no prose, no markdown fences) with keys:\n'
            '  "change_score": float from -1.0 to +1.0 '
            '(NEGATIVE = added/strengthened risk vs prior; POSITIVE = risk eased; 0 = unchanged)\n'
            '  "summary": one sentence on what changed\n'
            '  "confidence": float 0.0 to 1.0\n'
            '  "non_comparable_reason": null or short string if excerpts are not true risk sections\n'
            'If either excerpt is not a Risk Factors section, return '
            '{"change_score": null, "summary": "non-comparable excerpts", '
            '"confidence": 0.0, "non_comparable_reason": "..."}.\n\n'
            f'PRIOR YEAR:\n{prior_detail.section[:RISK_CAP]}\n\n'
            f'CURRENT YEAR:\n{current_detail.section[:RISK_CAP]}'
        )
    try:
        msg = client.messages.create(
            model=model,
            max_tokens=512,
            temperature=0,
            messages=[{'role': 'user', 'content': prompt}],
        )
        raw = _response_text(msg)
        if not raw:
            return {'change_score': None, 'summary': 'error: empty model response', 'confidence': None}
        parsed = _parse_json_object(raw)
        if parsed.get('non_comparable_reason'):
            return {
                'change_score': None,
                'summary': str(parsed.get('summary', 'non-comparable excerpts')),
                'confidence': parsed.get('confidence'),
                'non_comparable_reason': parsed.get('non_comparable_reason'),
            }
        score = parsed.get('change_score')
        if score is not None:
            score = float(score)
        conf = parsed.get('confidence')
        if conf is not None:
            conf = float(conf)
        return {
            'change_score': score,
            'summary': str(parsed.get('summary', '')),
            'confidence': conf,
            'non_comparable_reason': parsed.get('non_comparable_reason'),
        }
    except anthropic.APIStatusError as exc:
        return {'change_score': None, 'summary': f'API error ({exc.status_code}): {exc.message}'}
    except anthropic.APIError as exc:
        return {'change_score': None, 'summary': f'API error: {exc}'}
    except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
        return {'change_score': None, 'summary': f'parse error: {exc}'}
    except Exception as exc:
        return {'change_score': None, 'summary': f'error: {type(exc).__name__}: {exc}'}


def run(
    tickers: list[str],
    model: str,
    force: bool = False,
    universe: str | None = None,
    include_amendments: bool = False,
    cache_path: str = CACHE_PATH,
    filing_type: str = 'annual',
) -> None:
    if not os.environ.get('ANTHROPIC_API_KEY'):
        print('Set ANTHROPIC_API_KEY first')
        return

    tickers = _resolve_tickers(tickers, universe)
    if not tickers:
        print('No tickers to process.')
        return

    cache = load_cache(cache_path)
    rows: list[dict] = []

    do_annual = filing_type in {'annual', 'both'}
    do_quarterly = filing_type in {'quarterly', 'both'}

    for ticker in tickers:
        sym = ticker.upper()
        if do_annual:
            try:
                annuals = fetch_two_annuals(sym, include_amendments=include_amendments)
            except Exception as exc:
                print(f'{sym}: annual fetch error ({exc}), skipping.')
                annuals = []

            if len(annuals) < 2:
                print(f'{sym}: fewer than two base 10-K/20-F filings, skipping annual.')
            else:
                current_date, current_full, current_form = annuals[0]
                prior_date, prior_full, prior_form = annuals[1]
                cache_key = f'{sym}:{current_date}'
                cached = cache.get(cache_key)
                use_cache = _is_cache_usable(cached, force=force, source='annual')

                prior_detail = extract_risk_section_detail(prior_full, prior_form)
                current_detail = extract_risk_section_detail(current_full, current_form)
                extraction_ok = prior_detail.ok and current_detail.ok

                if not extraction_ok:
                    print(f'{sym}: annual extraction unreliable for {current_date}, not scoring.')
                    result = {'change_score': None, 'summary': EXTRACTION_FAIL_SUMMARY, 'ok': False}
                elif use_cache:
                    result = cached
                    print(f'{sym}: using cached annual score for {current_date}')
                else:
                    if force and cached is not None:
                        print(f'{sym}: re-scoring annual {current_date} (--force)')
                    result = score_change(
                        prior_detail,
                        current_detail,
                        model,
                        source='annual',
                        section_used='risk_factors',
                    )
                    result['ok'] = result.get('change_score') is not None
                    cache[cache_key] = {
                        'source': 'annual',
                        'comparison_type': 'year_over_year',
                        'section_used': 'risk_factors',
                        'match_method': 'annual_pair',
                        'change_score': result.get('change_score'),
                        'summary': result.get('summary'),
                        'confidence': result.get('confidence'),
                        'non_comparable_reason': result.get('non_comparable_reason'),
                        'prior_filing_date': prior_date,
                        'current_filing_date': current_date,
                        'form': current_form,
                        'prior_form': prior_form,
                        'model': model,
                        'scored_at': datetime.now(timezone.utc).isoformat(),
                        'extraction_quality': current_detail.quality_score,
                        'section_preview': current_detail.heading_preview[:500],
                        'ok': result['ok'],
                    }
                    save_cache(cache, cache_path)
                    print(f'{sym}: scored annual {current_date} ({current_form}) vs {prior_date} ({prior_form})')

                rows.append({
                    'ticker': sym,
                    'source': 'annual',
                    'filing_date': current_date,
                    'comparison_type': 'year_over_year',
                    'section_used': 'risk_factors',
                    'match_method': 'annual_pair',
                    'change_score': result.get('change_score'),
                    'summary': result.get('summary', ''),
                    'ok': bool(result.get('ok', False)),
                })

        if do_quarterly:
            pair = fetch_two_quarterlies(sym)
            if not pair.get('ok'):
                print(f'{sym}: quarterly {pair.get("reason", "insufficient filings")}, skipping.')
                continue

            current_date = pair['current_filing_date']
            prior_date = pair['prior_filing_date']
            current_form = pair.get('current_form', '10-Q')
            prior_form = pair.get('prior_form', '10-Q')
            cache_key = f'{sym}:{current_date}'
            cached = cache.get(cache_key)
            use_cache = _is_cache_usable(cached, force=force, source='quarterly')

            selection = select_quarterly_comparison_sections(
                pair['prior_full_text'],
                pair['current_full_text'],
            )
            extraction_ok = selection.ok and selection.section_used is not None

            if not extraction_ok:
                print(f'{sym}: quarterly non-comparable sections for {current_date}, not scoring.')
                result = {
                    'change_score': None,
                    'summary': 'non-comparable sections',
                    'confidence': None,
                    'non_comparable_reason': selection.reject_reason or 'non-comparable sections',
                    'ok': False,
                }
                cache[cache_key] = {
                    'source': 'quarterly',
                    'comparison_type': pair['comparison_type'],
                    'section_used': selection.section_used,
                    'match_method': pair['match_method'],
                    'change_score': None,
                    'summary': result['summary'],
                    'confidence': None,
                    'non_comparable_reason': result['non_comparable_reason'],
                    'prior_filing_date': prior_date,
                    'current_filing_date': current_date,
                    'form': current_form,
                    'prior_form': prior_form,
                    'current_period': pair.get('current_period'),
                    'prior_period': pair.get('prior_period'),
                    'model': model,
                    'scored_at': datetime.now(timezone.utc).isoformat(),
                    'extraction_quality': selection.current_detail.quality_score,
                    'section_preview': selection.current_detail.heading_preview[:500],
                    'candidate_lengths': selection.candidate_lengths,
                    'ok': False,
                }
                save_cache(cache, cache_path)
            elif use_cache:
                result = cached
                print(
                    f"{sym}: using cached quarterly score for {current_date} "
                    f"({result.get('comparison_type', pair['comparison_type'])}, "
                    f"{result.get('section_used', selection.section_used)})"
                )
            else:
                if force and cached is not None:
                    print(f'{sym}: re-scoring quarterly {current_date} (--force)')
                result = score_change(
                    selection.prior_detail,
                    selection.current_detail,
                    model,
                    source='quarterly',
                    section_used=selection.section_used,
                )
                result['ok'] = result.get('change_score') is not None
                cache[cache_key] = {
                    'source': 'quarterly',
                    'comparison_type': pair['comparison_type'],
                    'section_used': selection.section_used,
                    'match_method': pair['match_method'],
                    'change_score': result.get('change_score'),
                    'summary': result.get('summary'),
                    'confidence': result.get('confidence'),
                    'non_comparable_reason': result.get('non_comparable_reason'),
                    'prior_filing_date': prior_date,
                    'current_filing_date': current_date,
                    'form': current_form,
                    'prior_form': prior_form,
                    'current_period': pair.get('current_period'),
                    'prior_period': pair.get('prior_period'),
                    'model': model,
                    'scored_at': datetime.now(timezone.utc).isoformat(),
                    'extraction_quality': selection.current_detail.quality_score,
                    'section_preview': selection.current_detail.heading_preview[:500],
                    'candidate_lengths': selection.candidate_lengths,
                    'ok': result['ok'],
                }
                save_cache(cache, cache_path)
                print(
                    f"{sym}: scored quarterly {current_date} ({current_form}) vs {prior_date} ({prior_form}) "
                    f"[{pair['comparison_type']}, {selection.section_used}]"
                )

            rows.append({
                'ticker': sym,
                'source': 'quarterly',
                'filing_date': current_date,
                'comparison_type': pair.get('comparison_type'),
                'section_used': selection.section_used,
                'match_method': pair.get('match_method'),
                'change_score': result.get('change_score'),
                'summary': result.get('summary', ''),
                'ok': bool(result.get('ok', False)),
            })

    if not rows:
        print('No results.')
        return

    df = pd.DataFrame(rows)
    df['change_score'] = pd.to_numeric(df['change_score'], errors='coerce')
    df['ok'] = df['ok'].astype(bool)
    scored = df[df['ok']].sort_values('change_score', ascending=True, na_position='last')
    flagged = df[~df['ok']]

    pd.set_option('display.max_colwidth', None)
    print()
    if not scored.empty:
        print(scored.to_string(index=False))
    if not flagged.empty:
        if not scored.empty:
            print()
        print('FLAGGED — not scored, verify manually')
        print(flagged.to_string(index=False))
    print()
    print(
        f'Qualitative research overlay on ~{len(scored)} scored names — '
        'not a statistically validated factor.'
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description='10-K/20-F risk-factor change scores via EDGAR + Anthropic.',
    )
    parser.add_argument('tickers', nargs='*', default=[], help='Tickers to process')
    parser.add_argument('--model', default=DEFAULT_MODEL, help='Anthropic model id')
    parser.add_argument('--force', action='store_true', help='Re-score instead of using cache')
    parser.add_argument('--universe', default=None, help='Universe preset when no tickers given')
    parser.add_argument(
        '--filing-type',
        choices=['annual', 'quarterly', 'both'],
        default='annual',
        help='Filing source to score: annual, quarterly, or both (default: annual)',
    )
    parser.add_argument('--include-amendments', action='store_true',
                        help='Include 10-K/A and 20-F/A filings (default: base forms only)')
    parser.add_argument('--debug-extract', metavar='TICKER',
                        help='Print extraction diagnostics (no LLM call)')
    parser.add_argument('--cache-path', default=CACHE_PATH, help='Path to tenk cache JSON')
    args = parser.parse_args()

    if args.debug_extract:
        debug_extract(
            args.debug_extract,
            include_amendments=args.include_amendments,
            filing_type=args.filing_type,
        )
        return

    run(
        args.tickers, args.model, force=args.force, universe=args.universe,
        include_amendments=args.include_amendments, cache_path=args.cache_path,
        filing_type=args.filing_type,
    )


if __name__ == '__main__':
    main()
