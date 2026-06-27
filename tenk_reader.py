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
from datetime import datetime, timezone

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


def debug_extract(ticker: str, *, include_amendments: bool = False) -> None:
    sym = ticker.upper()
    print(f'=== debug extract: {sym} ===\n')
    annuals = fetch_two_annuals(sym, include_amendments=include_amendments)
    if not annuals:
        print('No filings fetched (check ticker or EDGAR connectivity).')
        return
    if len(annuals) < 2:
        print(f'Only {len(annuals)} filing(s) found; need two for year-over-year compare.')

    labels = ['current (newest)', 'prior']
    details: list[RiskExtraction] = []
    for i, (filing_date, full_text, form) in enumerate(annuals[:2]):
        label = labels[i] if i < len(labels) else f'filing_{i}'
        detail = extract_risk_section_detail(full_text, form)
        details.append(detail)
        print(f'--- {label}: {filing_date} ({form}) ---')
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


def _is_failed_cache(entry: dict) -> bool:
    if not isinstance(entry, dict):
        return True
    if not entry.get('ok', True):
        return True
    summary = str(entry.get('summary', ''))
    if entry.get('change_score') is None and summary in {'parse error', ''}:
        return True
    return summary.startswith(('parse error', 'API error', 'error:'))


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
) -> dict:
    if not prior_detail.ok or not current_detail.ok:
        return {
            'change_score': None,
            'summary': EXTRACTION_FAIL_SUMMARY,
            'confidence': None,
            'non_comparable_reason': 'extraction quality below threshold',
        }

    client = anthropic.Anthropic()
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

    for ticker in tickers:
        sym = ticker.upper()
        try:
            annuals = fetch_two_annuals(sym, include_amendments=include_amendments)
        except Exception as exc:
            print(f'{sym}: fetch error ({exc}), skipping.')
            continue

        if len(annuals) < 2:
            print(f'{sym}: fewer than two base 10-K/20-F filings, skipping.')
            continue

        current_date, current_full, current_form = annuals[0]
        prior_date, prior_full, prior_form = annuals[1]
        cache_key = f'{sym}:{current_date}'
        cached = cache.get(cache_key)
        use_cache = (
            cached is not None
            and not force
            and not _is_failed_cache(cached)
            and cached.get('ok', True)
        )

        prior_detail = extract_risk_section_detail(prior_full, prior_form)
        current_detail = extract_risk_section_detail(current_full, current_form)
        extraction_ok = prior_detail.ok and current_detail.ok

        if not extraction_ok:
            print(f'{sym}: extraction unreliable for {current_date}, not scoring.')
            result = {'change_score': None, 'summary': EXTRACTION_FAIL_SUMMARY, 'ok': False}
        elif use_cache:
            result = cached
            print(f'{sym}: using cached score for {current_date}')
        else:
            if force and cached is not None:
                print(f'{sym}: re-scoring {current_date} (--force)')
            result = score_change(prior_detail, current_detail, model)
            result['ok'] = result.get('change_score') is not None
            cache[cache_key] = {
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
            print(f'{sym}: scored {current_date} ({current_form}) vs {prior_date} ({prior_form})')

        rows.append({
            'ticker': sym,
            'filing_date': current_date,
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
    parser.add_argument('--include-amendments', action='store_true',
                        help='Include 10-K/A and 20-F/A filings (default: base forms only)')
    parser.add_argument('--debug-extract', metavar='TICKER',
                        help='Print extraction diagnostics (no LLM call)')
    parser.add_argument('--cache-path', default=CACHE_PATH, help='Path to tenk cache JSON')
    args = parser.parse_args()

    if args.debug_extract:
        debug_extract(args.debug_extract, include_amendments=args.include_amendments)
        return

    run(
        args.tickers, args.model, force=args.force, universe=args.universe,
        include_amendments=args.include_amendments, cache_path=args.cache_path,
    )


if __name__ == '__main__':
    main()
