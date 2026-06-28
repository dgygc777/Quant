"""Read-only helpers for tenk_cache.json (no EDGAR / LLM)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime


@dataclass
class TenkCacheEntry:
    ticker: str
    change_score: float | None
    filing_date: date | None
    prior_filing_date: date | None
    form: str | None
    prior_form: str | None
    ok: bool
    summary: str | None = None
    model: str | None = None
    scored_at: str | None = None
    extraction_quality: float | None = None
    section_preview: str | None = None
    source: str = 'annual'
    comparison_type: str | None = None
    section_used: str | None = None
    match_method: str | None = None


def _parse_filing_date(raw: str | None) -> date | None:
    if not raw:
        return None
    try:
        return datetime.strptime(str(raw)[:10], '%Y-%m-%d').date()
    except ValueError:
        return None


def _coerce_score(val) -> float | None:
    if val is None:
        return None
    try:
        score = float(val)
    except (TypeError, ValueError):
        return None
    return score


def load_tenk_scores(
    path: str = 'tenk_cache.json',
    *,
    include_failed: bool = False,
    source: str | None = 'annual',
) -> dict[str, TenkCacheEntry]:
    """Return newest valid cache entry per ticker, filtered by source first."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding='utf-8') as f:
            cache = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(cache, dict):
        return {}

    by_ticker: dict[str, tuple[date | None, TenkCacheEntry]] = {}
    for key, val in cache.items():
        if ':' not in key or not isinstance(val, dict):
            continue
        tkr, date_raw = key.split(':', 1)
        tkr = tkr.upper()
        entry_source = str(val.get('source') or 'annual').lower()
        if source is not None and entry_source != str(source).lower():
            continue
        filing_date = _parse_filing_date(date_raw) or _parse_filing_date(val.get('current_filing_date'))
        ok = bool(val.get('ok', True))
        score = _coerce_score(val.get('change_score'))
        if not include_failed:
            if not ok:
                continue
            if score is None:
                continue
            if filing_date is None:
                continue
        entry = TenkCacheEntry(
            ticker=tkr,
            change_score=score,
            filing_date=filing_date,
            prior_filing_date=_parse_filing_date(val.get('prior_filing_date')),
            form=val.get('form') or val.get('current_form'),
            prior_form=val.get('prior_form') or val.get('prior_filing_form'),
            ok=ok,
            summary=val.get('summary'),
            model=val.get('model'),
            scored_at=val.get('scored_at'),
            extraction_quality=val.get('extraction_quality'),
            section_preview=val.get('section_preview'),
            source=entry_source,
            comparison_type=val.get('comparison_type'),
            section_used=val.get('section_used'),
            match_method=val.get('match_method'),
        )
        prev = by_ticker.get(tkr)
        prev_date = prev[0] if prev else None
        if prev is None or (filing_date is not None and (prev_date is None or filing_date > prev_date)):
            by_ticker[tkr] = (filing_date, entry)
    return {t: e for t, (_, e) in by_ticker.items()}


def load_tenq_scores(
    path: str = 'tenk_cache.json',
    *,
    include_failed: bool = False,
) -> dict[str, TenkCacheEntry]:
    """Return newest valid quarterly 10-Q cache entry per ticker."""
    return load_tenk_scores(path, include_failed=include_failed, source='quarterly')
