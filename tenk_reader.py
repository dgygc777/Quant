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

import anthropic
import edgar
import pandas as pd


def _configure_edgar() -> None:
    """edgartools uses hishel.FileStorage; hishel 1.x removed it and breaks ticker lookup."""
    try:
        import hishel
        if not hasattr(hishel, "FileStorage"):
            edgar.httpclient.CACHE_ENABLED = False
            edgar.httpclient.close_clients()
    except ImportError:
        edgar.httpclient.CACHE_ENABLED = False
        edgar.httpclient.close_clients()


_configure_edgar()
edgar.set_identity("Dean Chen dean@example.com")

CACHE_PATH = "tenk_cache.json"
RISK_CAP = 25_000
DEFAULT_MODEL = "claude-haiku-4-5-20251001"

FALLBACK_TICKERS = [
    "NVDA", "AMD", "AVGO", "QCOM", "MU", "INTC", "MRVL", "AMAT", "LRCX", "KLAC",
    "TXN", "ADI", "MCHP", "ON", "COHR", "TSM", "ASML",
]

EXTRACTION_FAIL_SUMMARY = (
    "extraction unreliable — risk section not found "
    "(likely 20-F structure or section mismatch)"
)

RISK_CUES = [
    "risk", "adversely", "could", "uncertain", "no assurance",
    "fluctuat", "harm", "may not", "subject to",
]

_10K_END_MARKERS = [
    "item 1b", "item 2", "item 3", "unresolved staff comments",
]
_20F_END_MARKERS = [
    "item 4", "information on the company", "item 5",
    "operating and financial review", "directors, senior management",
]
_10K_START_PATTERNS = [
    r"item\s+1a\.?\s*risk\s*factors",
    r"item\s+1a\.?\s",
    r"risk\s*factors",
]
_20F_START_PATTERNS = [
    r"item\s+3\.?\s*key\s*information",
    r"item\s+3\.?\s*risk",
    r"item\s+3\.?\s",
    r"principal\s*risks",
    r"risk\s*factors",
]
_TOC_LINE = re.compile(r"risk\s*factors\s{4,}\d{1,4}\b", re.IGNORECASE)
_MIN_SLICE = 200  # skip TOC-sized slices ending at the next Item marker
_HEAD_WINDOW = 1_200
_RISK_HEADING = re.compile(
    r"(?:(?:^|\n|\r|[>\s])(?:item\s+1a\.?|item\s+3\.?\s*(?:key\s+information)?\s*[—\-:–]?\s*d\.?)\s*)?"
    r"(?:risk\s*factors|principal\s*risks)\b",
    re.IGNORECASE,
)
_RISK_FACTORS_HEADING = re.compile(
    r"(?:(?:^|\n|\r|[>\s])(?:item\s+1a\.?|item\s+3\.?\s*(?:key\s+information)?\s*[—\-:–]?\s*d\.?)\s*)?"
    r"risk\s*factors\b",
    re.IGNORECASE,
)
_PRINCIPAL_RISKS_HEADING = re.compile(r"\bprincipal\s*risks\b", re.IGNORECASE)
_FORWARD_LOOKING_REJECT = re.compile(
    r"(?:forward-looking|forward looking)\s+statements|"
    r"special\s+note\s+regarding\s+forward-looking|"
    r"cautionary\s+statement",
    re.IGNORECASE,
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
        return list(UNIVERSE_PRESETS["semis"])
    except Exception:
        pass
    try:
        from quant.universes import PRESETS  # type: ignore[attr-defined]
        return list(PRESETS["semis"])
    except Exception:
        pass
    try:
        from quant.universes import DEFAULT_UNIVERSE
        return list(DEFAULT_UNIVERSE)
    except Exception:
        return list(FALLBACK_TICKERS)


def _filing_text(filing) -> str:
    """Extract full text; edgartools API varies by version."""
    for method in ("text", "markdown"):
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
            if hasattr(obj, "text") and callable(obj.text):
                out = obj.text()
                if out:
                    return str(out)
            if hasattr(obj, "markdown") and callable(obj.markdown):
                out = obj.markdown()
                if out:
                    return str(out)
    except Exception:
        pass
    return ""


def _filing_date_str(filing) -> str:
    fd = getattr(filing, "filing_date", None)
    if fd is None:
        return "unknown"
    return str(fd)


def _filing_form(filing) -> str:
    form = getattr(filing, "form", None)
    return str(form) if form else "10-K"


def fetch_two_annuals(ticker: str) -> list[tuple[str, str, str]]:
    """Return up to two (filing_date, full_text, form) tuples, newest first."""
    try:
        company = edgar.Company(ticker)
        filings = company.get_filings(form=["10-K", "20-F"])
        if filings is None or len(filings) == 0:
            return []

        recent = filings.latest(2)
        items = []
        if hasattr(recent, "filing_date") and hasattr(recent, "accession_no"):
            items = [recent]
        else:
            try:
                n = len(recent)
            except TypeError:
                n = 0
            for i in range(min(n, 2)):
                items.append(recent[i])

        items = sorted(items, key=lambda f: _filing_date_str(f), reverse=True)[:2]
        out: list[tuple[str, str, str]] = []
        for filing in items:
            text = _filing_text(filing)
            if text:
                out.append((_filing_date_str(filing), text, _filing_form(filing)))
        time.sleep(0.3)
        return out
    except Exception:
        return []


def _find_starts(text: str, patterns: list[str]) -> list[int]:
    low = text.lower()
    starts: set[int] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, low):
            starts.add(match.start())
    return sorted(starts)


def _is_toc_slice(section: str) -> bool:
    head = section[:300]
    if _TOC_LINE.search(head):
        return True
    # TOC rows: Risk Factors + lots of whitespace + page number before next Item
    if len(section) < _MIN_SLICE and re.search(r"\s{6,}\d{1,4}\b", head):
        return True
    return False


def _candidate_score(section: str, start_pos: int, form_key: str) -> float:
    """Higher score means the slice is more likely to be the real risk section."""
    if not section or _is_toc_slice(section):
        return -1_000_000.0

    head = section[:_HEAD_WINDOW]
    head_low = head.lower()
    score = float(min(len(section), RISK_CAP)) / 1_000.0

    cue_count = risk_cue_count(section)
    cue_density = cue_count / max(len(section), 1) * 10_000
    score += min(cue_count, 80) * 4.0
    score += min(cue_density, 35.0) * 6.0

    if _RISK_FACTORS_HEADING.search(head):
        score += 360.0
    elif _PRINCIPAL_RISKS_HEADING.search(head):
        score += 60.0
    elif _RISK_HEADING.search(head):
        score += 120.0
    if re.match(r"\s*Risk\s+factors\b", head):
        score += 140.0
    if re.match(r"\s*risk\s+factors\b", head):
        score -= 260.0
    if re.search(r"\bd\.\s*risk\s*factors\b", head, re.IGNORECASE):
        score += 200.0
    if re.search(
        r"\brisk\s+factors\s+(?:should\s+be\s+considered|our\s+business\s+is\s+subject|we\s+wish\s+to\s+caution)",
        head,
        re.IGNORECASE,
    ):
        score += 120.0
    if form_key == "20-F" and re.search(r"\bitem\s+3\.?\s+key\s+information\b", head, re.IGNORECASE):
        score += 40.0

    if _FORWARD_LOOKING_REJECT.search(head):
        score -= 350.0
    if re.search(r"\bsee\s+[\"'“]?item\s+[13]\b", head_low):
        score -= 180.0
    if re.match(r"\s*risk\s+factors\s+related\s+to\b", head):
        score -= 320.0
    if re.search(r"\bfor\s+a\s+discussion\s+of\s+the\s+risk\s+related\s+to\b", head_low):
        score -= 220.0
    if re.search(r"\btable\s+of\s+contents\b", head_low):
        score -= 200.0
    if re.search(r"\bprincipal\s+risks\s+and\s+opportunities\b", head_low):
        score -= 280.0
    if re.search(r"\bsustainability|esg\s+committee|governance\b", head_low):
        score -= 120.0
    if cue_count < 15:
        score -= 300.0

    score += min(start_pos / max(RISK_CAP, 1), 20.0)
    return score


def extract_risk_section(text: str, form: str = "10-K") -> str:
    if not text:
        return ""

    form_key = str(form).upper()
    if form_key == "20-F":
        end_markers = _20F_END_MARKERS
        start_patterns = _20F_START_PATTERNS
    else:
        end_markers = _10K_END_MARKERS
        start_patterns = _10K_START_PATTERNS

    low = text.lower()
    starts = _find_starts(text, start_patterns)
    best = ""
    best_score = -1_000_000.0
    for st in starts:
        ends = [low.find(mk, st + 50) for mk in end_markers]
        ends = [e for e in ends if e != -1]
        end = min(ends) if ends else st + RISK_CAP
        chunk = text[st:end][:RISK_CAP]
        score = _candidate_score(chunk, st, form_key)
        if score > best_score:
            best = chunk
            best_score = score

    if not best:
        mid = len(text) // 2
        best = text[mid : mid + RISK_CAP]
    return best[:RISK_CAP]


def risk_cue_count(section: str) -> int:
    s = section.lower()
    return sum(s.count(c) for c in RISK_CUES)


def looks_like_risk_factors(section: str) -> bool:
    if not section or len(section) < 3000:
        return False
    return risk_cue_count(section) >= 15


def debug_extract(ticker: str) -> None:
    """Print extraction diagnostics for a ticker (no LLM / API key needed)."""
    sym = ticker.upper()
    print(f"=== debug extract: {sym} ===\n")

    annuals = fetch_two_annuals(sym)
    if not annuals:
        print("No filings fetched (check ticker or EDGAR connectivity).")
        return
    if len(annuals) < 2:
        print(f"Only {len(annuals)} filing(s) found; need two for year-over-year compare.")

    labels = ["current (newest)", "prior"]
    for i, (filing_date, full_text, form) in enumerate(annuals[:2]):
        label = labels[i] if i < len(labels) else f"filing_{i}"
        section = extract_risk_section(full_text, form)
        cues = risk_cue_count(section)
        ok = looks_like_risk_factors(section)
        preview = section[:500].replace("\n", " ")

        print(f"--- {label}: {filing_date} ({form}) ---")
        print(f"  full text length : {len(full_text):,}")
        print(f"  extracted length : {len(section):,}")
        print(f"  risk cue count   : {cues}  (need >= 15)")
        print(f"  looks_like_risk  : {ok}")
        print(f"  preview          : {preview!r}")
        print()

    if len(annuals) >= 2:
        f0, f1 = annuals[0][2], annuals[1][2]
        if f0 != f1:
            print(f"WARNING: form mismatch ({f1} vs {f0}) — YoY compare may be unreliable.")
        s0 = extract_risk_section(annuals[0][1], annuals[0][2])
        s1 = extract_risk_section(annuals[1][1], annuals[1][2])
        both_ok = looks_like_risk_factors(s0) and looks_like_risk_factors(s1)
        print(f"Comparable for scoring: {both_ok}")


def _is_failed_cache(entry: dict) -> bool:
    if not isinstance(entry, dict):
        return True
    summary = str(entry.get("summary", ""))
    if entry.get("change_score") is None and summary in {"parse error", ""}:
        return True
    return summary.startswith(("parse error", "API error", "error:"))


def _response_text(msg) -> str:
    parts: list[str] = []
    for block in msg.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(str(text))
    return "\n".join(parts).strip()


def _parse_json_object(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in response")

    chunk = raw[start : end + 1]
    try:
        return json.loads(chunk)
    except json.JSONDecodeError:
        score_m = re.search(r'"change_score"\s*:\s*(-?\d+(?:\.\d+)?)', chunk)
        if not score_m:
            raise
        summary_m = re.search(r'"summary"\s*:\s*"(.*?)"\s*\}?\s*$', chunk, re.DOTALL)
        return {
            "change_score": float(score_m.group(1)),
            "summary": summary_m.group(1) if summary_m else "",
        }


def load_cache() -> dict:
    if not os.path.isfile(CACHE_PATH):
        return {}
    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError, TypeError):
        return {}


def save_cache(cache: dict) -> None:
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def score_change(prior_text: str, current_text: str, model: str) -> dict:
    """One Anthropic call comparing prior vs current risk-factor text."""
    if not prior_text.strip() or not current_text.strip():
        return {"change_score": None, "summary": "error: empty risk-factor excerpt"}

    client = anthropic.Anthropic()
    prompt = (
        "Compare the PRIOR-year vs CURRENT-year risk-factor excerpts below.\n"
        "Respond with ONLY a JSON object (no prose, no markdown fences) with keys:\n"
        '  "change_score": float from -1.0 to +1.0 '
        "(NEGATIVE = added/strengthened risk, uncertainty, or hedged language vs prior; "
        "POSITIVE = risk eased; 0 = unchanged boilerplate)\n"
        '  "summary": one sentence on what changed\n'
        "If either excerpt does not appear to be a Risk Factors section "
        "(e.g. it contains financial statements or a business description "
        "instead of risk language), return "
        '{"change_score": null, "summary": "non-comparable excerpts"} '
        "instead of scoring.\n\n"
        f"PRIOR YEAR:\n{prior_text[:RISK_CAP]}\n\n"
        f"CURRENT YEAR:\n{current_text[:RISK_CAP]}"
    )
    try:
        msg = client.messages.create(
            model=model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = _response_text(msg)
        if not raw:
            return {"change_score": None, "summary": "error: empty model response"}
        parsed = _parse_json_object(raw)
        score = parsed.get("change_score")
        summary = parsed.get("summary", "")
        if score is not None:
            score = float(score)
        return {"change_score": score, "summary": str(summary)}
    except anthropic.APIStatusError as exc:
        return {"change_score": None, "summary": f"API error ({exc.status_code}): {exc.message}"}
    except anthropic.APIError as exc:
        return {"change_score": None, "summary": f"API error: {exc}"}
    except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
        return {"change_score": None, "summary": f"parse error: {exc}"}
    except Exception as exc:
        return {"change_score": None, "summary": f"error: {type(exc).__name__}: {exc}"}


def run(tickers: list[str], model: str, force: bool = False, universe: str | None = None) -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY first")
        return

    tickers = _resolve_tickers(tickers, universe)
    if not tickers:
        print("No tickers to process.")
        return

    cache = load_cache()
    rows: list[dict] = []

    for ticker in tickers:
        sym = ticker.upper()
        try:
            annuals = fetch_two_annuals(sym)
        except Exception as exc:
            print(f"{sym}: fetch error ({exc}), skipping.")
            continue

        if len(annuals) < 2:
            print(f"{sym}: fewer than two 10-K/20-F filings, skipping.")
            continue

        current_date, current_full, current_form = annuals[0]
        prior_date, prior_full, prior_form = annuals[1]
        cache_key = f"{sym}:{current_date}"
        cached = cache.get(cache_key)
        use_cache = (
            cached is not None
            and not force
            and not _is_failed_cache(cached)
            and cached.get("ok", True)
        )

        prior_risk = extract_risk_section(prior_full, prior_form)
        current_risk = extract_risk_section(current_full, current_form)
        extraction_ok = (
            looks_like_risk_factors(prior_risk)
            and looks_like_risk_factors(current_risk)
        )

        if not extraction_ok:
            print(f"{sym}: extraction unreliable for {current_date}, not scoring.")
            result = {"change_score": None, "summary": EXTRACTION_FAIL_SUMMARY, "ok": False}
        elif use_cache:
            result = cached
            print(f"{sym}: using cached score for {current_date}")
        else:
            if force and cached is not None:
                print(f"{sym}: re-scoring {current_date} (--force)")
            result = score_change(prior_risk, current_risk, model)
            result["ok"] = True
            cache[cache_key] = {
                "change_score": result.get("change_score"),
                "summary": result.get("summary"),
                "prior_filing_date": prior_date,
                "current_filing_date": current_date,
                "ok": True,
            }
            save_cache(cache)
            print(f"{sym}: scored {current_date} vs {prior_date}")

        rows.append({
            "ticker": sym,
            "filing_date": current_date,
            "change_score": result.get("change_score"),
            "summary": result.get("summary", ""),
            "ok": bool(result.get("ok", False)),
        })

    if not rows:
        print("No results.")
        return

    df = pd.DataFrame(rows)
    df["change_score"] = pd.to_numeric(df["change_score"], errors="coerce")
    df["ok"] = df["ok"].astype(bool)

    scored = df[df["ok"]].sort_values("change_score", ascending=True, na_position="last")
    flagged = df[~df["ok"]]

    pd.set_option("display.max_colwidth", None)
    print()
    if not scored.empty:
        print(scored.to_string(index=False))
    if not flagged.empty:
        if not scored.empty:
            print()
        print("FLAGGED — not scored, verify manually")
        print(flagged.to_string(index=False))
    print()
    print(
        f"Qualitative research overlay on ~{len(scored)} scored names — "
        "not a statistically validated factor."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="10-K/20-F risk-factor change scores via EDGAR + Anthropic.",
    )
    parser.add_argument(
        "tickers", nargs="*", default=[],
        help="Tickers to process (default: semis universe preset)",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Anthropic model id (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-score tickers instead of using cache (also retries prior failures)",
    )
    parser.add_argument(
        "--universe", default=None,
        help="Universe preset when no tickers given (e.g. nasdaq_top30, semis)",
    )
    parser.add_argument(
        "--debug-extract", metavar="TICKER",
        help="Print risk-section extraction diagnostics for TICKER (no LLM call)",
    )
    args = parser.parse_args()

    if args.debug_extract:
        debug_extract(args.debug_extract)
        return

    run(args.tickers, args.model, force=args.force, universe=args.universe)


if __name__ == "__main__":
    main()
