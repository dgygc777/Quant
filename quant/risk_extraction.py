"""Risk-factor section extraction from 10-K / 20-F plain text."""

from __future__ import annotations

import re
from dataclasses import dataclass

RISK_CAP = 25_000
MIN_SECTION_LEN = 3_000
MIN_QUALITY_SCORE = 150.0

RISK_CUES = [
    'risk', 'adversely', 'could', 'uncertain', 'no assurance',
    'fluctuat', 'harm', 'may not', 'subject to',
]

_10K_END_MARKERS = [
    'item 1b', 'item 2', 'item 3', 'unresolved staff comments',
]
_20F_END_MARKERS = [
    'item 4', 'information on the company', 'item 5',
    'operating and financial review', 'directors, senior management',
]
_10K_START_PATTERNS = [
    r'item\s+1a\.?\s*risk\s*factors',
    r'item\s+1a\.?\s',
    r'risk\s*factors',
]
_20F_START_PATTERNS = [
    r'item\s+3\.?\s*key\s*information',
    r'item\s+3\.?\s*d\.?\s*risk',
    r'item\s+3\.?\s*risk',
    r'item\s+3\.?\s',
    r'principal\s*risks',
    r'risk\s*factors',
]
_TOC_LINE = re.compile(r'risk\s*factors\s{4,}\d{1,4}\b', re.IGNORECASE)
_MIN_SLICE = 200
_HEAD_WINDOW = 1_200
_RISK_HEADING = re.compile(
    r'(?:(?:^|\n|\r|[>\s])(?:item\s+1a\.?|item\s+3\.?\s*(?:key\s+information)?\s*[—\-:–]?\s*d\.?)\s*)?'
    r'(?:risk\s*factors|principal\s*risks)\b',
    re.IGNORECASE,
)
_RISK_FACTORS_HEADING = re.compile(
    r'(?:(?:^|\n|\r|[>\s])(?:item\s+1a\.?|item\s+3\.?\s*(?:key\s+information)?\s*[—\-:–]?\s*d\.?)\s*)?'
    r'risk\s*factors\b',
    re.IGNORECASE,
)
_PRINCIPAL_RISKS_HEADING = re.compile(r'\bprincipal\s*risks\b', re.IGNORECASE)
_FORWARD_LOOKING_REJECT = re.compile(
    r'(?:forward-looking|forward looking)\s+statements|'
    r'special\s+note\s+regarding\s+forward-looking|'
    r'cautionary\s+statement',
    re.IGNORECASE,
)
_ANCHORED_10K = re.compile(
    r'(?:^|\n|\r|[>\s])item\s+1a\.?\s*(?:risk\s*factors)?',
    re.IGNORECASE,
)
_ANCHORED_20F = re.compile(
    r'(?:^|\n|\r|[>\s])item\s+3\.?\s*(?:key\s+information\s*[—\-:–]?\s*)?(?:d\.?\s*)?risk\s*factors',
    re.IGNORECASE,
)
_STANDALONE_RISK = re.compile(
    r'(?:^|\n|\r)\s*(?:RISK\s+FACTORS|Risk\s+Factors)\s*(?:\n|\r|$)',
)


@dataclass
class RiskExtraction:
    section: str
    ok: bool
    quality_score: float
    start: int
    end: int
    form: str
    cue_count: int
    heading_preview: str
    reject_reason: str | None = None


def risk_cue_count(section: str) -> int:
    s = section.lower()
    return sum(s.count(c) for c in RISK_CUES)


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
    if len(section) < _MIN_SLICE and re.search(r'\s{6,}\d{1,4}\b', head):
        return True
    return False


def _cross_reference_reason(head: str) -> str | None:
    head_low = head.lower()
    if re.search(r'risk\s+factors\s+titled', head, re.IGNORECASE):
        return 'cross-reference: risk factors titled'
    if re.search(r'risk\s+factors,\s*["\u201c]', head, re.IGNORECASE):
        return 'cross-reference: risk factors, "'
    if re.search(r'risk\s+factors[\.\u201d"]\s*(?:as\s+a\s+result|see\s|for\s+a)', head, re.IGNORECASE):
        return 'cross-reference: risk factors."'
    if re.search(r'\bsee\s+["\u201c]?item\s+[13]\b', head_low):
        return 'cross-reference: see Item 1A/3'
    if re.match(r'\s*risk\s+factors\s+related\s+to\b', head, re.IGNORECASE):
        return 'cross-reference: risk factors related to'
    if re.search(r'\bfor\s+a\s+discussion\s+of\b', head_low):
        return 'cross-reference: for a discussion of'
    if re.match(r'[^A-Z\n\r>]*risk\s+factors\b', head) and not re.match(r'\s*Risk\s+Factors\b', head):
        return 'inline/lowercase risk factors phrase'
    return None


def _candidate_score(section: str, start_pos: int, form_key: str) -> float:
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
    if re.match(r'\s*Risk\s+factors\b', head):
        score += 140.0
    if re.match(r'\s*risk\s+factors\b', head):
        score -= 260.0
    if re.search(r'\bd\.\s*risk\s*factors\b', head, re.IGNORECASE):
        score += 200.0
    if _ANCHORED_10K.search(head) or _ANCHORED_20F.search(head):
        score += 180.0
    if _STANDALONE_RISK.search(head):
        score += 160.0
    if re.search(
        r'\brisk\s+factors\s+(?:should\s+be\s+considered|our\s+business\s+is\s+subject|we\s+wish\s+to\s+caution)',
        head,
        re.IGNORECASE,
    ):
        score += 120.0
    if form_key == '20-F' and re.search(r'\bitem\s+3\.?\s+key\s+information\b', head, re.IGNORECASE):
        score += 40.0

    xref = _cross_reference_reason(head)
    if xref:
        score -= 450.0

    if _FORWARD_LOOKING_REJECT.search(head):
        score -= 350.0
    if re.search(r'\btable\s+of\s+contents\b', head_low):
        score -= 200.0
    if re.search(r'\bprincipal\s+risks\s+and\s+opportunities\b', head_low):
        score -= 280.0
    if re.search(r'\bsustainability|esg\s+committee|governance\b', head_low):
        score -= 120.0
    if cue_count < 15:
        score -= 300.0

    score += min(start_pos / max(RISK_CAP, 1), 20.0)
    return score


def extract_risk_section_detail(text: str, form: str = '10-K') -> RiskExtraction:
    form_key = str(form).upper().split('/')[0]  # 10-K/A -> 10-K for markers
    if not text:
        return RiskExtraction('', False, -1_000_000.0, 0, 0, form_key, 0, '', 'empty document')

    if form_key == '20-F':
        end_markers = _20F_END_MARKERS
        start_patterns = _20F_START_PATTERNS
    else:
        end_markers = _10K_END_MARKERS
        start_patterns = _10K_START_PATTERNS

    low = text.lower()
    starts = _find_starts(text, start_patterns)
    best = ''
    best_score = -1_000_000.0
    best_start = 0
    best_end = 0

    for st in starts:
        ends = [low.find(mk, st + 50) for mk in end_markers]
        ends = [e for e in ends if e != -1]
        end = min(ends) if ends else st + RISK_CAP
        chunk = text[st:end][:RISK_CAP]
        score = _candidate_score(chunk, st, form_key)
        if score > best_score:
            best = chunk
            best_score = score
            best_start = st
            best_end = min(end, st + RISK_CAP)

    reject_reason = None
    if best_score <= MIN_QUALITY_SCORE:
        reject_reason = _cross_reference_reason(best[:300]) or 'quality score below threshold'
    if not best:
        mid = len(text) // 2
        best = text[mid:mid + RISK_CAP]
        best_start = mid
        best_end = mid + len(best)
        best_score = _candidate_score(best, best_start, form_key)
        reject_reason = reject_reason or 'no risk heading match; middle fallback'

    heading_preview = best[:200].replace('\n', ' ')
    cue_count = risk_cue_count(best)
    xref = _cross_reference_reason(best[:300])
    ok = (
        best_score >= MIN_QUALITY_SCORE
        and len(best) >= MIN_SECTION_LEN
        and cue_count >= 15
        and xref is None
    )
    if xref:
        reject_reason = xref
    elif not ok and reject_reason is None:
        if len(best) < MIN_SECTION_LEN:
            reject_reason = 'section too short'
        elif cue_count < 15:
            reject_reason = 'insufficient risk language cues'
        else:
            reject_reason = 'quality score below threshold'

    return RiskExtraction(
        section=best[:RISK_CAP],
        ok=ok,
        quality_score=best_score,
        start=best_start,
        end=best_end,
        form=form_key,
        cue_count=cue_count,
        heading_preview=heading_preview,
        reject_reason=reject_reason,
    )


def extract_risk_section(text: str, form: str = '10-K') -> str:
    return extract_risk_section_detail(text, form).section


def looks_like_risk_factors(section: str, quality_score: float | None = None) -> bool:
    if not section or len(section) < MIN_SECTION_LEN:
        return False
    if risk_cue_count(section) < 15:
        return False
    if _cross_reference_reason(section[:300]):
        return False
    score = quality_score if quality_score is not None else _candidate_score(section, 0, '10-K')
    return score >= MIN_QUALITY_SCORE
