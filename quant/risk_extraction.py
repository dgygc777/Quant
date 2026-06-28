"""Risk-factor section extraction from annual and quarterly filing text."""

from __future__ import annotations

import re
from dataclasses import dataclass

RISK_CAP = 25_000
MIN_SECTION_LEN = 3_000
MIN_QUALITY_SCORE = 150.0
TENQ_MIN_BODY_LEN = 2_000
TENQ_RISK_USABLE_LEN = TENQ_MIN_BODY_LEN

# FUTURE: Extend this extraction layer with earnings-call transcript fetch paths.
# They should reuse the same cache/change_score schema, with source='transcript',
# and attach a transcript-tone/guidance-change score beside the filing risk score.

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
_TOC_LINE = re.compile(
    r'(?:risk\s*factors|management[’\']?s\s+discussion\s+and\s+analysis|'
    r'financial\s+condition\s+and\s+results\s+of\s+operations)'
    r'(?:\s+|[.\u2026]{2,})\d{1,4}\b',
    re.IGNORECASE,
)
_TOC_ITEM_LINE = re.compile(
    r'^\s*item\s+(?:\d+[a-z]?|[ivx]+)\b.*(?:\s{2,}|[.\u2026]{2,})\d{1,4}\s*$',
    re.IGNORECASE,
)
_TOC_PAGE_REF_LINE = re.compile(
    r'^\s*(?:item\s+(?:\d+[a-z]?|[ivx]+)\b|'
    r'risk\s+factors\b|management[’\']?s\s+discussion\b|'
    r'analysis\b|quantitative\s+and\s+qualitative\b|'
    r'controls\s+and\s+procedures\b|results\s+of\s+operations\b|'
    r'liquidity\s+and\s+capital\b|critical\s+accounting\b).*'
    r'(?:\bpages?\s*\d|\bnot\s+applicable\b|\d{1,4}\s*$)',
    re.IGNORECASE,
)
_TOC_RIGHT_PAGE_LINE = re.compile(
    r'^\s*[A-Za-z][A-Za-z0-9&(),/’\'\-\s]{2,90}'
    r'(?:\s{2,}|\s+pages?)\d{1,4}(?:-\d{1,4})?\s*$',
    re.IGNORECASE,
)
_INDEX_CONTEXT_LINE = re.compile(
    r'^\s*(?:form\s+10-q\s+cross-reference\s+index|cross-reference\s+index|'
    r'reference\s+index|table\s+of\s+contents|item\s+number\s+item|'
    r'part\s+i{1,2}\b.*|information\s*|'
    r'management[’\']?s\s+discussion\s+and\s*|analysis\s*|\(md&a\)\s*)$',
    re.IGNORECASE,
)
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
_ANNUAL_10K_END_RE = re.compile(
    r'\b(?:item\s+(?:1b|2|3)\b[\.:：\-–—]?|unresolved\s+staff\s+comments\b)',
    re.IGNORECASE,
)
_ANNUAL_20F_END_RE = re.compile(
    r'(?:^|\n|\r)[\s>|│┃║#*\-–—•╭╰╮╯]*'
    r'item\s+[45]\b[\.:：\-–—]?',
    re.IGNORECASE,
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
    section_used: str | None = None
    candidate_lengths: dict[str, int] | None = None


@dataclass
class QuarterlySectionSelection:
    prior_detail: RiskExtraction
    current_detail: RiskExtraction
    section_used: str | None
    ok: bool
    candidate_lengths: dict[str, dict[str, int]]
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
    if not section:
        return False
    head = section[:600].replace('\xa0', ' ')
    if re.search(
        r'(?im)^\s*(?:form\s+10-q\s+cross-reference\s+index|'
        r'cross-reference\s+index|reference\s+index)\s*$',
        head,
    ):
        return True
    if _TOC_LINE.search(head):
        return True
    if _TOC_ITEM_LINE.search(head):
        return True
    if len(section) < _MIN_SLICE and re.search(r'\s{6,}\d{1,4}\b', head):
        return True
    lines = [line.strip() for line in head.splitlines() if line.strip()]
    item_lines = [line for line in lines if _TOC_ITEM_LINE.search(line)]
    page_ref_lines = [line for line in lines if _TOC_PAGE_REF_LINE.search(line)]
    right_page_lines = [line for line in lines if _TOC_RIGHT_PAGE_LINE.search(line)]
    item_heading_lines = [
        line for line in lines
        if re.match(r'\s*item\s+\d+[a-z]?\b', line, re.IGNORECASE)
    ]
    if len(item_lines) >= 2:
        return True
    if len(page_ref_lines) >= 2:
        return True
    if len(right_page_lines) >= 2:
        return True
    if len(item_heading_lines) >= 2 and page_ref_lines:
        return True
    if lines and re.match(r'\s*item\s+\d+[a-z]?\b', lines[0], re.IGNORECASE):
        next_lines = '\n'.join(lines[1:4])
        if re.search(r'^\s*item\s+\d+[a-z]?\b', next_lines, re.IGNORECASE | re.MULTILINE):
            return True
    if len(section) < 700 and re.search(r'\bpart\s+i{1,2}\b', head, re.IGNORECASE) and len(item_lines) >= 1:
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

    anchored_annual_heading = _ANCHORED_10K.search(head) or _ANCHORED_20F.search(head)
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
    if anchored_annual_heading:
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
    if not anchored_annual_heading and re.search(r'\bsustainability|esg\s+committee|governance\b', head_low):
        score -= 120.0
    if cue_count < 15:
        score -= 300.0

    score += min(start_pos / max(RISK_CAP, 1), 20.0)
    return score


def _annual_end_pos(text: str, start: int, form_key: str) -> int:
    """Find the next annual item heading without stopping on inline cross-references."""
    end_re = _ANNUAL_20F_END_RE if form_key == '20-F' else _ANNUAL_10K_END_RE
    match = end_re.search(text, start + 50)
    if match:
        return match.start()
    return start + RISK_CAP


_LINE_PREFIX = r'(?:^|\n|\r)[\s>|│┃║#*\-–—•╭╰╮╯]*'
_ITEM_PUNCT = r'[\.:：\-–—]?'
_PART_RE = re.compile(_LINE_PREFIX + r'part\s+(i{1,2})\b' + _ITEM_PUNCT, re.IGNORECASE)
_ITEM_RE = re.compile(
    _LINE_PREFIX + r'item\s+(1a|1b|2|3|4|5|6)\b' + _ITEM_PUNCT + r'\s*(?:[^\n\r]*)',
    re.IGNORECASE,
)
_TENQ_ITEM1A_RE = re.compile(
    _LINE_PREFIX + r'item\s+1a\b' + _ITEM_PUNCT + r'\s*(?:risk\s+factors)?',
    re.IGNORECASE,
)
_TENQ_MDA_RE = re.compile(
    _LINE_PREFIX + r'item\s+2\b' + _ITEM_PUNCT
    + r'\s*(?:management[’\']?s\s+discussion\s+and\s+analysis|md&a)?',
    re.IGNORECASE,
)
_TENQ_ITEM3_4_RE = re.compile(_LINE_PREFIX + r'item\s+[34]\b' + _ITEM_PUNCT, re.IGNORECASE)
_TENQ_PART_II_RE = re.compile(_LINE_PREFIX + r'part\s+ii\b' + _ITEM_PUNCT, re.IGNORECASE)
_TENQ_PART_I_END_RE = re.compile(
    _LINE_PREFIX + r'(?:item\s+[34]\b' + _ITEM_PUNCT + r'|part\s+ii\b' + _ITEM_PUNCT + r')',
    re.IGNORECASE,
)
_TENQ_STANDALONE_MDA_RE = re.compile(
    _LINE_PREFIX
    + r'management[’\']?s\s+discussion\s+and\s+analysis',
    re.IGNORECASE,
)
_TENQ_STANDALONE_RISK_RE = re.compile(
    _LINE_PREFIX + r'risk\s+factors\b',
    re.IGNORECASE,
)
_TENQ_MARKET_RISK_RE = re.compile(
    _LINE_PREFIX + r'quantitative\s+and\s+qualitative\s+disclosures\s+about\s+market\s+risk\b',
    re.IGNORECASE,
)
_TENQ_CONTROLS_RE = re.compile(
    _LINE_PREFIX + r'controls\s+and\s+procedures\b',
    re.IGNORECASE,
)
_TENQ_RISK_AND_OTHER_RE = re.compile(
    _LINE_PREFIX + r'risk\s+factors\s+and\s+other\s+key\s+information\b',
    re.IGNORECASE,
)
_TENQ_ISSUER_PURCHASES_RE = re.compile(
    _LINE_PREFIX + r'issuer\s+purchases\s+of\s+equity\s+securities\b',
    re.IGNORECASE,
)
_MDA_HEADING_RE = re.compile(
    r'management[’\']?s\s+discussion\s+and\s+analysis|'
    r'\bmd&a\b|'
    r'financial\s+condition\s+and\s+results\s+of\s+operations',
    re.IGNORECASE,
)
_MDA_CUES = [
    'revenue', 'net sales', 'gross margin', 'operating income', 'liquidity',
    'cash flow', 'inventory', 'demand', 'customer', 'backlog', 'cost of',
    'results of operations', 'financial condition',
]


def _part_spans(text: str) -> dict[str, tuple[int, int]]:
    matches = []
    for match in _PART_RE.finditer(text):
        label = match.group(1).upper()
        if label in {'I', 'II'}:
            matches.append((label, match.start()))
    spans: dict[str, tuple[int, int]] = {}
    for idx, (label, start) in enumerate(matches):
        end = matches[idx + 1][1] if idx + 1 < len(matches) else len(text)
        spans.setdefault(label, (start, end))
    return spans


def _line_spans(text: str):
    pos = 0
    for line in text.splitlines(True):
        end = pos + len(line)
        yield pos, end, line
        pos = end
    if pos < len(text):
        yield pos, len(text), text[pos:]


def _toc_spans(text: str) -> list[tuple[int, int]]:
    """Locate likely table-of-contents regions so body matching can ignore them."""
    scan_limit = len(text)
    lines = [(st, en, line) for st, en, line in _line_spans(text[:scan_limit])]
    item_idxs = [
        idx for idx, (_, _, line) in enumerate(lines)
        if (
            _TOC_ITEM_LINE.search(line.replace('\xa0', ' '))
            or _TOC_PAGE_REF_LINE.search(line.replace('\xa0', ' '))
            or _TOC_RIGHT_PAGE_LINE.search(line.replace('\xa0', ' '))
        )
    ]
    if not item_idxs:
        return []

    runs: list[list[int]] = []
    run: list[int] = []
    prev = None
    for idx in item_idxs:
        if prev is None or idx - prev <= 12:
            run.append(idx)
        else:
            runs.append(run)
            run = [idx]
        prev = idx
    runs.append(run)
    viable_runs = [candidate for candidate in runs if len(candidate) >= 2]
    if not viable_runs:
        return []

    spans = []
    for toc_run in viable_runs:
        first_line = toc_run[0]
        last_line = toc_run[-1]
        start_line = first_line
        while start_line > 0 and first_line - start_line <= 8:
            prev_line = lines[start_line - 1][2].replace('\xa0', ' ')
            if _INDEX_CONTEXT_LINE.match(prev_line):
                start_line -= 1
                continue
            if re.match(r'^\s*$', prev_line):
                start_line -= 1
                continue
            break
        start = lines[start_line][0]
        end_line = last_line
        while end_line + 1 < len(lines):
            next_line = lines[end_line + 1][2].replace('\xa0', ' ')
            if re.match(r'^\s*$', next_line) or _INDEX_CONTEXT_LINE.match(next_line):
                end_line += 1
                continue
            break
        spans.append((start, lines[end_line][1]))
    return spans


def _mask_spans(text: str, spans: list[tuple[int, int]]) -> str:
    if not spans:
        return text
    chars = list(text)
    for start, end in spans:
        for idx in range(max(0, start), min(len(chars), end)):
            if chars[idx] not in '\r\n':
                chars[idx] = ' '
    return ''.join(chars)


def _strip_toc_for_matching(text: str) -> str:
    return _mask_spans(text, _toc_spans(text))


def _all_part_spans(search_text: str) -> list[tuple[str, tuple[int, int]]]:
    matches = []
    for match in _PART_RE.finditer(search_text):
        label = match.group(1).upper()
        if label in {'I', 'II'}:
            matches.append((label, match.start()))
    spans = []
    for idx, (label, start) in enumerate(matches):
        end = matches[idx + 1][1] if idx + 1 < len(matches) else len(search_text)
        spans.append((label, (start, end)))
    return spans


def _slice_part_item(
    source_text: str,
    search_text: str,
    part_span: tuple[int, int] | None,
    item_re: re.Pattern,
    *,
    end_re: re.Pattern | None = None,
    next_part_item: bool = False,
) -> list[tuple[str, int, int]]:
    if part_span is None:
        return []
    part_start, part_end = part_span
    part_search = search_text[part_start:part_end]
    slices = []
    for match in item_re.finditer(part_search):
        start = part_start + match.start()
        search_from = match.end()
        rel_end = len(part_search)
        if end_re is not None:
            end_match = end_re.search(part_search, search_from)
            if end_match:
                rel_end = min(rel_end, end_match.start())
        if next_part_item:
            for item_match in _ITEM_RE.finditer(part_search, search_from):
                item = item_match.group(1).lower()
                if item != '1a':
                    rel_end = min(rel_end, item_match.start())
                    break
        end = part_start + rel_end
        slices.append((source_text[start:end].strip(), start, end))
    return slices


def _normalized_head(section: str, n: int = 600) -> str:
    return section[:n].replace('\xa0', ' ')


def _is_quarterly_toc_slice(section: str) -> bool:
    if not section:
        return False
    head = _normalized_head(section)
    if _is_toc_slice(section):
        return True
    lines = [line.strip() for line in head.splitlines() if line.strip()]
    first = lines[0] if lines else head.strip()
    if 'table of contents' in head.lower():
        return True
    if len(section) < 600 and re.search(r'\s{2,}\d{1,4}\s*$', first):
        return True
    if len(section) < 600 and re.search(
        r'(risk\s+factors|management[’\']?s\s+discussion\s+and\s+analysis|'
        r'financial\s+condition\s+and\s+results\s+of\s+operations)\s+\d{1,4}\s*$',
        head,
        re.IGNORECASE,
    ):
        return True
    if len(section) < 250 and len(lines) <= 2 and not re.search(r'[.;:]\s+\w+', head):
        return True
    return False


def _mda_candidate_score(section: str, start_pos: int) -> float:
    if not section:
        return -1_000_000.0
    if _is_quarterly_toc_slice(section):
        return -1_000_000.0
    head = _normalized_head(section, _HEAD_WINDOW)
    low = section.lower()
    score = float(min(len(section), RISK_CAP)) / 50.0
    if _MDA_HEADING_RE.search(head):
        score += 500.0
    if re.search(r'(?:^|\n|\r)\s*item\s+2\b', head, re.IGNORECASE):
        score += 120.0
    score += sum(low.count(cue) for cue in _MDA_CUES) * 8.0
    if len(section) < 200:
        score -= 1_000.0
    score += min(start_pos / max(RISK_CAP, 1), 20.0)
    return score


def _quarterly_candidate_score(section: str, start_pos: int, section_used: str) -> float:
    if section_used == 'mda':
        return _mda_candidate_score(section, start_pos)
    if _is_quarterly_toc_slice(section):
        return -1_000_000.0
    return _candidate_score(section, start_pos, '10-Q')


def _best_part_item_candidate(
    source_text: str,
    search_text: str,
    part_label: str,
    item_re: re.Pattern,
    section_used: str,
    *,
    end_re: re.Pattern | None = None,
    next_part_item: bool = False,
) -> tuple[str, int, int]:
    best = ''
    best_start = 0
    best_end = 0
    best_len = -1
    best_fallback_len = -1
    fallback = ('', 0, 0)
    for label, span in _all_part_spans(search_text):
        if label != part_label:
            continue
        for section, start, end in _slice_part_item(
            source_text,
            search_text,
            span,
            item_re,
            end_re=end_re,
            next_part_item=next_part_item,
        ):
            section_len = len(section)
            if section_len > best_fallback_len:
                fallback = (section, start, end)
                best_fallback_len = section_len
            if _is_quarterly_toc_slice(section):
                continue
            if section_len > best_len:
                best = section
                best_start = start
                best_end = end
                best_len = section_len
    if not best and fallback[0]:
        return fallback
    return best, best_start, best_end


def _next_marker_pos(text: str, start: int, markers: list[re.Pattern]) -> int:
    positions = []
    for marker in markers:
        match = marker.search(text, start)
        if match:
            positions.append(match.start())
    return min(positions) if positions else len(text)


def _global_item_slices(
    source_text: str,
    search_text: str,
    item_re: re.Pattern,
    end_markers: list[re.Pattern],
) -> list[tuple[str, int, int]]:
    slices = []
    for match in item_re.finditer(search_text):
        start = match.start()
        end = _next_marker_pos(search_text, match.end(), end_markers)
        slices.append((source_text[start:end].strip(), start, end))
    return slices


def _best_global_item_candidate(
    source_text: str,
    search_text: str,
    item_re: re.Pattern,
    section_used: str,
    end_markers: list[re.Pattern],
) -> tuple[str, int, int]:
    best = ''
    best_start = 0
    best_end = 0
    best_len = -1
    best_fallback_len = -1
    fallback = ('', 0, 0)
    for section, start, end in _global_item_slices(source_text, search_text, item_re, end_markers):
        section_len = len(section)
        if section_len > best_fallback_len:
            fallback = (section, start, end)
            best_fallback_len = section_len
        if _is_quarterly_toc_slice(section):
            continue
        if section_len > best_len:
            best = section
            best_start = start
            best_end = end
            best_len = section_len
    if not best and fallback[0]:
        return fallback
    return best, best_start, best_end


def _tenq_detail(
    section: str,
    start: int,
    end: int,
    section_used: str,
    candidate_lengths: dict[str, int],
) -> RiskExtraction:
    section = section[:RISK_CAP]
    cue_count = risk_cue_count(section)
    toc_like = _is_quarterly_toc_slice(section)
    if section_used == 'risk_factors':
        score = _candidate_score(section, start, '10-Q')
        ok = len(section.strip()) >= TENQ_MIN_BODY_LEN and not toc_like
        if ok:
            reject_reason = None
        elif toc_like:
            reject_reason = 'quarterly risk section matched table of contents'
        elif section:
            reject_reason = 'quarterly risk section below body-length threshold'
        else:
            reject_reason = 'quarterly risk section not found'
    else:
        score = _mda_candidate_score(section, start)
        ok = len(section.strip()) >= TENQ_MIN_BODY_LEN and not toc_like
        if ok:
            reject_reason = None
        elif toc_like:
            reject_reason = 'quarterly MD&A matched table of contents'
        elif section:
            reject_reason = 'quarterly MD&A below body-length threshold'
        else:
            reject_reason = 'quarterly MD&A section not found'
    return RiskExtraction(
        section=section,
        ok=ok,
        quality_score=score,
        start=start,
        end=end,
        form='10-Q',
        cue_count=cue_count,
        heading_preview=section[:200].replace('\n', ' '),
        reject_reason=reject_reason,
        section_used=section_used,
        candidate_lengths=dict(candidate_lengths),
    )


def extract_quarterly_section_candidates(text: str) -> dict[str, RiskExtraction]:
    """Return part-aware 10-Q Item 1A and Part I Item 2 candidates."""
    candidate_lengths = {'risk_factors': 0, 'mda': 0}
    if not text:
        empty = _tenq_detail('', 0, 0, 'risk_factors', candidate_lengths)
        return {'risk_factors': empty, 'mda': _tenq_detail('', 0, 0, 'mda', candidate_lengths)}

    search_text = _strip_toc_for_matching(text)
    risk, risk_start, risk_end = _best_part_item_candidate(
        text,
        search_text,
        'II',
        _TENQ_ITEM1A_RE,
        'risk_factors',
        next_part_item=True,
    )
    mda, mda_start, mda_end = _best_part_item_candidate(
        text,
        search_text,
        'I',
        _TENQ_MDA_RE,
        'mda',
        end_re=_TENQ_PART_I_END_RE,
    )
    if not risk or len(risk) < TENQ_MIN_BODY_LEN or _is_quarterly_toc_slice(risk):
        risk, risk_start, risk_end = _best_global_item_candidate(
            text,
            search_text,
            _TENQ_ITEM1A_RE,
            'risk_factors',
            [_ITEM_RE],
        )
    if not risk or len(risk) < TENQ_MIN_BODY_LEN or _is_quarterly_toc_slice(risk):
        risk, risk_start, risk_end = _best_global_item_candidate(
            text,
            search_text,
            _TENQ_STANDALONE_RISK_RE,
            'risk_factors',
            [
                _TENQ_MARKET_RISK_RE,
                _TENQ_CONTROLS_RE,
                _TENQ_ISSUER_PURCHASES_RE,
                _ITEM_RE,
                _TENQ_PART_II_RE,
            ],
        )
    if not mda or len(mda) < TENQ_MIN_BODY_LEN or _is_quarterly_toc_slice(mda):
        mda, mda_start, mda_end = _best_global_item_candidate(
            text,
            search_text,
            _TENQ_MDA_RE,
            'mda',
            [_TENQ_ITEM3_4_RE, _TENQ_PART_II_RE],
        )
    if not mda or len(mda) < TENQ_MIN_BODY_LEN or _is_quarterly_toc_slice(mda):
        mda, mda_start, mda_end = _best_global_item_candidate(
            text,
            search_text,
            _TENQ_STANDALONE_MDA_RE,
            'mda',
            [
                _TENQ_RISK_AND_OTHER_RE,
                _TENQ_MARKET_RISK_RE,
                _TENQ_CONTROLS_RE,
                _TENQ_ITEM3_4_RE,
                _TENQ_PART_II_RE,
            ],
        )
    candidate_lengths = {'risk_factors': len(risk), 'mda': len(mda)}
    return {
        'risk_factors': _tenq_detail(risk, risk_start, risk_end, 'risk_factors', candidate_lengths),
        'mda': _tenq_detail(mda, mda_start, mda_end, 'mda', candidate_lengths),
    }


def select_quarterly_comparison_sections(
    prior_text: str,
    current_text: str,
) -> QuarterlySectionSelection:
    """Align quarterly filings to risk-vs-risk or MD&A-vs-MD&A before scoring."""
    prior = extract_quarterly_section_candidates(prior_text)
    current = extract_quarterly_section_candidates(current_text)
    candidate_lengths = {
        'prior': dict(prior['risk_factors'].candidate_lengths or {}),
        'current': dict(current['risk_factors'].candidate_lengths or {}),
    }
    prior_risk_len = candidate_lengths['prior'].get('risk_factors', 0)
    current_risk_len = candidate_lengths['current'].get('risk_factors', 0)

    prior_detail = prior['risk_factors']
    current_detail = current['risk_factors']
    if (
        prior_risk_len >= TENQ_RISK_USABLE_LEN
        and current_risk_len >= TENQ_RISK_USABLE_LEN
        and prior_detail.ok
        and current_detail.ok
    ):
        return QuarterlySectionSelection(
            prior_detail=prior_detail,
            current_detail=current_detail,
            section_used='risk_factors',
            ok=prior_detail.ok and current_detail.ok,
            candidate_lengths=candidate_lengths,
        )

    prior_mda = prior['mda']
    current_mda = current['mda']
    if prior_mda.ok and current_mda.ok:
        prior_mda.section_used = 'mda_both'
        current_mda.section_used = 'mda_both'
        return QuarterlySectionSelection(
            prior_detail=prior_mda,
            current_detail=current_mda,
            section_used='mda_both',
            ok=True,
            candidate_lengths=candidate_lengths,
        )

    fallback_prior = prior_mda if prior_mda.section else prior['risk_factors']
    fallback_current = current_mda if current_mda.section else current['risk_factors']
    return QuarterlySectionSelection(
        prior_detail=fallback_prior,
        current_detail=fallback_current,
        section_used=None,
        ok=False,
        candidate_lengths=candidate_lengths,
        reject_reason='non-comparable sections',
    )


def extract_risk_section_detail(text: str, form: str = '10-K') -> RiskExtraction:
    form_key = str(form).upper().split('/')[0]  # 10-K/A -> 10-K for markers
    if not text:
        return RiskExtraction('', False, -1_000_000.0, 0, 0, form_key, 0, '', 'empty document')

    if form_key == '10-Q':
        candidates = extract_quarterly_section_candidates(text)
        risk_detail = candidates['risk_factors']
        if risk_detail.ok:
            return risk_detail
        mda_detail = candidates['mda']
        if mda_detail.ok:
            return mda_detail
        if risk_detail.section:
            return risk_detail
        return mda_detail

    if form_key == '20-F':
        start_patterns = _20F_START_PATTERNS
    else:
        start_patterns = _10K_START_PATTERNS

    starts = _find_starts(text, start_patterns)
    best = ''
    best_score = -1_000_000.0
    best_start = 0
    best_end = 0

    for st in starts:
        end = _annual_end_pos(text, st, form_key)
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
