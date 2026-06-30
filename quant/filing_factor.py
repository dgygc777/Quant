"""
Point-in-time filing-risk factor.

Turns cached 10-K/20-F risk-language change scores into a ticker x date panel
that can be used as a NEGATIVE filter on a long-only book (exclude or
half-weight names whose latest filing flagged worsening risk language), and
walk-forward validated like any other signal.

Design principles
-----------------
* No look-ahead. A filing scored on ``filing_date`` only becomes active on the
  NEXT trading day (filing_date + 1), and stays active until a newer filing for
  that ticker supersedes it. The score panel at date d reflects only filings
  publicly available by close of d-1.
* Negative filter only. Filing risk is used to AVOID names, never to size up.
* Honest about evidence. Because annual filings update only ~once per year, a
  cache must contain a multi-year history across several names before the factor
  can be validated at all. ``filing_data_sufficiency`` gates this explicitly and
  the report refuses to claim validation on too-thin data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_FILING_THRESHOLD = -0.30
FILING_ACTIONS = ('exclude', 'half_weight')
DEFAULT_FILING_ACTION = 'exclude'
DEFAULT_ACTIVATION_LAG = 1  # trading days after filing_date before the score is live

# Sufficiency thresholds for "can this factor even be validated?"
# Cadence-aware: 10-Ks are annual (~1 update/name/yr) so an annual-only cache
# needs many years; 10-Q quarterly scoring (~4 updates/name/yr) reaches the same
# event count in far less calendar time. We require enough INDEPENDENT updates
# per name AND enough calendar span to cover more than one regime.
MIN_EVENTS_PER_NAME = 4
MIN_NAMES_WITH_HISTORY = 4
MIN_HISTORY_YEARS = 3.0
MIN_TOTAL_EVENTS = 20


@dataclass
class FilingEvent:
    ticker: str
    filing_date: pd.Timestamp
    score: float
    ok: bool
    source: str = 'annual'  # 'annual' (10-K/20-F) or 'quarterly' (10-Q)


def load_filing_events(cache_path: str = 'tenk_cache.json') -> dict[str, list[FilingEvent]]:
    """Parse the tenk cache into per-ticker filing events sorted by date.

    Only entries with ``ok`` true and a numeric ``change_score`` are kept (stale
    / failed extractions carry no usable signal). Returns {ticker: [events...]}.
    """
    try:
        with open(cache_path, 'r') as fh:
            raw = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return load_filing_events_from_dict(raw)


def load_filing_events_from_dict(raw: dict) -> dict[str, list[FilingEvent]]:
    by_ticker: dict[str, list[FilingEvent]] = {}
    for key, value in raw.items():
        if not isinstance(value, dict):
            continue
        ticker = key.split(':', 1)[0].strip().upper()
        date_str = value.get('current_filing_date') or (
            key.split(':', 1)[1] if ':' in key else None
        )
        if not date_str:
            continue
        try:
            filing_date = pd.Timestamp(date_str)
        except (ValueError, TypeError):
            continue
        score = value.get('change_score')
        ok = bool(value.get('ok', False)) and isinstance(score, (int, float)) and not pd.isna(score)
        source = str(value.get('source') or 'annual').lower()
        by_ticker.setdefault(ticker, []).append(
            FilingEvent(ticker, filing_date, float(score) if ok else float('nan'), ok, source)
        )
    for events in by_ticker.values():
        events.sort(key=lambda e: e.filing_date)
    return by_ticker


def build_filing_score_panel(
    events_by_ticker: dict[str, list[FilingEvent]],
    index: pd.DatetimeIndex,
    columns: list[str],
    activation_lag: int = DEFAULT_ACTIVATION_LAG,
) -> pd.DataFrame:
    """Build a point-in-time ticker x date score panel (no look-ahead).

    A filing's score activates ``activation_lag`` trading days after its filing
    date and persists until a newer filing supersedes it. Dates before the first
    activation (or names with no valid filing) are NaN = "no filing info".
    """
    if activation_lag < 0:
        raise ValueError('activation_lag must be >= 0')
    panel = pd.DataFrame(np.nan, index=index, columns=columns)
    for ticker in columns:
        events = events_by_ticker.get(ticker, [])
        col = panel.columns.get_loc(ticker)
        for event in events:
            if not event.ok:
                continue
            # First trading day strictly after filing_date, then +(_lag - 1) more.
            pos = index.searchsorted(event.filing_date, side='right')
            pos += max(0, activation_lag - 1)
            if pos >= len(index):
                continue
            panel.iloc[pos:, col] = event.score  # newer filings overwrite from activation
    return panel


def filing_data_sufficiency(
    events_by_ticker: dict[str, list[FilingEvent]],
    columns: list[str] | None = None,
    *,
    min_events_per_name: int = MIN_EVENTS_PER_NAME,
    min_names_with_history: int = MIN_NAMES_WITH_HISTORY,
    min_history_years: float = MIN_HISTORY_YEARS,
    min_total_events: int = MIN_TOTAL_EVENTS,
) -> dict:
    """Assess whether the cache has enough history to validate the factor."""
    if columns is not None:
        events_by_ticker = {t: events_by_ticker.get(t, []) for t in columns}
    per_name = {}
    all_dates: list[pd.Timestamp] = []
    total_events = 0
    source_counts: dict[str, int] = {}
    spacings_days: list[float] = []
    for ticker, events in events_by_ticker.items():
        valid = [e for e in events if e.ok]
        per_name[ticker] = len(valid)
        total_events += len(valid)
        for e in valid:
            all_dates.append(e.filing_date)
            source_counts[e.source] = source_counts.get(e.source, 0) + 1
        dates = sorted(e.filing_date for e in valid)
        spacings_days.extend((b - a).days for a, b in zip(dates, dates[1:]))
    names_with_history = sum(1 for n in per_name.values() if n >= min_events_per_name)
    if all_dates:
        span_years = (max(all_dates) - min(all_dates)).days / 365.25
        first_date, last_date = min(all_dates), max(all_dates)
    else:
        span_years, first_date, last_date = 0.0, None, None
    median_events = float(np.median(list(per_name.values()))) if per_name else 0.0
    median_spacing = float(np.median(spacings_days)) if spacings_days else float('nan')
    has_quarterly = source_counts.get('quarterly', 0) > 0

    reasons: list[str] = []
    if names_with_history < min_names_with_history:
        reasons.append(
            f'only {names_with_history} name(s) have >= {min_events_per_name} '
            f'scored updates; need {min_names_with_history}'
        )
    if span_years < min_history_years:
        reasons.append(
            f'history spans {span_years:.1f}y; need >= {min_history_years:.0f}y'
        )
    if total_events < min_total_events:
        reasons.append(f'only {total_events} total scored updates; need {min_total_events}')

    return {
        'validatable': not reasons,
        'reasons': reasons,
        'total_events': total_events,
        'names_total': len(per_name),
        'names_with_history': names_with_history,
        'median_events_per_name': median_events,
        'span_years': span_years,
        'first_filing': first_date,
        'last_filing': last_date,
        'events_per_name': per_name,
        'source_counts': source_counts,
        'median_spacing_days': median_spacing,
        'has_quarterly': has_quarterly,
    }


def apply_filing_filter(
    weights: pd.DataFrame,
    score_panel: pd.DataFrame,
    *,
    threshold: float = DEFAULT_FILING_THRESHOLD,
    action: str = DEFAULT_FILING_ACTION,
    renormalize: bool = True,
) -> pd.DataFrame:
    """Down-weight long positions in names whose active filing score <= threshold.

    ``exclude`` zeroes the position; ``half_weight`` halves it. With
    ``renormalize`` the long book is rescaled back to its original per-day gross
    (so the filter is a pure tilt, not a market-timing/cash call). No look-ahead:
    the score panel is point-in-time and only aligned, never shifted forward.
    """
    if action not in FILING_ACTIONS:
        raise ValueError(f'action must be one of {FILING_ACTIONS}')
    aligned = score_panel.reindex(index=weights.index, columns=weights.columns)
    flagged = aligned <= threshold  # NaN compares False -> no filter where no filing
    filtered = weights.copy()
    if action == 'exclude':
        filtered = filtered.mask(flagged & (weights > 0), 0.0)
    else:
        half = filtered.where(~(flagged & (weights > 0)), filtered * 0.5)
        filtered = half
    if renormalize:
        before = weights.where(weights > 0, 0.0).sum(axis=1)
        after = filtered.where(filtered > 0, 0.0).sum(axis=1)
        scale = (before / after).where(after > 1e-12, 0.0)
        pos = filtered.where(filtered > 0, 0.0).mul(scale, axis=0)
        neg = filtered.where(filtered < 0, 0.0)  # leave any short leg untouched
        filtered = pos + neg
    return filtered


def make_filing_filtered_strategy(
    score_panel: pd.DataFrame,
    *,
    threshold: float = DEFAULT_FILING_THRESHOLD,
    action: str = DEFAULT_FILING_ACTION,
    cost: float = 0.0005,
    score_mode: str = 'raw_momentum',
    beta_window: int = 126,
):
    """Long-only XS strategy with a point-in-time filing-risk negative filter."""
    from quant.models.cross_sectional import (
        build_weights,
        compute_scores,
        portfolio_returns,
    )

    def _strategy(panel: pd.DataFrame, **kw) -> pd.Series:
        scores = compute_scores(
            panel, mode='momentum',
            lookback=int(kw.get('lookback', 126)),
            skip=int(kw.get('skip', 21)),
            score_mode=kw.get('score_mode', score_mode),
            beta_window=int(kw.get('beta_window', beta_window)),
        )
        weights = build_weights(
            panel, scores,
            top_frac=float(kw.get('top_frac', 0.25)),
            rebalance=int(kw.get('rebalance', 5)),
            market_neutral=False,
        )
        weights = apply_filing_filter(
            weights, score_panel, threshold=threshold, action=action,
        )
        rets = panel.pct_change(fill_method=None)
        return portfolio_returns(weights, rets, cost)['strat_net']

    return _strategy


def print_filing_sufficiency(suff: dict) -> None:
    print('\n=== Filing-risk factor: data sufficiency ===')
    fb = suff['first_filing'].date() if suff['first_filing'] is not None else 'n/a'
    lb = suff['last_filing'].date() if suff['last_filing'] is not None else 'n/a'
    counts = suff.get('source_counts', {})
    mix = ', '.join(f'{k}={v}' for k, v in sorted(counts.items())) or 'none'
    spacing = suff.get('median_spacing_days', float('nan'))
    spacing_str = f'{spacing:.0f}d' if spacing == spacing else 'n/a'  # NaN check
    print(f'Scored updates: {suff["total_events"]} across {suff["names_total"]} names '
          f'({fb} -> {lb}, {suff["span_years"]:.1f}y span)')
    print(f'Cadence: {mix}  median update spacing: {spacing_str}')
    print(f'Names with >= {MIN_EVENTS_PER_NAME} updates: {suff["names_with_history"]}  '
          f'median updates/name: {suff["median_events_per_name"]:.0f}')
    if suff['validatable']:
        print('Verdict: SUFFICIENT — enough history to attempt walk-forward validation.')
    else:
        print('Verdict: INSUFFICIENT — cannot validate the filing factor yet:')
        for reason in suff['reasons']:
            print(f'  - {reason}')
        print('  The cache is a current-snapshot overlay, not a historical factor dataset.')
        if not suff.get('has_quarterly'):
            print('  Annual 10-Ks update only ~1x/name/yr. To densify to ~4x/name/yr, '
                  'add quarterly 10-Q scoring:')
            print('    python3 tenk_reader.py --tickers <T1,T2,...> --filing-type both')
        print('  (EDGAR holds the raw 10-K/20-F/10-Q filings back decades; scoring the '
              'historical pairs builds the multi-year series needed to validate.)')


def report_filing_factor_validation(
    panel: pd.DataFrame,
    xs_params: dict,
    *,
    cache_path: str = 'tenk_cache.json',
    events_by_ticker: dict[str, list[FilingEvent]] | None = None,
    threshold: float = DEFAULT_FILING_THRESHOLD,
    action: str = DEFAULT_FILING_ACTION,
    activation_lag: int = DEFAULT_ACTIVATION_LAG,
    cost: float = 0.0005,
    train: int = 504,
    test: int = 63,
    warmup: int | None = None,
    select: str = 'active_ir',
) -> dict:
    """Sufficiency-gated walk-forward report for the filing-risk negative filter.

    Always prints the data-sufficiency assessment first. Only runs (and reports)
    base-vs-filtered walk-forward validation when the cache clears the gate.
    """
    if events_by_ticker is None:
        events_by_ticker = load_filing_events(cache_path)
    suff = filing_data_sufficiency(events_by_ticker, columns=list(panel.columns))
    print_filing_sufficiency(suff)

    score_panel = build_filing_score_panel(
        events_by_ticker, panel.index, list(panel.columns), activation_lag=activation_lag,
    )
    active_cells = int((~score_panel.isna()).to_numpy().sum())
    flagged_cells = int((score_panel <= threshold).to_numpy().sum())
    print(f'Point-in-time panel: {active_cells} active ticker-days, '
          f'{flagged_cells} flagged at threshold {threshold:+.2f} '
          f'(action={action}, activation_lag={activation_lag}d)')

    out = {'sufficiency': suff, 'score_panel': score_panel}
    if not suff['validatable']:
        print('Filing-factor validation SKIPPED — see sufficiency verdict above.')
        return out

    from validate_cross_sectional import (
        active_oos_returns,
        equal_weight_oos_returns,
        information_ratio_ci,
        make_xs_strategy,
        validation_verdict,
        validation_verdict_reason,
    )
    from quant.metrics import metrics
    from quant.validation import walk_forward

    if warmup is None:
        warmup = max(40, int(xs_params.get('lookback', 126)) + int(xs_params.get('skip', 21)) + 5)
    grid = {k: [xs_params[k]] for k in ('lookback', 'skip', 'top_frac', 'rebalance') if k in xs_params}
    score_mode = xs_params.get('score_mode', 'raw_momentum')
    beta_window = int(xs_params.get('beta_window', 126))

    base_fn = make_xs_strategy('long_only', cost=cost, score_mode=score_mode, beta_window=beta_window)
    filt_fn = make_filing_filtered_strategy(
        score_panel, threshold=threshold, action=action, cost=cost,
        score_mode=score_mode, beta_window=beta_window,
    )
    common = dict(train=train, test=test, warmup=warmup, select=select)
    base_wf = walk_forward(base_fn, panel, grid, **common)
    filt_wf = walk_forward(filt_fn, panel, grid, **common)

    def _summary(wf):
        oos = wf['oos_metrics']
        bench = equal_weight_oos_returns(panel, wf['oos_returns'].index)
        active = active_oos_returns(wf['oos_returns'], bench)
        ir, lo, hi, _se = information_ratio_ci(active)
        verdict = validation_verdict(oos['sharpe'], ir, len(wf['folds']),
                                     ci_lower=lo, ci_upper=hi, selection_threshold=0.0)
        return oos, ir, lo, hi, verdict

    base_oos, base_ir, *_ = _summary(base_wf)
    filt_oos, filt_ir, filt_lo, filt_hi, filt_verdict = _summary(filt_wf)

    print('\n=== Filing-risk filter walk-forward (long-only, vs equal-weight benchmark) ===')
    print(f"{'Book':28}{'OOS Sharpe':>11}{'AnnRet':>9}{'ActiveIR':>10}")
    print(f"{'Base (no filter)':28}{base_oos['sharpe']:>11.2f}{base_oos['ann_return']:>9.1%}{base_ir:>+10.2f}")
    print(f"{'Filing-filtered':28}{filt_oos['sharpe']:>11.2f}{filt_oos['ann_return']:>9.1%}{filt_ir:>+10.2f}")
    print(f'Active IR delta (filtered - base): {filt_ir - base_ir:+.2f}')
    print(f'Filtered verdict: {filt_verdict}')
    if filt_ir <= base_ir:
        print('  -> The filing filter did NOT improve benchmark-relative performance OOS.')
    out.update({'base_wf': base_wf, 'filtered_wf': filt_wf,
                'base_ir': base_ir, 'filtered_ir': filt_ir})
    return out
