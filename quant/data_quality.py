"""Shared data-quality gates for cross-sectional panels."""

from __future__ import annotations

import pandas as pd


MIN_COVERAGE = 0.60
EXTREME_DAILY_RETURN = 0.35


def coverage_by_ticker(panel: pd.DataFrame) -> pd.Series:
    """Return each ticker's non-NaN fraction over the panel window."""
    if not isinstance(panel, pd.DataFrame):
        raise TypeError('panel must be a pandas DataFrame.')
    return panel.notna().mean()


def coverage_ok_by_ticker(
    panel: pd.DataFrame,
    min_coverage: float = MIN_COVERAGE,
) -> pd.Series:
    """Return a boolean coverage gate for each ticker."""
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError('min_coverage must be in [0, 1].')
    return coverage_by_ticker(panel) >= min_coverage


def filter_panel_by_coverage(
    panel: pd.DataFrame,
    min_coverage: float = MIN_COVERAGE,
    coverage: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Drop tickers whose non-NaN coverage is below min_coverage."""
    coverage = (
        coverage.reindex(panel.columns).fillna(0.0)
        if coverage is not None
        else coverage_by_ticker(panel)
    )
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError('min_coverage must be in [0, 1].')
    keep = coverage >= min_coverage
    filtered = panel.loc[:, keep].dropna(how='all')
    dropped = coverage.loc[~keep].sort_values()
    return filtered, coverage, dropped


def format_coverage(items: pd.Series) -> str:
    """Format ticker coverage percentages for console reports."""
    return ', '.join(f'{ticker} {cov:.0%}' for ticker, cov in items.items())


def winsorize_extreme_returns(
    prices: pd.DataFrame,
    limit: float = EXTREME_DAILY_RETURN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clip one-day price returns beyond +/-limit and rebuild affected paths.

    This keeps split artifacts or single-day data glitches from dominating
    ranks and covariance estimates. The returned records DataFrame lists each
    ticker-date that was clipped.
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError('prices must be a pandas DataFrame.')
    if limit <= 0:
        raise ValueError('limit must be positive.')

    adjusted = prices.astype(float).copy()
    records: list[dict] = []
    for ticker in adjusted.columns:
        valid = adjusted[ticker].dropna()
        if len(valid) < 2:
            continue
        returns = valid.pct_change(fill_method=None)
        clipped_returns = returns.clip(lower=-limit, upper=limit)
        mask = returns.abs() > limit
        if not bool(mask.any()):
            continue

        for date, original in returns.loc[mask].items():
            records.append({
                'date': date,
                'ticker': str(ticker),
                'original_return': float(original),
                'clipped_return': float(clipped_returns.loc[date]),
            })

        factors = 1.0 + clipped_returns
        factors.iloc[0] = 1.0
        rebuilt = valid.iloc[0] * factors.cumprod()
        adjusted.loc[rebuilt.index, ticker] = rebuilt

    clipped = pd.DataFrame.from_records(
        records,
        columns=['date', 'ticker', 'original_return', 'clipped_return'],
    )
    if not clipped.empty:
        clipped = clipped.sort_values(['date', 'ticker']).reset_index(drop=True)
    return adjusted, clipped


def format_clipped_returns(clipped: pd.DataFrame, max_items: int = 20) -> str:
    """Format clipped return records for a compact console line."""
    if clipped.empty:
        return 'none'
    cells = []
    for _, row in clipped.head(max_items).iterrows():
        date = row['date']
        date_s = date.date().isoformat() if hasattr(date, 'date') else str(date)
        cells.append(
            f"{date_s} {row['ticker']} "
            f"{row['original_return']:+.1%}->{row['clipped_return']:+.1%}"
        )
    if len(clipped) > max_items:
        cells.append(f'... (+{len(clipped) - max_items} more)')
    return ', '.join(cells)
