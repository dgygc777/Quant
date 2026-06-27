"""Shared parameter validation for CLI and library callers."""

from __future__ import annotations


def validate_top_frac(top_frac: float) -> float:
    if not (0 < top_frac <= 0.5):
        raise ValueError(f'top_frac must be in (0, 0.5]; got {top_frac}')
    return top_frac


def validate_xs_params(
    *,
    top_frac: float,
    rebalance: int,
    lookback: int | None = None,
    skip: int | None = None,
    short_window: int | None = None,
    years: int | None = None,
) -> None:
    validate_top_frac(top_frac)
    if rebalance < 1:
        raise ValueError(f'rebalance must be >= 1; got {rebalance}')
    if lookback is not None and lookback <= 0:
        raise ValueError(f'lookback must be > 0; got {lookback}')
    if skip is not None and skip < 0:
        raise ValueError(f'skip must be >= 0; got {skip}')
    if short_window is not None and short_window <= 0:
        raise ValueError(f'short_window must be > 0; got {short_window}')
    if years is not None and years <= 0:
        raise ValueError(f'years must be > 0; got {years}')
