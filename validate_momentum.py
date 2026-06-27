#!/usr/bin/env python3
"""
Walk-forward validation runner for the momentum model on real Yahoo Finance data.

Delegates optimization and walk-forward logic to quant.validation.
"""

from __future__ import annotations

import argparse

from quant.data import fetch_historical_prices
from quant.models.momentum import MomentumModel
from quant.validation import iter_param_grid, report_validation

GRID = {
    'lookback': [63, 126, 252],
    'skip': [0, 21],
    'target_vol': [0.10, 0.15, 0.20],
}


def auto_warmup(grid: dict) -> int:
    """Bars needed before the first valid momentum signal for the worst-case grid combo."""
    model = MomentumModel()
    return max(model.min_history_days(**c) for c in iter_param_grid(grid)) + 10


def min_signal_bars(grid: dict) -> int:
    """History needed before the widest lookback+skip combo in the grid is valid."""
    return max(p['lookback'] + p['skip'] for p in iter_param_grid(grid))


def adaptive_windows(
    n: int,
    grid: dict,
    preferred_train: int = 504,
    preferred_test: int = 63,
    min_test: int = 21,
) -> tuple[int, int, int] | None:
    """Fit train/test/warmup to the longest history available for this ticker.

    Uses preferred train/test when data allows. Otherwise maximizes train for
    at least one OOS fold while keeping enough warmup for the widest grid signal.
    """
    min_warmup = min_signal_bars(grid)
    warmup_cap = auto_warmup(grid)

    if n < min_warmup + min_test:
        return None

    test = min(preferred_test, n - min_warmup)
    test = max(min_test, test)
    if n < min_warmup + test:
        return None

    train = min(preferred_train, n - test)
    if train < min_warmup:
        train = n - test
    if train < min_warmup:
        return None

    warmup = min(warmup_cap, train)
    return train, test, warmup


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Walk-forward momentum validation on real price data.',
    )
    parser.add_argument('tickers', nargs='+', help='One or more ticker symbols')
    parser.add_argument('--years', type=int, default=10, help='Years of history')
    parser.add_argument(
        '--train', type=int, default=504,
        help='Train window in trading days (momentum wants train > warmup; try 756 if history allows)',
    )
    parser.add_argument('--test', type=int, default=63, help='Test window in trading days')
    parser.add_argument(
        '--warmup', type=int, default=None,
        help='Indicator warmup bars (default: auto from grid via auto_warmup)',
    )
    args = parser.parse_args()

    warmup = args.warmup if args.warmup is not None else auto_warmup(GRID)
    n_combos = sum(1 for _ in iter_param_grid(GRID))
    print(
        f'Walk-forward momentum: preferred train={args.train} test={args.test} '
        f'warmup_cap={warmup} grid_combos={n_combos} (windows adapt per ticker)'
    )

    strat = lambda p, **kw: MomentumModel().backtest(p, **kw)[0]['strat_net']

    for ticker in args.tickers:
        sym = ticker.upper()
        try:
            price = fetch_historical_prices(sym, args.years)
        except (ValueError, OSError) as exc:
            print(f'{sym}: fetch failed ({exc}), skipping.')
            continue

        n = len(price)
        windows = adaptive_windows(n, GRID, args.train, args.test)
        if windows is None:
            print(
                f'{sym}: only {n} bars — need at least '
                f'{min_signal_bars(GRID) + 21} for momentum signal + one test fold, skipping.'
            )
            continue
        train, test, tick_warmup = windows
        if args.warmup is not None:
            tick_warmup = args.warmup

        print(f'\n{sym}: {n} bars → train={train} test={test} warmup={tick_warmup}')
        report_validation(
            f'{sym} momentum',
            strat,
            price,
            GRID,
            train=train,
            test=test,
            warmup=tick_warmup,
        )


if __name__ == '__main__':
    main()
