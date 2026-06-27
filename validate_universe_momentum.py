#!/usr/bin/env python3
"""
Rank every ticker in a universe by walk-forward OUT-OF-SAMPLE momentum Sharpe.

Each name is validated individually (time-series momentum on that stock alone).
Train/test/warmup adapt per ticker to use the longest history available (short
listings like SNDK are included with smaller windows, not skipped).

  python3 validate_universe_momentum.py --universe semis
  python3 validate_universe_momentum.py --universe semis --verbose
"""

from __future__ import annotations

import argparse
import sys

from quant.data import fetch_historical_prices
from quant.models.momentum import MomentumModel
from quant.universes import DEFAULT_PRESET, get_universe
from quant.validation import iter_param_grid, optimize_full, walk_forward

from validate_momentum import GRID, adaptive_windows, auto_warmup, min_signal_bars


def validate_ticker(
    sym: str,
    years: int,
    preferred_train: int,
    preferred_test: int,
    warmup_override: int | None,
) -> dict | None:
    """Return summary row or None if skipped."""
    strat = lambda p, **kw: MomentumModel().backtest(p, **kw)[0]['strat_net']
    try:
        price = fetch_historical_prices(sym, years)
    except (ValueError, OSError) as exc:
        return {'ticker': sym, 'skip': f'fetch failed ({exc})'}

    n = len(price)
    windows = adaptive_windows(n, GRID, preferred_train, preferred_test)
    if windows is None:
        need = min_signal_bars(GRID) + 21
        return {'ticker': sym, 'skip': f'only {n} bars (need ~{need} for signal + test)'}

    train, test, warmup = windows
    if warmup_override is not None:
        warmup = min(warmup_override, train)

    _, naive_m = optimize_full(strat, price, GRID)
    wf = walk_forward(strat, price, GRID, train=train, test=test, warmup=warmup)
    oos = wf['oos_metrics']
    if not wf['folds']:
        return {'ticker': sym, 'skip': 'no walk-forward folds'}

    return {
        'ticker': sym,
        'bars': n,
        'train': train,
        'test': test,
        'warmup': warmup,
        'folds': len(wf['folds']),
        'oos_sharpe': oos['sharpe'],
        'oos_ann_return': oos['ann_return'],
        'oos_max_dd': oos['max_dd'],
        'naive_sharpe': naive_m['sharpe'],
        'overfit_tax': naive_m['sharpe'] - oos['sharpe'],
        'price': price,
        'strat': strat,
        'wf_train': train,
        'wf_test': test,
        'wf_warmup': warmup,
    }


def print_ranked_table(rows: list[dict], universe: str) -> None:
    ok = [r for r in rows if 'skip' not in r]
    skipped = [r for r in rows if 'skip' in r]
    ok.sort(key=lambda r: r['oos_sharpe'], reverse=True)

    print(f'\n=== Universe momentum OOS Sharpe ranking: {universe} ===')
    print('Per-stock time-series momentum walk-forward (not cross-sectional ranks).')
    print('Train/test/warmup adapt to each ticker\'s available history.\n')
    print(f'{"Rank":<5}{"Ticker":<7}{"OOS Sh":>8}{"OOS Ret":>9}{"OOS DD":>9}'
          f'{"Fld":>4}{"Trn":>5}{"Tst":>4}{"Wup":>5}{"Bars":>6}')
    print('-' * 68)
    for i, r in enumerate(ok, 1):
        print(f'{i:<5}{r["ticker"]:<7}{r["oos_sharpe"]:>8.2f}'
              f'{r["oos_ann_return"]:>+8.1%}{r["oos_max_dd"]:>9.1%}'
              f'{r["folds"]:>4}{r["train"]:>5}{r["test"]:>4}{r["warmup"]:>5}'
              f'{r["bars"]:>6}')

    if skipped:
        print(f'\nSkipped ({len(skipped)}):')
        for r in skipped:
            print(f'  {r["ticker"]}: {r["skip"]}')

    if ok:
        print(f'\nBest OOS Sharpe:  {ok[0]["ticker"]} ({ok[0]["oos_sharpe"]:.2f})')
        print(f'Worst OOS Sharpe: {ok[-1]["ticker"]} ({ok[-1]["oos_sharpe"]:.2f})')
        print('Trn/Tst/Wup = train/test/warmup bars used for that name.')
        print('Few folds (e.g. Fld=1) → noisier OOS Sharpe; compare cautiously vs Fld=31.')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Rank universe tickers by walk-forward OOS momentum Sharpe.',
    )
    parser.add_argument(
        '--universe', default=DEFAULT_PRESET,
        help=f'Universe preset (default: {DEFAULT_PRESET})',
    )
    parser.add_argument('--years', type=int, default=10)
    parser.add_argument(
        '--train', type=int, default=504,
        help='Preferred train window (adapted down when history is shorter)',
    )
    parser.add_argument(
        '--test', type=int, default=63,
        help='Preferred test window (adapted down when history is shorter)',
    )
    parser.add_argument(
        '--warmup', type=int, default=None,
        help='Override warmup cap (default: auto, capped by each ticker train)',
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Print full report_validation block for each ticker',
    )
    args = parser.parse_args()

    try:
        tickers = get_universe(args.universe)
    except ValueError as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)

    warmup_cap = args.warmup if args.warmup is not None else auto_warmup(GRID)
    n_combos = sum(1 for _ in iter_param_grid(GRID))
    print(
        f'Universe momentum validation: {args.universe} ({len(tickers)} names) '
        f'preferred train={args.train} test={args.test} warmup_cap={warmup_cap} '
        f'grid_combos={n_combos} (per-ticker adaptive windows)'
    )

    rows: list[dict] = []
    for ticker in tickers:
        row = validate_ticker(
            ticker, args.years, args.train, args.test, args.warmup,
        )
        if row is None:
            continue
        if args.verbose and 'skip' not in row:
            from quant.validation import report_validation
            report_validation(
                f'{row["ticker"]} momentum', row['strat'], row['price'], GRID,
                train=row['wf_train'], test=row['wf_test'], warmup=row['wf_warmup'],
            )
        if 'skip' in row:
            rows.append(row)
        else:
            for k in ('price', 'strat', 'wf_train', 'wf_test', 'wf_warmup'):
                row.pop(k, None)
            rows.append(row)

    print_ranked_table(rows, args.universe)


if __name__ == '__main__':
    main()
