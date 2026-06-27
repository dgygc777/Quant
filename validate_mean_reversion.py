#!/usr/bin/env python3
"""
Walk-forward validation for single-stock mean reversion on real Yahoo Finance data.

Compares naive full-history parameter search (overfit) vs honest out-of-sample
walk-forward Sharpe. Run from the project root:

    python3 validate_mean_reversion.py MU
    python3 validate_mean_reversion.py NVDA --years 5
"""

from __future__ import annotations

import argparse
import sys

from quant.data import fetch_historical_prices
from quant.models.mean_reversion import MeanReversionModel
from quant.validation import report_validation

DEFAULT_GRID = {
    'window': [10, 20, 30, 50],
    'entry_z': [-0.5, -1.0, -1.5, -2.0],
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Walk-forward mean-reversion validation on real price data.',
    )
    parser.add_argument('ticker', nargs='?', default='MU', help='Ticker symbol')
    parser.add_argument('--years', type=int, default=10, help='Years of history')
    parser.add_argument('--train', type=int, default=252, help='Train window (days)')
    parser.add_argument('--test', type=int, default=63, help='Test window (days)')
    parser.add_argument('--warmup', type=int, default=40, help='Indicator warmup bars')
    args = parser.parse_args()

    ticker = args.ticker.upper()
    price = fetch_historical_prices(ticker, args.years)
    if len(price) < args.train + args.test:
        print(
            f'Error: need at least {args.train + args.test} trading days; got {len(price)}.',
            file=sys.stderr,
        )
        sys.exit(1)

    strat = lambda p, **kw: MeanReversionModel().backtest(p, **kw)[0]['strat_net']
    report_validation(
        f'{ticker} mean-reversion (real data)',
        strat,
        price,
        DEFAULT_GRID,
        train=args.train,
        test=args.test,
        warmup=args.warmup,
    )


if __name__ == '__main__':
    main()
