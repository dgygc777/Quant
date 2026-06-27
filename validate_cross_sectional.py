#!/usr/bin/env python3
"""
Walk-forward validation for the cross-sectional momentum model on a real price panel.

How this connects to portfolio ranks / backtest
-----------------------------------------------
Cross-sectional model (cli.py portfolio ranks):
  • Ranks stocks in a universe TODAY using lookback, top_frac, rebalance.
  • Answers: "Who is long/short right now?"

validate_cross_sectional.py:
  • Uses the SAME ranking rules (backtest_xs / CrossSectionalModel).
  • Splits history into train → test windows.
  • On each train window, searches GRID for best lookback / top_frac / rebalance.
  • Applies those fixed params to the NEXT test window (out-of-sample).
  • Stitches test-window portfolio returns → one Sharpe (honest performance).
  • With --show-fold-ranks: prints which names were long/short at each test window.

Overfitting tax = naive full-history Sharpe minus OOS Sharpe. Positive means
tuning on all past data overstated how good the strategy looked.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from quant.data import fetch_panel
from quant.metrics import metrics
from quant.models.cross_sectional import CrossSectionalModel, backtest_xs
from quant.universes import DEFAULT_PRESET, get_universe
from quant.validation import iter_param_grid, optimize_full, walk_forward

GRID = {
    'lookback': [63, 126, 252],
    'top_frac': [0.25, 0.33],
    'rebalance': [5, 21],
}

# Largest lookback in GRID (252) + default momentum skip (21) − 3 bar buffer ≈ 270.
WARMUP = max(GRID['lookback']) + 18


def xs_strat(panel, **kw):
    """backtest_xs returns (DataFrame, n_rebalances); strategy returns are strat_net."""
    return backtest_xs(panel, mode='momentum', **kw)[0]['strat_net']


def _xs_model_params(grid_params: dict) -> dict:
    return {'mode': 'momentum', **grid_params}


def _format_leg(label: str, as_of: pd.Timestamp, weights: pd.Series,
                scores: pd.Series) -> None:
    print(f'{label} ({as_of.date()}):')
    longs = weights[weights > 0].sort_values(ascending=False)
    shorts = weights[weights < 0].sort_values()
    if longs.empty and shorts.empty:
        print('  (no active long/short book — insufficient scored names)')
        return
    if not longs.empty:
        print('  LONG:', end='')
        for tk, wt in longs.items():
            sc = scores.get(tk, float('nan'))
            sc_s = f'{sc:+.1%}' if not pd.isna(sc) else 'n/a'
            print(f'  {tk} wt {wt:+.0%} score {sc_s}', end='')
        print()
    if not shorts.empty:
        print('  SHORT:', end='')
        for tk, wt in shorts.items():
            sc = scores.get(tk, float('nan'))
            sc_s = f'{sc:+.1%}' if not pd.isna(sc) else 'n/a'
            print(f'  {tk} wt {wt:+.0%} score {sc_s}', end='')
        print()


def _book_at(model: CrossSectionalModel, panel: pd.DataFrame,
             params: dict) -> tuple[pd.Series, pd.Series, pd.Timestamp]:
    p = _xs_model_params(params)
    as_of = panel.index[-1]
    try:
        weights = model.current_weights(panel, **p)
        scores = model.current_ranks(panel, **p)
    except ValueError:
        weights = pd.Series(dtype=float)
        scores = pd.Series(dtype=float)
    return weights, scores, as_of


def print_oos_fold_ranks(panel: pd.DataFrame, param_grid: dict,
                         train: int, test: int, warmup: int,
                         select: str = 'sharpe') -> None:
    """Print long/short legs at test start/end for each walk-forward fold."""
    model = CrossSectionalModel()
    n = len(panel)
    pos = train
    fold_n = 0

    print('\n=== Walk-forward OOS rankings ===')
    print('Same cross-sectional rules as `portfolio ranks`, but parameters are')
    print('picked on the train window only; books below are on the test window.\n')

    while pos + test <= n:
        fold_n += 1
        tr_lo, tr_hi = pos - train, pos
        te_lo, te_hi = pos, pos + test

        best_params, best_is, best_score = None, None, -np.inf
        for params in iter_param_grid(param_grid):
            m = metrics(xs_strat(panel.iloc[tr_lo:tr_hi], **params))
            if m[select] > best_score:
                best_score, best_params, best_is = m[select], params, m

        w_start, s_start, d_start = _book_at(model, panel.iloc[:te_lo + 1], best_params)
        w_end, s_end, d_end = _book_at(model, panel.iloc[:te_hi], best_params)

        oos_ret = xs_strat(panel.iloc[max(0, te_lo - warmup):te_hi], **best_params)
        oos_m = metrics(oos_ret.reindex(panel.index[te_lo:te_hi]).dropna())

        print(f'--- Fold {fold_n} ---')
        print(f'Train: {panel.index[tr_lo].date()} → {panel.index[tr_hi - 1].date()}'
              f'  |  Test: {panel.index[te_lo].date()} → {panel.index[te_hi - 1].date()}')
        print(f'Params from train: {best_params}')
        print(f'Train Sharpe: {best_is["sharpe"]:.2f}  |  Test Sharpe: {oos_m["sharpe"]:.2f}')
        _format_leg('  Test START', d_start, w_start, s_start)
        _format_leg('  Test END  ', d_end, w_end, s_end)
        print()
        pos += test


def report_panel_validation(name, panel, param_grid, show_fold_ranks: bool = False,
                            **wf_kwargs) -> dict:
    """Panel walk-forward report — same three rows as quant.validation.report_validation."""
    _, full_m = optimize_full(xs_strat, panel, param_grid)
    wf = walk_forward(xs_strat, panel, param_grid, **wf_kwargs)
    oos = wf['oos_metrics']
    mean_is = np.mean([f['in_sample_sharpe'] for f in wf['folds']]) if wf['folds'] else 0.0

    print(f'\n=== {name} ===')
    print(f"{'':34}{'Sharpe':>9}{'AnnRet':>9}{'MaxDD':>9}")
    print(f"{'Naive full-history optimize':34}{full_m['sharpe']:>9.2f}"
          f"{full_m['ann_return']:>9.1%}{full_m['max_dd']:>9.1%}   <- the overfit trap")
    print(f"{'Walk-forward, in-sample (avg)':34}{mean_is:>9.2f}"
          f"{'':>9}{'':>9}   <- optimistic")
    print(f"{'Walk-forward, OUT-OF-SAMPLE':34}{oos['sharpe']:>9.2f}"
          f"{oos['ann_return']:>9.1%}{oos['max_dd']:>9.1%}   <- the honest number")
    gap = full_m['sharpe'] - oos['sharpe']
    print(f'\nOverfitting tax (naive - OOS Sharpe): {gap:.2f}')
    print(f'Folds: {len(wf["folds"])}')

    if show_fold_ranks:
        print_oos_fold_ranks(
            panel, param_grid,
            train=wf_kwargs['train'],
            test=wf_kwargs['test'],
            warmup=wf_kwargs.get('warmup', WARMUP),
        )

    return wf


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Walk-forward cross-sectional momentum validation on a universe panel.',
    )
    parser.add_argument(
        '--universe', default=DEFAULT_PRESET,
        help=f'Universe preset name (default: {DEFAULT_PRESET})',
    )
    parser.add_argument('--years', type=int, default=10, help='Years of history')
    parser.add_argument('--train', type=int, default=504, help='Train window (trading days)')
    parser.add_argument('--test', type=int, default=63, help='Test window (trading days)')
    parser.add_argument(
        '--warmup', type=int, default=None,
        help=f'Warmup bars before test window (default: {WARMUP})',
    )
    parser.add_argument(
        '--show-fold-ranks', action='store_true',
        help='Print long/short legs at test start/end for each walk-forward fold',
    )
    args = parser.parse_args()

    warmup = args.warmup if args.warmup is not None else WARMUP
    n_combos = sum(1 for _ in iter_param_grid(GRID))

    try:
        tickers = get_universe(args.universe)
    except ValueError as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)

    try:
        panel = fetch_panel(tickers, args.years)
    except (ValueError, OSError) as exc:
        print(f'Fetch failed: {exc}', file=sys.stderr)
        sys.exit(1)

    print(
        f'Walk-forward cross-sectional: universe={args.universe} '
        f'n={len(tickers)} bars={len(panel)} train={args.train} test={args.test} '
        f'warmup={warmup} grid_combos={n_combos}'
    )

    if len(panel) < args.train + args.test:
        print(
            f'Error: panel has {len(panel)} rows — need at least '
            f'{args.train + args.test} for one fold.',
            file=sys.stderr,
        )
        sys.exit(1)

    report_panel_validation(
        f'{args.universe} cross-sectional momentum',
        panel,
        GRID,
        show_fold_ranks=args.show_fold_ranks,
        train=args.train,
        test=args.test,
        warmup=warmup,
    )


if __name__ == '__main__':
    main()
