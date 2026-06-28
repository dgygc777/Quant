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
from quant.data_quality import (
    MIN_COVERAGE,
    filter_panel_by_coverage,
    format_coverage as _format_coverage,
)
from quant.metrics import metrics
from quant.combined_signal import backtest_xs_long_only
from quant.models.cross_sectional import CrossSectionalModel, backtest_xs
from quant.risk_model import WEIGHTING_METHODS
from quant.universes import DEFAULT_PRESET, get_universe
from quant.validation import iter_param_grid, optimize_full, walk_forward

BOOK_CHOICES = ('long_only', 'long_short')
MIN_VALIDATION_NAMES = 4
MIN_VALIDATION_FOLDS = 8
ACTIVE_IR_EDGE_MARGIN = 0.25
# Backward-compatible name for older callers/tests that imported this constant.
ACTIVE_SHARPE_EDGE_MARGIN = ACTIVE_IR_EDGE_MARGIN
VERDICT_EDGE = 'EDGE'
VERDICT_MATCHES = 'MATCHES BENCHMARK — captures sector beta, not alpha'
VERDICT_FAILS = 'FAILS'

GRID = {
    'lookback': [63, 126, 252],
    'top_frac': [0.25, 0.33],
    'rebalance': [5, 21],
}

# Largest lookback in GRID (252) + default momentum skip (21) − 3 bar buffer ≈ 270.
WARMUP = max(GRID['lookback']) + 18


def book_market_neutral(book: str) -> bool:
    if book not in BOOK_CHOICES:
        raise ValueError(f'unknown book: {book}')
    return book == 'long_short'


def book_description(book: str) -> str:
    if book == 'long_only':
        return 'LONG-ONLY top-k book (this is the gate for a long-only trader)'
    return 'LONG/SHORT dollar-neutral book (diagnostic spread, not the default long-only gate)'


def make_xs_strategy(book: str = 'long_only', weighting: str = 'equal'):
    """backtest_xs returns (DataFrame, n_rebalances); strategy returns are strat_net."""
    market_neutral = book_market_neutral(book)

    def _strategy(panel, **kw):
        return backtest_xs(
            panel,
            mode='momentum',
            market_neutral=market_neutral,
            weighting=weighting,
            **kw,
        )[0]['strat_net']

    return _strategy


def xs_strat(panel, **kw):
    """Backward-compatible default: validate the long-only book."""
    return make_xs_strategy('long_only')(panel, **kw)


def make_weighted_long_only_strategy(weighting: str):
    """Strategy adapter for validating long-only weighting schemes."""

    def _strategy(panel, **kw):
        return backtest_xs_long_only(
            panel,
            {
                'mode': 'momentum',
                **kw,
                'weighting': weighting,
            },
        )['strat_net']

    return _strategy


def _xs_model_params(grid_params: dict, book: str = 'long_only') -> dict:
    return {
        'mode': 'momentum',
        'market_neutral': book_market_neutral(book),
        **grid_params,
    }


def fold_count(n_rows: int, train: int, test: int) -> int:
    if n_rows < train + test:
        return 0
    return (n_rows - train) // test


def equal_weight_oos_returns(panel: pd.DataFrame, oos_index: pd.Index) -> pd.Series:
    """Equal-weight basket returns over the same OOS dates as walk_forward."""
    benchmark = panel.pct_change(fill_method=None).mean(axis=1)
    return benchmark.reindex(oos_index).dropna()


def active_oos_returns(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> pd.Series:
    """Align OOS strategy and benchmark returns, then subtract benchmark."""
    aligned = pd.concat([
        strategy_returns.rename('strategy'),
        benchmark_returns.rename('benchmark'),
    ], axis=1).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    return aligned['strategy'] - aligned['benchmark']


def information_ratio(active_returns: pd.Series) -> float:
    """Annualized information ratio of active-return OOS series."""
    active_returns = active_returns.dropna()
    if active_returns.empty:
        return 0.0
    std = float(active_returns.std())
    mean = float(active_returns.mean())
    if std <= 1e-15:
        if mean > 0.0:
            return float('inf')
        if mean < 0.0:
            return float('-inf')
        return 0.0
    return mean / std * np.sqrt(252.0)


def validation_verdict(
    strategy_oos_sharpe: float,
    active_information_ratio: float,
    folds: int,
    *,
    min_folds: int = MIN_VALIDATION_FOLDS,
    margin: float = ACTIVE_IR_EDGE_MARGIN,
) -> str:
    """Classify whether OOS active returns beat the same-universe benchmark."""
    if folds < min_folds or pd.isna(strategy_oos_sharpe) or strategy_oos_sharpe <= 0:
        return VERDICT_FAILS
    if pd.isna(active_information_ratio):
        return VERDICT_FAILS
    if active_information_ratio > margin:
        return VERDICT_EDGE
    if abs(active_information_ratio) <= margin:
        return VERDICT_MATCHES
    return VERDICT_FAILS


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
             params: dict, book: str = 'long_only') -> tuple[pd.Series, pd.Series, pd.Timestamp]:
    p = _xs_model_params(params, book)
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
                         select: str = 'sharpe',
                         book: str = 'long_only') -> None:
    """Print long/short legs at test start/end for each walk-forward fold."""
    model = CrossSectionalModel()
    strategy_fn = make_xs_strategy(book)
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
            m = metrics(strategy_fn(panel.iloc[tr_lo:tr_hi], **params))
            if m[select] > best_score:
                best_score, best_params, best_is = m[select], params, m

        w_start, s_start, d_start = _book_at(model, panel.iloc[:te_lo + 1], best_params, book)
        w_end, s_end, d_end = _book_at(model, panel.iloc[:te_hi], best_params, book)

        oos_ret = strategy_fn(panel.iloc[max(0, te_lo - warmup):te_hi], **best_params)
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
                            book: str = 'long_only', **wf_kwargs) -> dict:
    """Panel walk-forward report — same three rows as quant.validation.report_validation."""
    strategy_fn = make_xs_strategy(book)
    print(f'\nValidating: {book_description(book)}')
    print(
        f'Validation data context: names={panel.shape[1]} rows={len(panel)} '
        f'folds={fold_count(len(panel), wf_kwargs["train"], wf_kwargs["test"])}'
    )
    _, full_m = optimize_full(strategy_fn, panel, param_grid)
    wf = walk_forward(strategy_fn, panel, param_grid, **wf_kwargs)
    oos = wf['oos_metrics']
    benchmark_oos_returns = equal_weight_oos_returns(panel, wf['oos_returns'].index)
    benchmark_oos = metrics(benchmark_oos_returns)
    active_oos_sharpe = oos['sharpe'] - benchmark_oos['sharpe']
    active_returns = active_oos_returns(wf['oos_returns'], benchmark_oos_returns)
    active_ir = information_ratio(active_returns)
    verdict = validation_verdict(oos['sharpe'], active_ir, len(wf['folds']))
    wf['benchmark_oos_returns'] = benchmark_oos_returns
    wf['benchmark_oos_metrics'] = benchmark_oos
    wf['active_oos_returns'] = active_returns
    wf['active_oos_sharpe'] = active_oos_sharpe
    wf['information_ratio'] = active_ir
    wf['validation_verdict'] = verdict
    mean_is = np.mean([f['in_sample_sharpe'] for f in wf['folds']]) if wf['folds'] else 0.0

    print(f'\n=== {name} ===')
    print(f"{'':34}{'Sharpe':>9}{'AnnRet':>9}{'MaxDD':>9}")
    print(f"{'Naive full-history optimize':34}{full_m['sharpe']:>9.2f}"
          f"{full_m['ann_return']:>9.1%}{full_m['max_dd']:>9.1%}   <- the overfit trap")
    print(f"{'Walk-forward, in-sample (avg)':34}{mean_is:>9.2f}"
          f"{'':>9}{'':>9}   <- optimistic")
    print(f"{'Walk-forward, OUT-OF-SAMPLE':34}{oos['sharpe']:>9.2f}"
          f"{oos['ann_return']:>9.1%}{oos['max_dd']:>9.1%}   <- the honest number")
    print(f"{'Equal-weight benchmark OOS':34}{benchmark_oos['sharpe']:>9.2f}"
          f"{benchmark_oos['ann_return']:>9.1%}{benchmark_oos['max_dd']:>9.1%}   <- same OOS folds")
    gap = full_m['sharpe'] - oos['sharpe']
    print(f'\nOverfitting tax (naive - OOS Sharpe): {gap:.2f}')
    print(f'Active OOS Sharpe (strategy - benchmark): {active_oos_sharpe:+.2f}')
    print(f'Information ratio (active-return OOS): {active_ir:+.2f}')
    print(f'Validation verdict: {verdict} (IR edge margin {ACTIVE_IR_EDGE_MARGIN:.2f})')
    print(f'Folds: {len(wf["folds"])}')

    if show_fold_ranks:
        print_oos_fold_ranks(
            panel, param_grid,
            train=wf_kwargs['train'],
            test=wf_kwargs['test'],
            warmup=wf_kwargs.get('warmup', WARMUP),
            book=book,
        )

    return wf


def compare_weighting_validation(panel: pd.DataFrame, param_grid: dict, **wf_kwargs) -> dict:
    """Walk-forward validate long-only sizing schemes and the benchmark."""
    rows: dict[str, dict] = {}
    first_oos_index: pd.Index | None = None
    for weighting in WEIGHTING_METHODS:
        try:
            wf = walk_forward(
                make_weighted_long_only_strategy(weighting),
                panel,
                param_grid,
                **wf_kwargs,
            )
            oos = wf['oos_metrics']
            if first_oos_index is None:
                first_oos_index = wf['oos_returns'].index
            rows[weighting] = {
                'sharpe': oos['sharpe'],
                'ann_return': oos['ann_return'],
                'ann_vol': oos['ann_vol'],
                'max_dd': oos['max_dd'],
                'wf': wf,
                'error': None,
            }
        except (AssertionError, ValueError, TypeError, np.linalg.LinAlgError, FloatingPointError) as exc:
            rows[weighting] = {
                'sharpe': float('nan'),
                'ann_return': float('nan'),
                'ann_vol': float('nan'),
                'max_dd': float('nan'),
                'wf': None,
                'error': str(exc),
            }

    if first_oos_index is None:
        first_oos_index = pd.Index([])
    benchmark_returns = equal_weight_oos_returns(panel, first_oos_index)
    benchmark_metrics = metrics(benchmark_returns)
    rows['benchmark'] = {
        'sharpe': benchmark_metrics['sharpe'],
        'ann_return': benchmark_metrics['ann_return'],
        'ann_vol': benchmark_metrics['ann_vol'],
        'max_dd': benchmark_metrics['max_dd'],
        'wf': None,
        'error': None,
    }

    candidates = {
        name: row for name, row in rows.items()
        if name in WEIGHTING_METHODS and row['error'] is None and not pd.isna(row['sharpe'])
    }
    best_weighting = max(candidates, key=lambda name: candidates[name]['sharpe']) if candidates else None
    equal_sharpe = rows.get('equal', {}).get('sharpe')
    risk_parity_sharpe = rows.get('risk_parity', {}).get('sharpe')
    risk_parity_beats_equal = (
        equal_sharpe is not None
        and risk_parity_sharpe is not None
        and not pd.isna(equal_sharpe)
        and not pd.isna(risk_parity_sharpe)
        and risk_parity_sharpe > equal_sharpe
    )
    return {
        'rows': rows,
        'best_weighting': best_weighting,
        'risk_parity_beats_equal': risk_parity_beats_equal,
    }


def print_weighting_validation_report(summary: dict) -> None:
    """Print sizing-scheme walk-forward metrics."""
    rows = summary.get('rows', {})
    print('\n=== Sizing-scheme OOS validation (long-only book) ===')
    print(f"{'Weighting':<16}{'OOS Sharpe':>11}{'AnnRet':>9}{'Vol':>8}{'MaxDD':>9}")
    for name in list(WEIGHTING_METHODS) + ['benchmark']:
        row = rows.get(name)
        if not row:
            continue
        if row.get('error'):
            print(f'{name:<16} skipped ({row["error"]})')
            continue
        print(
            f"{name:<16}{row['sharpe']:>11.2f}"
            f"{row['ann_return']:>9.1%}{row['ann_vol']:>8.1%}{row['max_dd']:>9.1%}"
        )

    best = summary.get('best_weighting')
    if best:
        best_sharpe = rows[best]['sharpe']
        rp_note = (
            'risk_parity beats equal'
            if summary.get('risk_parity_beats_equal')
            else 'risk_parity does not beat equal'
        )
        print(f'Takeaway: best OOS Sharpe is {best} ({best_sharpe:.2f}); {rp_note}.')
    else:
        print('Takeaway: no sizing scheme produced an estimable OOS Sharpe.')


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
        '--book',
        choices=BOOK_CHOICES,
        default='long_only',
        help='Book to validate: long_only or long_short (default: long_only)',
    )
    parser.add_argument(
        '--min-coverage',
        type=float,
        default=MIN_COVERAGE,
        help='Minimum non-NaN ticker coverage before validation (default: 0.60)',
    )
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

    panel, coverage, dropped = filter_panel_by_coverage(panel, args.min_coverage)
    if len(dropped):
        print(f'Excluded (coverage < {args.min_coverage:.0%}): {_format_coverage(dropped)}')
    else:
        print(f'Excluded (coverage < {args.min_coverage:.0%}): none')
    print(f'Survived coverage filter: {panel.shape[1]}/{len(tickers)} names')

    print(
        f'Walk-forward cross-sectional: universe={args.universe} '
        f'n={panel.shape[1]} bars={len(panel)} train={args.train} test={args.test} '
        f'warmup={warmup} grid_combos={n_combos}'
    )
    print(
        f'Data context after filtering: surviving_names={panel.shape[1]} '
        f'usable_rows={len(panel)} folds={fold_count(len(panel), args.train, args.test)}'
    )

    if panel.shape[1] < MIN_VALIDATION_NAMES:
        print(
            f'Warning: universe is too thin to validate after coverage filter '
            f'({panel.shape[1]} names; need at least {MIN_VALIDATION_NAMES}).'
        )
        return

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
        book=args.book,
        train=args.train,
        test=args.test,
        warmup=warmup,
    )


if __name__ == '__main__':
    main()
