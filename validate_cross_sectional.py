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
from quant.models.cross_sectional import CrossSectionalModel, DEFAULT_XS_COST, backtest_xs
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
COST_SWEEP_BPS = [0, 5, 10, 20, 30, 50]

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


def cost_from_bps(cost_bps: float) -> float:
    """Convert basis points to decimal cost per unit turnover."""
    return float(cost_bps) / 10_000.0


def make_xs_strategy(
    book: str = 'long_only',
    weighting: str = 'equal',
    cost: float | pd.Series = DEFAULT_XS_COST,
):
    """backtest_xs returns (DataFrame, n_rebalances); strategy returns are strat_net."""
    market_neutral = book_market_neutral(book)

    def _strategy(panel, **kw):
        return backtest_xs(
            panel,
            mode='momentum',
            market_neutral=market_neutral,
            weighting=weighting,
            cost=cost,
            **kw,
        )[0]['strat_net']

    return _strategy


def xs_strat(panel, **kw):
    """Backward-compatible default: validate the long-only book."""
    return make_xs_strategy('long_only')(panel, **kw)


def make_weighted_long_only_strategy(
    weighting: str,
    cost: float | pd.Series = DEFAULT_XS_COST,
):
    """Strategy adapter for validating long-only weighting schemes."""
    return make_xs_strategy('long_only', weighting=weighting, cost=cost)


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


def buy_hold_equal_weight_benchmark(
    panel: pd.DataFrame,
    oos_index: pd.Index,
    cost: float = 0.0,
) -> pd.DataFrame:
    """Low-turnover equal-weight buy-and-hold benchmark on OOS dates."""
    rets = panel.pct_change(fill_method=None).reindex(oos_index).fillna(0.0)
    if rets.empty or rets.shape[1] == 0:
        return pd.DataFrame({
            'ret': pd.Series(dtype=float),
            'turnover': pd.Series(dtype=float),
            'cost': pd.Series(dtype=float),
        })

    weights = pd.Series(1.0 / rets.shape[1], index=rets.columns)
    rows = []
    first = True
    for dt, row in rets.iterrows():
        turnover = 1.0 if first else 0.0
        cost_paid = turnover * float(cost)
        gross = float((weights * row).sum())
        net = gross - cost_paid
        rows.append((dt, net, turnover, cost_paid))
        next_weights = weights * (1.0 + row)
        total = float(next_weights.sum())
        if total > 0.0:
            weights = next_weights / total
        first = False

    return pd.DataFrame.from_records(
        rows,
        columns=['date', 'ret', 'turnover', 'cost'],
    ).set_index('date')


def break_even_cost_bps(cost_bps: list[float], active_irs: list[float]) -> float | None:
    """Linearly interpolate the first cost where active IR crosses zero."""
    pairs = [
        (float(cost), float(ir))
        for cost, ir in zip(cost_bps, active_irs)
        if not pd.isna(ir) and np.isfinite(ir)
    ]
    if not pairs:
        return None
    if pairs[0][1] <= 0.0:
        return None
    for (c0, ir0), (c1, ir1) in zip(pairs, pairs[1:]):
        if ir0 == 0.0:
            return c0
        if ir0 > 0.0 and ir1 <= 0.0:
            if ir0 == ir1:
                return c1
            return c0 + (0.0 - ir0) * (c1 - c0) / (ir1 - ir0)
    return None


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


def report_panel_validation(
    name,
    panel,
    param_grid,
    show_fold_ranks: bool = False,
    book: str = 'long_only',
    cost: float | pd.Series = DEFAULT_XS_COST,
    **wf_kwargs,
) -> dict:
    """Panel walk-forward report — same three rows as quant.validation.report_validation."""
    strategy_fn = make_xs_strategy(book, cost=cost)
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


def compare_weighting_validation(
    panel: pd.DataFrame,
    param_grid: dict,
    cost: float | pd.Series = DEFAULT_XS_COST,
    **wf_kwargs,
) -> dict:
    """Walk-forward validate long-only sizing schemes and the benchmark."""
    rows: dict[str, dict] = {}
    first_oos_index: pd.Index | None = None
    for weighting in WEIGHTING_METHODS:
        try:
            wf = walk_forward(
                make_weighted_long_only_strategy(weighting, cost=cost),
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


def walk_forward_long_only_with_turnover(
    panel: pd.DataFrame,
    param_grid: dict,
    *,
    cost: float = DEFAULT_XS_COST,
    train: int = 252,
    test: int = 63,
    warmup: int = WARMUP,
    select: str = 'sharpe',
    fixed_folds: list[dict] | None = None,
) -> dict:
    """Canonical long-only walk-forward plus fold-matched OOS turnover."""
    if fixed_folds is None:
        strategy_fn = make_xs_strategy('long_only', cost=cost)
        base_wf = walk_forward(
            strategy_fn,
            panel,
            param_grid,
            train=train,
            test=test,
            warmup=warmup,
            select=select,
        )
        folds = base_wf['folds']
    else:
        folds = fixed_folds

    oos_chunks, turnover_chunks, cost_chunks, replayed_folds = [], [], [], []
    pos = train
    for fold in folds:
        te_lo, te_hi = pos, pos + test
        if te_hi > len(panel):
            break
        wlo = max(0, te_lo - warmup)
        df_full, _ = backtest_xs(
            panel.iloc[wlo:te_hi],
            mode='momentum',
            market_neutral=False,
            cost=cost,
            **fold['best_params'],
        )
        test_index = panel.index[te_lo:te_hi]
        oos = df_full['strat_net'].reindex(test_index).dropna()
        turnover = df_full['turnover'].reindex(oos.index).fillna(0.0)
        cost_paid = df_full['cost'].reindex(oos.index).fillna(0.0)
        oos_chunks.append(oos)
        turnover_chunks.append(turnover)
        cost_chunks.append(cost_paid)
        replayed = dict(fold)
        replayed['oos_sharpe'] = metrics(oos)['sharpe']
        replayed_folds.append(replayed)
        pos += test
    oos_returns = pd.concat(oos_chunks).sort_index() if oos_chunks else pd.Series(dtype=float)
    turnover = pd.concat(turnover_chunks).sort_index() if turnover_chunks else pd.Series(dtype=float)
    cost_paid = pd.concat(cost_chunks).sort_index() if cost_chunks else pd.Series(dtype=float)
    return {
        'oos_returns': oos_returns,
        'oos_metrics': metrics(oos_returns),
        'turnover': turnover,
        'cost_paid': cost_paid,
        'folds': replayed_folds,
    }


def active_ir_increase_warnings(rows: list[dict], tol: float = 1e-12) -> list[str]:
    """Return warnings for Active IR bumps; IR is not a monotonic cost-drag metric."""
    warnings = []
    prev_row = None
    for row in rows:
        ir = row.get('active_ir')
        if ir is None or pd.isna(ir) or not np.isfinite(ir):
            prev_row = row
            continue
        if prev_row is not None:
            prev_ir = prev_row.get('active_ir')
            if (
                prev_ir is not None
                and not pd.isna(prev_ir)
                and np.isfinite(prev_ir)
                and ir > prev_ir + tol
            ):
                warnings.append(
                    'Active IR increased as transaction cost rose: '
                    f"{prev_row['cost_bps']:.2f} bps -> {row['cost_bps']:.2f} bps "
                    f"({prev_ir:.12g} -> {ir:.12g})."
                )
        prev_row = row
    return warnings


def assert_cost_paid_non_decreasing(rows: list[dict], tol: float = 1e-12) -> None:
    """Costs paid must rise monotonically with the swept bps on the fixed book."""
    prev = None
    for row in rows:
        total_cost = row.get('strategy_cost_mean')
        if total_cost is None or pd.isna(total_cost) or not np.isfinite(total_cost):
            prev = row
            continue
        if prev is not None:
            prev_cost = prev.get('strategy_cost_mean')
            if (
                prev_cost is not None
                and not pd.isna(prev_cost)
                and np.isfinite(prev_cost)
                and total_cost + tol < prev_cost
            ):
                raise AssertionError(
                    'Strategy cost paid decreased as transaction cost rose: '
                    f"{prev['cost_bps']:.2f} bps -> {row['cost_bps']:.2f} bps "
                    f"({prev_cost:.12g} -> {total_cost:.12g})."
                )
        prev = row


def transaction_cost_sensitivity(
    panel: pd.DataFrame,
    param_grid: dict,
    cost_bps_levels: list[float] | None = None,
    selection_cost: float | None = None,
    **wf_kwargs,
) -> dict:
    """Sweep costs on one fixed walk-forward book and benchmark cost model."""
    levels = sorted({float(level) for level in (COST_SWEEP_BPS if cost_bps_levels is None else cost_bps_levels)})
    if not levels:
        return {'rows': [], 'break_even_bps': None, 'active_ir_warnings': []}
    selection_cost = cost_from_bps(levels[0]) if selection_cost is None else selection_cost
    selection_wf = walk_forward_long_only_with_turnover(
        panel,
        param_grid,
        cost=selection_cost,
        **wf_kwargs,
    )
    rows = []
    for bps in levels:
        cost = cost_from_bps(bps)
        if abs(cost - selection_cost) <= 1e-15:
            wf = selection_wf
        else:
            wf = walk_forward_long_only_with_turnover(
                panel,
                param_grid,
                cost=cost,
                fixed_folds=selection_wf['folds'],
                **wf_kwargs,
            )
        benchmark = buy_hold_equal_weight_benchmark(panel, wf['oos_returns'].index, cost=cost)
        active = active_oos_returns(wf['oos_returns'], benchmark['ret'])
        active_ir = information_ratio(active)
        verdict = validation_verdict(wf['oos_metrics']['sharpe'], active_ir, len(wf['folds']))
        strategy_events = int((wf['turnover'] > 1e-9).sum())
        strategy_turnover_mean = float(wf['turnover'].mean()) if len(wf['turnover']) else 0.0
        benchmark_turnover_mean = float(benchmark['turnover'].mean()) if len(benchmark) else 0.0
        strategy_cost_mean = float(wf['cost_paid'].mean()) if len(wf['cost_paid']) else 0.0
        benchmark_cost_mean = float(benchmark['cost'].mean()) if len(benchmark) else 0.0
        rows.append({
            'cost_bps': float(bps),
            'cost': cost,
            'sharpe': wf['oos_metrics']['sharpe'],
            'active_ir': active_ir,
            'active_mean_return': float(active.mean()) if len(active) else 0.0,
            'verdict': verdict,
            'strategy_turnover_mean': strategy_turnover_mean,
            'benchmark_turnover_mean': benchmark_turnover_mean,
            'strategy_cost_mean': strategy_cost_mean,
            'benchmark_cost_mean': benchmark_cost_mean,
            'active_cost_drag_mean': strategy_cost_mean - benchmark_cost_mean,
            'strategy_turnover_per_rebal': strategy_turnover_mean,
            'benchmark_turnover_per_rebal': benchmark_turnover_mean,
            'strategy_turnover_events': strategy_events,
            'wf': wf,
            'benchmark_returns': benchmark['ret'],
        })
    assert_cost_paid_non_decreasing(rows)
    ir_warnings = active_ir_increase_warnings(rows)

    breakeven = break_even_cost_bps(
        [row['cost_bps'] for row in rows],
        [row['active_ir'] for row in rows],
    )
    return {
        'rows': rows,
        'break_even_bps': breakeven,
        'active_ir_warnings': ir_warnings,
        'selection_cost_bps': float(selection_cost) * 10_000.0,
    }


def print_transaction_cost_sensitivity(
    summary: dict,
    label: str = 'custom',
    headline_cost_bps: float | None = None,
) -> None:
    """Print transaction-cost sensitivity table."""
    rows = summary.get('rows', [])
    print(f'\n=== Transaction-cost sensitivity (long-only book, {label}) ===')
    if rows:
        print(
            f"Strategy turnover(mean): {rows[0]['strategy_turnover_mean']:.2f}   "
            f"Benchmark turnover(mean): {rows[0]['benchmark_turnover_mean']:.2f}"
        )
        selection_cost_bps = summary.get('selection_cost_bps')
        if selection_cost_bps is not None:
            print(f'Sweep fold-selection schedule: fixed at {selection_cost_bps:.0f} bps.')
    print(f"{'Cost(bps RT)':<13}{'StratOOS Sharpe':>17}{'Active IR':>12}   Verdict")
    for row in rows:
        print(
            f"{row['cost_bps']:<13.0f}{row['sharpe']:>17.2f}"
            f"{row['active_ir']:>12.2f}   {row['verdict']}"
        )
    if summary.get('active_ir_warnings'):
        print(
            'Active IR shape note: cost paid is monotonic, but Active IR can rise '
            'when the active-return volatility denominator falls; inspect the '
            'cost rows as diagnostics, not as an optimizer.'
        )
    if not rows:
        print('Break-even cost: unavailable - no estimable rows.')
    elif rows[0]['active_ir'] <= 0.0:
        print('Break-even cost: already <=0 gross - no positive-cost break-even.')
    elif summary.get('break_even_bps') is None:
        print('Break-even cost: no crossing in swept range.')
    else:
        print(f"Break-even cost (Active IR crosses 0): ~{summary['break_even_bps']:.1f} bps")
    if rows:
        headline = (
            min(rows, key=lambda row: abs(row['cost_bps'] - headline_cost_bps))
            if headline_cost_bps is not None
            else rows[0]
        )
        print(
            'Cost takeaway: this book usually turns over far more than the buy-and-hold '
            'benchmark, so a low gross active IR is fragile; read the '
            f"{headline['cost_bps']:.0f} bps row as the realistic-cost verdict "
            f"({headline['verdict']})."
        )


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
