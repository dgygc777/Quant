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
import math
import sys
from statistics import NormalDist

import numpy as np
import pandas as pd

from quant.data import fetch_panel
from quant.data_quality import (
    MIN_COVERAGE,
    filter_panel_by_coverage,
    format_coverage as _format_coverage,
)
from quant.metrics import metrics
from quant.models.cross_sectional import (
    CrossSectionalModel,
    DEFAULT_BETA_WINDOW,
    DEFAULT_SCORE_MODE,
    DEFAULT_XS_COST,
    SCORE_MODES,
    backtest_xs,
    build_weights,
    compute_scores,
    portfolio_returns,
)
from quant.combined_signal import (
    CombinedParams,
    combined_long_only_weights,
    precompute_single_stock_signals,
)
from quant.risk_model import WEIGHTING_METHODS
from quant.universes import DEFAULT_PRESET, get_universe
from quant.validation import (
    iter_param_grid,
    optimize_full,
    selection_objective,
    walk_forward,
)

BOOK_CHOICES = ('long_only', 'long_short')
MIN_VALIDATION_NAMES = 4
MIN_VALIDATION_FOLDS = 8
ACTIVE_IR_EDGE_MARGIN = 0.25
DEFAULT_SELECTION_TRIALS = 20
IR_CI_BLOCK = 21
IR_CI_BOOT = 2000
IR_CI_ALPHA = 0.05
IR_CI_SEED = 0
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
    score_mode: str = DEFAULT_SCORE_MODE,
    beta_window: int = DEFAULT_BETA_WINDOW,
    hysteresis: dict | None = None,
    group_neutral: dict | None = None,
):
    """backtest_xs returns (DataFrame, n_rebalances); strategy returns are strat_net.

    ``score_mode``/``beta_window`` select the cross-sectional scoring rule (raw
    vs benchmark-relative vs residual momentum). ``hysteresis`` (if given) is a
    dict of build_weights turnover-control kwargs. ``group_neutral`` (if given)
    is a dict of peer-group construction kwargs (group_neutral/group_map/
    group_top_frac). All are fixed model choices, not tuned grid params; per-call
    grid kwargs still win if explicitly passed.
    """
    market_neutral = book_market_neutral(book)
    hysteresis = hysteresis or {}
    group_neutral = group_neutral or {}

    def _strategy(panel, **kw):
        kw.setdefault('score_mode', score_mode)
        kw.setdefault('beta_window', beta_window)
        for key, value in hysteresis.items():
            kw.setdefault(key, value)
        for key, value in group_neutral.items():
            kw.setdefault(key, value)
        return backtest_xs(
            panel,
            mode='momentum',
            market_neutral=market_neutral,
            weighting=weighting,
            cost=cost,
            **kw,
        )[0]['strat_net']

    return _strategy


def group_neutral_kwargs(
    use_group_neutral: bool = False,
    group_map: dict | None = None,
    group_top_frac: float | None = None,
) -> dict:
    """Bundle build_weights peer-group kwargs (empty when disabled)."""
    if not use_group_neutral:
        return {}
    if not group_map:
        raise ValueError('group-neutral construction requires a non-empty group_map.')
    out = {'group_neutral': True, 'group_map': group_map}
    if group_top_frac is not None:
        out['group_top_frac'] = float(group_top_frac)
    return out


def hysteresis_kwargs(
    use_hysteresis: bool = False,
    entry_rank_pct: float = 0.80,
    exit_rank_pct: float = 0.60,
    max_new_names_per_rebalance: int | None = None,
) -> dict:
    """Bundle build_weights turnover-control kwargs (empty when disabled)."""
    if not use_hysteresis:
        return {}
    return {
        'use_hysteresis': True,
        'entry_rank_pct': float(entry_rank_pct),
        'exit_rank_pct': float(exit_rank_pct),
        'max_new_names_per_rebalance': max_new_names_per_rebalance,
    }


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


def information_ratio_ci(
    active_returns: pd.Series,
    block: int = IR_CI_BLOCK,
    n_boot: int = IR_CI_BOOT,
    alpha: float = IR_CI_ALPHA,
    seed: int = IR_CI_SEED,
) -> tuple[float, float, float, float]:
    """Deterministic block-bootstrap confidence interval for active-return IR.

    Blocks preserve roughly monthly autocorrelation. If the OOS active-return
    series is too short for a meaningful block bootstrap, the point estimate is
    still returned and the interval fields are NaN; callers should treat that as
    "CI unavailable", not as an error.
    """
    active = pd.Series(active_returns).dropna()
    point_ir = information_ratio(active)
    block = int(block)
    n_boot = int(n_boot)
    if block <= 0:
        raise ValueError('block must be positive')
    if n_boot <= 0:
        raise ValueError('n_boot must be positive')
    if len(active) < 2 * block:
        return point_ir, float('nan'), float('nan'), float('nan')

    values = active.to_numpy(dtype=float)
    n = len(values)
    max_start = n - block
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n / block))
    boot_irs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        sample = np.concatenate([values[start:start + block] for start in starts])[:n]
        boot_irs[i] = information_ratio(pd.Series(sample))

    finite = boot_irs[np.isfinite(boot_irs)]
    if finite.size == 0:
        return point_ir, float('nan'), float('nan'), float('nan')
    lower = float(np.percentile(finite, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(finite, 100.0 * (1.0 - alpha / 2.0)))
    se = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    return point_ir, lower, upper, se


def expected_max_ir_under_null(n_trials: int, se: float) -> float:
    """Expected best IR from trying ``n_trials`` pure-noise strategies."""
    if se is None or pd.isna(se):
        return float('nan')
    se = float(se)
    if se < 0:
        raise ValueError('se must be non-negative')
    if not np.isfinite(se):
        return float('inf')
    if se == 0.0:
        return 0.0
    n_trials = int(n_trials)
    if n_trials <= 1:
        return 0.0

    gamma = 0.5772
    normal = NormalDist()
    p1 = min(max(1.0 - 1.0 / n_trials, 1e-12), 1.0 - 1e-12)
    p2 = min(max(1.0 - 1.0 / (n_trials * math.e), 1e-12), 1.0 - 1e-12)
    return se * ((1.0 - gamma) * normal.inv_cdf(p1) + gamma * normal.inv_cdf(p2))


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
    ci_lower: float | None = None,
    ci_upper: float | None = None,
    selection_threshold: float | None = None,
    min_folds: int = MIN_VALIDATION_FOLDS,
    margin: float = ACTIVE_IR_EDGE_MARGIN,
) -> str:
    """Classify whether OOS active returns beat the benchmark with evidence."""
    if folds < min_folds or pd.isna(strategy_oos_sharpe) or strategy_oos_sharpe <= 0:
        return VERDICT_FAILS
    if pd.isna(active_information_ratio):
        return VERDICT_FAILS
    ci_upper_ok = ci_upper is not None and not pd.isna(ci_upper) and np.isfinite(ci_upper)
    if ci_upper_ok and float(ci_upper) < 0.0:
        return VERDICT_FAILS
    ci_lower_ok = ci_lower is not None and not pd.isna(ci_lower) and np.isfinite(ci_lower)
    threshold_ok = (
        selection_threshold is not None
        and not pd.isna(selection_threshold)
        and np.isfinite(selection_threshold)
    )
    if (
        ci_lower_ok
        and threshold_ok
        and float(ci_lower) > margin
        and active_information_ratio > float(selection_threshold)
    ):
        return VERDICT_EDGE
    return VERDICT_MATCHES


def validation_verdict_reason(
    strategy_oos_sharpe: float,
    active_information_ratio: float,
    folds: int,
    *,
    ci_lower: float | None,
    ci_upper: float | None,
    selection_threshold: float | None,
    min_folds: int = MIN_VALIDATION_FOLDS,
    margin: float = ACTIVE_IR_EDGE_MARGIN,
) -> str:
    """Human-readable reason matching ``validation_verdict``."""
    if folds < min_folds:
        return f'only {folds} folds; need at least {min_folds}'
    if pd.isna(strategy_oos_sharpe) or strategy_oos_sharpe <= 0:
        return 'strategy OOS Sharpe is not positive'
    if pd.isna(active_information_ratio):
        return 'active-return IR is unavailable'
    if (
        ci_upper is not None
        and not pd.isna(ci_upper)
        and np.isfinite(ci_upper)
        and float(ci_upper) < 0.0
    ):
        return 'CI is confidently below zero active return'
    if ci_lower is None or pd.isna(ci_lower) or not np.isfinite(ci_lower):
        return 'CI unavailable; no EDGE from the point estimate alone'
    if float(ci_lower) <= margin:
        return 'CI straddles the edge margin; indistinguishable from beta'
    if (
        selection_threshold is None
        or pd.isna(selection_threshold)
        or not np.isfinite(selection_threshold)
    ):
        return 'selection-adjusted bar unavailable; no EDGE'
    if float(active_information_ratio) <= float(selection_threshold):
        return 'point IR does not clear the selection-adjusted null bar'
    return 'CI clears the edge margin and point IR clears the selection-adjusted null bar'


def _format_signed(value: float | None) -> str:
    if value is None or pd.isna(value):
        return 'n/a'
    return f'{value:+.2f}'


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
    n_trials: int = DEFAULT_SELECTION_TRIALS,
    ci_block: int = IR_CI_BLOCK,
    ci_n_boot: int = IR_CI_BOOT,
    ci_alpha: float = IR_CI_ALPHA,
    ci_seed: int = IR_CI_SEED,
    select: str = 'sharpe',
    score_mode: str = DEFAULT_SCORE_MODE,
    beta_window: int = DEFAULT_BETA_WINDOW,
    hysteresis: dict | None = None,
    turnover_penalty: float = 0.0,
    group_neutral: dict | None = None,
    **wf_kwargs,
) -> dict:
    """Panel walk-forward report — same three rows as quant.validation.report_validation."""
    hysteresis = hysteresis or {}
    group_neutral = group_neutral or {}
    strategy_fn = make_xs_strategy(
        book, cost=cost, score_mode=score_mode, beta_window=beta_window,
        hysteresis=hysteresis, group_neutral=group_neutral,
    )
    print(f'\nValidating: {book_description(book)}')
    print(
        f'Validation data context: names={panel.shape[1]} rows={len(panel)} '
        f'folds={fold_count(len(panel), wf_kwargs["train"], wf_kwargs["test"])}'
    )
    if score_mode != DEFAULT_SCORE_MODE:
        print(f'Score mode: {score_mode}' + (f' (beta_window={beta_window})'
                                             if score_mode == 'residual_momentum' else ''))
    if select != 'sharpe':
        print(f'Parameter selection objective: {select} (benchmark-relative)')
    if hysteresis:
        print(
            f"Rank hysteresis ON: entry>={hysteresis['entry_rank_pct']:.0%} "
            f"exit<{hysteresis['exit_rank_pct']:.0%}"
            + (f", max_new/rebal={hysteresis['max_new_names_per_rebalance']}"
               if hysteresis.get('max_new_names_per_rebalance') is not None else '')
        )
    if turnover_penalty:
        print(f'Turnover-penalized selection: objective = active_IR - {turnover_penalty:g} * avg_turnover')
    if group_neutral.get('group_neutral'):
        from quant.universes import groups_to_members
        group_map = group_neutral.get('group_map', {})
        members = groups_to_members({
            t: g for t, g in group_map.items() if t in panel.columns
        })
        gtf = group_neutral.get('group_top_frac')
        gtf_txt = f', group_top_frac={gtf:.0%}' if gtf is not None else ''
        print(f'Peer-group-neutral construction ON ({len(members)} groups{gtf_txt}):')
        for grp in sorted(members):
            print(f"  {grp:18} {', '.join(members[grp])}")
        try:
            first_combo = next(iter(iter_param_grid(param_grid)))
            current_book = CrossSectionalModel().current_weights(
                panel, mode='momentum', market_neutral=False,
                score_mode=score_mode, beta_window=beta_window,
                **group_neutral, **first_combo,
            )
            held = groups_to_members({
                t: group_map.get(t, 'broad_semis') for t in current_book.index
            })
            print(f'Current group-neutral book ({panel.index[-1].date()}):')
            for grp in sorted(held):
                names = ', '.join(
                    f'{t} {current_book[t]:.0%}' for t in held[grp]
                )
                print(f"  {grp:18} {names}")
        except (ValueError, StopIteration):
            print('  (current book unavailable — too few scored names)')
    objective_fn = None
    if turnover_penalty and turnover_penalty > 0.0:
        objective_fn = _turnover_penalized_objective(
            cost, score_mode, beta_window, hysteresis, turnover_penalty,
        )
    _, full_m = optimize_full(strategy_fn, panel, param_grid, select=select)
    wf = walk_forward(strategy_fn, panel, param_grid, select=select,
                      objective_fn=objective_fn, **wf_kwargs)
    oos = wf['oos_metrics']
    benchmark_oos_returns = equal_weight_oos_returns(panel, wf['oos_returns'].index)
    benchmark_oos = metrics(benchmark_oos_returns)
    active_oos_sharpe = oos['sharpe'] - benchmark_oos['sharpe']
    active_returns = active_oos_returns(wf['oos_returns'], benchmark_oos_returns)
    active_ir, ci_lower, ci_upper, ir_se = information_ratio_ci(
        active_returns,
        block=ci_block,
        n_boot=ci_n_boot,
        alpha=ci_alpha,
        seed=ci_seed,
    )
    selection_bar = expected_max_ir_under_null(n_trials, ir_se)
    verdict = validation_verdict(
        oos['sharpe'],
        active_ir,
        len(wf['folds']),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        selection_threshold=selection_bar,
    )
    verdict_reason = validation_verdict_reason(
        oos['sharpe'],
        active_ir,
        len(wf['folds']),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        selection_threshold=selection_bar,
    )
    aligned_oos = pd.concat([
        wf['oos_returns'].rename('strategy'),
        benchmark_oos_returns.rename('benchmark'),
    ], axis=1).dropna()
    benchmark_corr = (
        float(aligned_oos['strategy'].corr(aligned_oos['benchmark']))
        if len(aligned_oos) >= 2
        else float('nan')
    )
    wf['benchmark_oos_returns'] = benchmark_oos_returns
    wf['benchmark_oos_metrics'] = benchmark_oos
    wf['active_oos_returns'] = active_returns
    wf['active_oos_sharpe'] = active_oos_sharpe
    wf['information_ratio'] = active_ir
    wf['information_ratio_ci_lower'] = ci_lower
    wf['information_ratio_ci_upper'] = ci_upper
    wf['information_ratio_se'] = ir_se
    wf['information_ratio_ci_available'] = (
        not pd.isna(ci_lower)
        and not pd.isna(ci_upper)
        and np.isfinite(ci_lower)
        and np.isfinite(ci_upper)
    )
    wf['selection_expected_max_ir'] = selection_bar
    wf['selection_trials'] = int(n_trials)
    wf['benchmark_correlation'] = benchmark_corr
    wf['validation_verdict'] = verdict
    wf['validation_verdict_reason'] = verdict_reason
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
    confidence = int(round((1.0 - ci_alpha) * 100))
    if wf['information_ratio_ci_available']:
        ci_text = f'{_format_signed(ci_lower)}, {_format_signed(ci_upper)}'
    else:
        ci_text = 'unavailable - too few OOS active-return points'
    print(
        f'Information ratio (active-return OOS): {_format_signed(active_ir)}  '
        f'[{confidence}% CI: {ci_text}]  (block-bootstrap, seed={ci_seed})'
    )
    print(
        f'Selection-adjusted bar (N={int(n_trials)} trials): '
        f'expected max IR under null ≈ {_format_signed(selection_bar)}'
    )
    corr_note = ''
    if not pd.isna(benchmark_corr) and benchmark_corr > 0.90:
        corr_note = '   <- active return is a thin residual; IR is low-power'
    print(f'Benchmark correlation: {_format_signed(benchmark_corr)}{corr_note}')
    print(f'Validation verdict: {verdict} - {verdict_reason}')
    print(f'Folds: {len(wf["folds"])}')

    if book == 'long_only' and (hysteresis or turnover_penalty):
        to_wf = walk_forward_long_only_with_turnover(
            panel,
            param_grid,
            cost=cost,
            train=wf_kwargs['train'],
            test=wf_kwargs['test'],
            warmup=wf_kwargs.get('warmup', WARMUP),
            select=select,
            score_mode=score_mode,
            beta_window=beta_window,
            hysteresis=hysteresis,
            turnover_penalty=turnover_penalty,
        )
        mean_turnover = float(to_wf['turnover'].mean()) if len(to_wf['turnover']) else 0.0
        mean_cost = float(to_wf['cost_paid'].mean()) if len(to_wf['cost_paid']) else 0.0
        wf['oos_mean_turnover'] = mean_turnover
        wf['oos_mean_cost_drag'] = mean_cost
        print(
            f'OOS mean per-day turnover: {mean_turnover:.4f}   '
            f'mean daily cost drag: {mean_cost:.5f}'
        )

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


def _turnover_penalized_objective(cost, score_mode, beta_window, hysteresis, turnover_penalty):
    """In-sample objective = active IR - turnover_penalty * mean per-day turnover.

    Uses only the train slice passed in (no look-ahead): the benchmark and the
    turnover both come from that slice's own backtest.
    """
    hysteresis = hysteresis or {}

    def _obj(returns, price, params):
        kw = {'score_mode': score_mode, 'beta_window': beta_window, **hysteresis, **params}
        df, _ = backtest_xs(
            price, mode='momentum', market_neutral=False, cost=cost, **kw,
        )
        bench = price.pct_change(fill_method=None).mean(axis=1)
        active = active_oos_returns(df['strat_net'], bench)
        ir = information_ratio(active)
        avg_turnover = float(df['turnover'].mean()) if len(df) else 0.0
        if not np.isfinite(ir):
            return ir
        return ir - float(turnover_penalty) * avg_turnover

    return _obj


def walk_forward_long_only_with_turnover(
    panel: pd.DataFrame,
    param_grid: dict,
    *,
    cost: float = DEFAULT_XS_COST,
    train: int = 252,
    test: int = 63,
    warmup: int = WARMUP,
    select: str = 'sharpe',
    score_mode: str = DEFAULT_SCORE_MODE,
    beta_window: int = DEFAULT_BETA_WINDOW,
    hysteresis: dict | None = None,
    turnover_penalty: float = 0.0,
    fixed_folds: list[dict] | None = None,
) -> dict:
    """Canonical long-only walk-forward plus fold-matched OOS turnover."""
    hysteresis = hysteresis or {}
    if fixed_folds is None:
        strategy_fn = make_xs_strategy(
            'long_only', cost=cost, score_mode=score_mode, beta_window=beta_window,
            hysteresis=hysteresis,
        )
        objective_fn = None
        if turnover_penalty and turnover_penalty > 0.0:
            objective_fn = _turnover_penalized_objective(
                cost, score_mode, beta_window, hysteresis, turnover_penalty,
            )
        base_wf = walk_forward(
            strategy_fn,
            panel,
            param_grid,
            train=train,
            test=test,
            warmup=warmup,
            select=select,
            objective_fn=objective_fn,
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
        replay_kw = dict(fold['best_params'])
        replay_kw.setdefault('score_mode', score_mode)
        replay_kw.setdefault('beta_window', beta_window)
        for key, value in hysteresis.items():
            replay_kw.setdefault(key, value)
        df_full, _ = backtest_xs(
            panel.iloc[wlo:te_hi],
            mode='momentum',
            market_neutral=False,
            cost=cost,
            **replay_kw,
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


COMBINED_GRID = {
    'z_overextended': [1.0, 1.5, 2.0],
    'z_oversold': [-0.5, -1.0, -1.5],
    'require_momentum_buy': [True, False],
    'score_mode': ['raw_momentum', 'relative_momentum', 'residual_momentum'],
}


def _xs_single_combo_grid(xs_params: dict) -> dict:
    """One-combo grid (lists of length 1) of the fixed XS preset params."""
    keys = ('lookback', 'skip', 'top_frac', 'rebalance')
    return {k: [xs_params[k]] for k in keys if k in xs_params}


def _precompute_combined_panels(
    panel: pd.DataFrame, xs_params: dict, score_modes: list[str], beta_window: int,
):
    """Compute single-stock signal panels + per-score-mode XS leg weights once.

    Every panel here is point-in-time (rolling/shifted, backward-looking only),
    so slicing them to any window in the fold loop introduces no look-ahead.
    Reusing them across the grid is what makes combined walk-forward tractable.
    """
    lookback = xs_params.get('lookback', 126)
    skip = xs_params.get('skip', 21)
    top_frac = xs_params.get('top_frac', 0.25)
    rebalance = xs_params.get('rebalance', 5)
    precomputed = precompute_single_stock_signals(
        panel, mom_params={'lookback': lookback, 'skip': skip},
    )
    xs_w_by_mode = {}
    for sm in score_modes:
        scores = compute_scores(
            panel, mode='momentum', lookback=lookback, skip=skip,
            score_mode=sm, beta_window=beta_window,
        )
        xs_w_by_mode[sm] = build_weights(
            panel, scores, top_frac=top_frac, rebalance=rebalance, market_neutral=True,
        )
    return precomputed, xs_w_by_mode


def _combined_combo_returns(
    panel: pd.DataFrame, combo: dict, xs_w_by_mode: dict, precomputed, rebalance: int,
    cost: float | pd.Series,
) -> pd.Series:
    """Net daily returns of the combined long-only book for one grid combo."""
    combined = CombinedParams(
        z_overextended=float(combo['z_overextended']),
        z_oversold=float(combo['z_oversold']),
        require_momentum_buy=bool(combo['require_momentum_buy']),
        long_only_mode=True,
    )
    z_panel, mr_panel, mom_panel = precomputed
    xs_w = xs_w_by_mode[combo['score_mode']]
    weights, _ = combined_long_only_weights(
        panel, xs_w, z_panel, mr_panel, mom_panel, combined, rebalance,
    )
    rets = panel.pct_change(fill_method=None)
    return portfolio_returns(weights, rets, cost)['strat_net']


def walk_forward_combined_long_only(
    panel: pd.DataFrame,
    xs_params: dict,
    *,
    cost: float | pd.Series = DEFAULT_XS_COST,
    train: int = 252,
    test: int = 63,
    warmup: int = WARMUP,
    combined_grid: dict | None = None,
    beta_window: int = DEFAULT_BETA_WINDOW,
    select: str = 'active_ir',
) -> dict:
    """Walk-forward the combined long-only book, tuning combined params per fold.

    Single-stock signals and per-score-mode XS leg weights are precomputed once
    on the full panel (point-in-time), then sliced per fold. Parameters are
    chosen on the train window only (default objective: active IR vs the
    in-sample equal-weight benchmark), then applied to the next test window.
    """
    combined_grid = combined_grid or COMBINED_GRID
    score_modes = list(dict.fromkeys(combined_grid.get('score_mode', [DEFAULT_SCORE_MODE])))
    rebalance = xs_params.get('rebalance', 5)
    precomputed, xs_w_by_mode = _precompute_combined_panels(
        panel, xs_params, score_modes, beta_window,
    )
    combos = list(iter_param_grid(combined_grid))
    combo_returns = [
        _combined_combo_returns(panel, c, xs_w_by_mode, precomputed, rebalance, cost)
        for c in combos
    ]

    n = len(panel)
    pos = train
    oos_chunks: list[pd.Series] = []
    folds: list[dict] = []
    while pos + test <= n:
        tr_lo, tr_hi = pos - train, pos
        te_lo, te_hi = pos, pos + test
        train_idx = panel.index[tr_lo:tr_hi]
        test_idx = panel.index[te_lo:te_hi]
        train_price = panel.iloc[tr_lo:tr_hi]

        best_score, best_combo, best_idx, best_is_sharpe = -np.inf, None, None, float('nan')
        for i, (combo, r) in enumerate(zip(combos, combo_returns)):
            r_tr = r.reindex(train_idx).dropna()
            if len(r_tr) < 2:
                continue
            score = selection_objective(r_tr, train_price, select)
            if score > best_score:
                best_score, best_combo, best_idx = score, combo, i
                best_is_sharpe = metrics(r_tr)['sharpe']
        if best_idx is None:
            pos += test
            continue
        r_test = combo_returns[best_idx].reindex(test_idx).dropna()
        oos_chunks.append(r_test)
        folds.append({
            'train_end': panel.index[tr_hi - 1],
            'test_end': panel.index[te_hi - 1],
            'best_params': best_combo,
            'in_sample_sharpe': best_is_sharpe,
            'in_sample_score': best_score,
            'oos_sharpe': metrics(r_test)['sharpe'] if len(r_test) else float('nan'),
            'select': select,
        })
        pos += test

    oos_returns = pd.concat(oos_chunks).sort_index() if oos_chunks else pd.Series(dtype=float)
    return {
        'oos_returns': oos_returns,
        'oos_metrics': metrics(oos_returns),
        'folds': folds,
        'n_combos': len(combos),
    }


def _active_oos_summary(
    oos_returns: pd.Series,
    panel: pd.DataFrame,
    folds: int,
    n_trials: int,
    ci_block: int = IR_CI_BLOCK,
    ci_n_boot: int = IR_CI_BOOT,
    ci_alpha: float = IR_CI_ALPHA,
    ci_seed: int = IR_CI_SEED,
) -> dict:
    """Benchmark-relative OOS summary (Sharpe, active IR + CI, verdict)."""
    oos = metrics(oos_returns)
    bench_returns = equal_weight_oos_returns(panel, oos_returns.index)
    bench = metrics(bench_returns)
    active = active_oos_returns(oos_returns, bench_returns)
    active_ir, ci_lower, ci_upper, ir_se = information_ratio_ci(
        active, block=ci_block, n_boot=ci_n_boot, alpha=ci_alpha, seed=ci_seed,
    )
    selection_bar = expected_max_ir_under_null(n_trials, ir_se)
    verdict = validation_verdict(
        oos['sharpe'], active_ir, folds,
        ci_lower=ci_lower, ci_upper=ci_upper, selection_threshold=selection_bar,
    )
    reason = validation_verdict_reason(
        oos['sharpe'], active_ir, folds,
        ci_lower=ci_lower, ci_upper=ci_upper, selection_threshold=selection_bar,
    )
    aligned = pd.concat([
        oos_returns.rename('s'), bench_returns.rename('b'),
    ], axis=1).dropna()
    corr = float(aligned['s'].corr(aligned['b'])) if len(aligned) >= 2 else float('nan')
    return {
        'oos_metrics': oos,
        'benchmark_metrics': bench,
        'active_ir': active_ir,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'ir_se': ir_se,
        'selection_bar': selection_bar,
        'verdict': verdict,
        'verdict_reason': reason,
        'benchmark_correlation': corr,
    }


def report_combined_validation(
    panel: pd.DataFrame,
    xs_params: dict,
    *,
    cost: float | pd.Series = DEFAULT_XS_COST,
    train: int = 252,
    test: int = 63,
    warmup: int = WARMUP,
    combined_grid: dict | None = None,
    beta_window: int = DEFAULT_BETA_WINDOW,
    select: str = 'active_ir',
    ci_block: int = IR_CI_BLOCK,
    ci_n_boot: int = IR_CI_BOOT,
    ci_alpha: float = IR_CI_ALPHA,
    ci_seed: int = IR_CI_SEED,
) -> dict:
    """OOS active-IR report for the combined long-only book vs pure XS + benchmark.

    Both books are walk-forward validated on the same panel/folds with the same
    selection objective; the benchmark is the equal-weight universe on the same
    OOS dates. The combined book is only credited with EDGE if it clears the same
    benchmark-relative gates as the pure XS book — never on absolute return alone.
    """
    combined_grid = combined_grid or COMBINED_GRID
    n_combos = 1
    for values in combined_grid.values():
        n_combos *= max(1, len(values))

    print('\n=== Combined-signal walk-forward validation (long-only) ===')
    print(
        'XS rank + single-stock z-score / momentum confirmation + filing overlay. '
        'Tuned combined params per fold on the train window only.'
    )
    print(
        f"Combined grid: z_overextended={combined_grid.get('z_overextended')} "
        f"z_oversold={combined_grid.get('z_oversold')} "
        f"require_momentum_buy={combined_grid.get('require_momentum_buy')} "
        f"score_mode={combined_grid.get('score_mode')}  ->  {n_combos} combos/fold"
    )
    if select != 'sharpe':
        print(f'Parameter selection objective: {select} (benchmark-relative)')

    combined_wf = walk_forward_combined_long_only(
        panel, xs_params, cost=cost, train=train, test=test, warmup=warmup,
        combined_grid=combined_grid, beta_window=beta_window, select=select,
    )
    xs_strategy = make_xs_strategy(
        'long_only', cost=cost,
        score_mode=xs_params.get('score_mode', DEFAULT_SCORE_MODE),
        beta_window=beta_window,
    )
    xs_wf = walk_forward(
        xs_strategy, panel, _xs_single_combo_grid(xs_params),
        train=train, test=test, warmup=warmup, select=select,
    )

    combined_folds = len(combined_wf['folds'])
    xs_folds = len(xs_wf['folds'])
    combined_sum = _active_oos_summary(
        combined_wf['oos_returns'], panel, combined_folds, n_combos,
        ci_block=ci_block, ci_n_boot=ci_n_boot, ci_alpha=ci_alpha, ci_seed=ci_seed,
    )
    xs_sum = _active_oos_summary(
        xs_wf['oos_returns'], panel, xs_folds, 1,
        ci_block=ci_block, ci_n_boot=ci_n_boot, ci_alpha=ci_alpha, ci_seed=ci_seed,
    )
    benchmark = combined_sum['benchmark_metrics']

    print(f"\n{'Book':32}{'OOS Sharpe':>11}{'AnnRet':>9}{'MaxDD':>9}{'ActiveIR':>10}")
    xs_m = xs_sum['oos_metrics']
    cm = combined_sum['oos_metrics']
    print(f"{'Pure XS long-only':32}{xs_m['sharpe']:>11.2f}"
          f"{xs_m['ann_return']:>9.1%}{xs_m['max_dd']:>9.1%}"
          f"{_format_signed(xs_sum['active_ir']):>10}")
    print(f"{'Combined long-only':32}{cm['sharpe']:>11.2f}"
          f"{cm['ann_return']:>9.1%}{cm['max_dd']:>9.1%}"
          f"{_format_signed(combined_sum['active_ir']):>10}")
    print(f"{'Equal-weight benchmark':32}{benchmark['sharpe']:>11.2f}"
          f"{benchmark['ann_return']:>9.1%}{benchmark['max_dd']:>9.1%}"
          f"{'—':>10}   <- same OOS folds")

    confidence = int(round((1.0 - ci_alpha) * 100))
    ci_lower, ci_upper = combined_sum['ci_lower'], combined_sum['ci_upper']
    if not pd.isna(ci_lower) and not pd.isna(ci_upper) and np.isfinite(ci_lower) and np.isfinite(ci_upper):
        ci_text = f'{_format_signed(ci_lower)}, {_format_signed(ci_upper)}'
    else:
        ci_text = 'unavailable - too few OOS active-return points'
    print(
        f"\nCombined active IR: {_format_signed(combined_sum['active_ir'])}  "
        f'[{confidence}% CI: {ci_text}]  (block-bootstrap, seed={ci_seed})'
    )
    print(
        f'Selection-adjusted bar (N={n_combos} combos): '
        f"expected max IR under null ≈ {_format_signed(combined_sum['selection_bar'])}"
    )
    corr = combined_sum['benchmark_correlation']
    corr_note = '   <- active return is a thin residual; IR is low-power' if (
        not pd.isna(corr) and corr > 0.90) else ''
    print(f'Combined benchmark correlation: {_format_signed(corr)}{corr_note}')
    print(f"Combined verdict: {combined_sum['verdict']} - {combined_sum['verdict_reason']}")
    print(f'Folds: combined={combined_folds}  pure-XS={xs_folds}')
    print(
        '\nNote: the combined overlay only adds edge if it beats BOTH the pure XS '
        'book and the equal-weight benchmark OOS after costs; gating/confirmation '
        'usually trades breadth for lower turnover, not for higher active IR.'
    )

    return {
        'combined_wf': combined_wf,
        'combined_summary': combined_sum,
        'xs_wf': xs_wf,
        'xs_summary': xs_sum,
        'n_combos': n_combos,
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
    n_trials: int = DEFAULT_SELECTION_TRIALS,
    ci_block: int = IR_CI_BLOCK,
    ci_n_boot: int = IR_CI_BOOT,
    ci_alpha: float = IR_CI_ALPHA,
    ci_seed: int = IR_CI_SEED,
    score_mode: str = DEFAULT_SCORE_MODE,
    beta_window: int = DEFAULT_BETA_WINDOW,
    hysteresis: dict | None = None,
    turnover_penalty: float = 0.0,
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
        score_mode=score_mode,
        beta_window=beta_window,
        hysteresis=hysteresis,
        turnover_penalty=turnover_penalty,
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
                score_mode=score_mode,
                beta_window=beta_window,
                hysteresis=hysteresis,
                turnover_penalty=turnover_penalty,
                fixed_folds=selection_wf['folds'],
                **wf_kwargs,
            )
        benchmark = buy_hold_equal_weight_benchmark(panel, wf['oos_returns'].index, cost=cost)
        active = active_oos_returns(wf['oos_returns'], benchmark['ret'])
        active_ir, ci_lower, ci_upper, ir_se = information_ratio_ci(
            active,
            block=ci_block,
            n_boot=ci_n_boot,
            alpha=ci_alpha,
            seed=ci_seed,
        )
        selection_bar = expected_max_ir_under_null(n_trials, ir_se)
        verdict = validation_verdict(
            wf['oos_metrics']['sharpe'],
            active_ir,
            len(wf['folds']),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            selection_threshold=selection_bar,
        )
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
            'information_ratio_ci_lower': ci_lower,
            'information_ratio_ci_upper': ci_upper,
            'information_ratio_se': ir_se,
            'selection_expected_max_ir': selection_bar,
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
        'selection_trials': int(n_trials),
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
        '--n-trials',
        type=int,
        default=DEFAULT_SELECTION_TRIALS,
        help=(
            'Selection-adjustment trial count for the expected max null IR '
            f'(default: {DEFAULT_SELECTION_TRIALS}; rough presets x universes x parameter families explored)'
        ),
    )
    parser.add_argument(
        '--warmup', type=int, default=None,
        help=f'Warmup bars before test window (default: {WARMUP})',
    )
    parser.add_argument(
        '--show-fold-ranks', action='store_true',
        help='Print long/short legs at test start/end for each walk-forward fold',
    )
    parser.add_argument(
        '--score-mode',
        choices=list(SCORE_MODES),
        default=DEFAULT_SCORE_MODE,
        help=f'Cross-sectional scoring rule (default: {DEFAULT_SCORE_MODE})',
    )
    parser.add_argument(
        '--beta-window', type=int, default=DEFAULT_BETA_WINDOW,
        help=f'Rolling beta window for residual_momentum (default: {DEFAULT_BETA_WINDOW})',
    )
    parser.add_argument(
        '--select', choices=['sharpe', 'active_ir'], default='sharpe',
        help='Parameter selection objective on the train window (default: sharpe)',
    )
    parser.add_argument(
        '--use-hysteresis', action='store_true',
        help='Enable rank hysteresis (entry/exit bands) to reduce long-leg turnover',
    )
    parser.add_argument(
        '--entry-rank-pct', type=float, default=0.80,
        help='Hysteresis: enter a long when rank percentile >= this (default 0.80)',
    )
    parser.add_argument(
        '--exit-rank-pct', type=float, default=0.60,
        help='Hysteresis: hold a long until rank percentile < this (default 0.60)',
    )
    parser.add_argument(
        '--max-new-names-per-rebalance', type=int, default=None,
        help='Hysteresis: cap new long entries per rebalance (default: no cap)',
    )
    parser.add_argument(
        '--turnover-penalty', type=float, default=0.0,
        help='Selection objective = active_IR - penalty * avg_turnover (default 0)',
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
        n_trials=args.n_trials,
        select=args.select,
        score_mode=args.score_mode,
        beta_window=args.beta_window,
        hysteresis=hysteresis_kwargs(
            use_hysteresis=args.use_hysteresis,
            entry_rank_pct=args.entry_rank_pct,
            exit_rank_pct=args.exit_rank_pct,
            max_new_names_per_rebalance=args.max_new_names_per_rebalance,
        ),
        turnover_penalty=args.turnover_penalty,
        train=args.train,
        test=args.test,
        warmup=warmup,
    )


if __name__ == '__main__':
    main()
