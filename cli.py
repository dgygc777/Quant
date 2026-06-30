#!/usr/bin/env python3
"""
Quant CLI — multi-model analysis, backtesting, and paper trading.

Practice only. No real money. Yahoo Finance data (~15 min delayed).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from quant.data import (
    build_live_frame,
    fetch_daily_prices,
    fetch_historical_prices,
    fetch_live_quote,
    fetch_panel,
)
from quant.portfolio import (
    DEFAULT_CASH,
    DEFAULT_PORTFOLIO,
    get_model_state,
    load_portfolio,
    paper_buy,
    paper_sell,
    position_for,
    save_portfolio,
    set_model_state,
)
from quant.combined_signal import (
    CombinedParams,
    build_combined_snapshot_df,
    print_combined_signal_report,
    print_strategy_comparison,
    run_strategy_comparison,
)
from quant.data_quality import MIN_COVERAGE, coverage_by_ticker, format_coverage
from quant.models.cross_sectional import assess_panel_quality, print_panel_quality
from quant.params import validate_xs_params
from quant.models.cross_sectional import (
    DEFAULT_BETA_WINDOW,
    DEFAULT_SCORE_MODE,
    DEFAULT_TOP_FRAC,
    SCORE_MODES,
)
from quant.registry import get_model, get_panel_model, list_models, list_panel_models, resolve_models
from quant.report_builder import build_full_report, cli_usage_block
from quant.reporting import print_model_comparison, run_model_backtest, run_panel_backtest
from quant.risk_model import report_sizing
from quant.universe_analysis import (
    analyze_universe_backtests,
    print_backtest_table,
)
from quant.universes import (
    DEFAULT_PRESET,
    describe_preset,
    format_universes_listing,
    resolve_universe,
    validate_universe_size,
)
from quant.momentum_presets import (
    build_momentum_preset_rank_table,
    print_momentum_preset_rank_table,
    print_single_stock_momentum_comparison,
    print_xs_momentum_preset_comparison,
    resolve_momentum_params,
    run_single_stock_momentum_comparison,
    run_xs_momentum_preset_comparison,
    validate_momentum_params,
)

DEFAULT_COST = 0.0005
DEFAULT_COST_BPS = 10.0
DEFAULT_SELECTION_TRIALS = 20
MIN_RISK_REVIEW_NAMES = 6


def _cost_from_bps(cost_bps: float | None) -> float:
    if cost_bps is None:
        return DEFAULT_COST
    return float(cost_bps) / 10_000.0


def resolve_portfolio_universe(args) -> tuple[str, list[str]]:
    preset, tickers = resolve_universe(args.universe, getattr(args, 'tickers', None))
    validate_xs_params(
        top_frac=args.top_frac,
        rebalance=args.rebalance,
        short_window=args.short_window,
        years=args.years,
    )
    validate_universe_size(tickers, args.top_frac)
    return preset, tickers


def add_portfolio_params(p: argparse.ArgumentParser) -> None:
    p.add_argument('--universe', default=DEFAULT_PRESET,
                   help='Universe preset name (default: semis). Use "portfolio universes" to list.')
    p.add_argument('--tickers', default=None,
                   help='Custom comma-separated tickers (overrides --universe)')
    p.add_argument('--signal', default='momentum',
                   choices=['momentum', 'reversal', 'all', 'combo'],
                   help='Cross-sectional signal (default: momentum)')
    p.add_argument('--years', type=int, default=5)
    p.add_argument('--cost-bps', type=float, default=DEFAULT_COST_BPS,
                   help='Round-trip transaction cost in bps for portfolio comparison/validation (default 10)')
    p.add_argument('--n-trials', type=int, default=DEFAULT_SELECTION_TRIALS,
                   help='Selection-adjustment trial count for validation CI gate (default 20)')
    p.add_argument('--top-frac', type=float, default=DEFAULT_TOP_FRAC,
                   help=f'Fraction long / short per leg (default {DEFAULT_TOP_FRAC})')
    p.add_argument('--rebalance', type=int, default=5,
                   help='Rebalance every N trading days (default 5)')
    p.add_argument('--short-window', type=int, default=5,
                   help='Reversal signal lookback in days (default 5)')
    p.add_argument('--momentum-preset', default=None,
                   help='Named momentum lookback (mom_10d, mom_20d, mom_63d, mom_126d_skip21)')
    p.add_argument('--lookback', type=int, default=None,
                   help='Momentum lookback days (overrides --momentum-preset)')
    p.add_argument('--skip', type=int, default=None,
                   help='Momentum skip days (overrides --momentum-preset)')
    p.add_argument('--score-mode', choices=list(SCORE_MODES), default=DEFAULT_SCORE_MODE,
                   help=f'Cross-sectional scoring rule (default: {DEFAULT_SCORE_MODE}). '
                        'relative_momentum/residual_momentum are benchmark-relative.')
    p.add_argument('--beta-window', type=int, default=DEFAULT_BETA_WINDOW,
                   help=f'Rolling beta window for residual_momentum (default {DEFAULT_BETA_WINDOW})')
    p.add_argument('--select', choices=['sharpe', 'active_ir'], default='sharpe',
                   help='Walk-forward parameter selection objective (default: sharpe)')
    p.add_argument('--use-hysteresis', action='store_true',
                   help='Enable rank hysteresis (entry/exit bands) to cut long-leg turnover')
    p.add_argument('--entry-rank-pct', type=float, default=0.80,
                   help='Hysteresis: enter long when rank percentile >= this (default 0.80)')
    p.add_argument('--exit-rank-pct', type=float, default=0.60,
                   help='Hysteresis: hold long until rank percentile < this (default 0.60)')
    p.add_argument('--max-new-names-per-rebalance', type=int, default=None,
                   help='Hysteresis: cap new long entries per rebalance (default: no cap)')
    p.add_argument('--turnover-penalty', type=float, default=0.0,
                   help='Validation: objective = active_IR - penalty * avg_turnover (default 0)')
    p.add_argument('--group-neutral', action='store_true',
                   help='Peer-group-neutral long-only construction: equal capital per '
                        'semiconductor sub-industry, top names within each group')
    p.add_argument('--group-top-frac', type=float, default=None,
                   help='Group-neutral: fraction of names selected within each group '
                        '(default: same as --top-frac)')
    p.add_argument('--compare-momentum-presets', action='store_true',
                   help='Show current ranks across all momentum presets')
    p.add_argument('--no-explain', action='store_true')
    p.add_argument('--no-single-stock', action='store_true',
                   help='Skip per-stock mean-reversion & momentum analysis')
    p.add_argument('--z-overextended', type=float, default=1.5,
                   help='Combined: z above this → WAIT (default 1.5)')
    p.add_argument('--z-oversold', type=float, default=-1.0,
                   help='Combined: z below this → oversold bounce zone (default -1.0)')
    p.add_argument('--no-require-momentum-buy', action='store_true',
                   help='Combined: do not require momentum BUY for long entries')
    p.add_argument('--allow-short-candidates', action='store_true',
                   help='Combined: label SHORT_CANDIDATE (research only)')
    p.add_argument('--no-long-only', action='store_true',
                   help='Disable long-only interpretation of short leg')
    p.add_argument('--risk-threshold', type=float, default=-0.30,
                   help='10KΔ flag threshold for BUY conflicts (default -0.30)')
    p.add_argument('--tenk-cache', default='tenk_cache.json',
                   help='Path to tenk_reader cache JSON')
    p.add_argument('--validate', action='store_true',
                   help='Append lightweight walk-forward OOS summary (compare only)')
    p.add_argument('--validate-combined', action='store_true',
                   help='Also walk-forward validate the combined long-only book vs pure XS '
                        '+ benchmark (active-IR selection). Implies --validate.')


def combined_params_from_args(args) -> CombinedParams:
    return CombinedParams(
        z_overextended=args.z_overextended,
        z_oversold=args.z_oversold,
        require_momentum_buy=not args.no_require_momentum_buy,
        allow_short_candidates=args.allow_short_candidates,
        long_only_mode=not args.no_long_only,
    )


def panel_params_from_args(args) -> dict:
    mom_preset, lookback, skip = resolve_momentum_params(
        getattr(args, 'momentum_preset', None),
        getattr(args, 'lookback', None),
        getattr(args, 'skip', None),
    )
    return {
        'mode': args.signal if args.signal not in ('all', 'combo') else 'momentum',
        'top_frac': args.top_frac,
        'rebalance': args.rebalance,
        'short_window': args.short_window,
        'lookback': lookback,
        'skip': skip,
        'momentum_preset': mom_preset,
        'score_mode': getattr(args, 'score_mode', DEFAULT_SCORE_MODE),
        'beta_window': getattr(args, 'beta_window', DEFAULT_BETA_WINDOW),
    }


def _hysteresis_from_args(args) -> dict:
    """Bundle hysteresis kwargs from CLI args (empty dict when disabled)."""
    if not getattr(args, 'use_hysteresis', False):
        return {}
    return {
        'use_hysteresis': True,
        'entry_rank_pct': float(getattr(args, 'entry_rank_pct', 0.80)),
        'exit_rank_pct': float(getattr(args, 'exit_rank_pct', 0.60)),
        'max_new_names_per_rebalance': getattr(args, 'max_new_names_per_rebalance', None),
    }


def _group_neutral_from_args(args, tickers: list[str]) -> dict:
    """Bundle peer-group-neutral kwargs from CLI args (empty dict when disabled)."""
    if not getattr(args, 'group_neutral', False):
        return {}
    from quant.universes import build_group_map
    out = {'group_neutral': True, 'group_map': build_group_map(list(tickers))}
    gtf = getattr(args, 'group_top_frac', None)
    if gtf is not None:
        out['group_top_frac'] = float(gtf)
    return out


def mom_params_from_panel(params: dict) -> dict:
    return {'lookback': params['lookback'], 'skip': params['skip']}


def validation_grid_from_panel_params(params: dict) -> dict:
    """Single-combo WF grid matching the live cross-sectional rank settings."""
    return {
        'lookback': [int(params['lookback'])],
        'skip': [int(params['skip'])],
        'top_frac': [float(params['top_frac'])],
        'rebalance': [int(params['rebalance'])],
    }


def validation_warmup_from_panel_params(params: dict) -> int:
    """Enough prior bars for the selected momentum preset at test-window start."""
    return max(40, int(params['lookback']) + int(params['skip']) + 5)


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument('--portfolio', type=Path, default=DEFAULT_PORTFOLIO,
                   help='Path to paper portfolio JSON')


def add_model_args(p: argparse.ArgumentParser, default: str = 'mean-reversion') -> None:
    p.add_argument('--model', default=default,
                   help='Model slug: mean-reversion, momentum, or all (default: %(default)s)')


def add_strategy_params(p: argparse.ArgumentParser) -> None:
    """Shared + model-specific optional params."""
    p.add_argument('--window', type=int, default=20,
                   help='Mean-reversion rolling window in days')
    p.add_argument('--entry-z', type=float, default=-1.0,
                   help='Mean-reversion: enter when z < this')
    p.add_argument('--exit-z', type=float, default=0.0,
                   help='Mean-reversion: exit when z >= this')
    p.add_argument('--lookback', type=int, default=None,
                   help='Momentum: lookback window in days (default: classic 126)')
    p.add_argument('--skip', type=int, default=None,
                   help='Momentum: skip recent days in signal (default: classic 21)')
    p.add_argument('--momentum-preset', default=None,
                   help='Momentum: named preset (mom_10d, mom_20d, mom_63d, mom_126d_skip21)')
    p.add_argument('--vol-window', type=int, default=63,
                   help='Momentum: realized-vol rolling window (default 63)')
    p.add_argument('--target-vol', type=float, default=0.15,
                   help='Momentum: annual vol target for sizing (default 0.15)')
    p.add_argument('--max-leverage', type=float, default=3.0,
                   help='Momentum: max position weight (default 3.0)')
    p.add_argument('--no-vol-scale', action='store_true',
                   help='Momentum: disable volatility targeting')
    p.add_argument('--short', action='store_true',
                   help='Momentum: allow short positions (long/short mode)')


def model_params_from_args(model, args) -> dict:
    params = model.default_params()
    if model.slug == 'mean-reversion':
        if hasattr(args, 'window'):
            params['window'] = args.window
        if hasattr(args, 'entry_z'):
            params['entry_z'] = args.entry_z
        if hasattr(args, 'exit_z'):
            params['exit_z'] = args.exit_z
    elif model.slug == 'momentum':
        mom_preset, lb, sk = resolve_momentum_params(
            getattr(args, 'momentum_preset', None),
            getattr(args, 'lookback', None),
            getattr(args, 'skip', None),
        )
        params['lookback'] = lb
        params['skip'] = sk
        params['momentum_preset'] = mom_preset
        if hasattr(args, 'vol_window'):
            params['vol_window'] = args.vol_window
        if hasattr(args, 'target_vol'):
            params['target_vol'] = args.target_vol
        if hasattr(args, 'max_leverage'):
            params['max_leverage'] = args.max_leverage
        if hasattr(args, 'no_vol_scale'):
            params['vol_scale'] = not args.no_vol_scale
        if hasattr(args, 'short'):
            params['long_only'] = not args.short
    return params


def print_status(portfolio: dict) -> None:
    print('\n=== Paper portfolio ===')
    print(f'Cash:           ${portfolio["cash"]:,.2f}')
    total = portfolio['cash']
    if not portfolio['positions']:
        print('Positions:      (none)')
    for ticker, pos in portfolio['positions'].items():
        try:
            price, _ = fetch_live_quote(ticker)
            value = pos['shares'] * price
            pnl = (price - pos['avg_cost']) * pos['shares']
            pnl_pct = (price / pos['avg_cost'] - 1) if pos['avg_cost'] else 0.0
            total += value
            print(f'  {ticker}: {pos["shares"]:.0f} sh @ ${pos["avg_cost"]:.2f} '
                  f'→ ${price:.2f}  (${value:,.0f}, P&L {pnl:+,.0f} / {pnl_pct:+.1%})')
        except ValueError:
            print(f'  {ticker}: {pos["shares"]:.0f} shares (quote unavailable)')
    print(f'Total equity:   ${total:,.2f}')
    ret = total / portfolio['initial_cash'] - 1
    print(f'Return:         {ret:+.1%} vs ${portfolio["initial_cash"]:,.0f} start')


def print_history(portfolio: dict, limit: int = 20) -> None:
    trades = portfolio['trades'][-limit:]
    print(f'\n=== Last {len(trades)} trades ===')
    if not trades:
        print('(no trades yet)')
        return
    for t in trades:
        extra = t.get('cost', t.get('proceeds', 0))
        print(f'{t["time"][:19]}  {t["side"]:4} {t["shares"]:.0f} {t["ticker"]} '
              f'@ ${t["price"]:.2f}  (${extra:,.0f})  [{t.get("reason", "")}]')


def cmd_models_list(_args) -> None:
    print('\nSingle-asset models (one ticker):')
    for m in list_models():
        print(f'  {m.slug:<20} {m.name}')
        print(f'  {"":20} {m.description}')
    print('\nPanel models (stock universe):')
    for m in list_panel_models():
        print(f'  {m.slug:<20} {m.name}')
        print(f'  {"":20} {m.description}')


def cmd_signal(args) -> None:
    model = get_model(args.model)
    params = model_params_from_args(model, args)
    portfolio = load_portfolio(args.portfolio)
    ticker = args.ticker.upper()

    prices = fetch_daily_prices(ticker, model.min_history_days(**params))
    prices.name = ticker
    df, live, ts = build_live_frame(prices, model, **params)
    row = df.iloc[-1]
    state = get_model_state(portfolio, ticker, model.slug)
    in_pos = state.get('in_position', position_for(portfolio, ticker)['shares'] > 0)

    print(model.format_signal(ticker, row, live, ts, in_pos, **params))
    if not args.no_math:
        print(model.explain_math(**params))


def cmd_backtest(args) -> None:
    ticker = args.ticker.upper()
    hist = fetch_historical_prices(ticker, args.years)
    models = resolve_models(args.model)
    rows = []
    for model in models:
        params = model_params_from_args(model, args)
        summary = run_model_backtest(
            model, hist, DEFAULT_COST, not args.no_explain,
            args.years, ticker, **params,
        )
        rows.append(summary)
    if len(rows) > 1:
        print_model_comparison(rows)
        hold = rows[0]['hold']
        print(
            f'\nBuy & hold benchmark: {hold["ann_return"]:+.1%}/yr, '
            f'Sharpe {hold["sharpe"]:.2f}, max DD {hold["max_dd"]:.1%}'
        )


def cmd_report(args) -> None:
    portfolio = load_portfolio(args.portfolio)
    text = build_full_report(
        args.ticker, years=args.years, explain=not args.no_explain,
        model_slug=args.model, portfolio=portfolio,
    )
    print(text)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(text)
        print(f'\nReport saved to {args.save}')


def cmd_run(args) -> None:
    model = get_model(args.model)
    params = model_params_from_args(model, args)
    portfolio = load_portfolio(args.portfolio)
    ticker = args.ticker.upper()

    prices = fetch_daily_prices(ticker, model.min_history_days(**params))
    prices.name = ticker
    df, live, ts = build_live_frame(prices, model, **params)
    row = df.iloc[-1]
    state = get_model_state(portfolio, ticker, model.slug)
    in_pos = state.get('in_position', position_for(portfolio, ticker)['shares'] > 0)
    new_in_pos, action = model.next_action(row, in_pos, **params)

    print(model.format_signal(ticker, row, live, ts, in_pos, **params))
    sig_val = model.signal_value(row, **params)

    if action == 'BUY':
        paper_buy(portfolio, ticker, live, reason=f'{model.slug} signal')
        print(f'\n→ Paper BUY executed at ${live:.2f}')
    elif action == 'SELL':
        paper_sell(portfolio, ticker, live, reason=f'{model.slug} signal')
        print(f'\n→ Paper SELL executed at ${live:.2f}')
    else:
        print('\n→ No trade — conditions not met.')

    set_model_state(portfolio, ticker, model.slug, {
        'in_position': new_in_pos,
        'last_value': sig_val,
    })
    save_portfolio(portfolio, args.portfolio)


def cmd_watch(args) -> None:
    model = get_model(args.model)
    params = model_params_from_args(model, args)
    portfolio = load_portfolio(args.portfolio)
    ticker = args.ticker.upper()

    print(f'Watching {ticker} / {model.name} every {args.interval}s (Ctrl+C to stop).')
    print('Data: Yahoo Finance (free, typically ~15 min delayed).\n')
    while True:
        prices = fetch_daily_prices(ticker, model.min_history_days(**params))
        prices.name = ticker
        df, live, ts = build_live_frame(prices, model, **params)
        row = df.iloc[-1]
        state = get_model_state(portfolio, ticker, model.slug)
        in_pos = state.get('in_position', position_for(portfolio, ticker)['shares'] > 0)
        print(model.format_signal(ticker, row, live, ts, in_pos, **params))
        print(f'--- next refresh in {args.interval}s ---\n')
        time.sleep(args.interval)


def _print_single_stock_backtests(panel, weights, xs_scores, years, cost, skip: bool) -> None:
    if skip:
        return
    df = analyze_universe_backtests(panel, cost, weights, xs_scores)
    print_backtest_table(df, years)


def _print_single_stock_snapshots(panel, weights, xs_scores, skip: bool,
                                  combined: CombinedParams,
                                  mom_params: dict | None = None,
                                  momentum_preset: str = 'mom_126d_skip21',
                                  risk_threshold: float = -0.30,
                                  tenk_cache: str = 'tenk_cache.json',
                                  coverage: pd.Series | None = None) -> pd.DataFrame | None:
    if skip:
        return None
    df = build_combined_snapshot_df(
        panel, weights, xs_scores, combined,
        mom_params=mom_params, momentum_preset=momentum_preset,
        coverage=coverage, min_coverage=MIN_COVERAGE,
    )
    return print_combined_signal_report(
        df, combined, cache_path=tenk_cache, risk_threshold=risk_threshold,
    )


def _print_wf_validation(
    panel: pd.DataFrame,
    train: int = 504,
    test: int = 63,
    coverage: pd.Series | None = None,
    momentum_preset: str = 'custom',
    xs_params: dict | None = None,
    headline_cost: float = DEFAULT_COST,
    headline_cost_bps: float | None = None,
    n_trials: int = DEFAULT_SELECTION_TRIALS,
    select: str = 'sharpe',
    score_mode: str = DEFAULT_SCORE_MODE,
    beta_window: int = DEFAULT_BETA_WINDOW,
    hysteresis: dict | None = None,
    turnover_penalty: float = 0.0,
    group_neutral: dict | None = None,
    validate_combined: bool = False,
) -> dict | None:
    from validate_cross_sectional import (
        ACTIVE_IR_EDGE_MARGIN,
        COST_SWEEP_BPS,
        GRID,
        MIN_VALIDATION_NAMES,
        VERDICT_MATCHES,
        WARMUP,
        compare_weighting_validation,
        print_weighting_validation_report,
        print_transaction_cost_sensitivity,
        report_combined_validation,
        report_panel_validation,
        transaction_cost_sensitivity,
    )
    hysteresis = hysteresis or {}
    group_neutral = group_neutral or {}
    print('\n=== Walk-forward validation (same universe / XS rules) ===')
    print('Full fold detail: python3 validate_cross_sectional.py --universe <preset>')
    from quant.data_quality import filter_panel_by_coverage
    if xs_params is not None:
        validation_grid = validation_grid_from_panel_params(xs_params)
        validation_warmup = validation_warmup_from_panel_params(xs_params)
        validation_label = (
            f"{momentum_preset} live preset "
            f"(lookback={xs_params['lookback']}, skip={xs_params['skip']}, "
            f"top_frac={xs_params['top_frac']}, rebalance={xs_params['rebalance']})"
        )
        print(f'Validation strategy: {validation_label}')
    else:
        validation_grid = GRID
        validation_warmup = WARMUP
        validation_label = 'grid-optimized XS strategy (not the live preset)'
        print(f'Validation strategy: {validation_label}')
    panel, _, dropped = filter_panel_by_coverage(panel, MIN_COVERAGE, coverage=coverage)
    if len(dropped):
        print(f'Excluded from validation (coverage < {MIN_COVERAGE:.0%}): {format_coverage(dropped)}')
    print(f'Validation survivors: {panel.shape[1]} names, {len(panel)} rows')
    if panel.shape[1] < MIN_VALIDATION_NAMES:
        print(
            f'Validation skipped: universe is too thin after coverage filter '
            f'({panel.shape[1]} names; need at least {MIN_VALIDATION_NAMES}).'
        )
        return None
    if len(panel) < train + test:
        print(f'Panel too short for WF ({len(panel)} bars; need {train + test}).')
        return None
    wf = report_panel_validation(
        'Portfolio compare WF',
        panel,
        validation_grid,
        book='long_only',
        cost=headline_cost,
        n_trials=n_trials,
        select=select,
        score_mode=score_mode,
        beta_window=beta_window,
        hysteresis=hysteresis,
        turnover_penalty=turnover_penalty,
        group_neutral=group_neutral,
        train=train,
        test=test,
        warmup=validation_warmup,
    )
    oos = wf['oos_metrics']
    headline_sweep_bps = (
        float(headline_cost_bps)
        if headline_cost_bps is not None
        else float(headline_cost) * 10_000.0
    )
    sweep_levels = sorted({float(level) for level in COST_SWEEP_BPS} | {headline_sweep_bps})
    cost_sensitivity = transaction_cost_sensitivity(
        panel,
        validation_grid,
        cost_bps_levels=sweep_levels,
        selection_cost=headline_cost,
        n_trials=n_trials,
        score_mode=score_mode,
        beta_window=beta_window,
        hysteresis=hysteresis,
        turnover_penalty=turnover_penalty,
        train=train,
        test=test,
        warmup=validation_warmup,
    )
    print_transaction_cost_sensitivity(
        cost_sensitivity,
        label=validation_label,
        headline_cost_bps=headline_cost_bps,
    )
    sizing_validation = compare_weighting_validation(
        panel,
        validation_grid,
        train=train,
        test=test,
        warmup=validation_warmup,
        cost=headline_cost,
    )
    print_weighting_validation_report(sizing_validation)
    equal_row = sizing_validation.get('rows', {}).get('equal', {})
    equal_sharpe = equal_row.get('sharpe')
    if group_neutral.get('group_neutral'):
        print('Self-check: SKIPPED - sizing/cost diagnostics use the plain top-k book, '
              'not the group-neutral book, so OOS Sharpe is not expected to match.')
    elif equal_row.get('error') is None and equal_sharpe is not None and abs(equal_sharpe - oos['sharpe']) <= 1e-10:
        print('Self-check: OK - equal weighting matches main long-only OOS Sharpe.')
    else:
        print(
            'Self-check: WARNING - equal weighting OOS Sharpe diverges from '
            f"main long-only OOS Sharpe ({equal_sharpe} vs {oos['sharpe']:.6f})."
        )
    sweep_row = next(
        (
            row for row in cost_sensitivity.get('rows', [])
            if abs(row['cost_bps'] - headline_sweep_bps) <= 1e-9
        ),
        None,
    )
    if group_neutral.get('group_neutral'):
        print('Self-check: SKIPPED - cost sweep uses the plain top-k book, not the '
              'group-neutral book.')
    elif sweep_row and abs(sweep_row['sharpe'] - oos['sharpe']) <= 1e-6:
        print('Self-check: OK - cost sweep headline row matches main long-only OOS Sharpe.')
    else:
        sweep_sharpe = sweep_row.get('sharpe') if sweep_row else None
        print(
            'Self-check: WARNING - cost sweep headline row diverges from '
            f"main long-only OOS Sharpe ({sweep_sharpe} vs {oos['sharpe']:.6f})."
        )
    combined_validation = None
    if validate_combined and xs_params is not None:
        combined_validation = report_combined_validation(
            panel,
            xs_params,
            cost=headline_cost,
            train=train,
            test=test,
            warmup=validation_warmup,
            beta_window=beta_window,
            select=select if select != 'sharpe' else 'active_ir',
        )

    point_ir = wf.get('information_ratio')
    ci_lower = wf.get('information_ratio_ci_lower')
    ci_gate_case = (
        point_ir is not None
        and ci_lower is not None
        and not pd.isna(point_ir)
        and not pd.isna(ci_lower)
        and float(point_ir) > ACTIVE_IR_EDGE_MARGIN
        and float(ci_lower) <= ACTIVE_IR_EDGE_MARGIN
    )
    if ci_gate_case:
        if wf.get('validation_verdict') != VERDICT_MATCHES:
            raise AssertionError(
                'CI gate invariant failed: point IR clears margin but CI lower does not, '
                f"yet verdict is {wf.get('validation_verdict')!r}."
            )
        print('Self-check: OK - point IR clears margin but CI lower does not, so verdict stays MATCHES.')
    else:
        print('Self-check: OK - CI gate invariant armed for point-estimate-only edges.')
    return {
        'folds': len(wf['folds']),
        'oos_sharpe': oos['sharpe'],
        'oos_ann_return': oos['ann_return'],
        'oos_max_dd': oos['max_dd'],
        'benchmark_oos_sharpe': wf.get('benchmark_oos_metrics', {}).get('sharpe'),
        'benchmark_oos_ann_return': wf.get('benchmark_oos_metrics', {}).get('ann_return'),
        'benchmark_oos_max_dd': wf.get('benchmark_oos_metrics', {}).get('max_dd'),
        'active_oos_sharpe': wf.get('active_oos_sharpe'),
        'information_ratio': wf.get('information_ratio'),
        'information_ratio_ci_lower': wf.get('information_ratio_ci_lower'),
        'information_ratio_ci_upper': wf.get('information_ratio_ci_upper'),
        'information_ratio_se': wf.get('information_ratio_se'),
        'selection_expected_max_ir': wf.get('selection_expected_max_ir'),
        'selection_trials': wf.get('selection_trials'),
        'benchmark_correlation': wf.get('benchmark_correlation'),
        'validation_verdict': wf.get('validation_verdict'),
        'validation_verdict_reason': wf.get('validation_verdict_reason'),
        'sizing_validation': sizing_validation,
        'winning_weighting': sizing_validation.get('best_weighting'),
        'risk_parity_beats_equal': sizing_validation.get('risk_parity_beats_equal'),
        'cost_sensitivity': cost_sensitivity,
        'break_even_cost_bps': cost_sensitivity.get('break_even_bps'),
        'headline_cost_bps': headline_cost_bps,
        'validation_param_grid': validation_grid,
        'validation_warmup': validation_warmup,
        'validation_label': validation_label,
        'combined_validation': combined_validation,
    }


def _ranked_risk_pool(weights: pd.Series, xs_scores: pd.Series,
                      min_names: int = MIN_RISK_REVIEW_NAMES) -> tuple[list[str], list[str]]:
    active_longs = weights[weights > 0].index.tolist()
    scored = xs_scores.dropna().sort_values(ascending=False)
    if scored.empty:
        return active_longs, active_longs

    target_count = min(len(scored), max(len(active_longs), min_names))
    pool = scored.index[:target_count].tolist()
    for ticker in active_longs:
        if ticker not in pool:
            pool.append(ticker)
    return active_longs, pool


def _print_ranked_candidate_risk(panel: pd.DataFrame, weights: pd.Series,
                                 xs_scores: pd.Series, xs_params: dict,
                                 years: int,
                                 coverage: pd.Series | None = None) -> dict | None:
    active_longs, risk_pool = _ranked_risk_pool(weights, xs_scores)
    if not risk_pool:
        print('\n=== Ranked Candidate Risk Review ===')
        print('No ranked candidates to analyze.')
        return None

    ticker_coverage = coverage.reindex(panel.columns) if coverage is not None else coverage_by_ticker(panel)
    pool_coverage = ticker_coverage.reindex(risk_pool).fillna(0.0)
    coverage_ok = pool_coverage >= MIN_COVERAGE
    speculative = pool_coverage.loc[~coverage_ok].sort_values()
    risk_sized_pool = [ticker for ticker in risk_pool if bool(coverage_ok.get(ticker, False))]

    print('\n=== Ranked Candidate Risk Review ===')
    print(
        f"Signal: {xs_params.get('momentum_preset', 'custom')} "
        f"(lookback={xs_params.get('lookback')}, skip={xs_params.get('skip')})"
    )
    print(f'Active long leg ({len(active_longs)}): {_join_tickers(active_longs)}')
    print(f'Ranked candidate pool ({len(risk_pool)}): {_join_tickers(risk_pool)}')
    if len(speculative):
        print(
            'INSUFFICIENT HISTORY - speculative only, not validated or risk-sized '
            f'(coverage): {format_coverage(speculative)}'
        )
    if not risk_sized_pool:
        print('No coverage-approved candidate set remains for covariance sizing.')
        return {
            'active_longs': active_longs,
            'risk_pool': risk_pool,
            'risk_sized_pool': [],
            'speculative': {str(ticker): float(cov) for ticker, cov in speculative.items()},
            'coverage': {str(ticker): float(cov) for ticker, cov in pool_coverage.items()},
            'sample_rows': 0,
            'rc_pct_by_ticker': {},
        }

    returns = (
        panel[risk_sized_pool]
        .pct_change(fill_method=None)
        .dropna(how='all')
        .dropna(axis=1, how='all')
    )
    if returns.empty or returns.shape[1] == 0:
        print('Not enough return history for the coverage-approved risk-sized pool.')
        return {
            'active_longs': active_longs,
            'risk_pool': risk_pool,
            'risk_sized_pool': [],
            'speculative': {str(ticker): float(cov) for ticker, cov in speculative.items()},
            'coverage': {str(ticker): float(cov) for ticker, cov in pool_coverage.items()},
            'sample_rows': 0,
            'rc_pct_by_ticker': {},
        }

    risk_sized_pool = list(returns.columns)
    start_date = returns.index[0].date()
    end_date = returns.index[-1].date()
    print(f'Coverage-approved risk-sized pool ({len(risk_sized_pool)}): {_join_tickers(risk_sized_pool)}')
    if len(risk_sized_pool) > len([t for t in active_longs if t in risk_sized_pool]):
        print(
            'Risk review uses the top-ranked candidate pool, not only the active long leg, '
            'so concentration can inform selection before final sizing.'
        )
    print(
        f'Covariance window: {len(returns)} daily return rows, '
        f'{start_date} to {end_date} (--years {years})'
    )
    min_rows = max(5 * len(risk_sized_pool), 252)
    if len(returns) < min_rows:
        print(
            f'Warning: covariance estimate is data-limited ({len(returns)} rows < {min_rows}); '
            'min-variance weights in particular should not be trusted because they invert a noisy matrix.'
        )
    try:
        report = report_sizing(returns, label='Ranked Candidate Risk-Pool')
    except (AssertionError, ValueError, TypeError, FloatingPointError) as exc:
        print(f'Risk sizing unavailable: {exc}')
        return {
            'active_longs': active_longs,
            'risk_pool': risk_pool,
            'risk_sized_pool': risk_sized_pool,
            'speculative': {str(ticker): float(cov) for ticker, cov in speculative.items()},
            'coverage': {str(ticker): float(cov) for ticker, cov in pool_coverage.items()},
            'sample_rows': len(returns),
            'rc_pct_by_ticker': {},
        }

    equal_rc = report['equal_risk_contributions']
    max_ticker = str(equal_rc['pct_of_total'].idxmax())
    min_ticker = str(equal_rc['pct_of_total'].idxmin())
    return {
        'active_longs': active_longs,
        'risk_pool': risk_pool,
        'risk_sized_pool': risk_sized_pool,
        'speculative': {str(ticker): float(cov) for ticker, cov in speculative.items()},
        'coverage': {str(ticker): float(cov) for ticker, cov in pool_coverage.items()},
        'sample_rows': len(returns),
        'sample_start': start_date.isoformat(),
        'sample_end': end_date.isoformat(),
        'equal_ann_vol': float(report['portfolio_volatility'].get('equal', float('nan'))),
        'rc_pct_by_ticker': {
            str(ticker): float(pct)
            for ticker, pct in equal_rc['pct_of_total'].items()
        },
        'max_rc_ticker': max_ticker,
        'max_rc_pct': float(equal_rc.loc[max_ticker, 'pct_of_total']),
        'min_rc_ticker': min_ticker,
        'min_rc_pct': float(equal_rc.loc[min_ticker, 'pct_of_total']),
    }


def _fmt_pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return 'n/a'
    return f'{value:.1%}'


def _fmt_float(value: float | None) -> str:
    if value is None or pd.isna(value):
        return 'n/a'
    return f'{value:.2f}'


def _fmt_weighting(value: str | None) -> str:
    if not value:
        return 'n/a'
    return str(value).replace('_', '-')


def _join_tickers(tickers: list[str], max_items: int = 8) -> str:
    tickers = [str(t) for t in tickers]
    if not tickers:
        return 'none'
    if len(tickers) <= max_items:
        return ', '.join(tickers)
    return ', '.join(tickers[:max_items]) + f', ... (+{len(tickers) - max_items})'


def _ticker_set_from_tenk(tenk_df: pd.DataFrame | None, mask: pd.Series) -> set[str]:
    if tenk_df is None or tenk_df.empty:
        return set()
    return set(tenk_df[mask]['ticker'].astype(str))


def _buy_strategy_conclusion(
    preset: str,
    xs_params: dict,
    weights: pd.Series | None,
    xs_scores: pd.Series | None,
    validation_stats: dict | None,
    tenk_df: pd.DataFrame | None,
    sizing_stats: dict | None,
    coverage: pd.Series | None = None,
) -> str | None:
    if weights is None or xs_scores is None:
        return None

    active_longs = weights[weights > 0].index.tolist()
    risk_pool = list(sizing_stats.get('risk_pool', [])) if sizing_stats else []
    if not risk_pool:
        risk_pool = active_longs
    risk_sized_pool = list(sizing_stats.get('risk_sized_pool', [])) if sizing_stats else []
    speculative_map = dict(sizing_stats.get('speculative', {})) if sizing_stats else {}
    if coverage is not None:
        pool_coverage = coverage.reindex(risk_pool).fillna(0.0)
        for ticker, cov in pool_coverage.items():
            if cov < MIN_COVERAGE:
                speculative_map[str(ticker)] = float(cov)
        if not risk_sized_pool:
            risk_sized_pool = [
                ticker for ticker in risk_pool
                if float(pool_coverage.get(ticker, 0.0)) >= MIN_COVERAGE
            ]
    coverage_bad = set(speculative_map)
    if not active_longs and not risk_pool:
        return (
            'Validation verdict FAILS: no current ranked long leg is active, so this remains '
            'a research/watchlist screen only. Standing caveat: this is a curated-universe '
            'research overlay, not a validated edge.'
        )

    active_ranked = xs_scores.reindex(active_longs).dropna().sort_values(ascending=False)
    risk_ranked = xs_scores.reindex(risk_pool).dropna().sort_values(ascending=False)
    active_desc = ', '.join(
        f'{ticker} {score:+.1%}' for ticker, score in active_ranked.items()
        if not pd.isna(score)
    )
    if not active_desc:
        active_desc = _join_tickers(active_longs)
    risk_pool_desc = ', '.join(
        f'{ticker} {score:+.1%}' for ticker, score in risk_ranked.items()
        if not pd.isna(score)
    )
    if not risk_pool_desc:
        risk_pool_desc = _join_tickers(risk_pool)

    folds = int(validation_stats.get('folds', 0)) if validation_stats else 0
    oos_sharpe = validation_stats.get('oos_sharpe') if validation_stats else None
    oos_ann_return = validation_stats.get('oos_ann_return') if validation_stats else None
    oos_max_dd = validation_stats.get('oos_max_dd') if validation_stats else None
    benchmark_oos_sharpe = validation_stats.get('benchmark_oos_sharpe') if validation_stats else None
    active_oos_sharpe = validation_stats.get('active_oos_sharpe') if validation_stats else None
    active_ir = validation_stats.get('information_ratio') if validation_stats else None
    active_ir_ci_lower = validation_stats.get('information_ratio_ci_lower') if validation_stats else None
    active_ir_ci_upper = validation_stats.get('information_ratio_ci_upper') if validation_stats else None
    selection_bar = validation_stats.get('selection_expected_max_ir') if validation_stats else None
    selection_trials = validation_stats.get('selection_trials') if validation_stats else None
    verdict_reason = validation_stats.get('validation_verdict_reason') if validation_stats else None
    winning_weighting = validation_stats.get('winning_weighting') if validation_stats else None
    risk_parity_beats_equal = validation_stats.get('risk_parity_beats_equal') if validation_stats else None
    headline_cost_bps = validation_stats.get('headline_cost_bps') if validation_stats else None
    break_even_cost_bps = validation_stats.get('break_even_cost_bps') if validation_stats else None
    cost_rows = validation_stats.get('cost_sensitivity', {}).get('rows', []) if validation_stats else []
    cost_context = (
        f'at {_fmt_float(headline_cost_bps)} bps round-trip cost'
        if headline_cost_bps is not None
        else 'at the configured transaction cost'
    )
    if cost_rows and float(cost_rows[0].get('active_ir', 0.0)) <= 0.0:
        break_even_desc = 'break-even cost: already <=0 gross'
    elif break_even_cost_bps is None:
        break_even_desc = 'break-even cost: no crossing in swept range'
    else:
        break_even_desc = f'break-even cost: ~{_fmt_float(break_even_cost_bps)} bps'
    verdict = validation_stats.get('validation_verdict') if validation_stats else 'FAILS'
    validation_edge = verdict == 'EDGE'
    ir_evidence = f"active-return IR {_fmt_float(active_ir)}"
    if active_ir_ci_lower is not None or active_ir_ci_upper is not None:
        ir_evidence += (
            f" [95% CI {_fmt_float(active_ir_ci_lower)}, {_fmt_float(active_ir_ci_upper)}]"
        )
    if selection_bar is not None:
        trials_desc = f"N={selection_trials}" if selection_trials is not None else 'N=n/a'
        ir_evidence += f", selection bar {_fmt_float(selection_bar)} ({trials_desc})"
    if verdict_reason:
        ir_evidence += f", reason: {verdict_reason}"
    if validation_edge:
        validation_desc = (
            f"Validation verdict EDGE {cost_context}: strategy OOS Sharpe {_fmt_float(oos_sharpe)} vs "
            f"equal-weight benchmark {_fmt_float(benchmark_oos_sharpe)} "
            f"(Sharpe spread {_fmt_float(active_oos_sharpe)}, {ir_evidence}), "
            f"ann return {_fmt_pct(oos_ann_return)}, max DD {_fmt_pct(oos_max_dd)} "
            f"over {folds} folds; {break_even_desc}."
        )
    elif verdict == 'MATCHES BENCHMARK — captures sector beta, not alpha':
        validation_desc = (
            f"Validation verdict {verdict} {cost_context}: strategy OOS Sharpe {_fmt_float(oos_sharpe)} vs "
            f"benchmark {_fmt_float(benchmark_oos_sharpe)} "
            f"(Sharpe spread {_fmt_float(active_oos_sharpe)}, {ir_evidence}), "
            f"max DD {_fmt_pct(oos_max_dd)}, "
            f"folds {folds}; {break_even_desc}; this is a research candidate that does not beat the basket."
        )
    else:
        validation_desc = (
            f"Validation verdict FAILS {cost_context}: the ranking has no validated benchmark-relative edge "
            f"(strategy OOS Sharpe {_fmt_float(oos_sharpe)} vs benchmark "
            f"{_fmt_float(benchmark_oos_sharpe)}, Sharpe spread {_fmt_float(active_oos_sharpe)}, "
            f"{ir_evidence}, "
            f"max DD {_fmt_pct(oos_max_dd)}, folds {folds}; {break_even_desc}), so the momentum list is a "
            f"research/watchlist candidate screen only."
        )

    clean_vetted: list[str] = []
    no_filing_data: list[str] = []
    filing_flagged: list[str] = []
    filing_flag_desc = 'filing-risk conflicts: none'
    no_data_desc = 'no missing filing coverage in the ranked long leg'
    if tenk_df is not None and not tenk_df.empty:
        ticker_index = tenk_df.assign(ticker=tenk_df['ticker'].astype(str)).set_index('ticker', drop=False)
        actions = tenk_df['final_action'].astype(str)
        risk_flags = tenk_df['risk_flag'].fillna(False).astype(bool)
        tenk_ok = tenk_df['tenk_ok'].fillna(False).astype(bool) if 'tenk_ok' in tenk_df else pd.Series(
            False, index=tenk_df.index,
        )
        clean_mask = (actions == 'BUY') & ~risk_flags & tenk_ok
        flagged_mask = risk_flags

        clean_set = _ticker_set_from_tenk(tenk_df, clean_mask)
        flagged_report = tenk_df[flagged_mask]['ticker'].astype(str).tolist()
        filing_flagged = [ticker for ticker in risk_pool if ticker in set(flagged_report)]
        for ticker in risk_pool:
            if ticker in coverage_bad:
                continue
            if ticker not in ticker_index.index:
                no_filing_data.append(f'{ticker} (no filing data)')
            elif not bool(ticker_index.loc[ticker].get('tenk_ok', False)):
                no_filing_data.append(f'{ticker} (no filing data)')
            elif ticker in clean_set:
                clean_vetted.append(ticker)

        if flagged_report:
            filing_flag_desc = (
                'filing-risk conflicts: '
                + '; '.join(f'{ticker} flagged - verify filing before sizing' for ticker in flagged_report)
            )
        if no_filing_data:
            no_data_desc = 'filing coverage gaps: ' + ', '.join(no_filing_data)
    else:
        no_filing_data = [f'{ticker} (no filing data)' for ticker in risk_pool if ticker not in coverage_bad]
        no_data_desc = 'filing coverage gaps: ' + ', '.join(no_filing_data)

    if speculative_map:
        speculative_series = pd.Series(speculative_map).sort_values()
        speculative_desc = (
            'insufficient-history speculative bucket: '
            + format_coverage(speculative_series)
            + ' - not validated or risk-sized'
        )
    else:
        speculative_desc = 'insufficient-history speculative bucket: none'

    rc_pct = dict(sizing_stats.get('rc_pct_by_ticker', {})) if sizing_stats else {}
    if not risk_sized_pool and rc_pct:
        risk_sized_pool = [ticker for ticker in risk_pool if ticker in rc_pct]
    risk_contribution_pool = [ticker for ticker in risk_sized_pool if ticker in rc_pct]
    equal_share = 1.0 / len(risk_contribution_pool) if risk_contribution_pool else float('nan')
    concentration_threshold = 1.5 * equal_share if risk_contribution_pool else float('inf')
    concentration_flags = [
        ticker for ticker in risk_contribution_pool
        if float(rc_pct.get(ticker, 0.0)) > concentration_threshold
    ]
    if sizing_stats and risk_contribution_pool:
        if concentration_flags:
            recommended_weighting = winning_weighting if winning_weighting and winning_weighting != 'equal' else 'risk_parity'
            risk_desc = (
                'risk-concentration flags: '
                + '; '.join(
                    f'{ticker} is a disproportionate risk contributor '
                    f'({_fmt_pct(rc_pct[ticker])} vs equal-dollar {_fmt_pct(equal_share)})'
                    for ticker in concentration_flags
                )
                + f'; equal-dollar sizing is inappropriate - use {_fmt_weighting(recommended_weighting)} sizing instead'
            )
        else:
            risk_desc = (
                f"equal-weight risk contribution has no >1.5x concentration flag; highest is "
                f"{sizing_stats['max_rc_ticker']} at {_fmt_pct(sizing_stats['max_rc_pct'])} "
                f"with equal-weight annualized vol {_fmt_pct(sizing_stats['equal_ann_vol'])}"
            )
    else:
        risk_desc = 'equal-weight risk contribution was not available for a coverage-approved pool'
    if winning_weighting:
        rp_desc = (
            'risk_parity beat equal'
            if risk_parity_beats_equal
            else 'risk_parity did not beat equal'
        )
        risk_desc += f'; OOS sizing validation winner: {winning_weighting} ({rp_desc})'

    contradiction_parts = []
    for ticker in risk_pool:
        lenses = []
        if not validation_edge:
            lenses.append(
                'benchmark-relative validation'
                if verdict == 'MATCHES BENCHMARK — captures sector beta, not alpha'
                else 'negative validation regime'
            )
        if ticker in coverage_bad:
            lenses.append('insufficient history')
        if ticker in filing_flagged:
            lenses.append('filing-risk flag')
        if ticker in concentration_flags:
            lenses.append('risk-concentration flag')
        if lenses:
            contradiction_parts.append(
                f"{ticker} momentum-ranked but contradicted by {', '.join(lenses)} - "
                f"do not treat as high-conviction"
            )
    contradiction_desc = (
        'Contradictions: ' + '; '.join(contradiction_parts) + '.'
        if contradiction_parts
        else 'Contradictions: none among the ranked long names.'
    )

    if validation_edge:
        actionable = [
            ticker for ticker in active_longs
            if ticker in clean_vetted
            if ticker not in filing_flagged
            if ticker not in coverage_bad
        ]
        action_desc = (
            f"Actionable candidate list after filing and validation gates: {_join_tickers(actionable)}."
            if actionable
            else 'No actionable candidate survives the filing and validation gates.'
        )
    else:
        no_filing_tickers = {item.split()[0] for item in no_filing_data}
        research_names = [
            ticker if ticker not in no_filing_tickers
            else f'{ticker} (no filing data)'
            for ticker in risk_pool
            if ticker not in coverage_bad
        ]
        action_desc = f"Names to research: {', '.join(research_names) if research_names else 'none after coverage gate'}."

    return (
        f"{validation_desc} Momentum rank ({xs_params.get('momentum_preset', 'custom')}, "
        f"lookback={xs_params.get('lookback')}, skip={xs_params.get('skip')}) selected active "
        f"long leg {active_desc}; risk review pool {risk_pool_desc}; {speculative_desc}; "
        f"{filing_flag_desc}; {no_data_desc}; {risk_desc}. "
        f"{contradiction_desc} {action_desc} Standing caveat: this is a curated-universe "
        f"research overlay, not a validated edge."
    )


def _print_buy_strategy_conclusion(
    preset: str,
    xs_params: dict,
    weights: pd.Series | None,
    xs_scores: pd.Series | None,
    validation_stats: dict | None,
    tenk_df: pd.DataFrame | None,
    sizing_stats: dict | None,
    coverage: pd.Series | None = None,
) -> None:
    conclusion = _buy_strategy_conclusion(
        preset, xs_params, weights, xs_scores,
        validation_stats, tenk_df, sizing_stats, coverage=coverage,
    )
    if conclusion:
        print('\n=== Strategy Conclusion ===')
        print(conclusion)


def cmd_portfolio_universes(_args) -> None:
    print(format_universes_listing())


def cmd_portfolio_compare(args) -> None:
    preset, universe = resolve_portfolio_universe(args)
    panel = fetch_panel(universe, args.years)
    coverage = coverage_by_ticker(panel)
    quality = assess_panel_quality(panel)
    print_panel_quality(quality)
    xs_params = panel_params_from_args(args)
    xs_params['mode'] = 'momentum'
    validate_momentum_params(xs_params['lookback'], xs_params['skip'], n_days=len(panel))
    strategy_cost = _cost_from_bps(getattr(args, 'cost_bps', None))
    combined = combined_params_from_args(args)
    rows = run_strategy_comparison(panel, xs_params, combined, strategy_cost)
    print_strategy_comparison(rows)
    print(
        f'\nBenchmark note: equal-weight is the universe basket; cash/zero is shown for L/S context. '
        f'Headline strategy cost: {getattr(args, "cost_bps", DEFAULT_COST * 10_000):.1f} bps round-trip.'
    )
    validation_stats = None
    if (
        getattr(args, 'validate', False)
        or getattr(args, 'validate_combined', False)
        or getattr(args, 'group_neutral', False)
    ):
        validation_stats = _print_wf_validation(
            panel,
            coverage=coverage,
            momentum_preset=xs_params['momentum_preset'],
            xs_params=xs_params,
            headline_cost=strategy_cost,
            headline_cost_bps=getattr(args, 'cost_bps', DEFAULT_COST * 10_000),
            n_trials=getattr(args, 'n_trials', DEFAULT_SELECTION_TRIALS),
            select=getattr(args, 'select', 'sharpe'),
            score_mode=getattr(args, 'score_mode', DEFAULT_SCORE_MODE),
            beta_window=getattr(args, 'beta_window', DEFAULT_BETA_WINDOW),
            hysteresis=_hysteresis_from_args(args),
            turnover_penalty=getattr(args, 'turnover_penalty', 0.0),
            group_neutral=_group_neutral_from_args(args, list(panel.columns)),
            validate_combined=getattr(args, 'validate_combined', False),
        )
    weights = None
    scores = None
    tenk_df = None
    sizing_stats = None
    if not args.no_single_stock:
        model = get_panel_model()
        weights = model.current_weights(panel, **xs_params)
        scores = model.current_ranks(panel, **xs_params)
        tenk_df = _print_single_stock_snapshots(
            panel, weights, scores, False, combined,
            mom_params=mom_params_from_panel(xs_params),
            momentum_preset=xs_params['momentum_preset'],
            risk_threshold=args.risk_threshold,
            tenk_cache=args.tenk_cache,
            coverage=coverage,
        )
        sizing_stats = _print_ranked_candidate_risk(
            panel, weights, scores, xs_params, args.years, coverage=coverage,
        )
    print(f'\nUniverse preset: {preset} — {describe_preset(preset)}')
    _print_buy_strategy_conclusion(
        preset, xs_params, weights, scores,
        validation_stats, tenk_df, sizing_stats, coverage=coverage,
    )


def cmd_portfolio_backtest(args) -> None:
    if getattr(args, 'compare', False):
        cmd_portfolio_compare(args)
        return

    preset, universe = resolve_portfolio_universe(args)
    panel = fetch_panel(universe, args.years)
    coverage = coverage_by_ticker(panel)
    params = panel_params_from_args(args)
    validate_momentum_params(params['lookback'], params['skip'], n_days=len(panel))
    model = get_panel_model()
    xs_mode = args.signal if args.signal in ('momentum', 'reversal') else 'momentum'
    xs_params = {**params, 'mode': xs_mode}
    xs_scores = model.current_ranks(panel, **xs_params)
    try:
        xs_weights = model.current_weights(panel, **xs_params)
    except ValueError:
        xs_weights = pd.Series(dtype=float)
    rows = []

    if args.signal in ('momentum', 'reversal'):
        params = panel_params_from_args(args)
        params['mode'] = args.signal
        summary = run_panel_backtest(
            model, panel, DEFAULT_COST, not args.no_explain, args.years,
            f'XS {args.signal.title()} — {len(panel.columns)} names ({args.years}y)',
            **params,
        )
        rows.append(summary)
    elif args.signal == 'combo':
        params = panel_params_from_args(args)
        combo_df = model.backtest_combo(panel, cost=DEFAULT_COST, **params)
        from quant.metrics import metrics
        from quant.reporting import print_xs_backtest_report, xs_win_rate
        strat = metrics(combo_df['strat_net'])
        bench = metrics(combo_df['ret'])
        n_rebal = int(len(panel) / params['rebalance'])
        wr = xs_win_rate(combo_df)
        print_xs_backtest_report(
            f'XS Combo 50/50 — {len(panel.columns)} names ({args.years}y)',
            strat, bench, n_rebal, wr, explain=not args.no_explain,
        )
        rows.append({
            'label': 'Combo 50/50', 'ann_return': strat['ann_return'],
            'sharpe': strat['sharpe'], 'max_dd': strat['max_dd'], 'n_trades': n_rebal,
        })
    else:  # all
        for sig in ('momentum', 'reversal'):
            params = panel_params_from_args(args)
            params['mode'] = sig
            summary = run_panel_backtest(
                model, panel, DEFAULT_COST, not args.no_explain, args.years,
                f'XS {sig.title()} — {len(panel.columns)} names ({args.years}y)',
                **params,
            )
            rows.append(summary)
        if len(rows) > 1:
            print_model_comparison(rows)
            bench = rows[0]['hold']
            print(
                f'\nEqual-weight universe benchmark: {bench["ann_return"]:+.1%}/yr, '
                f'Sharpe {bench["sharpe"]:.2f}, max DD {bench["max_dd"]:.1%}'
            )

    _print_single_stock_backtests(
        panel, xs_weights, xs_scores, args.years, DEFAULT_COST,
        skip=args.no_single_stock,
    )
    if not args.no_single_stock:
        combined = combined_params_from_args(args)
        _print_single_stock_snapshots(
            panel, xs_weights, xs_scores, False, combined,
            mom_params=mom_params_from_panel(xs_params),
            momentum_preset=xs_params['momentum_preset'],
            risk_threshold=args.risk_threshold,
            tenk_cache=args.tenk_cache,
            coverage=coverage,
        )


def cmd_portfolio_momentum_compare(args) -> None:
    preset, universe = resolve_portfolio_universe(args)
    panel = fetch_panel(universe, args.years)
    base = {
        'top_frac': args.top_frac,
        'rebalance': args.rebalance,
        'market_neutral': True,
    }
    rows = run_xs_momentum_preset_comparison(panel, base, DEFAULT_COST)
    print_xs_momentum_preset_comparison(rows, preset, args.years)


def cmd_portfolio_ranks(args) -> None:
    preset, universe = resolve_portfolio_universe(args)
    panel = fetch_panel(universe, max(2, args.years))
    coverage = coverage_by_ticker(panel)
    print_panel_quality(assess_panel_quality(panel))
    params = panel_params_from_args(args)
    validate_momentum_params(params['lookback'], params['skip'], n_days=len(panel))

    if getattr(args, 'compare_momentum_presets', False):
        if args.signal != 'momentum':
            raise ValueError('--compare-momentum-presets requires --signal momentum')
        df = build_momentum_preset_rank_table(panel)
        print_momentum_preset_rank_table(df)
        return

    model = get_panel_model()
    if args.signal in ('all', 'combo'):
        raise ValueError('ranks requires --signal momentum or reversal')
    params['mode'] = args.signal
    weights = model.current_weights(panel, **params)
    scores = model.current_ranks(panel, **params)
    print(model.format_ranks(weights, scores, universe, preset_name=preset, **params))
    if not args.no_single_stock:
        combined = combined_params_from_args(args)
        _print_single_stock_snapshots(
            panel, weights, scores, False, combined,
            mom_params=mom_params_from_panel(params),
            momentum_preset=params['momentum_preset'],
            risk_threshold=args.risk_threshold,
            tenk_cache=args.tenk_cache,
            coverage=coverage,
        )
    if not getattr(args, 'no_math', False):
        print(model.explain_math(**params))


def add_trend_params(p: argparse.ArgumentParser) -> None:
    p.add_argument('--universe', default=DEFAULT_PRESET,
                   help='Universe preset name (default: semis)')
    p.add_argument('--tickers', default=None,
                   help='Custom comma-separated tickers (overrides --universe)')
    p.add_argument('--years', type=int, default=10)
    p.add_argument('--trend-mode', choices=['sma', 'tsmom'], default='sma',
                   help='Trend signal: price>SMA (sma) or trailing-return>0 (tsmom)')
    p.add_argument('--trend-windows', default=None,
                   help='Comma-separated trend windows to search (default 50,100,150,200,250)')
    p.add_argument('--vol-target', action='store_true',
                   help='Scale exposure toward a volatility target (de-risk in turbulent regimes)')
    p.add_argument('--target-vol', type=float, default=0.15,
                   help='Annualized vol target when --vol-target (default 0.15)')
    p.add_argument('--vol-window', type=int, default=63,
                   help='Realized-vol window for vol targeting (default 63)')
    p.add_argument('--max-leverage', type=float, default=1.0,
                   help='Max exposure when vol targeting (default 1.0 = no leverage)')
    p.add_argument('--cost-bps', type=float, default=DEFAULT_COST_BPS,
                   help='Cost in bps per unit exposure turnover (default 10)')
    p.add_argument('--cash-rate', type=float, default=0.0,
                   help='Annual cash/risk-free rate earned when out of the market (default 0)')
    p.add_argument('--train', type=int, default=504, help='Walk-forward train window (days)')
    p.add_argument('--test', type=int, default=63, help='Walk-forward test window (days)')


def cmd_portfolio_trend(args) -> None:
    from quant.data_quality import filter_panel_by_coverage
    from quant.trend_overlay import TREND_WINDOW_GRID, report_trend_overlay_validation

    preset, universe = resolve_universe(args.universe, getattr(args, 'tickers', None))
    panel = fetch_panel(universe, args.years)
    coverage = coverage_by_ticker(panel)
    print_panel_quality(assess_panel_quality(panel))
    panel, _, dropped = filter_panel_by_coverage(panel, MIN_COVERAGE, coverage=coverage)
    if len(dropped):
        print(f'Excluded (coverage < {MIN_COVERAGE:.0%}): {format_coverage(dropped)}')
    if panel.shape[1] < 2:
        raise ValueError('Need at least 2 names with adequate coverage to form a basket.')
    if args.trend_windows:
        windows = [int(w) for w in args.trend_windows.split(',') if w.strip()]
    else:
        windows = list(TREND_WINDOW_GRID)
    if len(panel) < args.train + args.test:
        raise ValueError(
            f'Panel too short for walk-forward ({len(panel)} bars; '
            f'need train+test = {args.train + args.test}).'
        )
    report_trend_overlay_validation(
        panel,
        train=args.train,
        test=args.test,
        mode=args.trend_mode,
        windows=windows,
        use_vol_target=getattr(args, 'vol_target', False),
        target_vol=args.target_vol,
        vol_window=args.vol_window,
        max_leverage=args.max_leverage,
        cost=_cost_from_bps(getattr(args, 'cost_bps', None)),
        cash_rate=args.cash_rate,
    )


def cmd_portfolio_filing(args) -> None:
    from quant.data_quality import filter_panel_by_coverage
    from quant.filing_factor import (
        DEFAULT_FILING_ACTION,
        DEFAULT_FILING_THRESHOLD,
        report_filing_factor_validation,
    )

    preset, universe = resolve_portfolio_universe(args)
    panel = fetch_panel(universe, args.years)
    coverage = coverage_by_ticker(panel)
    print_panel_quality(assess_panel_quality(panel))
    panel, _, dropped = filter_panel_by_coverage(panel, MIN_COVERAGE, coverage=coverage)
    if len(dropped):
        print(f'Excluded (coverage < {MIN_COVERAGE:.0%}): {format_coverage(dropped)}')
    xs_params = panel_params_from_args(args)
    xs_params['mode'] = 'momentum'
    validate_momentum_params(xs_params['lookback'], xs_params['skip'], n_days=len(panel))
    report_filing_factor_validation(
        panel,
        xs_params,
        cache_path=getattr(args, 'tenk_cache', 'tenk_cache.json'),
        threshold=getattr(args, 'filing_threshold', DEFAULT_FILING_THRESHOLD),
        action=getattr(args, 'filing_action', DEFAULT_FILING_ACTION),
        activation_lag=getattr(args, 'filing_activation_lag', 1),
        cost=_cost_from_bps(getattr(args, 'cost_bps', None)),
        train=getattr(args, 'train', 504),
        test=getattr(args, 'test', 63),
        select='active_ir',
    )


def cmd_stock_momentum_compare(args) -> None:
    ticker = args.ticker.upper()
    hist = fetch_historical_prices(ticker, args.years)
    validate_momentum_params(126, 21, n_days=len(hist))
    rows = run_single_stock_momentum_comparison(hist.rename(ticker), DEFAULT_COST)
    print_single_stock_momentum_comparison(rows, ticker, args.years)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Quant toolkit: multi-model backtesting and paper trading.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=cli_usage_block(),
    )
    sub = parser.add_subparsers(dest='command', required=True)

    p_models = sub.add_parser('models', help='List available models')
    p_models_sub = p_models.add_subparsers(dest='models_cmd', required=True)
    p_models_sub.add_parser('list', help='List all models')

    p_signal = sub.add_parser('signal', help='Current model signal for a ticker')
    p_signal.add_argument('ticker')
    add_model_args(p_signal)
    add_strategy_params(p_signal)
    add_common_args(p_signal)
    p_signal.add_argument('--no-math', action='store_true')

    p_bt = sub.add_parser('backtest', help='Historical backtest')
    p_bt.add_argument('ticker')
    add_model_args(p_bt, default='all')
    p_bt.add_argument('--years', type=int, default=5)
    p_bt.add_argument('--no-explain', action='store_true')
    add_strategy_params(p_bt)

    p_report = sub.add_parser('report', help='Full text report (signals + backtests)')
    p_report.add_argument('ticker')
    add_model_args(p_report, default='all')
    p_report.add_argument('--years', type=int, default=5)
    p_report.add_argument('--no-explain', action='store_true')
    p_report.add_argument('--save', type=Path, help='Save report to a text file')
    add_common_args(p_report)

    p_run = sub.add_parser('run', help='Apply model rules to paper portfolio')
    p_run.add_argument('ticker')
    add_model_args(p_run)
    add_strategy_params(p_run)
    add_common_args(p_run)

    p_buy = sub.add_parser('buy', help='Manual paper buy')
    p_buy.add_argument('ticker')
    add_common_args(p_buy)

    p_sell = sub.add_parser('sell', help='Manual paper sell')
    p_sell.add_argument('ticker')
    add_common_args(p_sell)

    p_status = sub.add_parser('status', help='Paper portfolio summary')
    add_common_args(p_status)

    p_hist = sub.add_parser('history', help='Trade history')
    p_hist.add_argument('--limit', type=int, default=20)
    add_common_args(p_hist)

    p_watch = sub.add_parser('watch', help='Poll signal periodically')
    p_watch.add_argument('ticker')
    add_model_args(p_watch)
    p_watch.add_argument('--interval', type=int, default=300)
    add_strategy_params(p_watch)
    add_common_args(p_watch)

    p_reset = sub.add_parser('reset', help='Reset paper portfolio')
    add_common_args(p_reset)

    p_port = sub.add_parser('portfolio', help='Cross-sectional multi-stock model')
    p_port_sub = p_port.add_subparsers(dest='portfolio_cmd', required=True)

    p_port_univ = p_port_sub.add_parser('universes', help='List universe presets')
    p_port_univ.add_argument('--no-note', action='store_true', help='Skip selection note')

    p_port_bt = p_port_sub.add_parser('backtest', help='Backtest long/short universe')
    add_portfolio_params(p_port_bt)
    p_port_bt.add_argument('--compare', action='store_true',
                           help='Run full strategy comparison (incl. combined signal)')

    p_port_cmp = p_port_sub.add_parser('compare', help='Compare all portfolio strategies')
    add_portfolio_params(p_port_cmp)

    p_port_ranks = p_port_sub.add_parser('ranks', help='Current long/short targets')
    add_portfolio_params(p_port_ranks)
    p_port_ranks.add_argument('--no-math', action='store_true')

    p_port_momcmp = p_port_sub.add_parser(
        'momentum-compare', help='Compare cross-sectional momentum lookbacks',
    )
    add_portfolio_params(p_port_momcmp)
    p_port_momcmp.set_defaults(signal='momentum')

    p_port_trend = p_port_sub.add_parser(
        'trend', help='Trend/vol-timing overlay on the basket (risk-adjusted, vs cash)',
    )
    add_trend_params(p_port_trend)

    p_port_filing = p_port_sub.add_parser(
        'filing', help='Point-in-time filing-risk negative filter (data-gated validation)',
    )
    add_portfolio_params(p_port_filing)
    p_port_filing.add_argument('--filing-threshold', type=float, default=-0.30,
                               help='Flag names with filing change-score <= this (default -0.30)')
    p_port_filing.add_argument('--filing-action', choices=['exclude', 'half_weight'],
                               default='exclude', help='Action on flagged names (default exclude)')
    p_port_filing.add_argument('--filing-activation-lag', type=int, default=1,
                               help='Trading days after filing date before score goes live (default 1)')
    p_port_filing.add_argument('--train', type=int, default=504, help='Walk-forward train window')
    p_port_filing.add_argument('--test', type=int, default=63, help='Walk-forward test window')

    p_stock = sub.add_parser('stock', help='Single-stock analysis tools')
    p_stock_sub = p_stock.add_subparsers(dest='stock_cmd', required=True)
    p_stock_momcmp = p_stock_sub.add_parser(
        'momentum-compare', help='Compare momentum lookbacks on one ticker',
    )
    p_stock_momcmp.add_argument('ticker')
    p_stock_momcmp.add_argument('--years', type=int, default=5)

    p_help = sub.add_parser('help', help='Show CLI usage')
    p_help.add_argument('topic', nargs='?', default='')

    args = parser.parse_args()

    try:
        if args.command == 'models' and args.models_cmd == 'list':
            cmd_models_list(args)
        elif args.command == 'signal':
            cmd_signal(args)
        elif args.command == 'backtest':
            cmd_backtest(args)
        elif args.command == 'report':
            cmd_report(args)
        elif args.command == 'run':
            cmd_run(args)
        elif args.command == 'buy':
            live, _ = fetch_live_quote(args.ticker)
            portfolio = load_portfolio(args.portfolio)
            paper_buy(portfolio, args.ticker, live, reason='manual buy')
            save_portfolio(portfolio, args.portfolio)
            print(f'Paper BUY {args.ticker.upper()} at ${live:.2f}')
        elif args.command == 'sell':
            live, _ = fetch_live_quote(args.ticker)
            portfolio = load_portfolio(args.portfolio)
            paper_sell(portfolio, args.ticker, live, reason='manual sell')
            save_portfolio(portfolio, args.portfolio)
            print(f'Paper SELL {args.ticker.upper()} at ${live:.2f}')
        elif args.command == 'status':
            print_status(load_portfolio(args.portfolio))
        elif args.command == 'history':
            print_history(load_portfolio(args.portfolio), args.limit)
        elif args.command == 'watch':
            cmd_watch(args)
        elif args.command == 'reset':
            portfolio = {
                'cash': DEFAULT_CASH,
                'initial_cash': DEFAULT_CASH,
                'positions': {},
                'strategy_state': {},
                'trades': [],
            }
            save_portfolio(portfolio, args.portfolio)
            print(f'Portfolio reset to ${DEFAULT_CASH:,.0f} cash.')
        elif args.command == 'portfolio' and args.portfolio_cmd == 'universes':
            cmd_portfolio_universes(args)
        elif args.command == 'portfolio' and args.portfolio_cmd == 'backtest':
            cmd_portfolio_backtest(args)
        elif args.command == 'portfolio' and args.portfolio_cmd == 'compare':
            cmd_portfolio_compare(args)
        elif args.command == 'portfolio' and args.portfolio_cmd == 'ranks':
            cmd_portfolio_ranks(args)
        elif args.command == 'portfolio' and args.portfolio_cmd == 'momentum-compare':
            cmd_portfolio_momentum_compare(args)
        elif args.command == 'portfolio' and args.portfolio_cmd == 'trend':
            cmd_portfolio_trend(args)
        elif args.command == 'portfolio' and args.portfolio_cmd == 'filing':
            cmd_portfolio_filing(args)
        elif args.command == 'stock' and args.stock_cmd == 'momentum-compare':
            cmd_stock_momentum_compare(args)
        elif args.command == 'help':
            print(cli_usage_block())
        else:
            parser.print_help()
    except (ValueError, KeyboardInterrupt) as exc:
        if isinstance(exc, KeyboardInterrupt):
            print('\nStopped.')
        else:
            print(f'Error: {exc}', file=sys.stderr)
            sys.exit(1)


if __name__ == '__main__':
    main()
