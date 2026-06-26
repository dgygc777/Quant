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

from quant.data import (
    build_live_frame,
    fetch_daily_prices,
    fetch_historical_prices,
    fetch_live_quote,
    fetch_panel,
    parse_universe,
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
from quant.models.cross_sectional import DEFAULT_UNIVERSE
from quant.registry import get_model, get_panel_model, list_models, list_panel_models, resolve_models
from quant.report_builder import build_full_report, cli_usage_block
from quant.reporting import print_model_comparison, run_model_backtest, run_panel_backtest

DEFAULT_UNIVERSE_STR = ','.join(DEFAULT_UNIVERSE)
DEFAULT_COST = 0.0005


def add_portfolio_params(p: argparse.ArgumentParser) -> None:
    p.add_argument('--universe', default=DEFAULT_UNIVERSE_STR,
                   help='Comma-separated tickers (default: semis basket)')
    p.add_argument('--signal', default='momentum',
                   choices=['momentum', 'reversal', 'all', 'combo'],
                   help='Cross-sectional signal (default: momentum)')
    p.add_argument('--years', type=int, default=5)
    p.add_argument('--top-frac', type=float, default=0.33,
                   help='Fraction long / short per leg (default 0.33)')
    p.add_argument('--rebalance', type=int, default=5,
                   help='Rebalance every N trading days (default 5)')
    p.add_argument('--short-window', type=int, default=5,
                   help='Reversal signal lookback in days (default 5)')
    p.add_argument('--lookback', type=int, default=126,
                   help='Momentum signal lookback (default 126)')
    p.add_argument('--skip', type=int, default=21,
                   help='Momentum signal skip days (default 21)')
    p.add_argument('--no-explain', action='store_true')


def panel_params_from_args(args) -> dict:
    return {
        'mode': args.signal if args.signal not in ('all', 'combo') else 'momentum',
        'top_frac': args.top_frac,
        'rebalance': args.rebalance,
        'short_window': args.short_window,
        'lookback': args.lookback,
        'skip': args.skip,
    }


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
    p.add_argument('--lookback', type=int, default=126,
                   help='Momentum: lookback window in days (default 126)')
    p.add_argument('--skip', type=int, default=21,
                   help='Momentum: skip recent days in signal (default 21)')
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
        if hasattr(args, 'lookback'):
            params['lookback'] = args.lookback
        if hasattr(args, 'skip'):
            params['skip'] = args.skip
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


def cmd_portfolio_backtest(args) -> None:
    universe = parse_universe(args.universe)
    panel = fetch_panel(universe, args.years)
    model = get_panel_model()
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


def cmd_portfolio_ranks(args) -> None:
    universe = parse_universe(args.universe)
    panel = fetch_panel(universe, max(2, args.years))
    model = get_panel_model()
    params = panel_params_from_args(args)
    if args.signal in ('all', 'combo'):
        raise ValueError('ranks requires --signal momentum or reversal')
    params['mode'] = args.signal
    weights = model.current_weights(panel, **params)
    scores = model.current_ranks(panel, **params)
    print(model.format_ranks(weights, scores, universe, **params))
    if not getattr(args, 'no_math', False):
        print(model.explain_math(**params))


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

    p_port_bt = p_port_sub.add_parser('backtest', help='Backtest long/short universe')
    add_portfolio_params(p_port_bt)

    p_port_ranks = p_port_sub.add_parser('ranks', help='Current long/short targets')
    add_portfolio_params(p_port_ranks)
    p_port_ranks.add_argument('--no-math', action='store_true')

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
        elif args.command == 'portfolio' and args.portfolio_cmd == 'backtest':
            cmd_portfolio_backtest(args)
        elif args.command == 'portfolio' and args.portfolio_cmd == 'ranks':
            cmd_portfolio_ranks(args)
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
