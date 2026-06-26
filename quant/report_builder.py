from __future__ import annotations

from datetime import datetime, timezone

from quant.data import build_live_frame, fetch_daily_prices, fetch_historical_prices
from quant.metrics import metrics
from quant.portfolio import position_for
from quant.registry import list_models, resolve_models
from quant.reporting import print_model_comparison, run_model_backtest


def cli_usage_block() -> str:
  return """
CLI quick reference
-------------------
  python cli.py models list
  python cli.py report TICKER [--years 5] [--model all]
  python cli.py signal TICKER --model mean-reversion|momentum
  python cli.py backtest TICKER --model mean-reversion|momentum|all [--years 5]
  python cli.py run TICKER --model mean-reversion|momentum
  python cli.py status
  python cli.py history
  python cli.py buy TICKER
  python cli.py sell TICKER
  python cli.py reset

  # Cross-sectional long/short (multi-stock universe)
  python cli.py portfolio backtest --universe MU,NVDA,AMD --signal all --years 5
  python cli.py portfolio ranks --universe MU,NVDA,AMD,TSM --signal momentum

Paper trading only. Data from Yahoo Finance (free, ~15 min delayed).
"""


def build_full_report(ticker: str, years: int = 5, explain: bool = True,
                      model_slug: str = 'all', cost: float = 0.0005,
                      portfolio=None) -> str:
    """Generate a complete text report for terminal output."""
    ticker = ticker.upper()
    models = resolve_models(model_slug)
    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    sections: list[str] = []

    header = [
        '=' * 72,
        f'QUANT ANALYSIS REPORT — {ticker}',
        f'Generated: {now}',
        'Data source: Yahoo Finance (free, typically ~15 min delayed)',
        'DISCLAIMER: Practice / research only. Not investment advice.',
        '=' * 72,
    ]
    sections.append('\n'.join(header))

    # Available models
    sections.append('\nAVAILABLE MODELS')
    sections.append('-' * 40)
    for m in list_models():
        sections.append(f'  {m.slug:<18} {m.name} — {m.description}')

    # Live signals
    sections.append('\n' + '=' * 72)
    sections.append('LIVE SIGNALS')
    sections.append('=' * 72)
    prices = fetch_daily_prices(ticker, lookback_days=max(
        m.min_history_days(**m.default_params()) for m in models
    ))
    prices.name = ticker
    for model in models:
        params = model.default_params()
        try:
            df, live, ts = build_live_frame(prices.copy(), model, **params)
            row = df.iloc[-1]
            in_pos = False
            if portfolio is not None:
                from quant.portfolio import get_model_state
                state = get_model_state(portfolio, ticker, model.slug)
                in_pos = state.get('in_position', position_for(portfolio, ticker)['shares'] > 0)
            sections.append('')
            sections.append(model.format_signal(ticker, row, live, ts, in_pos, **params))
            sections.append(model.explain_math(**params))
        except ValueError as exc:
            sections.append(f'\n[{model.name}] signal unavailable: {exc}')

    # Backtests
    hist = fetch_historical_prices(ticker, years)
    comparison_rows = []
    hold_metrics = None

    sections.append('\n' + '=' * 72)
    sections.append(f'BACKTESTS ({years}-year historical)')
    sections.append('=' * 72)

    for model in models:
        params = model.default_params()
        df, n_trades = model.backtest(hist, cost=cost, **params)
        strat = metrics(df['strat_net'])
        hold = metrics(df['ret'])
        if hold_metrics is None:
            hold_metrics = hold

        from quant.metrics import win_rate
        wr = win_rate(df)

        sections.append('')
        sections.append(f'--- {model.name} ({model.slug}) ---')
        sections.append(f'{"":18}{"STRATEGY":>12}{"BUY & HOLD":>12}')
        sections.append(f'{"Annual return":18}{strat["ann_return"]:>11.1%}{hold["ann_return"]:>12.1%}')
        sections.append(f'{"Annual vol":18}{strat["ann_vol"]:>11.1%}{hold["ann_vol"]:>12.1%}')
        sections.append(f'{"Sharpe ratio":18}{strat["sharpe"]:>12.2f}{hold["sharpe"]:>12.2f}')
        sections.append(f'{"Max drawdown":18}{strat["max_dd"]:>11.1%}{hold["max_dd"]:>12.1%}')
        sections.append(f'{"Trades":18}{n_trades:>12}{"-":>12}')
        sections.append(f'{"Win rate (days)":18}{wr:>11.1%}{"-":>12}')

        if explain:
            from quant.reporting import explain_backtest
            sections.append(explain_backtest(strat, hold, n_trades, wr))

        comparison_rows.append({
            'label': model.name,
            'ann_return': strat['ann_return'],
            'sharpe': strat['sharpe'],
            'max_dd': strat['max_dd'],
            'n_trades': n_trades,
        })

    if hold_metrics and len(comparison_rows) > 1:
        sections.append('\n--- Summary comparison ---')
        sections.append(f'{"Model":<18}{"Return":>10}{"Sharpe":>10}{"Max DD":>10}{"Trades":>8}')
        sections.append('-' * 56)
        for row in comparison_rows:
            sections.append(
                f'{row["label"]:<18}'
                f'{row["ann_return"]:>9.1%}'
                f'{row["sharpe"]:>10.2f}'
                f'{row["max_dd"]:>10.1%}'
                f'{row["n_trades"]:>8}'
            )
        sections.append(
            f'{"Buy & hold":<18}'
            f'{hold_metrics["ann_return"]:>9.1%}'
            f'{hold_metrics["sharpe"]:>10.2f}'
            f'{hold_metrics["max_dd"]:>10.1%}'
            f'{"—":>8}'
        )

        best_return = max(comparison_rows, key=lambda r: r['ann_return'])
        best_sharpe = max(comparison_rows, key=lambda r: r['sharpe'])
        sections.append('')
        sections.append(
            f'Highest return: {best_return["label"]} ({best_return["ann_return"]:+.1%}/yr)'
        )
        sections.append(
            f'Best risk-adjusted (Sharpe): {best_sharpe["label"]} (Sharpe {best_sharpe["sharpe"]:.2f})'
        )

    sections.append('\n' + '=' * 72)
    sections.append('HOW TO USE THESE MODELS')
    sections.append('=' * 72)
    sections.append(cli_usage_block().strip())

    return '\n'.join(sections)
