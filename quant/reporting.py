from __future__ import annotations

from quant.metrics import metrics, win_rate


def print_backtest_report(title: str, strat: dict, hold: dict, n_trades: int,
                          win_rate_val: float, explain: bool = True) -> None:
    """Print side-by-side strategy vs buy-and-hold metrics."""
    print(f'\n=== {title} ===')
    print(f'{"":18}{"STRATEGY":>12}{"BUY & HOLD":>12}')
    print(f'{"Annual return":18}{strat["ann_return"]:>11.1%}{hold["ann_return"]:>12.1%}')
    print(f'{"Annual vol":18}{strat["ann_vol"]:>11.1%}{hold["ann_vol"]:>12.1%}')
    print(f'{"Sharpe ratio":18}{strat["sharpe"]:>12.2f}{hold["sharpe"]:>12.2f}')
    print(f'{"Max drawdown":18}{strat["max_dd"]:>11.1%}{hold["max_dd"]:>12.1%}')
    print(f'{"Trades":18}{n_trades:>12}{"-":>12}')
    print(f'{"Win rate (days)":18}{win_rate_val:>11.1%}{"-":>12}')
    if explain:
        print(explain_backtest(strat, hold, n_trades, win_rate_val))


def explain_backtest(strat: dict, hold: dict, n_trades: int,
                     win_rate_val: float) -> str:
    """Plain-language guide to what each backtest number means."""
    beat_hold = strat['ann_return'] > hold['ann_return']
    smoother = strat['ann_vol'] < hold['ann_vol']
    better_risk_adj = strat['sharpe'] > hold['sharpe']
    shallower_dd = strat['max_dd'] > hold['max_dd']

    lines = [
        '',
        'What these numbers mean',
        '-----------------------',
        'Annual return',
        '  Average yearly gain/loss if you compounded daily returns over the period.',
        f'  Strategy: {strat["ann_return"]:+.1%}/yr  |  Buy & hold: {hold["ann_return"]:+.1%}/yr.',
        f'  → Strategy {"beat" if beat_hold else "lagged"} buy-and-hold on raw return.',
        '',
        'Annual vol (volatility)',
        '  How much daily returns swing around their average, scaled to a year.',
        '  Higher = a bumpier ride. Lower is not automatically better if return is also lower.',
        f'  Strategy vol {strat["ann_vol"]:.1%} vs hold {hold["ann_vol"]:.1%} '
        f'→ strategy was {"smoother" if smoother else "rougher"}.',
        '',
        'Sharpe ratio',
        '  Return earned per unit of risk (return ÷ volatility). Higher is better.',
        '  Rough guide: < 0 lose money on a risk-adjusted basis; ~1 acceptable; > 2 strong.',
        f'  Strategy Sharpe {strat["sharpe"]:.2f} vs hold {hold["sharpe"]:.2f} '
        f'→ strategy {"won" if better_risk_adj else "lost"} on risk-adjusted performance.',
        '',
        'Max drawdown',
        '  Worst peak-to-trough drop during the test — your largest underwater stretch.',
        '  Measures pain tolerance: -50% means $100k briefly became $50k before recovering.',
        f'  Strategy {strat["max_dd"]:.1%} vs hold {hold["max_dd"]:.1%} '
        f'→ strategy had a {"shallower" if shallower_dd else "deeper"} worst dip.',
        '',
        'Trades',
        f'  {n_trades} round-trip entries/exits in the backtest (each buy or sell counts).',
        '  More trades = more transaction costs and more chances to be wrong.',
        '  Only applies to the strategy column; buy-and-hold enters once and stays.',
        '',
        'Win rate (days)',
        f'  {win_rate_val:.1%} of days while the strategy was invested had a positive daily return.',
        '  A high win rate does NOT guarantee profitability — a few large losing days',
        '  can wipe out many small winners. Always read this alongside return and drawdown.',
        '',
        'How to read the comparison',
        '  A good strategy beats buy-and-hold on Sharpe or drawdown, not just return.',
        '  Past backtest results do not predict future performance.',
    ]
    return '\n'.join(lines)


def print_model_comparison(rows: list[dict]) -> None:
    """Print a summary table comparing multiple models."""
    print('\n=== Model comparison ===')
    print(f'{"Model":<18}{"Return":>10}{"Sharpe":>10}{"Max DD":>10}{"Trades":>8}')
    print('-' * 56)
    for row in rows:
        print(f'{row["label"]:<18}'
              f'{row["ann_return"]:>9.1%}'
              f'{row["sharpe"]:>10.2f}'
              f'{row["max_dd"]:>10.1%}'
              f'{row["n_trades"]:>8}')


def run_model_backtest(model, price, cost: float, explain: bool, years: int,
                       ticker: str, **params) -> dict:
    """Backtest one model and print the report. Returns summary dict."""
    df, n_trades = model.backtest(price, cost=cost, **params)
    strat = metrics(df['strat_net'])
    hold = metrics(df['ret'])
    wr = win_rate(df)
    print_backtest_report(
        f'Backtest: {ticker.upper()} / {model.name} ({years}y)',
        strat, hold, n_trades, wr, explain=explain,
    )
    return {
        'slug': model.slug,
        'label': model.name,
        'ann_return': strat['ann_return'],
        'sharpe': strat['sharpe'],
        'max_dd': strat['max_dd'],
        'n_trades': n_trades,
        'strat': strat,
        'hold': hold,
    }


def xs_win_rate(df: pd.DataFrame) -> float:
    r = df['strat_net'].dropna()
    return float((r > 0).mean()) if len(r) else float('nan')


def print_xs_backtest_report(title: str, strat: dict, bench: dict, n_rebal: int,
                             win_rate_val: float, explain: bool = True) -> None:
    """Print cross-sectional strategy vs equal-weight universe benchmark."""
    print(f'\n=== {title} ===')
    print(f'{"":18}{"STRATEGY":>12}{"EW UNIV":>12}')
    print(f'{"Annual return":18}{strat["ann_return"]:>11.1%}{bench["ann_return"]:>12.1%}')
    print(f'{"Annual vol":18}{strat["ann_vol"]:>11.1%}{bench["ann_vol"]:>12.1%}')
    print(f'{"Sharpe ratio":18}{strat["sharpe"]:>12.2f}{bench["sharpe"]:>12.2f}')
    print(f'{"Max drawdown":18}{strat["max_dd"]:>11.1%}{bench["max_dd"]:>12.1%}')
    print(f'{"Rebalances":18}{n_rebal:>12}{"-":>12}')
    print(f'{"Win rate (days)":18}{win_rate_val:>11.1%}{"-":>12}')
    if explain:
        print(explain_backtest(strat, bench, n_rebal, win_rate_val))


def run_panel_backtest(model, panel, cost: float, explain: bool, years: int,
                       label: str, **params) -> dict:
    df, n_rebal = model.backtest(panel, cost=cost, **params)
    strat = metrics(df['strat_net'])
    bench = metrics(df['ret'])
    wr = xs_win_rate(df)
    print_xs_backtest_report(label, strat, bench, n_rebal, wr, explain=explain)
    return {
        'slug': model.slug,
        'label': label,
        'ann_return': strat['ann_return'],
        'sharpe': strat['sharpe'],
        'max_dd': strat['max_dd'],
        'n_trades': n_rebal,
        'strat': strat,
        'hold': bench,
    }
