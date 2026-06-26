from __future__ import annotations

import pandas as pd

from quant.metrics import metrics, win_rate
from quant.models.mean_reversion import MeanReversionModel
from quant.models.momentum import MomentumModel


def _recommendation(model, row: pd.Series, in_position: bool = False) -> str:
    _, action = model.next_action(row, in_position)
    if action == 'BUY':
        return 'BUY'
    if action == 'SELL':
        return 'SELL'
    if in_position:
        return 'HOLD'
    return 'WAIT'


def snapshot_ticker(price: pd.Series, mr: MeanReversionModel, mom: MomentumModel) -> dict:
    """Latest single-stock mean-reversion and momentum readings."""
    ticker = str(price.name or price.index.name or 'TICKER')
    mr_p = mr.default_params()
    mom_p = mom.default_params()

    mr_df = mr.compute_indicators(price, **mr_p).dropna()
    mom_df = mom.compute_indicators(price, **mom_p).dropna()
    if mr_df.empty or mom_df.empty:
        raise ValueError(f'Insufficient history for {ticker}')

    mr_row = mr_df.iloc[-1]
    mom_row = mom_df.iloc[-1]
    return {
        'ticker': ticker,
        'price': float(mr_row['price']),
        'z': float(mr_row['z']),
        'mr_signal': _recommendation(mr, mr_row),
        'momentum': float(mom_row['momentum']),
        'mom_weight': float(mom_row['weight']),
        'mom_signal': _recommendation(mom, mom_row),
    }


def backtest_ticker(price: pd.Series, cost: float,
                    mr: MeanReversionModel, mom: MomentumModel) -> dict:
    """Historical single-stock backtest metrics for one name."""
    ticker = str(price.name or 'TICKER')
    mr_df, mr_trades = mr.backtest(price, cost=cost)
    mom_df, mom_trades = mom.backtest(price, cost=cost)
    mr_m = metrics(mr_df['strat_net'])
    mom_m = metrics(mom_df['strat_net'])
    hold_m = metrics(mr_df['ret'])
    return {
        'ticker': ticker,
        'hold_return': hold_m['ann_return'],
        'hold_sharpe': hold_m['sharpe'],
        'mr_return': mr_m['ann_return'],
        'mr_sharpe': mr_m['sharpe'],
        'mr_trades': mr_trades,
        'mr_win_rate': win_rate(mr_df),
        'mom_return': mom_m['ann_return'],
        'mom_sharpe': mom_m['sharpe'],
        'mom_trades': mom_trades,
        'mom_win_rate': win_rate(mom_df),
    }


def _xs_leg(ticker: str, weights: pd.Series) -> str:
    w = weights.get(ticker, 0.0)
    if w > 1e-9:
        return 'LONG'
    if w < -1e-9:
        return 'SHORT'
    return '—'


def analyze_universe_snapshots(panel: pd.DataFrame,
                               weights: pd.Series,
                               xs_scores: pd.Series) -> pd.DataFrame:
    mr, mom = MeanReversionModel(), MomentumModel()
    rows = []
    for ticker in panel.columns:
        snap = snapshot_ticker(panel[ticker].dropna().rename(ticker), mr, mom)
        snap['xs_score'] = float(xs_scores.get(ticker, float('nan')))
        snap['xs_leg'] = _xs_leg(ticker, weights)
        rows.append(snap)
    df = pd.DataFrame(rows)
    return df.sort_values('xs_score', ascending=False, na_position='last')


def analyze_universe_backtests(panel: pd.DataFrame, cost: float,
                               weights: pd.Series,
                               xs_scores: pd.Series) -> pd.DataFrame:
    mr, mom = MeanReversionModel(), MomentumModel()
    rows = []
    for ticker in panel.columns:
        row = backtest_ticker(panel[ticker].dropna().rename(ticker), cost, mr, mom)
        row['xs_score'] = float(xs_scores.get(ticker, float('nan')))
        row['xs_leg'] = _xs_leg(ticker, weights)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.sort_values('xs_score', ascending=False, na_position='last')


def print_snapshot_table(df: pd.DataFrame) -> None:
    print('\n=== Per-stock signals (mean-reversion & momentum) ===')
    print(f'{"Ticker":<7}{"XS leg":<7}{"XS score":>10}{"Price":>10}{"Z":>7}'
          f'{"MR sig":>8}{"Mom":>10}{"Mom sig":>9}')
    print('-' * 76)
    for _, r in df.iterrows():
        print(f'{r["ticker"]:<7}{r["xs_leg"]:<7}{r["xs_score"]:>+9.1%}'
              f'{r["price"]:>10.2f}{r["z"]:>+7.2f}'
              f'{r["mr_signal"]:>8}{r["momentum"]:>+9.1%}{r["mom_signal"]:>9}')


def print_backtest_table(df: pd.DataFrame, years: int) -> None:
    print(f'\n=== Per-stock backtests ({years}y) — vs buy & hold ===')
    print(f'{"Ticker":<7}{"XS leg":<7}{"Hold":>8}{"MR ret":>8}{"MR Sh":>7}'
          f'{"Mom ret":>8}{"Mom Sh":>7}{"MR tr":>6}{"Mom tr":>6}')
    print('-' * 72)
    for _, r in df.iterrows():
        print(f'{r["ticker"]:<7}{r["xs_leg"]:<7}{r["hold_return"]:>+7.1%}'
              f'{r["mr_return"]:>+7.1%}{r["mr_sharpe"]:>7.2f}'
              f'{r["mom_return"]:>+7.1%}{r["mom_sharpe"]:>7.2f}'
              f'{int(r["mr_trades"]):>6}{int(r["mom_trades"]):>6}')
