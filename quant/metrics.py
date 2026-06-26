from __future__ import annotations

import numpy as np
import pandas as pd


def metrics(returns: pd.Series) -> dict:
    returns = returns.dropna()
    if len(returns) == 0 or returns.std() == 0:
        return dict(ann_return=0.0, ann_vol=0.0, sharpe=0.0, max_dd=0.0)
    ann_return = (1 + returns).prod() ** (252.0 / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(252.0)
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252.0)
    equity = (1 + returns).cumprod()
    max_dd = (equity / equity.cummax() - 1).min()
    return dict(ann_return=ann_return, ann_vol=ann_vol, sharpe=sharpe, max_dd=max_dd)


def win_rate(df: pd.DataFrame) -> float:
    """Fraction of invested days with positive strategy return."""
    if 'weight' in df.columns:
        invested = df['weight'].shift(1).abs() > 1e-9
    else:
        invested = df['pos'].shift(1) == 1
    in_market = df['strat_net'][invested]
    return float((in_market > 0).mean()) if len(in_market) else float('nan')
