"""
Educational momentum backtest on synthetic data.

Uses the shared MomentumModel from quant.models. For real data:
    python cli.py backtest MU --model momentum
    python cli.py report MU
"""

from pathlib import Path

import numpy as np
import pandas as pd

from quant.metrics import metrics
from quant.models.momentum import MomentumModel
from quant.reporting import print_backtest_report


def simulate_prices(n_days=2520, annual_vol=0.20, P0=100.0, seed=1, trend=False):
    """trend=True -> persistent drift momentum can ride; trend=False -> random walk."""
    rng = np.random.default_rng(seed)
    sig = annual_vol / np.sqrt(252.0)
    eps = rng.normal(0.0, sig, n_days)
    if trend:
        rho = 0.98
        dn = rng.normal(0.0, sig * 0.12, n_days)
        drift = np.zeros(n_days)
        for t in range(1, n_days):
            drift[t] = rho * drift[t - 1] + dn[t]
        r = drift + eps
    else:
        r = eps
    return pd.Series(P0 * np.exp(np.cumsum(r)), name='price')


def report(label, price, **params):
    model = MomentumModel()
    df, n_trades = model.backtest(price, **params)
    strat = metrics(df['strat_net'])
    hold = metrics(df['ret'])
    from quant.metrics import win_rate
    print_backtest_report(label, strat, hold, n_trades, win_rate(df))
    return df


def row(label, m, n_trades=None):
    t = '' if n_trades is None else f'{n_trades:>9}'
    print(f"{label:24}{m['ann_return']:>10.1%}{m['ann_vol']:>9.1%}"
          f"{m['sharpe']:>9.2f}{m['max_dd']:>10.1%}{t}")


def header(title):
    print(f'\n=== {title} ===')
    print(f"{'':24}{'AnnRet':>10}{'Vol':>9}{'Sharpe':>9}{'MaxDD':>10}{'Trades':>9}")


if __name__ == '__main__':
    trend_px = simulate_prices(trend=True, seed=1)
    rand_px = simulate_prices(trend=False, seed=1)
    model = MomentumModel()

    header('TRENDING DATA (momentum should work)')
    row('Buy & hold', metrics(trend_px.pct_change()))
    plain_df, plain_trades = model.backtest(
        trend_px, vol_scale=False, long_only=False,
    )
    row('Momentum (plain L/S)', metrics(plain_df['strat_net']), plain_trades)
    vt_df, vt_trades = model.backtest(trend_px, long_only=False)
    row('Momentum (vol-targeted)', metrics(vt_df['strat_net']), vt_trades)

    header('RANDOM-WALK DATA (no trend to capture)')
    row('Buy & hold', metrics(rand_px.pct_change()))
    vt_r_df, vt_r_trades = model.backtest(rand_px, long_only=False)
    row('Momentum (vol-targeted)', metrics(vt_r_df['strat_net']), vt_r_trades)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    e_hold = (1 + trend_px.pct_change().dropna()).cumprod()
    e_plain = (1 + plain_df['strat_net']).cumprod()
    e_vt = (1 + vt_df['strat_net']).cumprod()
    plt.figure(figsize=(9, 4.8))
    plt.plot(e_hold.values, label='Buy & hold', lw=1.5)
    plt.plot(e_plain.values, label='Momentum (plain L/S)', lw=1.5)
    plt.plot(e_vt.values, label='Momentum (vol-targeted)', lw=1.8)
    plt.title('Momentum on trending data (growth of $1)')
    plt.xlabel('Trading day')
    plt.ylabel('Equity (×$1)')
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    out = Path(__file__).parent / 'momentum_curves.png'
    plt.savefig(out, dpi=130)
    print(f'\nSaved plot: {out}')
