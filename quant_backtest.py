"""
First quant backtest: z-score mean reversion (synthetic data).

Uses the mean-reversion model from quant.models. For real data, use:
    python cli.py backtest TICKER --model mean-reversion
    python cli.py report TICKER
"""

from pathlib import Path

import numpy as np
import pandas as pd

from quant.metrics import win_rate
from quant.models.mean_reversion import MeanReversionModel
from quant.reporting import print_backtest_report


def simulate_prices(n_days=1000, annual_vol=0.30, annual_drift=0.08,
                    revert=0.0, P0=100.0, seed=1):
    """Daily prices via a random walk in log space.

    revert > 0 injects NEGATIVE return autocorrelation: a positive return today
    makes a negative return tomorrow more likely -> genuine mean reversion the
    strategy can exploit. revert = 0 is a pure random walk (no edge to find).
    """
    rng = np.random.default_rng(seed)
    mu = annual_drift / 252.0
    sig = annual_vol / np.sqrt(252.0)
    eps = rng.normal(0.0, sig, n_days)
    r = np.zeros(n_days)
    for t in range(1, n_days):
        r[t] = mu - revert * r[t - 1] + eps[t]
    price = P0 * np.exp(np.cumsum(r))
    return pd.Series(price, name='price')


def report(label, price):
    model = MeanReversionModel()
    df, n_trades = model.backtest(price)
    from quant.metrics import metrics
    strat = metrics(df['strat_net'])
    hold = metrics(df['ret'])
    print_backtest_report(label, strat, hold, n_trades, win_rate(df))
    return df


if __name__ == '__main__':
    edge_price = simulate_prices(revert=0.35, seed=1)
    df_edge = report('MEAN-REVERTING DATA (a real edge exists)', edge_price)

    noise_price = simulate_prices(revert=0.0, seed=1)
    report('RANDOM-WALK DATA (no edge exists)', noise_price)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    eq_strat = (1 + df_edge['strat_net']).cumprod()
    eq_hold = (1 + df_edge['ret']).cumprod()
    plt.figure(figsize=(9, 4.8))
    plt.plot(eq_hold.values, label='Buy & hold', linewidth=1.6)
    plt.plot(eq_strat.values, label='Mean-reversion strategy', linewidth=1.6)
    plt.title('Equity curves on mean-reverting data (growth of $1)')
    plt.xlabel('Trading day')
    plt.ylabel('Equity (×$1)')
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    out = Path(__file__).parent / 'equity_curves.png'
    plt.savefig(out, dpi=130)
    print(f'\nSaved plot: {out}')
