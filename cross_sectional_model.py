"""
Cross-sectional (multi-stock) long/short portfolio model — synthetic demo.

For real data:
    python cli.py portfolio backtest --signal all --years 5
    python cli.py portfolio ranks --signal momentum
"""

from pathlib import Path

import pandas as pd

from quant.metrics import metrics
from quant.models.cross_sectional import (
    DEFAULT_UNIVERSE,
    CrossSectionalModel,
    simulate_panel,
)
from quant.reporting import explain_backtest


def line(label, m, extra=''):
    print(f"{label:26}{m['ann_return']:>10.1%}{m['ann_vol']:>9.1%}"
          f"{m['sharpe']:>9.2f}{m['max_dd']:>10.1%}{extra:>11}")


def header(title):
    print(f'\n=== {title} ===')
    print(f"{'':26}{'AnnRet':>10}{'Vol':>9}{'Sharpe':>9}{'MaxDD':>10}{'CorrBench':>11}")


if __name__ == '__main__':
    model = CrossSectionalModel()
    px = simulate_panel(DEFAULT_UNIVERSE, n_days=1500, seed=1)

    mom_df, _ = model.backtest(px, mode='momentum')
    rev_df, _ = model.backtest(
        px, mode='reversal', short_window=2, rebalance=2,
    )
    df = pd.DataFrame({
        'bench': mom_df['ret'],
        'mom': mom_df['strat_net'],
        'rev': rev_df['strat_net'],
    }).dropna()
    df['combo'] = 0.5 * df['mom'] + 0.5 * df['rev']

    header('SYNTHETIC PANEL (momentum + reversal embedded)')
    line('Equal-weight universe', metrics(df['bench']))
    line('XS Momentum (L/S)', metrics(df['mom']), f"{df['mom'].corr(df['bench']):>11.2f}")
    line('XS Reversal (L/S)', metrics(df['rev']), f"{df['rev'].corr(df['bench']):>11.2f}")
    line('Combined (50/50)', metrics(df['combo']), f"{df['combo'].corr(df['bench']):>11.2f}")
    print(f"\nCorrelation  momentum vs reversal: {df['mom'].corr(df['rev']):.2f}")
    print(explain_backtest(
        metrics(df['mom']), metrics(df['bench']), 0, float('nan'),
    ))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(9, 4.8))
    for col, lab, lw in [('bench', 'Equal-weight universe', 1.4),
                         ('mom', 'XS Momentum (L/S)', 1.6),
                         ('rev', 'XS Reversal (L/S)', 1.6),
                         ('combo', 'Combined 50/50', 1.9)]:
        plt.plot((1 + df[col]).cumprod().values, label=lab, lw=lw)
    plt.title('Cross-sectional long/short on a 12-stock universe (growth of $1)')
    plt.xlabel('Trading day')
    plt.ylabel('Equity (×$1)')
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    out = Path(__file__).parent / 'portfolio_curves.png'
    plt.savefig(out, dpi=130)
    print(f'\nSaved plot: {out}')
