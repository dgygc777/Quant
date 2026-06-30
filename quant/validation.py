"""
Walk-forward validation.

The problem this solves: if you pick a strategy's parameters by looking at the
WHOLE history and keeping whatever scored best, you are fitting to the past. The
backtest will look great and mean nothing -- that's overfitting.

Walk-forward fixes it by never letting the strategy see its own test data:
  1. Split time into consecutive folds.
  2. On each TRAIN window, search the parameter grid and keep the best params
     (this is "in-sample" -- optimistic, allowed to peek).
  3. Apply those FIXED params to the next TEST window (out-of-sample -- honest).
  4. Roll forward, stitch all the test-window returns into one equity curve.

The stitched out-of-sample curve is the honest estimate. The GAP between the
naive full-history Sharpe and the walk-forward Sharpe is your overfitting tax.

strategy_fn must take (price_slice, **params) and return a pd.Series of per-period
strategy returns. For your repo:
    from quant.models.mean_reversion import MeanReversionModel
    strat = lambda p, **kw: MeanReversionModel().backtest(p, **kw)[0]['strat_net']
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from quant.metrics import metrics


def iter_param_grid(grid: dict):
    """Yield every combination of parameters from a {name: [values]} grid."""
    keys = list(grid)
    for combo in product(*(grid[k] for k in keys)):
        yield dict(zip(keys, combo))


def _equal_weight_benchmark(price: pd.DataFrame | pd.Series) -> pd.Series:
    """Equal-weight per-day return of whatever names are in ``price``.

    Computed only from the supplied slice, so when called on a train window it
    introduces no look-ahead — the benchmark sees exactly the same dates/names
    the strategy saw in-sample.
    """
    if isinstance(price, pd.Series):
        return price.pct_change(fill_method=None)
    return price.pct_change(fill_method=None).mean(axis=1)


def active_information_ratio(returns: pd.Series, benchmark: pd.Series) -> float:
    """Annualized IR of (strategy - benchmark) active returns, aligned by date."""
    aligned = pd.concat([
        pd.Series(returns).rename('s'),
        pd.Series(benchmark).rename('b'),
    ], axis=1).dropna()
    if len(aligned) < 2:
        return -np.inf
    active = aligned['s'] - aligned['b']
    std = float(active.std())
    mean = float(active.mean())
    if std <= 1e-15:
        if mean > 0.0:
            return np.inf
        if mean < 0.0:
            return -np.inf
        return 0.0
    return mean / std * np.sqrt(252.0)


def selection_objective(returns: pd.Series, price_slice, select: str = 'sharpe') -> float:
    """Score a candidate's in-sample returns under the chosen selection metric.

    ``select='active_ir'`` ranks by benchmark-relative information ratio using a
    benchmark derived from the same in-sample slice (no look-ahead). Any other
    value falls back to the matching key from ``metrics()`` (e.g. 'sharpe').
    """
    if select == 'active_ir':
        benchmark = _equal_weight_benchmark(price_slice)
        return active_information_ratio(returns, benchmark)
    return metrics(returns)[select]


def optimize_full(strategy_fn, price, param_grid, select='sharpe'):
    """NAIVE baseline: optimize over the ENTIRE history (this is the overfit trap)."""
    best_params, best_metrics, best_score = None, None, -np.inf
    for params in iter_param_grid(param_grid):
        r = strategy_fn(price, **params)
        score = selection_objective(r, price, select)
        if score > best_score:
            best_score, best_params, best_metrics = score, params, metrics(r)
    return best_params, best_metrics


def walk_forward(strategy_fn, price, param_grid, train=252, test=63,
                 warmup=40, select='sharpe', objective_fn=None):
    """Rolling walk-forward. Returns folds, stitched OOS returns, and OOS metrics.

    ``select`` controls the in-sample selection objective. The default 'sharpe'
    preserves historical behavior; 'active_ir' selects parameters by
    benchmark-relative information ratio computed only on the train window.

    ``objective_fn(returns, price, params) -> float`` optionally overrides the
    scalar selection score (e.g. active IR minus a turnover penalty). It must
    use only in-sample data; it is evaluated on the train window exclusively.
    """
    n = len(price)
    fold_records, oos_chunks = [], []
    pos = train
    while pos + test <= n:
        tr_lo, tr_hi = pos - train, pos              # train window  [tr_lo, tr_hi)
        te_lo, te_hi = pos, pos + test               # test  window  [te_lo, te_hi)

        # 1) optimize ON TRAIN ONLY (in-sample, allowed to peek)
        train_slice = price.iloc[tr_lo:tr_hi]
        best_params, best_is, best_score = None, None, -np.inf
        for params in iter_param_grid(param_grid):
            r_train = strategy_fn(train_slice, **params)
            if objective_fn is not None:
                score = objective_fn(returns=r_train, price=train_slice, params=params)
            else:
                score = selection_objective(r_train, train_slice, select)
            if score > best_score:
                best_score, best_params, best_is = score, params, metrics(r_train)

        # 2) apply FIXED best params to TEST (out-of-sample, never peeked).
        #    Feed `warmup` extra prior bars so rolling indicators are valid at
        #    the test-window start, then keep ONLY returns inside the test window.
        wlo = max(0, te_lo - warmup)
        r_full = strategy_fn(price.iloc[wlo:te_hi], **best_params)
        r_test = r_full.reindex(price.index[te_lo:te_hi]).dropna()
        oos_chunks.append(r_test)

        fold_records.append({
            'train_end': price.index[tr_hi - 1],
            'test_end': price.index[te_hi - 1],
            'best_params': best_params,
            'in_sample_sharpe': best_is['sharpe'],
            'in_sample_score': best_score,
            'select': select,
            'oos_sharpe': metrics(r_test)['sharpe'],
        })
        pos += test                                   # non-overlapping test windows

    oos_returns = pd.concat(oos_chunks).sort_index() if oos_chunks else pd.Series(dtype=float)
    return {
        'oos_returns': oos_returns,
        'oos_metrics': metrics(oos_returns),
        'folds': fold_records,
    }


def report_validation(name, strategy_fn, price, param_grid, **wf_kwargs):
    """Print the overfit baseline vs the honest walk-forward result."""
    _, full_m = optimize_full(strategy_fn, price, param_grid)
    wf = walk_forward(strategy_fn, price, param_grid, **wf_kwargs)
    oos = wf['oos_metrics']
    mean_is = np.mean([f['in_sample_sharpe'] for f in wf['folds']]) if wf['folds'] else 0.0

    print(f"\n=== {name} ===")
    print(f"{'':34}{'Sharpe':>9}{'AnnRet':>9}{'MaxDD':>9}")
    print(f"{'Naive full-history optimize':34}{full_m['sharpe']:>9.2f}"
          f"{full_m['ann_return']:>9.1%}{full_m['max_dd']:>9.1%}   <- the overfit trap")
    print(f"{'Walk-forward, in-sample (avg)':34}{mean_is:>9.2f}"
          f"{'':>9}{'':>9}   <- optimistic")
    print(f"{'Walk-forward, OUT-OF-SAMPLE':34}{oos['sharpe']:>9.2f}"
          f"{oos['ann_return']:>9.1%}{oos['max_dd']:>9.1%}   <- the honest number")
    gap = full_m['sharpe'] - oos['sharpe']
    print(f"\nOverfitting tax (naive - OOS Sharpe): {gap:.2f}")
    print(f"Folds: {len(wf['folds'])}")
    return wf


# --------------------------- synthetic demo ---------------------------
def _simulate(n=1500, revert=0.0, annual_vol=0.30, seed=1):
    rng = np.random.default_rng(seed)
    sig = annual_vol / np.sqrt(252.0)
    eps = rng.normal(0.0, sig, n)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = -revert * r[t - 1] + eps[t]      # revert>0 => mean reversion (a real edge)
    return pd.Series(100.0 * np.exp(np.cumsum(r)),
                     index=pd.bdate_range('2018-01-01', periods=n), name='price')


if __name__ == '__main__':
    from quant.models.mean_reversion import MeanReversionModel
    strat = lambda p, **kw: MeanReversionModel().backtest(p, **kw)[0]['strat_net']
    grid = {'window': [10, 20, 30, 50], 'entry_z': [-0.5, -1.0, -1.5, -2.0]}

    edge = _simulate(revert=0.35, seed=1)      # real mean-reversion present
    noise = _simulate(revert=0.0, seed=1)      # pure random walk, no edge

    report_validation('MEAN-REVERTING DATA (a real edge exists)', strat, edge, grid)
    report_validation('RANDOM-WALK DATA (no edge -- watch the trap)', strat, noise, grid)
