"""
Trend / volatility-timing overlay on a long-only basket.

What edge this targets
----------------------
This is NOT a benchmark-relative (active-return) strategy. Cross-sectional
momentum on the semis universe matches the equal-weight basket; this module
asks a different question: can we time EXPOSURE to that basket to improve
risk-adjusted return (Sharpe) and cut drawdowns, judged against (a) the
always-invested basket and (b) cash?

Mechanism
---------
At the close of day t we compute an exposure in [0, max_leverage] from a trend
signal (and optionally a volatility target), using only data <= t. The exposure
is applied to the NEXT day's basket return (shift by one), so there is no
look-ahead. The uninvested fraction earns the cash rate. Transaction cost is
charged on changes in exposure (entering/exiting the basket).

    exposure[t]      = trend_gate[t] * vol_scale[t]        # data <= t
    gross_return[t]  = exposure[t-1] * basket_ret[t] + (1 - exposure[t-1]) * cash
    turnover[t]      = |exposure[t] - exposure[t-1]|
    net_return[t]    = gross_return[t] - turnover[t] * cost
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.metrics import metrics
from quant.validation import optimize_full, walk_forward

TREND_MODES = ('sma', 'tsmom')
DEFAULT_TREND_MODE = 'sma'
DEFAULT_TREND_WINDOW = 200
DEFAULT_VOL_WINDOW = 63
DEFAULT_TARGET_VOL = 0.15
DEFAULT_MAX_LEVERAGE = 1.0
DEFAULT_OVERLAY_COST = 0.0005
DEFAULT_CASH_RATE = 0.0

TREND_WINDOW_GRID = [50, 100, 150, 200, 250]

# Verdict thresholds (risk-adjusted, not benchmark-relative).
SHARPE_IMPROVE_MARGIN = 0.10
DD_IMPROVE_FRACTION = 0.15
MIN_OVERLAY_FOLDS = 6

VERDICT_IMPROVES = 'IMPROVES RISK-ADJUSTED'
VERDICT_DEFENSIVE = 'REDUCES DRAWDOWN ONLY — lower return, defensive'
VERDICT_NO_IMPROVE = 'NO IMPROVEMENT — overlay does not beat buy-and-hold'
VERDICT_INSUFFICIENT = 'INSUFFICIENT EVIDENCE — too few OOS folds'


def basket_price_and_returns(panel: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Equal-weight basket daily returns and the implied price index."""
    rets = panel.pct_change(fill_method=None).mean(axis=1)
    price = (1.0 + rets.fillna(0.0)).cumprod()
    return price, rets


def trend_exposure(
    basket_price: pd.Series,
    *,
    mode: str = DEFAULT_TREND_MODE,
    window: int = DEFAULT_TREND_WINDOW,
) -> pd.Series:
    """Binary in {0, 1}: 1 when the basket trend is up. Uses only data <= t.

    'sma'   -> price above its ``window``-day simple moving average.
    'tsmom' -> trailing ``window``-day return is positive.
    Bars without enough history are NaN (treated as flat by the backtest).
    """
    if int(window) <= 1:
        raise ValueError('window must be an integer > 1')
    window = int(window)
    if mode == 'sma':
        sma = basket_price.rolling(window, min_periods=window).mean()
        sig = (basket_price > sma).astype(float)
        return sig.where(sma.notna(), np.nan)
    if mode == 'tsmom':
        trailing = basket_price / basket_price.shift(window) - 1.0
        sig = (trailing > 0).astype(float)
        return sig.where(trailing.notna(), np.nan)
    raise ValueError(f"unknown trend mode: {mode} (choose from {TREND_MODES})")


def vol_target_scale(
    basket_returns: pd.Series,
    *,
    target_vol: float = DEFAULT_TARGET_VOL,
    vol_window: int = DEFAULT_VOL_WINDOW,
    max_leverage: float = DEFAULT_MAX_LEVERAGE,
) -> pd.Series:
    """Scale toward ``target_vol`` using trailing realized vol (data <= t)."""
    if target_vol <= 0:
        raise ValueError('target_vol must be positive')
    if int(vol_window) <= 1:
        raise ValueError('vol_window must be an integer > 1')
    realized = basket_returns.rolling(
        int(vol_window), min_periods=max(10, int(vol_window) // 2)
    ).std() * np.sqrt(252.0)
    scale = (float(target_vol) / realized).clip(upper=float(max_leverage))
    return scale.where(realized.notna() & (realized > 0), np.nan)


def backtest_trend_overlay(
    panel: pd.DataFrame,
    *,
    mode: str = DEFAULT_TREND_MODE,
    window: int = DEFAULT_TREND_WINDOW,
    use_vol_target: bool = False,
    target_vol: float = DEFAULT_TARGET_VOL,
    vol_window: int = DEFAULT_VOL_WINDOW,
    max_leverage: float = DEFAULT_MAX_LEVERAGE,
    cost: float = DEFAULT_OVERLAY_COST,
    cash_rate: float = DEFAULT_CASH_RATE,
) -> pd.DataFrame:
    """Backtest the exposure-timed basket. No look-ahead (exposure shifted)."""
    price, rets = basket_price_and_returns(panel)
    exposure = trend_exposure(price, mode=mode, window=window)
    cap = float(max_leverage) if use_vol_target else 1.0
    if use_vol_target:
        scale = vol_target_scale(
            rets, target_vol=target_vol, vol_window=vol_window, max_leverage=max_leverage,
        )
        exposure = exposure * scale
    exposure = exposure.fillna(0.0).clip(lower=0.0, upper=cap)

    daily_cash = (1.0 + float(cash_rate)) ** (1.0 / 252.0) - 1.0
    active = exposure.shift(1).fillna(0.0)
    gross = active * rets + (1.0 - active) * daily_cash
    turnover = exposure.diff().abs().fillna(0.0)
    cost_paid = turnover * float(cost)
    net = gross - cost_paid
    return pd.DataFrame({
        'overlay_net': net,
        'basket_ret': rets,
        'exposure': exposure,
        'turnover': turnover,
        'cost': cost_paid,
    }).dropna()


def make_trend_overlay_strategy(
    *,
    mode: str = DEFAULT_TREND_MODE,
    use_vol_target: bool = False,
    target_vol: float = DEFAULT_TARGET_VOL,
    vol_window: int = DEFAULT_VOL_WINDOW,
    max_leverage: float = DEFAULT_MAX_LEVERAGE,
    cost: float = DEFAULT_OVERLAY_COST,
    cash_rate: float = DEFAULT_CASH_RATE,
):
    """Adapter for quant.validation.walk_forward; returns overlay net returns."""
    def _strategy(panel: pd.DataFrame, **kw) -> pd.Series:
        return backtest_trend_overlay(
            panel,
            mode=kw.get('mode', mode),
            window=int(kw.get('window', DEFAULT_TREND_WINDOW)),
            use_vol_target=use_vol_target,
            target_vol=target_vol,
            vol_window=vol_window,
            max_leverage=max_leverage,
            cost=cost,
            cash_rate=cash_rate,
        )['overlay_net']
    return _strategy


def overlay_verdict(
    overlay_sharpe: float,
    basket_sharpe: float,
    overlay_max_dd: float,
    basket_max_dd: float,
    folds: int,
    *,
    sharpe_margin: float = SHARPE_IMPROVE_MARGIN,
    dd_fraction: float = DD_IMPROVE_FRACTION,
    min_folds: int = MIN_OVERLAY_FOLDS,
) -> str:
    """Classify the overlay on risk-adjusted grounds (vs always-invested basket)."""
    if folds < min_folds:
        return VERDICT_INSUFFICIENT
    if any(pd.isna(x) for x in (overlay_sharpe, basket_sharpe, overlay_max_dd, basket_max_dd)):
        return VERDICT_NO_IMPROVE
    # max_dd is negative; "less negative" is better. Improvement fraction vs basket.
    dd_improved = (
        basket_max_dd < 0
        and (abs(basket_max_dd) - abs(overlay_max_dd)) / abs(basket_max_dd) >= dd_fraction
    )
    if overlay_sharpe >= basket_sharpe + sharpe_margin and abs(overlay_max_dd) <= abs(basket_max_dd):
        return VERDICT_IMPROVES
    if dd_improved:
        return VERDICT_DEFENSIVE
    return VERDICT_NO_IMPROVE


def _overlay_oos_detail(panel, folds, train, test, warmup, **overlay_kwargs):
    """Replay each fold's selected window to collect OOS exposure/cost + per-fold
    overlay-vs-basket Sharpe. Same fold boundaries as walk_forward (no peek)."""
    rets_full = panel.pct_change(fill_method=None).mean(axis=1)
    pos = train
    expo_chunks, turn_chunks, cost_chunks, fold_stats = [], [], [], []
    for fold in folds:
        te_lo, te_hi = pos, pos + test
        if te_hi > len(panel):
            break
        wlo = max(0, te_lo - warmup)
        df = backtest_trend_overlay(
            panel.iloc[wlo:te_hi], window=int(fold['best_params']['window']), **overlay_kwargs,
        )
        test_idx = panel.index[te_lo:te_hi]
        expo = df['exposure'].reindex(test_idx).dropna()
        expo_chunks.append(expo)
        turn_chunks.append(df['turnover'].reindex(expo.index).fillna(0.0))
        cost_chunks.append(df['cost'].reindex(expo.index).fillna(0.0))
        ov_net = df['overlay_net'].reindex(expo.index).dropna()
        basket = rets_full.reindex(expo.index).dropna()
        fold_stats.append((metrics(ov_net)['sharpe'], metrics(basket)['sharpe']))
        pos += test

    def _cat(chunks):
        return pd.concat(chunks).sort_index() if chunks else pd.Series(dtype=float)

    return _cat(expo_chunks), _cat(turn_chunks), _cat(cost_chunks), fold_stats


def report_trend_overlay_validation(
    panel: pd.DataFrame,
    *,
    train: int = 504,
    test: int = 63,
    warmup: int | None = None,
    mode: str = DEFAULT_TREND_MODE,
    windows: list[int] | None = None,
    use_vol_target: bool = False,
    target_vol: float = DEFAULT_TARGET_VOL,
    vol_window: int = DEFAULT_VOL_WINDOW,
    max_leverage: float = DEFAULT_MAX_LEVERAGE,
    cost: float = DEFAULT_OVERLAY_COST,
    cash_rate: float = DEFAULT_CASH_RATE,
    select: str = 'sharpe',
) -> dict:
    """Walk-forward validate the trend overlay and print a risk-adjusted report.

    The overlay is judged on Sharpe and drawdown vs the ALWAYS-INVESTED equal-
    weight basket (and cash) on the same OOS folds — not on active return, since
    a defensive overlay is expected to give up upside for lower risk.
    """
    windows = list(windows) if windows else list(TREND_WINDOW_GRID)
    if warmup is None:
        warmup = max(windows) + (vol_window if use_vol_target else 0) + 5
    grid = {'window': windows}
    overlay_kwargs = dict(
        mode=mode, use_vol_target=use_vol_target, target_vol=target_vol,
        vol_window=vol_window, max_leverage=max_leverage, cost=cost, cash_rate=cash_rate,
    )
    strategy_fn = make_trend_overlay_strategy(**overlay_kwargs)

    _, full_m = optimize_full(strategy_fn, panel, grid, select=select)
    wf = walk_forward(strategy_fn, panel, grid, train=train, test=test,
                      warmup=warmup, select=select)
    oos = wf['oos_metrics']
    folds = wf['folds']
    oos_index = wf['oos_returns'].index

    basket_oos = panel.pct_change(fill_method=None).mean(axis=1).reindex(oos_index).dropna()
    basket_m = metrics(basket_oos)
    daily_cash = (1.0 + float(cash_rate)) ** (1.0 / 252.0) - 1.0
    cash_ann = (1.0 + daily_cash) ** 252.0 - 1.0

    exposure, turnover, cost_paid, fold_stats = _overlay_oos_detail(
        panel, folds, train, test, warmup, **overlay_kwargs,
    )
    mean_exposure = float(exposure.mean()) if len(exposure) else 0.0
    pct_invested = float((exposure > 1e-9).mean()) if len(exposure) else 0.0
    mean_turnover = float(turnover.mean()) if len(turnover) else 0.0
    mean_cost = float(cost_paid.mean()) if len(cost_paid) else 0.0
    fold_win_rate = (
        float(np.mean([1.0 if ov >= bk else 0.0 for ov, bk in fold_stats]))
        if fold_stats else 0.0
    )
    mean_is = np.mean([f['in_sample_sharpe'] for f in folds]) if folds else 0.0
    window_counts: dict[int, int] = {}
    for f in folds:
        w = int(f['best_params']['window'])
        window_counts[w] = window_counts.get(w, 0) + 1

    verdict = overlay_verdict(
        oos['sharpe'], basket_m['sharpe'], oos['max_dd'], basket_m['max_dd'], len(folds),
    )

    label = f'{mode.upper()} trend overlay' + (' + vol-target' if use_vol_target else '')
    print(f'\n=== {label} (walk-forward, judged vs basket + cash) ===')
    print(f'Universe basket: {panel.shape[1]} names, {len(panel)} rows  |  '
          f'windows tried: {windows}  |  select={select}')
    print(f'Exposure: trend gate in [0,1]'
          + (f' x vol-target {target_vol:.0%} (max {max_leverage:.1f}x)' if use_vol_target else '')
          + f'; cash rate {cash_ann:.1%}; cost {cost * 1e4:.0f} bps/unit turnover')
    print(f"\n{'':30}{'Sharpe':>9}{'AnnRet':>9}{'AnnVol':>9}{'MaxDD':>9}")
    print(f"{'Naive full-history optimize':30}{full_m['sharpe']:>9.2f}"
          f"{full_m['ann_return']:>9.1%}{full_m['ann_vol']:>9.1%}{full_m['max_dd']:>9.1%}   <- overfit trap")
    print(f"{'Overlay, in-sample (avg)':30}{mean_is:>9.2f}{'':>9}{'':>9}{'':>9}   <- optimistic")
    print(f"{'Overlay, OUT-OF-SAMPLE':30}{oos['sharpe']:>9.2f}"
          f"{oos['ann_return']:>9.1%}{oos['ann_vol']:>9.1%}{oos['max_dd']:>9.1%}   <- the honest number")
    print(f"{'Always-invested basket OOS':30}{basket_m['sharpe']:>9.2f}"
          f"{basket_m['ann_return']:>9.1%}{basket_m['ann_vol']:>9.1%}{basket_m['max_dd']:>9.1%}   <- buy & hold")
    print(f"{'Cash':30}{0.0:>9.2f}{cash_ann:>9.1%}{0.0:>9.1%}{0.0:>9.1%}")

    sharpe_delta = oos['sharpe'] - basket_m['sharpe']
    dd_delta = abs(basket_m['max_dd']) - abs(oos['max_dd'])
    print(f'\nOverfitting tax (naive - OOS Sharpe): {full_m["sharpe"] - oos["sharpe"]:.2f}')
    print(f'Sharpe vs basket: {sharpe_delta:+.2f}   '
          f'MaxDD reduction vs basket: {dd_delta:+.1%} (abs)')
    print(f'Time invested: {pct_invested:.0%} of OOS days  '
          f'(mean exposure {mean_exposure:.2f})')
    print(f'OOS mean per-day turnover: {mean_turnover:.4f}   '
          f'mean daily cost drag: {mean_cost:.5f}')
    print(f'Folds: {len(folds)}  |  folds overlay Sharpe >= basket: {fold_win_rate:.0%}')
    print(f'Window selected per fold: '
          + ', '.join(f'{w}d x{c}' for w, c in sorted(window_counts.items())))
    print(f'Verdict: {verdict}')
    if verdict == VERDICT_DEFENSIVE:
        print('  -> Lower return than buy-and-hold but materially smaller drawdowns; '
              'useful only if your mandate values capital preservation.')
    elif verdict == VERDICT_NO_IMPROVE:
        print('  -> After costs, timing the basket did not beat simply holding it.')

    wf.update({
        'basket_oos_metrics': basket_m,
        'sharpe_vs_basket': sharpe_delta,
        'maxdd_reduction': dd_delta,
        'mean_exposure': mean_exposure,
        'pct_invested': pct_invested,
        'mean_turnover': mean_turnover,
        'mean_cost_drag': mean_cost,
        'fold_win_rate': fold_win_rate,
        'verdict': verdict,
        'window_counts': window_counts,
    })
    return wf
