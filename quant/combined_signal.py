"""
Combined cross-sectional + single-stock decision layer.

Cross-sectional model  -> stock selection / relative strength
Single-stock models    -> timing / confirmation / overextension control

Execution assumption (no look-ahead)
------------------------------------
Signal and weights are fixed at the close of day t.
The position is active starting day t+1.

    portfolio_return[t] = sum_i weights[t-1, i] * returns[t, i]
    turnover[t]         = sum_i abs(weights[t, i] - weights[t-1, i])
    gross_return[t]     = sum_i weights[t-1, i] * returns[t, i]
    net_return[t]       = gross_return[t] - turnover[t] * cost

Transaction costs are charged on day t when weights[t-1] first becomes active
for return[t] (cost aligns with shifted execution).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant.metrics import metrics
from quant.models.cross_sectional import (
    build_weights,
    compute_scores,
    portfolio_returns,
)
from quant.models.mean_reversion import MeanReversionModel
from quant.models.momentum import MomentumModel


@dataclass
class CombinedParams:
    z_overextended: float = 1.5
    z_oversold: float = -1.0
    require_momentum_buy: bool = True
    allow_short_candidates: bool = False
    long_only_mode: bool = True

    def as_dict(self) -> dict:
        return {
            'z_overextended': self.z_overextended,
            'z_oversold': self.z_oversold,
            'require_momentum_buy': self.require_momentum_buy,
            'allow_short_candidates': self.allow_short_candidates,
            'long_only_mode': self.long_only_mode,
        }


def xs_leg_from_weight(w: float) -> str:
    if w > 1e-9:
        return 'LONG'
    if w < -1e-9:
        return 'SHORT'
    return '—'


def combined_decision(xs_leg: str, z: float, mr_sig: str, mom_sig: str,
                      p: CombinedParams) -> tuple[str, str]:
    """Return (final_action, reason) for one ticker."""
    if np.isnan(z):
        return 'WAIT', 'Insufficient data for z-score.'

    mom_ok = (mom_sig == 'BUY') if p.require_momentum_buy else True
    oversold = z < p.z_oversold
    mr_bounce = mr_sig == 'BUY' or oversold

    if xs_leg == 'LONG':
        if not mom_ok:
            return 'WAIT', 'XS long but momentum not confirmed.'
        if mom_sig == 'BUY' and z > p.z_overextended:
            return 'WAIT', 'Relative strength confirmed, but price is stretched. Wait for pullback.'
        if mom_sig == 'BUY' and z <= p.z_overextended:
            return 'BUY', 'XS long + momentum confirmed + not overextended.'
        return 'WAIT', 'XS long leg; conditions for entry not met.'

    if xs_leg == 'SHORT':
        if mr_bounce and mom_sig == 'BUY':
            return 'WATCH', 'Conflict: relatively weak, but individually oversold / possible bounce.'
        if (mom_sig != 'BUY' and mr_sig != 'BUY' and z >= p.z_oversold):
            if p.allow_short_candidates:
                return 'SHORT_CANDIDATE', 'Relative weak + no bullish single-stock confirmation.'
            return 'AVOID', 'Relative underperformer. Avoid for long-only portfolio.'
        return 'AVOID', 'Relative underperformer. Avoid for long-only portfolio.'

    return 'WAIT', 'Not in XS active book (middle of universe).'


def _signal_series(model, df: pd.DataFrame) -> pd.Series:
    """Stateless signal at each bar (in_position=False) for rebalance decisions."""
    out = []
    for _, row in df.iterrows():
        _, action = model.next_action(row, in_position=False)
        if action == 'BUY':
            out.append('BUY')
        elif action == 'SELL':
            out.append('SELL')
        else:
            out.append('WAIT')
    return pd.Series(out, index=df.index)


def precompute_single_stock_signals(
    panel: pd.DataFrame,
    mom_params: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-ticker z-score and MR / momentum signal panels."""
    mr_model, mom_model = MeanReversionModel(), MomentumModel()
    mom_p = {**mom_model.default_params(), **(mom_params or {})}
    z_panel = pd.DataFrame(index=panel.index, columns=panel.columns, dtype=float)
    mr_panel = pd.DataFrame(index=panel.index, columns=panel.columns, dtype=object)
    mom_panel = pd.DataFrame(index=panel.index, columns=panel.columns, dtype=object)

    for col in panel.columns:
        price = panel[col].dropna()
        mr_df = mr_model.compute_indicators(price.rename(col))
        mom_df = mom_model.compute_indicators(price.rename(col), **mom_p)
        z_panel.loc[mr_df.index, col] = mr_df['z']
        mr_panel.loc[mr_df.index, col] = _signal_series(mr_model, mr_df)
        mom_panel.loc[mom_df.index, col] = _signal_series(mom_model, mom_df)

    return z_panel, mr_panel, mom_panel


def apply_combined_to_row(xs_leg: str, z: float, mr_sig: str, mom_sig: str,
                          p: CombinedParams) -> tuple[str, str]:
    action, reason = combined_decision(xs_leg, z, mr_sig, mom_sig, p)
    if p.long_only_mode and action == 'SHORT_CANDIDATE':
        action = 'AVOID'
    return action, reason


def build_combined_snapshot_df(panel: pd.DataFrame, xs_weights: pd.Series,
                               xs_scores: pd.Series,
                               p: CombinedParams,
                               mom_params: dict | None = None,
                               momentum_preset: str = 'mom_126d_skip21') -> pd.DataFrame:
    """Latest combined signal row per ticker."""
    from quant.universe_analysis import snapshot_ticker

    mr_model, mom_model = MeanReversionModel(), MomentumModel()
    mom_p = {**mom_model.default_params(), **(mom_params or {})}
    rows = []
    for ticker in panel.columns:
        snap = snapshot_ticker(
            panel[ticker].dropna().rename(ticker), mr_model, mom_model, mom_params=mom_p,
        )
        leg = xs_leg_from_weight(float(xs_weights.get(ticker, 0.0)))
        action, reason = apply_combined_to_row(
            leg, snap['z'], snap['mr_signal'], snap['mom_signal'], p,
        )
        rows.append({
            **snap,
            'xs_score': float(xs_scores.get(ticker, float('nan'))),
            'xs_leg': leg,
            'momentum_preset': momentum_preset,
            'final_action': action,
            'reason': reason,
        })
    return pd.DataFrame(rows).sort_values('xs_score', ascending=False, na_position='last')


def print_combined_signal_report(df: pd.DataFrame, p: CombinedParams) -> None:
    print('\n=== Combined Signal Report ===')
    if p.long_only_mode:
        print('Long-only mode: short leg is treated as avoid/underweight, not actual short positions.')
    buys = df[df['final_action'] == 'BUY']['ticker'].tolist()
    if buys:
        print(f'Actionable BUY candidates ({len(buys)}): {", ".join(buys)}')
    print()
    print(f'{"Ticker":<7}{"XS":<6}{"XS score":>9}{"Price":>10}{"Z":>7}'
          f'{"MR":>5}{"Mom preset":<12}{"Mom score":>10}{"Mom sig":>8}{"Final":>14}{"  Reason"}')
    print('-' * 120)
    for _, r in df.iterrows():
        reason_short = r['reason'] if len(r['reason']) <= 38 else r['reason'][:35] + '...'
        mom_preset = r.get('momentum_preset', 'mom_126d_skip21')
        print(f'{r["ticker"]:<7}{r["xs_leg"]:<6}{r["xs_score"]:>+8.1%}'
              f'{r["price"]:>10.2f}{r["z"]:>+7.2f}'
              f'{r["mr_signal"]:>5}{mom_preset:<12}{r["momentum"]:>+9.1%}'
              f'{r["mom_signal"]:>8}{r["final_action"]:>14}  {reason_short}')


def _rebalance_days(n: int, rebalance: int) -> set[int]:
    return set(range(0, n, rebalance))


def backtest_combined_long_only(
    panel: pd.DataFrame,
    xs_params: dict,
    combined: CombinedParams,
    cost: float = 0.0005,
) -> tuple[pd.DataFrame, dict]:
    """
    Long-only strategy: equal-weight only names with Final action == BUY each rebalance.
    """
    rets = panel.pct_change(fill_method=None)
    scores = compute_scores(panel, mode=xs_params.get('mode', 'momentum'),
                            lookback=xs_params.get('lookback', 126),
                            skip=xs_params.get('skip', 21),
                            short_window=xs_params.get('short_window', 5))
    xs_w = build_weights(
        panel, scores,
        top_frac=xs_params.get('top_frac', 0.25),
        rebalance=xs_params.get('rebalance', 5),
        market_neutral=True,
    )
    mom_kw = {k: xs_params[k] for k in ('lookback', 'skip') if k in xs_params}
    z_panel, mr_panel, mom_panel = precompute_single_stock_signals(panel, mom_params=mom_kw)

    weights = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
    last_w = pd.Series(0.0, index=panel.columns)
    rebals = _rebalance_days(len(panel), xs_params.get('rebalance', 5))
    holdings_count = []

    for t in range(len(panel)):
        if t in rebals:
            buys = []
            for col in panel.columns:
                leg = xs_leg_from_weight(float(xs_w.iloc[t][col]))
                z = float(z_panel.iloc[t][col]) if not pd.isna(z_panel.iloc[t][col]) else float('nan')
                mr_sig = str(mr_panel.iloc[t][col]) if not pd.isna(mr_panel.iloc[t][col]) else 'WAIT'
                mom_sig = str(mom_panel.iloc[t][col]) if not pd.isna(mom_panel.iloc[t][col]) else 'WAIT'
                action, _ = apply_combined_to_row(leg, z, mr_sig, mom_sig, combined)
                if action == 'BUY':
                    buys.append(col)
            w = pd.Series(0.0, index=panel.columns)
            if buys:
                w[buys] = 1.0 / len(buys)
            last_w = w
            holdings_count.append(len(buys))
        weights.iloc[t] = last_w

    result = portfolio_returns(weights, rets, cost)
    stats = {
        'avg_holdings': float(np.mean(holdings_count)) if holdings_count else 0.0,
        'n_rebalances': len(holdings_count),
    }
    return result, stats


def backtest_xs_long_only(
    panel: pd.DataFrame,
    xs_params: dict,
    cost: float = 0.0005,
) -> pd.DataFrame:
    """Pure XS momentum long leg only — fully invested in top quantile."""
    rets = panel.pct_change(fill_method=None)
    scores = compute_scores(panel, mode=xs_params.get('mode', 'momentum'),
                            lookback=xs_params.get('lookback', 126),
                            skip=xs_params.get('skip', 21),
                            short_window=xs_params.get('short_window', 5))
    weights = build_weights(
        panel, scores,
        top_frac=xs_params.get('top_frac', 0.25),
        rebalance=xs_params.get('rebalance', 5),
        market_neutral=False,
    )
    return portfolio_returns(weights, rets, cost)


def backtest_xs_long_short(
    panel: pd.DataFrame,
    xs_params: dict,
    cost: float = 0.0005,
) -> pd.DataFrame:
    """Pure XS momentum long/short dollar-neutral book."""
    rets = panel.pct_change(fill_method=None)
    scores = compute_scores(panel, mode=xs_params.get('mode', 'momentum'),
                            lookback=xs_params.get('lookback', 126),
                            skip=xs_params.get('skip', 21),
                            short_window=xs_params.get('short_window', 5))
    weights = build_weights(
        panel, scores,
        top_frac=xs_params.get('top_frac', 0.25),
        rebalance=xs_params.get('rebalance', 5),
        market_neutral=True,
    )
    return portfolio_returns(weights, rets, cost)


def _strategy_stats(returns: pd.Series, bench: pd.Series, turnover: pd.Series,
                    n_rebal: int, avg_holdings: float = float('nan')) -> dict:
    m = metrics(returns)
    aligned = pd.concat([returns, bench], axis=1).dropna()
    corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1])) if len(aligned) > 1 else 0.0
    return {
        'ann_return': m['ann_return'],
        'ann_vol': m['ann_vol'],
        'sharpe': m['sharpe'],
        'max_dd': m['max_dd'],
        'turnover': float(turnover.mean()),
        'corr_bench': corr,
        'n_rebalances': n_rebal,
        'avg_holdings': avg_holdings,
    }


def run_strategy_comparison(
    panel: pd.DataFrame,
    xs_params: dict,
    combined: CombinedParams,
    cost: float = 0.0005,
) -> list[dict]:
    """Backtest all four strategies and return summary rows."""
    rets = panel.pct_change(fill_method=None)
    bench = rets.mean(axis=1)

    bench_m = metrics(bench)
    rows = [{
        'label': 'Equal-weight benchmark',
        'ann_return': bench_m['ann_return'],
        'ann_vol': bench_m['ann_vol'],
        'sharpe': bench_m['sharpe'],
        'max_dd': bench_m['max_dd'],
        'turnover': 0.0,
        'corr_bench': 1.0,
        'n_rebalances': 0,
        'avg_holdings': float(panel.shape[1]),
    }]

    xs_lo = backtest_xs_long_only(panel, xs_params, cost)
    rows.append({
        'label': 'XS momentum long-only',
        **_strategy_stats(xs_lo['strat_net'], bench, xs_lo['turnover'],
                          int((xs_lo['turnover'] > 1e-9).sum()),
                          avg_holdings=xs_params.get('top_frac', 0.25) * panel.shape[1]),
    })

    xs_ls = backtest_xs_long_short(panel, xs_params, cost)
    rows.append({
        'label': 'XS momentum long/short',
        **_strategy_stats(xs_ls['strat_net'], bench, xs_ls['turnover'],
                          int((xs_ls['turnover'] > 1e-9).sum())),
    })

    comb, comb_stats = backtest_combined_long_only(panel, xs_params, combined, cost)
    rows.append({
        'label': 'Combined signal long-only',
        **_strategy_stats(comb['strat_net'], bench, comb['turnover'],
                          comb_stats['n_rebalances'], comb_stats['avg_holdings']),
    })

    return rows


def print_strategy_comparison(rows: list[dict]) -> None:
    print('\n=== Strategy Comparison ===')
    print(f'{"Model":<28}{"AnnRet":>8}{"Vol":>8}{"Sharpe":>8}{"MaxDD":>8}'
          f'{"Turnover":>10}{"CorrBench":>11}{"AvgHold":>8}')
    print('-' * 95)
    for r in rows:
        ah = r.get('avg_holdings', float('nan'))
        ah_s = f'{ah:>7.1f}' if not np.isnan(ah) else f'{"—":>7}'
        print(f'{r["label"]:<28}{r["ann_return"]:>+7.1%}{r["ann_vol"]:>8.1%}'
              f'{r["sharpe"]:>8.2f}{r["max_dd"]:>8.1%}'
              f'{r["turnover"]:>10.4f}{r["corr_bench"]:>11.2f}{ah_s}')

    strat_rows = [r for r in rows if r['label'] != 'Equal-weight benchmark']
    if not strat_rows:
        return
    best_ret = max(strat_rows, key=lambda x: x['ann_return'])
    best_sharpe = max(strat_rows, key=lambda x: x['sharpe'])
    best_dd = max(strat_rows, key=lambda x: x['max_dd'])
    lowest_corr = min(strat_rows, key=lambda x: abs(x['corr_bench']))
    print()
    print(f'Best return:          {best_ret["label"]} ({best_ret["ann_return"]:+.1%}/yr)')
    print(f'Best Sharpe:          {best_sharpe["label"]} (Sharpe {best_sharpe["sharpe"]:.2f})')
    print(f'Lowest drawdown:      {best_dd["label"]} (max DD {best_dd["max_dd"]:.1%})')
    print(f'Lowest benchmark corr: {lowest_corr["label"]} (corr {lowest_corr["corr_bench"]:.2f})')
