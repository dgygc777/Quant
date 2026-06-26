from __future__ import annotations

import numpy as np
import pandas as pd

from quant.models.panel_base import PanelModel

DEFAULT_UNIVERSE = [
    'MU', 'NVDA', 'AMD', 'AVGO', 'QCOM', 'TSM',
    'AMAT', 'LRCX', 'KLAC', 'INTC', 'ASML', 'MRVL',
]


def simulate_panel(tickers, n_days=1500, seed=1, annual_idvol=0.25,
                   mkt_vol=0.15, mom_persist=0.97, rev_strength=0.60):
    """Synthetic price panel with cross-sectional momentum and reversal."""
    rng = np.random.default_rng(seed)
    idsig = annual_idvol / np.sqrt(252.0)
    mkt = rng.normal(0.05 / 252, mkt_vol / np.sqrt(252.0), n_days)
    cols = {}
    for tk in tickers:
        beta = rng.uniform(0.8, 1.2)
        dn = rng.normal(0.0, idsig * 0.07, n_days)
        drift = np.zeros(n_days)
        for t in range(1, n_days):
            drift[t] = mom_persist * drift[t - 1] + dn[t]
        e = rng.normal(0.0, idsig, n_days)
        idio = drift + e
        idio[1:] -= rev_strength * e[:-1]
        r = beta * mkt + idio
        cols[tk] = 100.0 * np.exp(np.cumsum(r))
    return pd.DataFrame(cols)


def compute_scores(prices: pd.DataFrame, mode: str = 'momentum',
                   lookback: int = 126, skip: int = 21, short_window: int = 5) -> pd.DataFrame:
    if mode == 'momentum':
        return prices.shift(skip) / prices.shift(skip + lookback) - 1.0
    if mode == 'reversal':
        return -(prices / prices.shift(short_window) - 1.0)
    raise ValueError(f'unknown mode: {mode}')


def build_weights(prices: pd.DataFrame, scores: pd.DataFrame, top_frac: float = 0.33,
                  rebalance: int = 5, market_neutral: bool = True) -> pd.DataFrame:
    """Daily target weights with periodic rebalance and hold-between logic."""
    n = prices.shape[1]
    k = max(1, int(round(top_frac * n)))
    rebal_days = set(range(0, len(prices), rebalance))
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    last_w = pd.Series(0.0, index=prices.columns)
    for t in range(len(prices)):
        if t in rebal_days:
            s = scores.iloc[t].dropna()
            if len(s) >= 2 * k:
                ranked = s.sort_values()
                w = pd.Series(0.0, index=prices.columns)
                w[ranked.index[-k:]] = 1.0 / k
                if market_neutral:
                    w[ranked.index[:k]] = -1.0 / k
                last_w = w
        weights.iloc[t] = last_w
    return weights


def backtest_xs(prices: pd.DataFrame, mode: str = 'momentum', top_frac: float = 0.33,
                rebalance: int = 5, cost: float = 0.0005, market_neutral: bool = True,
                **sig_kw) -> pd.DataFrame:
    rets = prices.pct_change()
    scores = compute_scores(prices, mode=mode, **sig_kw)
    weights = build_weights(prices, scores, top_frac, rebalance, market_neutral)

    port_gross = (weights.shift(1) * rets).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    port_net = port_gross - turnover * cost
    bench = rets.mean(axis=1)
    n_rebalances = int((weights.diff().abs().sum(axis=1) > 1e-9).sum())
    return pd.DataFrame({
        'strat_net': port_net,
        'ret': bench,
        'weights': [weights.iloc[i] for i in range(len(weights))],
    }).dropna(), n_rebalances


class CrossSectionalModel(PanelModel):
    """Rank a stock universe; long top quantile, short bottom (dollar-neutral)."""

    slug = 'cross-sectional'
    name = 'Cross-Sectional L/S'
    description = (
        'Long/short spread across many names: rank by momentum or short-term reversal, '
        'equal-weight top vs bottom quantiles.'
    )

    def default_params(self) -> dict:
        return {
            'mode': 'momentum',
            'lookback': 126,
            'skip': 21,
            'short_window': 5,
            'top_frac': 0.33,
            'rebalance': 5,
            'market_neutral': True,
        }

    def _sig_params(self, params: dict) -> dict:
        mode = params.get('mode', 'momentum')
        kw = {}
        if mode == 'momentum':
            kw['lookback'] = params.get('lookback', 126)
            kw['skip'] = params.get('skip', 21)
        else:
            kw['short_window'] = params.get('short_window', 5)
        return kw

    def backtest(self, panel: pd.DataFrame, cost: float = 0.0005,
                 **params) -> tuple[pd.DataFrame, int]:
        p = {**self.default_params(), **params}
        df, n_rebal = backtest_xs(
            panel,
            mode=p['mode'],
            top_frac=p['top_frac'],
            rebalance=p['rebalance'],
            cost=cost,
            market_neutral=p['market_neutral'],
            **self._sig_params(p),
        )
        return df.drop(columns=['weights'], errors='ignore'), n_rebal

    def backtest_combo(self, panel: pd.DataFrame, cost: float = 0.0005,
                       **params) -> pd.DataFrame:
        """50/50 blend of momentum and reversal legs."""
        mom_df, _ = backtest_xs(panel, mode='momentum', cost=cost, **params)
        rev_df, _ = backtest_xs(
            panel, mode='reversal', cost=cost,
            short_window=params.get('short_window', 5),
            top_frac=params.get('top_frac', 0.33),
            rebalance=params.get('rebalance', 5),
            market_neutral=params.get('market_neutral', True),
        )
        aligned = pd.DataFrame({
            'strat_net': 0.5 * mom_df['strat_net'] + 0.5 * rev_df['strat_net'],
            'ret': mom_df['ret'],
        }).dropna()
        return aligned

    def current_weights(self, panel: pd.DataFrame, **params) -> pd.Series:
        p = {**self.default_params(), **params}
        scores = compute_scores(panel, mode=p['mode'], **self._sig_params(p))
        n = panel.shape[1]
        k = max(1, int(round(p['top_frac'] * n)))
        s = scores.iloc[-1].dropna()
        if len(s) < 2 * k:
            raise ValueError(f'Need at least {2 * k} scored names; got {len(s)}.')
        ranked = s.sort_values()
        w = pd.Series(0.0, index=panel.columns)
        w[ranked.index[-k:]] = 1.0 / k
        if p['market_neutral']:
            w[ranked.index[:k]] = -1.0 / k
        return w[w.abs() > 1e-9]

    def current_ranks(self, panel: pd.DataFrame, **params) -> pd.Series:
        """Latest cross-sectional scores (higher = more long-tilted)."""
        p = {**self.default_params(), **params}
        return compute_scores(panel, mode=p['mode'], **self._sig_params(p)).iloc[-1].dropna()

    def explain_math(self, **params) -> str:
        p = {**self.default_params(), **params}
        mode = p['mode']
        if mode == 'momentum':
            signal = (
                f'momentum score = price[t-{p["skip"]}] / price[t-{p["skip"]}-{p["lookback"]}] - 1'
            )
            signal_desc = 'Long recent WINNERS, short recent LOSERS.'
        else:
            signal = f'reversal score = -(price / price[{p["short_window"]}d ago] - 1)'
            signal_desc = 'Long recent short-term LOSERS, short WINNERS.'

        return f"""
How the math works (cross-sectional long/short)
-----------------------------------------------
Universe: rank ALL stocks in the panel each rebalance ({p["rebalance"]} trading days).

Signal ({mode}):
  {signal}
  → {signal_desc}

Portfolio construction:
  1. Sort stocks by score; take top {p["top_frac"]:.0%} → equal-weight LONG leg.
  2. Bottom {p["top_frac"]:.0%} → equal-weight SHORT leg (dollar-neutral book).
  3. Hold weights until next rebalance; execute on NEXT day's returns.

Why a spread: momentum and reversal earn as cross-sectional factors across
many names — not as a single-stock directional bet.

Benchmark: equal-weight return of the full universe.
"""

    def format_ranks(self, weights: pd.Series, scores: pd.Series,
                     universe: list[str], **params) -> str:
        p = {**self.default_params(), **params}
        longs = weights[weights > 0].sort_values(ascending=False)
        shorts = weights[weights < 0].sort_values()
        lines = [
            f'=== Cross-Sectional Book ({p["mode"]}) ===',
            f'Universe ({len(universe)}): {", ".join(universe)}',
            f'Rebalance every: {p["rebalance"]} days  |  Top/bottom frac: {p["top_frac"]:.0%}',
            '',
            'LONG leg:',
        ]
        for tk, wt in longs.items():
            sc = scores.get(tk, float('nan'))
            lines.append(f'  {tk:<6} weight {wt:+.2%}   score {sc:+.1%}')
        lines.append('SHORT leg:')
        for tk, wt in shorts.items():
            sc = scores.get(tk, float('nan'))
            lines.append(f'  {tk:<6} weight {wt:+.2%}   score {sc:+.1%}')
        return '\n'.join(lines)
