from __future__ import annotations

import numpy as np
import pandas as pd

from quant.data_quality import EXTREME_DAILY_RETURN, coverage_by_ticker
from quant.params import validate_xs_params
from quant.risk_model import WEIGHTING_METHODS, size_long_leg

from quant.models.panel_base import PanelModel
from quant.universes import (
    DEFAULT_PEER_GROUP,
    DEFAULT_PRESET,
    DEFAULT_UNIVERSE,
    universe_selection_note,
)

# Re-export for backward compatibility.
DEFAULT_TOP_FRAC = 0.25
DEFAULT_RISK_LOOKBACK = 252
DEFAULT_XS_COST = 0.0005


def _normal_weighting_name(weighting: str) -> str:
    key = str(weighting).lower().replace('-', '_')
    if key not in WEIGHTING_METHODS:
        raise ValueError(
            "weighting must be one of: 'equal', 'inverse_vol', 'risk_parity', 'min_variance'."
        )
    return key


def _equal_long_weights(selected: pd.Index | list[str]) -> pd.Series:
    selected = list(selected)
    if not selected:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(selected), index=selected)


def _sized_long_weights(
    prices: pd.DataFrame,
    t: int,
    selected: pd.Index | list[str],
    weighting: str,
    risk_lookback: int = DEFAULT_RISK_LOOKBACK,
    risk_shrink: float = 0.2,
) -> pd.Series:
    """Size selected longs using only returns available through rebalance t."""
    selected = list(selected)
    if weighting == 'equal':
        return _equal_long_weights(selected)
    history = prices.iloc[:t + 1][selected]
    trailing = (
        history
        .pct_change(fill_method=None)
        .tail(risk_lookback)
        .dropna(how='all')
        .dropna()
    )
    min_rows = max(20, 5 * len(selected))
    if len(trailing) < min_rows:
        return _equal_long_weights(selected)
    try:
        weights = size_long_leg(trailing, method=weighting, shrink=risk_shrink).reindex(selected)
    except (AssertionError, ValueError, TypeError, np.linalg.LinAlgError, FloatingPointError):
        return _equal_long_weights(selected)
    if weights.isna().any() or float(weights.sum()) <= 0:
        return _equal_long_weights(selected)
    return weights / float(weights.sum())


def assess_panel_quality(prices: pd.DataFrame) -> dict:
    """Report stale dates, coverage, and suspicious one-day moves."""
    rets = prices.pct_change(fill_method=None)
    latest = prices.index[-1] if len(prices.index) else None
    excluded: list[tuple[str, str]] = []
    coverage = coverage_by_ticker(prices).to_dict()
    for col in prices.columns:
        s = prices[col].dropna()
        if s.empty:
            excluded.append((col, 'no data'))
            continue
        if s.index[-1] < latest:
            excluded.append((col, f'stale last bar {s.index[-1].date()}'))
        col_rets = rets[col].dropna()
        if (col_rets.abs() > EXTREME_DAILY_RETURN).any():
            excluded.append((col, 'extreme daily return (>35%)'))
    min_cov = min(coverage.values()) if coverage else 0.0
    return {
        'latest_date': latest,
        'coverage': coverage,
        'min_coverage': min_cov,
        'excluded': excluded,
    }


def print_panel_quality(report: dict) -> None:
    if report['excluded']:
        print('Panel data-quality notes:')
        for ticker, reason in report['excluded']:
            print(f'  {ticker}: {reason}')
    print(f'Latest panel date: {report["latest_date"]}')
    print(f'Min ticker coverage: {report["min_coverage"]:.1%}')


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


SCORE_MODES = (
    'raw_momentum',
    'risk_adjusted_momentum',
    'multi_horizon_composite',
    'relative_momentum',
    'residual_momentum',
)
DEFAULT_SCORE_MODE = 'raw_momentum'
DEFAULT_BETA_WINDOW = 126


def _raw_momentum(prices: pd.DataFrame, lookback: int, skip: int) -> pd.DataFrame:
    """Per-name price return over the lookback window, skipping recent `skip` days."""
    return prices.shift(skip) / prices.shift(skip + lookback) - 1.0


def _benchmark_price(prices: pd.DataFrame) -> pd.Series:
    """Equal-weight universe price index from per-day mean returns (point-in-time)."""
    bench_ret = prices.pct_change(fill_method=None).mean(axis=1)
    return (1.0 + bench_ret.fillna(0.0)).cumprod()


def _benchmark_momentum(prices: pd.DataFrame, lookback: int, skip: int) -> pd.Series:
    """Equal-weight benchmark momentum over the same lookback/skip window."""
    bench_px = _benchmark_price(prices)
    return bench_px.shift(skip) / bench_px.shift(skip + lookback) - 1.0


def _rolling_beta(stock_ret: pd.DataFrame, bench_ret: pd.Series,
                  beta_window: int) -> pd.DataFrame:
    """Rolling beta of each name to the benchmark using only past/current returns.

    beta_i,t = Cov(r_i, r_bench) / Var(r_bench) over the trailing ``beta_window``.
    Computed from the E[XY] - E[X]E[Y] identity so a single Series benchmark
    broadcasts cleanly across all columns without look-ahead.
    """
    min_periods = max(20, beta_window // 3)
    mean_stock = stock_ret.rolling(beta_window, min_periods=min_periods).mean()
    mean_bench = bench_ret.rolling(beta_window, min_periods=min_periods).mean()
    mean_prod = (
        stock_ret.mul(bench_ret, axis=0)
        .rolling(beta_window, min_periods=min_periods)
        .mean()
    )
    cov = mean_prod.sub(mean_stock.mul(mean_bench, axis=0), axis=0)
    var_bench = bench_ret.rolling(beta_window, min_periods=min_periods).var()
    return cov.div(var_bench, axis=0)


def compute_scores(
    prices: pd.DataFrame,
    mode: str = 'momentum',
    lookback: int = 126,
    skip: int = 21,
    short_window: int = 5,
    score_mode: str = DEFAULT_SCORE_MODE,
    beta_window: int = DEFAULT_BETA_WINDOW,
) -> pd.DataFrame:
    """Cross-sectional score per name.

    score_mode:
      raw_momentum — price return over lookback/skip (default)
      risk_adjusted_momentum — return / realized vol
      multi_horizon_composite — avg percentile rank of benchmark-relative
        momentum across 20d, 63d, and 126d skip-21 horizons
      relative_momentum — raw momentum minus equal-weight benchmark momentum
        over the same lookback/skip window
      residual_momentum — raw momentum minus beta * benchmark momentum, where
        beta is a rolling (``beta_window``) estimate from past returns; falls
        back to relative momentum where beta cannot be estimated

    All modes use only past/current prices (no look-ahead). The benchmark is the
    equal-weight mean of whatever names are present in ``prices``, so in a
    walk-forward slice it reflects only that slice's universe and dates.
    """
    if mode == 'momentum':
        validate_xs_params(top_frac=0.25, rebalance=1, lookback=lookback, skip=skip)
        if score_mode == 'raw_momentum':
            return _raw_momentum(prices, lookback, skip)
        if score_mode == 'risk_adjusted_momentum':
            ret = _raw_momentum(prices, lookback, skip)
            vol = prices.pct_change(fill_method=None).rolling(
                lookback, min_periods=max(5, lookback // 3)).std()
            return ret / vol.replace(0, np.nan)
        if score_mode == 'relative_momentum':
            raw = _raw_momentum(prices, lookback, skip)
            bench_mom = _benchmark_momentum(prices, lookback, skip)
            return raw.sub(bench_mom, axis=0)
        if score_mode == 'residual_momentum':
            if beta_window is None or int(beta_window) <= 1:
                raise ValueError('beta_window must be an integer > 1.')
            raw = _raw_momentum(prices, lookback, skip)
            bench_mom = _benchmark_momentum(prices, lookback, skip)
            stock_ret = prices.pct_change(fill_method=None)
            bench_ret = stock_ret.mean(axis=1)
            beta = _rolling_beta(stock_ret, bench_ret, int(beta_window))
            relative = raw.sub(bench_mom, axis=0)
            residual = raw.sub(beta.mul(bench_mom, axis=0), axis=0)
            # Fall back to relative momentum wherever beta is unavailable.
            return residual.where(beta.notna(), relative)
        if score_mode == 'multi_horizon_composite':
            horizons = [(20, 0), (63, 0), (126, 21)]
            ranks = []
            for lb, sk in horizons:
                raw = _raw_momentum(prices, lb, sk)
                bench_mom = _benchmark_momentum(prices, lb, sk)
                rel = raw.sub(bench_mom, axis=0)
                ranks.append(rel.rank(axis=1, pct=True))
            return sum(ranks) / len(ranks)
        raise ValueError(f'unknown score_mode: {score_mode}')
    if mode == 'reversal':
        validate_xs_params(top_frac=0.25, rebalance=1, short_window=short_window)
        return -(prices / prices.shift(short_window) - 1.0)
    raise ValueError(f'unknown mode: {mode}')


def _select_longs_hysteresis(
    s: pd.Series,
    prev_longs: list[str],
    k: int,
    entry_rank_pct: float,
    exit_rank_pct: float,
    max_new_names: int | None,
) -> list[str]:
    """Pick the long book with entry/exit rank bands to dampen turnover.

    A held name is retained while its rank percentile stays at or above
    ``exit_rank_pct``; a new name is only added once its percentile reaches
    ``entry_rank_pct``. Existing holds get priority for the ``k`` slots, which is
    what actually reduces churn versus rebuilding the top-k every rebalance.
    """
    if not 0.0 < exit_rank_pct <= entry_rank_pct <= 1.0:
        raise ValueError('require 0 < exit_rank_pct <= entry_rank_pct <= 1.')
    pct = s.rank(pct=True)
    kept = [t for t in prev_longs if t in pct.index and float(pct[t]) >= exit_rank_pct]
    kept.sort(key=lambda t: float(pct[t]), reverse=True)
    if len(kept) > k:
        kept = kept[:k]
    slots = k - len(kept)
    new_names: list[str] = []
    if slots > 0:
        held = set(kept)
        candidates = [
            t for t in pct.index
            if t not in held and float(pct[t]) >= entry_rank_pct
        ]
        candidates.sort(key=lambda t: float(pct[t]), reverse=True)
        if max_new_names is not None:
            candidates = candidates[:max(0, int(max_new_names))]
        new_names = candidates[:slots]
    return kept + new_names


def _group_neutral_long_weights(
    prices: pd.DataFrame,
    t: int,
    s: pd.Series,
    group_map: dict,
    group_top_frac: float,
    weighting: str,
    risk_lookback: int = DEFAULT_RISK_LOOKBACK,
    risk_shrink: float = 0.2,
) -> pd.Series:
    """Long-only weights that equal-weight peer groups, then names within a group.

    Within each group the top ``group_top_frac`` of scored names (at least one)
    is selected; active groups split capital equally; intra-group sizing follows
    ``weighting``. Scoring uses only ``s`` (data through rebalance ``t``), so
    there is no look-ahead. Total weight sums to 1 across selected names.
    """
    if not 0.0 < group_top_frac <= 1.0:
        raise ValueError('group_top_frac must be in (0, 1].')
    members_by_group: dict[str, list[str]] = {}
    for ticker in s.index:
        group = group_map.get(ticker, DEFAULT_PEER_GROUP)
        members_by_group.setdefault(group, []).append(ticker)

    selected_by_group: dict[str, list[str]] = {}
    for group, members in members_by_group.items():
        ranked = s[members].sort_values()
        kg = max(1, int(round(group_top_frac * len(members))))
        selected_by_group[group] = list(ranked.index[-kg:])

    active = [g for g, sel in selected_by_group.items() if sel]
    w = pd.Series(0.0, index=prices.columns)
    if not active:
        return w[w > 0]
    group_capital = 1.0 / len(active)
    for group in active:
        intra = _sized_long_weights(
            prices, t, selected_by_group[group], weighting,
            risk_lookback=risk_lookback, risk_shrink=risk_shrink,
        )
        w[intra.index] = w[intra.index].add(group_capital * intra, fill_value=0.0)
    return w[w > 0]


def build_weights(prices: pd.DataFrame, scores: pd.DataFrame, top_frac: float = 0.33,
                  rebalance: int = 5, market_neutral: bool = True,
                  weighting: str = 'equal',
                  risk_lookback: int = DEFAULT_RISK_LOOKBACK,
                  risk_shrink: float = 0.2,
                  *,
                  use_hysteresis: bool = False,
                  entry_rank_pct: float = 0.80,
                  exit_rank_pct: float = 0.60,
                  max_new_names_per_rebalance: int | None = None,
                  group_neutral: bool = False,
                  group_map: dict | None = None,
                  group_top_frac: float | None = None) -> pd.DataFrame:
    """Build daily target weights with periodic rebalance.

    When ``use_hysteresis`` is True, the long leg uses entry/exit rank bands so
    names are not churned every rebalance (lower turnover, lower cost drag). The
    short leg of a market-neutral book keeps the fresh bottom-k behavior. With
    hysteresis off, behavior is unchanged from the original top-k construction.

    When ``group_neutral`` is True (long-only books only), capital is spread
    equally across peer groups and the top ``group_top_frac`` names are picked
    within each group, so the book is not dominated by one sub-industry. This
    path takes precedence over hysteresis and requires ``group_map``.
    """
    validate_xs_params(top_frac=top_frac, rebalance=rebalance)
    weighting = _normal_weighting_name(weighting)
    if market_neutral and weighting != 'equal':
        raise ValueError('non-equal weighting is only implemented for the long-only book.')
    if group_neutral:
        if market_neutral:
            raise ValueError('group-neutral construction is long-only (set market_neutral=False).')
        if not group_map:
            raise ValueError('group_neutral=True requires a non-empty group_map.')
    gtf = top_frac if group_top_frac is None else float(group_top_frac)
    n = prices.shape[1]
    k = max(1, int(round(top_frac * n)))
    rebal_days = set(range(0, len(prices), rebalance))
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    last_w = pd.Series(0.0, index=prices.columns)
    prev_longs: list[str] = []
    for t in range(len(prices)):
        if t in rebal_days:
            s = scores.iloc[t].dropna()
            if group_neutral:
                if len(s) >= 1:
                    gw = _group_neutral_long_weights(
                        prices, t, s, group_map, gtf, weighting,
                        risk_lookback=risk_lookback, risk_shrink=risk_shrink,
                    )
                    if not gw.empty:
                        last_w = gw.reindex(prices.columns).fillna(0.0)
                        prev_longs = list(last_w[last_w > 0].index)
            elif len(s) >= 2 * k:
                ranked = s.sort_values()
                w = pd.Series(0.0, index=prices.columns)
                if use_hysteresis:
                    selected_longs = _select_longs_hysteresis(
                        s, prev_longs, k,
                        entry_rank_pct, exit_rank_pct,
                        max_new_names_per_rebalance,
                    )
                else:
                    selected_longs = list(ranked.index[-k:])
                if selected_longs:
                    long_w = _sized_long_weights(
                        prices,
                        t,
                        selected_longs,
                        weighting,
                        risk_lookback=risk_lookback,
                        risk_shrink=risk_shrink,
                    )
                    w[long_w.index] = long_w
                if market_neutral:
                    w[ranked.index[:k]] = -1.0 / k
                last_w = w
                prev_longs = list(w[w > 0].index)
        weights.iloc[t] = last_w
    return weights


def portfolio_returns(weights: pd.DataFrame, rets: pd.DataFrame,
                      cost: float | pd.Series = DEFAULT_XS_COST) -> pd.DataFrame:
    """Compute portfolio returns with no look-ahead and aligned transaction costs.

    Execution assumption
    --------------------
    Signal and weights are set at the close of day t.
    The new position becomes active on day t+1.

        portfolio_return[t] = sum_i weights[t-1, i] * returns[t, i]
        turnover[t]         = sum_i abs(weights[t, i] - weights[t-1, i])
        gross_return[t]     = sum_i weights[t-1, i] * returns[t, i]
        net_return[t]       = gross_return[t] - turnover[t] * cost

    ``cost`` may also be a per-ticker Series. In that case, each ticker's
    absolute weight change is multiplied by its own cost before summing.

    Costs are charged on day t when the position from the prior weight change
    becomes active (turnover[t] paired with gross_return[t]).
    """
    port_gross = (weights.shift(1) * rets).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    if isinstance(cost, pd.Series):
        cost_vec = cost.astype(float).reindex(weights.columns).fillna(DEFAULT_XS_COST)
        cost_paid = weights.diff().abs().mul(cost_vec, axis=1).sum(axis=1).fillna(0.0)
    else:
        cost_paid = turnover * float(cost)
    port_net = port_gross - cost_paid
    bench = rets.mean(axis=1)
    return pd.DataFrame({
        'strat_net': port_net,
        'ret': bench,
        'turnover': turnover,
        'cost': cost_paid,
        'gross': port_gross,
    }).dropna()


def backtest_xs(prices: pd.DataFrame, mode: str = 'momentum', top_frac: float = 0.33,
                rebalance: int = 5, cost: float | pd.Series = DEFAULT_XS_COST,
                market_neutral: bool = True,
                weighting: str = 'equal',
                risk_lookback: int = DEFAULT_RISK_LOOKBACK,
                risk_shrink: float = 0.2,
                use_hysteresis: bool = False,
                entry_rank_pct: float = 0.80,
                exit_rank_pct: float = 0.60,
                max_new_names_per_rebalance: int | None = None,
                group_neutral: bool = False,
                group_map: dict | None = None,
                group_top_frac: float | None = None,
                **sig_kw) -> pd.DataFrame:
    validate_xs_params(
        top_frac=top_frac,
        rebalance=rebalance,
        lookback=sig_kw.get('lookback'),
        skip=sig_kw.get('skip'),
        short_window=sig_kw.get('short_window'),
    )
    rets = prices.pct_change(fill_method=None)
    scores = compute_scores(prices, mode=mode, **sig_kw)
    weights = build_weights(
        prices,
        scores,
        top_frac,
        rebalance,
        market_neutral,
        weighting=weighting,
        risk_lookback=risk_lookback,
        risk_shrink=risk_shrink,
        use_hysteresis=use_hysteresis,
        entry_rank_pct=entry_rank_pct,
        exit_rank_pct=exit_rank_pct,
        max_new_names_per_rebalance=max_new_names_per_rebalance,
        group_neutral=group_neutral,
        group_map=group_map,
        group_top_frac=group_top_frac,
    )
    result = portfolio_returns(weights, rets, cost)
    n_rebalances = int((weights.diff().abs().sum(axis=1) > 1e-9).sum())
    return result, n_rebalances


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
            'top_frac': DEFAULT_TOP_FRAC,
            'rebalance': 5,
            'market_neutral': True,
            'weighting': 'equal',
            'risk_lookback': DEFAULT_RISK_LOOKBACK,
            'risk_shrink': 0.2,
            'score_mode': DEFAULT_SCORE_MODE,
            'beta_window': DEFAULT_BETA_WINDOW,
        }

    def _sig_params(self, params: dict) -> dict:
        mode = params.get('mode', 'momentum')
        kw = {}
        if mode == 'momentum':
            kw['lookback'] = params.get('lookback', 126)
            kw['skip'] = params.get('skip', 21)
            kw['score_mode'] = params.get('score_mode', DEFAULT_SCORE_MODE)
            kw['beta_window'] = params.get('beta_window', DEFAULT_BETA_WINDOW)
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
            weighting=p.get('weighting', 'equal'),
            risk_lookback=p.get('risk_lookback', DEFAULT_RISK_LOOKBACK),
            risk_shrink=p.get('risk_shrink', 0.2),
            **self._sig_params(p),
        )
        return df.dropna(), n_rebal

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
        s = scores.iloc[-1].dropna()
        if p.get('group_neutral'):
            if p.get('market_neutral'):
                raise ValueError('group-neutral construction is long-only.')
            group_map = p.get('group_map')
            if not group_map:
                raise ValueError('group_neutral=True requires a non-empty group_map.')
            if len(s) < 1:
                raise ValueError('Need at least 1 scored name for group-neutral book.')
            gtf = p['top_frac'] if p.get('group_top_frac') is None else float(p['group_top_frac'])
            w = _group_neutral_long_weights(
                panel, len(panel) - 1, s, group_map, gtf,
                _normal_weighting_name(p.get('weighting', 'equal')),
                risk_lookback=p.get('risk_lookback', DEFAULT_RISK_LOOKBACK),
                risk_shrink=p.get('risk_shrink', 0.2),
            )
            return w[w.abs() > 1e-9]
        n = panel.shape[1]
        k = max(1, int(round(p['top_frac'] * n)))
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
                     universe: list[str], preset_name: str = DEFAULT_PRESET,
                     **params) -> str:
        p = {**self.default_params(), **params}
        longs = weights[weights > 0].sort_values(ascending=False)
        shorts = weights[weights < 0].sort_values()
        mom_preset = params.get('momentum_preset', 'mom_126d_skip21')
        from quant.momentum_presets import format_momentum_header_lines
        lines = [
            f'=== Cross-Sectional Book ({p["mode"]}) ===',
            f'Universe preset: {preset_name}',
            f'Universe ({len(universe)}): {", ".join(universe)}',
        ]
        lines.extend(format_momentum_header_lines(
            p['mode'], mom_preset, p['lookback'], p['skip'], p['rebalance'],
        ))
        if p['mode'] != 'momentum':
            lines.append(
                f'Rebalance every: {p["rebalance"]} days  |  Top/bottom frac: {p["top_frac"]:.0%}'
            )
            lines.append('')
        else:
            lines.append(f'Top/bottom frac: {p["top_frac"]:.0%}')
            lines.append('')
        lines.append('LONG leg:')
        for tk, wt in longs.items():
            sc = scores.get(tk, float('nan'))
            lines.append(f'  {tk:<6} weight {wt:+.2%}   score {sc:+.1%}')
        lines.append('SHORT leg:')
        for tk, wt in shorts.items():
            sc = scores.get(tk, float('nan'))
            lines.append(f'  {tk:<6} weight {wt:+.2%}   score {sc:+.1%}')
        lines.append('')
        lines.append(universe_selection_note())
        return '\n'.join(lines)
