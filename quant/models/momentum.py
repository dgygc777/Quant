from __future__ import annotations

import numpy as np
import pandas as pd

from quant.models.base import TradingModel


class MomentumModel(TradingModel):
    """Time-series momentum with skip-window signal and volatility targeting.

    Signal: sign of past return over lookback, skipping the most recent `skip`
    days (classic 12-1 style convention on daily bars).
    Sizing: scale exposure inversely to realized vol for roughly constant risk.
    """

    slug = 'momentum'
    name = 'Momentum'
    description = (
        'Time-series momentum: ride medium-term trends (skip-window signal) '
        'with volatility-targeted sizing.'
    )

    def default_params(self) -> dict:
        return {
            'lookback': 126,
            'skip': 21,
            'vol_window': 63,
            'target_vol': 0.15,
            'vol_scale': True,
            'long_only': True,
            'max_leverage': 3.0,
        }

    def min_history_days(self, **params) -> int:
        p = {**self.default_params(), **params}
        return p['lookback'] + p['skip'] + p['vol_window'] + 30

    def compute_indicators(self, price: pd.Series, **params) -> pd.DataFrame:
        lookback = params.get('lookback', 126)
        skip = params.get('skip', 21)
        vol_window = params.get('vol_window', 63)
        target_vol = params.get('target_vol', 0.15)
        vol_scale = params.get('vol_scale', True)
        long_only = params.get('long_only', True)
        max_leverage = params.get('max_leverage', 3.0)

        df = pd.DataFrame({'price': price})
        df['ret'] = df['price'].pct_change()

        # Return from (t - skip - lookback) to (t - skip); both points are in the past.
        past = df['price'].shift(skip) / df['price'].shift(skip + lookback) - 1.0
        df['momentum'] = past

        if long_only:
            sign = (past > 0).astype(float)
        else:
            sign = np.sign(past).fillna(0.0)

        if vol_scale:
            realized_vol = df['ret'].rolling(vol_window).std() * np.sqrt(252.0)
            weight = sign * (target_vol / realized_vol)
            lo = 0.0 if long_only else -max_leverage
            weight = weight.clip(lower=lo, upper=max_leverage)
            df['realized_vol'] = realized_vol
        else:
            weight = sign * 1.0
            df['realized_vol'] = np.nan

        df['weight'] = weight
        df['pos'] = (weight > 0).astype(float)
        return df

    def next_action(self, row: pd.Series, in_position: bool,
                    **params) -> tuple[bool, str | None]:
        long_only = params.get('long_only', True)
        mom = row['momentum']
        weight = row['weight']
        if np.isnan(mom) or np.isnan(weight):
            return in_position, None

        if long_only:
            want_long = mom > 0
            if (not in_position) and want_long:
                return True, 'BUY'
            if in_position and not want_long:
                return False, 'SELL'
            return in_position, None

        # Long/short paper mode: treat any non-zero weight as invested.
        want_in = abs(weight) > 1e-9
        if (not in_position) and want_in:
            return True, 'BUY'
        if in_position and not want_in:
            return False, 'SELL'
        return in_position, None

    def backtest(self, price: pd.Series, cost: float = 0.0005,
                 **params) -> tuple[pd.DataFrame, int]:
        df = self.compute_indicators(price, **params)
        df['strat_gross'] = df['weight'].shift(1) * df['ret']
        turnover = df['weight'].diff().abs().fillna(0.0)
        df['strat_net'] = df['strat_gross'] - turnover * cost
        n_trades = int(turnover.gt(1e-9).sum())
        return df.dropna(), n_trades

    def signal_value(self, row: pd.Series, **params) -> float:
        return float(row['momentum'])

    def explain_math(self, **params) -> str:
        lookback = params.get('lookback', 126)
        skip = params.get('skip', 21)
        vol_window = params.get('vol_window', 63)
        target_vol = params.get('target_vol', 0.15)
        vol_scale = params.get('vol_scale', True)
        long_only = params.get('long_only', True)
        max_leverage = params.get('max_leverage', 3.0)
        sizing = (
            f'weight = sign × ({target_vol:.0%} / realized vol), '
            f'capped at {max_leverage:.1f}x'
            if vol_scale else
            'weight = sign × 1.0 (no vol scaling)'
        )
        direction = 'long when sign > 0, flat otherwise' if long_only else 'long/short by sign'
        return f"""
How the math works (time-series momentum)
-----------------------------------------
1. Lookback: {lookback} trading days of past return, skipping the most recent {skip}
   days (avoids short-term reversal contaminating the medium-term trend).

   momentum = price[t-{skip}] / price[t-{skip}-{lookback}] - 1

2. Direction signal: sign(momentum)
   → +1 if the skipped lookback return is positive (uptrend)
   →  0 if flat/negative (long-only mode stays out)

3. Position sizing: {sizing}
   Realized vol = {vol_window}-day rolling std of daily returns × √252.

4. Execution: trade on the NEXT day's return (no look-ahead).
   Transaction costs charged on weight changes (turnover).

5. Paper trading uses the direction signal only ({direction});
   vol targeting applies in backtests.

Why it might work: medium-term trends persist; skipping recent days avoids
short-term mean reversion. Vol targeting keeps risk roughly stable.
Why it often fails: sharp momentum crashes and choppy sideways markets.
"""

    def format_signal(self, ticker: str, row: pd.Series, live_price: float,
                      quote_ts: str, in_position: bool, **params) -> str:
        lookback = params.get('lookback', 126)
        skip = params.get('skip', 21)
        target_vol = params.get('target_vol', 0.15)
        vol_scale = params.get('vol_scale', True)
        long_only = params.get('long_only', True)

        mom = float(row['momentum'])
        weight = float(row['weight'])
        rv = row.get('realized_vol', np.nan)
        _, action = self.next_action(row, in_position, **params)

        if action == 'BUY':
            rec = 'BUY  — positive skipped lookback momentum (uptrend)'
        elif action == 'SELL':
            rec = 'SELL — momentum turned non-positive'
        elif in_position:
            rec = 'HOLD — in position, trend still positive'
        else:
            rec = 'WAIT — no positive momentum signal'

        rv_line = (
            f'Realized vol:   {float(rv):.1%}  (target {target_vol:.0%})'
            if vol_scale and not np.isnan(rv) else
            'Vol targeting:  off'
        )
        weight_line = (
            f'Target weight:  {weight:.2f}x  (vol-scaled)'
            if vol_scale else
            f'Target weight:  {weight:.2f}x'
        )

        lines = [
            f'=== {ticker.upper()} — {self.name} ===',
            f'Quote time:     {quote_ts}',
            f'Live price:     ${live_price:,.2f}',
            f'Momentum:       {mom:+.1%}  ({lookback}d lookback, skip {skip}d)',
            rv_line,
            weight_line,
            f'Mode:           {"long-only" if long_only else "long/short"}',
            f'In position:    {"yes" if in_position else "no"}',
            f'Recommendation: {rec}',
        ]
        return '\n'.join(lines)
