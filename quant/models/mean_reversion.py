from __future__ import annotations

import numpy as np
import pandas as pd

from quant.models.base import TradingModel


class MeanReversionModel(TradingModel):
    """Z-score mean reversion: buy oversold, sell on reversion to the mean."""

    slug = 'mean-reversion'
    name = 'Mean Reversion'
    description = 'Buy when price is oversold vs its rolling mean (z-score); sell on reversion.'

    def default_params(self) -> dict:
        return {'window': 20, 'entry_z': -1.0, 'exit_z': 0.0}

    def min_history_days(self, **params) -> int:
        window = params.get('window', self.default_params()['window'])
        return window + 30

    def compute_indicators(self, price: pd.Series, **params) -> pd.DataFrame:
        window = params.get('window', 20)
        df = pd.DataFrame({'price': price})
        df['ret'] = df['price'].pct_change()
        df['roll_mean'] = df['price'].rolling(window).mean()
        df['roll_std'] = df['price'].rolling(window).std()
        df['z'] = (df['price'] - df['roll_mean']) / df['roll_std']
        return df

    def next_action(self, row: pd.Series, in_position: bool,
                    **params) -> tuple[bool, str | None]:
        entry_z = params.get('entry_z', -1.0)
        exit_z = params.get('exit_z', 0.0)
        z = row['z']
        if np.isnan(z):
            return in_position, None
        if (not in_position) and z < entry_z:
            return True, 'BUY'
        if in_position and z >= exit_z:
            return False, 'SELL'
        return in_position, None

    def signal_value(self, row: pd.Series, **params) -> float:
        return float(row['z'])

    def explain_math(self, **params) -> str:
        window = params.get('window', 20)
        entry_z = params.get('entry_z', -1.0)
        exit_z = params.get('exit_z', 0.0)
        return f"""
How the math works (z-score mean reversion)
------------------------------------------
1. Rolling mean (μ): average close over the last {window} trading days.
2. Rolling std  (σ): how much price typically swings around that mean.
3. Z-score:        z = (price - μ) / σ

   z =  0  → price is exactly at its recent average
   z = -1  → price is 1 std dev BELOW average (oversold)
   z = +1  → price is 1 std dev ABOVE average (overbought)

4. Entry rule: go long when z < {entry_z} (oversold → bet on bounce).
5. Exit rule:  sell when z >= {exit_z} (price reverted toward the mean).

Why it might work: prices that stretch far below a short-term average sometimes
snap back. Why it often fails: trends can keep falling ("catching a falling knife").
"""

    def format_signal(self, ticker: str, row: pd.Series, live_price: float,
                      quote_ts: str, in_position: bool, **params) -> str:
        window = params.get('window', 20)
        entry_z = params.get('entry_z', -1.0)
        exit_z = params.get('exit_z', 0.0)
        z = float(row['z'])
        mu = float(row['roll_mean'])
        sigma = float(row['roll_std'])
        _, action = self.next_action(row, in_position, **params)

        if action == 'BUY':
            rec = 'BUY  — z-score below entry threshold (oversold)'
        elif action == 'SELL':
            rec = 'SELL — z-score reverted to exit threshold'
        elif in_position:
            rec = 'HOLD — in position, waiting for z >= exit threshold'
        else:
            rec = 'WAIT — not oversold enough to enter'

        lines = [
            f'=== {ticker.upper()} — {self.name} ===',
            f'Quote time:     {quote_ts}',
            f'Live price:     ${live_price:,.2f}',
            f'{window}-day mean (μ): ${mu:,.2f}',
            f'{window}-day std  (σ): ${sigma:,.2f}',
            f'Z-score:        {z:+.2f}   (entry < {entry_z}, exit >= {exit_z})',
            f'In position:    {"yes" if in_position else "no"}',
            f'Recommendation: {rec}',
        ]
        return '\n'.join(lines)
