from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class TradingModel(ABC):
    """Base class every analytical model must implement."""

    slug: str
    name: str
    description: str

    @abstractmethod
    def default_params(self) -> dict[str, Any]:
        """Default hyperparameters for this model."""

    @abstractmethod
    def compute_indicators(self, price: pd.Series, **params) -> pd.DataFrame:
        """Return a DataFrame with price, ret, and model-specific indicator columns."""

    @abstractmethod
    def next_action(self, row: pd.Series, in_position: bool,
                    **params) -> tuple[bool, str | None]:
        """Return (new_in_position, 'BUY'|'SELL'|None) for one bar."""

    def build_positions(self, df: pd.DataFrame, **params) -> pd.Series:
        """Historical position series (1 = long, 0 = flat) with hold logic."""
        pos = []
        in_position = False
        for _, row in df.iterrows():
            in_position, _ = self.next_action(row, in_position, **params)
            pos.append(1.0 if in_position else 0.0)
        return pd.Series(pos, index=df.index)

    def backtest(self, price: pd.Series, cost: float = 0.0005,
                 **params) -> tuple[pd.DataFrame, int]:
        df = self.compute_indicators(price, **params)
        df['pos'] = self.build_positions(df, **params)
        df['strat_gross'] = df['pos'].shift(1) * df['ret']
        trades = df['pos'].diff().abs().fillna(0.0)
        df['strat_net'] = df['strat_gross'] - trades * cost
        return df.dropna(), int(trades.sum())

    @abstractmethod
    def explain_math(self, **params) -> str:
        """Plain-language explanation of the model's math."""

    @abstractmethod
    def format_signal(self, ticker: str, row: pd.Series, live_price: float,
                      quote_ts: str, in_position: bool, **params) -> str:
        """Multi-line text block for the current live signal."""

    def signal_value(self, row: pd.Series, **params) -> float:
        """Primary indicator value used for state persistence."""
        raise NotImplementedError

    def min_history_days(self, **params) -> int:
        """Minimum daily bars needed for live signals."""
        return 120
