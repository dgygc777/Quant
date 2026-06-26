from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class PanelModel(ABC):
    """Base class for multi-asset / cross-sectional models."""

    slug: str
    name: str
    description: str

    @abstractmethod
    def default_params(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def backtest(self, panel: pd.DataFrame, cost: float = 0.0005,
                 **params) -> tuple[pd.DataFrame, int]:
        """Return (results_df with strat_net & bench columns, rebalance_count)."""

    @abstractmethod
    def current_weights(self, panel: pd.DataFrame, **params) -> pd.Series:
        """Target portfolio weights at the latest date."""

    @abstractmethod
    def explain_math(self, **params) -> str:
        pass
