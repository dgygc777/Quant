from __future__ import annotations

from quant.models.base import TradingModel
from quant.models.cross_sectional import CrossSectionalModel
from quant.models.mean_reversion import MeanReversionModel
from quant.models.momentum import MomentumModel
from quant.models.panel_base import PanelModel

MODELS: dict[str, TradingModel] = {
    MeanReversionModel.slug: MeanReversionModel(),
    MomentumModel.slug: MomentumModel(),
}

PANEL_MODELS: dict[str, PanelModel] = {
    CrossSectionalModel.slug: CrossSectionalModel(),
}


def list_models() -> list[TradingModel]:
    return list(MODELS.values())


def list_panel_models() -> list[PanelModel]:
    return list(PANEL_MODELS.values())


def get_model(slug: str) -> TradingModel:
    key = slug.lower().replace('_', '-')
    if key == 'all':
        raise ValueError('"all" is not a single model — use it with backtest/report commands.')
    if key not in MODELS:
        available = ', '.join(sorted(MODELS))
        raise ValueError(f'Unknown model "{slug}". Available: {available}')
    return MODELS[key]


def get_panel_model(slug: str = 'cross-sectional') -> PanelModel:
    key = slug.lower().replace('_', '-')
    if key not in PANEL_MODELS:
        available = ', '.join(sorted(PANEL_MODELS))
        raise ValueError(f'Unknown panel model "{slug}". Available: {available}')
    return PANEL_MODELS[key]


def resolve_models(slug: str) -> list[TradingModel]:
    """Return one model, or all models if slug is 'all'."""
    if slug.lower() == 'all':
        return list_models()
    return [get_model(slug)]
