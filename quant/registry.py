from __future__ import annotations

from quant.models.base import TradingModel
from quant.models.mean_reversion import MeanReversionModel
from quant.models.momentum import MomentumModel

MODELS: dict[str, TradingModel] = {
    MeanReversionModel.slug: MeanReversionModel(),
    MomentumModel.slug: MomentumModel(),
}


def list_models() -> list[TradingModel]:
    return list(MODELS.values())


def get_model(slug: str) -> TradingModel:
    key = slug.lower().replace('_', '-')
    if key == 'all':
        raise ValueError('"all" is not a single model — use it with backtest/report commands.')
    if key not in MODELS:
        available = ', '.join(sorted(MODELS))
        raise ValueError(f'Unknown model "{slug}". Available: {available}')
    return MODELS[key]


def resolve_models(slug: str) -> list[TradingModel]:
    """Return one model, or all models if slug is 'all'."""
    if slug.lower() == 'all':
        return list_models()
    return [get_model(slug)]
