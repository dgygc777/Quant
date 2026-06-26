"""
Named universe presets for cross-sectional portfolio models.

Cross-sectional signals are RELATIVE rankings within the chosen basket.
A stock can have positive momentum on its own but still land in the short leg
if it ranks near the bottom of the selected universe. Universe definition is
part of the model — mixing semis, mega-cap tech, and other names changes
what question the backtest answers.
"""

from __future__ import annotations

DEFAULT_PRESET = 'semis'

UNIVERSE_PRESETS: dict[str, list[str]] = {
    # Core semiconductor universe: designers, memory, foundry, packaging, equipment
    'semis': [
        'MU', 'SNDK', 'NVDA', 'AMD', 'AVGO', 'QCOM', 'TSM',
        'ASX', 'AMAT', 'LRCX', 'KLAC', 'INTC', 'ASML', 'MRVL',
    ],

    # Semiconductor equipment / wafer-fab tool makers
    'semi_equipment': [
        'AMAT', 'LRCX', 'KLAC', 'ASML',
    ],

    # Memory and storage names
    'memory_storage': [
        'MU', 'SNDK', 'WDC', 'STX',
    ],

    # Foundry / manufacturing / packaging exposure
    'foundry_packaging': [
        'TSM', 'ASX', 'INTC',
    ],

    # Mega-cap technology / platform companies
    'mega_cap_tech': [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'AVGO', 'TSLA',
    ],

    # AI infrastructure: chips, memory, foundry, equipment, cloud platforms
    'ai_infra': [
        'NVDA', 'AMD', 'AVGO', 'MU', 'SNDK', 'MRVL', 'TSM',
        'ASML', 'AMAT', 'LRCX', 'KLAC', 'MSFT', 'GOOGL', 'AMZN',
    ],

    # Broad tech / AI / semiconductor basket
    'broad_tech': [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
        'NVDA', 'AMD', 'AVGO', 'QCOM', 'TSM', 'ASML',
        'AMAT', 'LRCX', 'KLAC', 'MU', 'SNDK', 'MRVL', 'INTC',
    ],
}

# Backward compatibility for code that imports DEFAULT_UNIVERSE.
DEFAULT_UNIVERSE = UNIVERSE_PRESETS[DEFAULT_PRESET]

UNIVERSE_DESCRIPTIONS: dict[str, str] = {
    'semis': 'Which semiconductor names are strongest relative to other semiconductor names?',
    'semi_equipment': 'Which semi equipment names lead vs other equipment peers?',
    'memory_storage': 'Which memory and storage names lead vs other memory/storage peers?',
    'foundry_packaging': 'Which foundry/packaging names lead vs other manufacturing peers?',
    'mega_cap_tech': 'Which mega-cap tech names are strongest relative to other mega-cap tech names?',
    'ai_infra': 'Which AI infrastructure names are strongest relative to the AI infrastructure basket?',
    'broad_tech': 'Which broad tech names rank highest within a diversified tech basket?',
    'custom': 'Custom ticker list supplied by the user.',
}


def available_universes() -> list[str]:
    return sorted(UNIVERSE_PRESETS.keys())


def get_universe(name: str) -> list[str]:
    key = name.lower().replace('-', '_')
    if key not in UNIVERSE_PRESETS:
        avail = ', '.join(available_universes())
        raise ValueError(f'Unknown universe: {name}\nAvailable universes: {avail}')
    return list(UNIVERSE_PRESETS[key])


def parse_tickers(tickers: str) -> list[str]:
    """Parse comma-separated tickers; dedupe while preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for part in tickers.split(','):
        t = part.strip().upper()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def resolve_universe(universe: str = DEFAULT_PRESET,
                     tickers: str | None = None) -> tuple[str, list[str]]:
    """Return (preset_label, ticker_list). Custom tickers override preset name."""
    if tickers:
        parsed = parse_tickers(tickers)
        if not parsed:
            raise ValueError('No valid tickers in --tickers.')
        return 'custom', parsed
    return universe.lower().replace('-', '_'), get_universe(universe)


def validate_universe_size(tickers: list[str], top_frac: float) -> None:
    """Ensure enough names for long + short legs: n >= 2*k, k = round(top_frac * n)."""
    n = len(tickers)
    if n < 2:
        raise ValueError('Cross-sectional model needs at least 2 tickers.')
    k = max(1, int(round(top_frac * n)))
    needed = 2 * k
    if n < needed:
        raise ValueError(
            f'Universe has {n} ticker(s) but needs at least {needed} for '
            f'top_frac={top_frac:.0%} (k={k} long + {k} short). '
            f'Add more tickers or lower --top-frac.'
        )


def universe_selection_note() -> str:
    return (
        'Note: Cross-sectional ranks are relative to the selected universe.\n'
        'Changing the universe can change which stocks appear in the long/short legs.\n'
        'Use cleaner presets when you want the signal to answer a specific question.'
    )


def describe_preset(preset_name: str) -> str:
    return UNIVERSE_DESCRIPTIONS.get(preset_name, UNIVERSE_DESCRIPTIONS['custom'])


def format_universes_listing() -> str:
    lines = ['Available universe presets:', '']
    for name in available_universes():
        tickers = UNIVERSE_PRESETS[name]
        lines.append(f'{name} ({len(tickers)}):')
        lines.append(f'  {", ".join(tickers)}')
        lines.append(f'  → {describe_preset(name)}')
        lines.append('')
    lines.append(universe_selection_note())
    return '\n'.join(lines)
