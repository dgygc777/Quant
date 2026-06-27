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
        'TXN', 'SNPS', 'CDNS', 'ARM', 'GFS', 'AMKR', 'TER', 'COHR',
    ],

    # EDA and semiconductor IP / design tools
    'eda_ip': [
        'SNPS', 'CDNS', 'ARM',
    ],

    # Analog, mixed-signal, and power semiconductors
    'analog_power': [
        'TXN', 'ADI', 'NXPI', 'MCHP', 'ON', 'MPWR', 'STM',
    ],

    # Foundry and integrated-device manufacturing
    'foundry': [
        'TSM', 'GFS', 'UMC', 'INTC',
    ],

    # Semiconductor equipment (wafer-fab tools and related)
    'equipment': [
        'AMAT', 'LRCX', 'KLAC', 'ASML', 'TER', 'ONTO', 'KLIC', 'ACLS', 'ENTG',
    ],

    # AI connectivity: networking, optical, and high-speed interconnect
    'ai_connectivity': [
        'ALAB', 'CRDO', 'MRVL', 'AVGO', 'COHR',
    ],

    # Outsourced semiconductor assembly and test (OSAT) / packaging
    'osat': [
        'ASX', 'AMKR',
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

# Pool for ranking NASDAQ names by live market cap (dedupe GOOG → GOOGL).
NASDAQ_TOP30_CANDIDATES: list[str] = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'AVGO', 'TSLA', 'NFLX',
    'COST', 'AMD', 'PEP', 'CSCO', 'ADBE', 'QCOM', 'TXN', 'INTU', 'AMAT', 'ISRG',
    'BKNG', 'ARM', 'PANW', 'ADP', 'MU', 'LRCX', 'KLAC', 'SNPS', 'CDNS', 'REGN',
    'ASML', 'MELI', 'MAR', 'SBUX', 'ABNB', 'CRWD', 'FTNT', 'DXCM', 'MNST', 'ADSK',
    'PYPL', 'CHTR', 'CMCSA', 'GILD', 'HON', 'PDD', 'MRVL', 'LULU', 'WDAY', 'ORLY',
    'CTAS', 'PCAR', 'NXPI', 'MCHP', 'ON', 'TTD', 'ZS', 'DDOG', 'TEAM', 'IDXX',
]

# Static fallback — largest NASDAQ-listed names by market cap (approx. early 2026).
NASDAQ_TOP30_FALLBACK: list[str] = [
    'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'AVGO', 'TSLA', 'NFLX',
    'COST', 'AMD', 'PEP', 'CSCO', 'ADBE', 'QCOM', 'TXN', 'INTU', 'AMAT', 'ISRG',
    'BKNG', 'ARM', 'PANW', 'ADP', 'MU', 'LRCX', 'KLAC', 'SNPS', 'CDNS', 'REGN',
    'ASML',
]

UNIVERSE_PRESETS['nasdaq_top30'] = list(NASDAQ_TOP30_FALLBACK)

# Backward compatibility for code that imports DEFAULT_UNIVERSE.
DEFAULT_UNIVERSE = UNIVERSE_PRESETS[DEFAULT_PRESET]

UNIVERSE_DESCRIPTIONS: dict[str, str] = {
    'semis': 'Which semiconductor names are strongest relative to other semiconductor names?',
    'eda_ip': 'Which EDA/IP names lead vs other design-tool peers?',
    'analog_power': 'Which analog/power names lead vs peers?',
    'foundry': 'Which foundry/IDM names lead vs other foundries?',
    'equipment': 'Which semi-equipment names lead vs equipment peers?',
    'ai_connectivity': 'Which AI-connectivity names lead vs peers?',
    'osat': 'Which packaging/OSAT names lead vs peers?',
    'semi_equipment': 'Which semi equipment names lead vs other equipment peers?',
    'memory_storage': 'Which memory and storage names lead vs other memory/storage peers?',
    'foundry_packaging': 'Which foundry/packaging names lead vs other manufacturing peers?',
    'mega_cap_tech': 'Which mega-cap tech names are strongest relative to other mega-cap tech names?',
    'ai_infra': 'Which AI infrastructure names are strongest relative to the AI infrastructure basket?',
    'broad_tech': 'Which broad tech names rank highest within a diversified tech basket?',
    'nasdaq_top30': 'Largest ~30 NASDAQ-listed names by market cap (10-K risk-change scan).',
    'custom': 'Custom ticker list supplied by the user.',
}


def available_universes() -> list[str]:
    return sorted(UNIVERSE_PRESETS.keys())


def resolve_nasdaq_top30(live: bool = True) -> list[str]:
    """Return ~30 largest NASDAQ-listed names by market cap."""
    if live:
        try:
            import yfinance as yf

            caps: list[tuple[str, float]] = []
            for ticker in NASDAQ_TOP30_CANDIDATES:
                try:
                    info = yf.Ticker(ticker).fast_info
                    cap = info.get("market_cap") if hasattr(info, "get") else getattr(info, "market_cap", None)
                    if cap and cap > 0:
                        caps.append((ticker.upper(), float(cap)))
                except Exception:
                    continue
            if len(caps) >= 30:
                caps.sort(key=lambda x: x[1], reverse=True)
                out: list[str] = []
                for ticker, _ in caps:
                    if ticker == "GOOG":
                        ticker = "GOOGL"
                    if ticker not in out:
                        out.append(ticker)
                    if len(out) >= 30:
                        break
                return out
        except Exception:
            pass
    return list(NASDAQ_TOP30_FALLBACK)


def get_universe(name: str) -> list[str]:
    key = name.lower().replace('-', '_')
    if key == 'nasdaq_top30':
        return resolve_nasdaq_top30()
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
    from quant.params import validate_top_frac
    validate_top_frac(top_frac)
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
