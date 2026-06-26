"""
Named momentum lookback presets for single-stock and cross-sectional models.

Momentum score (no look-ahead at signal time):
    momentum[t] = price[t - skip] / price[t - skip - lookback] - 1

Parameters
----------
lookback : how far back the return window extends.
skip     : ignore the most recent `skip` days to reduce short-term reversal noise.
           skip=0 uses price[t] / price[t - lookback] - 1.

rebalance (portfolio) is separate: it controls how often weights update, not the
signal window. Positions still execute on the NEXT day's return (weights.shift(1)).
"""

from __future__ import annotations

import pandas as pd

from quant.metrics import metrics
from quant.models.momentum import MomentumModel

DEFAULT_MOMENTUM_PRESET = 'mom_126d_skip21'

MOMENTUM_PRESETS: dict[str, dict] = {
    'mom_10d': {
        'lookback': 10,
        'skip': 0,
        'description': '10-day short-term momentum',
    },
    'mom_20d': {
        'lookback': 20,
        'skip': 0,
        'description': '20-day / ~1-month momentum',
    },
    'mom_63d': {
        'lookback': 63,
        'skip': 0,
        'description': '63-day / ~3-month momentum',
    },
    'mom_126d_skip21': {
        'lookback': 126,
        'skip': 21,
        'description': '126-day momentum skipping the most recent 21 days',
    },
}

# Presets used in side-by-side comparison commands.
COMPARISON_PRESET_NAMES: list[str] = [
    'mom_10d',
    'mom_20d',
    'mom_63d',
    'mom_126d_skip21',
]

MOMENTUM_INTERPRETATION: dict[str, str] = {
    'mom_10d': (
        '10-day momentum: fast and reactive, but noisy. '
        'Higher turnover and more false signals.'
    ),
    'mom_20d': (
        '20-day momentum: roughly one trading month. '
        'Useful for short swing trading.'
    ),
    'mom_63d': (
        '63-day momentum: roughly one quarter. '
        'More stable trend signal.'
    ),
    'mom_126d_skip21': (
        '126-day skip-21 momentum: classic medium-term momentum. '
        'Slower, less noisy, skips recent short-term reversal.'
    ),
}


def available_momentum_presets() -> list[str]:
    return sorted(MOMENTUM_PRESETS.keys())


def get_momentum_preset(name: str) -> dict:
    key = name.lower().replace('-', '_')
    if key not in MOMENTUM_PRESETS:
        avail = ', '.join(available_momentum_presets())
        raise ValueError(f'Unknown momentum preset: {name}\nAvailable presets: {avail}')
    return dict(MOMENTUM_PRESETS[key])


def validate_momentum_params(lookback: int, skip: int, n_days: int | None = None) -> None:
    if lookback <= 0:
        raise ValueError(f'lookback must be positive, got {lookback}')
    if skip < 0:
        raise ValueError(f'skip must be >= 0, got {skip}')
    if n_days is not None and lookback + skip >= n_days:
        raise ValueError(
            f'Insufficient data: need more than {lookback + skip} trading days '
            f'(lookback={lookback} + skip={skip}), but only {n_days} available. '
            f'Use a longer --years history or a shorter lookback/skip.'
        )


def resolve_momentum_params(
    momentum_preset: str | None = None,
    lookback: int | None = None,
    skip: int | None = None,
) -> tuple[str, int, int]:
    """Return (preset_label, lookback, skip).

    Preset supplies defaults; explicit lookback/skip override preset values.
    When nothing is specified, uses DEFAULT_MOMENTUM_PRESET (126d / skip 21).
    """
    if momentum_preset is not None:
        p = get_momentum_preset(momentum_preset)
        preset_name = momentum_preset.lower().replace('-', '_')
        lb, sk = p['lookback'], p['skip']
    elif lookback is None and skip is None:
        p = MOMENTUM_PRESETS[DEFAULT_MOMENTUM_PRESET]
        preset_name = DEFAULT_MOMENTUM_PRESET
        lb, sk = p['lookback'], p['skip']
    else:
        base = MOMENTUM_PRESETS[DEFAULT_MOMENTUM_PRESET]
        preset_name = 'custom'
        lb = lookback if lookback is not None else base['lookback']
        sk = skip if skip is not None else base['skip']

    if lookback is not None:
        lb = lookback
    if skip is not None:
        sk = skip

    if momentum_preset is not None:
        ref = MOMENTUM_PRESETS[momentum_preset.lower().replace('-', '_')]
        if lb != ref['lookback'] or sk != ref['skip']:
            preset_name = 'custom'

    validate_momentum_params(lb, sk)
    return preset_name, lb, sk


def format_momentum_header_lines(
    mode: str,
    preset_name: str,
    lookback: int,
    skip: int,
    rebalance: int,
) -> list[str]:
    if mode != 'momentum':
        return []
    lines = [
        'Signal: momentum',
        f'Momentum preset: {preset_name}',
        f'Lookback: {lookback} trading days',
        f'Skip: {skip} trading days',
        f'Rebalance every: {rebalance} trading days',
        '',
    ]
    note = MOMENTUM_INTERPRETATION.get(preset_name)
    if note:
        lines.insert(4, note)
    return lines


def format_momentum_presets_listing() -> str:
    lines = ['Available momentum presets:', '']
    for name in available_momentum_presets():
        p = MOMENTUM_PRESETS[name]
        lines.append(
            f'{name}: lookback={p["lookback"]}, skip={p["skip"]} — {p["description"]}'
        )
        interp = MOMENTUM_INTERPRETATION.get(name)
        if interp:
            lines.append(f'  → {interp}')
        lines.append('')
    lines.append(
        'lookback controls how far back the signal looks; skip ignores recent days; '
        'rebalance is separate and controls portfolio update frequency.'
    )
    return '\n'.join(lines)


def run_xs_momentum_preset_comparison(
    panel: pd.DataFrame,
    base_params: dict,
    cost: float = 0.0005,
) -> list[dict]:
    """Cross-sectional momentum backtest for each comparison preset."""
    from quant.models.cross_sectional import backtest_xs

    rows = []
    for pname in COMPARISON_PRESET_NAMES:
        p = MOMENTUM_PRESETS[pname]
        params = {
            **base_params,
            'mode': 'momentum',
            'lookback': p['lookback'],
            'skip': p['skip'],
        }
        result, n_rebal = backtest_xs(
            panel,
            mode='momentum',
            top_frac=params.get('top_frac', 0.25),
            rebalance=params.get('rebalance', 5),
            cost=cost,
            market_neutral=params.get('market_neutral', True),
            lookback=p['lookback'],
            skip=p['skip'],
        )
        bench = panel.pct_change(fill_method=None).mean(axis=1)
        strat_m = metrics(result['strat_net'])
        aligned = pd.concat([result['strat_net'], bench], axis=1).dropna()
        corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1])) if len(aligned) > 1 else 0.0
        rows.append({
            'preset': pname,
            'lookback': p['lookback'],
            'skip': p['skip'],
            'ann_return': strat_m['ann_return'],
            'ann_vol': strat_m['ann_vol'],
            'sharpe': strat_m['sharpe'],
            'max_dd': strat_m['max_dd'],
            'turnover': float(result['turnover'].mean()),
            'corr_bench': corr,
            'n_rebalances': n_rebal,
        })
    return rows


def print_xs_momentum_preset_comparison(
    rows: list[dict],
    universe_preset: str,
    years: int,
) -> None:
    print('\n=== Cross-Sectional Momentum Lookback Comparison ===')
    print(f'Universe: {universe_preset}')
    print(f'Years: {years}')
    print()
    print(f'{"Preset":<18}{"Lookback":>9}{"Skip":>6}{"AnnRet":>8}{"Vol":>8}'
          f'{"Sharpe":>8}{"MaxDD":>8}{"Turnover":>10}{"CorrBench":>11}')
    print('-' * 87)
    for r in rows:
        print(f'{r["preset"]:<18}{r["lookback"]:>9}{r["skip"]:>6}'
              f'{r["ann_return"]:>+7.1%}{r["ann_vol"]:>8.1%}'
              f'{r["sharpe"]:>8.2f}{r["max_dd"]:>8.1%}'
              f'{r["turnover"]:>10.4f}{r["corr_bench"]:>11.2f}')
    print()
    print(f'Best return:     {max(rows, key=lambda x: x["ann_return"])["preset"]} '
          f'({max(rows, key=lambda x: x["ann_return"])["ann_return"]:+.1%}/yr)')
    print(f'Best Sharpe:     {max(rows, key=lambda x: x["sharpe"])["preset"]} '
          f'(Sharpe {max(rows, key=lambda x: x["sharpe"])["sharpe"]:.2f})')
    best_dd = max(rows, key=lambda x: x['max_dd'])
    print(f'Lowest drawdown: {best_dd["preset"]} (max DD {best_dd["max_dd"]:.1%})')
    lowest_to = min(rows, key=lambda x: x['turnover'])
    print(f'Lowest turnover: {lowest_to["preset"]} (turnover {lowest_to["turnover"]:.4f})')
    print()
    for pname in COMPARISON_PRESET_NAMES:
        print(MOMENTUM_INTERPRETATION.get(pname, ''))


def run_single_stock_momentum_comparison(
    price: pd.Series,
    cost: float = 0.0005,
) -> list[dict]:
    """Backtest each momentum preset on one ticker plus buy-and-hold."""
    model = MomentumModel()
    rows = []
    for pname in COMPARISON_PRESET_NAMES:
        p = MOMENTUM_PRESETS[pname]
        params = {'lookback': p['lookback'], 'skip': p['skip']}
        df, n_trades = model.backtest(price, cost=cost, **params)
        m = metrics(df['strat_net'])
        rows.append({
            'label': pname,
            'lookback': p['lookback'],
            'skip': p['skip'],
            'ann_return': m['ann_return'],
            'ann_vol': m['ann_vol'],
            'sharpe': m['sharpe'],
            'max_dd': m['max_dd'],
            'n_trades': n_trades,
        })
    hold_df, _ = model.backtest(price, cost=0.0)
    hold_m = metrics(hold_df['ret'])
    rows.append({
        'label': 'Buy & hold',
        'lookback': None,
        'skip': None,
        'ann_return': hold_m['ann_return'],
        'ann_vol': hold_m['ann_vol'],
        'sharpe': hold_m['sharpe'],
        'max_dd': hold_m['max_dd'],
        'n_trades': 0,
    })
    return rows


def print_single_stock_momentum_comparison(
    rows: list[dict],
    ticker: str,
    years: int,
) -> None:
    print(f'\n=== Single-Stock Momentum Lookback Comparison ===')
    print(f'Ticker: {ticker}')
    print(f'Years: {years}')
    print()
    print(f'{"Model":<18}{"Lookback":>9}{"Skip":>6}{"AnnRet":>8}{"Vol":>8}'
          f'{"Sharpe":>8}{"MaxDD":>8}{"Trades":>8}')
    print('-' * 74)
    for r in rows:
        lb = f'{r["lookback"]:>9}' if r['lookback'] is not None else f'{"—":>9}'
        sk = f'{r["skip"]:>6}' if r['skip'] is not None else f'{"—":>6}'
        print(f'{r["label"]:<18}{lb}{sk}'
              f'{r["ann_return"]:>+7.1%}{r["ann_vol"]:>8.1%}'
              f'{r["sharpe"]:>8.2f}{r["max_dd"]:>8.1%}'
              f'{r["n_trades"]:>8}')
    print()
    mom_rows = [r for r in rows if r['label'] != 'Buy & hold']
    if mom_rows:
        print(f'Best return:  {max(mom_rows, key=lambda x: x["ann_return"])["label"]}')
        print(f'Best Sharpe:  {max(mom_rows, key=lambda x: x["sharpe"])["label"]}')
        print(f'Lowest drawdown: {max(mom_rows, key=lambda x: x["max_dd"])["label"]}')


def build_momentum_preset_rank_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Latest cross-sectional momentum score and rank for each comparison preset."""
    from quant.models.cross_sectional import compute_scores

    preset_scores: dict[str, pd.Series] = {}
    preset_ranks: dict[str, pd.Series] = {}
    for pname in COMPARISON_PRESET_NAMES:
        p = MOMENTUM_PRESETS[pname]
        scores = compute_scores(
            panel, mode='momentum', lookback=p['lookback'], skip=p['skip'],
        ).iloc[-1].dropna()
        preset_scores[pname] = scores
        preset_ranks[pname] = scores.rank(ascending=False, method='min')

    rows = []
    for ticker in panel.columns:
        row: dict = {'ticker': ticker}
        for pname in COMPARISON_PRESET_NAMES:
            row[f'{pname}_score'] = float(preset_scores[pname].get(ticker, float('nan')))
            rank = preset_ranks[pname].get(ticker)
            row[f'{pname}_rank'] = int(rank) if rank is not None and not pd.isna(rank) else None
        rows.append(row)
    return pd.DataFrame(rows)


def print_momentum_preset_rank_table(df: pd.DataFrame) -> None:
    print('\n=== Current Momentum Ranks by Preset ===')
    print(
        'Interpretation: strong in 10d but weak in 126d = recent bounce, not durable trend; '
        'strong in both 20d and 126d = stronger confirmation; '
        'weak short-term but strong medium-term = pullback in a longer trend.'
    )
    print()
    header = f'{"Ticker":<7}'
    for pname in COMPARISON_PRESET_NAMES:
        short = pname.replace('mom_', '')
        header += f'{short:>10}{"Rank":>6}'
    print(header)
    print('-' * (7 + len(COMPARISON_PRESET_NAMES) * 16))
    for _, r in df.iterrows():
        line = f'{r["ticker"]:<7}'
        for pname in COMPARISON_PRESET_NAMES:
            sc = r[f'{pname}_score']
            rk = r[f'{pname}_rank']
            sc_s = f'{sc:+.1%}' if not pd.isna(sc) else '   n/a'
            rk_s = f'{rk:>5}' if rk is not None else '    —'
            line += f'{sc_s:>10}{rk_s:>6}'
        print(line)
