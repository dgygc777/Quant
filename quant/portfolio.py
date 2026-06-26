from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

DEFAULT_PORTFOLIO = Path(__file__).parent.parent / 'paper_portfolio.json'
DEFAULT_CASH = 100_000.0
DEFAULT_COST = 0.0005


def load_portfolio(path: Path = DEFAULT_PORTFOLIO) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text())
    return {
        'cash': DEFAULT_CASH,
        'initial_cash': DEFAULT_CASH,
        'positions': {},
        'strategy_state': {},
        'trades': [],
    }


def save_portfolio(portfolio: dict[str, Any], path: Path = DEFAULT_PORTFOLIO) -> None:
    path.write_text(json.dumps(portfolio, indent=2, default=str))


def position_for(portfolio: dict, ticker: str) -> dict:
    return portfolio['positions'].get(ticker.upper(), {'shares': 0.0, 'avg_cost': 0.0})


def model_state_key(ticker: str, model_slug: str) -> str:
    return f'{ticker.upper()}:{model_slug}'


def get_model_state(portfolio: dict, ticker: str, model_slug: str) -> dict:
    return portfolio['strategy_state'].get(model_state_key(ticker, model_slug), {})


def set_model_state(portfolio: dict, ticker: str, model_slug: str, state: dict) -> None:
    portfolio['strategy_state'][model_state_key(ticker, model_slug)] = state


def paper_buy(portfolio: dict, ticker: str, price: float,
              shares: Optional[float] = None, reason: str = 'manual',
              cost: float = DEFAULT_COST) -> dict:
    ticker = ticker.upper()
    pos = position_for(portfolio, ticker)
    if shares is None:
        shares = int(portfolio['cash'] / price)
    if shares <= 0:
        raise ValueError('Not enough cash to buy even 1 share.')
    total_cost = shares * price * (1 + cost)
    if total_cost > portfolio['cash']:
        shares = int(portfolio['cash'] / (price * (1 + cost)))
        total_cost = shares * price * (1 + cost)
    if shares <= 0:
        raise ValueError('Not enough cash after transaction costs.')

    old_shares, old_cost = pos['shares'], pos['avg_cost']
    new_shares = old_shares + shares
    new_avg = ((old_shares * old_cost) + (shares * price)) / new_shares

    portfolio['cash'] -= total_cost
    portfolio['positions'][ticker] = {'shares': new_shares, 'avg_cost': new_avg}
    portfolio['trades'].append({
        'time': datetime.now(timezone.utc).isoformat(),
        'ticker': ticker,
        'side': 'BUY',
        'shares': shares,
        'price': price,
        'cost': total_cost,
        'reason': reason,
    })
    return portfolio


def paper_sell(portfolio: dict, ticker: str, price: float,
               shares: Optional[float] = None, reason: str = 'manual',
               cost: float = DEFAULT_COST) -> dict:
    ticker = ticker.upper()
    pos = position_for(portfolio, ticker)
    if shares is None:
        shares = pos['shares']
    shares = min(shares, pos['shares'])
    if shares <= 0:
        raise ValueError(f'No shares to sell for {ticker}.')

    proceeds = shares * price * (1 - cost)
    remaining = pos['shares'] - shares
    if remaining <= 1e-9:
        portfolio['positions'].pop(ticker, None)
    else:
        portfolio['positions'][ticker] = {'shares': remaining, 'avg_cost': pos['avg_cost']}

    portfolio['cash'] += proceeds
    portfolio['trades'].append({
        'time': datetime.now(timezone.utc).isoformat(),
        'ticker': ticker,
        'side': 'SELL',
        'shares': shares,
        'price': price,
        'proceeds': proceeds,
        'reason': reason,
    })
    return portfolio
