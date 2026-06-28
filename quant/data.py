from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

from quant.data_quality import format_clipped_returns, winsorize_extreme_returns


def fetch_daily_prices(ticker: str, lookback_days: int = 120) -> pd.Series:
    """Download adjusted daily close prices."""
    start = (pd.Timestamp.now() - pd.DateOffset(days=lookback_days)).strftime('%Y-%m-%d')
    data = yf.download(
        ticker,
        start=start,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if data.empty:
        raise ValueError(f'No data returned for {ticker}. Check the ticker symbol.')
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    close = data['Close'].dropna()
    close.name = ticker.upper()
    return close


def parse_universe(text: str) -> list[str]:
    """Parse comma-separated ticker list."""
    return [t.strip().upper() for t in text.split(',') if t.strip()]


def fetch_panel(tickers: list[str], years: int) -> pd.DataFrame:
    """Download multi-year daily close panel for cross-sectional models."""
    tickers = [t.upper() for t in tickers]
    data = yf.download(
        tickers,
        start=(pd.Timestamp.now() - pd.DateOffset(years=years)).strftime('%Y-%m-%d'),
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if data.empty:
        raise ValueError(f'No data returned for universe: {", ".join(tickers)}')
    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close'].dropna(how='all')
    else:
        close = pd.DataFrame({tickers[0]: data['Close']}).dropna(how='all')
    close.columns = [str(c).upper() for c in close.columns]
    if close.shape[1] < 2:
        raise ValueError('Cross-sectional backtest needs at least 2 tickers with data.')
    close, clipped = winsorize_extreme_returns(close)
    if not clipped.empty:
        print('Clipped extreme daily returns (>35%): ' + format_clipped_returns(clipped))
    return close


def fetch_historical_prices(ticker: str, years: int) -> pd.Series:
    """Download multi-year daily close for backtesting."""
    data = yf.download(
        ticker,
        start=(pd.Timestamp.now() - pd.DateOffset(years=years)).strftime('%Y-%m-%d'),
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if data.empty:
        raise ValueError(f'No historical data for {ticker}')
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    return data['Close'].dropna()


def fetch_live_quote(ticker: str) -> tuple[float, str]:
    """Latest tradeable price and timestamp from Yahoo."""
    t = yf.Ticker(ticker)
    info = t.fast_info
    price = info.get('lastPrice') or info.get('regularMarketPrice')
    if price is None or (isinstance(price, float) and np.isnan(price)):
        hist = t.history(period='1d', interval='1m', auto_adjust=True)
        if hist.empty:
            raise ValueError(f'Cannot get live quote for {ticker}')
        price = float(hist['Close'].iloc[-1])
        ts = hist.index[-1].isoformat()
    else:
        price = float(price)
        ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    return price, ts


def append_live_price(prices: pd.Series, live_price: float) -> pd.Series:
    """Append or update today's bar with the latest quote."""
    today = pd.Timestamp.now().normalize()
    if prices.index[-1].normalize() >= today:
        prices = prices.copy()
        prices.iloc[-1] = live_price
    else:
        prices = pd.concat([prices, pd.Series({today: live_price})])
        prices.name = prices.name or 'price'
    return prices


def build_live_frame(price: pd.Series, model, **params) -> tuple[pd.DataFrame, float, str]:
    """History + live quote, with model indicators computed."""
    ticker = price.name or 'TICKER'
    live_price, quote_ts = fetch_live_quote(ticker)
    prices = append_live_price(price, live_price)
    df = model.compute_indicators(prices, **params)
    return df.dropna(), live_price, quote_ts
