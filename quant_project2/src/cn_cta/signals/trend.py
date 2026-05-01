"""Daily trend-following signals."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _to_close_frame(data: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    if {"date", "symbol", price_col}.issubset(data.columns):
        return data.pivot(index="date", columns="symbol", values=price_col).sort_index()
    return data.sort_index()


def _direction(condition_long: pd.DataFrame, condition_short: pd.DataFrame) -> pd.DataFrame:
    signal = pd.DataFrame(0.0, index=condition_long.index, columns=condition_long.columns)
    signal = signal.mask(condition_long, 1.0)
    signal = signal.mask(condition_short, -1.0)
    return signal.fillna(0.0)


def moving_average_breakout(
    data: pd.DataFrame,
    fast_window: int = 20,
    slow_window: int = 60,
    price_col: str = "close",
) -> pd.DataFrame:
    """Long when fast moving average is above slow average, short otherwise."""

    if fast_window <= 0 or slow_window <= 0:
        raise ValueError("moving average windows must be positive")
    if fast_window >= slow_window:
        raise ValueError("fast_window must be smaller than slow_window")

    close = _to_close_frame(data, price_col)
    fast = close.rolling(fast_window, min_periods=fast_window).mean()
    slow = close.rolling(slow_window, min_periods=slow_window).mean()
    return _direction(fast > slow, fast < slow)


def donchian_breakout(
    data: pd.DataFrame,
    entry_window: int = 55,
    exit_window: int | None = 20,
    price_col: str = "close",
) -> pd.DataFrame:
    """Donchian channel breakout with optional exit channel.

    The channel is shifted by one day to avoid using the current close when
    forming today's signal.
    """

    if entry_window <= 1:
        raise ValueError("entry_window must be greater than 1")
    if exit_window is not None and exit_window <= 1:
        raise ValueError("exit_window must be greater than 1")

    close = _to_close_frame(data, price_col)
    upper = close.rolling(entry_window, min_periods=entry_window).max().shift(1)
    lower = close.rolling(entry_window, min_periods=entry_window).min().shift(1)

    if exit_window is None:
        raw = _direction(close > upper, close < lower)
        return raw.replace(0.0, np.nan).ffill().fillna(0.0)

    exit_high = close.rolling(exit_window, min_periods=exit_window).max().shift(1)
    exit_low = close.rolling(exit_window, min_periods=exit_window).min().shift(1)

    position = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for symbol in close.columns:
        state = 0.0
        for idx in close.index:
            price = close.at[idx, symbol]
            if pd.isna(price):
                position.at[idx, symbol] = state
                continue

            if state > 0 and price < exit_low.at[idx, symbol]:
                state = 0.0
            elif state < 0 and price > exit_high.at[idx, symbol]:
                state = 0.0

            if price > upper.at[idx, symbol]:
                state = 1.0
            elif price < lower.at[idx, symbol]:
                state = -1.0

            position.at[idx, symbol] = state
    return position


def volatility_breakout(
    data: pd.DataFrame,
    lookback: int = 20,
    threshold: float = 1.5,
    price_col: str = "close",
) -> pd.DataFrame:
    """Signal when daily return exceeds a multiple of recent volatility."""

    if lookback <= 1:
        raise ValueError("lookback must be greater than 1")
    if threshold <= 0:
        raise ValueError("threshold must be positive")

    close = _to_close_frame(data, price_col)
    returns = close.pct_change()
    realized_vol = returns.rolling(lookback, min_periods=lookback).std()
    return _direction(returns > threshold * realized_vol, returns < -threshold * realized_vol)
