"""Position-level risk controls."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _wide(data: pd.DataFrame, value: str) -> pd.DataFrame:
    if {"date", "symbol", value}.issubset(data.columns):
        return data.pivot(index="date", columns="symbol", values=value).sort_index()
    return data.sort_index()


def atr(data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Average true range for each symbol."""

    if window <= 1:
        raise ValueError("window must be greater than 1")

    high = _wide(data, "high")
    low = _wide(data, "low")
    close = _wide(data, "close")
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            (high - low).stack(),
            (high - prev_close).abs().stack(),
            (low - prev_close).abs().stack(),
        ],
        axis=1,
    ).max(axis=1)
    true_range = true_range.unstack()
    return true_range.rolling(window, min_periods=window).mean()


def volatility_target_positions(
    signals: pd.DataFrame,
    close: pd.DataFrame,
    target_vol: float = 0.15,
    lookback: int = 60,
    max_leverage: float = 1.5,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Scale directional signals to a target annualized volatility."""

    if target_vol <= 0 or max_leverage <= 0:
        raise ValueError("target_vol and max_leverage must be positive")
    if lookback <= 1:
        raise ValueError("lookback must be greater than 1")

    returns = close.pct_change()
    realized_vol = returns.rolling(lookback, min_periods=lookback).std() * np.sqrt(periods_per_year)
    leverage = (target_vol / realized_vol).clip(upper=max_leverage)
    return (signals.reindex_like(close).fillna(0.0) * leverage).fillna(0.0)


def apply_trailing_stop(
    positions: pd.DataFrame,
    close: pd.DataFrame,
    stop_atr: pd.DataFrame,
    atr_multiple: float = 3.0,
) -> pd.DataFrame:
    """Set positions to zero after an ATR-based trailing stop is hit."""

    if atr_multiple <= 0:
        raise ValueError("atr_multiple must be positive")

    adjusted = positions.reindex_like(close).fillna(0.0).copy()
    for symbol in adjusted.columns:
        active_side = 0.0
        high_watermark = np.nan
        low_watermark = np.nan

        for idx in adjusted.index:
            side = np.sign(adjusted.at[idx, symbol])
            price = close.at[idx, symbol]
            band = stop_atr.at[idx, symbol] * atr_multiple if symbol in stop_atr.columns else np.nan

            if side == 0 or pd.isna(price) or pd.isna(band):
                if side == 0:
                    active_side = 0.0
                continue

            if side != active_side:
                active_side = side
                high_watermark = price
                low_watermark = price

            high_watermark = max(high_watermark, price)
            low_watermark = min(low_watermark, price)
            stop_hit = (side > 0 and price <= high_watermark - band) or (
                side < 0 and price >= low_watermark + band
            )
            if stop_hit:
                adjusted.at[idx, symbol] = 0.0
                active_side = 0.0

    return adjusted


def apply_drawdown_control(
    positions: pd.DataFrame,
    portfolio_returns: pd.Series,
    max_drawdown: float = 0.12,
    reduced_leverage: float = 0.5,
) -> pd.DataFrame:
    """Reduce future exposure when realized drawdown breaches a threshold."""

    if not 0 < max_drawdown < 1:
        raise ValueError("max_drawdown must be between 0 and 1")
    if not 0 <= reduced_leverage <= 1:
        raise ValueError("reduced_leverage must be between 0 and 1")

    equity = (1 + portfolio_returns.fillna(0.0)).cumprod()
    drawdown = equity / equity.cummax() - 1
    scale = pd.Series(1.0, index=positions.index)
    scale.loc[drawdown.reindex(positions.index).fillna(0.0) <= -max_drawdown] = reduced_leverage
    return positions.mul(scale.shift(1).fillna(1.0), axis=0)
