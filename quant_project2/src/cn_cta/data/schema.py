"""Schema validation for normalized daily market data."""

from __future__ import annotations

import pandas as pd

OHLCV_COLUMNS = ("date", "symbol", "open", "high", "low", "close", "volume")
OPTIONAL_CHINA_COLUMNS = (
    "limit_up",
    "limit_down",
    "paused",
    "multiplier",
    "margin_rate",
)


def validate_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize daily OHLCV data.

    The returned frame is sorted by `symbol` and `date`, with `date` converted
    to pandas datetime. The function raises `ValueError` for missing columns,
    duplicated symbol/date rows, invalid prices, or rows where high/low are
    inconsistent with open/close.
    """

    missing = [column for column in OHLCV_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")

    frame = data.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["symbol"] = frame["symbol"].astype(str)

    numeric_columns = ["open", "high", "low", "close", "volume"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    if frame.duplicated(["date", "symbol"]).any():
        raise ValueError("OHLCV data contains duplicated date/symbol rows")

    price_columns = ["open", "high", "low", "close"]
    if (frame[price_columns] <= 0).any().any():
        raise ValueError("OHLCV prices must be positive")
    if (frame["volume"] < 0).any():
        raise ValueError("volume must be non-negative")

    max_open_close = frame[["open", "close"]].max(axis=1)
    min_open_close = frame[["open", "close"]].min(axis=1)
    if (frame["high"] < max_open_close).any() or (frame["low"] > min_open_close).any():
        raise ValueError("high/low columns are inconsistent with open/close")

    if "paused" in frame.columns:
        frame["paused"] = frame["paused"].fillna(False).astype(bool)

    return frame.sort_values(["symbol", "date"]).reset_index(drop=True)
