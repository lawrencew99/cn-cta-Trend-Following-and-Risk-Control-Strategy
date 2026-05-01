"""Data adapter interfaces for China-market OHLCV data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol, Sequence

import pandas as pd


@dataclass(frozen=True)
class MarketDataRequest:
    """Request object consumed by market data adapters."""

    symbols: Sequence[str]
    start: date | str
    end: date | str
    frequency: str = "1d"
    fields: Sequence[str] | None = None


class MarketDataAdapter(Protocol):
    """Protocol for user-provided data sources.

    Implementations can wrap local CSV/Parquet files, vendor APIs, AkShare,
    Tushare, Wind, JoinQuant, or an internal research database.
    """

    def load_ohlcv(self, request: MarketDataRequest) -> pd.DataFrame:
        """Return normalized OHLCV data for the requested symbols."""
