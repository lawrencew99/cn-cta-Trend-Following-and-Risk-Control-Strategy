"""AKShare data adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Literal

import pandas as pd

from cn_cta.data.adapters import MarketDataRequest
from cn_cta.data.schema import validate_ohlcv

AssetType = Literal["etf", "futures"]


@dataclass(frozen=True)
class AkShareDataAdapter:
    """Load daily ETF or futures data through AKShare.

    `asset_type="etf"` uses `fund_etf_hist_em`.
    `asset_type="futures"` uses `futures_zh_daily_sina`, which is suitable for
    common main/continuous symbols such as `IF0` or `RB0`.
    """

    asset_type: AssetType = "etf"
    adjust: str = "qfq"
    api_kwargs: dict[str, Any] = field(default_factory=dict)

    def load_ohlcv(self, request: MarketDataRequest) -> pd.DataFrame:
        """Fetch and normalize OHLCV data from AKShare."""

        if request.frequency != "1d":
            raise ValueError("AkShareDataAdapter currently supports daily frequency only")

        frames = [self._load_symbol(symbol, request) for symbol in request.symbols]
        if not frames:
            raise ValueError("request.symbols must not be empty")
        return validate_ohlcv(pd.concat(frames, ignore_index=True))

    def _load_symbol(self, symbol: str, request: MarketDataRequest) -> pd.DataFrame:
        akshare = import_module("akshare")
        start_date = _format_akshare_date(request.start)
        end_date = _format_akshare_date(request.end)

        if self.asset_type == "etf":
            raw = akshare.fund_etf_hist_em(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=self.adjust,
                **self.api_kwargs,
            )
        elif self.asset_type == "futures":
            raw = akshare.futures_zh_daily_sina(symbol=symbol, **self.api_kwargs)
        else:
            raise ValueError(f"Unsupported AKShare asset_type: {self.asset_type}")

        normalized = normalize_akshare_ohlcv(raw, symbol)
        if self.asset_type == "futures":
            start = pd.to_datetime(request.start)
            end = pd.to_datetime(request.end)
            normalized = normalized[(normalized["date"] >= start) & (normalized["date"] <= end)]
        return normalized


def normalize_akshare_ohlcv(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Normalize common AKShare OHLCV column names into project schema."""

    if raw.empty:
        raise ValueError(f"AKShare returned empty data for {symbol}")

    frame = raw.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    rename_map = _resolve_columns(frame.columns)
    missing_targets = {"date", "open", "high", "low", "close", "volume"} - set(rename_map.values())
    if missing_targets:
        raise ValueError(f"Cannot map AKShare columns to OHLCV schema: {sorted(missing_targets)}")

    frame = frame.rename(columns=rename_map)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["symbol"] = symbol
    return frame[["date", "symbol", "open", "high", "low", "close", "volume"]]


def _resolve_columns(columns: pd.Index) -> dict[str, str]:
    aliases = {
        "date": {"date", "日期", "时间"},
        "open": {"open", "开盘", "开盘价"},
        "high": {"high", "最高", "最高价"},
        "low": {"low", "最低", "最低价"},
        "close": {"close", "收盘", "收盘价"},
        "volume": {"volume", "成交量", "vol"},
    }
    resolved: dict[str, str] = {}
    lowered = {str(column).lower(): column for column in columns}

    for target, names in aliases.items():
        for name in names:
            column = lowered.get(name.lower())
            if column is not None:
                resolved[str(column)] = target
                break
    return resolved


def _format_akshare_date(value: object) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")
