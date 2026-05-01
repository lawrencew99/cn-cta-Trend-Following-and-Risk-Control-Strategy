import sys
from types import SimpleNamespace

import pandas as pd

from cn_cta.data import AkShareDataAdapter, MarketDataRequest, normalize_akshare_ohlcv


def test_normalize_akshare_chinese_columns() -> None:
    raw = pd.DataFrame(
        {
            "日期": ["2024-01-02"],
            "开盘": [1.0],
            "最高": [1.1],
            "最低": [0.9],
            "收盘": [1.05],
            "成交量": [1000],
        }
    )

    data = normalize_akshare_ohlcv(raw, "510300")

    assert list(data.columns) == ["date", "symbol", "open", "high", "low", "close", "volume"]
    assert data.loc[0, "symbol"] == "510300"


def test_akshare_etf_adapter_uses_standard_schema(monkeypatch) -> None:
    def fund_etf_hist_em(**kwargs):
        assert kwargs["symbol"] == "510300"
        assert kwargs["start_date"] == "20240101"
        assert kwargs["end_date"] == "20240131"
        return pd.DataFrame(
            {
                "日期": ["2024-01-02", "2024-01-03"],
                "开盘": [1.0, 1.05],
                "最高": [1.1, 1.2],
                "最低": [0.9, 1.0],
                "收盘": [1.05, 1.1],
                "成交量": [1000, 1200],
            }
        )

    monkeypatch.setitem(sys.modules, "akshare", SimpleNamespace(fund_etf_hist_em=fund_etf_hist_em))
    adapter = AkShareDataAdapter(asset_type="etf")

    data = adapter.load_ohlcv(MarketDataRequest(["510300"], "2024-01-01", "2024-01-31"))

    assert data["date"].min() == pd.Timestamp("2024-01-02")
    assert data["close"].iloc[-1] == 1.1


def test_akshare_futures_adapter_filters_dates(monkeypatch) -> None:
    def futures_zh_daily_sina(**kwargs):
        assert kwargs["symbol"] == "IF0"
        return pd.DataFrame(
            {
                "date": ["2023-12-29", "2024-01-02", "2024-01-03"],
                "open": [1.0, 1.1, 1.2],
                "high": [1.1, 1.2, 1.3],
                "low": [0.9, 1.0, 1.1],
                "close": [1.05, 1.15, 1.25],
                "volume": [1000, 1200, 1300],
            }
        )

    monkeypatch.setitem(sys.modules, "akshare", SimpleNamespace(futures_zh_daily_sina=futures_zh_daily_sina))
    adapter = AkShareDataAdapter(asset_type="futures")

    data = adapter.load_ohlcv(MarketDataRequest(["IF0"], "2024-01-01", "2024-01-02"))

    assert list(data["date"]) == [pd.Timestamp("2024-01-02")]
