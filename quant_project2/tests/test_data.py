import pandas as pd
import pytest

from cn_cta.data import make_sample_ohlcv, validate_ohlcv


def test_sample_data_validates() -> None:
    data = make_sample_ohlcv(symbols=("510300.SH",), periods=30)

    validated = validate_ohlcv(data)

    assert list(validated["symbol"].unique()) == ["510300.SH"]
    assert validated["date"].is_monotonic_increasing


def test_validate_rejects_duplicate_symbol_date() -> None:
    row = {
        "date": "2024-01-02",
        "symbol": "IF.CFE",
        "open": 1.0,
        "high": 1.1,
        "low": 0.9,
        "close": 1.0,
        "volume": 100,
    }
    data = pd.DataFrame([row, row])

    with pytest.raises(ValueError, match="duplicated"):
        validate_ohlcv(data)
