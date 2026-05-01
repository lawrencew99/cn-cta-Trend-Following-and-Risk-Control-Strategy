"""Synthetic China-market-like daily data for demos and tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cn_cta.data.schema import validate_ohlcv


def make_sample_ohlcv(
    symbols: list[str] | tuple[str, ...] = ("510300.SH", "IF.CFE", "RB.SHF"),
    periods: int = 420,
    start: str = "2023-01-03",
    seed: int = 7,
) -> pd.DataFrame:
    """Create deterministic OHLCV data with trends and volatility clusters."""

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=periods)
    rows: list[pd.DataFrame] = []

    for index, symbol in enumerate(symbols):
        drift = 0.00015 + index * 0.00005
        volatility = 0.011 + index * 0.003
        cycle = np.sin(np.linspace(0, 8 * np.pi, periods)) * 0.002
        shocks = rng.normal(drift + cycle, volatility, periods)
        close = 100 * np.exp(np.cumsum(shocks))

        intraday_range = rng.uniform(0.003, 0.025, periods)
        open_ = close * (1 + rng.normal(0, 0.003, periods))
        high = np.maximum(open_, close) * (1 + intraday_range)
        low = np.minimum(open_, close) * (1 - intraday_range)
        volume = rng.integers(100_000, 2_000_000, periods)

        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": symbol,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "paused": False,
                    "multiplier": 1.0 if symbol.endswith(".SH") else 300.0,
                    "margin_rate": 1.0 if symbol.endswith(".SH") else 0.12,
                }
            )
        )

    return validate_ohlcv(pd.concat(rows, ignore_index=True))
