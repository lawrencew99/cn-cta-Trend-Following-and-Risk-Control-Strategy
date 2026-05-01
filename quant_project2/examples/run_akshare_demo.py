"""Run the CTA workflow with AKShare daily data."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cn_cta.analysis import performance_summary
from cn_cta.backtest import BacktestConfig, run_backtest
from cn_cta.data import AkShareDataAdapter, MarketDataRequest
from cn_cta.risk import volatility_target_positions
from cn_cta.signals import donchian_breakout


def main() -> None:
    adapter = AkShareDataAdapter(asset_type="etf", adjust="qfq")
    data = adapter.load_ohlcv(
        MarketDataRequest(
            symbols=["510300"],
            start="2021-01-01",
            end="2024-12-31",
        )
    )
    close = data.pivot(index="date", columns="symbol", values="close")

    signals = donchian_breakout(data, entry_window=55, exit_window=20)
    positions = volatility_target_positions(signals, close, target_vol=0.12, max_leverage=1.0)
    result = run_backtest(close, positions, BacktestConfig(commission_bps=2.0, slippage_bps=1.0))

    print("AKShare symbols:", ", ".join(close.columns))
    print(pd.Series(performance_summary(result.returns)))


if __name__ == "__main__":
    main()
