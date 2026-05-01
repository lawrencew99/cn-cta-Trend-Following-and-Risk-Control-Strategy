"""Run a complete CTA research workflow on synthetic China-market data."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cn_cta.analysis import monte_carlo_resample, performance_summary, return_attribution, stress_test
from cn_cta.backtest import BacktestConfig, run_backtest
from cn_cta.data import make_sample_ohlcv
from cn_cta.risk import apply_drawdown_control, apply_trailing_stop, atr, volatility_target_positions
from cn_cta.signals import donchian_breakout


def main() -> None:
    data = make_sample_ohlcv()
    close = data.pivot(index="date", columns="symbol", values="close")

    signals = donchian_breakout(data, entry_window=55, exit_window=20)
    base_positions = volatility_target_positions(signals, close, target_vol=0.12, max_leverage=1.2)
    stopped_positions = apply_trailing_stop(base_positions, close, atr(data, window=14), atr_multiple=3.0)

    config = BacktestConfig(commission_bps=2.0, slippage_bps=1.0)
    preliminary = run_backtest(close, stopped_positions, config)
    controlled_positions = apply_drawdown_control(
        stopped_positions,
        preliminary.returns,
        max_drawdown=0.12,
        reduced_leverage=0.5,
    )
    result = run_backtest(close, controlled_positions, config)

    summary = performance_summary(result.returns)
    attribution = return_attribution(result.pnl_by_symbol, result.positions, result.costs)
    stress = stress_test(result.returns, {"one_day_liquidity_shock": -0.02, "mild_rebound": 0.005})
    mc = monte_carlo_resample(result.returns, n_paths=500, seed=11)

    pd.set_option("display.precision", 4)
    print("Performance summary")
    print(pd.Series(summary))
    print("\nReturn attribution")
    print(attribution)
    print("\nStress test")
    print(stress[["annual_return", "max_drawdown", "var_95", "cvar_95"]])
    print("\nMonte Carlo quantiles")
    print(mc.quantile([0.05, 0.5, 0.95]))


if __name__ == "__main__":
    main()
