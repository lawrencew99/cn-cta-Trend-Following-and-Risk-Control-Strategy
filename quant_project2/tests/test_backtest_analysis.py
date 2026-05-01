import pandas as pd
import pytest

from cn_cta.analysis import monte_carlo_resample, performance_summary, return_attribution, stress_test
from cn_cta.backtest import BacktestConfig, run_backtest


def test_backtest_applies_one_day_delay_and_costs() -> None:
    close = pd.DataFrame({"ETF": [100.0, 101.0, 102.0]}, index=pd.date_range("2024-01-01", periods=3))
    positions = pd.DataFrame({"ETF": [1.0, 1.0, 1.0]}, index=close.index)

    result = run_backtest(close, positions, BacktestConfig(commission_bps=0.0, slippage_bps=0.0))

    assert result.returns.iloc[0] == 0.0
    assert result.returns.iloc[1] == pytest.approx(0.01)
    assert result.equity_curve.iloc[-1] > result.equity_curve.iloc[0]


def test_analysis_outputs_are_populated() -> None:
    returns = pd.Series([0.01, -0.02, 0.015, -0.005, 0.012])
    pnl = pd.DataFrame({"A": [0.01, -0.01, 0.005, 0.0, 0.004]})
    positions = pd.DataFrame({"A": [1.0, 1.0, 0.5, 0.0, 1.0]})
    costs = pd.Series([0.0, 0.001, 0.001, 0.0, 0.001])

    summary = performance_summary(returns)
    attribution = return_attribution(pnl, positions, costs)
    stressed = stress_test(returns, {"shock": -0.01})
    paths = monte_carlo_resample(returns, n_paths=20, seed=1)

    assert "sharpe" in summary
    assert "net_return" in attribution.columns
    assert "shock" in stressed.index
    assert len(paths) == 20
