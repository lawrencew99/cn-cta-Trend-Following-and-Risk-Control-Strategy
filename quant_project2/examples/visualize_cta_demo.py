"""Visualize the CTA demo workflow and save the report as a PNG."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cn_cta.analysis import monte_carlo_resample, performance_summary, return_attribution, stress_test
from cn_cta.backtest import BacktestConfig, BacktestResult, run_backtest
from cn_cta.data import make_sample_ohlcv
from cn_cta.risk import apply_drawdown_control, apply_trailing_stop, atr, volatility_target_positions
from cn_cta.signals import donchian_breakout


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "outputs" / "cta_demo_visualization.png"


@dataclass(frozen=True)
class DemoOutputs:
    """Container for numerical outputs used by the visualization."""

    result: BacktestResult
    summary: dict[str, float]
    attribution: pd.DataFrame
    stress: pd.DataFrame
    monte_carlo: pd.DataFrame


def run_demo_backtest(mc_paths: int = 500) -> DemoOutputs:
    """Run the same synthetic CTA workflow used by the text demo."""

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

    return DemoOutputs(
        result=result,
        summary=performance_summary(result.returns),
        attribution=return_attribution(result.pnl_by_symbol, result.positions, result.costs),
        stress=stress_test(result.returns, {"one_day_liquidity_shock": -0.02, "mild_rebound": 0.005}),
        monte_carlo=monte_carlo_resample(result.returns, n_paths=mc_paths, seed=11),
    )


def _format_pct(value: float) -> str:
    return f"{value:.1%}"


def plot_strategy_report(outputs: DemoOutputs, output: Path, show: bool = False) -> None:
    """Create a multi-panel report and save it to a PNG file."""

    result = outputs.result
    equity = result.equity_curve
    drawdown = equity / equity.cummax() - 1

    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(16, 13), constrained_layout=True)
    fig.suptitle(
        "CTA Demo Visualization "
        f"(Ann. Return {_format_pct(outputs.summary['annual_return'])}, "
        f"Sharpe {outputs.summary['sharpe']:.2f}, "
        f"Max DD {_format_pct(outputs.summary['max_drawdown'])})",
        fontsize=14,
    )

    ax = axes[0, 0]
    equity.plot(ax=ax, color="tab:blue")
    ax.set_title("Equity Curve")
    ax.set_ylabel("Capital")
    ax.grid(True, alpha=0.3)
    ax.legend(["Portfolio"], loc="upper left")

    ax = axes[0, 1]
    drawdown.plot(ax=ax, color="tab:red")
    ax.fill_between(drawdown.index, drawdown, 0, color="tab:red", alpha=0.15)
    ax.set_title("Portfolio Drawdown")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    result.returns.plot.hist(ax=ax, bins=40, color="tab:purple", alpha=0.75)
    ax.axvline(-outputs.summary["var_95"], color="tab:orange", linestyle="--", label="VaR 95%")
    ax.axvline(-outputs.summary["cvar_95"], color="tab:red", linestyle="--", label="CVaR 95%")
    ax.set_title("Daily Return Distribution")
    ax.set_xlabel("Daily return")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.1%}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    outputs.attribution["net_return"].plot.bar(ax=ax, color="tab:green")
    ax.set_title("Net Return Attribution")
    ax.set_ylabel("Net return")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[2, 0]
    result.positions.plot(ax=ax)
    ax.set_title("Position Exposure")
    ax.set_ylabel("Leverage")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    ax = axes[2, 1]
    ax.scatter(
        outputs.monte_carlo["max_drawdown"],
        outputs.monte_carlo["terminal_return"],
        s=12,
        alpha=0.45,
        color="tab:cyan",
        edgecolors="none",
    )
    ax.axhline(outputs.monte_carlo["terminal_return"].median(), color="tab:blue", linestyle="--", label="Median terminal")
    ax.axvline(outputs.monte_carlo["max_drawdown"].median(), color="tab:red", linestyle="--", label="Median drawdown")
    ax.set_title("Monte Carlo Outcomes")
    ax.set_xlabel("Max drawdown")
    ax.set_ylabel("Terminal return")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(output, dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize the CTA demo workflow and save a PNG report.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="PNG output path.")
    parser.add_argument("--show", action="store_true", help="Show the chart window after saving.")
    parser.add_argument("--mc-paths", type=int, default=500, help="Number of Monte Carlo bootstrap paths.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mc_paths <= 0:
        raise ValueError("--mc-paths must be positive")

    outputs = run_demo_backtest(mc_paths=args.mc_paths)
    plot_strategy_report(outputs, args.output, show=args.show)
    print(f"Saved visualization to {args.output}")


if __name__ == "__main__":
    main()
