"""Vectorized daily backtest engine."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    """Backtest assumptions for daily China-market research."""

    initial_capital: float = 1_000_000.0
    commission_bps: float = 2.0
    slippage_bps: float = 1.0
    stamp_duty_bps: float = 0.0
    periods_per_year: int = 252

    @property
    def cost_rate(self) -> float:
        return (self.commission_bps + self.slippage_bps + self.stamp_duty_bps) / 10_000


@dataclass(frozen=True)
class BacktestResult:
    """Container returned by `run_backtest`."""

    equity_curve: pd.Series
    returns: pd.Series
    gross_returns: pd.Series
    costs: pd.Series
    positions: pd.DataFrame
    turnover: pd.Series
    asset_returns: pd.DataFrame
    pnl_by_symbol: pd.DataFrame


def run_backtest(
    close: pd.DataFrame,
    positions: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Run a close-to-close daily backtest with one-bar signal delay."""

    cfg = config or BacktestConfig()
    close = close.sort_index()
    positions = positions.reindex_like(close).fillna(0.0).clip(-10.0, 10.0)

    asset_returns = close.pct_change().fillna(0.0)
    tradable_positions = positions.shift(1).fillna(0.0)
    pnl_by_symbol = tradable_positions * asset_returns
    gross_returns = pnl_by_symbol.sum(axis=1)

    turnover = positions.diff().abs().sum(axis=1).fillna(positions.abs().sum(axis=1))
    costs = turnover * cfg.cost_rate
    net_returns = gross_returns - costs
    equity_curve = cfg.initial_capital * (1 + net_returns).cumprod()

    return BacktestResult(
        equity_curve=equity_curve.rename("equity"),
        returns=net_returns.rename("returns"),
        gross_returns=gross_returns.rename("gross_returns"),
        costs=costs.rename("costs"),
        positions=positions,
        turnover=turnover.rename("turnover"),
        asset_returns=asset_returns,
        pnl_by_symbol=pnl_by_symbol,
    )
