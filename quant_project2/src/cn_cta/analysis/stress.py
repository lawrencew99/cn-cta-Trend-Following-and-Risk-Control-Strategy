"""Stress testing and Monte Carlo resampling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cn_cta.analysis.performance import performance_summary


def stress_test(
    returns: pd.Series,
    scenarios: dict[str, pd.Series | float],
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Apply deterministic return shocks and summarize the result."""

    rows: dict[str, dict[str, float]] = {}
    base = returns.fillna(0.0)
    for name, shock in scenarios.items():
        if isinstance(shock, pd.Series):
            stressed = base.add(shock.reindex(base.index).fillna(0.0), fill_value=0.0)
        else:
            stressed = base + float(shock)
        rows[name] = performance_summary(stressed, periods_per_year)
    return pd.DataFrame.from_dict(rows, orient="index")


def monte_carlo_resample(
    returns: pd.Series,
    n_paths: int = 1_000,
    path_length: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Bootstrap daily returns into terminal return and drawdown outcomes."""

    clean = returns.dropna().to_numpy()
    if clean.size == 0:
        return pd.DataFrame(columns=["terminal_return", "max_drawdown"])
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")

    rng = np.random.default_rng(seed)
    length = path_length or clean.size
    samples = rng.choice(clean, size=(n_paths, length), replace=True)
    equity = np.cumprod(1 + samples, axis=1)
    running_max = np.maximum.accumulate(equity, axis=1)
    drawdowns = equity / running_max - 1
    return pd.DataFrame(
        {
            "terminal_return": equity[:, -1] - 1,
            "max_drawdown": drawdowns.min(axis=1),
        }
    )
