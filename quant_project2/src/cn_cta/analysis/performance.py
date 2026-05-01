"""Portfolio performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cn_cta.risk.metrics import conditional_value_at_risk, value_at_risk


def performance_summary(
    returns: pd.Series,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Return common CTA performance metrics."""

    clean = returns.fillna(0.0)
    if clean.empty:
        return {
            "annual_return": 0.0,
            "annual_volatility": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "var_95": 0.0,
            "cvar_95": 0.0,
        }

    equity = (1 + clean).cumprod()
    annual_return = float(equity.iloc[-1] ** (periods_per_year / len(clean)) - 1)
    annual_volatility = float(clean.std(ddof=0) * np.sqrt(periods_per_year))
    excess_return = annual_return - risk_free_rate
    sharpe = excess_return / annual_volatility if annual_volatility > 0 else 0.0
    drawdown = equity / equity.cummax() - 1
    max_drawdown = float(drawdown.min())
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0.0

    return {
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": float(sharpe),
        "max_drawdown": max_drawdown,
        "calmar": float(calmar),
        "var_95": value_at_risk(clean, 0.95),
        "cvar_95": conditional_value_at_risk(clean, 0.95),
    }
