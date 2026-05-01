"""Tail-risk metrics."""

from __future__ import annotations

import pandas as pd


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR as a positive loss number."""

    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    clean = returns.dropna()
    if clean.empty:
        return 0.0
    return float(-clean.quantile(1 - confidence))


def conditional_value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical CVaR/expected shortfall as a positive loss number."""

    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    clean = returns.dropna()
    if clean.empty:
        return 0.0
    threshold = clean.quantile(1 - confidence)
    tail = clean[clean <= threshold]
    if tail.empty:
        return 0.0
    return float(-tail.mean())
