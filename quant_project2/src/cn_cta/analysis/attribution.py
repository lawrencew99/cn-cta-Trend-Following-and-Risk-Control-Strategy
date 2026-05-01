"""Return attribution for CTA strategy diagnostics."""

from __future__ import annotations

import pandas as pd


def return_attribution(
    pnl_by_symbol: pd.DataFrame,
    positions: pd.DataFrame,
    costs: pd.Series,
) -> pd.DataFrame:
    """Break portfolio return into direction, holding, leverage, and costs."""

    aligned_positions = positions.reindex_like(pnl_by_symbol).fillna(0.0)
    exposure = aligned_positions.abs()
    direction_component = pnl_by_symbol.where(aligned_positions.shift(1).fillna(0.0) != 0, 0.0).sum()
    holding_days = (exposure > 0).sum()
    average_leverage = exposure.mean()

    summary = pd.DataFrame(
        {
            "direction_return": direction_component,
            "holding_days": holding_days.astype(float),
            "average_abs_leverage": average_leverage,
            "gross_return": pnl_by_symbol.sum(),
        }
    )
    summary["cost_share"] = 0.0
    gross_total = summary["gross_return"].abs().sum()
    if gross_total > 0:
        summary["cost_share"] = costs.sum() * summary["gross_return"].abs() / gross_total
    summary["net_return"] = summary["gross_return"] - summary["cost_share"]
    return summary.sort_values("net_return", ascending=False)
