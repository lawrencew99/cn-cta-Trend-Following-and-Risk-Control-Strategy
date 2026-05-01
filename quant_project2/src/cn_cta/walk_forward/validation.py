"""Walk-forward parameter validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import pandas as pd


@dataclass(frozen=True)
class WalkForwardResult:
    """Selected parameters and validation performance for one window."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    params: dict[str, object]
    train_score: float
    validation_score: float


def generate_windows(
    index: pd.Index,
    train_size: int,
    validation_size: int,
    step: int | None = None,
) -> list[tuple[pd.Index, pd.Index]]:
    """Generate rolling train/validation index slices."""

    if train_size <= 0 or validation_size <= 0:
        raise ValueError("train_size and validation_size must be positive")
    step = step or validation_size
    if step <= 0:
        raise ValueError("step must be positive")

    windows: list[tuple[pd.Index, pd.Index]] = []
    start = 0
    while start + train_size + validation_size <= len(index):
        train_idx = index[start : start + train_size]
        validation_idx = index[start + train_size : start + train_size + validation_size]
        windows.append((train_idx, validation_idx))
        start += step
    return windows


def walk_forward_optimize(
    close: pd.DataFrame,
    parameter_grid: Iterable[dict[str, object]],
    strategy: Callable[[pd.DataFrame, dict[str, object]], pd.Series],
    score: Callable[[pd.Series], float],
    train_size: int,
    validation_size: int,
    step: int | None = None,
) -> list[WalkForwardResult]:
    """Select parameters on train windows and evaluate them out-of-sample."""

    params_list = list(parameter_grid)
    if not params_list:
        raise ValueError("parameter_grid must not be empty")

    results: list[WalkForwardResult] = []
    for train_idx, validation_idx in generate_windows(close.index, train_size, validation_size, step):
        train_close = close.loc[train_idx]
        validation_close = close.loc[validation_idx]

        train_scores = [(params, score(strategy(train_close, params))) for params in params_list]
        best_params, best_train_score = max(train_scores, key=lambda item: item[1])
        validation_score = score(strategy(validation_close, best_params))

        results.append(
            WalkForwardResult(
                train_start=pd.Timestamp(train_idx[0]),
                train_end=pd.Timestamp(train_idx[-1]),
                validation_start=pd.Timestamp(validation_idx[0]),
                validation_end=pd.Timestamp(validation_idx[-1]),
                params=dict(best_params),
                train_score=float(best_train_score),
                validation_score=float(validation_score),
            )
        )
    return results
