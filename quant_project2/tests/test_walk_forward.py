import pandas as pd

from cn_cta.walk_forward import generate_windows, walk_forward_optimize


def test_generate_windows() -> None:
    index = pd.date_range("2024-01-01", periods=10)

    windows = generate_windows(index, train_size=4, validation_size=2, step=2)

    assert len(windows) == 3
    assert len(windows[0][0]) == 4
    assert len(windows[0][1]) == 2


def test_walk_forward_selects_parameters() -> None:
    close = pd.DataFrame({"A": range(20)}, index=pd.date_range("2024-01-01", periods=20))
    grid = [{"edge": 1.0}, {"edge": 2.0}]

    def strategy(_: pd.DataFrame, params: dict[str, object]) -> pd.Series:
        return pd.Series([float(params["edge"])], index=["score"])

    def score(result: pd.Series) -> float:
        return float(result.iloc[0])

    results = walk_forward_optimize(close, grid, strategy, score, train_size=8, validation_size=4)

    assert results
    assert all(result.params["edge"] == 2.0 for result in results)
