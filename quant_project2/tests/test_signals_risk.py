import pandas as pd

from cn_cta.data import make_sample_ohlcv
from cn_cta.risk import atr, volatility_target_positions
from cn_cta.signals import donchian_breakout, moving_average_breakout, volatility_breakout


def test_trend_signals_have_expected_shape() -> None:
    data = make_sample_ohlcv(symbols=("510300.SH", "IF.CFE"), periods=120)

    ma = moving_average_breakout(data, fast_window=10, slow_window=30)
    donchian = donchian_breakout(data, entry_window=20, exit_window=10)
    breakout = volatility_breakout(data, lookback=15, threshold=1.0)

    assert ma.shape == donchian.shape == breakout.shape == (120, 2)
    assert set(ma.stack().dropna().unique()).issubset({-1.0, 0.0, 1.0})


def test_volatility_target_positions_are_bounded() -> None:
    data = make_sample_ohlcv(symbols=("510300.SH",), periods=100)
    close = data.pivot(index="date", columns="symbol", values="close")
    signals = pd.DataFrame(1.0, index=close.index, columns=close.columns)

    positions = volatility_target_positions(signals, close, target_vol=0.1, max_leverage=0.7)

    assert positions.max().max() <= 0.7
    assert positions.min().min() >= 0.0


def test_atr_is_positive_after_warmup() -> None:
    data = make_sample_ohlcv(symbols=("IF.CFE",), periods=40)

    result = atr(data, window=14)

    assert result.dropna().iloc[-1, 0] > 0
