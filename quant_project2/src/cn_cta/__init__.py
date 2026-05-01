"""China-market CTA trend-following research toolkit."""

from cn_cta.backtest import BacktestConfig, BacktestResult, run_backtest
from cn_cta.data import MarketDataAdapter, MarketDataRequest, validate_ohlcv
from cn_cta.signals import donchian_breakout, moving_average_breakout, volatility_breakout

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "MarketDataAdapter",
    "MarketDataRequest",
    "donchian_breakout",
    "moving_average_breakout",
    "run_backtest",
    "validate_ohlcv",
    "volatility_breakout",
]
