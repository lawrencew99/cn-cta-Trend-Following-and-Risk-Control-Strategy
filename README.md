# cn-cta

`cn-cta` 是一个面向中国交易市场日频 CTA 趋势跟踪研究的 Python 工程包。聚焦 A 股 ETF、指数期货和商品期货，提供数据契约、趋势信号、风险控制、回测、绩效归因和滚动参数验证。

项目内置 AKShare 适配器，也保留 `MarketDataAdapter` 协议，方便继续接入 Tushare、券商数据或内部研究数据库。

## 安装

```bash
pip install -e ".[dev]"
```

如果只需要安装 AKShare 数据源依赖：

```bash
pip install -e ".[akshare]"
```

## 快速开始

```bash
python examples/run_cta_demo.py
```

示例会使用内置模拟数据运行 Donchian 趋势跟踪策略，经过波动率目标和回撤控制后生成回测结果、绩效指标、风险指标和收益归因。

生成 CTA demo 可视化报告：

```bash
python examples/visualize_cta_demo.py
```

默认会保存图片到 `outputs/cta_demo_visualization.png`。可以用 `--output` 指定输出路径，用 `--show` 在保存后弹出图表窗口。

使用 AKShare ETF 日频数据：

```python
from cn_cta.data import AkShareDataAdapter, MarketDataRequest

adapter = AkShareDataAdapter(asset_type="etf", adjust="qfq")
data = adapter.load_ohlcv(
    MarketDataRequest(symbols=["510300"], start="2022-01-01", end="2024-12-31")
)
```

使用 AKShare 期货日频数据：

```python
adapter = AkShareDataAdapter(asset_type="futures")
data = adapter.load_ohlcv(
    MarketDataRequest(symbols=["IF0", "RB0"], start="2022-01-01", end="2024-12-31")
)
```

## 标准行情字段

数据适配器返回的 `pandas.DataFrame` 至少需要包含：

- `date`：交易日
- `symbol`：标的代码
- `open`、`high`、`low`、`close`：OHLC 价格
- `volume`：成交量

可选字段包括 `limit_up`、`limit_down`、`paused`、`multiplier`、`margin_rate`，用于中国市场涨跌停、停牌、合约乘数和保证金研究。

## 模块划分

- `cn_cta.data`：数据协议、schema 校验、模拟行情、AKShare 适配器。
- `cn_cta.signals`：均线突破、Donchian 通道、波动率突破。
- `cn_cta.risk`：ATR、止损止盈、波动率目标、最大回撤控制、VaR/CVaR。
- `cn_cta.backtest`：日频回测、交易成本、权益曲线。
- `cn_cta.analysis`：绩效指标、压力测试、蒙特卡洛重采样、收益归因。
- `cn_cta.walk_forward`：滚动训练/验证窗口和参数筛选。

## 设计边界

用于研究和项目展示，不构成投资建议

