# Gold Price Predictor

本项目通过抓取国际黄金价格数据构建近 3 个月的日度数据集，并使用机器学习对未来 1/3/5 个交易日的涨跌概率进行预测。输出内容用于风险参考，不构成交易建议。

## 功能概览

- 爬取国际黄金价格（默认使用 Stooq 的 XAUUSD 数据）
- 数据清洗与特征工程（收益率、波动率、技术指标、滞后特征）
- 机器学习模型（Logistic Regression / Random Forest）
- 滚动训练与评估（Walk Forward）
- 可视化趋势与收益率分布

## 目录结构

```
.
├── analysis
│   ├── feature_engineering.py
│   └── indicators.py
├── crawler
│   └── fetch_price.py
├── model
│   ├── predict.py
│   └── train.py
├── visualization
│   └── charts.py
├── config
│   └── settings.yaml
├── data
│   ├── raw
│   └── processed
├── main.py
└── README.md
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行项目

```bash
python main.py
```

运行后会生成：

- `data/raw/gold_prices_raw.csv`
- `data/processed/gold_prices_processed.csv`
- `data/processed/models/*`（模型与指标）
- `data/processed/charts/*`（可视化图表与预测结果）

## 数据字段示例

| date | open | high | low | close | volume |
| --- | --- | --- | --- | --- | --- |

## 风险与限制

- 黄金价格受突发事件影响强，模型无法预测黑天鹅。
- 当前仅使用价格数据，未引入宏观指标（美元指数、利率等）。

## 后续可扩展方向

- 接入 LBMA / COMEX 等数据源
- 引入 DXY、10Y 美债收益率
- 使用 LSTM / Transformer 时间序列模型
- 构建实时预测仪表盘（Streamlit）
