# GoldEyesAI - 黄金价格预测系统

## 项目简介

GoldEyesAI 是一个专注于黄金价格预测的智能系统，通过抓取国际黄金价格数据构建日度数据集，并使用机器学习技术对未来 1/3/5 个交易日的涨跌概率进行预测。系统采用滚动训练与评估方法，确保模型能够适应市场变化，为投资者提供风险参考。

**免责声明**：本系统输出内容仅用于风险参考，不构成任何投资建议。投资有风险，入市需谨慎。

## 核心功能

- **数据采集**：自动从 Stooq 抓取 XAUUSD（黄金/美元）历史价格数据
- **数据处理**：清洗数据、计算收益率、波动率等基础特征
- **特征工程**：构建多种技术指标（MA、EMA、RSI、MACD、布林带等）和滞后特征
- **模型训练**：使用 Logistic Regression 和 Random Forest 两种模型进行滚动训练
- **模型评估**：采用 Walk Forward 方法评估模型性能，计算准确率和 AUC 指标
- **概率预测**：输出未来 1/3/5 个交易日的涨跌概率
- **可视化**：生成价格趋势图和收益率分布图

## 目录结构

```
.
├── analysis/            # 分析模块
│   ├── feature_engineering.py  # 特征工程
│   └── indicators.py           # 技术指标计算
├── crawler/             # 数据抓取模块
│   └── fetch_price.py          # 价格数据抓取
├── model/               # 模型模块
│   ├── predict.py              # 模型预测
│   └── train.py                # 模型训练
├── visualization/       # 可视化模块
│   └── charts.py               # 图表生成
├── config/              # 配置目录
│   └── settings.yaml           # 系统配置
├── data/                # 数据存储
│   ├── raw/                    # 原始数据
│   └── processed/              # 处理后数据
│       ├── models/             # 模型文件
│       └── charts/             # 图表文件
├── main.py              # 主程序
├── requirements.txt     # 依赖文件
└── README.md            # 项目说明
```

## 技术栈

- **数据处理**：Pandas、NumPy
- **网络请求**：Requests
- **机器学习**：Scikit-learn
- **模型存储**：Joblib
- **可视化**：Matplotlib
- **配置管理**：PyYAML

## 快速开始

### 环境要求

- Python 3.7+
- pip 包管理器

### 安装依赖

```bash
# 在项目根目录执行
pip install -r requirements.txt
```

### 运行系统

```bash
# 在项目根目录执行
python main.py
```

### 运行流程

1. 系统加载配置文件 `config/settings.yaml`
2. 抓取指定天数的黄金价格数据
3. 进行数据清洗和特征工程
4. 对每个预测周期（1/3/5天）进行：
   - 构建标签
   - 滚动训练模型
   - 选择最佳模型
   - 生成预测结果
5. 生成可视化图表
6. 保存所有结果到指定目录

## 配置说明

系统配置文件位于 `config/settings.yaml`，可根据需要调整以下参数：

### 项目配置
- `data_days`：抓取的历史数据天数（默认90天）
- `horizons`：预测周期列表（默认[1, 3, 5]天）

### 数据源配置
- `provider`：数据提供商（默认stooq）
- `stooq_url`：Stooq API URL

### 存储配置
- `raw_csv`：原始数据存储路径
- `processed_csv`：处理后数据存储路径
- `model_dir`：模型存储目录

### 特征配置
- `ma_windows`：移动平均线窗口
- `ema_windows`：指数移动平均线窗口
- `rsi_period`：RSI指标周期
- `macd_fast/slow/signal`：MACD指标参数
- `bollinger_window/std`：布林带参数
- `lag_days`：滞后特征天数

### 模型配置
- `random_state`：随机种子
- `train_window`：训练窗口大小
- `test_window`：测试窗口大小
- `min_train_size`：最小训练集大小

### 可视化配置
- `output_dir`：图表输出目录

## 输出结果

运行后会生成以下文件：

### 数据文件
- `data/raw/gold_prices_raw.csv`：原始价格数据
- `data/processed/gold_prices_processed.csv`：处理后的数据（含特征和标签）

### 模型文件
- `data/processed/models/horizon_*/`：每个预测周期的模型目录
  - `logistic.joblib`：逻辑回归模型
  - `random_forest.joblib`：随机森林模型
  - `metrics.json`：模型性能指标

### 图表文件
- `data/processed/charts/close_trend.png`：黄金价格趋势图
- `data/processed/charts/return_distribution.png`：收益率分布图
- `data/processed/charts/predictions.json`：预测结果（涨跌概率）

## 预测结果解读

预测结果以 JSON 格式存储在 `data/processed/charts/predictions.json` 文件中，示例：

```json
[
  {
    "horizon": 1,
    "probability_up": 0.65,
    "probability_down": 0.35
  },
  {
    "horizon": 3,
    "probability_up": 0.58,
    "probability_down": 0.42
  },
  {
    "horizon": 5,
    "probability_up": 0.52,
    "probability_down": 0.48
  }
]
```

- `horizon`：预测周期（交易日数）
- `probability_up`：上涨概率
- `probability_down`：下跌概率

## 风险与限制

- **数据限制**：当前仅使用价格数据，未引入宏观经济指标（如美元指数、利率、通胀率等）
- **模型限制**：机器学习模型无法预测黑天鹅事件和极端市场波动
- **预测精度**：短期预测精度相对较高，长期预测精度会降低
- **数据源风险**：依赖 Stooq 数据源的稳定性和准确性

## 常见问题

### Q: 运行时出现网络错误怎么办？
A: 检查网络连接，确保能够访问 Stooq 网站。如果问题持续，可尝试修改 `settings.yaml` 中的数据源配置。

### Q: 预测结果不准确怎么办？
A: 预测结果受多种因素影响，包括市场环境变化、数据质量等。可尝试调整特征参数或增加更多特征来改善模型性能。

### Q: 如何增加预测周期？
A: 修改 `settings.yaml` 中的 `horizons` 配置，添加需要的预测周期（如 [1, 3, 5, 10]）。

### Q: 如何更换数据源？
A: 可修改 `crawler/fetch_price.py` 文件，添加新的数据源抓取函数，并在 `main.py` 中调用。

## 后续扩展方向

- **数据源扩展**：接入 LBMA、COMEX 等权威黄金价格数据源
- **特征扩展**：引入宏观经济指标（美元指数、10年期美债收益率、通胀率等）
- **模型扩展**：使用 LSTM、Transformer 等时间序列模型
- **实时预测**：构建 Streamlit 仪表盘，提供实时预测和可视化
- **多资产预测**：扩展到其他贵金属（如白银、铂金）和大宗商品
- **风险评估**：增加风险评估模块，计算不同预测结果下的风险水平

## 贡献指南

欢迎对本项目提出改进建议和贡献代码。如有问题或建议，请通过 GitHub Issues 提交。

## 许可证

本项目采用 MIT 许可证。

---

**GoldEyesAI** - 用人工智能洞察黄金市场趋势
