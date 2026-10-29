# 美股量化研究：阶段一——数据与风险刻画

用于学习美股行情数据处理、基础风险指标和组合相关性分析的 Python 项目。当前版本只做研究分析，不生成买卖信号，也不是实盘交易系统。

## 功能

- 下载 SPY、SOXX、NVDA、GOOGL 日线行情并缓存为 CSV。
- 检查数据日期、重复行、缺失值、OHLC 合法性和异常大幅日收益。
- 使用带下载元数据的本地缓存；缓存过期、覆盖范围不完整或缺失元数据时自动刷新。
- 计算日收益率、复合年化收益率（CAGR）、年化波动率、最大回撤和相对 SPY 的 Beta。
- 计算组合总收益、年化收益/波动率、Sharpe、Sortino、最大回撤、历史 VaR/CVaR，以及各持仓对组合波动率的贡献。
- 提供横截面动量 v1 策略、单期信号滞后、基准对比、交易成本/滑点假设和可复现回测输出。
- 计算 20/60/120 个观测窗口的滚动波动率和回撤，并支持自定义压力情景。
- 输出相关矩阵及每一对资产的共同观测数量。
- 输出最大回撤的峰值/谷值/恢复日期，以及每次分析的数据截止时间。
- 绘制个股完整可用区间价格图、共同样本净值图、回撤图和相关矩阵图。
- 使用离线单元测试验证核心计算，不要求测试时访问行情 API。

行情来自 yfinance 对 Yahoo Finance 数据的访问，适合入门和探索，不应视为经过审计的机构级数据源。

## 项目结构

    phase1_risk_analysis/
    ├── README.md
    ├── requirements.txt
    ├── requirements-dev.txt
    ├── .gitignore
    ├── phase1_risk_analysis.py
    ├── position_scenarios.py
    ├── portfolio_risk.py
    ├── backtest_engine.py
    ├── tests/
    │   ├── test_phase1_risk_analysis.py
    │   ├── test_position_scenarios.py
    │   ├── test_portfolio_risk.py
    │   └── test_backtest_engine.py
    ├── .github/workflows/tests.yml
    ├── data/raw/       # 本地行情缓存；CSV 默认不提交
    └── outputs/        # 运行后生成的表格与图片；默认不提交

## 获取项目

```bash
git clone https://github.com/chloe6888/phase1_risk_analysis.git
cd phase1_risk_analysis
```

## 环境要求

- Python 3.10 或更高版本
- 首次下载数据时可以访问 Yahoo Finance
- 安装 Python 依赖时可以访问配置的包源

## 安装与运行

macOS / Linux：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python phase1_risk_analysis.py
```

Windows PowerShell：

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python phase1_risk_analysis.py
```

脚本会在缓存缺失、缓存超过默认 24 小时、缓存覆盖范围不足或元数据缺失时尝试重新下载。需要强制更新所有标的时运行：

```bash
python phase1_risk_analysis.py --refresh

# 仅在有意接受旧数据且刷新失败时，才使用缓存回退
python phase1_risk_analysis.py --allow-stale-cache
```

## 配置

在脚本顶部调整以下参数：

- `TICKERS`：研究标的列表，默认 SPY、SOXX、NVDA、GOOGL。
- `BENCHMARK`：Beta 基准，默认 SPY，必须包含在 `TICKERS` 中。
- `START_DATE`：下载起始日期，默认 2015-01-01。
- `END_DATE`：默认取纽约当地当前日期，下载接口的结束日期不包含当天。

建议第一次使用默认配置，成功运行后再添加标的。

## 输出文件

| 文件 | 内容 |
|---|---|
| `data/raw/<TICKER>_daily.csv` | 各标的整理后的行情缓存，包含可用 OHLCV 和公司行为字段 |
| `data/raw/<TICKER>_daily.meta.json` | 缓存来源、请求日期范围、下载时间、有效数据起止日和行数 |
| `data/raw/adjusted_close_panel.csv` | 调整收盘价面板 |
| `outputs/data_quality_report.csv` | 数据完整性摘要 |
| `outputs/daily_returns.csv` | 调整收盘价计算的日收益率 |
| `outputs/risk_metrics.csv` | CAGR、年化波动率、最大回撤和 Beta 等指标 |
| `outputs/correlation_matrix.csv` | 日收益率相关矩阵 |
| `outputs/correlation_observation_counts.csv` | 每一对资产相关系数使用的共同收益观测数量 |
| `outputs/max_drawdown_details.csv` | 最大回撤的高点、低点、恢复日期和日历天数 |
| `outputs/analysis_metadata.json` | 分析时间、参数、数据源及各标的最后数据日期 |
| `outputs/closing_prices.png` | 每只标的各自完整的有效价格区间 |
| `outputs/net_value.png` | 共同样本起点归一化为 100 的净值比较 |
| `outputs/drawdown.png` | 从历史高点计算的回撤曲线 |
| `outputs/correlation_matrix.png` | 相关矩阵热力图 |
| `outputs/portfolio_risk_summary.json` | 组合汇总指标及权重口径 |
| `outputs/portfolio_risk_contributions.csv` | 各持仓对组合波动率的贡献 |
| `outputs/portfolio_daily_returns.csv` | 资产日收益及组合日收益 |
| `outputs/portfolio_drawdown.csv` | 组合净值、运行高点和回撤序列 |
| `outputs/portfolio_rolling_risk.csv` | 滚动波动率和滚动最大回撤 |
| `outputs/portfolio_stress_tests.csv` | 用户定义压力情景下的组合损益 |
| `outputs/backtest_summary.json` | 策略及基准的收益风险指标、相对表现和方法说明 |
| `outputs/backtest_daily_returns.csv` | 策略毛收益、成本、净收益、基准收益和换手率 |
| `outputs/backtest_applied_weights.csv` | 实际用于每个收益区间的滞后权重 |
| `outputs/backtest_equity_curve.csv` | 策略与基准的归一化净值曲线 |

## 组合风险分析

先运行 `phase1_risk_analysis.py` 生成 `data/raw/adjusted_close_panel.csv`，再按实际组合权重运行：

```bash
python portfolio_risk.py --weights GOOGL=0.60,SPY=0.40
```

可选地加入压力情景和组合市值，测算不同资产冲击下的组合影响：

```bash
python portfolio_risk.py \
  --weights GOOGL=0.60,SPY=0.40 \
  --portfolio-value 10000 \
  --scenario tech_selloff:GOOGL=-0.25,SPY=-0.10 \
  --scenario broad_selloff:GOOGL=-0.20,SPY=-0.20
```

权重会自动归一化，当前版本只支持非负权重（不支持做空）。压力情景中的收益冲击使用小数表示，例如 `-0.20` 表示下跌 20%；未指定的资产冲击按 0 处理。历史 VaR/CVaR 是一个观测周期的历史尾部估计，不是最大损失保证。组合日收益使用目标权重按每个观测区间再平衡的假设；若你要模拟买入后长期不调仓的持有组合，需要在后续版本增加 buy-and-hold 权重漂移模型。

## 策略回测 v1

当前内置一个便于审计的横截面动量基线：按过去 `lookback` 个观测的收益排名，跳过最近 `skip_recent` 个观测，等权持有排名靠前的 `top_n` 个非基准资产，并与基准比较。

```bash
python backtest_engine.py \
  --benchmark SPY \
  --lookback 126 \
  --skip-recent 21 \
  --top-n 2 \
  --transaction-cost-bps 5 \
  --slippage-bps 5
```

脚本会输出总收益、复合年化收益率、年化波动率、Sharpe、Sortino、最大回撤、胜率、换手率、基准相对表现、跟踪误差、信息比率和 Beta，以及每日收益、应用权重与净值曲线。首次使用真实数据运行前，请确认价格区间、基准和样本外评估方案。

**重要假设：** 信号生成于收盘后，权重至少滞后一条观测再应用到收益，避免同日信号偷看同日收益。交易成本以目标权重变化估算；目前尚未完整模拟持仓权重漂移、订单成交价、税费、停牌/涨跌停约束、最终平仓成本。该回测是策略研究原型，不是实盘收益预期，也不能仅凭样本内结果得出策略长期有效的结论。

## 计算口径与局限

- 历史总收益分析使用 Adj Close 作为调整后价格代理；这不是实际可成交的历史报价。
- CAGR 按实际日历时间跨度年化；年化波动率使用日收益样本标准差乘以平方根 252。
- Beta 以匹配日期的资产和 SPY 日收益率计算。
- 相关系数和 Beta 描述历史关系，不代表因果关系；压力期的相关性可能显著上升。
- 缺失价格不会自动填充为零收益。若下载数据存在缺口，必须检查数据质量报告。
- 当前版本不模拟交易成本、滑点、税费、做空约束或订单执行。
- 当前四个标的不代表整个美股市场；历史最大回撤也不是未来损失上限。不要仅凭这些指标直接推导安全仓位。

## 持仓加仓情景工具

`position_scenarios.py` 用来计算现有持仓的市值、未实现盈亏、不同加仓价格下的新成本均价，以及在压力价格下的盈亏。不预测价格，也不自动生成买卖信号。个人持仓参数在运行时传入，不要把个人信息提交到公共仓库。

示例：

```bash
python position_scenarios.py \
  --ticker GOOGL \
  --shares 10 \
  --avg-cost 100 \
  --price 95 \
  --levels 95 90 85 80 \
  --add-shares 1 2 3 \
  --stress-prices 80 75 70 \
  --history-csv data/raw/GOOGL_daily.csv
```

命令中的价格和持仓数量均为虚构示例，运行前请替换为最新可核验的报价和自己的持仓数据。指定 `--history-csv` 时，CSV 需包含 `Date` 索引以及 `Close` 或 `Adj Close` 列。程序会输出 CSV 至 `outputs/`。


## 五批累计计划与风险预算

连续执行的多批加仓请使用 `--staged-prices` 与 `--staged-shares`。程序会按批次累计股数、成本均价、现金投入和压力价格下的盈亏。

可以用 `--max-add-cash` 和 `--max-total-shares` 设置资金及持股上限；超过限制时会拒绝该累计计划。还可使用 `--max-stress-loss-pct 0.10` 限制所给压力价格下的损失不超过成本基数的 10%。设置该项时必须同时提供 `--stress-prices`；若计划不满足压力损失限制，程序会拒绝累计计划。独立情景会在结果表中标出违规项目。

```bash
python position_scenarios.py \
  --ticker GOOGL \
  --shares 10 \
  --avg-cost 100 \
  --price 95 \
  --staged-prices 95 90 85 80 75 \
  --staged-shares 1 1 1 1 1 \
  --stress-prices 80 75 \
  --max-add-cash 500 \
  --max-total-shares 15 \
  --max-stress-loss-pct 0.25
```

命令中的行情和持仓数字只是规划示例，并非当前报价或买入建议。独立情景仍使用 `--levels` 与 `--add-shares`；这两项生成的是彼此独立的情景，不是按顺序执行的累计计划。

## 自动化测试

安装开发依赖后，在项目根目录运行：

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

测试使用人工构造的离线数据，不代表真实市场行情验证，也不替代数据源交叉核验。

仓库配置了 GitHub Actions，会在向 `main` 分支推送或提交针对 `main` 的 Pull Request 时运行这些测试。

## 后续路线

1. **数据与组合风险**：已实现基础行情数据治理、组合风险贡献、滚动风险和压力测试。
2. **横截面动量 v1**：已实现可配置回测基线、基准比较、单期信号滞后与初步交易成本估算。
3. **稳健性实验**：下一步测试不同动量窗口、调仓频率、资产池、权重上限和市场区间，保留实验记录。
4. **现实约束 v2**：完善持仓权重漂移、成本/滑点模型、再平衡频率和最终平仓成本。
5. **回测审计**：系统检查未来函数、幸存者偏差、多重试验、样本外表现和数据披露时点。
6. **资产配置与持续监控**：组合约束优化、定时报告、异常检测和风险提醒。
7. **模拟运行评估**：先纸面运行，并明确规则失效条件和风险预算；未完成审计前不接实盘交易接口。

---

本项目仅用于编程学习与量化研究，不构成投资、税务或法律建议。历史表现不代表未来结果。
