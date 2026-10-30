# 美股量化研究：阶段一——数据与风险刻画

用于学习美股行情数据处理、风险指标和组合相关性分析的 Python 项目。当前版本可基于明确配置的风险预算输出可解释的仓位调整方案，但不预测未来收益、不自动下单，也不是实盘交易系统。

## 功能

- 下载 SPY、SOXX、NVDA、GOOGL 日线行情并缓存为 CSV。
- 检查数据日期、重复行、缺失值、OHLC 合法性和异常大幅日收益。
- 使用带下载元数据的本地缓存；缓存过期、覆盖范围不完整或缺失元数据时自动刷新。
- 计算日收益率、复合年化收益率（CAGR）、年化波动率、最大回撤和相对 SPY 的 Beta。
- 计算组合总收益、年化收益/波动率、Sharpe、Sortino、最大回撤、历史 VaR/CVaR，以及各持仓对组合波动率的贡献。
- 提供漂移感知的横截面动量回测：持仓权重自然漂移，按实际成交额计交易成本，并与旧的目标权重日再平衡算法并排比较。
- 支持日/周/月三种调仓频率；周/月在自然周期最后一个可用交易日收盘形成调仓，并用于下一收益区间。
- 支持按日期提供国债年化收益率序列，转换为日无风险收益率后计算 Sharpe/Sortino。
- 支持参数网格实验、样本内/样本外指标分离，以及按自然年拆分的策略表现审查。
- 支持滚动 walk-forward：每个测试窗口只用之前的训练窗口选择参数，并输出连续、互不重叠的样本外收益序列。
- 通过统一入口按顺序运行数据、组合风险、基准回测、参数验证和滚动验证，并生成汇总研究报告与运行清单。
- 提供只在本机读取研究产物的 Streamlit 仪表盘，可查看组合风险、策略净值、参数实验、walk-forward 和综合报告，并从侧边栏提交研究任务。
- 仓位集中度规则分开配置：单一股票默认 15%，整体风险主题默认 45%，行业子主题默认 25%；超限仅提示复核，不自动交易。
- 支持导入本地持仓 CSV，按持股数量与最新可用收盘价计算市值和组合权重，并检测单一仓位和主题集中度；持仓文件被 Git 忽略，不提交到公开仓库。
- 增加 MU、SKHY、DRAM 存储主题决策模块：估算 DRAM ETF 对 MU/SKHY 的穿透敞口，输出主题上限、20/60 日趋势诊断和三种目标配置情景；仅作风险预算与复核，不自动下单。
- 计算 20/60/120 个观测窗口的滚动波动率和回撤，并支持自定义压力情景。
- 如果组合所含股票的上市历史较短，默认滚动风险窗口会按共同收益观测数自动缩短；历史不足时明确标注，不会因滚动图窗口不足而中断整份报告。
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
    ├── strategy_validation.py
    ├── strategy_research.py
    ├── portfolio_optimizer.py
    ├── paper_trading.py
    ├── risk_monitor.py
    ├── walk_forward_validation.py
    ├── run_research.py
    ├── dashboard.py
    ├── holdings_manager.py
    ├── memory_allocation.py
    ├── requirements-dashboard.txt
    ├── start_dashboard.sh
    ├── start_dashboard.ps1
    ├── tests/
    │   ├── test_phase1_risk_analysis.py
    │   ├── test_position_scenarios.py
    │   ├── test_portfolio_risk.py
    │   ├── test_backtest_engine.py
    │   ├── test_strategy_validation.py
    │   ├── test_strategy_research.py
    │   ├── test_portfolio_optimizer.py
    │   ├── test_paper_trading.py
    │   ├── test_risk_monitor.py
    │   ├── test_walk_forward_validation.py
    │   ├── test_run_research.py
    │   ├── test_dashboard.py
    │   └── test_holdings_manager.py
    ├── .github/workflows/tests.yml
    ├── data/private/   # 个人持仓导入文件；禁止提交到公开仓库
    ├── data/raw/       # 本地行情缓存；CSV 默认不提交
    └── outputs/        # 运行后生成的表格、图片和报告；默认不提交

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
| `data/raw/adjusted_close_panel.csv` | 历史总收益风险分析使用的调整收盘价面板 |
| `data/raw/close_price_panel.csv` | 按实际持股数量估值使用的未调整收盘价面板 |
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
| `outputs/strategy_parameter_sweep.csv` | 不同参数下的样本内、样本外收益风险表现 |
| `outputs/strategy_yearly_performance.csv` | 各参数下逐自然年的策略与基准表现 |
| `outputs/strategy_validation_metadata.json` | 测试期起点、参数网格规模及跳过组合数量 |
| `outputs/walk_forward_folds.csv` | 每轮训练区间、测试区间、训练选出的参数及样本外结果 |
| `outputs/walk_forward_oos_returns.csv` | 各轮拼接后的非重叠样本外收益序列 |
| `outputs/walk_forward_summary.json` | walk-forward 总体样本外表现和验证口径 |
| `outputs/holdings_valuation.csv` | 按最新有效未调整收盘价估算的持仓市值、成本和盈亏 |
| `outputs/holdings_latest_price_weights.csv` | 每只持仓最近可用价格日期、市值与组合权重 |
| `outputs/holdings_summary.json` | 导入持仓的整体市值、成本和集中度摘要 |
| `outputs/holdings_reconciliation.json` | 导入持仓行合计与券商汇总金额之间的差异 |
| `outputs/holdings_risk_alerts.json` | 单一持仓与风险主题集中度提醒 |
| `outputs/memory_allocation_summary.json` | MU/SKHY/DRAM 主题占比、DRAM 穿透敞口和总体风险预算判断 |
| `outputs/memory_position_diagnostics.csv` | 三项持仓最新收盘价、仓位权重、20/60 日动量与波动率诊断 |
| `outputs/memory_allocation_scenarios.csv` | 平衡、ETF 核心、直接持股三种持仓结构的目标股数与建议动作 |
| `outputs/research_report.md` | 汇总数据覆盖、组合风险、基准回测和样本外验证的报告 |
| `outputs/research_run_manifest.json` | 本次研究使用的参数和运行记录（保存在本地输出目录，不提交到仓库） |

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

## 策略回测 v2

当前内置一个便于审计的横截面动量基线：按过去 `lookback` 个观测的收益排名，跳过最近 `skip_recent` 个观测，等权持有排名靠前的 `top_n` 个非基准资产，并与基准比较。回测采用**权重漂移模型**：目标权重在调仓收盘后应用到下一收益区间；期间不强制日度再平衡，资产涨跌会改变实际组合权重；每次调仓成本按漂移后实际权重与新目标权重的绝对差额计算。

```bash
python backtest_engine.py \
  --benchmark SPY \
  --lookback 126 \
  --skip-recent 21 \
  --top-n 2 \
  --rebalance-freq weekly \
  --transaction-cost-bps 5 \
  --slippage-bps 5

# 也可选择 daily 或 monthly
```

脚本会输出总收益、复合年化收益率、年化波动率、Sharpe、Sortino、最大回撤、胜率、换手率、基准相对表现、跟踪误差、信息比率和 Beta，以及每日收益、应用权重与净值曲线。首次使用真实数据运行前，请确认价格区间、基准和样本外评估方案。

**重要假设：** 信号在收盘后形成，并在该收盘执行后应用到下一条收益区间，避免使用同日未来收益。日/周/月频率分别在每个交易日、每个自然周最后一个可用交易日、每个自然月最后一个可用交易日调仓。换手定义为漂移后的风险资产权重与目标风险资产权重之间绝对差额之和；买卖两侧均计入成交额。成本与滑点按该实际成交额计算，回测结束时还会按最后一期的实际风险资产权重计入期末清仓成本。旧的“目标权重乘每日收益”算法也会输出，供新旧口径审计，但该旧口径仍保留其未计漂移交易和期末清仓的历史算法局限。当前仍未模拟真实订单股数、成交价冲击、税费和停牌/涨跌停约束，因此这是更接近真实持仓的研究近似，而不是订单级回测。

## 调仓频率与参数稳定性专项评估

完成一次常规研究流程、更新好调整收盘价面板后，可以运行专项脚本，使用同一份本地行情比较日度、周度和月度调仓，并检查参数选择是否稳定：

```bash
python strategy_stability_report.py \
  --prices data/raw/adjusted_close_panel.csv \
  --benchmark SPY \
  --test-start 2023-01-01 \
  --baseline-lookback 126 \
  --baseline-skip-recent 21 \
  --baseline-top-n 2 \
  --lookbacks 63,126,252 \
  --skip-recents 5,21 \
  --top-ns 1,2,3 \
  --rebalance-freqs daily,weekly,monthly \
  --train-window 252 \
  --test-window 63 \
  --transaction-cost-bps 5 \
  --slippage-bps 5 \
  --output-dir outputs/stability_review
```

三种频率的基准比较共享相同的信号参数、行情区间和初始资金起点。动量信号首次可用后，较低频率策略在等到下一次计划调仓之前按现金持有，不会因为首次持仓日期较晚而悄悄缩短评估区间。旧的目标权重算法会与新漂移模型并列输出。

专项报告还会运行参数×频率网格和 walk-forward：网格汇总测试期超额收益、Sharpe 中位数及有效组合数；walk-forward 汇总每一折选中的 lookback、skip-recent、top-N 和调仓频率，统计各参数入选比例。它不会根据全区间收益自动挑选“赢家”。

结果写入 outputs/stability_review，包括 cadence_comparison.csv、strategy_parameter_sweep.csv、walk_forward_folds.csv、walk_forward_parameter_stability.csv、净值/收益序列、运行参数与行情文件 SHA-256，以及 stability_report.md。可以另加 `--risk-free-yield-csv data/private/treasury_yield.csv`，文件必须包含 Date 和 annual_yield 两列，年化收益率用小数表示。

test-start 应根据实际行情起止日期设定，并留出足够的训练期和测试期；若历史不足，可明确添加 `--skip-walk-forward`，但不能把单次参数网格当作滚动样本外验证。不要用测试期结果反复调参。

## 参数网格与样本外验证

回测指标不能只看全区间结果。使用 `strategy_validation.py` 可以预先指定样本外区间，逐一比较不同参数，并输出训练期、测试期和逐年表现。示例：

```bash
python strategy_validation.py \
  --test-start 2023-01-01 \
  --lookbacks 63,126,252 \
  --skip-recents 5,21 \
  --top-ns 1,2,3 \
  --rebalance-freqs daily,weekly,monthly
```

默认同时测试日、周、月三种调仓频率；可通过 `--rebalance-freqs daily,weekly,monthly` 显式配置。程序会报告所有参数×调仓频率组合，不会自动选出“最优参数”。建议先依据训练期设定参数，再把测试期作为未参与挑选的评估区间；测试期结果一旦被反复用于选参数，就不再是真正独立的样本外证据。

**无风险利率序列（可选）：** 可使用 `--risk-free-yield-csv path/to/treasury_yield.csv` 提供 `Date,annual_yield` 两列。日期应覆盖回测区间，`annual_yield` 使用小数年化收益率（例如 0.045 代表 4.5%，不能填 4.5）。没有新观测的日期沿用最近一次收益率，并按 `(1 + annual_yield)^(1/252) - 1` 换算为日收益率。未提供文件时，仍使用 `--risk-free-rate` 的固定年化值（默认 0）。这是一种现金收益率近似，不等于投资者可实现的实际每日收益。

## 多策略比较与过拟合审计

新增 `strategy_research.py`，用于在相同价格面板、基准、训练/测试区间、调仓频率、无风险收益和交易成本假设下比较多种简单策略基线：

- `equal_weight`：非基准资产等权配置；
- `cross_sectional_momentum`：横截面动量；
- `time_series_momentum`：时间序列动量；
- `trend_following`：均线趋势过滤；
- `mean_reversion`：低于滚动均线的均值回归基线；
- `low_volatility`：历史实现波动率较低的资产。

示例：

```bash
python strategy_research.py \
  --test-start 2023-01-01 \
  --strategies equal_weight,cross_sectional_momentum,time_series_momentum,trend_following,mean_reversion,low_volatility \
  --lookback 126 \
  --skip-recent 21 \
  --top-n 2 \
  --rebalance-freq monthly \
  --transaction-cost-bps 5 \
  --slippage-bps 5
```

输出文件：

| 文件 | 内容 |
|---|---|
| `outputs/strategy_comparison.csv` | 每种候选策略的训练期、测试期和基准比较指标 |
| `outputs/strategy_selected_oos_returns.csv` | 仅按训练期 Sharpe 选出的策略在测试期的逐日收益 |
| `outputs/strategy_research_summary.json` | 区间、候选策略、选优结果、DSR 和限制说明 |
| `outputs/strategy_research_report.md` | 人类可读的多策略研究报告 |

**审计边界：** 候选胜出者只按训练期 Sharpe 选择，测试期数据不参与选优。DSR 是近似统计诊断，使用候选数量近似试验次数并假设试验独立；策略之间高度相关时，修正结果并非严格显著性保证。单次留出测试也无法消除反复查看测试结果造成的窥探偏差，仍应保留从未参与调参的最终区间，并结合滚动 Walk-forward 结果。以上策略均为价格数据基线，不包含基本面、行业因子和真实订单执行。

## 约束组合优化

新增 `portfolio_optimizer.py`，使用共同完整的日收益样本估计年化协方差矩阵，并在以下约束下求解长期多头配置：

- 固定风险资产预算（未投入的比例保留为现金）；
- 单一资产最大权重；
- 按 ticker 分配的互斥行业/主题分组上限；
- 不允许做空，且会在约束不可行时明确报错。

默认目标是历史最小方差；也可以选择均值—方差目标，但其预期收益来自历史算术平均值，噪声较大，必须谨慎解释。

基础运行示例：

```bash
python portfolio_optimizer.py \
  --risky-budget 0.95 \
  --asset-max-weight 0.35 \
  --objective minimum_variance
```

可以通过 JSON 文件指定更细的约束：

```json
{"MU": 0.15, "SKHY": 0.10, "DRAM": 0.20}
```

以上内容可保存为 `asset_caps.json`，对应 ticker 最大权重；另外两个文件分别采用 `{"MU":"memory","SKHY":"memory","DRAM":"memory","GOOGL":"technology"}` 的 ticker 分组格式，以及 `{"memory":0.25,"technology":0.35}` 的分组最大权重格式。运行时通过 `--asset-caps-json`、`--groups-json` 和 `--group-caps-json` 指定。

输出包括 `portfolio_optimized_weights.csv`、`portfolio_optimization_summary.json` 和 `portfolio_optimization_report.md`。报告同时比较约束下的优化组合与约束下等权组合，并输出资产风险贡献。

**使用边界：** 该优化器只依据历史协方差（均值—方差模式还会使用历史平均收益）求解，不预测未来收益。历史最小方差不保证未来风险更低；分组必须互斥，每只资产只能属于一个组。当前不考虑交易成本、税费、流动性、最小交易单位和真实订单执行，也不会下单。

## 纸面交易模拟

新增 `paper_trading.py`，将优化器或人工编制的目标权重转换为离线模拟订单。目标 CSV 必须包含 `ticker,weight`；初始持仓 CSV 必须包含 `ticker,shares`。默认使用 `data/raw/close_price_panel.csv` 的最近可用收盘价，卖出先于买入，模拟成交价按配置滑点向不利方向调整，手续费按成交金额计算；现金不足时按比例缩减买单，整数股模式向下取整。

仅预览订单：

```bash
python paper_trading.py \
  --targets-csv outputs/portfolio_optimized_weights.csv \
  --holdings-csv data/private/holdings_snapshot.csv \
  --cash 5000 \
  --prices data/raw/close_price_panel.csv
```

默认不修改任何持仓账本。只有明确加入 `--commit-ledger` 时，才将模拟后的现金和股数写入本地 JSON 账本；后续运行会从账本读取模拟账户。输出包括 `paper_trade_orders.csv`、`paper_trading_holdings_after.csv` 和 `paper_trading_summary.json`。

**边界：** 这不是券商接口，不会发送订单。模拟采用参考收盘价加不利滑点，不模拟盘中成交、排队、市场冲击、税费、停牌或涨跌幅限制；纸面账本也不代表券商真实持仓。

## 组合风险监控

新增 `risk_monitor.py`，按当前持仓计算个股权重和主题/分组敞口，并检查单股上限、分组上限、历史年化波动率、历史最大回撤、单期 95% VaR/CVaR、历史不足和价格数据过期。

```bash
python risk_monitor.py \
  --holdings-csv data/private/holdings_snapshot.csv \
  --prices data/raw/close_price_panel.csv \
  --adjusted-prices data/raw/adjusted_close_panel.csv \
  --cash 5000 \
  --single-name-limit 0.15 \
  --volatility-limit 0.40 \
  --drawdown-limit 0.25
```

持仓 CSV 可增加 `risk_group` 列；也可以通过 `--groups-json` 与 `--group-caps-json` 覆盖分组和上限。输出包括 `risk_monitor_positions.csv`、`risk_monitor_alerts.csv`、`risk_monitor_daily_path.csv`、`risk_monitor_snapshot.json` 和 `risk_monitor_report.md`。

**历史口径限制：** 历史风险路径将当前股数应用于过去的调整收盘价，并假设现金余额不变；它不是按历史交易、入金、出金及公司行动重建的真实账户净值。因此历史波动率、回撤和 VaR/CVaR 是风险代理估计，而不是实际账户历史业绩或损失上限。阈值是用户配置的风险预算规则，监控只产生告警，不自动交易。

## 一键运行完整研究流程

统一入口按顺序执行行情下载与数据检查、组合风险及压力测试、动量基准回测、参数网格验证和滚动 walk-forward，并在全部必需步骤成功后生成综合报告。例子中的权重、日期与冲击仅用于展示命令格式，请根据实际投资组合和想要的测试区间修改：

```bash
python run_research.py \
  --holdings-csv data/private/holdings_snapshot.csv \
  --test-start 2023-01-01 \
  --lookbacks 63,126,252 \
  --skip-recents 5,21 \
  --top-ns 1,2,3 \
  --train-window 252 \
  --test-window 63 \
  --rebalance-freqs daily,weekly,monthly \
  --transaction-cost-bps 5 \
  --slippage-bps 5
```

如果需要强制下载行情，可增加 `--refresh`。只有你明确接受旧数据回退风险时，才使用 `--allow-stale-cache`。历史不足以支持 walk-forward 时，可显式加 `--skip-walk-forward`；其余阶段仍会执行，但这次报告将不会包含滚动验证结果。执行中任一必需的子步骤失败，统一入口会停止，不会把不完整流程标记为成功。

研究结果写入 `outputs/`，包括 `research_report.md` 和 `research_run_manifest.json`。请先阅读报告中的数据覆盖、成本口径与局限，再解读收益数字。

## 导入个人持仓

持仓 CSV 的必需列为 `ticker,shares,average_cost,snapshot_price`；可选列为 `name,risk_group,subgroup,currency,notes,quantity_source`。股票数量、平均成本和截图参考价应来自自己的券商记录，数据文件放在 `data/private/holdings_snapshot.csv`，该目录已被 `.gitignore` 排除。

通过一键入口导入后，程序会根据持仓代码下载并检查历史行情，用最近有效的未调整收盘价估算市值与组合权重，生成成本/盈亏表和单一仓位集中度提醒。如果同目录存在 `holdings_snapshot.csv.meta.json`，程序还会把行项目合计与券商截图总额对账，明确显示差异，而不是静默消除。

**重要：** 股数必须与券商端核实。截图没有直接显示持股数量时，不应仅凭反推数字执行交易。最近历史收盘价不是实时可成交报价；现金、债券、其他账户、汇率及融资负债未导入时，权重只代表导入的股票/ETF 仓位。

## MU / SKHY / DRAM 存储持仓决策模块

这个模块针对直接持有 MU、SKHY，同时持有存储 ETF DRAM 的重叠敞口。它使用最近有效的未调整收盘价计算实际股数市值；用调整收盘价计算 20/60 个交易日收益、20 日年化波动率和简单趋势状态；并把 DRAM 内部对 MU/SKHY 的估算持仓加入发行人穿透敞口。它不预测股票未来涨跌，也不连接券商下单。

运行完整研究流程并导入包含 MU、SKHY、DRAM 的本地持仓 CSV 后，结果会自动写入 `outputs/memory_allocation_summary.json`、`outputs/memory_position_diagnostics.csv` 和 `outputs/memory_allocation_scenarios.csv`；仪表盘也会显示“存储持仓”页。

也可以在已经下载好两份行情面板后单独运行：

```bash
python memory_allocation.py \
  --holdings-csv data/private/holdings_snapshot.csv \
  --prices data/raw/close_price_panel.csv \
  --adjusted-prices data/raw/adjusted_close_panel.csv \
  --max-memory-weight-pct 25 \
  --max-issuer-exposure-pct 12
```

当前默认策略参数是一个**可修改的保守风险预算起点**，不是历史回测得出的最优组合：
- 总存储主题（MU + SKHY + DRAM）占参考组合不超过 25%；
- 默认平衡方案目标权重：MU 8%、SKHY 6%、DRAM 5%，合计 19%；
- 同时输出 ETF 核心方案（仅 DRAM 15%）和直接持股方案（MU 8%、SKHY 8%，不持 DRAM），以便比较“ETF 分散”和“发行人集中”之间的取舍；
- 自动增加 `cap_only_preserve_etf` 最小调整方案：固定 DRAM 当前股数，在满足总主题与单发行人穿透敞口上限的前提下，选择保留市值最大的 MU/SKHY 整数股组合；若有同等解，再优先接近同比例减仓。
- 若当前股数高于目标股数，标记 `REDUCE_TO_TARGET`；若低于目标股数，默认标记 `HOLD_NO_AUTO_ADD`，不会自动补仓。
- 新增加仓许可门槛：主题至少低于上限 3 个百分点、单一发行人穿透敞口不超限，且 20/60 日趋势确认后，才标记 `STAGED_ADD_REVIEW_ONLY`；这仍是复核信号，不代表自动买入。

DRAM 穿透权重默认使用 2026-10-29 公布的最新可查持仓：按 10 月 29 日每日持仓，将 MU 直接持股及 swap 合计约 27.87%、SK hynix 直接持股、swap 及 ADR 合计约 21.82%。它们可能通过普通持股和掉期等多种方式形成，基金仓位会变动，需定期更新；这不是基金官方永久固定权重。若要修改平衡方案，可使用 `--target-weights MU=8,SKHY=6,DRAM=5`。建议在仪表盘“开始一次研究”中填写“账户总净资产”：用持仓 CSV 相同币种，按股票、ETF、现金及其他资产合计后扣除融资和负债。该数值会传入 `--portfolio-value`，作为存储主题总仓位和发行人穿透占比的分母；留空（0）时，百分比仅按导入的上市证券持仓计算，可能高估主题占整个账户的比例。

**使用边界：** 目标权重、单发行人上限和主题上限是策略假设，不等于买卖信号。趋势指标只帮助识别短期转弱，不代表基本面价值判断。执行调仓前仍需核对真实持股数量、最新基金穿透持仓、公司估值、财报、税费、流动性与保证金要求。

## 个股估值研究（DCF + 远期 PE）

仪表盘新增“个股估值”页，首批配置标的为 GOOG、NVDA、MU、SKHY。估值任务与组合研究任务使用独立的成功结果指针，估值失败不会覆盖上一次成功估值，也不会覆盖组合研究报告。

可在项目根目录运行 `python run_valuation.py --tickers GOOG,NVDA,MU,SKHY`。首次运行会从 Yahoo Finance 获取报价、远期 EPS、自由现金流、现金、负债和经营指标；默认缓存有效期为 7 天。强制刷新可增加 `--refresh`。只有明确接受价格与财务数据过期风险时，才使用 `--allow-stale-cache`。

DCF 与远期 PE 分别输出悲观、基准和乐观情景。买入观察线按基准估值与安全边际计算，卖出复核线按基准估值与人工设置的溢价计算；这些是估值复核区间，不是价格预测或自动交易信号。所有假设可在 `config/valuation_assumptions.json` 编辑，并应在每次财报发布后复核。

周期股 MU 的单期自由现金流可能处于周期高点或低点，应优先核对跨周期正常化现金流。SKHY 的 ADR 每股口径未经核验时会暂停每股估值；不得自行把韩国普通股 EPS/股数直接当作 ADR 口径。

估值原始快照缓存在 `data/fundamentals/`（已加入 Git 忽略规则），每次后台估值运行保存在 `outputs/runs/<task_id>/artifacts/`，成功后更新 `outputs/latest_valuation_success.json`。组合研究继续使用 `outputs/latest_success.json`，两类结果不会互相覆盖。

仪表盘的“投资决策复核”页按股票代码关联当前持仓与估值情景。只有当前导入的代码和股数与最近一次组合研究一致时，才展示那次研究的行情估值与权重；如果持仓已经变化，会提示重新运行，避免将旧结果误当作当前持仓。
## 本地仪表盘

安装可选依赖后，在项目根目录运行：

```bash
# macOS / Linux
bash start_dashboard.sh

# Windows PowerShell
.\start_dashboard.ps1

# GitHub Codespaces（在已安装 requirements-dashboard.txt 的环境中）
python -m streamlit run dashboard.py --server.address 0.0.0.0 --server.port 8501
```

Codespaces 启动后，在底部 Ports 面板打开 8501 端口。建议端口可见性保持 Private。启动脚本优先使用项目已有的 `.venv`，否则使用当前 Python 环境；仅在 Streamlit 缺失时安装可选依赖。

仪表盘以中文为主，研究入口放在页面主体，打开页面即可上传持仓或填写组合权重。研究入口提供“简易模式/专业模式”：简易模式只需确认持仓输入、对比基准和样本外检验起始日期，其余参数采用明确展示的默认值；专业模式开放回看窗口、训练/检验窗口、交易成本、滑点和数据刷新等设置。侧边栏可切换指标卡收益差值的颜色习惯（美股绿涨红跌 / A 股红涨绿跌），方向箭头与正负号会同时保留，避免只靠颜色传递信息。账户概览优先显示滚动样本外验证，组合风险页为波动率、最大回撤、夏普、索提诺、VaR 和 CVaR 提供解释与例子，并用规则模板生成大白话摘要。`research_report.md` 与仪表盘统一采用中文标题和指标术语。账户概览、组合风险、策略回测、稳健性检验、存储主题和研究报告分别显示在独立页面标签中。提交研究后由独立后台工作进程执行，页面不再同步等待整个研究流程；同一时刻只允许一个研究任务。任务状态和日志分别写入 `outputs/runs/<task_id>/status.json` 与 `worker.log`。

每次运行的研究产物写入独立的 `outputs/runs/<task_id>/artifacts/`。只有当进程成功退出且同时生成 `research_report.md` 和 `research_run_manifest.json` 后，才原子更新 `outputs/latest_success.json`。运行失败或进程意外退出时，上一次完整成功的报告仍然可用；仪表盘从该指针读取结果。首次部署时，如果还没有成功运行记录，会兼容读取旧版 `outputs/` 根目录下的结果。

可在侧边栏上传个人持仓 CSV，文件会保存在已忽略的 `data/private/` 文件夹。勾选“根据导入持仓自动计算组合权重”后即可按持仓运行，也可手工输入权重。请确认股数与券商记录一致，并注意没有导入的现金、融资负债和其他资产不会自动纳入组合权重。

此平台只用于研究，不连接券商、不发送真实订单。请不要将包含个人财务数据的服务公开部署到互联网；若需多人或公网访问，必须先增加身份认证、授权与数据隔离。

## 滚动 Walk-forward 验证

单次样本外切分仍可能受到选定日期的影响。滚动验证会在每轮中使用过去的训练窗口选择参数，并仅在紧随其后的测试窗口报告结果；测试窗口长度和训练窗口都以对齐后的收益观测数计。

```bash
python walk_forward_validation.py \
  --lookbacks 63,126,252 \
  --skip-recents 5,21 \
  --top-ns 1,2,3 \
  --train-window 252 \
  --test-window 63 \
  --selection-metric sharpe_ratio
```

当前版本要求至少有一整个训练窗口和一个测试窗口的共同收益历史；所有测试窗口不重叠。每轮以训练期指定指标选择参数，再输出后续测试期的策略和基准表现。多轮 walk-forward 仍不能消除多重试验、市场状态变化和幸存者偏差；建议保留额外未触碰的最终留出区间。

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

1. **数据与组合风险**：已实现行情质量检查、组合风险贡献、滚动风险和压力测试。
2. **策略回测 v1**：已实现横截面动量基线、信号滞后、基准比较及初步交易成本估算。
3. **滚动策略验证**：已增加 walk-forward 训练/测试切分，后续继续完善过拟合审计和策略对照。
4. **现实交易约束**：完善持仓权重漂移、成交价格、再平衡频率和最终平仓成本。
5. **研究质量控制**：加入更多市场阶段、资产池与策略对照，记录实验参数并检查幸存者偏差和多重试验。
6. **资产配置与持续监控**：组合约束优化、定时报告、异常检测和风险提醒。
7. **研究报告与模拟运行**：自动汇总结果后先纸面运行，并明确规则失效条件和风险预算；未完成审计前不接实盘交易接口。

---

本项目仅用于编程学习与量化研究，不构成投资、税务或法律建议。历史表现不代表未来结果。
