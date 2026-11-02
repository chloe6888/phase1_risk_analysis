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
    ├── factor_risk.py
    ├── paper_trading.py
    ├── risk_monitor.py
    ├── monitoring_runner.py
    ├── walk_forward_validation.py
    ├── run_research.py
    ├── dashboard.py
    ├── holdings_manager.py
    ├── etf_lookthrough.py
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
    │   ├── test_factor_risk.py
    │   ├── test_paper_trading.py
    │   ├── test_risk_monitor.py
    │   ├── test_monitoring_runner.py
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

## 组合因子风险分解

新增 `factor_risk.py`，将组合历史收益对用户提供的因子代理收益做同期 OLS 回归，帮助拆解市场、半导体、成长、小盘、利率、黄金等代理因子的历史暴露和方差贡献。因子名称由输入价格面板的列名决定，不会默认假设某个因子来源。

输入包括：
- 组合资产调整收盘价面板（`--asset-prices`）；
- 用户选择的因子代理 ETF 调整收盘价面板（`--factor-prices`）；
- 组合权重 CSV，至少包含 `ticker,weight`；可直接使用组合优化输出的权重文件（`--weights-csv`）。

示例：

```bash
python factor_risk.py \
  --asset-prices data/raw/adjusted_close_panel.csv \
  --factor-prices data/raw/factor_proxy_prices.csv \
  --weights-csv outputs/portfolio_optimized_weights.csv
```

输出包括：
- `portfolio_factor_exposures.csv`：每个因子的回归 Beta、波动率、与组合收益相关性及方差贡献；
- `portfolio_factor_daily_returns.csv`：对齐后的组合收益、因子收益、拟合收益和残差；
- `portfolio_factor_summary.json`：R²、年化 Alpha、因子解释波动、残差波动、共线性诊断和样本区间；
- `portfolio_factor_report.md`：可读报告。

组合收益按输入权重在每个观测收盘重新平衡，未投入比例视为零收益现金。若代理因子高度相关，程序会提示 Beta 和单因子贡献不稳定；完全共线时会拒绝拟合。负的因子方差贡献会标记为风险抵消项，而不是错误地截断成零。

**重要边界：** 该分析是同期历史共变解释，不是因果模型或收益预测。用户提供的 ETF 仅是风险代理，不等同于官方 Fama-French 学术因子数据；当前组合权重也不是历史逐笔持仓重建。残差风险包含未纳入因子、个股事件、数据误差和模型设定误差。

## 仪表盘研究工具箱

本地 Streamlit 仪表盘新增“研究工具箱”页签，提供四类 CLI 模块的交互入口：

- **约束组合优化**：输入风险资产预算、单股上限、风险主题分组和主题上限，运行 `portfolio_optimizer.py`，并展示优化权重与风险贡献。
- **因子风险分解**：上传因子代理调整收盘价面板，并选择组合权重文件，运行 `factor_risk.py`，展示 Beta、R²、解释波动率和残差风险。
- **纸面交易**：上传目标权重和持仓 CSV，运行 `paper_trading.py`。可选 ADV 流动性限制和市场冲击；默认仅预览，必须显式勾选才会提交本地模拟账本。
- **风险监控**：根据当前持仓和本地价格面板运行 `risk_monitor.py`，展示单股/主题集中度、历史波动率、回撤与数据时效告警。

上传的工具输入保存在已被 Git 忽略的 `data/private/tool_inputs/`；工具生成物分开放在 `outputs/tools/`，不会覆盖统一研究流程的报告或估值输出。工具只在本机执行，不连接券商。若缺少价格面板，请先运行一次完整研究流程。

## 纸面交易模拟

新增 `paper_trading.py`，将优化器或人工编制的目标权重转换为离线模拟订单。目标 CSV 必须包含 `ticker,weight`；初始持仓 CSV 必须包含 `ticker,shares`。默认使用 `data/raw/close_price_panel.csv` 的最近可用收盘价，卖出先于买入，模拟成交价按配置滑点向不利方向调整，手续费按成交金额计算；现金不足时按比例缩减买单，整数股模式向下取整。

如果已通过行情下载流程获得带有 `Volume` 字段的 `data/raw/<TICKER>_daily.csv` 文件，还可以启用流动性约束：默认取最近 20 个有效交易日的平均成交量（ADV），将每笔模拟成交股数限制在 ADV × 最大参与率以内。未能成交的数量保留在订单记录中，不会假装全部成交。

仅预览订单：

```bash
python paper_trading.py \
  --targets-csv outputs/portfolio_optimized_weights.csv \
  --holdings-csv data/private/holdings_snapshot.csv \
  --cash 5000 \
  --prices data/raw/close_price_panel.csv \
  --volume-dir data/raw \
  --volume-window 20 \
  --max-volume-age-days 7 \
  --max-participation-rate 0.05 \
  --market-impact-coefficient-bps 15
```

默认不修改任何持仓账本。只有明确加入 `--commit-ledger` 时，才将模拟后的现金和股数写入本地 JSON 账本；后续运行会从账本读取模拟账户。输出包括 `paper_trade_orders.csv`、`paper_trading_holdings_after.csv` 和 `paper_trading_summary.json`。

**边界：** 这不是券商接口，不会发送订单。未提供 `--volume-dir` 时不启用 ADV 成交量上限；提供后，每笔成交量按最大参与率截断，额外滑点采用可配置的平方根参与率近似（系数不是经市场校准的参数）。它仍不模拟盘中成交、排队优先级、动态盘口、真实市场冲击曲线、税费、停牌或涨跌幅限制；纸面账本也不代表券商真实持仓。

## 组合风险监控

新增 `risk_monitor.py`，按当前持仓计算个股权重、一级行业/主题和二级子主题敞口，并检查单股上限、一级分组上限、二级子主题上限、历史年化波动率、历史最大回撤、单期 95% VaR/CVaR、历史不足和价格数据过期。报告同时输出基于当前权重与历史协方差矩阵的单项资产风险贡献百分比，贡献合计约为 100%；该指标是当前权重的静态边际风险分解，不是历史盈亏归因。共同日收益少于 126 个时会触发风险估计仅属初步结果的告警；126 至 251 个观测会在摘要中标记为不足一年，但不会制造独立告警事件。

```bash
python risk_monitor.py \
  --holdings-csv data/private/holdings_snapshot.csv \
  --prices data/raw/close_price_panel.csv \
  --adjusted-prices data/raw/adjusted_close_panel.csv \
  --cash 5000 \
  --single-name-limit 0.15 \
  --group-caps-json risk_profiles/balanced_group_caps.json \
  --subgroup-caps-json risk_profiles/balanced_subgroup_caps.json \
  --volatility-limit 0.30 \
  --drawdown-limit 0.20
```

持仓 CSV 可增加 `risk_group` 和 `subgroup` 列；也可以通过 `--groups-json`、`--group-caps-json` 与 `--subgroup-caps-json` 覆盖一级分组、二级子主题和上限。仓库提供 `risk_profiles/balanced_group_caps.json` 与 `risk_profiles/balanced_subgroup_caps.json` 作为平衡型的初始复核阈值；这些上限只触发告警，不自动生成卖出指令。输出包括 `risk_monitor_positions.csv`、`risk_monitor_alerts.csv`、`risk_monitor_daily_path.csv`、`risk_monitor_snapshot.json` 和 `risk_monitor_report.md`。

**历史口径限制：** 历史风险路径将当前股数应用于过去的调整收盘价，并假设现金余额不变；它不是按历史交易、入金、出金及公司行动重建的真实账户净值。因此历史波动率、回撤和 VaR/CVaR 是风险代理估计，而不是实际账户历史业绩或损失上限。阈值是用户配置的风险预算规则，监控只产生告警，不自动交易。

## 定时风险监控与每日/每周报告

新增 `monitoring_runner.py`，在现有只读风险监控基础上保存每次执行的时间戳快照，并比较前后两次结果，识别告警的首次触发、持续存在和解除。它不在 Python 进程里常驻运行，适合由 macOS/Linux cron、Windows 任务计划程序或服务器 systemd timer 定期调用。

每日执行一次：

```bash
python monitoring_runner.py \
  --holdings-csv data/private/holdings_snapshot.csv \
  --prices data/raw/close_price_panel.csv \
  --adjusted-prices data/raw/adjusted_close_panel.csv \
  --group-caps-json risk_profiles/balanced_group_caps.json \
  --subgroup-caps-json risk_profiles/balanced_subgroup_caps.json \
  --cash 5000 \
  --mode daily
```

每周生成近 7 日汇总：

```bash
python monitoring_runner.py \
  --holdings-csv data/private/holdings_snapshot.csv \
  --prices data/raw/close_price_panel.csv \
  --adjusted-prices data/raw/adjusted_close_panel.csv \
  --cash 5000 \
  --mode weekly \
  --history-days 7
```

Linux/macOS 可在 crontab 中配置（示例为每个工作日美东收盘后约 30 分钟；服务器时区需自行核实）：

```cron
30 16 * * 1-5 cd /path/to/phase1_risk_analysis && .venv/bin/python monitoring_runner.py --mode daily --holdings-csv data/private/holdings_snapshot.csv --cash 5000
0 17 * * 5 cd /path/to/phase1_risk_analysis && .venv/bin/python monitoring_runner.py --mode weekly --holdings-csv data/private/holdings_snapshot.csv --cash 5000
```

Windows 可在任务计划程序中设置工作目录为项目根目录、程序为虚拟环境的 `python.exe`、参数为 `monitoring_runner.py --mode daily --holdings-csv data/private/holdings_snapshot.csv --cash 5000`；每周另建一个任务，将模式改为 `weekly`。请按实际行情更新时间和服务器时区调整触发时间，避免在日线数据尚未更新时误判为数据过期。

运行产物位于 `outputs/monitoring/`：
- `snapshots/`：每次运行独立的不可覆盖 JSON 快照；
- `latest_snapshot.json`：最近一次的完整风险、持仓和告警快照；
- `monitoring_state.json`：上次活跃告警状态，用于识别触发、持续和解除；
- `latest_alert_events.json`：本次告警事件，可供后续通知渠道读取；
- `alert_event_history.jsonl`：跨运行追加保存的告警事件历史，包含快照 ID，可供通知消费者去重和补偿处理；
- `monitoring_daily_report.md`：每日风险快照报告；
- `monitoring_weekly_summary.json` 与 `monitoring_weekly_report.md`：最近指定日数内已有快照的统计。

**重要边界：** 监控程序只生成本地快照和告警事件文件，不会自动发送邮件、聊天消息或下单。若机器关机、行情未更新或计划任务失败，该时间段不会有监控快照；周报会显示实际保存的快照数量和覆盖时间，而不会伪造连续监控。当前历史风险路径仍以当前股数回放历史调整价格，是风险代理，不是历史真实账户净值。

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

## DRAM ETF 穿透持仓导入与来源审计

新增 `etf_lookthrough.py`，用于导入基金管理人下载的 DRAM 日度持仓 CSV，按发行人聚合直接股票、总收益互换和 ADR 等持仓行，并保存日期、来源名称、原始文件 SHA-256、权重合计及未映射行告警。默认识别 Micron（MU）和 SK hynix（SKHY）；其他名称可以通过 JSON issuer map 扩展。

Roundhill 官方 DRAM 页面提供 `Download CSV` 持仓入口；官方说明其 Top Holdings 权重会合并股票与总收益互换敞口，且 Top Holdings 一般不包含现金管理类持仓。建议下载完整持仓 CSV，不要只抄 Top Holdings。导入时必须明确填写文件对应的持仓日期。

```bash
python etf_lookthrough.py \
  --holdings-csv data/private/DRAM_daily_holdings.csv \
  --as-of 2026-10-29 \
  --source-name "Roundhill issuer holdings CSV"
```

模块会生成带日期的原始标准化持仓、发行人分项敞口 CSV 和 JSON 快照。随后可以将 JSON 快照传入存储持仓决策模块：

```bash
python memory_allocation.py \
  --holdings-csv data/private/holdings_snapshot.csv \
  --dram-lookthrough-json outputs/etf_holdings/DRAM_lookthrough_2026-10-29.json
```

该快照会覆盖手工默认比例，并把来源日期、来源名称和原始 CSV SHA-256 写入 `memory_allocation_summary.json`。若完整持仓总权重不在 95%–105% 范围内，或存在未映射的非现金持仓，程序会明确告警；不会把未识别项目静默计入 MU/SKHY。现金和现金等价物会单独统计。

**使用边界：** 导入是基于用户从官方页面下载的文件，不依赖未公开或未经确认的固定 API 地址，也不在后台自动抓取。CSV 文件日期必须与真实持仓日期一致；完成导入后仍应检查未映射行、总权重和证券类型。发行人合计将直接股票、掉期和 ADR 归入同一经济发行人风险敞口，但这些工具的法律和交易属性并不相同。

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

## 估值数据质量与口径核验

新增 `fundamentals_validation.py`，在估值模型计算前独立核验报价和财报日期、缓存新鲜度、DCF 现金流币种、EPS 币种、每股单位确认状态，以及远期 EPS / 自由现金流 / 股数的可用性。

核验分成 DCF 与 PE 两个独立 gate：例如报价为 USD、报告 FCF 为 KRW 时只暂停未换汇的 DCF；只有 EPS 币种和 ADS 每股单位也存在问题时才暂停 PE。若数据源将 trailing EPS 作为 forward EPS 缺失时的回退值，核验器会拒绝将它用于远期 PE。市值与“股价 × 股数”差异较大时会提示检查股数及 ADR 单位，不会静默替用户换算。

运行估值时可以调整日期阈值：

```bash
python run_valuation.py \
  --tickers GOOG,NVDA,MU,SKHY \
  --max-quote-age-days 7 \
  --max-financial-age-days 180 \
  --max-cache-age-hours 168
```

新增输出：

| 文件 | 内容 |
|---|---|
| `outputs/valuation_data_quality.csv` | 每个标的的核验状态、DCF/PE gate 和主要发现 |
| `outputs/valuation_validation_summary.json` | 完整核验记录、阈值、日期年龄和发现代码 |

当对应 gate 未通过时，估值结果中的相关情景值与观察线会被清空，而不是仅给出提示后仍展示可能误用的价格区间。核验是内部一致性检查，不等同于从公司原始财报独立审计供应商数据；DRAM 等周期股仍需人工确认正常化 FCF。

## 个股估值研究（DCF + 远期 PE）

仪表盘新增“个股估值”页。默认估值范围现在对应实际持有的七只经营性公司：AVGO、GOOG、LRCX、MRVL、MU、OKTA、SKHY。NVDA 保留为可选研究标的，但不属于当前组合默认估值列表。DRAM 和 QQQ 是 ETF，不使用公司 DCF/PE；应通过带日期的 ETF 持仓文件分析底层发行人敞口。估值任务与组合研究任务使用独立的成功结果指针，估值失败不会覆盖上一次成功估值，也不会覆盖组合研究报告。

可在项目根目录运行 `python run_valuation.py`，默认分析配置中的七只公司持仓；也可显式运行 `python run_valuation.py --tickers AVGO,GOOG,LRCX,MRVL,MU,OKTA,SKHY`。首次运行会从 Yahoo Finance 获取报价、远期 EPS、自由现金流、现金、负债和经营指标；默认缓存有效期为 7 天。强制刷新可增加 `--refresh`。只有明确接受价格与财务数据过期风险时，才使用 `--allow-stale-cache`。

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

## 真实账户数据验收（本机执行）

在用个人持仓数据解释风险和调整仓位前，先运行本地验收器。它只读取本机文件，不联网、不上传 CSV，也不会连接券商或下单。

~~~bash
python real_data_acceptance.py \
  --holdings-csv data/private/holdings_snapshot.csv \
  --close-prices data/raw/close_price_panel.csv \
  --adjusted-prices data/raw/adjusted_close_panel.csv
~~~

默认读取 data/private/holdings_snapshot.csv、data/raw/close_price_panel.csv 和 data/raw/adjusted_close_panel.csv，并将脱敏结果写入 outputs/acceptance/real_data_acceptance.md 与 real_data_acceptance.json。报告不包含股数、成本价、市值或账户余额。若行情文件尚未覆盖所有持仓标的，可先按实际代码补齐行情：

~~~bash
python phase1_risk_analysis.py --tickers SPY,SOXX,NVDA,GOOGL,MU,SKHY,DRAM
~~~

请将示例代码替换为持仓 CSV 中的真实代码；数据下载成功后再运行验收器。也可以使用 --as-of-date YYYY-MM-DD 重现某个历史验收日期，或用 --max-staleness-days 和 --minimum-common-observations 调整数据门槛。

**验收判定：**
- PASS：基础字段、价格覆盖、价格日期、共同历史和（如提供）券商汇总核对均通过。
- WARNING：可继续检查，但存在短历史、单日异常跳变、集中度提醒或未提供券商汇总值等情况。
- BLOCKED：持仓字段/行情缺失、价格过期、共同历史不足、券商汇总差异超限，或币种口径未确认。阻断时不要继续把结果解释为完整账户风险。

**币种限制：** 当前组合导入器只支持单一计价币种。持仓 CSV 必须包含 currency 列并为每笔持仓明确填写报价币种，例如美股为 USD、港股为 HKD；缺失/空白币种会阻断验收，已识别的非美元交易所后缀还会校验币种是否匹配。若同一账户同时有 USD 与 HKD 等持仓，本阶段会阻断组合级合并分析。下一步必须先实现有日期依据、可追溯的 FX 转换与基准币种核算，不能把两种货币的名义价格直接相加。

如需与券商汇总值对账，可在 holdings_snapshot.csv.meta.json 中本地保存 reported_market_value、reported_unrealized_pnl 和可选 reconciliation_tolerance。该文件同样属于私人数据，勿提交到 Git。只有提供了券商汇总值，验收器才会进行对应对账；没有汇总值会显示警告，而不会误报为已完成对账。



## 个人真实组合：一键行情更新与本地验收

项目附带 `config/portfolio_universe.json`，其中只保存研究标的清单、计价币种和风险分类，不保存个人持仓数量或成本。默认研究范围是九项持仓，加上 `SPY`（市场基准）和 `SOXX`（半导体行业参照）。

1. 将个人持仓 CSV 放入 `data/private/holdings_snapshot.csv`。该文件至少需要 `ticker, shares, average_cost, snapshot_price, currency, risk_group, subgroup` 字段。`snapshot_price` 仅为兼容现有导入格式的备用快照；一旦读取到未调整价格面板，组合市值会使用面板中的最新有效 `Close` 重估。
2. 安装依赖：`python -m pip install -r requirements.txt`。
3. 在 Codespaces 终端执行：`python run_personal_portfolio.py --refresh`。
4. 阅读 `outputs/acceptance/real_data_acceptance.md`。`BLOCKED` 表示不能进入正式风险分析；`WARNING` 表示数据可读但有需披露的限制。行情缓存位于 `data/raw/`，个人持仓位于 `data/private/`，二者都被 Git 忽略。

可使用 `python run_personal_portfolio.py --use-cache` 复用未过期且覆盖范围符合标的上市历史的缓存；对于配置中声明了 `regular_way_start_date` 的新上市证券（如 SKHY），缓存只要求与下载元数据及最新日期一致，不会因为其无法覆盖 2015 年之前的上市历史而每天重复下载。数据更新使用 yfinance/Yahoo Finance，适合研究和学习，不是审计级行情服务。此命令只发送标的代码请求行情，不会上传持股数量、成本、账户余额或私有 CSV。

### 共同历史长度说明

`SKHY` 是 2026-07-13 开始常规交易的 Nasdaq ADR（1 份 ADR 代表 10 股韩国普通股）。因此，九项持仓的共同收益历史会被它的上市时间限制；本项目的最低验收门槛为 60 个共同日收益观测，目标研究长度为 126 个，约一年参考长度为 252 个。低于 60 个会阻断验收；60 至 125 个可以进入初步风险分析但会明确告警；达到 126 个仍会提示不足一年；达到 252 个才不因历史长度触发警告。不得将韩国本股的 KRW 价格直接拼接为 SKHY 的 USD 价格序列。

### ETF 穿透

`etf_lookthrough.py` 接受带明确 as-of 日期的 ETF 完整持仓 CSV，合并直接股票、TRS/掉期和 ADR 对发行人的敞口。DRAM 的持仓会随时间变化，因此应从 [Roundhill 官方 DRAM 页面](https://www.roundhillinvestments.com/etf/dram/)取得对应日期的完整持仓，再导入：

```bash
python etf_lookthrough.py --holdings-csv data/private/DRAM_holdings.csv --as-of YYYY-MM-DD --source-name "Roundhill official holdings"
```

导入后先检查未映射权重、现金权重和完整度，再用于组合穿透；不要将 ETF 里的 MU/SKHY 敞口与直接持仓简单重复相加。

## 第二阶段：个人组合统一风险报告与 ETF 穿透

在第一阶段数据验收通过后，运行：

```bash
python run_personal_risk_analysis.py
```

如果你已确认账户现金余额，可在命令中显式传入：

```bash
python run_personal_risk_analysis.py --cash 2500
```

如果暂时不提供现金，程序会临时按 0 处理，并在报告中明确标记“仅按导入的九项证券计算分母”；这不代表已核实账户现金为 0。

报告输出在 outputs/personal_risk/personal_portfolio_risk_report.md，并同时生成：
- 当前持仓权重、一级风险主题和二级子主题敞口；
- 基于调整收盘价的当前持仓风险贡献；
- 20、60、120 日滚动年化波动率和滚动回撤（历史不足的窗口不会填成 0）；
- 持仓收益相关性矩阵及每一对标的的共同观测数，默认至少 60 个共同日收益才显示相关系数；
- 95% 单期历史 VaR/CVaR 和可编辑的静态压力场景；
- 直接持股与已导入 ETF 底层发行人的合并敞口。

### DRAM 与 QQQ 的 dated look-through

ETF 底层敞口必须使用与披露日期对应的 holdings CSV；禁止将某日的固定比例当成永久比例。官方数据入口：
- [Roundhill DRAM ETF 官方页面](https://www.roundhillinvestments.com/etf/dram/)
- [Invesco QQQ Trust 官方页面及 Holdings Export](https://www.invesco.com/us/en/financial-products/etfs/invesco-qqq-trust-series-1.html)

将官方持仓文件分别放入本地 data/private/，然后使用文件披露日期导入。DRAM 默认识别 MU 和 SK hynix；QQQ 使用项目内的 config/qqq_issuer_patterns.json，只映射本组合关心的发行人：

```bash
python etf_lookthrough.py --fund-ticker DRAM --holdings-csv data/private/DRAM_holdings.csv --as-of YYYY-MM-DD --source-name "Roundhill official holdings"

python etf_lookthrough.py --fund-ticker QQQ --holdings-csv data/private/QQQ_holdings.csv --as-of YYYY-MM-DD --source-name "Invesco QQQ official holdings" --issuer-map-json config/qqq_issuer_patterns.json
```

然后把两份生成的 JSON 快照传给风险报告：

```bash
python run_personal_risk_analysis.py \
  --lookthrough-snapshot outputs/etf_holdings/DRAM_lookthrough_YYYY-MM-DD.json \
  --lookthrough-snapshot outputs/etf_holdings/QQQ_lookthrough_YYYY-MM-DD.json
```

只导入一只 ETF 时，报告会把另一只 ETF 标记为“缺少快照”；快照超过 14 个自然日会标记过期。未被 issuer map 识别的 ETF 持仓会计入 unmapped weight，不会假装已完成全 ETF 穿透。QQQ 内 GOOG/GOOGL 归为同一 Alphabet 发行人。合并后的发行人敞口是重叠口径，不能把各发行人比例再相加当成组合权重。

### 风险解释边界

- 历史组合收益路径使用“当前股数 × 历史复权价格”构造代理，不会复原过去的成交、股数变化、入金和出金。
- 波动率、最大回撤、VaR/CVaR 受 SKHY ADR 上市历史较短影响；共同样本不足时，相关系数或滚动窗口会缺失并给出样本数。
- 压力情景是用户可编辑的单期静态冲击假设，不是价格预测，也不模拟危机时相关性变化。
- 发行人穿透只显示 issuer map 识别的标的；其他 ETF 成分会保留在 unmapped weight 中。
- 所有输出均在本地 ignored 的 outputs/ 下；私有持仓 CSV 和 ETF 原始持仓文件不得提交到 GitHub。系统只生成研究报告，不连接券商、不执行真实订单。

默认的三个平衡压力情景位于 risk_profiles/balanced_stress_scenarios.json，可根据风险预算修改。它们是研究假设，不是回撤上限或交易指令。


## 第三阶段：实际持仓、平衡、保守与最低方差比较

在第一阶段行情数据验收通过后执行：

```bash
python portfolio_allocation_comparison.py
```

如果你已核实账户现金余额，可显式补充现金：

```bash
python portfolio_allocation_comparison.py --cash 2500
```

不提供现金时，当前实际持仓权重以导入的九项证券市值为分母，现金仍标记为“未提供”，不是推断账户现金为 0。目标组合里的剩余预算则单独建模为现金（现金日收益假设为 0）。

方案包括：
- 当前实际持仓：按确认股数 × 最新未复权 Close 计算现值权重；
- 平衡型（受约束等权）：85% 风险资产预算，15% 目标现金；
- 平衡型（历史最低方差）：在相同 85% 风险资产预算与分层风险上限下，最小化历史协方差风险；
- 保守型（受约束等权）：70% 风险资产预算，30% 目标现金，并收紧单股、主题和子主题上限。

分层配置约束位于 risk_profiles/personal_allocation_profiles.json。优化器现在支持单股上限、一级风险主题上限、二级子主题上限及总风险资产预算；子主题必须归属于唯一的一级主题。各方案输出在 outputs/allocation_comparison/，包括权重、主题敞口、风险贡献、历史风险指标、相对实际持仓的单边换手估计和 Walk-forward 样本外记录。

Walk-forward 默认使用 60 个共同日收益训练、随后 21 个交易日测试，训练窗口严格截止在测试开始前一个交易日。若少于 3 个样本外区段或总样本外观测少于 60，报告会标记为 limited evidence。由于 SKHY Nasdaq ADR 历史较短，当前阶段很可能只有少量样本外区段；不应将这类结果解释为稳定统计证据。

### 解释边界
- 最低方差模型只优化历史协方差，不预测未来收益，也不代表最低回撤或更高收益。
- 平衡/保守方案是受约束等权的参考配置，现金比例是目标预算，不是从账户读取的真实现金。
- 比较采用固定权重的历史日收益代理，不复原实际历史调仓；Walk-forward 才按测试区段滚动拟合，但仍未计入交易费、税费、滑点或市场冲击。
- 配置比较只生成研究报告，不连接券商、不执行真实订单；最终执行前需核对券商可交易整手/碎股规则和价格。


### 当前持仓估值范围与假设纪律

config/valuation_assumptions.json 默认范围包含七只实际持有的经营性股票：AVGO、GOOG、LRCX、MRVL、MU、OKTA 和 SKHY；NVDA 保留为可选研究标的，不进入默认组合估值。DRAM、QQQ 是基金份额，不是经营性公司；它们的研究入口是基金持仓穿透和发行人风险敞口，不是将 ETF 的财务字段套入公司 DCF/PE。配置中的增长率、折现率、终值增长率和合理 PE 均为人工设定的敏感性分析起点，不是分析师一致预期或目标价。

AVGO 兼具半导体与基础设施软件业务，LRCX、MRVL、MU、SKHY 按周期性行业处理；在没有跨周期正常化 FCF 前，DCF 结果必须结合周期分析，置信度保持低。OKTA 的 PE 依赖供应商能否提供正数、有效的 forward EPS；若缺失或非正则禁用 PE。SKHY 的 1 ADS 对应 10 股韩国普通股，其 ADR 每 ADS 的 EPS/稀释股数单位仍未核验，因此 DCF 与 PE 均保持暂停；不能自行把普通股 KRW 财务数值当作 ADS USD 每股数值。只有取得可追溯证据后，才允许修改 per_share_basis_verified。


## 第五阶段：规则化加仓、持有与减仓复核

先完成行情验收、组合风险报告和估值快照，再执行：

\`\`\`bash
python run_position_decision_review.py
\`\`\`

只有在你已核实账户可用现金时，才显式传入：

\`\`\`bash
python run_position_decision_review.py --cash 2500
\`\`\`

规则配置位于 `risk_profiles/position_decision_policy.json`，报告输出在 `outputs/position_decisions/`。核心门槛：

- 任一单股、一级主题或二级子主题超过上限：标记“减仓/再平衡复核”，不是自动卖出。
- 进入“加仓研究观察”前，DCF 和远期 PE 必须同时有有效价格线，估值数据日期/财务日期必须合格，本地收盘价与估值供应商报价偏差不超过配置值，估值置信度至少为中等。
- 现金没有明确提供时，系统会阻止加仓建议，并标注“未核实”，不会把缺失现金伪装成 0。
- DRAM、QQQ 不使用公司 DCF/PE，标为 ETF 穿透分析；若主题或子主题超限仍可触发组合层面的复核。

“最大可加仓金额”只是现金余额与单股/主题/子主题剩余额度共同约束下的理论上限，不是下单金额、目标仓位或投资建议。阈值和估值假设都需要在每次财报与市场数据更新后重新检查。该模块没有券商连接，不提交订单，也不保证收益。


## 第六阶段：Streamlit 个人研究工作流

启动仪表盘：

```bash
streamlit run dashboard.py
```

首页导航已增加“资产配置比较”，并在以下页面加入本地运行入口：
- “投资决策复核”：运行加仓/持有/减仓规则复核；
- “组合风险”：生成统一风险报告，覆盖滚动风险、相关性、VaR/CVaR、情景压力和 ETF 穿透状态；
- “资产配置比较”：对照当前持仓、平衡型受约束等权、平衡型最小方差和保守型受约束等权。

每个面板的现金复选框默认不勾选。只有明确核实账户现金后，才勾选并填写金额；不填写时系统按“现金未提供”处理，不代表账户现金已核实为 0。所有按钮调用项目内白名单脚本，运行在当前 Codespace，本地报告写入 outputs 目录；不会上传 CSV 或提交订单。

建议的实际操作顺序：
1. 在 Codespaces 文件区确认 data/private/holdings_snapshot.csv 已更新；
2. 运行 `python run_personal_portfolio.py --refresh` 验收最新行情；
3. 运行 `python run_valuation.py --refresh` 刷新七只持有公司估值（SKHY 仍会因每 ADS 财务单位未核验而被阻断）；
4. 分别运行组合风险、资产配置比较和规则复核，或直接使用仪表盘对应按钮；
5. 导入最新 DRAM/QQQ 官方持仓后，再重复运行组合风险报告，以获得真实日期的底层发行人穿透结果。

页面中的规则结论、配置比较和风险测算都是研究复核工具，不是交易建议或收益承诺。
