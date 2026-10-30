# 美股量化研究：阶段一——数据与风险刻画

用于学习美股行情数据处理、风险指标和组合相关性分析的 Python 项目。当前版本可基于明确配置的风险预算输出可解释的仓位调整方案，但不预测未来收益、不自动下单，也不是实盘交易系统。

## 功能

- 下载 SPY、SOXX、NVDA、GOOGL 日线行情并缓存为 CSV。
- 检查数据日期、重复行、缺失值、OHLC 合法性和异常大幅日收益。
- 使用带下载元数据的本地缓存；缓存过期、覆盖范围不完整或缺失元数据时自动刷新。
- 计算日收益率、复合年化收益率（CAGR）、年化波动率、最大回撤和相对 SPY 的 Beta。
- 计算组合总收益、年化收益/波动率、Sharpe、Sortino、最大回撤、历史 VaR/CVaR，以及各持仓对组合波动率的贡献。
- 提供横截面动量 v1 策略、单期信号滞后、基准对比、交易成本/滑点假设和可复现回测输出。
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

## 参数网格与样本外验证

回测指标不能只看全区间结果。使用 `strategy_validation.py` 可以预先指定样本外区间，逐一比较不同参数，并输出训练期、测试期和逐年表现。示例：

```bash
python strategy_validation.py \
  --test-start 2023-01-01 \
  --lookbacks 63,126,252 \
  --skip-recents 5,21 \
  --top-ns 1,2,3
```

程序会报告所有参数组合，不会自动选出“最优参数”。建议先依据训练期设定参数，再把测试期作为未参与挑选的评估区间；测试期结果一旦被反复用于选参数，就不再是真正独立的样本外证据。当前模块是单次时间切分和逐年报告，后续仍需加入滚动 walk-forward、更多资产池和策略失效条件审查。

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
- 自动增加 `cap_only_preserve_etf` 最小调整方案：固定 DRAM 当前股数，只按比例缩减 MU/SKHY 直接持仓，使三项合计尽量回到主题上限内。
- 若当前股数高于目标股数，标记 `REDUCE_TO_TARGET`；若低于目标股数，默认标记 `HOLD_NO_AUTO_ADD`，不会自动补仓。
- 新增加仓许可门槛：主题至少低于上限 3 个百分点、单一发行人穿透敞口不超限，且 20/60 日趋势确认后，才标记 `STAGED_ADD_REVIEW_ONLY`；这仍是复核信号，不代表自动买入。

DRAM 穿透权重默认使用 2026-10-29 公布的最新可查持仓：按 10 月 29 日每日持仓，将 MU 直接持股及 swap 合计约 27.87%、SK hynix 直接持股及 swap 合计约 21.14%。它们可能通过普通持股和掉期等多种方式形成，基金仓位会变动，需定期更新；这不是基金官方永久固定权重。若要修改平衡方案，可使用 `--target-weights MU=8,SKHY=6,DRAM=5`。只有在提供 `--portfolio-value` 或从其他途径输入完整净资产（含现金及负债）时，输出权重才代表整个账户；否则百分比仅按导入的证券持仓计算。

**使用边界：** 目标权重、单发行人上限和主题上限是策略假设，不等于买卖信号。趋势指标只帮助识别短期转弱，不代表基本面价值判断。执行调仓前仍需核对真实持股数量、最新基金穿透持仓、公司估值、财报、税费、流动性与保证金要求。

## 本地仪表盘

安装可选依赖后，在项目根目录运行：

```bash
# macOS / Linux
bash start_dashboard.sh

# Windows PowerShell
.\start_dashboard.ps1
```

仪表盘从 `outputs/` 读取已有结果；可在侧边栏上传个人持仓 CSV，文件会保存在已忽略的 `data/private/` 文件夹。勾选“使用导入持仓自动计算组合权重”后即可启动完整研究流程，也可手工填写权重。它默认仅用于本机交互，不执行真实交易，也不会因为打开页面就自动下载行情。请不要将本地服务直接暴露到公网；若需要多人或公网访问，必须先部署身份认证、授权与数据隔离。

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
