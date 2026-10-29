# 美股量化研究：阶段一——数据与风险刻画

用于学习美股行情数据处理、基础风险指标和组合相关性分析的 Python 项目。当前版本只做研究分析，不生成买卖信号，也不是实盘交易系统。

## 功能

- 下载 SPY、SOXX、NVDA 日线行情并缓存为 CSV。
- 检查数据日期、重复行、缺失值和关键价格字段。
- 计算日收益率、复合年化收益率（CAGR）、年化波动率、最大回撤和相对 SPY 的 Beta。
- 输出日收益相关矩阵。
- 绘制价格、净值、回撤和相关矩阵图。
- 使用离线单元测试验证核心计算，不要求测试时访问行情 API。

行情来自 yfinance 对 Yahoo Finance 数据的访问，适合入门和探索，不应视为经过审计的机构级数据源。

## 项目结构

    phase1_risk_analysis/
    ├── README.md
    ├── requirements.txt
    ├── requirements-dev.txt
    ├── .gitignore
    ├── phase1_risk_analysis.py
    ├── tests/
    │   └── test_phase1_risk_analysis.py
    ├── .github/workflows/tests.yml
    ├── data/raw/       # 本地行情缓存；CSV 默认不提交
    └── outputs/        # 运行后生成的表格与图片；默认不提交

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

脚本默认优先读取本地缓存。首次运行时，缺失的标的会从数据源下载。需要强制更新所有标的时运行：

```bash
python phase1_risk_analysis.py --refresh
```

## 配置

在脚本顶部调整以下参数：

- `TICKERS`：研究标的列表，默认 SPY、SOXX、NVDA。
- `BENCHMARK`：Beta 基准，默认 SPY，必须包含在 `TICKERS` 中。
- `START_DATE`：下载起始日期，默认 2015-01-01。
- `END_DATE`：默认取纽约当地当前日期，下载接口的结束日期不包含当天。

建议第一次使用默认配置，成功运行后再添加标的。

## 输出文件

| 文件 | 内容 |
|---|---|
| `data/raw/<TICKER>_daily.csv` | 各标的整理后的行情缓存，包含可用 OHLCV 和公司行为字段 |
| `data/raw/adjusted_close_panel.csv` | 调整收盘价面板 |
| `outputs/data_quality_report.csv` | 数据完整性摘要 |
| `outputs/daily_returns.csv` | 调整收盘价计算的日收益率 |
| `outputs/risk_metrics.csv` | CAGR、年化波动率、最大回撤和 Beta 等指标 |
| `outputs/correlation_matrix.csv` | 日收益率相关矩阵 |
| `outputs/closing_prices.png` | 各标的价格序列 |
| `outputs/net_value.png` | 共同样本起点归一化为 100 的净值比较 |
| `outputs/drawdown.png` | 从历史高点计算的回撤曲线 |
| `outputs/correlation_matrix.png` | 相关矩阵热力图 |

## 计算口径与局限

- 历史总收益分析使用 Adj Close 作为调整后价格代理；这不是实际可成交的历史报价。
- CAGR 按实际日历时间跨度年化；年化波动率使用日收益样本标准差乘以平方根 252。
- Beta 以匹配日期的资产和 SPY 日收益率计算。
- 相关系数和 Beta 描述历史关系，不代表因果关系；压力期的相关性可能显著上升。
- 缺失价格不会自动填充为零收益。若下载数据存在缺口，必须检查数据质量报告。
- 当前版本不模拟交易成本、滑点、税费、做空约束或订单执行。
- 当前三个标的不代表整个美股市场；历史最大回撤也不是未来损失上限。不要仅凭这些指标直接推导安全仓位。

## 自动化测试

安装开发依赖后，在项目根目录运行：

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

测试使用人工构造的离线数据，不代表真实市场行情验证，也不替代数据源交叉核验。

仓库配置了 GitHub Actions，会在向 `main` 分支推送或提交针对 `main` 的 Pull Request 时运行这些测试。

## 后续路线

1. **数据与风险刻画**：验证数据、指标、净值和相关矩阵。
2. **横截面动量 v1**：定义因子和调仓规则，输出五组收益并手工追踪信号、分组与收益的时间关系。
3. **稳健性实验**：对照公开因子，改变窗口、股票池和权重，并记录所有实验。
4. **现实约束 v2**：加入换手率、成本和成交价格假设。
5. **回测审计**：检查未来函数、幸存者偏差、多重试验和样本外表现。
6. **试运行评估**：只有规则、失效条件、风险预算和监控机制书面明确后，才评估是否进行模拟交易。

---

本项目仅用于编程学习与量化研究，不构成投资、税务或法律建议。历史表现不代表未来结果。
