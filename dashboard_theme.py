"""Chinese-first visual theme for the personal investment research dashboard."""
from __future__ import annotations

from typing import Any


DASHBOARD_CSS = """
:root {
  color-scheme: light;
  --quant-ink: #17283f;
  --quant-muted: #62738a;
  --quant-line: #e3eaf2;
  --quant-surface: #ffffff;
  --quant-canvas: #f3f6fa;
  --quant-teal: #087f8c;
  --quant-teal-dark: #06636e;
  --quant-navy: #10243d;
}

html, body, [class*="css"] {
  font-family: Inter, "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC",
    "Source Han Sans SC", system-ui, sans-serif;
}
[data-testid="stAppViewContainer"] {
  background: var(--quant-canvas);
  color: var(--quant-ink);
}
[data-testid="stHeader"] {
  background: rgba(243, 246, 250, .88);
}
[data-testid="stMainBlockContainer"], .block-container {
  max-width: 1540px;
  padding-top: 1.15rem;
  padding-bottom: 3rem;
}
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }

/* Main title banner */
.st-key-dashboard_hero {
  position: relative;
  overflow: hidden;
  padding: 1.55rem 1.8rem 1.45rem;
  margin: .15rem 0 1.25rem;
  border: 1px solid rgba(255,255,255,.12);
  border-radius: 22px;
  background:
    radial-gradient(circle at 88% 12%, rgba(54, 182, 180, .27), transparent 27%),
    linear-gradient(118deg, #10243d 0%, #183b58 58%, #11616b 100%);
  box-shadow: 0 14px 34px rgba(16, 36, 61, .14);
}
.st-key-dashboard_hero h1 {
  margin: .3rem 0 .35rem;
  padding: 0;
  color: #f7fbff !important;
  font-size: clamp(1.75rem, 3vw, 2.35rem);
  font-weight: 750;
  letter-spacing: .015em;
}
.st-key-dashboard_hero [data-testid="stMarkdownContainer"] p {
  color: rgba(239, 247, 252, .84);
}
.quant-hero-eyebrow {
  color: #8de0d9;
  font-size: .76rem;
  font-weight: 750;
  letter-spacing: .15em;
}
.quant-hero-description {
  max-width: 850px;
  margin: .35rem 0 .8rem;
  color: rgba(239, 247, 252, .86);
  font-size: .96rem;
  line-height: 1.65;
}
.quant-hero-tags {
  display: flex;
  flex-wrap: wrap;
  gap: .5rem;
  margin-top: .85rem;
}
.quant-hero-tags span {
  padding: .28rem .65rem;
  border: 1px solid rgba(217, 245, 244, .22);
  border-radius: 999px;
  background: rgba(255,255,255,.08);
  color: #e4f7f5;
  font-size: .76rem;
}

/* Navigation */
.stTabs [data-baseweb="tab-list"] {
  gap: .42rem;
  flex-wrap: wrap;
  padding: .4rem .25rem .55rem;
  border-bottom: 1px solid var(--quant-line);
}
.stTabs [data-baseweb="tab"] {
  min-height: 2.65rem;
  padding: .35rem .9rem;
  border: 1px solid transparent;
  border-radius: 11px 11px 8px 8px;
  background: #eaf0f6;
  color: #4d6077;
  font-weight: 650;
}
.stTabs [data-baseweb="tab"]:hover {
  border-color: #c8d8e5;
  color: var(--quant-teal-dark);
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
  border-color: #b7dfe0;
  background: #e0f2f1;
  color: #075e68;
}
.stTabs [data-baseweb="tab-highlight"] {
  background: var(--quant-teal);
}

/* Sidebar */
section[data-testid="stSidebar"] {
  border-right: 1px solid rgba(255,255,255,.07);
  background: linear-gradient(180deg, #10243d 0%, #142e49 60%, #123b4b 100%);
}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] .stCaption {
  color: #e5edf6;
}
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
  color: #f1f6fb;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label {
  color: #edf5fb;
}
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea,
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
  border-color: #c7d4e1;
  background: #f8fbfd;
  color: #17283f;
}

/* Cards, metrics and content sections */
[data-testid="stMetric"] {
  min-height: 112px;
  padding: .95rem 1.05rem;
  border: 1px solid var(--quant-line);
  border-radius: 15px;
  background: var(--quant-surface);
  box-shadow: 0 4px 14px rgba(26, 48, 75, .045);
}
[data-testid="stMetricLabel"] {
  color: var(--quant-muted);
  font-size: .84rem;
  font-weight: 650;
}
[data-testid="stMetricValue"] {
  color: var(--quant-ink);
  font-weight: 760;
  letter-spacing: -.025em;
}
[data-testid="stMetricDelta"] {
  font-size: .82rem;
  font-weight: 650;
}
[data-testid="stVerticalBlockBorderWrapper"] {
  border-color: var(--quant-line);
  border-radius: 16px;
  background: rgba(255,255,255,.8);
}
h2, h3 {
  color: var(--quant-ink);
  letter-spacing: -.015em;
}
h2 {
  margin-top: 1.25rem;
  padding-bottom: .45rem;
  border-bottom: 1px solid var(--quant-line);
  font-weight: 740;
}
h3 { font-weight: 700; }
[data-testid="stCaptionContainer"] {
  color: var(--quant-muted);
  line-height: 1.55;
}

/* Controls */
.stButton > button, .stFormSubmitButton > button {
  min-height: 2.55rem;
  border-radius: 10px;
  font-weight: 680;
  transition: transform .12s ease, box-shadow .12s ease, border-color .12s ease;
}
.stButton > button[kind="primary"],
.stFormSubmitButton > button[kind="primary"] {
  border: 1px solid #087f8c;
  background: linear-gradient(110deg, #087f8c, #0b7180);
  color: #fff;
  box-shadow: 0 5px 13px rgba(8, 127, 140, .17);
}
.stButton > button[kind="primary"]:hover,
.stFormSubmitButton > button[kind="primary"]:hover {
  border-color: #06636e;
  background: #066f7a;
  box-shadow: 0 7px 18px rgba(8, 127, 140, .22);
  transform: translateY(-1px);
}
.stButton > button:not([kind="primary"]),
.stFormSubmitButton > button:not([kind="primary"]) {
  border-color: #d7e1eb;
  background: #fff;
  color: #30445c;
}
input, textarea, [data-baseweb="select"] > div {
  border-radius: 9px !important;
}
[data-testid="stFileUploader"] {
  padding: .55rem;
  border: 1px dashed #b8c9d9;
  border-radius: 13px;
  background: #f9fbfd;
}
[data-testid="stExpander"] {
  overflow: hidden;
  border: 1px solid var(--quant-line);
  border-radius: 13px;
  background: #fff;
}
[data-testid="stAlert"] {
  border-radius: 13px;
}

/* Tables and charts */
[data-testid="stDataFrame"], [data-testid="stTable"] {
  overflow: hidden;
  border: 1px solid var(--quant-line);
  border-radius: 12px;
  background: #fff;
}
[data-testid="stDataFrame"] [role="grid"] {
  border-radius: 12px;
}
[data-testid="stVegaLiteChart"], [data-testid="stPlotlyChart"],
[data-testid="stArrowVegaLiteChart"], [data-testid="stDeckGlJsonChart"] {
  overflow: hidden;
  border: 1px solid var(--quant-line);
  border-radius: 14px;
  background: #fff;
  padding: .35rem;
}
hr {
  border-color: var(--quant-line);
}

@media (max-width: 900px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; }
  .st-key-dashboard_hero { padding: 1.2rem 1.15rem; border-radius: 16px; }
  .stTabs [data-baseweb="tab-list"] { gap: .25rem; }
  .stTabs [data-baseweb="tab"] { padding: .3rem .58rem; font-size: .84rem; }
}
"""


def apply_dashboard_theme(st: Any) -> None:
    """Inject the dashboard's responsive, Chinese-first visual theme."""
    st.markdown(f"<style>{DASHBOARD_CSS}</style>", unsafe_allow_html=True)


def render_dashboard_header(st: Any) -> None:
    """Render a branded title area while keeping the page title accessible to Streamlit."""
    with st.container(key="dashboard_hero"):
        st.markdown(
            '<div class="quant-hero-eyebrow">个人投资研究 · 数据驱动的组合管理</div>',
            unsafe_allow_html=True,
        )
        st.title("个人投资研究工作台")
        st.markdown(
            '<p class="quant-hero-description">把持仓、风险、估值与策略验证放在同一处，'
            '让每一次投资调整都有数据依据。</p>'
            '<div class="quant-hero-tags">'
            '<span>中文操作界面</span><span>本地数据分析</span>'
            '<span>风险优先</span><span>不连接券商 · 不自动交易</span></div>',
            unsafe_allow_html=True,
        )
