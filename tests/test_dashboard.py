"""Offline tests for dashboard data helpers and command construction."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

import dashboard as app


def test_read_json_returns_none_for_missing_and_validates_object(tmp_path: Path) -> None:
    assert app.read_json(tmp_path / "missing.json") is None
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"ok": True}), encoding="utf-8")
    assert app.read_json(path) == {"ok": True}
    path.write_text(json.dumps([1, 2]), encoding="utf-8")
    try:
        app.read_json(path)
    except ValueError as exc:
        assert "JSON object" in str(exc)
    else:
        raise AssertionError("Expected invalid JSON root to be rejected.")


def test_read_csv_handles_date_index_and_missing_outputs(tmp_path: Path) -> None:
    path = tmp_path / "values.csv"
    pd.DataFrame({"value": [1.0, 2.0]}, index=pd.to_datetime(["2024-01-01", "2024-01-02"])).to_csv(path)
    result = app.read_csv(path, date_index=True)
    assert result is not None
    assert result.index[0] == pd.Timestamp("2024-01-01")
    assert result["value"].tolist() == [1.0, 2.0]
    assert app.read_csv(tmp_path / "missing.csv") is None


def test_output_status_checks_key_artifacts(tmp_path: Path) -> None:
    (tmp_path / "research_report.md").write_text("# report", encoding="utf-8")
    status = app.output_status(tmp_path)
    assert status["research_report.md"] is True
    assert status["analysis_metadata.json"] is False
    assert len(status) >= 8


def test_make_pipeline_command_builds_argv_without_shell_interpolation() -> None:
    command = app.make_pipeline_command(
        weights="GOOGL=0.6,SPY=0.4",
        test_start=date(2023, 1, 1),
        benchmark="SPY",
        lookbacks="63,126",
        skip_recents="5,21",
        top_ns="1,2",
        refresh=True,
        skip_walk_forward=True,
    )
    assert command[0]
    assert "--weights" in command
    assert command[command.index("--weights") + 1] == "GOOGL=0.6,SPY=0.4"
    assert command[command.index("--test-start") + 1] == "2023-01-01"
    assert "--refresh" in command
    assert "--skip-walk-forward" in command
    # Values are separate argv tokens; no shell is involved in how the app runs commands.
    crafted = app.make_pipeline_command(
        weights="GOOGL=0.6; touch /tmp/should-not-exist",
        test_start="2023-01-01",
    )
    assert crafted[crafted.index("--weights") + 1] == "GOOGL=0.6; touch /tmp/should-not-exist"


def test_streamlit_dashboard_initializes_without_existing_outputs() -> None:
    from streamlit.testing.v1 import AppTest

    page = AppTest.from_file(str(app.ROOT / "dashboard.py"), default_timeout=20).run()
    assert not page.exception
    assert page.title
    assert page.title[0].value == "个人投资研究工作台"


def test_make_pipeline_command_uses_imported_holdings_instead_of_manual_weights() -> None:
    command = app.make_pipeline_command(
        weights=None,
        holdings_csv="data/private/holdings_snapshot.csv",
        test_start="2023-01-01",
        benchmark="SPY",
    )
    assert "--holdings-csv" in command
    assert command[command.index("--holdings-csv") + 1] == "data/private/holdings_snapshot.csv"
    assert "--weights" not in command


def test_make_pipeline_command_rejects_both_or_neither_portfolio_inputs() -> None:
    import pytest

    with pytest.raises(ValueError, match="exactly one"):
        app.make_pipeline_command(weights=None, holdings_csv=None, test_start="2023-01-01")
    with pytest.raises(ValueError, match="exactly one"):
        app.make_pipeline_command(weights="SPY=1", holdings_csv="data/private/holdings.csv", test_start="2023-01-01")


def test_dashboard_formats_missing_metrics_in_chinese() -> None:
    assert app._pct(None) == "暂无数据"
    assert app._number(None) == "暂无数据"


def test_display_frame_translates_common_columns_and_actions() -> None:
    frame = pd.DataFrame({
        "ticker": ["MU"],
        "shares": [10],
        "scenario": ["balanced"],
        "action": ["HOLD_NO_AUTO_ADD"],
    })
    shown = app._display_frame(frame)
    assert shown is not None
    assert shown.columns.tolist() == ["股票代码", "持股数量", "方案", "建议动作"]
    assert shown.iloc[0]["方案"] == "平衡方案"
    assert shown.iloc[0]["建议动作"] == "暂不自动加仓"


def test_report_view_translates_primary_headings() -> None:
    localized = app._localized_report(
        "# Personal Quantitative Research Report\n"
        "## Data coverage\n"
        "## Portfolio risk\n"
    )
    assert "# 综合投资研究报告" in localized
    assert "## 行情数据覆盖" in localized
    assert "## 组合风险分析" in localized


def test_task_step_and_value_labels_are_chinese_first() -> None:
    assert app._task_step_label("1/5 Data download, cache, and validation") == "第 1 步：行情下载与数据质量检查"
    assert app.VALUE_LABELS["REDUCE_TO_TARGET"] == "按目标仓位减仓复核"


def test_memory_explanations_are_translated_to_chinese() -> None:
    reason = app._localized_decision_reason(
        "Current MU position exceeds this scenario's target weight (8.0%)."
    )
    assert reason == "当前 MU 持仓高于此方案目标仓位（8.0%）。"
    summary = {
        "memory_sleeve_weight_pct": 28.0,
        "max_memory_weight_pct": 25.0,
        "memory_value_above_limit": 300.0,
        "overall_action": "REDUCE_MEMORY_SLEEVE",
    }
    assert "高于设定的 25.0% 上限" in app._memory_overall_reason(summary)
    assert "现金、债券、其他账户及融资负债" in app._memory_denominator_note(
        "Cash, bonds, other accounts and financing are not included."
    )


def test_holdings_match_snapshot_requires_same_tickers_and_share_counts() -> None:
    imported = pd.DataFrame({"ticker": ["MU", "GOOG"], "shares": [10, 2]})
    analyzed = pd.DataFrame({"ticker": ["GOOG", "MU"], "shares": [2, 10]})
    changed = pd.DataFrame({"ticker": ["GOOG", "MU"], "shares": [2, 9]})
    missing = pd.DataFrame({"ticker": ["GOOG"], "shares": [2]})
    assert app._holdings_match_snapshot(imported, analyzed)
    assert not app._holdings_match_snapshot(imported, changed)
    assert not app._holdings_match_snapshot(imported, missing)


def test_build_decision_review_marks_holdings_and_watchlist_separately() -> None:
    holdings = pd.DataFrame({
        "ticker": ["MU"],
        "shares": [10],
        "average_cost": [100.0],
        "snapshot_price": [120.0],
        "market_value": [1200.0],
        "weight_pct": [60.0],
        "currency": ["USD"],
    })
    valuation = pd.DataFrame({
        "ticker": ["MU", "GOOG"],
        "company_name": ["Micron", "Alphabet"],
        "currency": ["USD", "USD"],
        "current_price": [125.0, 180.0],
        "price_date": ["2026-10-29", "2026-10-29"],
        "dcf_base": [140.0, 200.0],
        "pe_base": [135.0, 195.0],
        "dcf_buy_below": [105.0, 150.0],
        "pe_buy_below": [101.0, 146.0],
        "dcf_sell_review_above": [168.0, 240.0],
        "pe_sell_review_above": [162.0, 234.0],
        "dcf_view": ["between_thresholds", "between_thresholds"],
        "pe_view": ["between_thresholds", "between_thresholds"],
        "overall_confidence": ["low", "medium"],
        "data_quality": ["complete_vendor_snapshot", "complete_vendor_snapshot"],
        "limitations": ["周期股，需正常化现金流", ""],
    })
    result = app._build_decision_review(holdings, valuation)
    assert result is not None
    by_ticker = result.set_index("股票代码")
    assert by_ticker.loc["MU", "持仓状态"] == "当前持仓"
    assert by_ticker.loc["MU", "持股数量"] == 10
    assert by_ticker.loc["GOOG", "持仓状态"] == "观察标的"
    assert pd.isna(by_ticker.loc["GOOG", "持股数量"])


def test_valuation_display_frame_translates_status_and_confidence() -> None:
    frame = pd.DataFrame({
        "ticker": ["MU"],
        "dcf_view": ["below_buy_threshold"],
        "pe_view": ["between_thresholds"],
        "overall_confidence": ["low"],
    })
    shown = app._valuation_display_frame(frame)
    assert shown is not None
    assert shown.loc[0, "DCF 价格区间判断"] == "低于买入观察线"
    assert shown.loc[0, "PE 价格区间判断"] == "处于两条观察线之间"
    assert shown.loc[0, "估值可信度"] == "低"


def test_risk_exposure_frame_sums_groups_to_one_hundred_percent() -> None:
    positions = pd.DataFrame({
        "risk_group": ["technology", "technology", "consumer"],
        "market_value": [30.0, 20.0, 50.0],
    })
    result = app._risk_exposure_frame(positions, "risk_group", "风险主题")
    assert round(float(result["持仓占比"].sum()), 8) == 100.0
    assert result.iloc[0]["风险主题"] == "consumer"
