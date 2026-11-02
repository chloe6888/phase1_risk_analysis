"""Tests for the guarded local portfolio research runner."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import run_personal_portfolio_suite as app


def test_blocked_acceptance_skips_risk_and_allocation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(app.data_pipeline, "run_pipeline", lambda **kwargs: {
        "status": "BLOCKED",
        "as_of_date": "2026-11-02",
        "position_count": 9,
        "common_return_observations": 42,
        "report_markdown": "acceptance.md",
        "report_json": "acceptance.json",
    })
    monkeypatch.setattr(
        app.risk_analysis,
        "run_personal_risk_report",
        lambda **kwargs: calls.append("risk"),
    )
    monkeypatch.setattr(
        app.allocation,
        "run_allocation_comparison",
        lambda **kwargs: calls.append("allocation"),
    )

    result = app.run_portfolio_suite(output_dir=tmp_path)

    assert result["status"] == "BLOCKED"
    assert result["stages"]["risk_analysis"]["status"] == "SKIPPED"
    assert result["stages"]["allocation_comparison"]["status"] == "SKIPPED"
    assert calls == []
    manifest = json.loads((tmp_path / "personal_portfolio_suite.json").read_text(encoding="utf-8"))
    assert manifest["privacy"]["share_quantities_included_in_suite_manifest"] is False
    assert manifest["privacy"]["cost_basis_included_in_suite_manifest"] is False


def test_warning_acceptance_runs_risk_and_allocation_and_preserves_unknown_cash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    def fake_data(**kwargs: object) -> dict[str, object]:
        calls["data"] = kwargs
        return {
            "status": "WARNING",
            "as_of_date": "2026-11-02",
            "position_count": 9,
            "common_return_observations": 80,
            "short_history_tickers": ["SKHY"],
            "report_markdown": "acceptance.md",
            "report_json": "acceptance.json",
        }

    def fake_risk(**kwargs: object) -> dict[str, object]:
        calls["risk"] = kwargs
        return {
            "summary": {
                "report_markdown": "risk.md",
                "history_quality": "under_one_trading_year",
                "portfolio_return_observations": 80,
                "alert_count": 3,
                "lookthrough_status": "partial_or_missing_or_stale",
                "lookthrough_funds_missing": ["DRAM", "QQQ"],
            }
        }

    def fake_allocation(**kwargs: object) -> dict[str, object]:
        calls["allocation"] = kwargs
        return {
            "summary": {
                "report_path": "allocation.md",
                "common_return_observations": 80,
                "walk_forward": {
                    "status": "limited_evidence",
                    "fold_count": 1,
                    "total_out_of_sample_observations": 20,
                },
            }
        }

    monkeypatch.setattr(app.data_pipeline, "run_pipeline", fake_data)
    monkeypatch.setattr(app.risk_analysis, "run_personal_risk_report", fake_risk)
    monkeypatch.setattr(app.allocation, "run_allocation_comparison", fake_allocation)

    result = app.run_portfolio_suite(
        output_dir=tmp_path,
        refresh=False,
        cash=None,
        snapshot_paths=[],
        as_of_date="2026-11-02",
    )

    assert result["status"] == "WARNING"
    assert result["cash_source"] == "not_provided"
    assert "不代表已核实的全部资产" in result["cash_note"]
    assert calls["data"]["refresh"] is False
    assert calls["risk"]["cash"] is None
    assert calls["allocation"]["cash"] is None
    assert calls["risk"]["as_of_date"] == "2026-11-02"
    assert calls["allocation"]["as_of_date"] == "2026-11-02"
    assert calls["risk"]["snapshot_paths"] == []
    assert len(result["warnings"]) >= 4

    report = (tmp_path / "personal_portfolio_suite.md").read_text(encoding="utf-8")
    assert "ETF 穿透" in report
    assert "现金余额为零" in report
    assert "Walk-forward 样本外证据不足" in report


def test_stage_failure_is_reported_as_partial_when_other_stage_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app.data_pipeline, "run_pipeline", lambda **kwargs: {
        "status": "PASS",
        "as_of_date": "2026-11-02",
        "position_count": 9,
        "common_return_observations": 260,
    })

    def fail_risk(**kwargs: object) -> None:
        raise RuntimeError("synthetic risk failure")

    monkeypatch.setattr(app.risk_analysis, "run_personal_risk_report", fail_risk)
    monkeypatch.setattr(app.allocation, "run_allocation_comparison", lambda **kwargs: {
        "summary": {
            "report_path": "allocation.md",
            "common_return_observations": 260,
            "walk_forward": {"status": "multiple_oos_folds", "fold_count": 4},
        }
    })

    result = app.run_portfolio_suite(output_dir=tmp_path)

    assert result["status"] == "PARTIAL"
    assert result["stages"]["risk_analysis"]["status"] == "FAILED"
    assert result["stages"]["allocation_comparison"]["status"] == "PASS"
    assert "RuntimeError" in result["stages"]["risk_analysis"]["error"]


def test_negative_cash_is_rejected_before_pipeline_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    def should_not_run(**kwargs: object) -> None:
        raise AssertionError("data pipeline should not start with invalid cash")

    monkeypatch.setattr(app.data_pipeline, "run_pipeline", should_not_run)
    with pytest.raises(ValueError, match="non-negative USD"):
        app.run_portfolio_suite(cash=-1.0)
