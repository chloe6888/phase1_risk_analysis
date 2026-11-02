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
    monkeypatch.setattr(app.valuation_pipeline, "run", lambda *args, **kwargs: calls.append("valuation"))
    monkeypatch.setattr(
        app.decision_rules,
        "run_position_decision_review",
        lambda **kwargs: calls.append("decision"),
    )

    result = app.run_portfolio_suite(output_dir=tmp_path)

    assert result["status"] == "BLOCKED"
    assert result["stages"]["risk_analysis"]["status"] == "SKIPPED"
    assert result["stages"]["allocation_comparison"]["status"] == "SKIPPED"
    assert result["stages"]["valuation_research"]["status"] == "SKIPPED"
    assert result["stages"]["position_decision_review"]["status"] == "SKIPPED"
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

    universe = {
        "holding_tickers": ["AAA", "BBB", "ETF"],
        "symbols": {
            "AAA": {"instrument_type": "equity"},
            "BBB": {"instrument_type": "equity"},
            "ETF": {"instrument_type": "etf"},
        },
    }
    monkeypatch.setattr(app.data_pipeline, "load_universe", lambda path: universe)
    monkeypatch.setattr(app.valuation_pipeline, "load_config", lambda path: {
        "tickers": {"AAA": {}, "BBB": {}}
    })

    def fake_valuation(symbols: list[str], **kwargs: object) -> list[dict[str, object]]:
        calls["valuation"] = {"symbols": symbols, **kwargs}
        return [
            {"ticker": "AAA", "data_validation_status": "PASS"},
            {"ticker": "BBB", "data_validation_status": "BLOCKED"},
        ]

    def fake_decision(**kwargs: object) -> tuple[None, dict[str, object]]:
        calls["decision"] = kwargs
        return None, {
            "position_count": 3,
            "decision_counts": {"HOLD": 1, "ETF_LOOKTHROUGH_ONLY": 1, "BLOCK_ADD_CASH_UNVERIFIED": 1},
            "cash_source": "not_provided",
            "report_path": "decisions.md",
        }

    monkeypatch.setattr(app.data_pipeline, "run_pipeline", fake_data)
    monkeypatch.setattr(app.risk_analysis, "run_personal_risk_report", fake_risk)
    monkeypatch.setattr(app.allocation, "run_allocation_comparison", fake_allocation)
    monkeypatch.setattr(app.valuation_pipeline, "run", fake_valuation)
    monkeypatch.setattr(app.decision_rules, "run_position_decision_review", fake_decision)

    result = app.run_portfolio_suite(
        output_dir=tmp_path,
        valuation_output_dir=tmp_path / "valuation",
        decision_output_dir=tmp_path / "decisions",
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
    assert calls["valuation"]["symbols"] == ["AAA", "BBB"]
    assert calls["valuation"]["refresh"] is False
    assert calls["valuation"]["output_dir"] == tmp_path / "valuation"
    assert calls["decision"]["cash"] is None
    assert calls["decision"]["valuation_path"] == tmp_path / "valuation" / "valuation_summary.csv"
    assert result["stages"]["valuation_research"]["status"] == "WARNING"
    assert result["stages"]["position_decision_review"]["status"] == "COMPLETED"
    assert len(result["warnings"]) >= 6

    report = (tmp_path / "personal_portfolio_suite.md").read_text(encoding="utf-8")
    assert "ETF 穿透" in report
    assert "现金余额为零" in report
    assert "Walk-forward 样本外证据不足" in report
    assert "基本面估值" in report
    assert "加减仓规则复核" in report


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
    universe = {
        "holding_tickers": ["AAA"],
        "symbols": {"AAA": {"instrument_type": "equity"}},
    }
    monkeypatch.setattr(app.data_pipeline, "load_universe", lambda path: universe)
    monkeypatch.setattr(app.valuation_pipeline, "load_config", lambda path: {"tickers": {"AAA": {}}})
    monkeypatch.setattr(app.valuation_pipeline, "run", lambda symbols, **kwargs: [
        {"ticker": "AAA", "data_validation_status": "PASS"}
    ])
    monkeypatch.setattr(app.decision_rules, "run_position_decision_review", lambda **kwargs: (
        None, {"position_count": 1, "decision_counts": {"HOLD": 1}, "cash_source": "not_provided", "report_path": "decisions.md"}
    ))

    result = app.run_portfolio_suite(output_dir=tmp_path)

    assert result["status"] == "PARTIAL"
    assert result["stages"]["risk_analysis"]["status"] == "FAILED"
    assert result["stages"]["allocation_comparison"]["status"] == "PASS"
    assert result["stages"]["valuation_research"]["status"] == "PASS"
    assert result["stages"]["position_decision_review"]["status"] == "COMPLETED"
    assert "RuntimeError" in result["stages"]["risk_analysis"]["error"]


@pytest.mark.parametrize("invalid_cash", [-1.0, float("nan"), float("inf"), float("-inf")])
def test_invalid_cash_is_rejected_before_pipeline_execution(
    invalid_cash: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def should_not_run(**kwargs: object) -> None:
        raise AssertionError("data pipeline should not start with invalid cash")

    monkeypatch.setattr(app.data_pipeline, "run_pipeline", should_not_run)
    with pytest.raises(ValueError, match="non-negative USD"):
        app.run_portfolio_suite(cash=invalid_cash)


def test_valuation_failure_skips_decisions_to_avoid_stale_valuation_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app.data_pipeline, "run_pipeline", lambda **kwargs: {
        "status": "PASS", "position_count": 1, "common_return_observations": 260,
    })
    monkeypatch.setattr(app.risk_analysis, "run_personal_risk_report", lambda **kwargs: {
        "summary": {"report_markdown": "risk.md", "history_quality": "one_year_or_more", "alert_count": 0, "lookthrough_status": "not_applicable"}
    })
    monkeypatch.setattr(app.allocation, "run_allocation_comparison", lambda **kwargs: {
        "summary": {"report_path": "allocation.md", "common_return_observations": 260, "walk_forward": {"status": "multiple_oos_folds", "fold_count": 4}}
    })
    monkeypatch.setattr(app.data_pipeline, "load_universe", lambda path: {
        "holding_tickers": ["AAA"], "symbols": {"AAA": {"instrument_type": "equity"}}
    })
    monkeypatch.setattr(app.valuation_pipeline, "load_config", lambda path: {"tickers": {"AAA": {}}})
    monkeypatch.setattr(app.valuation_pipeline, "run", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("network failure")))
    monkeypatch.setattr(app.decision_rules, "run_position_decision_review", lambda **kwargs: pytest.fail("stale valuation must not be used"))

    result = app.run_portfolio_suite(output_dir=tmp_path)

    assert result["status"] == "PARTIAL"
    assert result["stages"]["valuation_research"]["status"] == "FAILED"
    assert result["stages"]["position_decision_review"]["status"] == "SKIPPED"
