"""Offline tests for the unified research runner and report renderer."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import run_research as app


def test_render_report_combines_risk_backtest_and_validation() -> None:
    report = app.render_report(
        generated_at="2026-01-01T00:00:00+00:00",
        weights="AAA=0.6,SPY=0.4",
        benchmark="SPY",
        data={
            "source": "synthetic",
            "requested_start": "2020-01-01",
            "requested_end_exclusive": "2026-01-01",
            "data_last_dates": {"AAA": "2025-12-31", "SPY": "2025-12-31"},
        },
        portfolio={
            "start_date": "2020-01-02", "end_date": "2025-12-31", "observations": 1500,
            "total_return": 0.25, "annualized_return": 0.04, "annualized_volatility": 0.15,
            "sharpe_ratio": 0.6, "sortino_ratio": 0.9, "max_drawdown": -0.2,
            "historical_var_one_period": 0.02, "historical_cvar_one_period": 0.03,
        },
        backtest={
            "start_date": "2021-01-01", "end_date": "2025-12-31", "observations": 1200,
            "transaction_cost_bps": 5, "slippage_bps": 5, "excess_total_return": 0.03,
            "annualized_turnover": 4.0,
            "strategy": {"total_return": 0.3, "annualized_return": 0.06, "annualized_volatility": 0.2,
                         "sharpe_ratio": 0.5, "sortino_ratio": 0.7, "max_drawdown": -0.25},
            "benchmark": {"total_return": 0.27, "annualized_return": 0.055, "annualized_volatility": 0.18,
                          "sharpe_ratio": 0.4, "sortino_ratio": 0.6, "max_drawdown": -0.22},
        },
        validation={"requested_combinations": 12, "successful_combinations": 10, "skipped_combinations": 2},
        walk_forward={
            "train_window_observations": 252, "test_window_observations": 63,
            "folds_successful": 8, "folds_total": 8, "oos_observations": 504,
            "out_of_sample_excess_total_return": 0.01,
            "out_of_sample_strategy": {"total_return": 0.2, "annualized_return": 0.07,
                "annualized_volatility": 0.18, "sharpe_ratio": 0.5, "max_drawdown": -0.15},
            "out_of_sample_benchmark": {"total_return": 0.19, "annualized_return": 0.065,
                "annualized_volatility": 0.16, "sharpe_ratio": 0.45, "max_drawdown": -0.13},
        },
    )
    assert "Portfolio risk" in report
    assert "Baseline momentum backtest" in report
    assert "Parameter sensitivity" in report
    assert "Rolling walk-forward evaluation" in report
    assert "Historical one-period VaR" in report
    assert "does not place trades" in report
    assert "25.00%" in report


def test_write_report_handles_missing_optional_outputs(tmp_path: Path) -> None:
    (tmp_path / "analysis_metadata.json").write_text(json.dumps({"source": "synthetic"}), encoding="utf-8")
    report_path = app.write_report(tmp_path, weights="AAA=1", benchmark="SPY")
    text = report_path.read_text(encoding="utf-8")
    assert report_path.exists()
    assert "Data coverage" in text
    assert "Portfolio risk" not in text
    assert "Rolling walk-forward evaluation" not in text


def test_write_report_reads_machine_readable_outputs(tmp_path: Path) -> None:
    data = {"source": "synthetic", "requested_start": "2020-01-01",
            "requested_end_exclusive": "2021-01-01", "data_last_dates": {"SPY": "2020-12-31"}}
    portfolio = {"start_date": "2020-01-01", "end_date": "2020-12-31", "observations": 250,
                 "total_return": 0.1, "annualized_return": 0.1, "annualized_volatility": 0.2,
                 "sharpe_ratio": 0.5, "sortino_ratio": 0.6, "max_drawdown": -0.1,
                 "historical_var_one_period": 0.02, "historical_cvar_one_period": 0.03}
    (tmp_path / "analysis_metadata.json").write_text(json.dumps(data), encoding="utf-8")
    (tmp_path / "portfolio_risk_summary.json").write_text(json.dumps(portfolio), encoding="utf-8")
    report_path = app.write_report(tmp_path, weights="SPY=1", benchmark="SPY")
    rendered = report_path.read_text(encoding="utf-8")
    assert "SPY=1" in rendered
    assert "10.00%" in rendered
    assert "Historical one-period CVaR" in rendered


def test_report_shows_imported_holdings_reconciliation_and_alerts() -> None:
    report = app.render_report(
        generated_at="2026-01-01T00:00:00+00:00",
        weights="AAA=0.6,BBB=0.4",
        benchmark="SPY",
        data=None,
        portfolio=None,
        backtest=None,
        validation=None,
        walk_forward=None,
        holdings_summary={
            "position_count": 2,
            "currency": "USD",
            "market_value": 1000.0,
            "cost_basis": 1020.0,
            "unrealized_pnl": -20.0,
            "unrealized_return_pct": -1.96,
            "largest_position_ticker": "AAA",
            "largest_position_weight_pct": 60.0,
            "largest_risk_group": "semiconductor_related",
            "largest_risk_group_weight_pct": 100.0,
        },
        holdings_alerts=[{
            "level": "warning", "message": "Risk theme 'semiconductor_related' weighs 100.0%.",
        }],
        holdings_reconciliation={
            "market_value_delta": 1.5,
            "unrealized_pnl_delta": 2.0,
            "within_tolerance": False,
        },
    )
    assert "Imported holdings and concentration controls" in report
    assert "Snapshot reconciliation" in report
    assert "verify shares" in report
    assert "semiconductor_related" in report
    assert "cash, bonds" in report
