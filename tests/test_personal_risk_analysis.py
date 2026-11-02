"""Tests for the unified personal portfolio risk report."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from run_personal_risk_analysis import analyze_personal_risk, _write_report


def _synthetic_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=180)
    tickers = ["AAA", "BBB", "ETF1"]
    returns = pd.DataFrame(
        rng.normal(0.0004, 0.012, size=(len(dates), len(tickers))),
        index=dates,
        columns=tickers,
    )
    prices = 100.0 * (1.0 + returns).cumprod()
    holdings = pd.DataFrame([
        {"ticker": "AAA", "shares": 8, "risk_group": "semiconductor_related", "subgroup": "memory"},
        {"ticker": "BBB", "shares": 5, "risk_group": "internet_platform", "subgroup": "search_cloud_ai"},
        {"ticker": "ETF1", "shares": 2, "risk_group": "broad_tech_etf", "subgroup": "nasdaq_100"},
    ])
    return holdings, prices, prices.copy()


def _stress_profile() -> dict:
    return {
        "schema_version": 1,
        "scenarios": [
            {
                "name": "technology_shock",
                "label": "科技股下跌",
                "group_shocks": {
                    "semiconductor_related": -0.20,
                    "internet_platform": -0.12,
                    "broad_tech_etf": -0.15,
                },
            },
            {
                "name": "memory_shock",
                "label": "存储主题下跌",
                "subgroup_shocks": {"memory": -0.30},
                "ticker_shocks": {"AAA": -0.35},
            },
        ],
        "notes": ["synthetic assumptions for test only"],
    }


def test_unified_report_has_risk_contribution_rolling_correlation_and_stress() -> None:
    holdings, close, adjusted = _synthetic_inputs()
    result = analyze_personal_risk(
        holdings,
        close,
        adjusted,
        group_caps={
            "semiconductor_related": 0.50,
            "internet_platform": 0.50,
            "broad_tech_etf": 0.50,
        },
        subgroup_caps={
            "memory": 0.50,
            "search_cloud_ai": 0.50,
            "nasdaq_100": 0.50,
        },
        stress_profile=_stress_profile(),
        cash=None,
        fund_tickers=["ETF1"],
        minimum_correlation_observations=60,
        as_of_date=pd.Timestamp.today().date(),
    )
    summary = result["summary"]
    assert summary["cash_source"] == "not_provided_assumed_zero"
    assert summary["history_quality"] == "under_one_trading_year"
    assert summary["rolling_windows_used"] == [20, 60, 120]
    assert result["positions"]["risk_contribution_pct"].sum() == pytest.approx(100.0, abs=1e-5)
    assert result["correlations"].shape == (3, 3)
    assert result["correlation_observations"].to_numpy().min() >= 60
    assert set(result["stress"]["scenario"]) == {"technology_shock", "memory_shock"}
    assert (result["stress"]["estimated_pnl"] < 0).all()
    assert summary["lookthrough_funds_missing"] == ["ETF1"]


def test_cash_is_included_in_denominator_and_stress_has_zero_cash_shock() -> None:
    holdings, close, adjusted = _synthetic_inputs()
    result = analyze_personal_risk(
        holdings,
        close,
        adjusted,
        group_caps={
            "semiconductor_related": 0.60,
            "internet_platform": 0.60,
            "broad_tech_etf": 0.60,
        },
        subgroup_caps={"memory": 0.60, "search_cloud_ai": 0.60, "nasdaq_100": 0.60},
        stress_profile=_stress_profile(),
        cash=100.0,
        fund_tickers=[],
        minimum_correlation_observations=60,
        as_of_date=pd.Timestamp.today().date(),
    )
    assert result["summary"]["cash_source"] == "user_provided"
    assert result["summary"]["cash"] == pytest.approx(100.0)
    assert result["summary"]["cash_weight"] > 0
    assert result["stress"].shape[0] == 2


def test_report_writer_creates_private_local_outputs(tmp_path: Path) -> None:
    holdings, close, adjusted = _synthetic_inputs()
    result = analyze_personal_risk(
        holdings,
        close,
        adjusted,
        group_caps={
            "semiconductor_related": 0.60,
            "internet_platform": 0.60,
            "broad_tech_etf": 0.60,
        },
        subgroup_caps={"memory": 0.60, "search_cloud_ai": 0.60, "nasdaq_100": 0.60},
        stress_profile=_stress_profile(),
        cash=100.0,
        fund_tickers=[],
        minimum_correlation_observations=60,
        as_of_date=pd.Timestamp.today().date(),
    )
    _write_report(result, tmp_path)
    expected = [
        "personal_risk_summary.json",
        "personal_risk_positions.csv",
        "personal_risk_alerts.csv",
        "personal_risk_rolling.csv",
        "personal_risk_asset_correlations.csv",
        "personal_risk_correlation_observations.csv",
        "personal_risk_stress_tests.csv",
        "personal_portfolio_lookthrough.csv",
        "personal_portfolio_lookthrough_funds.csv",
        "personal_portfolio_lookthrough_summary.json",
        "personal_portfolio_risk_report.md",
    ]
    assert all((tmp_path / name).is_file() for name in expected)
    summary = json.loads((tmp_path / "personal_risk_summary.json").read_text(encoding="utf-8"))
    assert summary["position_count"] == 3
