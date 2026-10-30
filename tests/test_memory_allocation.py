"""Offline tests for MU/SKHY/DRAM allocation guardrails."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import memory_allocation as app


def sample_holdings() -> pd.DataFrame:
    return pd.DataFrame([
        {"ticker":"MU","name":"Micron","shares":10,"average_cost":95,"snapshot_price":100,"risk_group":"semiconductor_related","subgroup":"memory","currency":"USD"},
        {"ticker":"SKHY","name":"SK hynix","shares":20,"average_cost":48,"snapshot_price":50,"risk_group":"semiconductor_related","subgroup":"memory","currency":"USD"},
        {"ticker":"DRAM","name":"Roundhill Memory ETF","shares":40,"average_cost":20,"snapshot_price":25,"risk_group":"semiconductor_related","subgroup":"memory","currency":"USD"},
        {"ticker":"GOOG","name":"Alphabet","shares":10,"average_cost":190,"snapshot_price":200,"risk_group":"internet_platform","subgroup":"search_cloud_ai","currency":"USD"},
        {"ticker":"SPY","name":"SPDR S&P 500 ETF","shares":1,"average_cost":490,"snapshot_price":500,"risk_group":"broad_market_etf","subgroup":"broad_market","currency":"USD"},
    ])


def sample_prices(n: int = 90) -> pd.DataFrame:
    dates = pd.bdate_range("2026-05-01", periods=n)
    t = np.arange(n, dtype=float)
    frame = pd.DataFrame({
        "MU": 60 + 0.5*t + 3*np.sin(t/5),
        "SKHY": 30 + 0.2*t + 2*np.sin(t/7),
        "DRAM": 15 + 0.12*t + np.sin(t/8),
        "GOOG": 100 + 0.25*t + np.sin(t/4),
        "SPY": 450 + 0.3*t + np.sin(t/6),
    }, index=dates)
    expected_last_prices = {"MU": 100.0, "SKHY": 50.0, "DRAM": 25.0, "GOOG": 200.0, "SPY": 500.0}
    for ticker, target in expected_last_prices.items():
        frame[ticker] *= target / float(frame[ticker].iloc[-1])
    return frame


def test_calculate_memory_allocation_flags_concentration_and_overlap() -> None:
    summary, positions, decisions = app.analyze_memory_positions(
        sample_holdings(), sample_prices(), adjusted_close=sample_prices(),
        max_memory_weight_pct=25, max_issuer_exposure_pct=12,
    )
    assert summary["memory_sleeve_market_value"] == pytest.approx(3000)
    assert summary["imported_listed_positions_market_value"] == pytest.approx(5500)
    assert summary["memory_sleeve_weight_pct"] == pytest.approx(3000/5500*100)
    assert summary["overall_action"] == "REDUCE_MEMORY_SLEEVE"
    assert summary["lookthrough_exposure_values"]["MU"] == pytest.approx(1000 + 1000*0.2787)
    assert summary["lookthrough_exposure_values"]["SKHY"] == pytest.approx(1000 + 1000*0.2182)
    assert len(summary["issuer_alerts"]) == 2
    assert set(positions.index) == {"MU", "SKHY", "DRAM"}
    assert {"balanced", "etf_core", "direct_only"}.issubset(set(decisions["scenario"]))


def test_balanced_targets_calculate_reductions_without_auto_adds() -> None:
    _, _, decisions = app.analyze_memory_positions(sample_holdings(), sample_prices())
    balanced = decisions[decisions["scenario"] == "balanced"].set_index("ticker")
    assert balanced.loc["MU", "target_shares"] == 4
    assert balanced.loc["MU", "share_delta"] == -6
    assert balanced.loc["MU", "action"] == "REDUCE_TO_TARGET"
    assert balanced.loc["SKHY", "target_shares"] == 7
    assert balanced.loc["SKHY", "share_delta"] == -13
    assert balanced.loc["DRAM", "target_shares"] == 11
    assert balanced.loc["DRAM", "share_delta"] == -29
    etf_core = decisions[decisions["scenario"] == "etf_core"].set_index("ticker")
    assert etf_core.loc["MU", "target_shares"] == 0
    assert etf_core.loc["DRAM", "action"] == "REDUCE_TO_TARGET"


def test_user_supplied_portfolio_equity_changes_weight_denominator() -> None:
    summary, _, _ = app.analyze_memory_positions(
        sample_holdings(), sample_prices(), portfolio_value=10000,
    )
    assert summary["reference_portfolio_value"] == pytest.approx(10000)
    assert summary["reference_portfolio_value_type"] == "user_supplied_net_portfolio_equity"
    assert summary["memory_sleeve_weight_pct"] == pytest.approx(30)


def test_trend_diagnostics_reports_short_history_and_trend_values() -> None:
    prices = sample_prices()
    diagnostics = app.calculate_trend_diagnostics(prices)
    assert diagnostics.loc["MU", "return_20d_pct"] is not None
    assert diagnostics.loc["MU", "return_60d_pct"] is not None
    assert diagnostics.loc["MU", "realized_volatility_20d_annualized_pct"] >= 0
    short = app.calculate_trend_diagnostics(prices.iloc[:15])
    assert short.loc["DRAM", "trend_flag"] == "INSUFFICIENT_HISTORY"


def test_allocation_rejects_missing_memory_tickers_and_bad_caps() -> None:
    incomplete = sample_holdings()[sample_holdings()["ticker"] != "DRAM"]
    with pytest.raises(ValueError, match="must include MU, SKHY and DRAM"):
        app.analyze_memory_positions(incomplete, sample_prices())
    with pytest.raises(ValueError, match="max_memory_weight_pct"):
        app.analyze_memory_positions(sample_holdings(), sample_prices(), max_memory_weight_pct=101)


def test_rejects_invalid_lookthrough() -> None:
    with pytest.raises(ValueError, match="cannot sum above 100%"):
        app.analyze_memory_positions(
            sample_holdings(), sample_prices(), dram_lookthrough={"MU":0.7,"SKHY":0.4}
        )


def test_target_weight_parser_validates_symbols_and_values() -> None:
    assert app.parse_target_weights("MU=8,SKHY=6") == {"MU":8.0,"SKHY":6.0,"DRAM":0.0}
    with pytest.raises(Exception):
        app.parse_target_weights("MU=-1,SKHY=6")
    with pytest.raises(Exception):
        app.parse_target_weights("MU=8,MU=10")
    with pytest.raises(Exception):
        app.parse_target_weights("NVDA=8")


def test_cli_writes_memory_decision_outputs(tmp_path: Path) -> None:
    holdings_path = tmp_path / "holdings.csv"
    sample_holdings().to_csv(holdings_path, index=False)
    prices_path = tmp_path / "close_prices.csv"
    sample_prices().to_csv(prices_path)
    adjusted_path = tmp_path / "adjusted_prices.csv"
    sample_prices().to_csv(adjusted_path)
    output_dir = tmp_path / "output"
    result = subprocess.run(
        [
            sys.executable, str(Path(__file__).resolve().parents[1] / "memory_allocation.py"),
            "--holdings-csv", str(holdings_path),
            "--prices", str(prices_path),
            "--adjusted-prices", str(adjusted_path),
            "--output-dir", str(output_dir),
        ],
        check=True, capture_output=True, text=True,
    )
    summary = json.loads((output_dir / "memory_allocation_summary.json").read_text())
    assert summary["overall_action"] == "REDUCE_MEMORY_SLEEVE"
    for filename in (
        "memory_position_diagnostics.csv",
        "memory_allocation_scenarios.csv",
        "memory_allocation_summary.json",
    ):
        assert (output_dir / filename).exists()
        assert (output_dir / filename).stat().st_size > 0
    assert '"overall_action"' in result.stdout


def test_cap_only_scenario_preserves_dram_and_minimizes_direct_stock_trim() -> None:
    summary, positions, decisions = app.analyze_memory_positions(
        sample_holdings(), sample_prices(), max_memory_weight_pct=25,
    )
    cap = decisions[decisions["scenario"] == "cap_only_preserve_etf"].set_index("ticker")
    assert cap.loc["MU", "target_shares"] == 2
    assert cap.loc["SKHY", "target_shares"] == 3
    assert cap.loc["DRAM", "target_shares"] == 40
    assert cap.loc["MU", "action"] == "REDUCE_TO_TARGET"
    assert cap.loc["SKHY", "action"] == "REDUCE_TO_TARGET"
    assert cap.loc["DRAM", "action"] == "HOLD"
    assert cap["scenario_total_target_weight_pct"].iloc[0] <= 25
    assert positions["add_permission"].eq("NO_ADD_MEMORY_LIMIT").all()


def test_latest_dram_lookthrough_defaults_are_dated() -> None:
    assert app.DEFAULT_LOOKTHROUGH_ASOF == "2026-10-29"
    assert app.DEFAULT_DRAM_LOOKTHROUGH["MU"] == pytest.approx(0.2787)
    assert app.DEFAULT_DRAM_LOOKTHROUGH["SKHY"] == pytest.approx(0.2182)
