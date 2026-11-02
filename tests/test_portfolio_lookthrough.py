"""Tests for direct-plus-ETF issuer look-through."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from portfolio_lookthrough import analyze_portfolio_lookthrough


def _snapshot(path: Path, fund: str, as_of: str, exposures: dict[str, float]) -> Path:
    payload = {
        "fund_ticker": fund,
        "as_of": as_of,
        "source_name": f"{fund} official holdings",
        "source_file": f"{fund}.csv",
        "issuer_weights_pct": exposures,
        "mapped_target_issuer_weight_pct": sum(exposures.values()),
        "unmapped_weight_pct": 100.0 - sum(exposures.values()),
        "cash_or_cash_equivalent_weight_pct": 0.0,
        "warnings": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    holdings = pd.DataFrame([
        {"ticker": "GOOG", "shares": 2},
        {"ticker": "DRAM", "shares": 1},
        {"ticker": "QQQ", "shares": 1},
    ])
    dates = pd.date_range("2026-10-28", periods=3, freq="B")
    prices = pd.DataFrame({
        "GOOG": [98.0, 99.0, 100.0],
        "DRAM": [49.0, 49.5, 50.0],
        "QQQ": [198.0, 199.0, 200.0],
    }, index=dates)
    return holdings, prices


def test_combines_direct_and_indirect_exposure_without_double_counting(tmp_path: Path) -> None:
    holdings, prices = _inputs()
    dram = _snapshot(tmp_path / "dram.json", "DRAM", "2026-10-30", {"MU": 30.0, "SKHY": 20.0})
    qqq = _snapshot(tmp_path / "qqq.json", "QQQ", "2026-10-30", {"GOOG": 10.0, "AVGO": 5.0})

    exposure, funds, summary = analyze_portfolio_lookthrough(
        holdings, prices, [dram, qqq], cash=50.0, as_of_date="2026-10-30"
    )
    indexed = exposure.set_index("issuer")
    assert indexed.loc["GOOG", "direct_weight_pct"] == pytest.approx(40.0)
    assert indexed.loc["GOOG", "QQQ_indirect_weight_pct"] == pytest.approx(4.0)
    assert indexed.loc["GOOG", "total_effective_exposure_pct"] == pytest.approx(44.0)
    assert indexed.loc["MU", "DRAM_indirect_weight_pct"] == pytest.approx(3.0)
    assert indexed.loc["SKHY", "DRAM_indirect_weight_pct"] == pytest.approx(2.0)
    assert indexed.loc["AVGO", "QQQ_indirect_weight_pct"] == pytest.approx(2.0)
    assert funds["status"].eq("loaded").all()
    assert summary["cash_source"] == "user_provided"
    assert summary["fund_snapshots_missing"] == []


def test_missing_or_stale_fund_snapshots_are_explicitly_reported(tmp_path: Path) -> None:
    holdings, prices = _inputs()
    exposure, funds, summary = analyze_portfolio_lookthrough(
        holdings, prices, [], cash=None, as_of_date="2026-10-30"
    )
    assert set(summary["fund_snapshots_missing"]) == {"DRAM", "QQQ"}
    assert set(funds["status"]) == {"missing_snapshot"}
    assert summary["cash_source"] == "not_provided_assumed_zero"
    assert exposure.set_index("issuer").loc["GOOG", "direct_weight_pct"] == pytest.approx(40.0)

    old = _snapshot(tmp_path / "dram_old.json", "DRAM", "2026-09-01", {"MU": 20.0})
    _, funds, summary = analyze_portfolio_lookthrough(
        holdings, prices, [old], cash=0, as_of_date="2026-10-30",
        max_snapshot_age_calendar_days=14,
    )
    status = funds.set_index("fund_ticker").loc["DRAM", "status"]
    assert status == "stale_snapshot"
    assert any("过期" in item for item in summary["warnings"])
    assert "QQQ" in summary["fund_snapshots_missing"]


def test_duplicate_snapshots_for_same_fund_are_rejected(tmp_path: Path) -> None:
    holdings, prices = _inputs()
    one = _snapshot(tmp_path / "dram_1.json", "DRAM", "2026-10-29", {"MU": 20.0})
    two = _snapshot(tmp_path / "dram_2.json", "DRAM", "2026-10-30", {"MU": 25.0})
    with pytest.raises(ValueError, match="More than one snapshot"):
        analyze_portfolio_lookthrough(
            holdings, prices, [one, two], as_of_date="2026-10-30"
        )
