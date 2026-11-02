"""Offline tests for local-only real-data acceptance checks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import real_data_acceptance as acceptance


def _write_prices(path: Path, tickers: list[str], *, rows: int = 280, end: str = "2026-02-10") -> None:
    dates = pd.bdate_range(end=end, periods=rows)
    frame = pd.DataFrame(index=dates)
    for i, ticker in enumerate(tickers):
        frame[ticker] = 50.0 + i * 10 + np.arange(rows) * (0.03 + i * 0.005)
    frame.index.name = "Date"
    frame.to_csv(path)


def _write_holdings(path: Path, *, tickers: list[str] | None = None, currencies: list[str] | None = None) -> None:
    tickers = tickers or ["AAA", "BBB"]
    currencies = currencies or ["USD"] * len(tickers)
    frame = pd.DataFrame({
        "ticker": tickers,
        "shares": [2.0 + i for i in range(len(tickers))],
        "average_cost": [40.0 + i for i in range(len(tickers))],
        "snapshot_price": [50.0 + i for i in range(len(tickers))],
        "currency": currencies,
        "risk_group": ["technology"] * len(tickers),
        "subgroup": ["software"] * len(tickers),
    })
    frame.to_csv(path, index=False)


def test_acceptance_passes_with_complete_single_currency_data(tmp_path: Path) -> None:
    holdings = tmp_path / "holdings.csv"
    close = tmp_path / "close.csv"
    adjusted = tmp_path / "adjusted.csv"
    _write_holdings(holdings)
    _write_prices(close, ["AAA", "BBB"])
    _write_prices(adjusted, ["AAA", "BBB"])
    Path(str(holdings) + ".meta.json").write_text(
        json.dumps({"reported_market_value": 253.0, "reported_unrealized_pnl": 50.0}),
        encoding="utf-8",
    )

    result = acceptance.run_acceptance(
        holdings, close, adjusted, tmp_path / "reports", as_of_date="2026-02-10"
    )

    assert result["status"] == "WARNING"  # Concentration rules flag the intentionally two-name test portfolio.
    assert not any(item["level"] == "BLOCKER" for item in result["checks"])
    assert result["overview"]["position_count"] == 2
    assert result["overview"]["common_return_observations"] >= 252
    assert result["privacy"]["position_details_included"] is False
    report = (tmp_path / "reports" / "real_data_acceptance.md").read_text(encoding="utf-8")
    assert "不包含持股数量、成本价、市值或账户余额" in report
    assert "美元" not in report


def test_mixed_currencies_are_blocked_not_silently_combined(tmp_path: Path) -> None:
    holdings = tmp_path / "holdings.csv"
    close = tmp_path / "close.csv"
    adjusted = tmp_path / "adjusted.csv"
    _write_holdings(holdings, currencies=["USD", "HKD"])
    _write_prices(close, ["AAA", "BBB"])
    _write_prices(adjusted, ["AAA", "BBB"])

    result = acceptance.run_acceptance(
        holdings, close, adjusted, tmp_path / "reports", as_of_date="2026-02-10"
    )

    assert result["status"] == "BLOCKED"
    assert any(item["code"] == "MIXED_CURRENCIES" for item in result["checks"])


def test_missing_held_ticker_blocks_price_panel_acceptance(tmp_path: Path) -> None:
    holdings = tmp_path / "holdings.csv"
    close = tmp_path / "close.csv"
    adjusted = tmp_path / "adjusted.csv"
    _write_holdings(holdings)
    _write_prices(close, ["AAA"])
    _write_prices(adjusted, ["AAA", "BBB"])

    result = acceptance.run_acceptance(
        holdings, close, adjusted, tmp_path / "reports", as_of_date="2026-02-10"
    )

    assert result["status"] == "BLOCKED"
    assert any(item["code"] == "RAW_TICKER_COVERAGE" for item in result["checks"])


def test_stale_ticker_price_is_blocked(tmp_path: Path) -> None:
    holdings = tmp_path / "holdings.csv"
    close = tmp_path / "close.csv"
    adjusted = tmp_path / "adjusted.csv"
    _write_holdings(holdings)
    dates = pd.bdate_range(end="2026-02-10", periods=280)
    raw = pd.DataFrame({"AAA": np.arange(280) + 50.0, "BBB": np.arange(280) + 60.0}, index=dates)
    raw.loc[raw.index[-8:], "BBB"] = np.nan
    raw.index.name = "Date"
    raw.to_csv(close)
    _write_prices(adjusted, ["AAA", "BBB"])

    result = acceptance.run_acceptance(
        holdings, close, adjusted, tmp_path / "reports", as_of_date="2026-02-10"
    )

    assert result["status"] == "BLOCKED"
    assert any(item["code"] == "RAW_PRICE_STALE" for item in result["checks"])


def test_short_common_history_blocks_below_configured_minimum(tmp_path: Path) -> None:
    holdings = tmp_path / "holdings.csv"
    close = tmp_path / "close.csv"
    adjusted = tmp_path / "adjusted.csv"
    _write_holdings(holdings)
    _write_prices(close, ["AAA", "BBB"], rows=120)
    _write_prices(adjusted, ["AAA", "BBB"], rows=50)

    result = acceptance.run_acceptance(
        holdings, close, adjusted, tmp_path / "reports", as_of_date="2026-02-10",
        minimum_common_observations=60,
    )

    assert result["status"] == "BLOCKED"
    check = next(item for item in result["checks"] if item["code"] == "COMMON_RETURN_HISTORY")
    assert check["level"] == "BLOCKER"


def test_broker_snapshot_mismatch_is_blocked(tmp_path: Path) -> None:
    holdings = tmp_path / "holdings.csv"
    close = tmp_path / "close.csv"
    adjusted = tmp_path / "adjusted.csv"
    _write_holdings(holdings)
    _write_prices(close, ["AAA", "BBB"])
    _write_prices(adjusted, ["AAA", "BBB"])
    Path(str(holdings) + ".meta.json").write_text(
        json.dumps({"reported_market_value": 999999.0, "reconciliation_tolerance": 1.0}),
        encoding="utf-8",
    )

    result = acceptance.run_acceptance(
        holdings, close, adjusted, tmp_path / "reports", as_of_date="2026-02-10"
    )

    assert result["status"] == "BLOCKED"
    check = next(item for item in result["checks"] if item["code"] == "BROKER_RECONCILIATION")
    assert check["level"] == "BLOCKER"


def test_missing_holdings_writes_blocked_report(tmp_path: Path) -> None:
    result = acceptance.run_acceptance(
        tmp_path / "missing.csv",
        tmp_path / "close.csv",
        tmp_path / "adjusted.csv",
        tmp_path / "reports",
        as_of_date="2026-02-10",
    )
    assert result["status"] == "BLOCKED"
    assert (tmp_path / "reports" / "real_data_acceptance.json").exists()
    assert (tmp_path / "reports" / "real_data_acceptance.md").exists()


def test_non_us_ticker_requires_explicit_matching_currency(tmp_path: Path) -> None:
    holdings = tmp_path / "holdings.csv"
    close = tmp_path / "close.csv"
    adjusted = tmp_path / "adjusted.csv"
    pd.DataFrame({
        "ticker": ["00001.HK"],
        "shares": [10],
        "average_cost": [100],
        "snapshot_price": [110],
    }).to_csv(holdings, index=False)
    _write_prices(close, ["00001.HK"])
    _write_prices(adjusted, ["00001.HK"])

    result = acceptance.run_acceptance(
        holdings, close, adjusted, tmp_path / "reports", as_of_date="2026-02-10"
    )

    assert result["status"] == "BLOCKED"
    assert any(item["code"] == "EXCHANGE_CURRENCY_UNVERIFIED" for item in result["checks"])


def test_stale_adjusted_price_for_one_holding_is_blocked(tmp_path: Path) -> None:
    holdings = tmp_path / "holdings.csv"
    close = tmp_path / "close.csv"
    adjusted = tmp_path / "adjusted.csv"
    _write_holdings(holdings)
    _write_prices(close, ["AAA", "BBB"])
    dates = pd.bdate_range(end="2026-02-10", periods=280)
    frame = pd.DataFrame({
        "AAA": 50.0 + np.arange(280) * 0.03,
        "BBB": 60.0 + np.arange(280) * 0.04,
    }, index=dates)
    frame.loc[frame.index[-8:], "BBB"] = np.nan
    frame.index.name = "Date"
    frame.to_csv(adjusted)

    result = acceptance.run_acceptance(
        holdings, close, adjusted, tmp_path / "reports", as_of_date="2026-02-10"
    )

    assert result["status"] == "BLOCKED"
    assert any(item["code"] == "ADJUSTED_PRICE_TICKER_STALE" for item in result["checks"])
