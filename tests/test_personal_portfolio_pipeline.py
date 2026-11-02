"""Offline tests for the personal portfolio data pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import run_personal_portfolio as app


def _config(path: Path) -> dict:
    config = {
        "schema_version": 1,
        "base_currency": "USD",
        "benchmark": "SPY",
        "reference_tickers": ["SPY", "SOXX"],
        "start_date": "2015-01-01",
        "minimum_common_observations": 60,
        "target_common_observations": 126,
        "symbols": {
            "AAA": {
                "name": "Test asset A", "role": "holding", "currency": "USD",
                "risk_group": "technology", "subgroup": "software",
            },
            "BBB": {
                "name": "Test asset B", "role": "holding", "currency": "USD",
                "risk_group": "technology", "subgroup": "hardware",
            },
            "SPY": {"name": "S&P 500 ETF", "role": "benchmark", "currency": "USD"},
            "SOXX": {"name": "Semiconductor ETF", "role": "sector_reference", "currency": "USD"},
        },
    }
    path.write_text(json.dumps(config), encoding="utf-8")
    return config


def _holdings(path: Path, *, currencies: list[str] | None = None, tickers: list[str] | None = None) -> None:
    tickers = tickers or ["AAA", "BBB"]
    currencies = currencies or ["USD"] * len(tickers)
    rows = []
    for ticker, currency in zip(tickers, currencies):
        rows.append({
            "ticker": ticker, "shares": 2, "average_cost": 40,
            "snapshot_price": 50, "currency": currency,
            "risk_group": "technology",
            "subgroup": "software" if ticker == "AAA" else "hardware",
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def test_default_universe_is_nine_holdings_plus_two_references() -> None:
    universe = app.load_universe()
    assert len(universe["holding_tickers"]) == 9
    assert universe["benchmark"] == "SPY"
    assert universe["reference_tickers"] == ["SPY", "SOXX"]
    assert universe["symbols"]["GOOG"]["role"] == "holding"
    assert universe["symbols"]["SKHY"]["regular_way_start_date"] == "2026-07-13"


def test_validate_holdings_requires_exact_ticker_coverage_and_tags(tmp_path: Path) -> None:
    config_path = tmp_path / "universe.json"
    _config(config_path)
    universe = app.load_universe(config_path)
    holdings_path = tmp_path / "holdings.csv"
    _holdings(holdings_path)
    loaded = app.validate_holdings_file(holdings_path, universe)
    assert set(loaded["ticker"]) == {"AAA", "BBB"}

    _holdings(holdings_path, tickers=["AAA"])
    with pytest.raises(ValueError, match="coverage mismatch"):
        app.validate_holdings_file(holdings_path, universe)


def test_blank_or_missing_currency_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "universe.json"
    _config(config_path)
    universe = app.load_universe(config_path)
    holdings_path = tmp_path / "holdings.csv"
    _holdings(holdings_path, currencies=["USD", ""])
    with pytest.raises(ValueError, match="explicit currency"):
        app.validate_holdings_file(holdings_path, universe)

    pd.DataFrame([{
        "ticker": "AAA", "shares": 2, "average_cost": 40, "snapshot_price": 50,
        "risk_group": "technology", "subgroup": "software",
    }]).to_csv(holdings_path, index=False)
    with pytest.raises(ValueError, match="currency column"):
        app.validate_holdings_file(holdings_path, universe)


def test_mixed_currency_and_taxonomy_mismatch_are_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "universe.json"
    _config(config_path)
    universe = app.load_universe(config_path)
    holdings_path = tmp_path / "holdings.csv"
    _holdings(holdings_path, currencies=["USD", "HKD"])
    with pytest.raises(ValueError, match="mixed currencies"):
        app.validate_holdings_file(holdings_path, universe)

    _holdings(holdings_path)
    data = pd.read_csv(holdings_path)
    data.loc[data["ticker"] == "AAA", "subgroup"] = "semiconductor"
    data.to_csv(holdings_path, index=False)
    with pytest.raises(ValueError, match="subgroup"):
        app.validate_holdings_file(holdings_path, universe)


def test_run_pipeline_refreshes_reference_and_held_tickers_then_accepts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "universe.json"
    _config(config_path)
    holdings_path = tmp_path / "holdings.csv"
    _holdings(holdings_path)
    calls: dict[str, object] = {}

    def fake_market_data(**kwargs) -> None:
        calls["market"] = kwargs

    def fake_acceptance(*args, **kwargs) -> dict:
        calls["acceptance"] = (args, kwargs)
        return {
            "status": "WARNING",
            "as_of_date": "2026-02-10",
            "overview": {"position_count": 2, "common_return_observations": 80},
        }

    monkeypatch.setattr(app.market_data, "main", fake_market_data)
    monkeypatch.setattr(app.acceptance, "run_acceptance", fake_acceptance)
    result = app.run_pipeline(
        config_path=config_path,
        holdings_path=holdings_path,
        close_prices_path=tmp_path / "close.csv",
        adjusted_prices_path=tmp_path / "adjusted.csv",
        output_dir=tmp_path / "reports",
        refresh=True,
        as_of_date="2026-02-10",
    )
    assert result["status"] == "WARNING"
    assert result["private_values_included"] is False
    market_kwargs = calls["market"]
    assert set(market_kwargs["tickers"]) == {"SPY", "SOXX", "AAA", "BBB"}
    assert market_kwargs["refresh"] is True
    assert market_kwargs["benchmark"] == "SPY"
    assert market_kwargs["short_history_tickers"] == []
    acceptance_args, acceptance_kwargs = calls["acceptance"]
    assert acceptance_args[0] == holdings_path
    assert acceptance_kwargs["minimum_common_observations"] == 60
    assert acceptance_kwargs["as_of_date"] == "2026-02-10"

@pytest.mark.parametrize(
    ("arguments", "expected_refresh"),
    [(["--refresh"], True), (["--use-cache"], False), ([], True)],
)
def test_cli_refresh_modes_are_supported(
    arguments: list[str],
    expected_refresh: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    calls: dict[str, object] = {}
    monkeypatch.setattr(sys, "argv", ["run_personal_portfolio.py", *arguments])

    def fake_pipeline(**kwargs: object) -> dict[str, str]:
        calls.update(kwargs)
        return {"status": "PASS"}

    monkeypatch.setattr(app, "run_pipeline", fake_pipeline)
    app.main()
    assert calls["refresh"] is expected_refresh
