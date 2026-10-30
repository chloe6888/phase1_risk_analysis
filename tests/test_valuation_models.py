"""Offline tests for valuation arithmetic, data handling and cache semantics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from valuation_data import fetch_yahoo_snapshot, get_snapshot
from valuation_models import dcf_valuation, pe_valuation, value_ticker, validate_assumptions


BASE_ASSUMPTIONS = {
    "margin_of_safety": 0.25,
    "overvaluation_premium": 0.20,
    "forecast_years": 5,
    "cyclical": False,
    "scenarios": {
        "bear": {"fcf_growth": 0.03, "cost_of_equity": 0.11, "terminal_growth": 0.02, "fair_pe": 16},
        "base": {"fcf_growth": 0.08, "cost_of_equity": 0.10, "terminal_growth": 0.025, "fair_pe": 20},
        "bull": {"fcf_growth": 0.12, "cost_of_equity": 0.09, "terminal_growth": 0.03, "fair_pe": 24},
    },
}


def test_dcf_calculates_positive_equity_value_per_share() -> None:
    result = dcf_valuation(
        free_cash_flow=100.0,
        diluted_shares=10.0,
        annual_growth=0.05,
        cost_of_equity=0.10,
        terminal_growth=0.02,
        forecast_years=5,
    )
    assert result["equity_value"] > 0
    assert result["intrinsic_value_per_share"] == pytest.approx(result["equity_value"] / 10.0)
    assert result["forecast_years"] == 5.0


def test_dcf_rejects_discount_rate_not_above_terminal_growth() -> None:
    with pytest.raises(ValueError, match="greater than terminal_growth"):
        dcf_valuation(
            free_cash_flow=100,
            diluted_shares=10,
            annual_growth=0.05,
            cost_of_equity=0.02,
            terminal_growth=0.02,
        )


def test_pe_calculates_eps_times_multiple() -> None:
    assert pe_valuation(forward_eps=5.0, fair_pe=20.0)["intrinsic_value_per_share"] == 100.0


def test_assumption_validation_requires_three_cases() -> None:
    bad = json.loads(json.dumps(BASE_ASSUMPTIONS))
    bad["scenarios"].pop("bull")
    with pytest.raises(ValueError, match="Missing bull"):
        validate_assumptions(bad)


def test_value_ticker_returns_separate_dcf_and_pe_thresholds() -> None:
    fundamentals = {
        "company_name": "Example Corp",
        "currency": "USD",
        "current_price": 100.0,
        "price_date": "2026-10-29",
        "fetched_at_utc": "2026-10-30T00:00:00+00:00",
        "forward_eps": 5.0,
        "eps_source": "test",
        "free_cash_flow": 1_000_000.0,
        "fcf_source": "test",
        "diluted_shares": 1_000_000.0,
        "cash": 100.0,
        "debt": 50.0,
        "data_quality": "test",
    }
    result = value_ticker("TEST", fundamentals, BASE_ASSUMPTIONS)
    assert result["dcf_base"] is not None
    assert result["pe_base"] == 100.0
    assert result["pe_buy_below"] == 75.0
    assert result["pe_sell_review_above"] == 120.0
    assert result["dcf_view"] in {"below_buy_threshold", "between_thresholds", "above_sell_review_threshold"}


def test_value_ticker_disables_dcf_when_fcf_is_not_positive() -> None:
    fundamentals = {
        "current_price": 50,
        "forward_eps": 2,
        "free_cash_flow": -1,
        "diluted_shares": 100,
        "cash": 0,
        "debt": 0,
    }
    result = value_ticker("LOSS", fundamentals, BASE_ASSUMPTIONS)
    assert result["dcf_base"] is None
    assert result["pe_base"] == 40.0
    assert "自由现金流" in result["limitations"]


def test_cyclical_ticker_is_always_low_confidence_without_normalized_cycle_review() -> None:
    assumptions = json.loads(json.dumps(BASE_ASSUMPTIONS))
    assumptions["cyclical"] = True
    fundamentals = {
        "current_price": 50, "forward_eps": 4, "free_cash_flow": 100,
        "diluted_shares": 10, "cash": 0, "debt": 0,
    }
    result = value_ticker("MEM", fundamentals, assumptions)
    assert result["overall_confidence"] == "low"
    assert "周期行业" in result["limitations"]


def test_yahoo_adapter_uses_info_fields_and_historical_close() -> None:
    class FakeTicker:
        info = {
            "shortName": "Test Corporation",
            "currency": "USD",
            "sharesOutstanding": 1000,
            "totalCash": 200,
            "totalDebt": 50,
            "marketCap": 100_000,
            "forwardEps": 3.5,
            "freeCashflow": 500,
            "mostRecentQuarter": 1793232000,
        }

        def history(self, period: str, auto_adjust: bool) -> pd.DataFrame:
            assert period == "5d"
            assert auto_adjust is False
            return pd.DataFrame(
                {"Close": [99.0, 100.0]},
                index=pd.to_datetime(["2026-10-28", "2026-10-29"]),
            )

    result = fetch_yahoo_snapshot("TEST", ticker_factory=lambda _symbol: FakeTicker())
    assert result["current_price"] == 100
    assert result["price_date"] == "2026-10-29"
    assert result["forward_eps"] == 3.5
    assert result["free_cash_flow"] == 500
    assert result["diluted_shares"] == 1000
    assert result["cash"] == 200
    assert result["debt"] == 50


def test_get_snapshot_reuses_fresh_cache_and_marks_stale_fallback(tmp_path: Path) -> None:
    cached = {
        "ticker": "TEST",
        "fetched_at_utc": "2099-01-01T00:00:00+00:00",
        "current_price": 10,
    }
    cache = tmp_path / "TEST.json"
    cache.write_text(json.dumps(cached), encoding="utf-8")
    result = get_snapshot(
        "TEST",
        cache_dir=tmp_path,
        fetcher=lambda _symbol: (_ for _ in ()).throw(AssertionError("should use fresh cache")),
    )
    assert result["stale_data"] is False
    assert result["current_price"] == 10


def test_get_snapshot_requires_explicit_stale_cache_fallback(tmp_path: Path) -> None:
    cached = {
        "ticker": "TEST",
        "fetched_at_utc": "2000-01-01T00:00:00+00:00",
        "current_price": 10,
    }
    (tmp_path / "TEST.json").write_text(json.dumps(cached), encoding="utf-8")
    with pytest.raises(RuntimeError):
        get_snapshot(
            "TEST",
            refresh=True,
            allow_stale_cache=False,
            cache_dir=tmp_path,
            fetcher=lambda _symbol: (_ for _ in ()).throw(RuntimeError("network failure")),
        )
    result = get_snapshot(
        "TEST",
        refresh=True,
        allow_stale_cache=True,
        cache_dir=tmp_path,
        fetcher=lambda _symbol: (_ for _ in ()).throw(RuntimeError("network failure")),
    )
    assert result["stale_data"] is True
    assert result["current_price"] == 10

def test_repository_starter_assumptions_validate_for_watchlist() -> None:
    from pathlib import Path

    from run_valuation import load_config, ticker_assumptions

    config_path = Path(__file__).resolve().parents[1] / "config" / "valuation_assumptions.json"
    config = load_config(config_path)
    assert {"GOOG", "NVDA", "MU", "SKHY"}.issubset(config["tickers"])
    for symbol in ("GOOG", "NVDA", "MU", "SKHY"):
        validate_assumptions(ticker_assumptions(config, symbol))


def test_run_valuation_writes_report_and_manifest_without_network(
    tmp_path: Path, monkeypatch
) -> None:
    import run_valuation

    config = {
        "default_settings": {
            "margin_of_safety": 0.25,
            "overvaluation_premium": 0.20,
            "forecast_years": 5,
        },
        "tickers": {"TEST": BASE_ASSUMPTIONS},
    }
    config_path = tmp_path / "assumptions.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(
        run_valuation,
        "get_snapshot",
        lambda _ticker, **_kwargs: {
            "ticker": "TEST",
            "company_name": "Example",
            "currency": "USD",
            "current_price": 100,
            "price_date": "2026-10-29",
            "fetched_at_utc": "2026-10-30T00:00:00+00:00",
            "financial_period": "2026-09-30",
            "forward_eps": 5,
            "eps_source": "test",
            "free_cash_flow": 1_000_000,
            "fcf_source": "test",
            "diluted_shares": 1_000_000,
            "cash": 100,
            "debt": 50,
            "market_cap": 100_000_000,
            "data_quality": "test",
            "stale_data": False,
            "fetch_warnings": [],
        },
    )

    rows = run_valuation.run(
        ["TEST"],
        config_path=config_path,
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "outputs",
    )

    assert rows[0]["ticker"] == "TEST"
    for name in (
        "valuation_summary.csv",
        "valuation_fundamentals.json",
        "valuation_assumptions_used.json",
        "valuation_report.md",
        "valuation_run_manifest.json",
    ):
        assert (tmp_path / "outputs" / name).is_file()
    assert "基准情景估值" in (tmp_path / "outputs" / "valuation_report.md").read_text(encoding="utf-8") or "买入观察线" in (
        tmp_path / "outputs" / "valuation_report.md"
    )


def test_currency_mismatch_blocks_dcf_and_pe_values() -> None:
    fundamentals = {
        "current_price": 100,
        "currency": "USD",
        "financial_currency": "KRW",
        "currency_mismatch": True,
        "forward_eps": 5,
        "free_cash_flow": 100,
        "diluted_shares": 10,
        "cash": 100,
        "debt": 0,
    }
    result = value_ticker("ADR", fundamentals, BASE_ASSUMPTIONS)
    assert result["dcf_base"] is None
    assert result["pe_base"] is None
    assert result["overall_confidence"] == "low"
    assert "未进行货币换算" in result["limitations"]
