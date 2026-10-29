"""Offline unit tests for the phase-one risk analysis calculations."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import phase1_risk_analysis as app


def test_cagr_volatility_and_drawdown_are_calculated() -> None:
    # Four observations with an exact 25% drawdown from 120 to 90.
    index = pd.to_datetime(["2020-01-01", "2020-04-01", "2020-07-01", "2021-01-01"])
    prices = pd.DataFrame(
        {
            "SPY": [100.0, 120.0, 90.0, 110.0],
            "ASSET": [50.0, 55.0, 60.0, 65.0],
        },
        index=index,
    )

    result = app.compute_risk_metrics(prices, benchmark="SPY")

    assert result.loc["SPY", "maximum_drawdown"] == pytest.approx(-0.25)
    years = (index[-1] - index[0]).days / 365.25
    expected_cagr = (110.0 / 100.0) ** (1.0 / years) - 1.0
    assert result.loc["SPY", "CAGR"] == pytest.approx(expected_cagr)
    assert result.loc["SPY", "annualized_volatility"] > 0
    assert result.loc["SPY", "beta_vs_SPY"] == pytest.approx(1.0)


def test_beta_recovers_known_linear_return_exposure() -> None:
    rng = np.random.default_rng(7)
    benchmark_returns = rng.normal(0.0004, 0.01, 300)
    asset_returns = 2.0 * benchmark_returns
    index = pd.bdate_range("2022-01-03", periods=301)

    prices = pd.DataFrame(
        {
            "SPY": 100.0 * np.r_[1.0, np.cumprod(1.0 + benchmark_returns)],
            "ASSET": 50.0 * np.r_[1.0, np.cumprod(1.0 + asset_returns)],
        },
        index=index,
    )

    result = app.compute_risk_metrics(prices, benchmark="SPY")
    assert result.loc["ASSET", "beta_vs_SPY"] == pytest.approx(2.0, abs=1e-10)


def test_missing_price_does_not_create_bridged_return() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    prices = pd.DataFrame(
        {
            "SPY": [100.0, 101.0, 102.0, 103.0],
            "ASSET": [100.0, np.nan, 105.0, 106.0],
        },
        index=index,
    )
    returns = prices.pct_change(fill_method=None)

    assert pd.isna(returns.loc[index[1], "ASSET"])
    assert pd.isna(returns.loc[index[2], "ASSET"])
    assert returns.loc[index[3], "ASSET"] == pytest.approx(106.0 / 105.0 - 1.0)


def test_compute_risk_metrics_requires_benchmark() -> None:
    prices = pd.DataFrame(
        {"ASSET": [100.0, 101.0, 102.0]},
        index=pd.date_range("2024-01-01", periods=3),
    )
    with pytest.raises(ValueError, match="Benchmark"):
        app.compute_risk_metrics(prices, benchmark="SPY")


def test_nonpositive_adjusted_price_is_rejected() -> None:
    prices = pd.DataFrame(
        {"SPY": [100.0, 101.0, 102.0], "ASSET": [10.0, 0.0, 11.0]},
        index=pd.date_range("2024-01-01", periods=3),
    )
    with pytest.raises(ValueError, match="Non-positive"):
        app.compute_risk_metrics(prices, benchmark="SPY")


def test_data_quality_report_flags_missing_core_fields() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    data = pd.DataFrame(
        {
            "Open": [100.0, np.nan],
            "Close": [100.5, 101.5],
            "Adj Close": [100.5, 101.5],
            "Volume": [1000.0, 1100.0],
            "Dividends": [0.0, 0.0],
            "Stock Splits": [0.0, 0.0],
        },
        index=index,
    )
    result = app.build_data_quality_report({"TEST": data})
    assert result.loc["TEST", "missing_open"] == 1
    assert result.loc["TEST", "missing_close"] == 0
    assert result.loc["TEST", "missing_volume"] == 0


def test_cache_reader_loads_saved_csv(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    monkeypatch.setattr(app, "RAW_DIR", raw_dir)
    data = pd.DataFrame(
        {
            "Open": [10.0, 11.0],
            "Close": [10.5, 11.5],
            "Adj Close": [10.5, 11.5],
            "Volume": [100, 200],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    data.index.name = "Date"
    data.to_csv(raw_dir / "TEST_daily.csv")

    loaded = app.load_cached_ticker("TEST")
    assert loaded is not None
    assert loaded.index.equals(data.index)
    assert loaded["Adj Close"].tolist() == [10.5, 11.5]


def test_save_plots_creates_expected_files(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    out_dir = tmp_path / "outputs"
    monkeypatch.setattr(app, "OUT_DIR", out_dir)
    index = pd.bdate_range("2024-01-02", periods=80)
    t = np.arange(len(index), dtype=float)
    prices = pd.DataFrame(
        {
            "SPY": 100 * np.exp(0.001 * t + 0.01 * np.sin(t / 5)),
            "SOXX": 100 * np.exp(0.0015 * t + 0.02 * np.sin(t / 6)),
            "NVDA": 100 * np.exp(0.002 * t + 0.03 * np.sin(t / 7)),
        },
        index=index,
    )

    app.save_plots(prices)

    for name in (
        "closing_prices.png",
        "net_value.png",
        "drawdown.png",
        "correlation_matrix.png",
    ):
        assert (out_dir / name).exists()
        assert (out_dir / name).stat().st_size > 0


def test_get_ticker_data_prefers_local_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    cached = pd.DataFrame(
        {"Adj Close": [100.0, 101.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    monkeypatch.setattr(app, "load_cached_ticker", lambda ticker: cached)
    monkeypatch.setattr(app, "cache_is_usable", lambda *args, **kwargs: (True, "fresh cache (1h old)"))
    monkeypatch.setattr(
        app,
        "download_one_ticker",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("download should not run")),
    )

    result = app.get_ticker_data("TEST", "2024-01-01", "2024-02-01")
    pd.testing.assert_frame_equal(result, cached)


def test_refresh_bypasses_local_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    refreshed = pd.DataFrame(
        {"Adj Close": [110.0]},
        index=pd.to_datetime(["2024-02-01"]),
    )
    monkeypatch.setattr(
        app,
        "load_cached_ticker",
        lambda ticker: (_ for _ in ()).throw(AssertionError("cache should be bypassed")),
    )
    monkeypatch.setattr(app, "download_one_ticker", lambda *args, **kwargs: refreshed)

    result = app.get_ticker_data("TEST", "2024-01-01", "2024-03-01", refresh=True)
    pd.testing.assert_frame_equal(result, refreshed)


def test_cache_is_rejected_when_metadata_is_missing(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(app, "RAW_DIR", tmp_path)
    data = pd.DataFrame(
        {"Adj Close": [100.0, 101.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    usable, reason = app.cache_is_usable(
        "TEST", data, "2024-01-01", "2024-02-01", now=pd.Timestamp("2024-02-02", tz="UTC").to_pydatetime()
    )
    assert usable is False
    assert "metadata missing" in reason


def test_cache_is_rejected_when_stale_or_range_is_incomplete(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import json
    from datetime import datetime, timezone

    monkeypatch.setattr(app, "RAW_DIR", tmp_path)
    meta_path = app.cache_metadata_path("TEST")
    data = pd.DataFrame(
        {"Adj Close": [100.0, 101.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    now = datetime(2024, 2, 2, tzinfo=timezone.utc)
    meta = {
        "ticker": "TEST",
        "downloaded_at_utc": "2024-01-30T00:00:00+00:00",
        "requested_start": "2024-01-01",
        "requested_end": "2024-02-01",
    }
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    usable, reason = app.cache_is_usable("TEST", data, "2024-01-01", "2024-02-01", max_age_hours=24, now=now)
    assert not usable
    assert "exceeds" in reason

    meta["downloaded_at_utc"] = "2024-02-01T23:00:00+00:00"
    meta["requested_end"] = "2024-01-15"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    usable, reason = app.cache_is_usable("TEST", data, "2024-01-01", "2024-02-01", max_age_hours=24, now=now)
    assert not usable
    assert "ended before" in reason


def test_fresh_cache_is_accepted(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import json
    from datetime import datetime, timezone

    monkeypatch.setattr(app, "RAW_DIR", tmp_path)
    now = datetime(2024, 2, 2, 12, tzinfo=timezone.utc)
    meta = {
        "ticker": "TEST",
        "downloaded_at_utc": "2024-02-02T00:00:00+00:00",
        "requested_start": "2024-01-01",
        "requested_end": "2024-02-02",
    }
    app.cache_metadata_path("TEST").write_text(json.dumps(meta), encoding="utf-8")
    data = pd.DataFrame(
        {"Adj Close": [100.0, 101.0]},
        index=pd.to_datetime(["2024-01-02", "2024-02-01"]),
    )
    usable, reason = app.cache_is_usable("TEST", data, "2024-01-01", "2024-02-02", max_age_hours=24, now=now)
    assert usable
    assert "fresh cache" in reason


def test_get_ticker_data_uses_only_usable_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    cached = pd.DataFrame(
        {"Adj Close": [100.0, 101.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    monkeypatch.setattr(app, "load_cached_ticker", lambda ticker: cached)
    monkeypatch.setattr(app, "cache_is_usable", lambda *args, **kwargs: (True, "fresh cache (1h old)"))
    monkeypatch.setattr(
        app,
        "download_one_ticker",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("download should not run")),
    )
    result = app.get_ticker_data("TEST", "2024-01-01", "2024-02-01")
    pd.testing.assert_frame_equal(result, cached)


def test_max_drawdown_details_returns_peak_trough_and_recovery() -> None:
    prices = pd.Series(
        [100.0, 120.0, 90.0, 110.0, 121.0],
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
    )
    result = app.max_drawdown_details(prices)
    assert result["maximum_drawdown"] == pytest.approx(-0.25)
    assert result["peak_date"] == "2024-01-02"
    assert result["trough_date"] == "2024-01-03"
    assert result["recovery_date"] == "2024-01-05"
    assert result["recovery_days_calendar"] == 3


def test_correlation_observation_counts_are_pairwise() -> None:
    returns = pd.DataFrame({"A": [0.1, 0.2, np.nan], "B": [0.1, np.nan, 0.3]})
    counts = app.build_correlation_observation_counts(returns)
    assert counts.loc["A", "A"] == 2
    assert counts.loc["B", "B"] == 2
    assert counts.loc["A", "B"] == 1
