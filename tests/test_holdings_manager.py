"""Offline tests for private holdings import, valuation, and concentration guardrails."""
from __future__ import annotations

import pandas as pd
import pytest

import holdings_manager as app


def sample_holdings() -> pd.DataFrame:
    return pd.DataFrame([
        {"ticker":"AAA","name":"Alpha","shares":10,"average_cost":90,"snapshot_price":100,"risk_group":"semiconductor"},
        {"ticker":"BBB","name":"Beta","shares":5,"average_cost":110,"snapshot_price":100,"risk_group":"semiconductor"},
        {"ticker":"CCC","name":"Gamma","shares":2,"average_cost":50,"snapshot_price":60,"risk_group":"software"},
    ])


def test_load_holdings_normalizes_and_computes_snapshot_summary(tmp_path) -> None:
    path=tmp_path/"holdings.csv"
    sample_holdings().to_csv(path,index=False)
    frame=app.load_holdings(path)
    positions,summary=app.summarize_holdings(frame)
    assert len(positions)==3
    assert summary["position_count"]==3
    assert summary["market_value"]==pytest.approx(1620)
    assert summary["unrealized_pnl"]==pytest.approx(70)
    assert positions.iloc[0]["ticker"]=="AAA"


def test_load_holdings_rejects_missing_duplicate_and_invalid_rows(tmp_path) -> None:
    path=tmp_path/"bad.csv"
    pd.DataFrame({"ticker":["AAA"],"shares":[1]}).to_csv(path,index=False)
    with pytest.raises(ValueError,match="missing required columns"):
        app.load_holdings(path)
    data=sample_holdings()
    pd.concat([data,data.iloc[[0]]]).to_csv(path,index=False)
    with pytest.raises(ValueError,match="Duplicate tickers"):
        app.load_holdings(path)
    data=sample_holdings()
    data.loc[0,"shares"]=0
    data.to_csv(path,index=False)
    with pytest.raises(ValueError,match="greater than zero"):
        app.load_holdings(path)


def test_load_holdings_rejects_mixed_currencies(tmp_path) -> None:
    data=sample_holdings()
    data["currency"]=["USD","HKD","USD"]
    path=tmp_path/"mixed.csv"
    data.to_csv(path,index=False)
    with pytest.raises(ValueError,match="Mixed currencies"):
        app.load_holdings(path)


def test_reference_prices_override_screenshot_prices() -> None:
    frame=sample_holdings()
    positions,summary=app.summarize_holdings(frame,reference_prices={"AAA":120,"BBB":95})
    aaa=positions.set_index("ticker").loc["AAA"]
    assert aaa["reference_price"]==pytest.approx(120)
    assert aaa["market_value"]==pytest.approx(1200)
    assert summary["market_value"]==pytest.approx(1795)


def test_weights_from_latest_prices_tracks_dates_and_staleness() -> None:
    holdings=sample_holdings()
    dates=pd.to_datetime(["2024-01-01","2024-01-02","2024-01-03"])
    prices=pd.DataFrame({"AAA":[9,10,11],"BBB":[20,21,22],"CCC":[30,31,32]},index=dates)
    weights,valuation=app.weights_from_price_panel(holdings,prices)
    assert sum(weights.values())==pytest.approx(1.0)
    assert weights["AAA"]==pytest.approx(110/(110+110+64))
    assert set(valuation["price_date"])=={"2024-01-03"}
    stale=prices.copy()
    stale.loc[dates[-1],"BBB"]=float("nan")
    stale.loc[dates[-2],"BBB"]=float("nan")
    with pytest.raises(ValueError,match="older than the panel end"):
        app.weights_from_price_panel(holdings,stale,max_staleness_calendar_days=0)


def test_risk_guardrails_flag_single_and_group_concentration() -> None:
    frame=sample_holdings()
    positions,_=app.summarize_holdings(frame)
    alerts=app.risk_guardrail_alerts(positions,single_position_limit_pct=50,risk_group_limit_pct=50)
    assert any(x["kind"]=="risk_group_concentration" and x["name"]=="semiconductor" for x in alerts)
    single=app.risk_guardrail_alerts(positions,single_position_limit_pct=30,risk_group_limit_pct=100)
    assert any(x["kind"]=="single_position_concentration" for x in single)


def test_reconcile_snapshot_reports_deltas_instead_of_hiding_them() -> None:
    result=app.reconcile_snapshot(100.5,-4.2,reported_market_value=99,reported_unrealized_pnl=-5,tolerance=1)
    assert result["market_value_delta"]==pytest.approx(1.5)
    assert result["unrealized_pnl_delta"]==pytest.approx(0.8)
    assert result["within_tolerance"] is False


def test_subgroup_limit_is_separate_from_broad_risk_group_limit() -> None:
    frame = pd.DataFrame([
        {"ticker":"MU","shares":3,"average_cost":100,"snapshot_price":100,"risk_group":"semiconductor_related","subgroup":"memory"},
        {"ticker":"SKHY","shares":20,"average_cost":100,"snapshot_price":100,"risk_group":"semiconductor_related","subgroup":"memory"},
        {"ticker":"GOOG","shares":10,"average_cost":100,"snapshot_price":100,"risk_group":"internet_platform","subgroup":"search_cloud_ai"},
        {"ticker":"OKTA","shares":10,"average_cost":100,"snapshot_price":100,"risk_group":"enterprise_security","subgroup":"identity_security_software"},
    ])
    positions, _ = app.summarize_holdings(frame)
    alerts = app.risk_guardrail_alerts(positions, risk_group_limit_pct=45, subgroup_limit_pct=25)
    assert not any(
        a["kind"] == "risk_group_concentration" and a["name"] == "semiconductor_related"
        for a in alerts
    )
    memory_alert = next(
        a for a in alerts if a["kind"] == "subgroup_concentration" and a["name"] == "memory"
    )
    assert memory_alert["weight_pct"] > 25
    assert memory_alert["threshold_pct"] == 25


def test_guardrail_rejects_invalid_subgroup_limit() -> None:
    positions, _ = app.summarize_holdings(sample_holdings())
    with pytest.raises(ValueError, match="subgroup_limit_pct"):
        app.risk_guardrail_alerts(positions, subgroup_limit_pct=0)
