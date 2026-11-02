"""Tests for personal allocation scenarios and leakage-safe walk-forward checks."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_allocation_comparison import compare_allocations, walk_forward_minimum_variance


def _inputs(n: int = 180):
    rng = np.random.default_rng(20261102)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    common = rng.normal(0.0003, 0.008, n)
    series = {
        "AAA": common + rng.normal(0.0002, 0.010, n),
        "BBB": common * 0.8 + rng.normal(0.0001, 0.007, n),
        "CCC": rng.normal(0.0004, 0.009, n),
        "DDD": rng.normal(0.0002, 0.006, n),
    }
    returns = pd.DataFrame(series, index=dates)
    prices = 100.0 * (1.0 + returns).cumprod()
    holdings = pd.DataFrame([
        {"ticker":"AAA","shares":5,"average_cost":100,"snapshot_price":float(prices["AAA"].iloc[-1]),"currency":"USD","risk_group":"semiconductor_related","subgroup":"memory"},
        {"ticker":"BBB","shares":8,"average_cost":100,"snapshot_price":float(prices["BBB"].iloc[-1]),"currency":"USD","risk_group":"semiconductor_related","subgroup":"semiconductor_equipment"},
        {"ticker":"CCC","shares":4,"average_cost":100,"snapshot_price":float(prices["CCC"].iloc[-1]),"currency":"USD","risk_group":"internet_platform","subgroup":"search_cloud_ai"},
        {"ticker":"DDD","shares":6,"average_cost":100,"snapshot_price":float(prices["DDD"].iloc[-1]),"currency":"USD","risk_group":"internet_platform","subgroup":"identity_security_software"},
    ])
    return holdings, prices, prices.copy()


def _profiles():
    return {
        "minimum_common_observations":60,
        "walk_forward_train_observations":60,
        "walk_forward_test_observations":21,
        "profiles":{
            "balanced":{
                "risky_budget":0.85,
                "default_asset_max_weight":0.30,
                "group_max_weights":{"semiconductor_related":0.55,"internet_platform":0.55},
                "subgroup_max_weights":{"memory":0.30,"semiconductor_equipment":0.30,"search_cloud_ai":0.40,"identity_security_software":0.30},
            },
            "conservative":{
                "risky_budget":0.70,
                "default_asset_max_weight":0.25,
                "group_max_weights":{"semiconductor_related":0.40,"internet_platform":0.40},
                "subgroup_max_weights":{"memory":0.20,"semiconductor_equipment":0.20,"search_cloud_ai":0.30,"identity_security_software":0.20},
            },
        },
        "notes":["Synthetic test profiles."],
    }


def test_compare_allocations_respects_hierarchical_caps_and_compares_four_scenarios() -> None:
    holdings, close, adjusted = _inputs()
    result = compare_allocations(
        holdings, close, adjusted, _profiles(),
        as_of_date=pd.Timestamp.today().date(),
    )
    metrics = result["metrics"].set_index("scenario")
    assert set(metrics.index) == {
        "actual_holdings",
        "balanced_capped_equal_weight",
        "balanced_minimum_variance",
        "conservative_capped_equal_weight",
    }
    assert metrics.loc["actual_holdings", "cash_weight"] == pytest.approx(0.0)
    assert metrics.loc["balanced_capped_equal_weight", "cash_weight"] == pytest.approx(0.15)
    assert metrics.loc["conservative_capped_equal_weight", "cash_weight"] == pytest.approx(0.30)

    weights = result["weights"]
    for scenario in ("balanced_capped_equal_weight","balanced_minimum_variance"):
        frame = weights[weights["scenario"] == scenario]
        assert frame["target_weight"].sum() == pytest.approx(1.0)
        assert frame.loc[frame["ticker"] != "CASH", "target_weight"].max() <= 0.30 + 1e-7
    exposure = result["exposures"]
    for scenario in ("balanced_capped_equal_weight","balanced_minimum_variance"):
        group = exposure[(exposure["scenario"] == scenario) & (exposure["exposure_type"] == "group")].set_index("name")
        subgroup = exposure[(exposure["scenario"] == scenario) & (exposure["exposure_type"] == "subgroup")].set_index("name")
        assert group.loc["semiconductor_related","weight"] <= 0.55 + 1e-7
        assert subgroup.loc["memory","weight"] <= 0.30 + 1e-7
    assert result["summary"]["cash_source"] == "not_provided_assumed_zero"


def test_walk_forward_uses_prior_only_training_and_marks_short_history() -> None:
    holdings, _, adjusted = _inputs()
    profile = {
        **_profiles()["profiles"]["balanced"],
        "groups": {row.ticker: row.risk_group for row in holdings.itertuples(index=False)},
        "subgroups": {row.ticker: row.subgroup for row in holdings.itertuples(index=False)},
    }
    folds, summary = walk_forward_minimum_variance(
        adjusted,
        profile=profile,
        train_observations=60,
        test_observations=21,
        minimum_training_observations=60,
    )
    assert summary["no_lookahead"] is True
    assert summary["fold_count"] >= 3
    assert summary["status"] == "multiple_oos_folds"
    assert (pd.to_datetime(folds["train_end"]) < pd.to_datetime(folds["test_start"])).all()
    assert set(folds["strategy"]) == {"minimum_variance","balanced_capped_equal_weight"}

    short_folds, short_summary = walk_forward_minimum_variance(
        adjusted.iloc[:60],
        profile=profile,
        train_observations=60,
        test_observations=21,
        minimum_training_observations=60,
    )
    assert short_folds.empty
    assert short_summary["status"] == "insufficient_history"


def test_current_allocation_includes_cash_only_when_user_supplies_it() -> None:
    holdings, close, adjusted = _inputs()
    result = compare_allocations(
        holdings, close, adjusted, _profiles(),
        cash=1000.0,
        as_of_date=pd.Timestamp.today().date(),
    )
    assert result["summary"]["cash_source"] == "user_provided"
    actual = result["weights"].query("scenario == 'actual_holdings'").set_index("ticker")
    assert actual.loc["CASH","target_weight"] > 0
    assert actual["target_weight"].sum() == pytest.approx(1.0)
