import pandas as pd
import pytest

import position_scenarios as app


def test_position_summary_marks_existing_holding_to_market() -> None:
    result = app.position_summary(shares=5, average_cost=100, current_price=95)
    assert result["total_cost"] == pytest.approx(500)
    assert result["market_value"] == pytest.approx(475)
    assert result["unrealized_pnl"] == pytest.approx(-25)
    assert result["unrealized_return_pct"] == pytest.approx(-0.05)


def test_add_scenario_calculates_new_average_cost_and_cash() -> None:
    result = app.build_add_scenarios(
        shares=5,
        average_cost=100,
        reference_price=95,
        entry_prices=[90],
        add_share_counts=[3],
        stress_prices=[80],
    ).iloc[0]
    assert result["cash_required"] == pytest.approx(270)
    assert result["total_shares_after_add"] == 8
    assert result["new_average_cost"] == pytest.approx(96.25)
    assert result["pnl_if_price_80"] == pytest.approx(8 * 80 - (5 * 100 + 3 * 90))


def test_add_scenario_rejects_nonpositive_prices() -> None:
    with pytest.raises(ValueError, match="positive"):
        app.build_add_scenarios(5, 100, 95, [0], [1], [])


def test_history_technical_snapshot(tmp_path) -> None:
    path = tmp_path / "history.csv"
    df = pd.DataFrame(
        {"Close": [float(100 + i) for i in range(60)]},
        index=pd.bdate_range("2024-01-01", periods=60),
    )
    df.index.name = "Date"
    df.to_csv(path)
    result = app.analyze_history(path)
    assert result["history_observations"] == 60
    assert result["sma20"] == pytest.approx(sum(range(140, 160)) / 20)
    assert result["sma50"] == pytest.approx(sum(range(110, 160)) / 50)


def test_staged_plan_is_cumulative_across_all_tranches() -> None:
    result = app.build_staged_plan(
        shares=10,
        average_cost=100,
        reference_price=95,
        tranche_prices=[95, 90, 85, 80, 75],
        tranche_shares=[1, 1, 1, 1, 1],
        stress_prices=[80, 75],
        max_add_cash=500,
        max_total_shares=15,
    )
    last = result.iloc[-1]
    assert result["tranche"].tolist() == [1, 2, 3, 4, 5]
    assert last["cumulative_added_shares"] == 5
    assert last["cumulative_additional_cash"] == pytest.approx(425)
    assert last["total_shares_after_tranche"] == 15
    assert last["total_cost_basis_after_tranche"] == pytest.approx(1425)
    assert last["average_cost_after_tranche"] == pytest.approx(95)
    assert last["pnl_if_price_75"] == pytest.approx(15 * 75 - 1425)


def test_staged_plan_rejects_cash_budget_overrun() -> None:
    with pytest.raises(ValueError, match="max_add_cash"):
        app.build_staged_plan(
            shares=10, average_cost=100, reference_price=95,
            tranche_prices=[95, 90], tranche_shares=[2, 2],
            max_add_cash=300,
        )


def test_staged_plan_rejects_total_share_limit_overrun() -> None:
    with pytest.raises(ValueError, match="max_total_shares"):
        app.build_staged_plan(
            shares=10, average_cost=100, reference_price=95,
            tranche_prices=[95, 90], tranche_shares=[2, 2],
            max_total_shares=12,
        )


def test_staged_plan_accepts_skipped_tranche() -> None:
    result = app.build_staged_plan(
        shares=10, average_cost=100, reference_price=95,
        tranche_prices=[95, 90, 85], tranche_shares=[1, 0, 1],
    )
    assert result["status"].tolist() == ["PLANNED", "SKIPPED", "PLANNED"]
    assert result.iloc[1]["total_shares_after_tranche"] == 11
    assert result.iloc[2]["total_shares_after_tranche"] == 12


def test_add_scenarios_flags_budget_violations_without_hiding_them() -> None:
    result = app.build_add_scenarios(
        shares=10, average_cost=100, reference_price=95,
        entry_prices=[75, 120], add_share_counts=[1, 5], stress_prices=[],
        max_add_cash=100, max_total_shares=12,
    )
    assert result.loc[(result.entry_price == 75) & (result.add_shares == 1), "within_budget"].iloc[0]
    assert not result.loc[(result.entry_price == 120) & (result.add_shares == 5), "within_budget"].iloc[0]


def test_position_summary_rejects_non_finite_inputs() -> None:
    with pytest.raises(ValueError, match="finite"):
        app.position_summary(shares=5, average_cost=float("nan"), current_price=95)


def test_staged_plan_rejects_stress_loss_limit_breach() -> None:
    with pytest.raises(ValueError, match="stress-loss limit"):
        app.build_staged_plan(
            shares=5, average_cost=100, reference_price=95,
            tranche_prices=[95], tranche_shares=[1], stress_prices=[60],
            max_stress_loss_pct=0.10,
        )


def test_independent_scenarios_flag_stress_loss_limit_breach() -> None:
    result = app.build_add_scenarios(
        shares=5, average_cost=100, reference_price=95,
        entry_prices=[95], add_share_counts=[1], stress_prices=[60],
        max_stress_loss_pct=0.10,
    )
    assert not result.loc[0, "within_budget"]
    assert "stress price" in result.loc[0, "budget_reason"]
