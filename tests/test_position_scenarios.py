import pandas as pd
import pytest

import position_scenarios as app


def test_position_summary_marks_existing_holding_to_market() -> None:
    result = app.position_summary(shares=13, average_cost=306, current_price=289.43)
    assert result["total_cost"] == pytest.approx(3978)
    assert result["market_value"] == pytest.approx(3762.59)
    assert result["unrealized_pnl"] == pytest.approx(-215.41)
    assert result["unrealized_return_pct"] == pytest.approx(-215.41 / 3978)


def test_add_scenario_calculates_new_average_cost_and_cash() -> None:
    result = app.build_add_scenarios(
        shares=13,
        average_cost=306,
        reference_price=289.43,
        entry_prices=[280],
        add_share_counts=[3],
        stress_prices=[267.67],
    ).iloc[0]
    assert result["cash_required"] == pytest.approx(840)
    assert result["total_shares_after_add"] == 16
    assert result["new_average_cost"] == pytest.approx(301.125)
    assert result["pnl_if_price_267.67"] == pytest.approx(16 * 267.67 - (13 * 306 + 3 * 280))


def test_add_scenario_rejects_nonpositive_prices() -> None:
    with pytest.raises(ValueError, match="positive"):
        app.build_add_scenarios(13, 306, 289.43, [0], [1], [])


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
        shares=13,
        average_cost=306,
        reference_price=290,
        tranche_prices=[290, 280, 268, 255, 240],
        tranche_shares=[2, 2, 2, 3, 3],
        stress_prices=[255, 240],
        max_add_cash=3200,
        max_total_shares=25,
    )
    last = result.iloc[-1]
    assert result["tranche"].tolist() == [1, 2, 3, 4, 5]
    assert last["cumulative_added_shares"] == 12
    assert last["cumulative_additional_cash"] == pytest.approx(3161)
    assert last["total_shares_after_tranche"] == 25
    assert last["total_cost_basis_after_tranche"] == pytest.approx(7139)
    assert last["average_cost_after_tranche"] == pytest.approx(7139 / 25)
    assert last["pnl_if_price_240"] == pytest.approx(25 * 240 - 7139)


def test_staged_plan_rejects_cash_budget_overrun() -> None:
    with pytest.raises(ValueError, match="max_add_cash"):
        app.build_staged_plan(
            shares=13, average_cost=306, reference_price=290,
            tranche_prices=[290, 280], tranche_shares=[2, 2],
            max_add_cash=1000,
        )


def test_staged_plan_rejects_total_share_limit_overrun() -> None:
    with pytest.raises(ValueError, match="max_total_shares"):
        app.build_staged_plan(
            shares=13, average_cost=306, reference_price=290,
            tranche_prices=[290, 280], tranche_shares=[2, 2],
            max_total_shares=16,
        )


def test_staged_plan_accepts_skipped_tranche() -> None:
    result = app.build_staged_plan(
        shares=13, average_cost=306, reference_price=290,
        tranche_prices=[290, 280, 268], tranche_shares=[2, 0, 1],
    )
    assert result["status"].tolist() == ["PLANNED", "SKIPPED", "PLANNED"]
    assert result.iloc[1]["total_shares_after_tranche"] == 15
    assert result.iloc[2]["total_shares_after_tranche"] == 16


def test_add_scenarios_flags_budget_violations_without_hiding_them() -> None:
    result = app.build_add_scenarios(
        shares=13, average_cost=306, reference_price=290,
        entry_prices=[250, 300], add_share_counts=[1, 5], stress_prices=[],
        max_add_cash=1000, max_total_shares=16,
    )
    assert result.loc[(result.entry_price == 250) & (result.add_shares == 1), "within_budget"].iloc[0]
    assert not result.loc[(result.entry_price == 300) & (result.add_shares == 5), "within_budget"].iloc[0]


def test_position_summary_rejects_non_finite_inputs() -> None:
    with pytest.raises(ValueError, match="finite"):
        app.position_summary(shares=13, average_cost=float("nan"), current_price=290)
