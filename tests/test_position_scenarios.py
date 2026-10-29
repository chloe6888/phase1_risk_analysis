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
