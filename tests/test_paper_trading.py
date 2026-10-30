"""Offline tests for simulated paper rebalancing."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import paper_trading as app


def test_simulation_sells_before_buying_and_accounts_for_friction() -> None:
    holdings = pd.DataFrame({"ticker": ["AAA"], "shares": [10]})
    targets = {"AAA": 0.3, "BBB": 0.3}
    orders, after, summary = app.simulate_rebalance(
        holdings, targets, {"AAA": 100.0, "BBB": 50.0},
        cash=1000.0, transaction_cost_bps=10, slippage_bps=20,
    )
    assert set(orders["side"]) == {"SELL", "BUY"}
    assert orders.loc[orders["side"] == "SELL", "filled_shares"].sum() == pytest.approx(4)
    assert orders.loc[orders["side"] == "BUY", "filled_shares"].sum() == pytest.approx(12)
    assert after.set_index("ticker").loc["AAA", "shares"] == pytest.approx(6)
    assert after.set_index("ticker").loc["BBB", "shares"] == pytest.approx(12)
    assert summary["total_simulated_friction"] > 0
    assert summary["cash_after"] >= 0
    assert summary["nav_after"] < summary["nav_before"]


def test_buy_orders_scale_down_when_cash_is_insufficient() -> None:
    holdings = pd.DataFrame({"ticker": ["AAA"], "shares": [10]})
    orders, after, summary = app.simulate_rebalance(
        holdings, {"BBB": 0.95}, {"AAA": 100.0, "BBB": 100.0},
        cash=0.0, transaction_cost_bps=0, slippage_bps=0,
    )
    buy = orders.loc[orders["side"] == "BUY"].iloc[0]
    assert summary["buy_scale_due_to_cash"] < 1
    assert buy["status"] == "PARTIALLY_FILLED"
    assert after.set_index("ticker").loc["BBB", "shares"] < buy["desired_target_shares"]
    assert summary["cash_after"] >= 0


def test_fractional_share_mode_and_no_trade_case() -> None:
    holdings = pd.DataFrame({"ticker": ["AAA"], "shares": [1.5]})
    orders, after, summary = app.simulate_rebalance(
        holdings, {"AAA": 0.5}, {"AAA": 100.0},
        cash=200.0, fractional_shares=True,
    )
    assert orders.empty
    assert after.iloc[0]["shares"] == pytest.approx(1.5)
    assert summary["orders_count"] == 0


def test_rebalance_rejects_bad_weights_prices_and_cash() -> None:
    holdings = pd.DataFrame({"ticker": ["AAA"], "shares": [1]})
    with pytest.raises(ValueError, match="no more than 100"):
        app.simulate_rebalance(
            holdings, {"AAA": 0.8, "BBB": 0.4}, {"AAA": 100, "BBB": 100}, cash=0
        )
    with pytest.raises(ValueError, match="Missing current prices"):
        app.simulate_rebalance(holdings, {"BBB": 0.5}, {"AAA": 100}, cash=0)
    with pytest.raises(ValueError, match="cash"):
        app.simulate_rebalance(holdings, {"AAA": 0.5}, {"AAA": 100}, cash=-1)


def test_paper_ledger_is_previewed_unless_explicitly_written(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    initial = pd.DataFrame({"ticker": ["AAA"], "shares": [2]})
    holdings = pd.DataFrame({"ticker": ["AAA"], "shares": [1]})
    app.write_account_ledger(ledger, holdings, 50.0, last_simulation_summary={"nav_after": 150.0})
    state = json.loads(ledger.read_text())
    assert state["cash"] == 50.0
    assert state["positions"] == [{"ticker": "AAA", "shares": 1}]
    account, cash, exists = app.read_account(
        ledger, initial_holdings_csv=tmp_path / "unused.csv", initial_cash=999,
    )
    assert exists is True
    assert cash == 50.0
    assert app.normalize_holdings(account) == {"AAA": 1.0}
    assert app.normalize_holdings(initial) == {"AAA": 2.0}


def test_cli_writes_preview_and_commits_only_when_requested(tmp_path: Path) -> None:
    index = pd.bdate_range("2025-01-01", periods=20)
    prices = pd.DataFrame({"AAA": [100 + i for i in range(20)], "BBB": [50 + i for i in range(20)]}, index=index)
    price_file = tmp_path / "prices.csv"
    prices.to_csv(price_file)
    targets = tmp_path / "targets.csv"
    pd.DataFrame({"ticker": ["AAA", "BBB"], "weight": [0.3, 0.3]}).to_csv(targets, index=False)
    holdings = tmp_path / "holdings.csv"
    pd.DataFrame({"ticker": ["AAA"], "shares": [10]}).to_csv(holdings, index=False)
    ledger = tmp_path / "ledger.json"
    output = tmp_path / "output"
    script = Path(__file__).resolve().parents[1] / "paper_trading.py"
    command = [
        sys.executable, str(script), "--prices", str(price_file),
        "--targets-csv", str(targets), "--holdings-csv", str(holdings),
        "--cash", "1000", "--ledger-file", str(ledger), "--output-dir", str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert not ledger.exists()
    summary = json.loads((output / "paper_trading_summary.json").read_text())
    assert summary["ledger_committed"] is False
    subprocess.run(command + ["--commit-ledger"], check=True, capture_output=True, text=True)
    assert ledger.exists()
    state = json.loads(ledger.read_text())
    assert state["cash"] >= 0
