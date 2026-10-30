"""Offline tests for date-stamped ETF issuer look-through imports."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import etf_lookthrough as app


def sample_csv(path: Path) -> Path:
    frame = pd.DataFrame([
        {"Ticker": "", "Name": "Micron Technology Inc.", "Identifier": "US5951121038", "ETF Weight": "3.00%", "Shares": "500000", "Market Value": "100000000"},
        {"Ticker": "", "Name": "MICRON TECHNOLOGY INC SWAP NM", "Identifier": "TRS MU", "ETF Weight": "16.72%", "Shares": "1000000", "Market Value": "500000000"},
        {"Ticker": "", "Name": "MICRON TECHNOLOGY INC.-SWAP-GOLD-L", "Identifier": "TRS MU GS", "ETF Weight": "9.94%", "Shares": "700000", "Market Value": "300000000"},
        {"Ticker": "", "Name": "SK hynix Inc", "Identifier": "KR7000660001", "ETF Weight": "15.38%", "Shares": "300000", "Market Value": "400000000"},
        {"Ticker": "", "Name": "SK HYNIX INC-SWAP-GOLD-L", "Identifier": "TRS 000660", "ETF Weight": "5.76%", "Shares": "100000", "Market Value": "150000000"},
        {"Ticker": "SKHY", "Name": "SK hynix Inc ADR", "Identifier": "US7845571052", "ETF Weight": "0.68%", "Shares": "850000", "Market Value": "164500000"},
        {"Ticker": "FGXXX", "Name": "First American Government Obligations", "Identifier": "Cash", "ETF Weight": "48.52%", "Shares": "1000000", "Market Value": "485200000"},
    ])
    frame.to_csv(path, index=False)
    return path


def test_issuer_import_aggregates_direct_swap_and_adr_rows(tmp_path: Path) -> None:
    path = sample_csv(tmp_path / "dram.csv")
    rows, summary, manifest = app.import_etf_holdings(path, as_of="2026-10-29")
    exposures = summary.set_index("issuer")
    assert manifest["weight_total_in_range"] is True
    assert manifest["total_weight_pct"] == pytest.approx(100.0)
    assert manifest["issuer_weights_pct"]["MU"] == pytest.approx(29.66)
    assert manifest["issuer_weights_pct"]["SKHY"] == pytest.approx(21.82)
    assert exposures.loc["SKHY", "direct_equity"] == pytest.approx(15.38)
    assert exposures.loc["SKHY", "swap"] == pytest.approx(5.76)
    assert exposures.loc["SKHY", "adr"] == pytest.approx(0.68)
    assert len(rows) == 7
    assert len(manifest["source_sha256"]) == 64


def test_decimal_weights_autodetect_and_override(tmp_path: Path) -> None:
    path = tmp_path / "decimal.csv"
    pd.DataFrame([
        {"Ticker": "MU", "Name": "Micron", "ETF Weight": 0.25},
        {"Ticker": "SKHY", "Name": "SK hynix ADR", "ETF Weight": 0.20},
        {"Ticker": "CASH", "Name": "Cash", "ETF Weight": 0.55},
    ]).to_csv(path, index=False)
    _, _, auto = app.import_etf_holdings(path, as_of="2026-10-29")
    assert auto["weights_were_decimal"] is True
    assert auto["issuer_weights_pct"]["MU"] == pytest.approx(25.0)
    _, _, forced = app.import_etf_holdings(path, as_of="2026-10-29", decimal_weights=False)
    assert forced["issuer_weights_pct"]["MU"] == pytest.approx(0.25)


def test_import_warns_when_export_is_incomplete_and_tracks_unmapped_weight(tmp_path: Path) -> None:
    path = tmp_path / "partial.csv"
    pd.DataFrame([
        {"Ticker": "MU", "Name": "Micron", "ETF Weight": "20%"},
        {"Ticker": "XYZ", "Name": "Other issuer", "ETF Weight": "25%"},
    ]).to_csv(path, index=False)
    _, _, manifest = app.import_etf_holdings(path, as_of="2026-10-29")
    assert manifest["weight_total_in_range"] is False
    assert manifest["unmapped_weight_pct"] == pytest.approx(25.0)
    assert len(manifest["warnings"]) == 2


def test_import_rejects_bad_weight_units_and_invalid_rows(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame([
        {"Ticker": "MU", "Name": "Micron", "ETF Weight": "125%"},
    ]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="exceeds 100"):
        app.import_etf_holdings(path, as_of="2026-10-29")


def test_import_requires_valid_date_and_schema(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame([{"Ticker": "MU", "Shares": 10}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="weight column"):
        app.import_etf_holdings(path, as_of="2026-10-29")
    path = tmp_path / "good.csv"
    pd.DataFrame([{"Ticker": "MU", "Name": "Micron", "Weight": "10%"}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="as_of"):
        app.import_etf_holdings(path, as_of="not-a-date")


def test_snapshot_roundtrip_and_validations(tmp_path: Path) -> None:
    path = sample_csv(tmp_path / "dram.csv")
    _, _, manifest = app.import_etf_holdings(path, as_of="2026-10-29")
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps(manifest), encoding="utf-8")
    weights, asof, loaded = app.load_issuer_snapshot(snapshot)
    assert asof == "2026-10-29"
    assert weights == pytest.approx({"MU": 0.2966, "SKHY": 0.2182})
    bad = tmp_path / "bad_snapshot.json"
    bad.write_text(json.dumps({"issuer_weights_pct": {"MU": 120}, "as_of": "2026-10-29"}), encoding="utf-8")
    with pytest.raises(ValueError, match="between 0 and 100"):
        app.load_issuer_snapshot(bad)


def test_cli_writes_immutable_dated_snapshot_artifacts(tmp_path: Path) -> None:
    source = sample_csv(tmp_path / "issuer.csv")
    output = tmp_path / "out"
    from subprocess import run
    import sys
    script = Path(__file__).resolve().parents[1] / "etf_lookthrough.py"
    result = run(
        [sys.executable, str(script), "--holdings-csv", str(source), "--as-of", "2026-10-29", "--output-dir", str(output)],
        check=True, capture_output=True, text=True,
    )
    assert (output / "DRAM_holdings_2026-10-29.csv").exists()
    assert (output / "DRAM_issuer_exposures_2026-10-29.csv").exists()
    snapshot = output / "DRAM_lookthrough_2026-10-29.json"
    assert snapshot.exists()
    payload = json.loads(snapshot.read_text())
    assert payload["issuer_weights_pct"]["SKHY"] == pytest.approx(21.82)
    assert '"as_of": "2026-10-29"' in result.stdout
