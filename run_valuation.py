#!/usr/bin/env python3
"""Run transparent DCF/forward-PE scenario valuations for a watchlist."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from valuation_data import DEFAULT_CACHE_DIR, get_snapshot
from valuation_models import value_ticker

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config" / "valuation_assumptions.json"
DEFAULT_OUTPUT_DIR = Path(os.environ.get("PHASE1_OUTPUT_DIR", str(ROOT / "outputs"))).resolve()


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("tickers"), dict):
        raise ValueError("Valuation config must contain a ticker mapping.")
    return payload


def ticker_assumptions(config: dict[str, Any], symbol: str) -> dict[str, Any]:
    defaults = config.get("default_settings", {})
    per_ticker = config["tickers"].get(symbol)
    if not isinstance(defaults, dict) or not isinstance(per_ticker, dict):
        raise ValueError(f"No valuation assumptions configured for {symbol}.")
    result = dict(per_ticker)
    result.update({key: value for key, value in defaults.items() if key not in result})
    return result


def make_error_row(symbol: str, message: str) -> dict[str, Any]:
    return {
        "ticker": symbol,
        "company_name": symbol,
        "currency": None,
        "current_price": None,
        "price_date": None,
        "data_fetched_at_utc": None,
        "financial_period": None,
        "data_source": "Yahoo Finance via yfinance",
        "forward_eps": None,
        "eps_source": None,
        "free_cash_flow": None,
        "valuation_fcf": None,
        "fcf_basis": "unavailable",
        "fcf_source": None,
        "diluted_shares": None,
        "cash": None,
        "debt": None,
        "market_cap": None,
        "dcf_bear": None,
        "dcf_base": None,
        "dcf_bull": None,
        "pe_bear": None,
        "pe_base": None,
        "pe_bull": None,
        "dcf_buy_below": None,
        "dcf_sell_review_above": None,
        "pe_buy_below": None,
        "pe_sell_review_above": None,
        "dcf_view": "unavailable",
        "pe_view": "unavailable",
        "overall_confidence": "low",
        "data_quality": "fetch_failed",
        "limitations": message,
        "error": message,
    }


def fmt_money(value: Any, currency: str = "USD") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if pd.isna(number):
        return "N/A"
    return f"{currency} {number:,.2f}"


def fmt_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if pd.isna(number):
        return "N/A"
    return f"{number:,.2f}"


def render_report(rows: list[dict[str, Any]], generated_at: str, symbols: list[str]) -> str:
    lines = [
        "# 个股估值研究报告",
        "",
        f"- 生成时间（UTC）：{generated_at}",
        f"- 标的：{', '.join(symbols)}",
        "- 数据来源：Yahoo Finance，通过 yfinance 获取；请核对公司最新财报和官方披露。",
        "- 估值方法：五年股权现金流代理 DCF + 远期 PE；熊市/基准/牛市情景分别输出。",
        "",
        "## 情景估值结果",
        "",
        "| 股票 | 价格 | DCF熊 | DCF基准 | DCF牛 | PE熊 | PE基准 | PE牛 | DCF买入观察线 | PE买入观察线 | 数据质量 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        currency = str(row.get("currency") or "USD")
        lines.append(
            f"| {row['ticker']} | {fmt_money(row.get('current_price'), currency)} "
            f"| {fmt_money(row.get('dcf_bear'), currency)} | {fmt_money(row.get('dcf_base'), currency)} "
            f"| {fmt_money(row.get('dcf_bull'), currency)} | {fmt_money(row.get('pe_bear'), currency)} "
            f"| {fmt_money(row.get('pe_base'), currency)} | {fmt_money(row.get('pe_bull'), currency)} "
            f"| {fmt_money(row.get('dcf_buy_below'), currency)} | {fmt_money(row.get('pe_buy_below'), currency)} "
            f"| {row.get('overall_confidence', 'low')} |"
        )
    lines += [
        "",
        "## 数据与假设核验",
        "",
    ]
    for row in rows:
        currency = str(row.get("currency") or "USD")
        lines += [
            f"### {row['ticker']} — {row.get('company_name') or row['ticker']}",
            "",
            f"- 当前参考价格：{fmt_money(row.get('current_price'), currency)}（日期：{row.get('price_date') or '未知'}）",
            f"- 远期 EPS：{fmt_number(row.get('forward_eps'))}（来源：{row.get('eps_source') or '缺失'}）",
            f"- 数据源自由现金流：{fmt_money(row.get('free_cash_flow'), currency)}（来源：{row.get('fcf_source') or '缺失'}）",
            f"- DCF 实际使用的现金流：{fmt_money(row.get('valuation_fcf'), currency)}（口径：{row.get('fcf_basis') or '不可用'}）",
            f"- 股数（数据源）：{fmt_number(row.get('diluted_shares'))}",
            f"- 现金 / 债务：{fmt_money(row.get('cash'), currency)} / {fmt_money(row.get('debt'), currency)}",
            f"- 财务报表期间：{row.get('financial_period') or '未提供'}",
            f"- 数据获取时间：{row.get('data_fetched_at_utc') or '未提供'}",
            f"- DCF 区间判断：{row.get('dcf_view') or '不可用'}；PE 区间判断：{row.get('pe_view') or '不可用'}",
            f"- 置信度：{row.get('overall_confidence') or 'low'}",
            f"- 限制：{row.get('limitations') or '无'}",
            f"- 数据问题：{row.get('data_quality') or '未知'}",
            "",
        ]
    lines += [
        "## 如何解读买卖观察线",
        "",
        "- DCF 与 PE 是两种独立的估值交叉检查，不自动合成为一个目标价。",
        "- 买入观察线 = 基准情景估值 × (1 - 安全边际)；卖出复核线 = 基准情景估值 × (1 + 高估溢价)。它们是由配置假设计算出的复核区间，不是自动交易信号。",
        "- 若远期 EPS 缺失/非正数，PE 估值不计算；若自由现金流缺失/非正数，DCF 估值不计算。",
        "- Yahoo 的 CFO 减资本开支自由现金流只作为股权现金流代理，未逐项重建 FCFE；DCF 采用配置的股权资本成本折现，不再加现金或减债务。它不是完整的机构级 DCF。",
        "- MU、SKHY 等存储股具有周期性；如果没有跨周期正常化自由现金流，单期 FCF DCF 的参考价值很低。建议在 config/valuation_assumptions.json 填入经过人工研究的 normalized_fcf。",
        "- 默认增长率、折现率、终值增长率、PE 倍数和安全边际均为可编辑的人工起始假设，不代表分析师一致预期或公司指引。",
        "- 本报告不纳入技术趋势、仓位约束、现金及融资负债，不会连接券商或自动下单。",
        "",
    ]
    return "\n".join(lines)


def run(
    symbols: list[str],
    *,
    config_path: Path = DEFAULT_CONFIG,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    refresh: bool = False,
    allow_stale_cache: bool = False,
    max_cache_age_hours: float = 168.0,
) -> list[dict[str, Any]]:
    config = load_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    raw_snapshots: dict[str, Any] = {}
    for symbol in symbols:
        try:
            assumptions = ticker_assumptions(config, symbol)
            snapshot = get_snapshot(
                symbol,
                refresh=refresh,
                allow_stale_cache=allow_stale_cache,
                max_cache_age_hours=max_cache_age_hours,
                cache_dir=cache_dir,
            )
            raw_snapshots[symbol] = snapshot
            row = value_ticker(symbol, snapshot, assumptions)
            warnings = snapshot.get("fetch_warnings", [])
            if warnings:
                row["provider_warnings"] = " | ".join(str(item) for item in warnings)
            rows.append(row)
        except Exception as exc:
            rows.append(make_error_row(symbol, f"{type(exc).__name__}: {exc}"))

    generated_at = datetime.now(timezone.utc).isoformat()
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "valuation_summary.csv", index=False, float_format="%.8f")
    (output_dir / "valuation_fundamentals.json").write_text(
        json.dumps(raw_snapshots, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    assumptions_used = {
        symbol: ticker_assumptions(config, symbol)
        for symbol in symbols if symbol in config["tickers"]
    }
    (output_dir / "valuation_assumptions_used.json").write_text(
        json.dumps(assumptions_used, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report = render_report(rows, generated_at, symbols)
    (output_dir / "valuation_report.md").write_text(report, encoding="utf-8")
    manifest = {
        "generated_at_utc": generated_at,
        "tickers": symbols,
        "data_source": "Yahoo Finance via yfinance",
        "config_path": str(config_path),
        "cache_dir": str(cache_dir),
        "refresh": refresh,
        "allow_stale_cache": allow_stale_cache,
        "max_cache_age_hours": max_cache_age_hours,
        "successful_snapshots": len(raw_snapshots),
        "failed_snapshots": len(symbols) - len(raw_snapshots),
        "note": "Scenario valuation only; no trade execution or guaranteed price target.",
    }
    (output_dir / "valuation_run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(summary[[
        "ticker", "current_price", "dcf_base", "pe_base",
        "dcf_buy_below", "pe_buy_below", "overall_confidence",
    ]].to_string(index=False))
    print(f"\nReport saved: {output_dir / 'valuation_report.md'}")
    return rows


def main() -> None:
    config = load_config(DEFAULT_CONFIG)
    available = sorted(config["tickers"].keys())
    parser = argparse.ArgumentParser(description="Run DCF and forward-PE scenarios; no trade orders are placed.")
    parser.add_argument("--tickers", default=",".join(available), help="Comma-separated configured symbols.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--refresh", action="store_true", help="Force refresh Yahoo Finance snapshots.")
    parser.add_argument("--allow-stale-cache", action="store_true", help="Allow stale cached data only if refresh fails.")
    parser.add_argument("--max-cache-age-hours", type=float, default=168.0)
    args = parser.parse_args()
    if args.max_cache_age_hours <= 0:
        parser.error("--max-cache-age-hours must be positive.")
    symbols = [item.strip().upper() for item in args.tickers.split(",") if item.strip()]
    if not symbols or len(set(symbols)) != len(symbols):
        parser.error("Provide unique ticker symbols.")
    config = load_config(args.config)
    missing = sorted(set(symbols) - set(config["tickers"]))
    if missing:
        parser.error(f"No assumptions configured for: {', '.join(missing)}")
    run(
        symbols,
        config_path=args.config,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        refresh=args.refresh,
        allow_stale_cache=args.allow_stale_cache,
        max_cache_age_hours=args.max_cache_age_hours,
    )


if __name__ == "__main__":
    main()
