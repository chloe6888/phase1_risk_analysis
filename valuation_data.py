#!/usr/bin/env python3
"""Yahoo Finance/yfinance snapshot adapter with explicit field provenance and cache."""
from __future__ import annotations

import json
import math
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = ROOT / "data" / "fundamentals"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _date_from_epoch(value: Any) -> str | None:
    number = _number(value)
    if number is None or number <= 0:
        return None
    try:
        return datetime.fromtimestamp(number, tz=timezone.utc).date().isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def _normalize_label(value: Any) -> str:
    return "".join(character.lower() for character in str(value) if character.isalnum())


def _find_row(frame: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series | None:
    if frame is None or frame.empty:
        return None
    normalized = {_normalize_label(index): index for index in frame.index}
    for candidate in candidates:
        key = _normalize_label(candidate)
        if key in normalized:
            return frame.loc[normalized[key]]
    for candidate in candidates:
        key = _normalize_label(candidate)
        for normalized_key, original in normalized.items():
            if key and (key in normalized_key or normalized_key in key):
                return frame.loc[original]
    return None


def _latest_value(frame: pd.DataFrame, candidates: tuple[str, ...]) -> tuple[float | None, str | None]:
    row = _find_row(frame, candidates)
    if row is None:
        return None, None
    series = pd.to_numeric(row, errors="coerce").dropna()
    if series.empty:
        return None, None
    series = series.sort_index()
    date_label = pd.Timestamp(series.index[-1]).date().isoformat() if not isinstance(series.index[-1], (int, float)) else str(series.index[-1])
    return _number(series.iloc[-1]), date_label


def _ttm_fcf_from_statement(frame: pd.DataFrame, periods: int) -> tuple[float | None, str | None]:
    if frame is None or frame.empty:
        return None, None
    ordered = frame.copy()
    try:
        ordered = ordered.reindex(sorted(ordered.columns), axis=1)
    except TypeError:
        pass
    free_cash_flow, _ = _latest_value(ordered, ("Free Cash Flow", "FreeCashFlow"))
    if free_cash_flow is not None and periods == 1:
        return free_cash_flow, "statement_free_cash_flow"
    ocf_row = _find_row(ordered, (
        "Operating Cash Flow", "Cash Flow From Continuing Operating Activities",
        "Total Cash From Operating Activities", "Operating Cashflow",
    ))
    capex_row = _find_row(ordered, ("Capital Expenditure", "Capital Expenditures", "Capital Expenditure Reported"))
    if ocf_row is None or capex_row is None:
        return None, None
    ocf = pd.to_numeric(ocf_row, errors="coerce").dropna().sort_index().tail(periods)
    capex = pd.to_numeric(capex_row, errors="coerce").dropna().sort_index().tail(periods)
    if len(ocf) < periods or len(capex) < periods:
        return None, None
    # Yahoo cash-flow statements normally report capital expenditure as a negative cash flow.
    fcf = float(ocf.sum() + capex.sum()) if float(capex.sum()) < 0 else float(ocf.sum() - capex.sum())
    end_period = str(pd.Timestamp(ocf.index[-1]).date())
    return (fcf, f"operating_cash_flow_minus_capex_last_{periods}_periods:{end_period}")


def fetch_yahoo_snapshot(ticker: str, ticker_factory: Callable[[str], Any] | None = None) -> dict[str, Any]:
    """Fetch current price, forward EPS, FCF, cash, debt and shares using yfinance."""
    if ticker_factory is None:
        import yfinance as yf
        ticker_factory = yf.Ticker

    symbol = ticker.strip().upper()
    instrument = ticker_factory(symbol)
    errors: list[str] = []
    try:
        info = instrument.info or {}
        if not isinstance(info, Mapping):
            info = {}
    except Exception as exc:
        info = {}
        errors.append(f"info: {type(exc).__name__}: {exc}")

    history = None
    try:
        history = instrument.history(period="5d", auto_adjust=False)
    except Exception as exc:
        errors.append(f"history: {type(exc).__name__}: {exc}")
    price = None
    price_date = None
    if isinstance(history, pd.DataFrame) and not history.empty and "Close" in history.columns:
        close = pd.to_numeric(history["Close"], errors="coerce").dropna()
        if not close.empty:
            price = _number(close.iloc[-1])
            price_date = pd.Timestamp(close.index[-1]).date().isoformat()

    fast_info: Mapping[str, Any] = {}
    if price is None:
        try:
            fast = instrument.fast_info
            if isinstance(fast, Mapping):
                fast_info = fast
                price = _number(fast.get("last_price") or fast.get("lastPrice"))
        except Exception as exc:
            errors.append(f"fast_info: {type(exc).__name__}: {exc}")

    if price is None:
        price = _number(info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose"))
        if price is not None:
            price_date = _date_from_epoch(info.get("regularMarketTime"))
    currency = str(info.get("currency") or info.get("financialCurrency") or "USD").upper()
    financial_currency = str(info.get("financialCurrency") or currency).upper()
    currency_mismatch = currency != financial_currency
    shares = _number(info.get("sharesOutstanding") or info.get("impliedSharesOutstanding"))
    cash = _number(info.get("totalCash"))
    debt = _number(info.get("totalDebt"))
    market_cap = _number(info.get("marketCap"))
    forward_eps = _number(info.get("forwardEps"))
    eps_source = "Yahoo Finance info.forwardEps" if forward_eps is not None else None
    if forward_eps is None:
        forward_eps = _number(info.get("trailingEps"))
        if forward_eps is not None:
            eps_source = "Yahoo Finance info.trailingEps (forward EPS unavailable)"

    fcf = _number(info.get("freeCashflow"))
    fcf_source = "Yahoo Finance info.freeCashflow" if fcf is not None else None
    financial_period = _date_from_epoch(info.get("mostRecentQuarter")) or _date_from_epoch(info.get("lastFiscalYearEnd"))
    if fcf is None:
        try:
            ttm_cashflow = getattr(instrument, "ttm_cashflow", None)
            fcf, fcf_source = _ttm_fcf_from_statement(ttm_cashflow, 1)
            if fcf_source:
                fcf_source = f"yfinance ttm_cashflow {fcf_source}"
        except Exception as exc:
            errors.append(f"ttm_cashflow: {type(exc).__name__}: {exc}")
    if fcf is None:
        try:
            quarterly_cashflow = getattr(instrument, "quarterly_cashflow", None)
            fcf, fcf_source = _ttm_fcf_from_statement(quarterly_cashflow, 4)
            if fcf_source:
                fcf_source = f"yfinance quarterly_cashflow {fcf_source}"
        except Exception as exc:
            errors.append(f"quarterly_cashflow: {type(exc).__name__}: {exc}")

    if shares is None or cash is None or debt is None:
        try:
            balance_sheet = instrument.balance_sheet
            if shares is None:
                shares, _ = _latest_value(balance_sheet, ("Ordinary Shares Number", "Share Issued", "Common Stock Shares Outstanding"))
            if cash is None:
                cash, _ = _latest_value(balance_sheet, ("Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"))
            if debt is None:
                debt, _ = _latest_value(balance_sheet, ("Total Debt", "Long Term Debt And Capital Lease Obligation"))
        except Exception as exc:
            errors.append(f"balance_sheet: {type(exc).__name__}: {exc}")

    if financial_period is None:
        try:
            quarterly_cashflow = getattr(instrument, "quarterly_cashflow", None)
            if quarterly_cashflow is not None and len(quarterly_cashflow.columns):
                financial_period = str(pd.Timestamp(max(quarterly_cashflow.columns)).date())
        except Exception:
            pass

    fetched_at = utc_now()
    completeness = [
        ("current_price", price),
        ("forward_eps", forward_eps),
        ("free_cash_flow", fcf),
        ("diluted_shares", shares),
        ("cash", cash),
        ("debt", debt),
    ]
    missing = [name for name, value in completeness if value is None]
    data_quality = "complete_vendor_snapshot" if not missing else f"partial_vendor_snapshot; missing={','.join(missing)}"
    return {
        "ticker": symbol,
        "company_name": info.get("shortName") or info.get("longName") or symbol,
        "currency": currency,
        "financial_currency": financial_currency,
        "currency_mismatch": currency_mismatch,
        "current_price": price,
        "price_date": price_date,
        "fetched_at_utc": fetched_at,
        "financial_period": financial_period,
        "data_source": "Yahoo Finance via yfinance",
        "forward_eps": forward_eps,
        "eps_source": eps_source,
        "free_cash_flow": fcf,
        "fcf_source": fcf_source,
        "diluted_shares": shares,
        "cash": cash,
        "debt": debt,
        "market_cap": market_cap,
        "missing_fields": missing,
        "data_quality": data_quality,
        "fetch_warnings": errors,
        "stale_data": False,
    }


def load_cached_snapshot(ticker: str, cache_dir: str | Path = DEFAULT_CACHE_DIR) -> dict[str, Any] | None:
    path = Path(cache_dir) / f"{ticker.strip().upper()}.json"
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def save_cached_snapshot(snapshot: Mapping[str, Any], cache_dir: str | Path = DEFAULT_CACHE_DIR) -> Path:
    path = Path(cache_dir) / f"{str(snapshot['ticker']).upper()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(snapshot), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)
    return path


def cache_age_hours(snapshot: Mapping[str, Any]) -> float | None:
    fetched = snapshot.get("fetched_at_utc")
    if not fetched:
        return None
    try:
        value = datetime.fromisoformat(str(fetched).replace("Z", "+00:00"))
        return max(0.0, (datetime.now(timezone.utc) - value).total_seconds() / 3600.0)
    except (TypeError, ValueError):
        return None


def get_snapshot(
    ticker: str,
    *,
    refresh: bool = False,
    allow_stale_cache: bool = False,
    max_cache_age_hours: float = 168.0,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    fetcher: Callable[[str], dict[str, Any]] = fetch_yahoo_snapshot,
) -> dict[str, Any]:
    """Use a fresh cache; optionally fall back to stale cache when refresh fails."""
    if max_cache_age_hours <= 0 or not math.isfinite(max_cache_age_hours):
        raise ValueError("max_cache_age_hours must be finite and positive.")
    cached = load_cached_snapshot(ticker, cache_dir)
    age = cache_age_hours(cached) if cached else None
    if cached and not refresh and age is not None and age <= max_cache_age_hours:
        cached = dict(cached)
        cached["cache_age_hours"] = age
        price_date = cached.get("price_date")
        price_is_old = False
        if price_date:
            try:
                price_is_old = (date.today() - date.fromisoformat(str(price_date))).days > 7
            except ValueError:
                price_is_old = True
        cached["stale_data"] = bool(price_is_old)
        return cached
    try:
        snapshot = fetcher(ticker.strip().upper())
        save_cached_snapshot(snapshot, cache_dir)
        snapshot = dict(snapshot)
        snapshot["cache_age_hours"] = 0.0
        return snapshot
    except Exception:
        if allow_stale_cache and cached is not None:
            cached = dict(cached)
            cached["cache_age_hours"] = age
            cached["stale_data"] = True
            return cached
        raise
