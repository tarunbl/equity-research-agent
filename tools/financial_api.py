"""
tools/financial_api.py
======================
Live financial + valuation + analyst data via yfinance (Yahoo Finance).
No API key required. Supports any valid stock ticker.

Design
------
Each data category (revenue growth, valuation multiples, analyst consensus,
earnings calendar) is isolated into its own helper function. This makes
exception handling flat and readable — a failure in one category doesn't
cascade into others.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    _YFINANCE_AVAILABLE = True
except ImportError:
    _YFINANCE_AVAILABLE = False


def get_financial_data(ticker: str) -> dict[str, Any] | None:
    """
    Fetch live data for any ticker from Yahoo Finance.

    Returns a flat dict of financial metrics, price data, analyst consensus,
    and earnings calendar, or None if the ticker is invalid / unreachable.
    """
    if not _YFINANCE_AVAILABLE:
        raise RuntimeError("yfinance not installed. Run: pip install yfinance")

    ticker = ticker.upper()

    try:
        t    = yf.Ticker(ticker)
        info = t.info or {}

        if not info.get("longName") and not info.get("shortName"):
            return None

        return {
            "ticker":       ticker,
            "company_name": info.get("longName") or info.get("shortName", ticker),
            "sector":       info.get("sector", "N/A"),

            # Core financials
            "revenue_growth_pct":   _revenue_growth(t),
            "gross_margin_pct":     _pct(info.get("grossMargins")),
            "profit_margin_pct":    _pct(info.get("profitMargins")),
            "pe_ratio":             _val(info.get("trailingPE")),
            "debt_to_equity":       _val(info.get("debtToEquity")),
            "current_ratio":        _val(info.get("currentRatio")),
            "return_on_equity_pct": _pct(info.get("returnOnEquity")),
            "data_quality_hint":    _data_quality(info),

            # Price & momentum
            "current_price":    _val(info.get("currentPrice") or info.get("regularMarketPrice")),
            "week_52_high":     _val(info.get("fiftyTwoWeekHigh")),
            "week_52_low":      _val(info.get("fiftyTwoWeekLow")),
            "pct_off_52w_high": _pct_off_high(info),

            # Valuation multiples
            "forward_pe":     _val(info.get("forwardPE")),
            "peg_ratio":      _val(info.get("pegRatio")),
            "price_to_sales": _val(info.get("priceToSalesTrailing12Months")),
            "price_to_book":  _val(info.get("priceToBook")),
            "ev_to_ebitda":   _val(info.get("enterpriseToEbitda")),

            # Analyst consensus
            "analyst_count":          int(info.get("numberOfAnalystOpinions") or 0),
            "analyst_recommendation": str(info.get("recommendationKey") or "none").lower(),
            "recommendation_mean":    _val(info.get("recommendationMean")),  # 1=Strong Buy, 5=Strong Sell
            "mean_price_target":      _val(info.get("targetMeanPrice")),
            "low_price_target":       _val(info.get("targetLowPrice")),
            "high_price_target":      _val(info.get("targetHighPrice")),
            "upside_to_target_pct":   _upside(info),

            # Earnings calendar
            **_earnings_calendar(t),

            "notes": f"Live — Yahoo Finance. Sector: {info.get('sector', 'N/A')}.",
        }

    except Exception as exc:
        logger.warning(f"[financial_api] Failed to fetch {ticker}: {exc}")
        return None


# ── Category helpers ──────────────────────────────────────────────────────────

def _revenue_growth(t: Any) -> float:
    """Compute YoY revenue growth from the last two annual income statements."""
    try:
        fin = t.financials
        if fin is None or fin.empty:
            return 0.0
        for label in ("Total Revenue", "Revenue"):
            if label in fin.index:
                rev = fin.loc[label]
                if len(rev) >= 2 and rev.iloc[1] and rev.iloc[1] != 0:
                    return round(
                        ((rev.iloc[0] - rev.iloc[1]) / abs(rev.iloc[1])) * 100, 2
                    )
    except Exception:
        pass
    return 0.0


def _data_quality(info: dict[str, Any]) -> float:
    """Fraction of key financial fields that are non-null/non-zero."""
    fields = [
        info.get("grossMargins"),
        info.get("profitMargins"),
        info.get("trailingPE"),
        info.get("debtToEquity"),
        info.get("currentRatio"),
        info.get("returnOnEquity"),
    ]
    filled = sum(1 for f in fields if f is not None and f != 0)
    return round(filled / len(fields), 2)


def _pct_off_high(info: dict[str, Any]) -> float:
    """Percentage below 52-week high (negative = below)."""
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    high  = info.get("fiftyTwoWeekHigh")
    if price and high and high > 0:
        return round(((price - high) / high) * 100, 1)
    return 0.0


def _upside(info: dict[str, Any]) -> float:
    """Implied upside to analyst mean price target."""
    price  = info.get("currentPrice") or info.get("regularMarketPrice")
    target = info.get("targetMeanPrice")
    if price and target and price > 0:
        return round(((target - price) / price) * 100, 1)
    return 0.0


def _earnings_calendar(t: Any) -> dict[str, Any]:
    """
    Extract next earnings date and proximity flag.
    Returns safe defaults on any failure — earnings data is best-effort.
    """
    earnings_date           = None
    earnings_within_14_days = False

    try:
        cal = t.calendar
        if cal is None:
            return {"earnings_date": None, "earnings_within_14_days": False}

        raw_date = None
        if isinstance(cal, dict):
            ed_raw = cal.get("Earnings Date")
            if ed_raw is not None:
                items = list(ed_raw) if hasattr(ed_raw, "__iter__") else [ed_raw]
                raw_date = items[0] if items else None
        elif hasattr(cal, "T"):  # DataFrame
            if "Earnings Date" in cal.index:
                raw_date = cal.loc["Earnings Date"].iloc[0]

        if raw_date is not None:
            earnings_date = str(raw_date)[:10]
            ed_dt = datetime.fromisoformat(earnings_date)
            # Make timezone-naive for comparison
            if ed_dt.tzinfo is not None:
                ed_dt = ed_dt.replace(tzinfo=None)
            days = (ed_dt - datetime.now()).days
            earnings_within_14_days = 0 <= days <= 14

    except Exception:
        pass  # earnings date is informational — never crash on it

    return {
        "earnings_date":           earnings_date,
        "earnings_within_14_days": earnings_within_14_days,
    }


# ── Value helpers ─────────────────────────────────────────────────────────────

def _pct(value: Any) -> float:
    """0–1 ratio → percentage, 2 dp. Returns 0.0 for None."""
    if value is None:
        return 0.0
    try:
        return round(float(value) * 100, 2)
    except (TypeError, ValueError):
        return 0.0


def _val(value: Any, dp: int = 2) -> float:
    """Safe round. Returns 0.0 for None or non-numeric."""
    if value is None:
        return 0.0
    try:
        return round(float(value), dp)
    except (TypeError, ValueError):
        return 0.0
