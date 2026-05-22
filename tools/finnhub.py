"""
tools/finnhub.py
================
Finnhub integration for insider transactions and multi-source news.
Free tier: 60 API calls/minute — well within pipeline needs.

Setup: add FINNHUB_API_KEY to .env (free at finnhub.io)

Two functions are exposed:

  get_insider_intelligence(ticker)
    → SEC Form 4 insider transactions for the last 90 days.
      Whether executives are buying or selling their own stock is one
      of the strongest forward-looking signals available for free.

  get_finnhub_news(ticker)
    → News headlines from multiple publishers (Reuters, Seeking Alpha,
      Benzinga, MarketBeat, etc.) — distinct from Yahoo Finance's feed.
      Used by news_aggregator to enrich the News Agent's headline set.
"""
from __future__ import annotations

import os
import logging
from datetime import datetime, timedelta
from typing import Any

import requests

logger = logging.getLogger(__name__)

_BASE_URL = "https://finnhub.io/api/v1"
_TIMEOUT  = 10   # seconds


def get_insider_intelligence(ticker: str) -> dict[str, Any]:
    """
    Fetch and summarise insider buy/sell transactions (last 90 days).

    Returns
    -------
    {
        "signal":       "bullish" | "bearish" | "neutral" | "unavailable",
        "summary":      str,   # human-readable summary for the LLM
        "net_shares":   int,   # positive = net buying, negative = net selling
        "notable":      list[str],  # top 3 transactions described in words
        "source":       str,
    }
    """
    api_key = os.getenv("FINNHUB_API_KEY", "")
    if not api_key:
        return _empty_insider("Finnhub API key not configured")

    try:
        resp = requests.get(
            f"{_BASE_URL}/stock/insider-transactions",
            params={"symbol": ticker, "token": api_key},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        return _summarise_insider_data(data)

    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 403:
            logger.info("[finnhub] Free tier does not include insider data for this ticker")
        else:
            logger.warning(f"[finnhub] Insider transactions HTTP error: {exc}")
    except Exception as exc:
        logger.warning(f"[finnhub] Insider transactions error: {exc}")

    return _empty_insider("Data unavailable")


def get_finnhub_news(ticker: str, days_back: int = 7) -> list[str]:
    """
    Fetch recent news headlines from Finnhub's multi-publisher feed.
    Returns plain headline strings (same format as yfinance news).
    Returns empty list if API key is not configured.
    """
    api_key = os.getenv("FINNHUB_API_KEY", "")
    if not api_key:
        return []

    to_date   = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    try:
        resp = requests.get(
            f"{_BASE_URL}/company-news",
            params={
                "symbol": ticker,
                "from":   from_date,
                "to":     to_date,
                "token":  api_key,
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        items = resp.json()

        headlines = []
        for item in items:
            headline = item.get("headline") or item.get("summary", "")
            if headline and headline.strip():
                headlines.append(headline.strip())

        return headlines[:15]  # Cap at 15 to avoid overwhelming the prompt

    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 403:
            logger.info("[finnhub] Free tier limit — no news returned")
        else:
            logger.warning(f"[finnhub] News HTTP error: {exc}")
    except Exception as exc:
        logger.warning(f"[finnhub] News fetch error: {exc}")

    return []


# ── Insider data summarisation ────────────────────────────────────────────────

def _summarise_insider_data(transactions: list[dict]) -> dict[str, Any]:
    """
    Summarise raw Form 4 transaction data into human-readable intelligence.

    Filters to last 90 days, aggregates buy vs sell volumes, identifies
    the most significant transactions.
    """
    cutoff     = datetime.now() - timedelta(days=90)
    buys:  list[dict] = []
    sells: list[dict] = []

    for tx in transactions:
        try:
            tx_date = datetime.strptime(tx.get("transactionDate", ""), "%Y-%m-%d")
            if tx_date < cutoff:
                continue
            code   = tx.get("transactionCode", "")
            shares = int(tx.get("share", 0) or 0)
            value  = float(tx.get("value", 0) or 0)
            name   = tx.get("name", "Unknown insider")

            entry = {
                "name":   name,
                "shares": shares,
                "value":  value,
                "date":   tx.get("transactionDate", ""),
            }

            if code == "P":   # Purchase
                buys.append(entry)
            elif code == "S":  # Sale
                sells.append(entry)
        except (ValueError, TypeError):
            continue

    if not buys and not sells:
        return _empty_insider("No insider transactions in last 90 days")

    total_bought = sum(t["shares"] for t in buys)
    total_sold   = sum(t["shares"] for t in sells)
    net_shares   = total_bought - total_sold
    buy_value    = sum(t["value"] for t in buys)
    sell_value   = sum(t["value"] for t in sells)

    # Determine signal
    if net_shares > 0 and buy_value > sell_value * 0.5:
        signal = "bullish"
    elif net_shares < 0 and sell_value > buy_value * 0.5:
        signal = "bearish"
    else:
        signal = "neutral"

    # Build human-readable summary
    parts = []
    if buys:
        parts.append(
            f"Insiders bought {total_bought:,} shares "
            f"(${buy_value/1_000_000:.1f}M) in last 90 days"
        )
    if sells:
        parts.append(
            f"insiders sold {total_sold:,} shares "
            f"(${sell_value/1_000_000:.1f}M)"
        )
    summary = "; ".join(parts) if parts else "Mixed insider activity"

    # Notable transactions (top 3 by value)
    all_tx = [
        {**t, "direction": "bought"} for t in buys
    ] + [
        {**t, "direction": "sold"} for t in sells
    ]
    all_tx.sort(key=lambda x: x["value"], reverse=True)
    notable = [
        f"{t['name']} {t['direction']} "
        f"{t['shares']:,} shares "
        f"(${t['value']/1_000_000:.1f}M) on {t['date']}"
        for t in all_tx[:3]
    ]

    return {
        "signal":     signal,
        "summary":    summary,
        "net_shares": net_shares,
        "notable":    notable,
        "source":     "Finnhub (SEC Form 4)",
    }


def _empty_insider(reason: str) -> dict[str, Any]:
    return {
        "signal":     "unavailable",
        "summary":    reason,
        "net_shares": 0,
        "notable":    [],
        "source":     "Finnhub",
    }
