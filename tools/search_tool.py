"""
tools/search_tool.py
====================
News aggregator — combines Yahoo Finance and Finnhub headlines.

Deduplicates across sources using normalised title comparison so the
News Agent receives a clean, diverse headline set without repeats.
"""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Max headlines returned from each source before dedup
_YAHOO_LIMIT   = 10
_FINNHUB_LIMIT = 15

try:
    import yfinance as yf
    _YFINANCE_AVAILABLE = True
except ImportError:
    _YFINANCE_AVAILABLE = False


def get_news(ticker: str) -> list[str] | None:
    """
    Aggregate recent news headlines for a ticker from all available sources.

    Sources (in priority order):
      1. Yahoo Finance (yfinance) — always fetched
      2. Finnhub — fetched only if FINNHUB_API_KEY is configured

    Returns deduplicated list of headline strings, or None if no data.
    """
    ticker = ticker.upper()
    all_headlines: list[dict[str, str]] = []

    # Source 1: Yahoo Finance
    yahoo = _fetch_yahoo_news(ticker)
    for h in yahoo[:_YAHOO_LIMIT]:
        all_headlines.append({"text": h, "source": "Yahoo Finance"})

    # Source 2: Finnhub (graceful — skipped if key absent)
    try:
        from tools.finnhub import get_finnhub_news
        finnhub = get_finnhub_news(ticker)
        for h in finnhub[:_FINNHUB_LIMIT]:
            all_headlines.append({"text": h, "source": "Finnhub"})
    except Exception as exc:
        logger.debug(f"[search_tool] Finnhub news skipped: {exc}")

    if not all_headlines:
        return None

    deduplicated = _deduplicate(all_headlines)
    sources_used = list({h["source"] for h in deduplicated})
    logger.info(
        f"[search_tool] {ticker}: {len(deduplicated)} headlines "
        f"from {sources_used}"
    )

    return [h["text"] for h in deduplicated]


# ── Source fetchers ───────────────────────────────────────────────────────────

def _fetch_yahoo_news(ticker: str) -> list[str]:
    """Fetch headlines from Yahoo Finance via yfinance."""
    if not _YFINANCE_AVAILABLE:
        return []
    try:
        t     = yf.Ticker(ticker)
        items = t.news or []
        headlines = []
        for item in items:
            title = (
                item.get("title")
                or item.get("headline")
                or item.get("content", {}).get("title")
                or ""
            )
            if title.strip():
                headlines.append(title.strip())
        return headlines
    except Exception as exc:
        logger.debug(f"[search_tool] Yahoo news error: {exc}")
    return []


# ── Deduplication ─────────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Normalise a headline for duplicate detection."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:80]  # Compare first 80 normalised chars


def _deduplicate(headlines: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    Remove near-duplicate headlines across sources.
    Keeps the first occurrence; Yahoo Finance takes priority
    (it is listed first in the input).
    """
    seen: set[str] = set()
    unique: list[dict[str, str]] = []

    for item in headlines:
        key = _normalise(item["text"])
        if key and key not in seen:
            seen.add(key)
            unique.append(item)

    return unique
