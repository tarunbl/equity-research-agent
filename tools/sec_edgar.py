"""
tools/sec_edgar.py
==================
SEC EDGAR integration — no API key required.

Fetches two types of data from the official SEC EDGAR API:

  1. Risk Factors   — Item 1A from the latest 10-K filing.
                      These are the company's own legal disclosures of material
                      risks. Written by lawyers who face liability if they omit
                      real risks. More reliable than any analyst opinion.

  2. Material Events — Recent 8-K filings (last 30 days).
                      8-Ks disclose significant events: earnings results,
                      leadership changes, acquisitions, legal proceedings.
                      A CEO departure 2 weeks ago belongs in a risk assessment.

Rate limiting
-------------
SEC requires: max 10 requests/second, User-Agent header with name + email.
Configure SEC_USER_AGENT in your .env: "Your Name your@email.com"
"""
from __future__ import annotations

import os
import re
import logging
from datetime import datetime, timedelta
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ── EDGAR API endpoints ───────────────────────────────────────────────────────
_TICKERS_URL     = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_ARCHIVES_URL    = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}/{doc}"
_DEFAULT_UA      = "EquityResearchAgent research@example.com"

# Extraction limits
_MAX_RISK_FACTORS = 5
_MAX_8K_EVENTS    = 8
_MAX_DOC_BYTES    = 600_000   # 600KB — enough to reach Item 1A in most 10-Ks

# Module-level CIK cache: avoids re-fetching the 2MB tickers file per run
_cik_cache: dict[str, int | None] = {}


def get_sec_intelligence(ticker: str) -> dict[str, Any]:
    """
    Fetch SEC EDGAR intelligence for a ticker.

    Returns
    -------
    {
        "risk_factors":           list[str],   # from latest 10-K Item 1A
        "recent_material_events": list[dict],  # 8-K filings, last 30 days
        "latest_10k_date":        str | None,
        "source":                 str,
    }

    All fields default to empty/None on any failure — SEC data is
    best-effort enrichment, never a hard dependency.
    """
    headers = {
        "User-Agent": os.getenv("SEC_USER_AGENT", _DEFAULT_UA),
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json",
    }

    try:
        cik = _resolve_cik(ticker, headers)
        if not cik:
            logger.info(f"[sec_edgar] No CIK found for {ticker}")
            return _empty()

        submissions = _fetch_submissions(cik, headers)
        if not submissions:
            return _empty()

        return {
            "risk_factors":           _extract_risk_factors(cik, submissions, headers),
            "recent_material_events": _extract_8k_events(submissions),
            "latest_10k_date":        _latest_10k_date(submissions),
            "source":                 "SEC EDGAR",
        }

    except Exception as exc:
        logger.warning(f"[sec_edgar] Failed for {ticker}: {exc}")
        return _empty()


# ── CIK resolution ────────────────────────────────────────────────────────────

def _resolve_cik(ticker: str, headers: dict) -> int | None:
    """Look up SEC CIK for a ticker. Result is cached for the process lifetime."""
    ticker_upper = ticker.upper()
    if ticker_upper in _cik_cache:
        return _cik_cache[ticker_upper]

    try:
        resp = requests.get(_TICKERS_URL, headers=headers, timeout=10)
        resp.raise_for_status()
        for entry in resp.json().values():
            if entry.get("ticker", "").upper() == ticker_upper:
                cik = int(entry["cik_str"])
                _cik_cache[ticker_upper] = cik
                return cik
    except Exception as exc:
        logger.debug(f"[sec_edgar] CIK lookup error: {exc}")

    _cik_cache[ticker_upper] = None
    return None


# ── Submissions ───────────────────────────────────────────────────────────────

def _fetch_submissions(cik: int, headers: dict) -> dict | None:
    """Fetch the company submissions JSON (contains full filing history)."""
    try:
        resp = requests.get(
            _SUBMISSIONS_URL.format(cik=cik),
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.debug(f"[sec_edgar] Submissions fetch error: {exc}")
    return None


# ── 8-K material events ───────────────────────────────────────────────────────

def _extract_8k_events(submissions: dict) -> list[dict[str, str]]:
    """
    Extract recent 8-K material event descriptions.
    8-Ks cover: earnings results, CEO changes, acquisitions, legal actions.
    """
    events: list[dict[str, str]] = []
    cutoff = datetime.now() - timedelta(days=30)

    try:
        recent = submissions.get("filings", {}).get("recent", {})
        forms  = recent.get("form",        [])
        dates  = recent.get("filingDate",  [])
        items  = recent.get("items",       [])  # 8-K item codes

        for i, (form, date_str) in enumerate(zip(forms, dates)):
            if form not in ("8-K", "8-K/A"):
                continue
            try:
                if datetime.strptime(date_str, "%Y-%m-%d") < cutoff:
                    continue
                item_code   = items[i] if i < len(items) else ""
                description = _describe_8k_item(item_code) or "Material event disclosed"
                events.append({"date": date_str, "event": description})
                if len(events) >= _MAX_8K_EVENTS:
                    break
            except (ValueError, IndexError):
                continue

    except Exception as exc:
        logger.debug(f"[sec_edgar] 8-K extraction error: {exc}")

    return events


_8K_ITEM_MAP: dict[str, str] = {
    "1.01": "Material definitive agreement",
    "1.02": "Termination of material agreement",
    "1.03": "Bankruptcy or receivership",
    "2.01": "Acquisition or disposition of assets",
    "2.02": "Results of operations (earnings)",
    "2.03": "Material direct financial obligation",
    "2.04": "Triggering events affecting obligations",
    "2.05": "Costs associated with exit activities",
    "2.06": "Material impairment",
    "3.01": "Delisting or failure to satisfy listing rule",
    "3.03": "Material modification to rights of shareholders",
    "4.01": "Changes in registrant's certifying accountant",
    "4.02": "Non-reliance on prior financial statements",
    "5.01": "Changes in control",
    "5.02": "Departure/appointment of directors or officers",
    "5.03": "Amendments to articles of incorporation",
    "5.05": "Shareholder action",
    "7.01": "Regulation FD disclosure",
    "8.01": "Other events",
    "9.01": "Financial statements and exhibits",
}

_RISK_SIGNALS: frozenset[str] = frozenset([
    "risk", "may", "could", "might", "uncertain", "depend", "rely",
    "competition", "regulation", "fail", "loss", "decline", "adverse",
    "volatile", "fluctuat", "negatively", "harm", "impair", "disrupt",
])


def _describe_8k_item(item_code: str) -> str:
    """Map 8-K item codes to human-readable descriptions."""
    for code, desc in _8K_ITEM_MAP.items():
        if code in item_code:
            return desc
    return item_code if item_code else "Material event"


# ── 10-K risk factors ─────────────────────────────────────────────────────────

def _latest_10k_date(submissions: dict) -> str | None:
    try:
        recent = submissions.get("filings", {}).get("recent", {})
        for form, date in zip(recent.get("form", []), recent.get("filingDate", [])):
            if form in ("10-K", "10-K405"):
                return date
    except Exception:
        pass
    return None


def _extract_risk_factors(
    cik: int, submissions: dict, headers: dict
) -> list[str]:
    """
    Extract top risk factor headings from the latest 10-K.
    Streams the document and stops after Item 1A section is found.
    """
    try:
        recent      = submissions.get("filings", {}).get("recent", {})
        forms       = recent.get("form",            [])
        accessions  = recent.get("accessionNumber", [])
        docs        = recent.get("primaryDocument", [])

        for form, accession, doc in zip(forms, accessions, docs):
            if form not in ("10-K", "10-K405"):
                continue

            acc_clean = accession.replace("-", "")
            doc_url   = _ARCHIVES_URL.format(cik=cik, acc_clean=acc_clean, doc=doc)

            content = _stream_document(doc_url, headers, _MAX_DOC_BYTES)
            if content:
                factors = _parse_risk_factor_headings(content)
                if factors:
                    return factors

            break  # Only try the latest 10-K

    except Exception as exc:
        logger.debug(f"[sec_edgar] Risk factor extraction error: {exc}")

    return []


def _stream_document(url: str, headers: dict, max_bytes: int) -> str:
    """Stream a filing document up to max_bytes."""
    try:
        resp = requests.get(url, headers=headers, timeout=20, stream=True)
        resp.raise_for_status()
        content = b""
        for chunk in resp.iter_content(chunk_size=16_384):
            content += chunk
            if len(content) >= max_bytes:
                break
        return content.decode("utf-8", errors="replace")
    except Exception as exc:
        logger.debug(f"[sec_edgar] Document stream error for {url}: {exc}")
    return ""


def _parse_risk_factor_headings(html: str) -> list[str]:
    """
    Extract risk factor headings from 10-K HTML text.

    Strategy:
    1. Strip HTML tags and decode entities
    2. Locate Item 1A section using regex
    3. Extract sentences that look like risk headings:
       - 5–30 words
       - Contain risk-signalling language
    """
    # Strip HTML
    text = re.sub(r"<[^>]+>",      " ", html)
    text = re.sub(r"&[a-zA-Z]+;",  " ", text)
    text = re.sub(r"&#\d+;",       " ", text)
    text = re.sub(r"\s+",          " ", text).strip()

    # Find Item 1A section
    m = re.search(
        r"ITEM\s+1A\.?\s*RISK\s+FACTORS(.*?)(?:ITEM\s+1B|ITEM\s+2|\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return []

    section = m.group(1)[:8_000]  # First 8KB of the section

    # Split into candidate sentences
    candidates = re.split(r"(?<=[.?!])\s+(?=[A-Z])", section)

    headings: list[str] = []

    for candidate in candidates[:50]:
        candidate = candidate.strip().strip(".")
        words = candidate.split()
        if 5 <= len(words) <= 30:
            if any(sig in candidate.lower() for sig in _RISK_SIGNALS):
                headings.append(candidate)
                if len(headings) >= _MAX_RISK_FACTORS:
                    break

    return headings


# ── Defaults ──────────────────────────────────────────────────────────────────

def _empty() -> dict[str, Any]:
    return {
        "risk_factors":           [],
        "recent_material_events": [],
        "latest_10k_date":        None,
        "source":                 "SEC EDGAR (unavailable)",
    }
