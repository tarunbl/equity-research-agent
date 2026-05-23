"""
utils/context_builder.py
========================
Minimum-sufficient context for each agent in the pipeline.

Phase 3 additions
-----------------
  build_risk_context() now accepts risk_intelligence — SEC 10-K risk factors,
  recent 8-K material events, and Finnhub insider transaction summary.
  These are passed directly into the risk agent's reasoning context.
"""
from __future__ import annotations
from typing import Any


def build_financial_context(ticker: str, raw_data: dict[str, Any]) -> dict[str, Any]:
    return {
        "ticker":            ticker,
        "raw_data":          raw_data,
        "data_quality_hint": raw_data.get("data_quality_hint", 1.0),
    }


def build_valuation_context(
    ticker: str, raw_financial: dict[str, Any]
) -> dict[str, Any]:
    return {"ticker": ticker, "raw_financial": raw_financial}


def build_risk_context(
    financial:         dict[str, Any],
    news:              dict[str, Any],
    valuation:         dict[str, Any],
    risk_intelligence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Context for the Risk Agent.

    risk_intelligence (optional) adds:
      - SEC 10-K risk factors    — company's own legal risk disclosures
      - Recent 8-K events        — material events from last 30 days
      - Insider transaction data — are executives buying or selling?
    """
    financial_healthy = (
        financial.get("revenue_growth_pct", 0) > 0
        and financial.get("debt_to_equity", 99.0) < 2.0
        and financial.get("profit_margin_pct", 0) > 5.0
    )
    sentiment_negative = news.get("sentiment_score", 0.0) < -0.3
    sentiment_positive = news.get("sentiment_score", 0.0) > 0.3
    conflicting_signals = (
        (financial_healthy and sentiment_negative)
        or (not financial_healthy and sentiment_positive)
    )

    ri = risk_intelligence or {}

    return {
        "ticker":               financial["ticker"],
        "company_name":         financial["company_name"],
        # Financials
        "revenue_growth_pct":   financial["revenue_growth_pct"],
        "gross_margin_pct":     financial["gross_margin_pct"],
        "profit_margin_pct":    financial["profit_margin_pct"],
        "pe_ratio":             financial["pe_ratio"],
        "debt_to_equity":       financial["debt_to_equity"],
        "current_ratio":        financial["current_ratio"],
        "return_on_equity_pct": financial["return_on_equity_pct"],
        "financial_anomalies":  financial.get("anomalies", []),
        # News
        "overall_sentiment":    news["overall_sentiment"],
        "sentiment_score":      news["sentiment_score"],
        "negative_pct":         news["negative_pct"],
        "key_themes":           news["key_themes"],
        "top_headlines":        news["top_headlines"][:5],
        # Derived
        "conflicting_signals":  conflicting_signals,
        # Valuation context
        "current_price":           valuation.get("current_price"),
        "pct_off_52w_high":        valuation.get("pct_off_52w_high"),
        "valuation_stance":        valuation.get("valuation_stance"),
        "earnings_within_14_days": valuation.get("earnings_within_14_days", False),
        # Phase 3: SEC + insider intelligence
        "sec_risk_factors":           ri.get("sec_risk_factors",           []),
        "recent_material_events":     ri.get("recent_material_events",     []),
        "latest_10k_date":            ri.get("latest_10k_date"),
        "insider_signal":             ri.get("insider_signal",             "unavailable"),
        "insider_summary":            ri.get("insider_summary",            ""),
        "insider_notable":            ri.get("insider_notable",            []),
        "intelligence_sources":       ri.get("sources",                    []),
    }


def build_recommendation_context(
    financial:     dict[str, Any],
    risk:          dict[str, Any],
    valuation:     dict[str, Any],
    raw_financial: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    raw_financial: the unstructured dict directly from financial_api.
    Needed to recover recommendation_mean and high_price_target, which are
    dropped when the FinancialAgent structures its LLM output schema.
    """
    raw = raw_financial or {}
    return {
        "ticker":                  financial["ticker"],
        "company_name":            financial["company_name"],
        "revenue_growth_pct":      financial["revenue_growth_pct"],
        "profit_margin_pct":       financial["profit_margin_pct"],
        "pe_ratio":                financial["pe_ratio"],
        "debt_to_equity":          financial["debt_to_equity"],
        "risk_level":              risk["risk_level"],
        "risk_score":              risk["risk_score"],
        "conflicting_signals":     risk["conflicting_signals"],
        "financial_risk_summary":  risk["financial_risk_summary"],
        "sentiment_risk_summary":  risk["sentiment_risk_summary"],
        "top_risk_factors":        risk["top_risk_factors"],
        "mitigating_factors":      risk["mitigating_factors"],
        "current_price":           valuation.get("current_price"),
        "pct_off_52w_high":        valuation.get("pct_off_52w_high"),
        "week_52_high":            valuation.get("week_52_high"),
        "forward_pe":              valuation.get("forward_pe"),
        "peg_ratio":               valuation.get("peg_ratio"),
        "analyst_count":           valuation.get("analyst_count"),
        "analyst_consensus":       valuation.get("analyst_consensus"),
        "mean_price_target":       valuation.get("mean_price_target"),
        "upside_to_target_pct":    valuation.get("upside_to_target_pct"),
        "valuation_stance":        valuation.get("valuation_stance"),
        "valuation_insights":      valuation.get("key_insights", []),
        "earnings_date":           valuation.get("earnings_date"),
        "earnings_within_14_days": valuation.get("earnings_within_14_days", False),
        # Read from raw API data — not routed through LLM schemas where fields get dropped
        "recommendation_mean":     raw.get("recommendation_mean", 0.0),
        "high_price_target":       raw.get("high_price_target", 0.0),
    }


def build_formatter_context(
    financial:      dict[str, Any],
    news:           dict[str, Any],
    valuation:      dict[str, Any],
    risk:           dict[str, Any],
    recommendation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "ticker":         financial["ticker"],
        "company_name":   financial["company_name"],
        "financial":      financial,
        "news":           news,
        "valuation":      valuation,
        "risk":           risk,
        "recommendation": recommendation,
    }
