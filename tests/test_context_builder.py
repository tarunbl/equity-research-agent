"""
tests/test_context_builder.py
==============================
Unit tests for the context builder utilities.

Verifies that:
  - Each build_*_context() function returns only the expected fields.
  - conflicting_signals is computed correctly for all four combinations.
  - Context scoping prevents downstream agents from accessing upstream data
    they should not see.

No Anthropic API key is required.
"""
from __future__ import annotations

import pytest
from utils.context_builder import (
    build_financial_context,
    build_formatter_context,
    build_recommendation_context,
    build_risk_context,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def healthy_financial():
    return {
        "ticker": "AAPL", "company_name": "Apple Inc.",
        "revenue_growth_pct": 8.2, "gross_margin_pct": 44.5,
        "profit_margin_pct": 25.3, "pe_ratio": 28.4,
        "debt_to_equity": 1.52, "current_ratio": 1.07,
        "return_on_equity_pct": 147.2, "data_quality_score": 0.95,
        "anomalies": [], "confidence": 0.90, "escalate": False,
        "escalation_reason": None,
    }


@pytest.fixture
def weak_financial():
    return {
        "ticker": "GME", "company_name": "GameStop Corp.",
        "revenue_growth_pct": -28.4, "gross_margin_pct": 21.3,
        "profit_margin_pct": -4.2, "pe_ratio": -8.1,
        "debt_to_equity": 4.20, "current_ratio": 2.10,
        "return_on_equity_pct": -12.3, "data_quality_score": 0.85,
        "anomalies": ["Negative P/E"], "confidence": 0.88,
        "escalate": False, "escalation_reason": None,
    }


@pytest.fixture
def positive_news():
    return {
        "ticker": "AAPL", "sentiment_score": 0.55, "overall_sentiment": "positive",
        "positive_pct": 70.0, "neutral_pct": 20.0, "negative_pct": 10.0,
        "top_headlines": [], "key_themes": ["AI", "services growth"],
        "headlines_analyzed": 7, "confidence": 0.88,
    }


@pytest.fixture
def negative_news():
    return {
        "ticker": "GME", "sentiment_score": -0.60, "overall_sentiment": "negative",
        "positive_pct": 10.0, "neutral_pct": 15.0, "negative_pct": 75.0,
        "top_headlines": [], "key_themes": ["revenue decline", "debt concerns"],
        "headlines_analyzed": 8, "confidence": 0.85,
    }


@pytest.fixture
def low_risk():
    return {
        "ticker": "AAPL", "risk_level": "low", "risk_score": 2.1,
        "conflicting_signals": False,
        "financial_risk_summary": "Healthy balance sheet.",
        "sentiment_risk_summary": "Positive market narrative.",
        "sector_risk_summary": "Stable tech sector.",
        "top_risk_factors": ["China exposure"],
        "mitigating_factors": ["Strong cash flow"],
        "confidence": 0.91,
    }


@pytest.fixture
def high_risk():
    return {
        "ticker": "GME", "risk_level": "high", "risk_score": 8.4,
        "conflicting_signals": False,
        "financial_risk_summary": "Extreme leverage.",
        "sentiment_risk_summary": "Predominantly negative.",
        "sector_risk_summary": "Structural retail decline.",
        "top_risk_factors": ["High D/E", "Revenue decline"],
        "mitigating_factors": ["Cash buffer"],
        "confidence": 0.87,
    }


# ── build_financial_context ───────────────────────────────────────────────────

class TestBuildFinancialContext:
    def test_includes_ticker(self, healthy_financial):
        raw = {"data_quality_hint": 0.95, **healthy_financial}
        ctx = build_financial_context("AAPL", raw)
        assert ctx["ticker"] == "AAPL"

    def test_surfaces_quality_hint(self, healthy_financial):
        raw = {"data_quality_hint": 0.55, **healthy_financial}
        ctx = build_financial_context("AAPL", raw)
        assert ctx["data_quality_hint"] == 0.55

    def test_defaults_quality_hint_to_one(self, healthy_financial):
        # When hint is not in raw data, defaults to 1.0
        ctx = build_financial_context("AAPL", healthy_financial)
        assert ctx["data_quality_hint"] == 1.0


# ── build_risk_context — conflicting_signals logic ────────────────────────────

class TestConflictingSignals:
    """Verify all four healthy/weak × positive/negative combinations."""

    def test_healthy_financial_positive_news_no_conflict(
        self, healthy_financial, positive_news
    ):
        ctx = build_risk_context(healthy_financial, positive_news)
        assert ctx["conflicting_signals"] is False

    def test_healthy_financial_negative_news_is_conflict(
        self, healthy_financial, negative_news
    ):
        ctx = build_risk_context(healthy_financial, negative_news)
        assert ctx["conflicting_signals"] is True

    def test_weak_financial_negative_news_no_conflict(
        self, weak_financial, negative_news
    ):
        ctx = build_risk_context(weak_financial, negative_news)
        assert ctx["conflicting_signals"] is False

    def test_weak_financial_positive_news_is_conflict(
        self, weak_financial, positive_news
    ):
        ctx = build_risk_context(weak_financial, positive_news)
        assert ctx["conflicting_signals"] is True

    def test_surfaces_debt_to_equity_for_rules(self, healthy_financial, positive_news):
        ctx = build_risk_context(healthy_financial, positive_news)
        assert ctx["debt_to_equity"] == healthy_financial["debt_to_equity"]

    def test_limits_headlines_to_five(self, healthy_financial, positive_news):
        positive_news["top_headlines"] = [{"headline": f"h{i}"} for i in range(10)]
        ctx = build_risk_context(healthy_financial, positive_news)
        assert len(ctx["top_headlines"]) == 5


# ── build_recommendation_context ─────────────────────────────────────────────

class TestBuildRecommendationContext:
    def test_surfaces_risk_level(self, healthy_financial, low_risk):
        ctx = build_recommendation_context(healthy_financial, low_risk)
        assert ctx["risk_level"] == "low"

    def test_does_not_include_raw_news(self, healthy_financial, low_risk):
        ctx = build_recommendation_context(healthy_financial, low_risk)
        # News headlines should NOT be in recommendation context
        assert "top_headlines" not in ctx
        assert "negative_pct" not in ctx

    def test_includes_key_financial_metrics(self, healthy_financial, low_risk):
        ctx = build_recommendation_context(healthy_financial, low_risk)
        assert "revenue_growth_pct" in ctx
        assert "profit_margin_pct" in ctx
        assert "debt_to_equity" in ctx


# ── build_formatter_context ───────────────────────────────────────────────────

class TestBuildFormatterContext:
    def test_includes_all_sections(
        self, healthy_financial, positive_news, low_risk
    ):
        rec = {"ticker": "AAPL", "signal": "buy", "conviction": "high",
               "time_horizon": "12 months", "rationale": "Strong growth.",
               "bull_case": "Services boom.", "bear_case": "Regulation.",
               "key_catalysts": ["AI"], "key_risks": ["China"],
               "confidence": 0.9}
        ctx = build_formatter_context(healthy_financial, positive_news, low_risk, rec)
        for key in ["ticker", "company_name", "financial", "news", "risk", "recommendation"]:
            assert key in ctx

    def test_formatter_receives_full_financial(
        self, healthy_financial, positive_news, low_risk
    ):
        rec = {"ticker": "AAPL", "signal": "buy", "conviction": "high",
               "time_horizon": "12 months", "rationale": "r",
               "bull_case": "b", "bear_case": "b",
               "key_catalysts": [], "key_risks": [], "confidence": 0.9}
        ctx = build_formatter_context(healthy_financial, positive_news, low_risk, rec)
        # Formatter should have full financial data, not a subset
        assert ctx["financial"]["gross_margin_pct"] == healthy_financial["gross_margin_pct"]
