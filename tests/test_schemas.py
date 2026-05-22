"""
tests/test_schemas.py
=====================
Unit tests for Pydantic v2 output schemas.

These tests verify that:
  - Valid data passes schema validation cleanly.
  - Invalid data (wrong types, out-of-range values) raises ValidationError.
  - Enum fields reject unexpected values.

No Anthropic API key is required — all tests are pure Python.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from models.schemas import (
    AgentMetrics,
    Conviction,
    FinancialOutput,
    FormatterOutput,
    HeadlineSentiment,
    NewsOutput,
    RecommendationOutput,
    RiskLevel,
    RiskOutput,
    Signal,
)


# ── FinancialOutput ───────────────────────────────────────────────────────────

class TestFinancialOutput:
    def _valid(self) -> dict:
        return {
            "ticker": "AAPL", "company_name": "Apple Inc.",
            "revenue_growth_pct": 8.2, "gross_margin_pct": 44.5,
            "profit_margin_pct": 25.3, "pe_ratio": 28.4,
            "debt_to_equity": 1.52, "current_ratio": 1.07,
            "return_on_equity_pct": 147.2, "data_quality_score": 0.95,
            "confidence": 0.90,
        }

    def test_valid_passes(self):
        out = FinancialOutput(**self._valid())
        assert out.ticker == "AAPL"
        assert out.escalate is False
        assert out.anomalies == []

    def test_confidence_out_of_range(self):
        data = self._valid()
        data["confidence"] = 1.5   # > 1.0 not allowed
        with pytest.raises(ValidationError):
            FinancialOutput(**data)

    def test_data_quality_score_out_of_range(self):
        data = self._valid()
        data["data_quality_score"] = -0.1  # < 0.0 not allowed
        with pytest.raises(ValidationError):
            FinancialOutput(**data)

    def test_anomalies_are_optional(self):
        data = self._valid()
        data["anomalies"] = ["Negative P/E detected"]
        out = FinancialOutput(**data)
        assert len(out.anomalies) == 1

    def test_escalation_fields_optional(self):
        out = FinancialOutput(**self._valid())
        assert out.escalation_reason is None


# ── NewsOutput ────────────────────────────────────────────────────────────────

class TestNewsOutput:
    def _valid(self) -> dict:
        return {
            "ticker": "TSLA", "headlines_analyzed": 8,
            "overall_sentiment": "negative", "sentiment_score": -0.35,
            "positive_pct": 25.0, "neutral_pct": 25.0, "negative_pct": 50.0,
            "top_headlines": [
                {"headline": "Tesla misses delivery estimates", "sentiment": "negative", "score": -0.6}
            ],
            "key_themes": ["delivery miss", "price cuts", "competition"],
            "confidence": 0.82,
        }

    def test_valid_passes(self):
        out = NewsOutput(**self._valid())
        assert out.ticker == "TSLA"
        assert len(out.top_headlines) == 1
        assert isinstance(out.top_headlines[0], HeadlineSentiment)

    def test_confidence_bounds(self):
        data = self._valid()
        data["confidence"] = -0.1
        with pytest.raises(ValidationError):
            NewsOutput(**data)

    def test_key_themes_list(self):
        out = NewsOutput(**self._valid())
        assert isinstance(out.key_themes, list)


# ── RiskOutput ────────────────────────────────────────────────────────────────

class TestRiskOutput:
    def _valid(self) -> dict:
        return {
            "ticker": "GME", "risk_level": "high", "risk_score": 8.4,
            "conflicting_signals": False,
            "financial_risk_summary": "Extreme leverage and declining revenue.",
            "sentiment_risk_summary": "Predominantly negative market narrative.",
            "sector_risk_summary": "Physical retail structural decline.",
            "top_risk_factors": ["High D/E ratio", "Declining revenue", "CEO departure"],
            "mitigating_factors": ["Cash on hand", "Low short-term debt"],
            "confidence": 0.88,
        }

    def test_valid_passes(self):
        out = RiskOutput(**self._valid())
        assert out.risk_level == RiskLevel.HIGH
        assert out.risk_score == 8.4

    def test_invalid_risk_level(self):
        data = self._valid()
        data["risk_level"] = "extreme"   # not a valid RiskLevel
        with pytest.raises(ValidationError):
            RiskOutput(**data)

    def test_risk_score_float(self):
        out = RiskOutput(**self._valid())
        assert isinstance(out.risk_score, float)


# ── RecommendationOutput ──────────────────────────────────────────────────────

class TestRecommendationOutput:
    def _valid(self) -> dict:
        return {
            "ticker": "AAPL", "signal": "buy", "conviction": "high",
            "time_horizon": "12–18 months", "rationale": "Strong services growth.",
            "bull_case": "Services reach $40B revenue.", "bear_case": "Macro headwinds.",
            "key_catalysts": ["AI integration", "India expansion"],
            "key_risks": ["Regulatory scrutiny", "China revenue"],
            "confidence": 0.87,
        }

    def test_valid_passes(self):
        out = RecommendationOutput(**self._valid())
        assert out.signal == Signal.BUY
        assert out.conviction == Conviction.HIGH

    def test_invalid_signal(self):
        data = self._valid()
        data["signal"] = "maybe"
        with pytest.raises(ValidationError):
            RecommendationOutput(**data)

    def test_invalid_conviction(self):
        data = self._valid()
        data["conviction"] = "very_high"
        with pytest.raises(ValidationError):
            RecommendationOutput(**data)


# ── AgentMetrics ──────────────────────────────────────────────────────────────

class TestAgentMetrics:
    def test_defaults(self):
        m = AgentMetrics(agent_name="risk", model_used="claude-sonnet-4-6")
        assert m.input_tokens == 0
        assert m.escalated is False
        assert m.escalation_reason is None
        assert m.latency_ms == 0.0

    def test_with_escalation(self):
        m = AgentMetrics(
            agent_name="risk", model_used="claude-opus-4-6",
            input_tokens=900, output_tokens=500,
            escalated=True, escalation_reason="D/E > 3.0",
            escalation_layer="rule", latency_ms=3200.0,
        )
        assert m.escalated is True
        assert m.escalation_layer == "rule"
