"""
tests/test_escalation.py
========================
Unit tests for the rule-based escalation logic in BaseAgent.

Tests the check_rule_escalation() method with all supported operators
across all agent escalation rule configurations.

No Anthropic API key is required — all tests mock the AsyncAnthropic client
and test only the rule evaluation logic.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agents.base_agent import BaseAgent
from config import ESCALATION_RULES


# ── Minimal concrete subclass for testing ─────────────────────────────────────

class _StubAgent(BaseAgent):
    """Minimal BaseAgent subclass that satisfies the abstract interface."""

    def __init__(self, agent_name: str) -> None:
        # Bypass __init__ to avoid needing session/logger/API client
        self.agent_name           = agent_name
        self.escalation_rules     = ESCALATION_RULES.get(agent_name, [])
        self.confidence_threshold = None
        self.session              = MagicMock()
        self.logger               = MagicMock()

    def get_system_prompt(self) -> str:
        return "stub"

    async def run(self, context):
        return {}


# ── Financial Agent escalation rules ─────────────────────────────────────────

class TestFinancialEscalationRules:
    def setup_method(self):
        self.agent = _StubAgent("financial")

    def test_triggers_when_quality_below_threshold(self):
        triggered, reason = self.agent.check_rule_escalation(
            {"data_quality_hint": 0.4}
        )
        assert triggered is True
        assert "quality" in reason.lower()

    def test_does_not_trigger_at_threshold(self):
        triggered, _ = self.agent.check_rule_escalation(
            {"data_quality_hint": 0.6}   # exactly at threshold — should NOT trigger
        )
        assert triggered is False

    def test_does_not_trigger_above_threshold(self):
        triggered, _ = self.agent.check_rule_escalation(
            {"data_quality_hint": 0.9}
        )
        assert triggered is False

    def test_missing_field_does_not_trigger(self):
        triggered, _ = self.agent.check_rule_escalation({})
        assert triggered is False


# ── Risk Agent escalation rules ───────────────────────────────────────────────

class TestRiskEscalationRules:
    def setup_method(self):
        self.agent = _StubAgent("risk")

    def test_triggers_on_extreme_leverage(self):
        triggered, reason = self.agent.check_rule_escalation(
            {"debt_to_equity": 4.2, "conflicting_signals": False}
        )
        assert triggered is True
        assert "leverage" in reason.lower() or "D/E" in reason

    def test_triggers_on_conflicting_signals(self):
        triggered, reason = self.agent.check_rule_escalation(
            {"debt_to_equity": 0.5, "conflicting_signals": True}
        )
        assert triggered is True
        assert "conflict" in reason.lower()

    def test_does_not_trigger_on_clean_data(self):
        triggered, _ = self.agent.check_rule_escalation(
            {"debt_to_equity": 1.5, "conflicting_signals": False}
        )
        assert triggered is False

    def test_triggers_on_boundary_de(self):
        # D/E of exactly 3.0 should NOT trigger (rule is strictly >)
        triggered, _ = self.agent.check_rule_escalation(
            {"debt_to_equity": 3.0, "conflicting_signals": False}
        )
        assert triggered is False

    def test_triggers_just_above_boundary(self):
        triggered, _ = self.agent.check_rule_escalation(
            {"debt_to_equity": 3.01, "conflicting_signals": False}
        )
        assert triggered is True

    def test_first_matching_rule_wins(self):
        """Both rules triggered — should return on the first one."""
        triggered, reason = self.agent.check_rule_escalation(
            {"debt_to_equity": 5.0, "conflicting_signals": True}
        )
        assert triggered is True
        # First rule is debt_to_equity
        assert "leverage" in reason.lower() or "D/E" in reason


# ── Recommendation Agent escalation rules ─────────────────────────────────────

class TestRecommendationEscalationRules:
    def setup_method(self):
        self.agent = _StubAgent("recommendation")

    def test_triggers_on_high_risk(self):
        triggered, reason = self.agent.check_rule_escalation(
            {"risk_level": "high"}
        )
        assert triggered is True
        assert "high" in reason.lower() or "risk" in reason.lower()

    def test_does_not_trigger_on_medium_risk(self):
        triggered, _ = self.agent.check_rule_escalation(
            {"risk_level": "medium"}
        )
        assert triggered is False

    def test_does_not_trigger_on_low_risk(self):
        triggered, _ = self.agent.check_rule_escalation(
            {"risk_level": "low"}
        )
        assert triggered is False


# ── Formatter Agent — no rules ────────────────────────────────────────────────

class TestFormatterEscalationRules:
    def setup_method(self):
        self.agent = _StubAgent("formatter")

    def test_never_triggers(self):
        triggered, _ = self.agent.check_rule_escalation(
            {"debt_to_equity": 99.0, "risk_level": "high", "data_quality_hint": 0.0}
        )
        assert triggered is False


# ── Operator coverage ─────────────────────────────────────────────────────────

class TestOperators:
    """Test all four supported operators directly using a custom rule list."""

    def _make_agent_with_rule(self, operator: str, value, field: str = "x") -> _StubAgent:
        agent = _StubAgent("financial")
        agent.escalation_rules = [
            {"field": field, "operator": operator, "value": value, "reason": "test"}
        ]
        return agent

    def test_less_than_triggers(self):
        agent = self._make_agent_with_rule("<", 5)
        triggered, _ = agent.check_rule_escalation({"x": 4})
        assert triggered is True

    def test_less_than_no_trigger(self):
        agent = self._make_agent_with_rule("<", 5)
        triggered, _ = agent.check_rule_escalation({"x": 5})
        assert triggered is False

    def test_greater_than_triggers(self):
        agent = self._make_agent_with_rule(">", 3)
        triggered, _ = agent.check_rule_escalation({"x": 4})
        assert triggered is True

    def test_equals_triggers(self):
        agent = self._make_agent_with_rule("==", "high")
        triggered, _ = agent.check_rule_escalation({"x": "high"})
        assert triggered is True

    def test_not_equals_triggers(self):
        agent = self._make_agent_with_rule("!=", "active")
        triggered, _ = agent.check_rule_escalation({"x": "inactive"})
        assert triggered is True
