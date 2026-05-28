"""
agents/base_agent.py
====================
Abstract base class for all pipeline agents.

Responsibilities
----------------
  1. Model routing     — reads default/escalate model from config
  2. Prompt caching    — cache_control on system prompt
  3. Retry w/ backoff  — handles transient API errors
  4. Rule escalation   — pre-LLM Python rule check; zero token cost
  5. Confidence check  — post-LLM escalation on low self-reported confidence
  6. JSON extraction   — multi-strategy parser (code fence → raw JSON → regex)
  7. Usage logging     — one metrics entry per agent call (accumulates both
                         calls when confidence escalation triggers)

Concrete agents implement only:
  get_system_prompt() → str
  run(context: dict)  → dict
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Tuple

from anthropic import (
    AsyncAnthropic,
    APIConnectionError,
    APIStatusError,
    RateLimitError,
)

from config import (
    API_TIMEOUT_SECS,
    CONFIDENCE_THRESHOLDS,
    ESCALATION_RULES,
    MAX_RETRIES,
    MODEL_ROUTING,
    RETRY_BASE_DELAY_SECS,
    TOKEN_BUDGETS,
)
from memory.session_store import SessionStore
from models.schemas import AgentMetrics
from monitoring.logger import RunLogger


class AgentError(Exception):
    """Raised when an agent cannot complete its task after all retries."""


class BaseAgent(ABC):

    def __init__(
        self,
        agent_name: str,
        session: SessionStore,
        logger: RunLogger,
    ) -> None:
        self.agent_name           = agent_name
        self.client               = AsyncAnthropic(timeout=API_TIMEOUT_SECS)
        self.model_config         = MODEL_ROUTING[agent_name]
        self.max_tokens           = TOKEN_BUDGETS[agent_name]
        self.confidence_threshold = CONFIDENCE_THRESHOLDS.get(agent_name)
        self.escalation_rules     = ESCALATION_RULES.get(agent_name, [])
        self.session              = session
        self.logger               = logger

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the static system prompt for this agent."""

    @abstractmethod
    async def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Execute the agent task. Must return a validated output dict."""

    # ── Rule-based escalation ─────────────────────────────────────────────────

    def check_rule_escalation(
        self, context: dict[str, Any]
    ) -> tuple[bool, str]:
        """
        Evaluate escalation rules against input context.
        Runs BEFORE the LLM — zero token cost.
        Returns (should_escalate, reason).
        """
        for rule in self.escalation_rules:
            actual = context.get(rule["field"])
            if actual is None:
                continue
            op, threshold = rule["operator"], rule["value"]
            triggered = (
                (op == "<"  and actual < threshold)  or
                (op == ">"  and actual > threshold)  or
                (op == "==" and actual == threshold) or
                (op == "!=" and actual != threshold)
            )
            if triggered:
                return True, rule["reason"]
        return False, ""

    # ── Core LLM call ─────────────────────────────────────────────────────────

    async def call_llm(
        self,
        user_message: str,
        context_for_rules: dict[str, Any] | None = None,
        force_escalate: bool = False,
        system_prompt: str | None = None,
    ) -> Tuple[dict[str, Any], AgentMetrics]:
        """
        Full lifecycle LLM call:
          Pre-call  : rule-based escalation check (free)
          Call      : prompt-cached request with exponential-backoff retry
          Post-call : confidence-based escalation check
          Logging   : ONE metrics entry per call_llm() invocation,
                      accumulating tokens from both calls if escalation triggers

        Returns (parsed_output_dict, AgentMetrics).
        """
        # ── Determine starting model ──────────────────────────────────────────
        rule_escalated    = False
        escalation_reason = ""
        escalation_layer: str | None = None

        if force_escalate:
            model             = self.model_config["escalate"]
            rule_escalated    = True
            escalation_reason = "Forced escalation"
            escalation_layer  = "forced"
        elif context_for_rules:
            rule_escalated, escalation_reason = self.check_rule_escalation(context_for_rules)
            model            = self.model_config["escalate"] if rule_escalated else self.model_config["default"]
            escalation_layer = "rule" if rule_escalated else None
        else:
            model = self.model_config["default"]

        # ── First API call ────────────────────────────────────────────────────
        response, latency_ms = await self._call_with_retry(model, user_message, system_prompt)
        parsed               = self._extract_json(response.content[0].text)

        # Accumulate usage from all calls (may grow if confidence escalation triggers)
        usage = _sum_usage(response.usage)

        # ── Confidence-based escalation ───────────────────────────────────────
        conf_escalated = False
        if (
            not rule_escalated
            and self.confidence_threshold is not None
            and isinstance(parsed, dict)
        ):
            confidence = parsed.get("confidence", 1.0)
            if confidence < self.confidence_threshold:
                escalated_model = self.model_config["escalate"]
                if escalated_model != model:
                    conf_escalated    = True
                    escalation_layer  = "confidence"
                    escalation_reason = (
                        f"Confidence {confidence:.2f} below "
                        f"threshold {self.confidence_threshold}"
                    )
                    esc_response, esc_latency = await self._call_with_retry(
                        escalated_model, user_message, system_prompt
                    )
                    # Accumulate — single metrics entry covers both calls
                    usage        = _add_usage(usage, _sum_usage(esc_response.usage))
                    latency_ms  += esc_latency
                    parsed       = self._extract_json(esc_response.content[0].text)
                    model        = escalated_model

        # ── Log single metrics entry ──────────────────────────────────────────
        escalated = rule_escalated or conf_escalated
        metrics   = AgentMetrics(
            agent_name         = self.agent_name,
            model_used         = model,
            input_tokens       = usage["input"],
            output_tokens      = usage["output"],
            cache_write_tokens = usage["cache_write"],
            cache_read_tokens  = usage["cache_read"],
            escalated          = escalated,
            escalation_reason  = escalation_reason if escalated else None,
            escalation_layer   = escalation_layer  if escalated else None,
            latency_ms         = latency_ms,
        )
        self.logger.log(metrics)
        return parsed, metrics

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _call_with_retry(
        self, model: str, user_message: str, system_prompt: str | None = None
    ) -> Tuple[Any, float]:
        """
        Anthropic API call with exponential-backoff retry.
        Retries on RateLimitError and transient connection errors.
        Raises AgentError after MAX_RETRIES exhausted.
        Returns (response, latency_ms).
        """
        start = time.perf_counter()
        for attempt in range(MAX_RETRIES):
            try:
                prompt   = system_prompt if system_prompt is not None else self.get_system_prompt()
                response = await self.client.messages.create(
                    model      = model,
                    max_tokens = self.max_tokens,
                    system     = [
                        {
                            "type":          "text",
                            "text":          prompt,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    messages = [{"role": "user", "content": user_message}],
                )
                return response, (time.perf_counter() - start) * 1000

            except RateLimitError as exc:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_BASE_DELAY_SECS * (2 ** attempt))
                    continue
                raise AgentError(
                    f"[{self.agent_name}] Rate limit exceeded after "
                    f"{MAX_RETRIES} retries"
                ) from exc

            except (APIConnectionError, APIStatusError) as exc:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_BASE_DELAY_SECS * (2 ** attempt))
                    continue
                raise AgentError(
                    f"[{self.agent_name}] API error after {MAX_RETRIES} retries: {exc}"
                ) from exc

        raise AgentError(f"[{self.agent_name}] Retry loop exhausted")  # unreachable

    def _extract_json(self, text: str) -> dict[str, Any]:
        """
        Extract a JSON object from LLM output.

        Strategy order:
          1. <output>...</output> tags  (preferred format, complete)
          2. <output>...               (truncated — max_tokens hit, closing tag missing)
          3. ```json ... ```            (markdown code block)
          4. First { ... } block        (last resort)

        For truncated fragments, attempts common JSON completions before failing.
        Raises AgentError if no valid JSON can be extracted.
        """
        candidate = self._find_json_candidate(text)
        if candidate is None:
            raise AgentError(
                f"[{self.agent_name}] No JSON found in LLM response. "
                f"First 200 chars: {text[:200]!r}"
            )

        result = self._parse_with_recovery(candidate)
        if result is None:
            raise AgentError(
                f"[{self.agent_name}] Could not parse JSON from response. "
                f"Fragment (first 200 chars): {candidate[:200]!r}"
            )
        return result

    def _find_json_candidate(self, text: str) -> str | None:
        """Locate the JSON string within LLM output text."""
        # 1. Complete <output> tags
        m = re.search(r"<output>(.*?)</output>", text, re.DOTALL)
        if m:
            return m.group(1).strip()

        # 2. Truncated <output> tag (response cut by max_tokens)
        m = re.search(r"<output>(.*)", text, re.DOTALL)
        if m:
            return m.group(1).strip()

        # 3. Markdown code block
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.DOTALL)
        if m:
            return m.group(1).strip()

        # 4. First JSON object in the response
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            return m.group(0)

        return None

    def _parse_with_recovery(self, text: str) -> dict[str, Any] | None:
        """Parse JSON, trying common completions for truncated responses."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Common truncation patterns and their completions
        for suffix in ("}", "\n}", ',"escalate":false,"escalation_reason":null}'):
            try:
                return json.loads(text + suffix)
            except json.JSONDecodeError:
                continue

        return None


# ── Module-level helpers ──────────────────────────────────────────────────────

def _sum_usage(usage: Any) -> dict[str, int]:
    """Extract token counts from an Anthropic usage object."""
    return {
        "input":       getattr(usage, "input_tokens",                   0) or 0,
        "output":      getattr(usage, "output_tokens",                  0) or 0,
        "cache_write": getattr(usage, "cache_creation_input_tokens",    0) or 0,
        "cache_read":  getattr(usage, "cache_read_input_tokens",        0) or 0,
    }


def _add_usage(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    """Sum two usage dicts element-wise."""
    return {k: a.get(k, 0) + b.get(k, 0) for k in a}
