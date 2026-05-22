"""
agents/financial_agent.py
==========================
Fetches and structures financial data for a given ticker.

Model routing
-------------
  Default  : claude-haiku-4-5   — extraction and structuring
  Escalates: claude-sonnet-4-6  — when data_quality_hint < 0.6 (pre-LLM)
                                   or confidence < 0.70 (post-LLM)

Performance note
----------------
get_financial_data() is synchronous yfinance I/O. It runs in a thread pool
executor so the async event loop is not blocked, allowing Stage 1 to run
financial + news fetches in true parallel (not serialised around yfinance).
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from pydantic import ValidationError

from agents.base_agent import AgentError, BaseAgent
from config import YFINANCE_TIMEOUT_SECS
from memory.session_store import SessionStore
from models.schemas import FinancialOutput
from monitoring.logger import RunLogger
from tools.financial_api import get_financial_data
from utils.context_builder import build_financial_context


class FinancialAgent(BaseAgent):

    def __init__(self, session: SessionStore, logger: RunLogger) -> None:
        super().__init__("financial", session, logger)

    def get_system_prompt(self) -> str:
        return """You are a Financial Data Structuring Agent.

YOUR ONLY JOB: Receive raw financial data and return it as a clean, structured JSON object.
No opinions. No commentary. Structure and validate data only.

RULES:
- Map each raw field to its output field exactly.
- Flag anomalies (negative P/E on profitable company, missing fields, impossible ratios)
  in the "anomalies" array as plain-English strings.
- data_quality_score: 1.0 = all fields present and consistent. 0.0 = critical gaps.
- confidence: 0.80–0.95 when data is complete. Below 0.60 only if major fields missing.
- escalate: true ONLY for unresolvable data quality issues.

CRITICAL: Respond ONLY with valid JSON enclosed in <output></output> tags.

Output schema:
{
  "ticker":               string,
  "company_name":         string,
  "revenue_growth_pct":   number,
  "gross_margin_pct":     number,
  "profit_margin_pct":    number,
  "pe_ratio":             number,
  "debt_to_equity":       number,
  "current_ratio":        number,
  "return_on_equity_pct": number,
  "data_quality_score":   number (0.0–1.0),
  "anomalies":            array of strings,
  "confidence":           number (0.0–1.0),
  "escalate":             boolean,
  "escalation_reason":    string or null
}"""

    async def run(self, context: dict[str, Any]) -> dict[str, Any]:
        ticker = context["ticker"]

        # Fetch in thread pool — yfinance is blocking I/O
        try:
            loop     = asyncio.get_event_loop()
            raw_data = await asyncio.wait_for(
                loop.run_in_executor(None, get_financial_data, ticker),
                timeout=YFINANCE_TIMEOUT_SECS,
            )
        except asyncio.TimeoutError:
            raise AgentError(
                f"[financial] Timed out fetching data for {ticker} "
                f"(>{YFINANCE_TIMEOUT_SECS}s)"
            )

        if not raw_data:
            raise AgentError(
                f"[financial] No data found for '{ticker}'. "
                f"Check the ticker is valid on Yahoo Finance."
            )

        self.session.set("financial_raw", raw_data)

        scoped      = build_financial_context(ticker, raw_data)
        user_message = (
            f"Structure the following raw financial data for {ticker}.\n\n"
            f"Raw data:\n{json.dumps(raw_data, indent=2)}\n\n"
            f"Return structured output inside <output></output> tags."
        )

        parsed, _ = await self.call_llm(
            user_message,
            context_for_rules={"data_quality_hint": scoped["data_quality_hint"]},
        )

        try:
            validated = FinancialOutput(**parsed)
        except ValidationError as exc:
            raise AgentError(
                f"[financial] Output schema validation failed: {exc}"
            ) from exc

        output = validated.model_dump()
        self.session.set("financial_output", output)
        return output
