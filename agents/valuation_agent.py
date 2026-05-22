"""
agents/valuation_agent.py
==========================
Computes a structured valuation snapshot: price context, multiples,
analyst consensus, and earnings proximity.

This agent was added in Phase 2 after the system issued a SELL on ELF
when 16 Wall Street analysts rated it BUY with 89% implied upside.
Without price and consensus data, fundamental weakness looks identical
to a fundamentally weak stock that's already priced for disaster.

Model routing
-------------
  Default  : claude-haiku-4-5   — structured analysis, no deep reasoning
  Escalates: claude-sonnet-4-6  — on low confidence only
"""
from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from agents.base_agent import AgentError, BaseAgent
from memory.session_store import SessionStore
from models.schemas import ValuationOutput
from monitoring.logger import RunLogger


class ValuationAgent(BaseAgent):

    def __init__(self, session: SessionStore, logger: RunLogger) -> None:
        super().__init__("valuation", session, logger)

    def get_system_prompt(self) -> str:
        return """You are a Valuation Analysis Agent.

YOUR JOB: Assess whether a stock is cheap, fairly valued, or expensive.
You receive price data, valuation multiples, analyst consensus, and earnings timing.

RULES:
- valuation_stance: "cheap" | "fair" | "expensive"
  · cheap     = trading significantly below historical/sector norms AND/OR analyst targets
  · fair      = reasonably priced relative to growth and peers
  · expensive = stretched multiples relative to growth prospects
- pct_off_52w_high: context for how much the stock has already corrected
  (e.g. -63% means 63% below its 52-week high — a major reset already occurred)
- upside_to_target_pct: positive = upside to analyst consensus target
- analyst_consensus: pass through exactly as received
- key_insights: 3–5 specific, numbered observations. Examples:
  · "Trading at 20x forward P/E vs 5-year avg of 40x — significant multiple compression"
  · "16 analysts rate BUY with $103 mean target (+89% upside)"
  · "Earnings in 6 days — recommendation carries elevated uncertainty"
  · "Stock down 63% from 52-week high of $150 — substantial reset already priced in"
- earnings_within_14_days: flag true if earnings date is within 14 days from now
- confidence: 0.80–0.95 when data is present. Below 0.60 only if most fields are missing.

BE FACTUAL — report what the data says, not opinions.

CRITICAL: Respond ONLY with valid JSON enclosed in <output></output> tags. Nothing else.

Required output schema:
{
  "ticker":                 string,
  "current_price":          number,
  "week_52_high":           number,
  "week_52_low":            number,
  "pct_off_52w_high":       number,
  "trailing_pe":            number,
  "forward_pe":             number,
  "peg_ratio":              number,
  "price_to_sales":         number,
  "ev_to_ebitda":           number,
  "analyst_count":          integer,
  "analyst_consensus":      string,
  "mean_price_target":      number,
  "upside_to_target_pct":   number,
  "valuation_stance":       "cheap" | "fair" | "expensive",
  "earnings_date":          string or null,
  "earnings_within_14_days": boolean,
  "key_insights":           array of strings (3–5),
  "confidence":             number (0.0–1.0),
  "escalate":               boolean,
  "escalation_reason":      string or null
}"""

    async def run(self, context: dict[str, Any]) -> dict[str, Any]:
        ticker   = context["ticker"]
        raw      = context["raw_financial"]

        user_message = (
            f"Produce a valuation snapshot for {ticker} "
            f"({raw.get('company_name', ticker)}).\n\n"
            f"Price & Momentum:\n"
            f"  Current Price:    ${raw.get('current_price')}\n"
            f"  52-Week High:     ${raw.get('week_52_high')}\n"
            f"  52-Week Low:      ${raw.get('week_52_low')}\n"
            f"  % Off 52W High:   {raw.get('pct_off_52w_high')}%\n\n"
            f"Valuation Multiples:\n"
            f"  Trailing P/E:     {raw.get('pe_ratio')}\n"
            f"  Forward P/E:      {raw.get('forward_pe')}\n"
            f"  PEG Ratio:        {raw.get('peg_ratio')}\n"
            f"  Price/Sales:      {raw.get('price_to_sales')}\n"
            f"  EV/EBITDA:        {raw.get('ev_to_ebitda')}\n\n"
            f"Analyst Consensus:\n"
            f"  Rating:           {raw.get('analyst_recommendation')}\n"
            f"  Rating Mean:      {raw.get('recommendation_mean')} (1.0=Strong Buy, 5.0=Strong Sell)\n"
            f"  # Analysts:       {raw.get('analyst_count')}\n"
            f"  Mean Target:      ${raw.get('mean_price_target')}\n"
            f"  High Target:      ${raw.get('high_price_target')}\n"
            f"  Low Target:       ${raw.get('low_price_target')}\n"
            f"  Upside to Mean:   {raw.get('upside_to_target_pct')}%\n\n"
            f"Earnings:\n"
            f"  Next Earnings Date:        {raw.get('earnings_date') or 'Unknown'}\n"
            f"  Earnings Within 14 Days:   {raw.get('earnings_within_14_days')}\n\n"
            f"Return your valuation snapshot inside <output></output> tags."
        )

        parsed, _ = await self.call_llm(user_message)

        try:
            validated = ValuationOutput(**parsed)
        except ValidationError as exc:
            raise AgentError(f"[valuation] Schema validation failed: {exc}") from exc

        output = validated.model_dump()
        self.session.set("valuation_output", output)
        return output
