"""
agents/recommendation_agent.py
===============================
Recommendation NARRATIVE agent.

This agent no longer decides the BUY/HOLD/SELL signal.
That is done deterministically by agents/scorer.py.

This agent receives the pre-computed signal and writes the professional
analysis explaining it — rationale, bull/bear cases, catalysts, risks,
and consensus reconciliation.

Separation of concerns
----------------------
  scorer.py             : WHAT  (deterministic, reproducible)
  recommendation_agent  : WHY   (LLM narrative, professional tone)

Model routing
-------------
  Default  : claude-sonnet-4-6
  Escalates: claude-opus-4-6 when risk_level == "high" (pre-LLM rule)
             or confidence < 0.40 (post-LLM)
"""
from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from agents.base_agent import AgentError, BaseAgent
from agents.scorer import ScoredSignal, format_scorecard
from memory.session_store import SessionStore
from models.schemas import RecommendationOutput
from monitoring.logger import RunLogger


class RecommendationAgent(BaseAgent):

    def __init__(self, session: SessionStore, logger: RunLogger) -> None:
        super().__init__("recommendation", session, logger)

    def get_system_prompt(self) -> str:
        return """You are a Senior Equity Research Analyst writing the narrative
section of an investment recommendation report.

THE SIGNAL HAS ALREADY BEEN DETERMINED by a quantitative scoring model.
Your job is to write clear, professional analysis EXPLAINING the signal —
not to second-guess or override it.

YOUR TASKS:
  rationale:       2–3 sentences explaining the core investment thesis.
                   Reference the key scoring factors explicitly.
  bull_case:       1 sentence on the best-case scenario.
  bear_case:       1 sentence on the worst-case scenario.
  key_catalysts:   2–4 specific events that could drive outperformance.
  key_risks:       2–4 specific risks that could undermine the thesis.
  consensus_note:  1–2 sentences on the analyst consensus alignment.
  upside_downside_note: State both mean AND high price target with implied
                   upside/downside. e.g. "Mean target $305 (+3.6%); bull
                   case target $400 (+35.7%) from analyst high estimates."
  earnings_flag:   If earnings within 14 days, state the date and that it
                   adds near-term uncertainty. Otherwise null.

RULES:
  - Do NOT change the signal or conviction — those come from the scorer.
  - Reference specific numbers from the scoring factors.
  - Be direct and professional — no generic filler phrases.
  - confidence: your confidence in the QUALITY of your narrative (0.80–0.95).
  - BE CONCISE — rationale/cases are 1–2 sentences each, not paragraphs.

CRITICAL: Respond ONLY with valid JSON enclosed in <output></output> tags.

Output schema:
{
  "ticker":               string,
  "signal":               string,
  "conviction":           string,
  "time_horizon":         string,
  "rationale":            string,
  "bull_case":            string,
  "bear_case":            string,
  "key_catalysts":        array of strings (2–4),
  "key_risks":            array of strings (2–4),
  "consensus_alignment":  string,
  "consensus_note":       string,
  "upside_downside_note": string,
  "earnings_flag":        string or null,
  "confidence":           number (0.0–1.0),
  "escalate":             boolean,
  "escalation_reason":    string or null
}"""

    async def run(self, context: dict[str, Any]) -> dict[str, Any]:
        ticker       = context["ticker"]
        scored: ScoredSignal = context["scored_signal"]

        user_message = (
            f"Write the narrative analysis for {ticker} "
            f"({context.get('company_name', ticker)}).\n\n"

            f"━━ PRE-COMPUTED SIGNAL (DO NOT CHANGE) ━━\n"
            f"{format_scorecard(scored)}\n\n"

            f"━━ CONTEXT FOR YOUR NARRATIVE ━━\n"
            f"Current Price:        ${context.get('current_price')}\n"
            f"52W High:             ${context.get('week_52_high')}\n"
            f"% Off 52W High:       {context.get('pct_off_52w_high')}%\n"
            f"Forward P/E:          {context.get('forward_pe')}\n"
            f"Analyst Consensus:    {str(context.get('analyst_consensus','')).upper()}\n"
            f"Mean Target:          ${context.get('mean_price_target')} "
            f"({context.get('upside_to_target_pct'):+.1f}%)\n"
            f"High Target:          ${context.get('high_price_target')}\n"
            f"Risk Level:           {str(context.get('risk_level','')).upper()} "
            f"({context.get('risk_score')}/10)\n"
            f"Financial Risk:       {context.get('financial_risk_summary')}\n"
            f"Sentiment Risk:       {context.get('sentiment_risk_summary')}\n"
            f"Top Risk Factors:     {context.get('top_risk_factors', [])}\n"
            f"Mitigating Factors:   {context.get('mitigating_factors', [])}\n"
            f"Revenue Growth:       {context.get('revenue_growth_pct')}%\n"
            f"Profit Margin:        {context.get('profit_margin_pct')}%\n"
            f"Earnings Date:        {context.get('earnings_date') or 'Unknown'}\n"
            f"Earnings <14d:        {context.get('earnings_within_14_days')}\n\n"

            f"Consensus alignment is: {scored.consensus_alignment.upper()}\n"
            f"Time horizon should reflect the conviction level "
            f"({scored.conviction}) and risk profile.\n\n"

            f"Return your narrative analysis inside <output></output> tags.\n"
            f"Use {scored.signal.upper()} as the signal and "
            f"{scored.conviction.upper()} as the conviction in your output."
        )

        rule_context = {"risk_level": context.get("risk_level", "low")}
        parsed, _    = await self.call_llm(user_message, context_for_rules=rule_context)

        # Always inject scorer values — LLM must not override them
        parsed["signal"]              = scored.signal
        parsed["conviction"]          = scored.conviction
        parsed["consensus_alignment"] = scored.consensus_alignment
        parsed.setdefault("ticker",   ticker)

        try:
            validated = RecommendationOutput(**parsed)
        except ValidationError as exc:
            raise AgentError(
                f"[recommendation] Schema validation failed: {exc}"
            ) from exc

        output = validated.model_dump()
        output["score_factors"] = scored.factors   # Carry through for formatter
        self.session.set("recommendation_output", output)
        return output
