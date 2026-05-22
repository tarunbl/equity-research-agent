"""
agents/risk_agent.py
====================
Multi-axis risk analysis — now uses three data tiers:

  Tier 1 (always): Financial metrics + news sentiment
  Tier 2 (always): Valuation context (price reset, earnings proximity)
  Tier 3 (when configured):
    - SEC 10-K risk factors   — company's own legal disclosures
    - SEC 8-K material events — significant events last 30 days
    - Insider transactions    — executives buying or selling their own stock

Model routing
-------------
  Default  : claude-sonnet-4-6
  Escalates: claude-opus-4-6  — D/E > 3.0, conflicting signals, or confidence < 0.40
"""
from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from agents.base_agent import AgentError, BaseAgent
from memory.session_store import SessionStore
from models.schemas import RiskOutput
from monitoring.logger import RunLogger


class RiskAgent(BaseAgent):

    def __init__(self, session: SessionStore, logger: RunLogger) -> None:
        super().__init__("risk", session, logger)

    def get_system_prompt(self) -> str:
        return """You are a Senior Financial Risk Analyst.

YOUR JOB: Perform a comprehensive risk assessment across four axes:
  1. Financial risk    — leverage, liquidity, profitability, anomalies
  2. Sentiment risk    — news narrative, key negative themes
  3. Sector risk       — industry headwinds
  4. Intelligence risk — SEC disclosures, material events, insider behaviour

TIER 3 DATA RULES (when provided):
  - SEC Risk Factors: these are legal disclosures — take them seriously.
    A company disclosing "we may be unable to compete" is a real risk.
  - Recent 8-K Events: a CEO departure or earnings miss in the last 30 days
    significantly changes the risk picture — weight it heavily.
  - Insider Transactions:
    · "bearish" signal (net selling) → adds 0.5–1.5 to risk_score
    · "bullish" signal (net buying)  → reduces risk_score 0.5–1.0
    · "neutral" or "unavailable"     → no adjustment

VALUATION CONTEXT:
  - pct_off_52w_high: stock down 50%+ has priced in significant risk already
  - earnings_within_14_days: adds near-term uncertainty (+0.5 to score)

SCORING:
  - risk_level: "low" (0–3), "medium" (4–6), "high" (7–10)
  - risk_score: precise 0.0–10.0
  - top_risk_factors: 3–5 specific, evidence-backed factors
  - mitigating_factors: 2–4 genuine positives
  - confidence: 0.80–0.95 when data is clear; below 0.50 only if genuinely ambiguous
  - Each summary: 1–2 direct sentences. No verbosity.

CRITICAL: Respond ONLY with valid JSON enclosed in <output></output> tags.

Output schema:
{
  "ticker":                 string,
  "risk_level":             "low" | "medium" | "high",
  "risk_score":             number (0.0–10.0),
  "conflicting_signals":    boolean,
  "financial_risk_summary": string,
  "sentiment_risk_summary": string,
  "sector_risk_summary":    string,
  "top_risk_factors":       array of strings (3–5),
  "mitigating_factors":     array of strings (2–4),
  "confidence":             number (0.0–1.0),
  "escalate":               boolean,
  "escalation_reason":      string or null
}"""

    async def run(self, context: dict[str, Any]) -> dict[str, Any]:
        ticker = context["ticker"]

        # Build the intelligence section only if data is available
        intel_section = self._build_intelligence_section(context)

        user_message = (
            f"Perform a risk assessment for {ticker} "
            f"({context.get('company_name', ticker)}).\n\n"

            f"━━ FINANCIAL METRICS ━━\n"
            f"  Revenue Growth:  {context.get('revenue_growth_pct')}%\n"
            f"  Gross Margin:    {context.get('gross_margin_pct')}%\n"
            f"  Profit Margin:   {context.get('profit_margin_pct')}%\n"
            f"  P/E Ratio:       {context.get('pe_ratio')}\n"
            f"  Debt/Equity:     {context.get('debt_to_equity')}\n"
            f"  Current Ratio:   {context.get('current_ratio')}\n"
            f"  ROE:             {context.get('return_on_equity_pct')}%\n"
            f"  Anomalies:       {context.get('financial_anomalies', [])}\n\n"

            f"━━ NEWS SENTIMENT ━━\n"
            f"  Overall:     {context.get('overall_sentiment')} "
            f"(score: {context.get('sentiment_score')})\n"
            f"  Negative:    {context.get('negative_pct')}%\n"
            f"  Key Themes:  {context.get('key_themes', [])}\n"
            f"  Headlines:   {json.dumps(context.get('top_headlines', []), indent=2)}\n\n"

            f"━━ VALUATION CONTEXT ━━\n"
            f"  Price vs 52W High: {context.get('pct_off_52w_high')}%\n"
            f"  Valuation Stance:  {context.get('valuation_stance')}\n"
            f"  Earnings <14d:     {context.get('earnings_within_14_days')}\n"
            f"  Conflicting Signals: {context.get('conflicting_signals')}\n\n"

            f"{intel_section}"
            f"Return your risk assessment inside <output></output> tags."
        )

        rule_context = {
            "debt_to_equity":     context.get("debt_to_equity", 0.0),
            "conflicting_signals": context.get("conflicting_signals", False),
        }

        parsed, _ = await self.call_llm(user_message, context_for_rules=rule_context)

        try:
            validated = RiskOutput(**parsed)
        except ValidationError as exc:
            raise AgentError(f"[risk] Schema validation failed: {exc}") from exc

        output = validated.model_dump()
        self.session.set("risk_output", output)
        return output

    def _build_intelligence_section(self, context: dict[str, Any]) -> str:
        """
        Build the SEC + insider intelligence section of the prompt.
        Returns empty string if no intelligence data is available,
        so the agent degrades gracefully when Finnhub/SEC are not configured.
        """
        sec_risks  = context.get("sec_risk_factors",       [])
        events     = context.get("recent_material_events", [])
        insider_sig = context.get("insider_signal",        "unavailable")
        insider_sum = context.get("insider_summary",       "")
        insider_not = context.get("insider_notable",       [])
        sources     = context.get("intelligence_sources",  [])

        if not sec_risks and not events and insider_sig == "unavailable":
            return ""   # No Tier 3 data available — skip entirely

        lines = ["━━ EXTERNAL INTELLIGENCE ━━\n"]
        lines.append(f"  Sources: {', '.join(sources) if sources else 'SEC EDGAR'}\n")

        if sec_risks:
            lines.append(f"\n  SEC 10-K Risk Factors (legal disclosures):\n")
            for risk in sec_risks:
                lines.append(f"    · {risk}\n")

        if events:
            lines.append(f"\n  Recent Material Events (8-K filings, last 30 days):\n")
            for ev in events:
                lines.append(f"    · [{ev.get('date', '?')}] {ev.get('event', '?')}\n")

        if insider_sig != "unavailable":
            lines.append(f"\n  Insider Transactions (last 90 days):\n")
            lines.append(f"    Signal: {insider_sig.upper()}\n")
            lines.append(f"    Summary: {insider_sum}\n")
            if insider_not:
                lines.append(f"    Notable:\n")
                for n in insider_not:
                    lines.append(f"      · {n}\n")

        lines.append("\n")
        return "".join(lines)
