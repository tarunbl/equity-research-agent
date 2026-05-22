"""
agents/formatter_agent.py
=========================
Assembles all agent outputs into professional prose, then renders a Rich
terminal report and saves a JSON file.

Rendering is split into focused sub-methods so each section is independently
readable and testable. The LLM generates content; Python renders it — these
two concerns are kept strictly separate.
"""
from __future__ import annotations

import json
import os
from datetime import date
from typing import Any

from pydantic import ValidationError
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agents.base_agent import AgentError, BaseAgent
from memory.session_store import SessionStore
from models.schemas import FormatterOutput
from monitoring.logger import RunLogger

console = Console()


class FormatterAgent(BaseAgent):

    def __init__(self, session: SessionStore, logger: RunLogger) -> None:
        super().__init__("formatter", session, logger)

    def get_system_prompt(self) -> str:
        return """You are a Financial Report Writing Agent.

YOUR JOB: Write concise, professional prose sections for an equity research report.

SECTIONS (2–3 sentences each):
  executive_summary   — the single most important takeaway for an investor
  financial_overview  — key financial trends and their implications
  valuation_snapshot  — price context, multiples, analyst consensus (MUST include:
                        % off 52w high, analyst rating, implied upside/downside)
  sentiment_analysis  — market narrative and why it matters
  risk_assessment     — headline risk picture in plain English
  recommendation_text — signal, conviction, core rationale (MUST reference
                        consensus alignment and analyst target)

RULES:
  - Reference specific numbers: prices, percentages, ratios, targets
  - Do NOT add analysis beyond what the data supports
  - confidence: 0.85+ when inputs are complete; below 0.60 only if sparse data

CRITICAL: Respond ONLY with valid JSON enclosed in <output></output> tags.

Output schema:
{
  "ticker":              string,
  "company_name":        string,
  "report_date":         string (YYYY-MM-DD),
  "executive_summary":   string,
  "financial_overview":  string,
  "valuation_snapshot":  string,
  "sentiment_analysis":  string,
  "risk_assessment":     string,
  "recommendation_text": string,
  "signal":              "BUY" | "HOLD" | "SELL",
  "conviction":          string,
  "time_horizon":        string,
  "risk_level":          string,
  "consensus_alignment": string,
  "upside_downside_note": string,
  "earnings_flag":       string or null,
  "confidence":          number (0.0–1.0),
  "escalate":            boolean,
  "escalation_reason":   string or null
}"""

    async def run(self, context: dict[str, Any]) -> dict[str, Any]:
        financial      = context["financial"]
        news           = context["news"]
        valuation      = context["valuation"]
        risk           = context["risk"]
        recommendation = context["recommendation"]
        ticker         = context["ticker"]

        user_message = self._build_prompt(ticker, context)
        parsed, _    = await self.call_llm(user_message)

        # Inject fields that Python owns — override anything the LLM may have set
        self._inject_fields(parsed, recommendation, risk, valuation)

        try:
            validated = FormatterOutput(**parsed)
        except ValidationError as exc:
            raise AgentError(f"[formatter] Schema validation failed: {exc}") from exc

        output = validated.model_dump()
        self.session.set("report_output", output)

        self._render(output, financial, news, valuation, risk, recommendation)
        self._save_json(output, financial, news, valuation, risk, recommendation)
        return output

    # ── Prompt construction ───────────────────────────────────────────────────

    def _build_prompt(self, ticker: str, ctx: dict[str, Any]) -> str:
        f  = ctx["financial"]
        n  = ctx["news"]
        v  = ctx["valuation"]
        ri = ctx["risk"]
        rc = ctx["recommendation"]
        return (
            f"Write report sections for {ticker} ({ctx.get('company_name', ticker)}).\n\n"
            f"FINANCIALS: Revenue {f['revenue_growth_pct']:+.1f}% | "
            f"Profit margin {f['profit_margin_pct']:.1f}% | "
            f"P/E {f['pe_ratio']:.1f}x | D/E {f['debt_to_equity']:.2f}\n\n"
            f"VALUATION: ${v.get('current_price'):.2f} current | "
            f"{v.get('pct_off_52w_high'):.1f}% off 52w high | "
            f"Stance: {v.get('valuation_stance')} | "
            f"Analyst: {str(v.get('analyst_consensus','')).upper()} "
            f"({v.get('analyst_count')} analysts, target ${v.get('mean_price_target'):.2f}, "
            f"implied {v.get('upside_to_target_pct'):+.1f}%)\n\n"
            f"SENTIMENT: {n['overall_sentiment']} (score {n['sentiment_score']:+.2f}) | "
            f"Themes: {', '.join(n.get('key_themes', []))}\n\n"
            f"RISK: {ri['risk_level'].upper()} ({ri['risk_score']:.1f}/10)\n"
            f"Top risks: {'; '.join(ri.get('top_risk_factors', []))}\n\n"
            f"RECOMMENDATION: {rc['signal'].upper()} | "
            f"Conviction: {rc['conviction'].upper()} | "
            f"Horizon: {rc['time_horizon']}\n"
            f"Consensus: {rc.get('consensus_alignment','').upper()} — "
            f"{rc.get('consensus_note','')}\n"
            f"Upside note: {rc.get('upside_downside_note','')}\n"
            f"Earnings flag: {rc.get('earnings_flag') or 'None'}\n\n"
            f"Today: {date.today().isoformat()}\n\n"
            f"Return all sections inside <output></output> tags."
        )

    # ── Field injection & fallbacks ───────────────────────────────────────────

    def _inject_fields(
        self,
        parsed:         dict[str, Any],
        recommendation: dict[str, Any],
        risk:           dict[str, Any],
        valuation:      dict[str, Any],
    ) -> None:
        """
        Inject fields that Python owns outright (not LLM-generated),
        and provide fallbacks for optional LLM fields if truncated.
        """
        parsed["signal"]              = recommendation["signal"].upper()
        parsed["conviction"]          = recommendation["conviction"].upper()
        parsed["risk_level"]          = risk["risk_level"].upper()
        parsed["consensus_alignment"] = recommendation.get("consensus_alignment", "")
        parsed["upside_downside_note"] = recommendation.get("upside_downside_note", "")
        parsed["earnings_flag"]        = recommendation.get("earnings_flag")

        if not parsed.get("valuation_snapshot"):
            parsed["valuation_snapshot"] = self._build_valuation_snapshot(valuation)

    def _build_valuation_snapshot(self, v: dict[str, Any]) -> str:
        """Python fallback for valuation_snapshot if LLM omitted it."""
        upside = v.get("upside_to_target_pct", 0)
        upside_str = f"+{upside:.1f}%" if upside >= 0 else f"{upside:.1f}%"
        pct_off = abs(v.get("pct_off_52w_high", 0))
        return (
            f"Trading at ${v.get('current_price', 0):.2f}, {pct_off:.0f}% below the "
            f"52-week high of ${v.get('week_52_high', 0):.2f}. "
            f"Forward P/E of {v.get('forward_pe', 0):.1f}x; "
            f"valuation stance: {v.get('valuation_stance', 'N/A')}. "
            f"Analyst consensus: {str(v.get('analyst_consensus', 'N/A')).upper()} "
            f"({v.get('analyst_count', 0)} analysts, mean target "
            f"${v.get('mean_price_target', 0):.2f}, implied upside {upside_str})."
        )

    # ── Terminal rendering ────────────────────────────────────────────────────

    def _render(
        self, report, financial, news, valuation, risk, recommendation
    ) -> None:
        self._render_header(report, risk, valuation, recommendation)
        self._render_metrics(financial, valuation)
        self._render_narrative(report)
        self._render_recommendation(report, recommendation)

    def _render_header(self, report, risk, valuation, recommendation) -> None:
        signal    = report["signal"]
        risk_lvl  = report["risk_level"]
        sig_color = {"BUY": "bold green", "HOLD": "bold yellow", "SELL": "bold red"}.get(
            signal, "bold white"
        )
        risk_color = {"LOW": "green", "MEDIUM": "yellow", "HIGH": "red"}.get(
            risk_lvl, "white"
        )
        align       = report["consensus_alignment"].upper()
        align_color = "green" if align == "ALIGNED" else "yellow"

        console.rule(f"[bold cyan]Equity Research Report — {report['ticker']}[/bold cyan]")
        console.print()

        lines = [
            f"  [bold]Company:[/bold]      {report['company_name']}  ({report['ticker']})",
            f"  [bold]Date:[/bold]         {report['report_date']}",
            f"  [bold]Signal:[/bold]       [{sig_color}]{signal}[/{sig_color}]  "
            f"(Conviction: {report['conviction']})",
            f"  [bold]Time Horizon:[/bold] {report['time_horizon']}",
            f"  [bold]Risk Level:[/bold]   [{risk_color}]{risk_lvl}[/{risk_color}]  "
            f"(Score: {risk['risk_score']:.1f}/10)",
            f"  [bold]Consensus:[/bold]    [{align_color}]{align}[/{align_color}]  "
            f"— {report['upside_downside_note']}",
        ]
        if report.get("earnings_flag"):
            lines.append(
                f"  [bold yellow]⚠  Earnings:[/bold yellow]   {report['earnings_flag']}"
            )

        console.print(Panel(
            "\n".join(lines),
            border_style=sig_color.replace("bold ", ""),
            padding=(1, 2),
        ))
        console.print()

    def _render_metrics(self, financial, valuation) -> None:
        fin = Table(
            title="Financial Metrics", box=box.SIMPLE_HEAD,
            show_header=True, header_style="bold cyan",
        )
        fin.add_column("Metric", style="bold", width=20)
        fin.add_column("Value",  justify="right", width=12)
        for name, val in [
            ("Revenue Growth",    f"{financial['revenue_growth_pct']:+.1f}%"),
            ("Gross Margin",      f"{financial['gross_margin_pct']:.1f}%"),
            ("Profit Margin",     f"{financial['profit_margin_pct']:+.1f}%"),
            ("Trailing P/E",      f"{financial['pe_ratio']:.1f}x"),
            ("Forward P/E",       f"{valuation.get('forward_pe', 0):.1f}x"),
            ("PEG Ratio",         f"{valuation.get('peg_ratio', 0):.2f}"),
            ("Debt / Equity",     f"{financial['debt_to_equity']:.2f}"),
            ("Current Ratio",     f"{financial['current_ratio']:.2f}"),
            ("Return on Equity",  f"{financial['return_on_equity_pct']:.1f}%"),
        ]:
            fin.add_row(name, val)

        upside = valuation.get("upside_to_target_pct", 0)
        upside_disp = (
            Text(f"+{upside:.1f}%", style="green") if upside >= 0
            else Text(f"{upside:.1f}%", style="red")
        )
        val_t = Table(
            title="Valuation & Consensus", box=box.SIMPLE_HEAD,
            show_header=True, header_style="bold cyan",
        )
        val_t.add_column("Metric", style="bold", width=18)
        val_t.add_column("Value",  justify="right", width=14)
        val_t.add_row("Current Price",  f"${valuation.get('current_price', 0):.2f}")
        val_t.add_row("52W High",       f"${valuation.get('week_52_high', 0):.2f}")
        val_t.add_row("% Off 52W High", f"{valuation.get('pct_off_52w_high', 0):.1f}%")
        val_t.add_row("Analyst Rating", str(valuation.get("analyst_consensus", "N/A")).upper())
        val_t.add_row("# Analysts",     str(valuation.get("analyst_count", 0)))
        val_t.add_row("Mean Target",    f"${valuation.get('mean_price_target', 0):.2f}")
        val_t.add_row("Implied Upside", upside_disp)

        console.print(Columns([fin, val_t], padding=(0, 4)))
        console.print()

    def _render_narrative(self, report) -> None:
        for title, key in [
            ("Executive Summary",  "executive_summary"),
            ("Valuation Snapshot", "valuation_snapshot"),
            ("Financial Overview", "financial_overview"),
            ("Sentiment Analysis", "sentiment_analysis"),
            ("Risk Assessment",    "risk_assessment"),
        ]:
            console.print(Panel(
                report[key],
                title=f"[bold]{title}[/bold]",
                border_style="dim cyan",
                padding=(1, 2),
            ))
            console.print()

    def _render_recommendation(self, report, recommendation) -> None:
        signal    = report["signal"]
        sig_color = {"BUY": "bold green", "HOLD": "bold yellow", "SELL": "bold red"}.get(
            signal, "bold white"
        )
        align       = report["consensus_alignment"].upper()
        align_color = "green" if align == "ALIGNED" else "yellow"

        content = (
            f"[{sig_color}]● {signal}[/{sig_color}]  |  "
            f"Conviction: {report['conviction']}  |  "
            f"Horizon: {report['time_horizon']}\n\n"
            f"{report['recommendation_text']}\n\n"
            f"[bold]Consensus:[/bold] [{align_color}]{align}[/{align_color}]"
            f" — {recommendation.get('consensus_note', '')}\n"
            f"[bold]Catalysts:[/bold]  {', '.join(recommendation.get('key_catalysts', []))}\n"
            f"[bold]Key Risks:[/bold]  {', '.join(recommendation.get('key_risks', []))}"
        )
        console.print(Panel(
            content,
            title="[bold]Recommendation[/bold]",
            border_style=sig_color.replace("bold ", ""),
            padding=(1, 2),
        ))
        console.print()

        # Score breakdown table
        factors = recommendation.get("score_factors", [])
        if factors:
            self._render_score_breakdown(factors, recommendation)

    def _render_score_breakdown(self, factors: list, recommendation: dict) -> None:
        """Render the quantitative scoring breakdown — shows exactly why the signal was generated."""
        table = Table(
            title="[bold]Quantitative Score Breakdown[/bold]",
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
        )
        table.add_column("Factor",       style="bold", width=28)
        table.add_column("Value",                      width=20)
        table.add_column("Score",        justify="right", width=7)
        table.add_column("Note",         style="dim",  width=40)

        total = 0.0
        for f in factors:
            contrib = float(f.get("contribution", 0))
            total  += contrib
            if contrib > 0:
                score_disp = Text(f"+{contrib:.1f}", style="green")
            elif contrib < 0:
                score_disp = Text(f"{contrib:.1f}", style="red")
            else:
                score_disp = Text("  0.0", style="dim")

            table.add_row(
                f.get("factor", ""),
                f.get("value", ""),
                score_disp,
                f.get("note", ""),
            )

        table.add_section()
        signal = recommendation.get("signal", "").upper()
        sig_color = {"BUY": "green", "HOLD": "yellow", "SELL": "red"}.get(signal, "white")
        table.add_row(
            "[bold]TOTAL SCORE[/bold]",
            "",
            Text(f"{total:+.1f}", style=f"bold {sig_color}"),
            f"→  [{sig_color}]{signal}[/{sig_color}]  ({recommendation.get('conviction', '').upper()} conviction)",
        )

        console.print(table)
        console.print()

    # ── File output ───────────────────────────────────────────────────────────

    def _save_json(self, report, financial, news, valuation, risk, recommendation) -> None:
        os.makedirs("output", exist_ok=True)
        path = f"output/{report['ticker']}_{report['report_date']}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "report":         report,
                    "financial":      financial,
                    "news":           news,
                    "valuation":      valuation,
                    "risk":           risk,
                    "recommendation": recommendation,
                },
                f,
                indent=2,
                default=str,
            )
        console.print(f"[dim]📄 Report saved → {path}[/dim]\n")
