"""
main.py
=======
Pure Python pipeline — zero LLM calls here.

Phase 3 pipeline
----------------
    Stage 1 (fully parallel):
      FinancialAgent   — yfinance financial data
      NewsAgent        — Yahoo Finance + Finnhub news (aggregated)
      _fetch_risk_intel — SEC 10-K + 8-K + Finnhub insider transactions

    Stage 2: Valuation (Haiku)
    Stage 3: Risk       (Sonnet → Opus) — now uses Tier 3 intelligence
    Stage 4: Recommendation (Sonnet → Opus)
    Stage 5: Formatter  (Haiku)

Risk intelligence runs in parallel with Stage 1 LLM agents — adds zero
wall-clock time while providing substantially richer context to Stage 3.
"""
from __future__ import annotations

import asyncio
import logging
import sys
import time
import uuid
from typing import Any, NoReturn

from dotenv import load_dotenv
from rich.console import Console

from agents.base_agent import AgentError
from agents.scorer import compute_signal
from agents.financial_agent import FinancialAgent
from agents.formatter_agent import FormatterAgent
from agents.news_agent import NewsAgent
from agents.recommendation_agent import RecommendationAgent
from agents.risk_agent import RiskAgent
from agents.valuation_agent import ValuationAgent
from config import YFINANCE_TIMEOUT_SECS
from memory.session_store import SessionStore
from monitoring.logger import RunLogger
from utils.context_builder import (
    build_formatter_context,
    build_recommendation_context,
    build_risk_context,
    build_valuation_context,
)

load_dotenv()
console = Console()
logging.basicConfig(level=logging.WARNING)  # Suppress debug noise


# ── Risk intelligence fetch (parallel with Stage 1 agents) ───────────────────

async def _fetch_risk_intelligence(
    ticker: str, session: SessionStore
) -> dict[str, Any]:
    """
    Fetch SEC EDGAR and Finnhub intelligence concurrently.
    Runs in parallel with FinancialAgent and NewsAgent in Stage 1.
    Returns safely even if both sources are unavailable.
    """
    from tools.sec_edgar import get_sec_intelligence
    from tools.finnhub  import get_insider_intelligence

    loop = asyncio.get_event_loop()

    try:
        sec_data, insider_data = await asyncio.gather(
            asyncio.wait_for(
                loop.run_in_executor(None, get_sec_intelligence, ticker),
                timeout=YFINANCE_TIMEOUT_SECS,
            ),
            asyncio.wait_for(
                loop.run_in_executor(None, get_insider_intelligence, ticker),
                timeout=YFINANCE_TIMEOUT_SECS,
            ),
            return_exceptions=True,   # Don't let one failure cancel the other
        )
    except Exception:
        sec_data    = None
        insider_data = None

    # Handle per-result exceptions (from return_exceptions=True)
    if isinstance(sec_data, Exception):
        sec_data = None
    if isinstance(insider_data, Exception):
        insider_data = None

    sec    = sec_data    or {}
    insider = insider_data or {}

    sources = []
    if sec.get("risk_factors") or sec.get("recent_material_events"):
        sources.append("SEC EDGAR")
    if insider.get("signal") not in (None, "unavailable"):
        sources.append("Finnhub")

    intel = {
        "sec_risk_factors":           sec.get("risk_factors",           []),
        "recent_material_events":     sec.get("recent_material_events", []),
        "latest_10k_date":            sec.get("latest_10k_date"),
        "insider_signal":             insider.get("signal",     "unavailable"),
        "insider_summary":            insider.get("summary",    ""),
        "insider_notable":            insider.get("notable",    []),
        "sources":                    sources,
    }

    session.set("risk_intelligence", intel)
    return intel


# ── Main pipeline ─────────────────────────────────────────────────────────────

async def run_pipeline(ticker: str) -> None:
    run_id  = str(uuid.uuid4())[:8]
    session = SessionStore(run_id=run_id)
    logger  = RunLogger(ticker=ticker, run_id=run_id)

    console.print(
        f"\n[bold cyan]▶ Equity Research Pipeline[/bold cyan]  "
        f"[dim]ticker={ticker}  run={run_id}[/dim]\n"
    )

    # ── Stage 1: All data collection in parallel ──────────────────────────────
    try:
        t1 = time.perf_counter()
        with console.status(
            "[dim]Stage 1/5 — fetching financials, news & intelligence...[/dim]",
            spinner="dots",
        ):
            financial_output, news_output, risk_intel = await asyncio.gather(
                FinancialAgent(session, logger).run({"ticker": ticker}),
                NewsAgent(session, logger).run({"ticker": ticker}),
                _fetch_risk_intelligence(ticker, session),
            )

        sources = risk_intel.get("sources", [])
        source_note = f"  [dim]intelligence: {', '.join(sources) or 'Yahoo Finance only'}[/dim]"
        console.print(
            f"[green]✓[/green] Stage 1 complete  "
            f"[dim]({time.perf_counter()-t1:.1f}s)[/dim]{source_note}"
        )
    except AgentError as exc:
        raise AgentError(f"Stage 1 (data collection) failed — {exc}") from exc

    # ── Stage 2: Valuation snapshot ───────────────────────────────────────────
    try:
        t2 = time.perf_counter()
        with console.status(
            "[dim]Stage 2/5 — computing valuation & consensus (Haiku)...[/dim]",
            spinner="dots",
        ):
            valuation_output = await ValuationAgent(session, logger).run(
                build_valuation_context(
                    ticker=ticker,
                    raw_financial=session.get("financial_raw"),
                )
            )
        console.print(
            f"[green]✓[/green] Stage 2 complete  "
            f"[dim]({time.perf_counter()-t2:.1f}s)[/dim]"
        )
    except AgentError as exc:
        raise AgentError(f"Stage 2 (valuation) failed — {exc}") from exc

    # ── Stage 3: Risk analysis ────────────────────────────────────────────────
    try:
        t3 = time.perf_counter()
        with console.status(
            "[dim]Stage 3/5 — running risk analysis (Sonnet)...[/dim]",
            spinner="dots",
        ):
            risk_output = await RiskAgent(session, logger).run(
                build_risk_context(
                    financial         = financial_output,
                    news              = news_output,
                    valuation         = valuation_output,
                    risk_intelligence = risk_intel,
                )
            )
        console.print(
            f"[green]✓[/green] Stage 3 complete  "
            f"[dim]({time.perf_counter()-t3:.1f}s)[/dim]"
        )
    except AgentError as exc:
        raise AgentError(f"Stage 3 (risk analysis) failed — {exc}") from exc

    # ── Stage 4: Score signal + generate narrative ────────────────────────────
    try:
        t4 = time.perf_counter()

        # 4a: Deterministic scoring (Python, no LLM, instant)
        rec_ctx = build_recommendation_context(
            financial     = financial_output,
            risk          = risk_output,
            valuation     = valuation_output,
            raw_financial = session.get("financial_raw"),
        )
        scored_signal = compute_signal(rec_ctx)
        console.print(
            f"[cyan]  Score: {scored_signal.score:+.1f}  →  "
            f"{scored_signal.signal.upper()} "
            f"({scored_signal.conviction.upper()}, "
            f"{scored_signal.consensus_alignment})[/cyan]"
        )

        # 4b: LLM narrative explaining the signal
        with console.status(
            "[dim]Stage 4/5 — writing recommendation narrative (Sonnet)...[/dim]",
            spinner="dots",
        ):
            rec_ctx["scored_signal"] = scored_signal
            recommendation_output = await RecommendationAgent(session, logger).run(rec_ctx)

        console.print(
            f"[green]✓[/green] Stage 4 complete  "
            f"[dim]({time.perf_counter()-t4:.1f}s)[/dim]\n"
        )
    except AgentError as exc:
        raise AgentError(f"Stage 4 (recommendation) failed — {exc}") from exc

    # ── Stage 5: Format & render ──────────────────────────────────────────────
    try:
        with console.status(
            "[dim]Stage 5/5 — assembling report (Haiku)...[/dim]",
            spinner="dots",
        ):
            await FormatterAgent(session, logger).run(
                build_formatter_context(
                    financial_output, news_output, valuation_output,
                    risk_output, recommendation_output,
                )
            )
    except AgentError as exc:
        raise AgentError(f"Stage 5 (formatting) failed — {exc}") from exc

    logger.print_summary()


# ── Entry point ───────────────────────────────────────────────────────────────

def validate_ticker(ticker: str) -> None:
    clean = ticker.replace(".", "").replace("-", "")
    if not clean.isalnum() or not (1 <= len(clean) <= 10):
        console.print(
            f"\n[bold red]Invalid ticker:[/bold red] '{ticker}'\n"
            f"[dim]Examples: AAPL  TSLA  ELF  MSFT  BRK-B[/dim]\n"
        )
        sys.exit(1)


def main() -> NoReturn:
    if len(sys.argv) != 2:
        console.print(
            "\n[bold]Usage:[/bold]  python main.py <TICKER>\n"
            "[dim]Examples: AAPL  TSLA  ELF  MSFT  NVDA  BRK-B[/dim]\n"
        )
        sys.exit(1)

    ticker = sys.argv[1].upper()
    validate_ticker(ticker)

    try:
        asyncio.run(run_pipeline(ticker))
    except AgentError as exc:
        console.print(f"\n[bold red]Error:[/bold red] {exc}\n")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
