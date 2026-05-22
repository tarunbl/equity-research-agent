"""
monitoring/logger.py
====================
Tracks token usage, latency, escalation, and cost for every agent call.
Prints a Rich-formatted summary table at the end of each run.

Design note: one AgentMetrics entry is logged per agent call. The base_agent
accumulates tokens from both the default and escalated model call into a
single entry — so the table always shows exactly one row per agent.
"""
from __future__ import annotations

import time
from typing import List

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from config import COST_PER_MILLION
from models.schemas import AgentMetrics

console = Console()


class RunLogger:

    def __init__(self, ticker: str, run_id: str) -> None:
        self.ticker     = ticker
        self.run_id     = run_id
        self._agents:   List[AgentMetrics] = []
        self._run_start = time.perf_counter()

    def log(self, metrics: AgentMetrics) -> None:
        """Record metrics for one completed agent call."""
        metrics.estimated_cost_usd = self._estimate_cost(metrics)
        self._agents.append(metrics)

    def print_summary(self) -> None:
        """Print the full run summary to the terminal."""
        if not self._agents:
            return
        self._render_table()
        self._render_footer()

    # ── Cost estimation ───────────────────────────────────────────────────────

    def _estimate_cost(self, m: AgentMetrics) -> float:
        rates = COST_PER_MILLION.get(m.model_used, {})
        if not rates:
            return 0.0
        return round(
            (m.input_tokens       / 1_000_000) * rates["input"]       +
            (m.output_tokens      / 1_000_000) * rates["output"]      +
            (m.cache_write_tokens / 1_000_000) * rates["cache_write"] +
            (m.cache_read_tokens  / 1_000_000) * rates["cache_read"],
            6,
        )

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _render_table(self) -> None:
        table = Table(
            title=f"[bold cyan]Run Summary — {self.ticker}[/bold cyan]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold white on dark_blue",
            border_style="cyan",
            padding=(0, 1),
        )
        table.add_column("Agent",      style="bold", width=16)
        table.add_column("Model",                    width=22)
        table.add_column("In",         justify="right", width=6)
        table.add_column("Out",        justify="right", width=6)
        table.add_column("Cached",     justify="right", width=7)
        table.add_column("Time",       justify="right", width=8)
        table.add_column("Escalated",  justify="center", width=11)
        table.add_column("Cost",       justify="right", width=10)

        for a in self._agents:
            model_disp = (
                a.model_used
                .replace("claude-", "")
                .replace("-20251001", "")
            )
            esc = (
                Text("⬆ YES", style="bold yellow") if a.escalated
                else Text("✓  No", style="dim green")
            )
            table.add_row(
                a.agent_name,
                model_disp,
                str(a.input_tokens),
                str(a.output_tokens),
                str(a.cache_read_tokens),
                f"{a.latency_ms:.0f}ms",
                esc,
                f"${a.estimated_cost_usd:.5f}",
            )

        console.print()
        console.print(table)

    def _render_footer(self) -> None:
        total_wall     = (time.perf_counter() - self._run_start) * 1000
        total_in       = sum(a.input_tokens       for a in self._agents)
        total_out      = sum(a.output_tokens      for a in self._agents)
        total_cached   = sum(a.cache_read_tokens  for a in self._agents)
        total_cost     = sum(a.estimated_cost_usd for a in self._agents)
        escalated      = [a for a in self._agents if a.escalated]

        lines = [
            f"[bold]Total Tokens[/bold]    {total_in:,} in / {total_out:,} out",
            f"[bold]Cache Savings[/bold]   {total_cached:,} tokens read from cache",
            f"[bold]Wall Time[/bold]       {total_wall:.0f}ms",
            f"[bold]Est. Cost[/bold]       [green]${total_cost:.5f}[/green]",
            f"[bold]Escalations[/bold]     {len(escalated)} of {len(self._agents)} agents",
        ]
        for a in escalated:
            layer = f" [dim]({a.escalation_layer})[/dim]" if a.escalation_layer else ""
            lines.append(
                f"  [yellow]↳ {a.agent_name}[/yellow]: {a.escalation_reason}{layer}"
            )

        console.print(Panel(
            "\n".join(lines),
            title="[bold]Pipeline Complete[/bold]",
            border_style="cyan",
            padding=(1, 2),
        ))
        console.print()
