"""
agents/scorer.py
================
Deterministic investment signal scorer.

Produces BUY / HOLD / SELL with explicit weighted factors.
Same inputs always produce the same output — no LLM variance.

Why this exists
---------------
LLMs are excellent at synthesis and narrative generation. They are NOT
reliable for consistently applying specific quantitative rules — the same
inputs can produce different signals across runs depending on phrasing,
token sampling, and reasoning paths.

This module separates concerns:
  - scorer.py    : WHAT the signal is  (deterministic Python)
  - recommendation_agent.py : WHY it makes sense (LLM narrative)

Scoring factors
---------------
  1. Analyst rating mean     (primary driver)
  2. Bull case upside        (high price target)
  3. Mean target upside
  4. Risk level
  5. Price reset / momentum
  6. Insider signal
  7. Revenue growth
  8. Profit margin
  9. Earnings proximity      (modifier)
  10. Conflicting signals    (modifier)

Signal thresholds
-----------------
  Score ≥ 4.5  →  BUY  high
  Score ≥ 2.5  →  BUY  medium
  Score ≥ 1.0  →  BUY  low
  Score ≥ -1.0 →  HOLD medium
  Score ≥ -2.5 →  HOLD low
  Score < -2.5 →  SELL
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScoredSignal:
    """Complete output of the scoring engine."""
    signal:              str          # "buy" | "hold" | "sell"
    conviction:          str          # "high" | "medium" | "low"
    score:               float        # total weighted score
    max_possible:        float        # positive-factors-only ceiling
    consensus_alignment: str          # "aligned" | "divergent"
    factors:             list[dict] = field(default_factory=list)
    # Each factor: {factor, value, contribution, note}


def compute_signal(context: dict[str, Any]) -> ScoredSignal:
    """
    Compute a deterministic investment signal from pipeline context.

    Uses quantitative inputs from financial, valuation, risk, and
    intelligence stages. No LLM involved.
    """
    factors: list[dict] = []
    total = 0.0

    def add(factor: str, value: Any, contrib: float, note: str) -> None:
        nonlocal total
        total += contrib
        factors.append({
            "factor":       factor,
            "value":        str(value),
            "contribution": round(contrib, 2),
            "note":         note,
        })

    # ── 1. Analyst rating consensus ───────────────────────────────────────────
    # recommendation_mean: 1.0 = Strong Buy, 5.0 = Strong Sell
    rec_mean = float(context.get("recommendation_mean") or 0.0)
    if rec_mean == 0.0:
        add("Analyst Rating",    "N/A",              0.0, "No consensus data available")
    elif rec_mean <= 1.5:
        add("Analyst Rating",    f"{rec_mean:.1f}",  3.0, "Strong Buy consensus")
    elif rec_mean <= 2.0:
        add("Analyst Rating",    f"{rec_mean:.1f}",  2.0, "Buy consensus")
    elif rec_mean <= 2.5:
        add("Analyst Rating",    f"{rec_mean:.1f}",  1.0, "Weak Buy / upper Hold")
    elif rec_mean <= 3.0:
        add("Analyst Rating",    f"{rec_mean:.1f}",  0.0, "Hold consensus")
    elif rec_mean <= 3.5:
        add("Analyst Rating",    f"{rec_mean:.1f}", -1.0, "Mild Underperform")
    elif rec_mean <= 4.0:
        add("Analyst Rating",    f"{rec_mean:.1f}", -2.0, "Sell consensus")
    else:
        add("Analyst Rating",    f"{rec_mean:.1f}", -3.0, "Strong Sell consensus")

    # ── 2. Bull case: upside to high price target ─────────────────────────────
    current     = float(context.get("current_price") or 0.0)
    high_target = float(context.get("high_price_target") or 0.0)
    if current > 0 and high_target > 0:
        upside_high = ((high_target - current) / current) * 100
        if upside_high >= 40:
            add("Bull Case (High Target)", f"+{upside_high:.1f}%",  2.0,
                f"${high_target:.2f} high target — significant bull case")
        elif upside_high >= 20:
            add("Bull Case (High Target)", f"+{upside_high:.1f}%",  1.5,
                f"${high_target:.2f} high target — meaningful bull case")
        elif upside_high >= 10:
            add("Bull Case (High Target)", f"+{upside_high:.1f}%",  1.0,
                f"${high_target:.2f} high target — moderate bull case")
        elif upside_high >= 0:
            add("Bull Case (High Target)", f"+{upside_high:.1f}%",  0.5,
                f"${high_target:.2f} high target — limited bull case")
        else:
            add("Bull Case (High Target)", f"{upside_high:.1f}%",  -1.0,
                f"${high_target:.2f} high target below current price")
    else:
        add("Bull Case (High Target)", "N/A", 0.0, "High price target not available")

    # ── 3. Mean target upside ─────────────────────────────────────────────────
    upside_mean = float(context.get("upside_to_target_pct") or 0.0)
    if upside_mean >= 30:
        add("Mean Target Upside", f"+{upside_mean:.1f}%",  1.5, "Mean target implies strong upside")
    elif upside_mean >= 15:
        add("Mean Target Upside", f"+{upside_mean:.1f}%",  1.0, "Mean target implies solid upside")
    elif upside_mean >= 5:
        add("Mean Target Upside", f"+{upside_mean:.1f}%",  0.5, "Mean target implies modest upside")
    elif upside_mean >= 0:
        add("Mean Target Upside", f"+{upside_mean:.1f}%",  0.0, "Near mean analyst target")
    else:
        add("Mean Target Upside", f"{upside_mean:.1f}%",  -0.5, "Stock above mean analyst target")

    # ── 4. Risk level ─────────────────────────────────────────────────────────
    risk_level = str(context.get("risk_level") or "medium").lower()
    if risk_level == "low":
        add("Risk Level",  "LOW",     1.0, "Low fundamental risk — supports higher signal")
    elif risk_level == "medium":
        add("Risk Level",  "MEDIUM",  0.0, "Moderate risk — neutral impact")
    else:
        add("Risk Level",  "HIGH",   -1.5, "High fundamental risk — dampens conviction")

    # ── 5. Price reset / mean-reversion potential ─────────────────────────────
    pct_off = float(context.get("pct_off_52w_high") or 0.0)
    if pct_off <= -50:
        add("Price Reset",    f"{pct_off:.1f}% off 52w high", +1.5,
            "Deep discount — significant mean-reversion potential")
    elif pct_off <= -30:
        add("Price Reset",    f"{pct_off:.1f}% off 52w high", +1.0,
            "Meaningful pullback — risk/reward improving")
    elif pct_off <= -15:
        add("Price Reset",    f"{pct_off:.1f}% off 52w high", +0.5,
            "Moderate pullback from highs")
    elif pct_off <= -5:
        add("Price Reset",    f"{pct_off:.1f}% off 52w high",  0.0,
            "Moderate discount — neutral")
    else:
        add("Price Momentum", f"{pct_off:.1f}% off 52w high", -0.5,
            "At or near 52-week high — near-term upside may be priced in")

    # ── 6. Insider signal ─────────────────────────────────────────────────────
    insider = str(context.get("insider_signal") or "unavailable").lower()
    if insider == "bullish":
        add("Insider Activity", "BULLISH",     +0.5, "Net insider buying — executives signalling confidence")
    elif insider == "bearish":
        add("Insider Activity", "BEARISH",     -0.5, "Net insider selling — executives reducing exposure")
    else:
        add("Insider Activity", "NEUTRAL/N/A",  0.0, "No significant insider signal")

    # ── 7. Revenue growth ─────────────────────────────────────────────────────
    rev_growth = float(context.get("revenue_growth_pct") or 0.0)
    if rev_growth >= 20:
        add("Revenue Growth",  f"+{rev_growth:.1f}%",  +0.5, "Strong revenue growth")
    elif rev_growth >= 0:
        add("Revenue Growth",  f"+{rev_growth:.1f}%",  +0.2, "Positive revenue growth")
    elif rev_growth >= -10:
        add("Revenue Growth",  f"{rev_growth:.1f}%",   -0.3, "Mild revenue contraction")
    else:
        add("Revenue Growth",  f"{rev_growth:.1f}%",   -0.7, "Significant revenue decline")

    # ── 8. Profitability ──────────────────────────────────────────────────────
    margin = float(context.get("profit_margin_pct") or 0.0)
    if margin >= 15:
        add("Profit Margin",  f"{margin:.1f}%",  +0.3, "Strong profitability")
    elif margin >= 5:
        add("Profit Margin",  f"{margin:.1f}%",  +0.1, "Adequate profitability")
    elif margin >= 0:
        add("Profit Margin",  f"{margin:.1f}%",   0.0, "Near breakeven")
    else:
        add("Profit Margin",  f"{margin:.1f}%",  -0.5, "Loss-making — earnings risk")

    # ── 9. Earnings proximity (modifier) ──────────────────────────────────────
    if context.get("earnings_within_14_days"):
        add("Earnings Risk", "within 14 days", -0.3,
            "Upcoming earnings add near-term uncertainty")

    # ── 10. Conflicting signals (modifier) ────────────────────────────────────
    if context.get("conflicting_signals"):
        add("Conflicting Signals", "Yes", -0.5,
            "Financial and sentiment signals contradict each other")

    # ── Map score to signal + conviction ──────────────────────────────────────
    if total >= 4.5:
        signal, conviction = "buy",  "high"
    elif total >= 2.5:
        signal, conviction = "buy",  "medium"
    elif total >= 1.0:
        signal, conviction = "buy",  "low"
    elif total >= -1.0:
        signal, conviction = "hold", "medium"
    elif total >= -2.5:
        signal, conviction = "hold", "low"
    elif total >= -4.0:
        signal, conviction = "sell", "medium"
    else:
        signal, conviction = "sell", "high"

    # ── Consensus alignment (deterministic) ───────────────────────────────────
    analyst_key     = str(context.get("analyst_consensus") or "").lower()
    analyst_bullish = analyst_key in ("buy", "strongbuy", "strong_buy", "outperform")
    analyst_bearish = analyst_key in ("sell", "strongsell", "strong_sell", "underperform")

    if signal == "buy"  and analyst_bullish:    alignment = "aligned"
    elif signal == "sell" and analyst_bearish:  alignment = "aligned"
    elif signal == "hold" and not analyst_bullish and not analyst_bearish:
        alignment = "aligned"
    else:
        alignment = "divergent"

    max_possible = 3.0 + 2.0 + 1.5 + 1.0 + 1.5 + 0.5 + 0.5 + 0.3

    return ScoredSignal(
        signal              = signal,
        conviction          = conviction,
        score               = round(total, 2),
        max_possible        = round(max_possible, 1),
        consensus_alignment = alignment,
        factors             = factors,
    )


def format_scorecard(signal: ScoredSignal) -> str:
    """Format the scoring breakdown as a readable string for LLM prompts."""
    lines = [
        f"Signal: {signal.signal.upper()}  |  "
        f"Conviction: {signal.conviction.upper()}  |  "
        f"Score: {signal.score:.1f} / {signal.max_possible:.1f}\n",
        "Scoring Factors:",
    ]
    for f in signal.factors:
        sign = "+" if f["contribution"] > 0 else ("" if f["contribution"] == 0 else "")
        contrib_str = f"{sign}{f['contribution']:.1f}".rjust(6)
        lines.append(
            f"  {contrib_str}  {f['factor']:<28} {f['value']:<18}  {f['note']}"
        )
    return "\n".join(lines)
