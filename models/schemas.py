"""
models/schemas.py
=================
Pydantic v2 output schemas for every agent in the pipeline.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


class Signal(str, Enum):
    BUY  = "buy"
    HOLD = "hold"
    SELL = "sell"


class Conviction(str, Enum):
    HIGH   = "high"
    MEDIUM = "medium"
    LOW    = "low"


class ConsensusAlignment(str, Enum):
    ALIGNED   = "aligned"
    DIVERGENT = "divergent"


# ── Financial Agent output ─────────────────────────────────────────────────────

class FinancialOutput(BaseModel):
    ticker:               str
    company_name:         str
    revenue_growth_pct:   float
    gross_margin_pct:     float
    profit_margin_pct:    float
    pe_ratio:             float
    debt_to_equity:       float
    current_ratio:        float
    return_on_equity_pct: float
    data_quality_score:   float = Field(ge=0.0, le=1.0)
    anomalies:            List[str] = []
    confidence:           float = Field(ge=0.0, le=1.0)
    escalate:             bool = False
    escalation_reason:    Optional[str] = None


# ── News Sentiment Agent output ────────────────────────────────────────────────

class HeadlineSentiment(BaseModel):
    headline:  str
    sentiment: str
    score:     float


class NewsOutput(BaseModel):
    ticker:             str
    headlines_analyzed: int
    overall_sentiment:  str
    sentiment_score:    float
    positive_pct:       float
    neutral_pct:        float
    negative_pct:       float
    top_headlines:      List[HeadlineSentiment]
    key_themes:         List[str]
    confidence:         float = Field(ge=0.0, le=1.0)
    escalate:           bool = False
    escalation_reason:  Optional[str] = None


# ── Valuation Agent output ─────────────────────────────────────────────────────

class ValuationOutput(BaseModel):
    """
    Structured valuation snapshot including price context, multiples,
    analyst consensus, and earnings proximity.

    This is the key output missing from the original system that caused
    the ELF SELL error — the agent had no idea the stock was 64% off
    its high with 16 analysts rating it BUY.
    """
    ticker:                  str
    current_price:           float
    week_52_high:            float
    week_52_low:             float
    pct_off_52w_high:        float   # negative = below high, e.g. -63.5
    trailing_pe:             float
    forward_pe:              float
    peg_ratio:               float
    price_to_sales:          float
    ev_to_ebitda:            float
    analyst_count:           int
    analyst_consensus:       str     # e.g. "buy", "hold", "strongBuy"
    mean_price_target:       float
    upside_to_target_pct:    float   # e.g. +89.0 means 89% implied upside
    valuation_stance:        str     # "cheap" | "fair" | "expensive"
    earnings_date:           Optional[str]
    earnings_within_14_days: bool
    key_insights:            List[str]   # 3–5 bullet observations
    confidence:              float = Field(ge=0.0, le=1.0)
    escalate:                bool = False
    escalation_reason:       Optional[str] = None


# ── Risk Analysis Agent output ─────────────────────────────────────────────────

class RiskOutput(BaseModel):
    ticker:                 str
    risk_level:             RiskLevel
    risk_score:             float
    conflicting_signals:    bool
    financial_risk_summary: str
    sentiment_risk_summary: str
    sector_risk_summary:    str
    top_risk_factors:       List[str]
    mitigating_factors:     List[str]
    confidence:             float = Field(ge=0.0, le=1.0)
    escalate:               bool = False
    escalation_reason:      Optional[str] = None


# ── Recommendation Agent output ────────────────────────────────────────────────

class RecommendationOutput(BaseModel):
    ticker:               str
    signal:               Signal
    conviction:           Conviction
    time_horizon:         str
    rationale:            str
    bull_case:            str
    bear_case:            str
    key_catalysts:        List[str]
    key_risks:            List[str]
    # Phase 2 additions — analyst consensus reconciliation
    consensus_alignment:  ConsensusAlignment
    consensus_note:       str    # Why you agree or disagree with analyst consensus
    upside_downside_note: str    # e.g. "Analyst mean target $103 implies +89% upside"
    earnings_flag:        Optional[str] = None  # Warning if earnings within 14 days
    confidence:           float = Field(ge=0.0, le=1.0)
    escalate:             bool = False
    escalation_reason:    Optional[str] = None


# ── Formatter Agent output ─────────────────────────────────────────────────────

class FormatterOutput(BaseModel):
    ticker:              str
    company_name:        str
    report_date:         str
    executive_summary:   str
    financial_overview:  str
    valuation_snapshot:  str
    sentiment_analysis:  str
    risk_assessment:     str
    recommendation_text: str
    signal:              str
    conviction:          str
    time_horizon:        str
    risk_level:          str
    consensus_alignment: str
    upside_downside_note: str
    earnings_flag:       Optional[str] = None


# ── Per-agent run metrics ─────────────────────────────────────────────────────

class AgentMetrics(BaseModel):
    agent_name:         str
    model_used:         str
    input_tokens:       int = 0
    output_tokens:      int = 0
    cache_write_tokens: int = 0
    cache_read_tokens:  int = 0
    escalated:          bool = False
    escalation_reason:  Optional[str] = None
    escalation_layer:   Optional[str] = None
    latency_ms:         float = 0.0
    estimated_cost_usd: float = 0.0
