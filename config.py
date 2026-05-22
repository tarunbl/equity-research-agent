"""
config.py
=========
Single source of truth for all agent configuration.

All agents are registered in ONE place — no dict mutations, no appended patches.
Change routing or budgets here; zero changes needed in agent code.
"""
from __future__ import annotations

# ── Model identifiers ─────────────────────────────────────────────────────────
MODELS: dict[str, str] = {
    "haiku":  "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus":   "claude-opus-4-6",
}

# ── Per-agent model routing ───────────────────────────────────────────────────
# Rule: cheapest model that reliably handles the task.
# Escalate surgically — only the failing agent re-runs on a stronger model.
# Opus is never the default; only via escalation from Sonnet.
MODEL_ROUTING: dict[str, dict[str, str]] = {
    "financial":      {"default": MODELS["haiku"],  "escalate": MODELS["sonnet"]},
    "news":           {"default": MODELS["haiku"],  "escalate": MODELS["sonnet"]},
    "valuation":      {"default": MODELS["haiku"],  "escalate": MODELS["sonnet"]},
    "risk":           {"default": MODELS["sonnet"], "escalate": MODELS["opus"]},
    "recommendation": {"default": MODELS["sonnet"], "escalate": MODELS["opus"]},
    "formatter":      {"default": MODELS["haiku"],  "escalate": MODELS["sonnet"]},
}

# ── Max output tokens per agent ───────────────────────────────────────────────
# Sized to cover realistic outputs with ~15% headroom.
# Increase only when a specific agent consistently truncates.
TOKEN_BUDGETS: dict[str, int] = {
    "financial":      500,   # Structured JSON, bounded schema
    "news":           600,   # Scored headlines list
    "valuation":      600,   # Structured snapshot
    "risk":           1000,  # Multi-axis analysis with lists
    "recommendation": 1100,  # Verdict + consensus reconciliation
    "formatter":      2200,  # 6 prose sections + metadata
}

# ── Confidence thresholds ─────────────────────────────────────────────────────
# Keep LOW. LLMs self-report conservatively — a Sonnet returning 0.62 on a
# clean analysis is NOT genuinely uncertain. Tight thresholds (0.65+) trigger
# unnecessary Opus escalations (+20–30s per trigger).
# None = skip confidence check for that agent entirely.
CONFIDENCE_THRESHOLDS: dict[str, float | None] = {
    "financial":      None,   # Extraction task — skip
    "news":           None,   # Classification task — skip
    "valuation":      None,   # Factual reporting — skip
    "risk":           0.40,   # Only escalate on genuine ambiguity
    "recommendation": 0.40,   # Only escalate on genuinely split evidence
    "formatter":      None,   # Assembly task — skip
}

# ── Rule-based escalation ─────────────────────────────────────────────────────
# Evaluated in Python BEFORE calling the LLM — zero token cost.
# Operators: "<" | ">" | "==" | "!="
ESCALATION_RULES: dict[str, list[dict]] = {
    "financial": [
        {
            "field":    "data_quality_hint",
            "operator": "<",
            "value":    0.6,
            "reason":   "Raw data quality below threshold — needs stronger model",
        },
    ],
    "news":      [],  # Pre-call handled via keyword heuristic in NewsAgent.run()
    "valuation": [],
    "risk": [
        {
            "field":    "debt_to_equity",
            "operator": ">",
            "value":    3.0,
            "reason":   "Extreme leverage (D/E > 3.0) — complex risk reasoning required",
        },
        {
            "field":    "conflicting_signals",
            "operator": "==",
            "value":    True,
            "reason":   "Conflicting financial vs sentiment signals detected",
        },
    ],
    "recommendation": [
        {
            "field":    "risk_level",
            "operator": "==",
            "value":    "high",
            "reason":   "High-risk company — investment judgment requires Opus",
        },
    ],
    "formatter": [],
}

# ── Retry settings ────────────────────────────────────────────────────────────
MAX_RETRIES:           int   = 3
RETRY_BASE_DELAY_SECS: float = 1.0   # Doubles per attempt: 1s → 2s → 4s

# ── API timeout ───────────────────────────────────────────────────────────────
API_TIMEOUT_SECS: float = 90.0

# ── yfinance I/O timeout ──────────────────────────────────────────────────────
# Applied per-call when fetching financial data and news in the thread pool.
YFINANCE_TIMEOUT_SECS: float = 30.0

# ── Approximate cost per million tokens (USD) ─────────────────────────────────
# For run-summary estimation only. See anthropic.com/pricing for current rates.
# Caching: write = 1.25× base input, read = 0.10× base input.
COST_PER_MILLION: dict[str, dict[str, float]] = {
    MODELS["haiku"]: {
        "input":       0.80,
        "output":      4.00,
        "cache_write": 1.00,
        "cache_read":  0.08,
    },
    MODELS["sonnet"]: {
        "input":       3.00,
        "output":      15.00,
        "cache_write": 3.75,
        "cache_read":  0.30,
    },
    MODELS["opus"]: {
        "input":       15.00,
        "output":      75.00,
        "cache_write": 18.75,
        "cache_read":  1.50,
    },
}
