# 📈 Equity Research Multi-Agent System

> A production-grade multi-agent AI framework that generates structured equity research reports using intelligent model routing, parallel execution, and cost-optimised LLM orchestration — built on the Anthropic Claude API.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![Anthropic](https://img.shields.io/badge/Anthropic-Claude_API-D97757?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)
![Status](https://img.shields.io/badge/Status-Portfolio_Project-blue?style=flat)

---

## 🎯 What It Does

Given a stock ticker (`AAPL`, `TSLA`, `GME`, `NVDA`), the system spins up five specialised AI agents that work in parallel and in sequence to produce a full equity research brief — covering financials, news sentiment, risk analysis, and a buy/hold/sell recommendation — in under 10 seconds.

**The key engineering challenge:** doing this intelligently and cheaply. Not every task needs the same model. A sentiment classifier doesn't need the same firepower as a risk analyst reconciling conflicting signals. This system routes each agent to the right model automatically, escalating to a stronger (and more expensive) model only when the task demands it.

---

## 🏗️ Architecture

```
                        ┌─────────────────────────────┐
                        │   main.py  (Python Pipeline) │
                        │   No LLM — pure orchestration│
                        └──────────────┬──────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │   STAGE 1 — Parallel Data Collection │
                    │                                      │
            ┌───────▼──────┐                   ┌──────────▼──────┐
            │  Financial    │                   │   News Sentiment │
            │  Agent        │                   │   Agent          │
            │  [Haiku]      │                   │   [Haiku]        │
            └───────┬───────┘                   └──────────┬───────┘
                    │                                       │
                    └──────────────────┬────────────────────┘
                                       │
                            ┌──────────▼──────────┐
                            │   STAGE 2 — Analysis  │
                            │   Risk Agent          │
                            │   [Sonnet → Opus*]    │
                            └──────────┬────────────┘
                                       │
                            ┌──────────▼──────────┐
                            │   STAGE 3 — Judgment  │
                            │   Recommendation Agent│
                            │   [Sonnet → Opus*]    │
                            └──────────┬────────────┘
                                       │
                            ┌──────────▼──────────┐
                            │   STAGE 4 — Render    │
                            │   Formatter Agent     │
                            │   [Haiku]             │
                            └──────────┬────────────┘
                                       │
                          ┌────────────▼──────────────┐
                          │  Rich Terminal Report      │
                          │  + JSON output file        │
                          └────────────────────────────┘

  * Opus used only when escalation is triggered (rule-based or low confidence)
```

---

## ⚡ Intelligent Model Routing

This is the core design differentiator. Each agent is assigned the cheapest model capable of doing its job well. A stronger model is called **only when necessary**.

| Agent | Default Model | Escalates To | Escalation Trigger |
|-------|--------------|--------------|-------------------|
| Financial Agent | `claude-haiku-4-5` | `claude-sonnet-4-6` | `data_quality < 0.6` |
| News Agent | `claude-haiku-4-5` | `claude-sonnet-4-6` | `negative_pct > 50%` (post-call) |
| Risk Agent | `claude-sonnet-4-6` | `claude-opus-4-6` | `D/E > 3.0` or conflicting signals |
| Recommendation Agent | `claude-sonnet-4-6` | `claude-opus-4-6` | `risk_level == "high"` |
| Formatter Agent | `claude-haiku-4-5` | `claude-sonnet-4-6` | Low confidence only |

**Two-layer escalation strategy:**
1. **Rule-based (pre-LLM):** Python checks raw data against domain thresholds — zero token cost.
2. **Confidence-based (post-LLM):** If the model self-reports low confidence, the task re-runs on the stronger model.

**Additional optimisations:**
- Prompt caching on all system prompts (`cache_control: ephemeral`)
- Strict per-agent `max_tokens` budgets
- Minimum-sufficient context per agent (no full session dumps)
- `asyncio.gather()` for parallel Stage 1 execution

---

## 🗂️ Project Structure

```
equity-research-agent/
├── main.py                      # Pipeline entry point (pure Python, no LLM)
├── config.py                    # Model routing, budgets, escalation rules, costs
├── agents/
│   ├── base_agent.py            # Abstract base: routing, caching, retry, escalation
│   ├── financial_agent.py       # Extracts & validates financial metrics
│   ├── news_agent.py            # Classifies headline sentiment
│   ├── risk_agent.py            # Multi-axis risk analysis
│   ├── recommendation_agent.py  # Buy / Hold / Sell with rationale
│   └── formatter_agent.py       # Assembles and renders final report
├── tools/
│   ├── financial_api.py         # Mock financial data (AAPL, TSLA, GME, NVDA)
│   └── search_tool.py           # Mock news headlines (4 sentiment profiles)
├── models/
│   └── schemas.py               # Pydantic v2 output schemas for all agents
├── memory/
│   └── session_store.py         # Per-run in-memory key-value store
├── monitoring/
│   └── logger.py                # Token tracking, cost estimation, Rich summary
├── utils/
│   └── context_builder.py       # Scopes min-sufficient context per agent
├── tests/
│   ├── test_schemas.py          # Pydantic validation tests
│   ├── test_context_builder.py  # Context scoping unit tests
│   └── test_escalation.py       # Escalation rule logic tests
├── docs/
│   └── ARCHITECTURE.md          # Deep-dive design decisions
├── .github/
│   └── workflows/ci.yml         # GitHub Actions: lint + test
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/equity-research-agent.git
cd equity-research-agent
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env and add your Anthropic API key
```

### 3. Run

```bash
# Healthy company — no escalation (demonstrates normal routing)
python main.py AAPL

# Mixed signals — Risk agent escalates on conflicting data
python main.py TSLA

# High-risk company — multiple escalations triggered
python main.py GME

# Extraordinary growth — tests high-confidence Sonnet path
python main.py NVDA
```

---

## 📊 Sample Output

```
╭─────────────────── Equity Research Agent — GME ──────────────────╮
│                                                                    │
│  RECOMMENDATION:  ██ SELL  (Conviction: HIGH)                     │
│  Time Horizon:    12–18 months                                     │
│  Risk Level:      🔴 HIGH  (Score: 8.4 / 10)                      │
│                                                                    │
╰────────────────────────────────────────────────────────────────────╯

╭─── Run Summary ────────────────────────────────────────────────────╮
│  Agent              Model            In    Out  Cached  Escalated  │
│  ─────────────────────────────────────────────────────────────── │
│  financial          haiku-4-5        243   312     —      ✓ No    │
│  news               haiku-4-5        589   287     —    ⬆ YES     │
│  risk               sonnet→opus      934   521    180   ⬆ YES     │
│  recommendation     sonnet→opus      712   487    180   ⬆ YES     │
│  formatter          haiku-4-5        621   831     —      ✓ No    │
│  ─────────────────────────────────────────────────────────────── │
│  Total tokens:   3,099 in / 2,438 out   |  Est. cost: $0.03211    │
│  Cache savings:  360 tokens reused      |  Wall time: 11.2s        │
│  Escalations:    3 of 5 agents (rule-based × 2, confidence × 1)   │
╰────────────────────────────────────────────────────────────────────╯
```

---

## 🧠 Framework: 10-Step Agent Design

This project implements a complete 10-step multi-agent design framework:

| Step | Concept | Where It Lives |
|------|---------|---------------|
| 1 | Clear task definition | Each agent has one focused responsibility |
| 2 | Right LLM for the task | `config.py` → `MODEL_ROUTING` |
| 3 | System instructions | `get_system_prompt()` in each agent |
| 4 | Agent logic & control flow | `agents/base_agent.py` + `main.py` pipeline |
| 5 | Tool & API integration | `tools/financial_api.py`, `tools/search_tool.py` |
| 6 | Memory (short-term) | `memory/session_store.py` — per-run session state |
| 7 | Multi-agent coordination | DAG in `main.py`: parallel → sequential stages |
| 8 | Test & validate | `tests/` — schema, escalation, context tests |
| 9 | Monitor & feedback | `monitoring/logger.py` — token tracking, cost, latency |
| 10 | Deploy-ready structure | `.env` config, `requirements.txt`, CI workflow |

---

## 🔧 Extending the System

**Add a new ticker:**  Add mock data to `tools/financial_api.py` and `tools/search_tool.py`.

**Add a new agent:**  Extend `BaseAgent`, define `get_system_prompt()` and `run()`, register in `main.py`.

**Connect real APIs:**  Replace `tools/financial_api.py` with Alpha Vantage / Yahoo Finance. Replace `tools/search_tool.py` with a real news API.

**Change routing rules:**  All routing logic lives in `config.py` — no agent code changes needed.

---

## 📄 License

MIT — see [LICENSE](LICENSE)

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)
