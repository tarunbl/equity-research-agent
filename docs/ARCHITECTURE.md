# Architecture

Design decisions behind the Equity Research Multi-Agent System.

---

## Why the Orchestrator Is Pure Python

The orchestrator in most tutorial projects is itself an LLM agent. This is a mistake for fixed pipelines.

The orchestrator's job here is deterministic: validate input → start parallel tasks → collect results → pass to next stage. No judgment is required. Using an LLM for this wastes tokens, introduces non-determinism, and creates an extra failure point.

**Rule:** Use an LLM only where human-like judgment is needed. Routing a fixed DAG is not judgment — it is plumbing.

```
main.py  →  pure Python asyncio pipeline
         →  no LLM calls
         →  deterministic, fast, free
```

---

## Why Two-Layer Escalation

A single escalation strategy is insufficient:

| Strategy | Strength | Weakness |
|----------|----------|---------|
| Rule-based only | Deterministic, zero cost | Blind to novel edge cases |
| Confidence-based only | Catches emergent uncertainty | LLMs are poorly calibrated; may be confident when wrong |

**The layered approach:**
1. **Rule-based (pre-LLM):** Python checks raw data against known thresholds before any LLM call. Zero tokens spent. Catches obvious high-risk situations (e.g. D/E > 3.0).
2. **Confidence-based (post-LLM):** After the default model responds, its self-reported confidence score is checked. If below threshold, the task re-runs on the escalated model. Catches subtle cases rules miss.

Rules run first because they are free. Confidence checks cost one LLM call before potentially triggering a second on a stronger model.

---

## Why Context Scoping Matters

A common anti-pattern is passing the full session state to every agent:

```python
# Bad: every agent sees everything
agent.run(session.get_all())

# Good: each agent receives only what it needs
agent.run(context_builder.build_risk_context(financial, news))
```

Passing full context has three costs:
1. **Token cost:** Every extra token in the prompt is billed, even if the agent ignores it.
2. **Quality cost:** Irrelevant context can distract reasoning models.
3. **Security cost:** In production, agents shouldn't have access to data they don't need.

`utils/context_builder.py` defines exactly what each agent receives. The risk agent gets financial ratios and sentiment scores — not raw API responses. The recommendation agent gets the risk summary — not the full news feed.

---

## Why the News Agent Escalation Is Post-Call

Most escalation rules run pre-LLM (before any tokens are spent). The news agent is a special case: its escalation trigger is `negative_pct > 50%`, but `negative_pct` is *computed* by the LLM — it is not available in the raw input.

The solution: run the news agent on Haiku first, parse the output, then check `negative_pct`. If above threshold, re-run on Sonnet. This is handled explicitly in `news_agent.run()` rather than in `base_agent`, keeping the base class clean for the common case.

---

## Why Pydantic Schemas Between Agents

Without explicit schemas, inter-agent communication becomes fragile:

- Agent A returns `{"risk": "high"}` — Agent B expects `{"risk_level": "high"}` — silent failure.
- Agent A omits a field Agent B assumes exists — `KeyError` in production.

Every agent in this system returns a validated Pydantic model. If an agent's output doesn't match its schema, a `ValidationError` is raised immediately — not silently propagated through the pipeline.

```python
# In every agent's run() method:
validated = RiskOutput(**parsed)   # raises ValidationError if schema mismatch
output = validated.model_dump()    # only clean, typed data passes downstream
```

---

## Why Prompt Caching on System Prompts

Each agent's system prompt is static — it doesn't change between runs. Without caching, the same system prompt tokens are billed on every call.

With `cache_control: ephemeral`, Anthropic stores the computed representation of the system prompt. On the second call within 5 minutes (e.g. running TSLA after AAPL), those tokens are read from cache at ~10% of the normal input price.

**Note:** Haiku 4.5 requires a minimum of 4,096 tokens for cache activation. For shorter system prompts, the API silently skips caching without error. As the system grows and system prompts are enriched with examples and guardrails, they will naturally cross this threshold.

---

## Why Strict `max_tokens` Budgets

Each agent has a hard ceiling on its output:

| Agent | Budget | Reason |
|-------|--------|--------|
| Financial | 450 | JSON schema — predictable, bounded |
| News | 350 | Scored headline list — short by design |
| Risk | 650 | Analysis needs room but not essays |
| Recommendation | 550 | Verdict + rationale, not a report |
| Formatter | 950 | Must assemble prose sections |

Without these budgets, models — especially Sonnet and Opus — will produce verbose prose where structured data was requested. Tight budgets enforce conciseness and prevent cost overruns on repeat runs.

---

## Why the Formatter Agent Uses Haiku

The formatter's job is templating: receive structured data from all prior agents, write clean professional prose for each section, return a structured object for Python to render.

This is a completion task, not a reasoning task. Haiku handles it well. The formatter never makes investment judgments — those are locked in by the time it receives its inputs.

The Python rendering layer (`rich`) handles terminal formatting. The LLM handles content. These are kept deliberately separate.

---

## Trade-offs and Known Limitations

**Mock data only:** The tools layer uses in-memory mock data. Connecting to real APIs (Alpha Vantage, Bloomberg, news feeds) is a straightforward extension but introduces rate limits, authentication, and data normalisation complexity.

**No persistent memory:** The session store is in-memory and scoped to a single run. A production system would persist run history to a database, enabling trend analysis across multiple research runs on the same ticker.

**Haiku caching threshold:** Short system prompts don't cache on Haiku. Enriching prompts with few-shot examples would both improve quality and cross the caching threshold — a natural future improvement.

**Single-turn agents:** Each agent makes one LLM call (two if escalated). A ReAct or multi-turn inner loop could improve reasoning quality for the risk and recommendation agents at the cost of latency and tokens.
