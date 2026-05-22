# Setup Guide

Complete setup instructions for the Equity Research Multi-Agent System.

---

## Prerequisites

- Python 3.11 or 3.12
- `pip`
- An Anthropic account with API access

---

## Step 1 — Clone and Install

```bash
git clone https://github.com/YOUR_USERNAME/equity-research-agent.git
cd equity-research-agent

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

---

## Step 2 — Configure API Keys

Copy the example environment file and fill it in:

```bash
cp .env.example .env
```

Then open `.env` and add your keys (instructions for each below).

---

## API Key Reference

### 🔴 Required: Anthropic

The pipeline uses Claude models for all analysis. Without this key nothing runs.

1. Go to **https://console.anthropic.com/settings/keys**
2. Click **Create Key**
3. Copy the key (starts with `sk-ant-...`)
4. Add to `.env`:

```
ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxxxxxxxxxx
```

---

### 🟡 Recommended: Finnhub (Free Tier)

Adds two data sources the base system lacks:
- **Multi-source news** from Reuters, Seeking Alpha, Benzinga, and others — cross-validates Yahoo Finance headlines
- **Insider transactions** (SEC Form 4) — whether executives are buying or selling their own stock

**Free tier:** 60 API calls/minute — more than enough for this pipeline.

1. Go to **https://finnhub.io** and click **Get free API key**
2. Sign up (no credit card required)
3. Your key is shown on the dashboard immediately
4. Add to `.env`:

```
FINNHUB_API_KEY=your_finnhub_key_here
```

If this key is absent, the pipeline runs normally with Yahoo Finance news only.

---

### 🟢 Free, No Key: SEC EDGAR

Adds SEC filing data:
- **10-K Risk Factors** — the company's own legal disclosure of material risks
- **Recent 8-K Filings** — material events from the last 30 days (earnings, leadership changes, acquisitions)

No account or API key needed. The SEC only requires you to identify yourself via a `User-Agent` header.

Add to `.env`:

```
SEC_USER_AGENT=Your Name your@email.com
```

Use your real name and email. The SEC uses this to contact you if your usage is unusual — not for any other purpose. Example:

```
SEC_USER_AGENT=Jane Smith jane.smith@gmail.com
```

If this is absent, a generic user agent is used and SEC data may be rate-limited or blocked.

---

## Final `.env` File

A complete `.env` looks like this:

```
# Required
ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxxxxxxxxxx

# Recommended (free)
FINNHUB_API_KEY=d1a2b3c4d5e6f7g8h9i0j1k2

# Free, no sign-up needed
SEC_USER_AGENT=Jane Smith jane@example.com
```

---

## Step 3 — Run

```bash
# Any valid stock ticker
python main.py AAPL
python main.py ELF
python main.py MSFT
python main.py BRK-B
```

---

## What Each Run Does

```
Stage 1  Financial data       Yahoo Finance (yfinance)
         News headlines        Yahoo Finance + Finnhub (if key set)
         Risk intelligence     SEC EDGAR 10-K + 8-K + Finnhub insiders

Stage 2  Valuation snapshot   Price context, analyst consensus

Stage 3  Risk analysis        Uses all Stage 1 data including SEC + insiders

Stage 4  Recommendation       Consensus-aware buy/hold/sell

Stage 5  Report               Terminal output + JSON saved to output/
```

---

## Data Sources Summary

| Source | Data | Key Required | Cost |
|--------|------|-------------|------|
| Yahoo Finance (yfinance) | Financials, price, analyst consensus, news | No | Free |
| SEC EDGAR | 10-K risk factors, 8-K material events | No (email in User-Agent) | Free |
| Finnhub | Multi-source news, insider transactions | Yes (free) | Free tier |

---

## Troubleshooting

**`ModuleNotFoundError`**
Make sure you activated your virtual environment:
```bash
source venv/bin/activate
```
Then install dependencies: `pip install -r requirements.txt`

**`401 authentication_error`**
Your Anthropic API key is wrong or missing. Check `.env` has no quotes around the key:
```
ANTHROPIC_API_KEY=sk-ant-api03-...    # correct
ANTHROPIC_API_KEY="sk-ant-api03-..."  # wrong — remove quotes
```

**`No data found for 'XYZ'`**
The ticker is not listed on Yahoo Finance. Try the exact exchange symbol (e.g. `BRK-B` not `BRKB`).

**SEC EDGAR returns no risk factors**
Some smaller companies have filings in formats that are harder to parse. The pipeline continues without SEC data in this case.

**Pipeline is slow (>45 seconds)**
This usually means one or both Sonnet agents escalated to Opus. Check the run summary table — escalated agents are marked with `⬆ YES`. Escalation is intentional for high-risk companies but adds ~25s per trigger.
