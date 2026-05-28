# Contributing

Contributions are welcome — bug fixes, new tickers, real API integrations, or new agent types.

## Setup

```bash
git clone https://github.com/tarunbl/equity-research-agent.git
cd equity-research-agent
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install pytest
```

## Running Tests

```bash
pytest tests/ -v
```

Tests are written to run without an Anthropic API key — LLM calls are mocked where needed.

## Pull Request Guidelines

- One logical change per PR
- All tests must pass
- Add tests for new behaviour
- Follow the existing code style (type hints, docstrings, clean imports)
- Update `docs/ARCHITECTURE.md` if you change a design decision

## Adding a New Agent

1. Create `agents/your_agent.py` extending `BaseAgent`
2. Implement `get_system_prompt()` and `run()`
3. Add routing config to `config.py`
4. Add output schema to `models/schemas.py`
5. Wire into the pipeline in `main.py`
6. Add context scoping in `utils/context_builder.py`

## Adding a New Ticker

Add mock data to both `tools/financial_api.py` and `tools/search_tool.py`. Use realistic values — the escalation rules are calibrated to real-world thresholds, so extreme values will trigger escalation paths.
