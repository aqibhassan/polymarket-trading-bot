---
paths:
  - "tests/**/*.py"
---

# Testing Rules

- Use pytest with pytest-asyncio for async tests
- Test files mirror src: `src/engine/signals.py` → `tests/engine/test_signals.py`
- Fixtures in `tests/conftest.py` and domain-specific `tests/<domain>/conftest.py`
- Use `Decimal` for all price/quantity assertions — no float comparison
- Mock exchange APIs — NEVER hit real endpoints in tests
- Use `freezegun` or `time-machine` for time-dependent tests
- Property-based testing with `hypothesis` for strategy logic
- Minimum coverage: 80% overall, 95% for risk module
- Integration tests in `tests/integration/` — require Docker (mark with @pytest.mark.integration)
- Backtest result assertions: Sharpe, max drawdown, win rate — not just P&L
