---
paths:
  - "src/strategies/**/*.py"
  - "tests/strategies/**/*.py"
---

# Strategy Code Rules

- Every strategy MUST extend `BaseStrategy` from `src/strategies/base.py`
- MUST implement: `generate_signals()`, `on_fill()`, `on_cancel()`
- MUST define `REQUIRED_PARAMS` class variable listing all config keys
- MUST have a corresponding TOML config in `config/strategies/`
- NEVER hardcode symbol names, thresholds, or position sizes
- All numeric params loaded from config with explicit types (int/float/Decimal)
- Use `Decimal` for prices and quantities — NEVER float for money
- Every strategy must have ≥80% test coverage
- Test with deterministic data fixtures, not live/random data
- Log signal generation with strategy_id, symbol, direction, strength
