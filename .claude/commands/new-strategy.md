# /new-strategy

Create a new trading strategy with all required files.

<instructions>
When the user runs /new-strategy, ask for:
1. Strategy name (snake_case)
2. Brief description
3. Primary signal type (volatility, momentum, mean-reversion, spread)

Then create ALL of these files:
- `src/strategies/{name}.py` — Strategy class extending BaseStrategy
- `tests/strategies/test_{name}.py` — Test file with fixtures
- `config/strategies/{name}.toml` — Default parameters

The strategy class MUST:
- Extend BaseStrategy from src/strategies/base.py
- Implement generate_signals(), on_fill(), on_cancel()
- Define REQUIRED_PARAMS class variable
- Use Decimal for all prices/quantities
- Include docstring with strategy description and parameters
- Have type hints on all methods

The test file MUST:
- Test signal generation with known data
- Test edge cases (empty data, single tick, extreme values)
- Test parameter validation
- Use fixtures, not hardcoded test data

The config MUST:
- Include all REQUIRED_PARAMS with sensible defaults
- Include comments explaining each parameter
- Include [risk] section with strategy-specific limits
</instructions>
