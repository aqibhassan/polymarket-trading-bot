---
paths:
  - "src/risk/**/*.py"
  - "tests/risk/**/*.py"
---

# Risk Management Rules

- Risk module is the FINAL gate before any order reaches execution
- NEVER bypass risk checks — no "skip risk" flags, no backdoors
- Pre-trade checks (all must pass):
  - Position size ≤ max_position_pct of portfolio
  - Total exposure ≤ max_exposure_pct
  - Daily drawdown < max_daily_drawdown (default 5%)
  - Correlation check: reject if would create >0.8 correlated position cluster
- Kill switch: auto-flatten all positions if daily loss exceeds threshold
- All risk parameters in `config/` — NEVER hardcoded
- P&L calculations must use mid-price, not last trade
- Log every risk rejection with: reason, attempted_order, current_state
