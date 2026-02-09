---
paths:
  - "src/execution/**/*"
  - "execution-rs/**/*"
---

# Execution Layer Rules

- Rust code lives in `execution-rs/` with PyO3 bindings
- NEVER block the event loop — all I/O must be async
- Order IDs must be UUID v7 (time-sortable)
- Every order state transition MUST be logged to audit trail
- Implement circuit breaker: halt after N consecutive failures (configurable)
- Rate limiter per exchange — respect documented limits with 20% safety margin
- WebSocket reconnection with exponential backoff (base 1s, max 60s, jitter)
- All prices as string-encoded decimals across the Rust ↔ Python boundary
- Reject any order missing stop-loss before it reaches the exchange
