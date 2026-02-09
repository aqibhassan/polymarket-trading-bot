---
paths:
  - "src/data/**/*.py"
  - "tests/data/**/*.py"
---

# Data Pipeline Rules

- All timestamps MUST be UTC — never local timezone
- Store timestamps as timezone-aware datetime objects
- Use nanosecond precision where exchange provides it
- WebSocket handlers must track sequence numbers to detect gaps
- On gap detection: log warning, request snapshot, reconcile
- OHLCV bars built from tick stream, not exchange pre-built (more accurate)
- TimescaleDB hypertables: partition by time (1 day chunks)
- Redis keys prefixed with `mvhe:` namespace to avoid collisions
- Data validation on ingestion: reject ticks with price ≤ 0 or future timestamps
- Backfill support: historical data fetched via REST, same schema as live
