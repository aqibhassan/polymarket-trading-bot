# Micro-Volatility Harvesting Engine (MVHE)

## Identity
- **Codename**: MVHE
- **Purpose**: Systematic micro-volatility capture on Polymarket BTC 15m prediction markets
- **Stage**: Engine ready — paper trading
- **Owner**: Future Syncs Limited (UK No. 15950168)

## Stack
- **All Components**: Python 3.12+ (strategy, execution, backtesting, analytics)
- **Data**: TimescaleDB (OHLCV/tick), Redis (state/cache), ClickHouse (analytics)
- **Infra**: Docker Compose (dev), K8s (prod), Grafana (monitoring)

## CRITICAL RULES — NEVER VIOLATE
- NEVER commit secrets/keys/credentials to git
- NEVER execute live trades without `--live` flag + confirmation prompt
- MomentumConfirmation uses hold-to-settlement (no stop loss) — resolution guard at minute 14
- ALL sizing MUST respect `config.risk.max_position_pct` (default 2%)
- Paper trade first — use `--paper` flag
- Log every order attempt, fill, rejection to audit trail

## Commands
```
make dev        # Docker Compose dev stack
make test       # Full test suite
make backtest   # Run backtesting suite
make lint       # ruff + mypy
make paper      # Start paper trading
```

## Architecture
See `docs/brain/ARCHITECTURE.md` for system design.
Subdirectory CLAUDE.md files load on-demand — see `.claude/rules/` for domain rules.

## Memory Systems (Auto-Active)
- **memory-bank** MCP: Project context in `.memory-bank/`
- **knowledge-graph** MCP: Entities/relations in `.aim/` (local) + `~/.claude/knowledge-graph/` (global)
- Both are 100% local JSONL/markdown — no cloud, no cost

## Key Conventions
- Type hints everywhere (Python) — `mypy --strict`
- All strategies inherit `BaseStrategy` in `src/strategies/base.py`
- Config via TOML files in `config/` — never hardcode values
- Tests mirror src structure: `src/engine/foo.py` → `tests/engine/test_foo.py`
- Branch naming: `feat/`, `fix/`, `refactor/`, `test/`
