# Architecture Decision Records

## ADR-001: Python-First Execution (Rust Deferred)
- **Date**: 2026-02-08
- **Status**: Accepted (updated)
- **Context**: Need fast prototyping for strategies but sub-ms execution
- **Decision**: Python for all components initially; Rust via PyO3 deferred to optimization phase
- **Consequence**: Faster iteration, single build system; Rust can be added when latency matters

## ADR-002: TimescaleDB over InfluxDB
- **Date**: 2026-02-08
- **Status**: Accepted
- **Context**: Need time-series with SQL compatibility for complex analytics
- **Decision**: TimescaleDB (PostgreSQL extension) — free, OSS, SQL native
- **Consequence**: Can use full PostgreSQL ecosystem, pgvector for embeddings later

## ADR-003: Momentum-Following over Contrarian
- **Date**: 2026-02-08
- **Status**: Accepted
- **Context**: Original FalseSentimentStrategy (contrarian: fade crowd) vs MomentumConfirmation (follow BTC trend)
- **Decision**: MomentumConfirmation — data showed BTC 15m candles are momentum-driven, not mean-reverting
- **Evidence**: 89.6% accuracy (momentum) vs ~75% (contrarian), 14.5pp advantage across all regimes
- **Consequence**: FalseSentimentStrategy kept for reference but superseded

## ADR-004: Hold-to-Settlement (No Stop Loss)
- **Date**: 2026-02-08
- **Status**: Accepted
- **Context**: Binary markets pay $1.00 correct / $0.00 wrong. Should we use stop losses or take profit?
- **Decision**: Hold to settlement with resolution guard at minute 14
- **Evidence**: Stop losses trigger on sigmoid token price noise, destroying 5-10% of winning trades. Early exit at 40% TP captures <1% of trades. Settlement is where profit comes from.
- **Consequence**: Higher variance per trade but 89.4% accuracy makes it optimal

## ADR-005: Tiered Entry Thresholds by Minute
- **Date**: 2026-02-08
- **Status**: Accepted
- **Context**: Flat threshold vs time-varying thresholds for entry signals
- **Decision**: Tiered entry: min8@0.10%, min9@0.08%, min10@0.05%
- **Evidence**: Later minutes have inherently higher accuracy (less time for reversal), so lower thresholds are safe. Grid search over 132K candles found this optimal.
- **Consequence**: More trades (6,602) with better P&L than flat threshold

## ADR-006: Quarter-Kelly Position Sizing
- **Date**: 2026-02-08
- **Status**: Accepted
- **Context**: Fixed $100 sizing gave 21.73% ruin probability despite 89.4% accuracy (binary payout asymmetry: win +$35 avg, lose -$100)
- **Decision**: Binary Kelly criterion f* = (p-P)/(1-P), multiplied by 0.25 (quarter-Kelly), capped at 2% balance and 10% market volume
- **Evidence**: Monte Carlo simulation: ruin 0.00%, max drawdown <10% at 99th percentile, $10K starting capital sufficient
- **Consequence**: Lower individual trade sizes but eliminates ruin risk entirely

## ADR-007: Polymarket Dynamic Fee Model
- **Date**: 2026-02-08
- **Status**: Accepted
- **Context**: Need accurate fee modeling for P&L calculations
- **Decision**: fee = position_size * 0.25 * P^2 * (1-P)^2 (Polymarket 15-min dynamic taker fee)
- **Evidence**: Peaks at 1.56% at 50/50 odds, drops to ~0.2% at extremes. Average $0.88/trade, only 3.1% of gross profit. Strategy profitable up to 23.3x current fees.
- **Consequence**: Minimal fee impact; maker fee = 0%, settlement fee = 0%

## ADR-008: Sigmoid YES Price Model
- **Date**: 2026-02-08
- **Status**: Accepted
- **Context**: Need to model YES/NO token prices from BTC cumulative return
- **Decision**: prob = 1 / (1 + exp(-cum_return / 0.07)), sensitivity 0.07 matches backtest
- **Consequence**: Smooth price curve, avoids token price noise that triggers false stops

## ADR-009: TOML Config over YAML/JSON
- **Date**: 2026-02-08
- **Status**: Accepted
- **Context**: Strategy parameters need human-readable, comment-friendly format
- **Decision**: TOML for all config — native Python support via tomllib (3.11+)
- **Consequence**: Simple, readable, no external dependencies

## ADR-010: Local-First Memory Stack
- **Date**: 2026-02-08
- **Status**: Accepted
- **Context**: Need persistent Claude Code memory without cloud/paid services
- **Decision**: memory-bank MCP (project markdown) + mcp-knowledge-graph (JSONL)
- **Consequence**: All memory is local files, git-backupable, zero cost
