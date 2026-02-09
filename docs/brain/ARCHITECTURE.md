# MVHE System Architecture

## Strategy: Momentum Confirmation on BTC 15m Candle Markets
Follow BTC momentum within 15-minute windows on Polymarket prediction markets.
Buy YES/NO tokens based on cumulative return direction, hold to binary settlement ($1/$0).
Validated: 89.4% accuracy on 2,471 real Polymarket trades, $20.49 net EV/trade.

## High-Level Data Flow
```
Binance WS (BTCUSDT 1m) → Collect into 15-min windows ─────────────────┐
                                                                         │
PolymarketScanner (Gamma API) → Active market discovery ────────────────┤
                                                                         ├→ MomentumConfirmationStrategy
Sigmoid model: cum_return → YES/NO price ───────────────────────────────┤     ↓
                                                                         │  PositionSizer (quarter-Kelly)
                                                                         │     ↓
                                                                         │  CostCalculator (dynamic fees)
                                                                         │     ↓
                                                                         │  RiskManager (5-point gate)
                                                                         │     ↓
                                                                         └→ ExecutionBridge → Paper/Live
                                                                              ↓
                                                                           AuditTrail
```

## Production Bot Flow (src/bot.py)
```
BotOrchestrator._run_swing_loop():
  1. BinanceWSFeed.get_candles() → latest 1m candle
  2. Window boundary detection (15-min alignment)
  3. At window start: PolymarketScanner.scan_active_markets()
  4. Compute cumulative BTC return from window open
  5. Sigmoid model: cum_return → YES token price (sensitivity 0.07)
  6. MomentumConfirmationStrategy.generate_signals()
  7. For ENTRY signals:
     a. PositionSizer.calculate_binary() → Kelly-sized position
     b. CostCalculator.calculate_binary() → fee estimate
     c. RiskManager.check_order() → approved/rejected
     d. ExecutionBridge.submit_order() → paper or live fill
  8. Minute 14: force-close (resolution guard)
  9. Window boundary: reset state
```

## Primary Strategy: MomentumConfirmationStrategy

**Core insight**: BTC 15m candles are momentum-driven, not mean-reverting.

**Tiered entry (validated on 2,471 real Polymarket trades):**
- Minute 8: enter if |cum_return| > 0.10% (91.5% accuracy)
- Minute 9: enter if |cum_return| > 0.08% (87.5% accuracy)
- Minute 10: enter if |cum_return| > 0.05% (86.0% accuracy)
- First matching tier wins per window

**Exit: Hold to settlement** (binary $1.00 correct / $0.00 wrong)
- No stop loss, no profit target — stops destroy winning trades on sigmoid noise
- Resolution guard exits at minute 14

**Position sizing: Quarter-Kelly**
- Formula: f* = (p - P) / (1 - P), multiplied by 0.25
- Capped at 2% of balance, 10% of market volume
- Ruin probability: 0.00%, max drawdown <10% at 99th percentile

**Fee model: Polymarket dynamic taker**
- fee = position_size * 0.25 * P^2 * (1-P)^2
- Average $0.88/trade, 3.1% of gross profit

## Build Phases — ALL COMPLETE (553 tests)
| Phase | Status | Description |
|-------|--------|-------------|
| 0. Foundation | **COMPLETE** | Models, config, base strategy, protocols, CLI |
| 1. Data Layer | **COMPLETE** (1,294 LOC) | Binance WS, Polymarket client/WS/scanner, aggregator, DB/cache |
| 2. Signal Engine | **COMPLETE** (767 LOC) | Trend analyzer, threshold, book analyzer, exit manager |
| 3. Risk Management | **COMPLETE** (880 LOC, 95%+ cov) | Risk manager, cost calc, Kelly sizer, kill switch, portfolio |
| 4. Execution | **COMPLETE** (603 LOC) | Order manager, paper trader, bridge, circuit breaker, audit |
| 5. Strategy + Bot | **COMPLETE** | MomentumConfirmation, FalseSentiment, bot orchestrator, CLI |
| 6. Backtesting | **COMPLETE** (1,009 LOC) | Event-driven engine, simulator, metrics, validator |
| 7. Validation | **COMPLETE** | 2-year backtest, real Polymarket data, Kelly sizing, fee model |
| 8. Engine Integration | **COMPLETE** | All components wired in production bot loop |

## Service Boundaries

### 1. Data Pipeline (`src/data/`) — 1,294 LOC
- `BinanceWSFeed` — BTCUSDT 1m kline stream, reconnection with exponential backoff
- `PolymarketClient` — REST wrapper with rate limiting, credential masking
- `PolymarketWSFeed` — subscribe to market channel, emit MarketState + OrderBookSnapshot
- `PolymarketScanner` — Gamma API discovery, filters "Bitcoin Up or Down" markets, 5s cache
- `CandleAggregator` — aggregate ticks into OHLCV, rolling buffer
- `MarketResolver` — find active BTC 15m markets, extract token IDs
- `TimeseriesDB` — TimescaleDB async wrapper with hypertable creation
- `StateCache` — Redis wrapper with `mvhe:` prefix

### 2. Signal Engine (`src/engine/`) — 767 LOC
- `TrendAnalyzer` — last 5 BTC candles → direction/strength/momentum
- `DynamicThreshold` — base 0.59 + time scaling
- `BookAnalyzer` — depth analysis + spoofing detection (30s window, 50% cancel rate)
- `LiquidityFilter` — hour-of-day volume/spread filtering
- `MarketSpeedTracker` — correction speed measurement
- `ExitManager` — 6 exit types (profit target, trailing, hard stop, max time, resolution, force)
- `ConfidenceScorer` — multi-factor weighted scoring
- `FalseSentimentSignalGenerator` — orchestrator (used by FalseSentimentStrategy)

### 3. Strategy Layer (`src/strategies/`)
- `BaseStrategy` — abstract with REQUIRED_PARAMS, generate_signals(), on_fill(), on_cancel()
- `MomentumConfirmationStrategy` — **PRIMARY** — tiered entry, hold-to-settlement (35 tests)
- `FalseSentimentStrategy` — legacy contrarian approach (superseded but functional)

### 4. Risk Management (`src/risk/`) — 880 LOC, 95%+ coverage
- `RiskManager` — pre-trade gate with 5 checks (kill switch, stop loss, position size, drawdown, book impact)
- `CostCalculator` — Polymarket dynamic fee curve: 0.25 * P^2 * (1-P)^2 + 5 bps slippage
- `PositionSizer` — Binary Kelly: f* = (p-P)/(1-P) * kelly_multiplier, capped at max_position_pct
- `TimeGuard` — configurable entry/exit minute bounds
- `KillSwitch` — 5% daily drawdown circuit breaker, Redis-persisted
- `Portfolio` — open/closed positions, drawdown tracking, P&L

### 5. Execution (`src/execution/`) — 603 LOC
- `OrderManager` — submit, cancel, track state machine
- `CircuitBreaker` — halt after N consecutive failures
- `RateLimiter` — token bucket per exchange
- `PaperTrader` — simulated fills with 5 bps slippage
- `ExecutionBridge` — routes to PaperTrader (paper) or OrderManager (live)
- `AuditTrail` — append-only logging of all order events

### 6. Backtesting (`src/backtesting/`) — 1,009 LOC
- `Engine` — event-driven simulator, no lookahead bias
- `DataLoader` — CSV/DB loading, gap/duplicate detection
- `FillSimulator` — realistic slippage model
- `Metrics` — Sharpe, Sortino, max drawdown, Calmar, win rate, profit factor, total PnL, trade count
- `Validator` — bootstrap CI, Monte Carlo permutation, overfitting detection
- `Reporter` — JSON output, equity curve data

### 7. Bot + CLI
- `src/bot.py` — BotOrchestrator: async event loop with swing loop for 1m→15m trading
- `src/cli.py` — argparse: --paper, --live (requires confirmation), --backtest, --strategy

## Data Models
All in `src/models/` using Pydantic v2 with Decimal precision:
- `Candle` — OHLCV with direction/body_size/range_size properties
- `OrderBookLevel` / `OrderBookSnapshot` — depth with best_bid/best_ask/spread
- `MarketState` — Polymarket market state with dominant_side/minutes_elapsed
- `Position` — entry/exit tracking with realized P&L
- `Order` / `Fill` — order lifecycle with fill percentage tracking
- `Signal` / `Confidence` — trading signal with confidence breakdown
- `TrendResult` — BTC trend analysis output

## Config (config/default.toml)
```toml
[strategy.momentum_confirmation]
entry_minute_start = 8
entry_minute_end = 10
hold_to_settlement = true
min_confidence = 0.70
estimated_win_prob = 0.884

[[strategy.momentum_confirmation.entry_tiers]]
minute = 8
threshold_pct = 0.10

[risk]
max_position_pct = 0.02
max_daily_drawdown_pct = 0.05
kelly_multiplier = 0.25

[risk.fees]
fee_model = "polymarket_dynamic"
fee_constant = 0.25
slippage_bps = 5

[bot]
tick_interval_seconds = 5

[paper]
initial_balance = 10000

[binance]
ws_url = "wss://stream.binance.com:9443/ws/btcusdt@kline_1m"

[polymarket]
clob_url = "https://clob.polymarket.com"
chain_id = 137
```

## Exchange Connectivity
- **Binance**: BTCUSDT 1m kline WebSocket (aggregated into 15m windows)
- **Polymarket**: Gamma API (market discovery) + CLOB REST API (trading)
- Both via official APIs with rate limit tracking

## Key ADRs
- ADR-001: Python-first execution (Rust deferred to optimization phase)
- ADR-002: TimescaleDB for time-series with SQL
- ADR-003: Momentum-following beats contrarian (data showed 14.5pp advantage)
- ADR-004: Hold-to-settlement optimal for binary markets (stops destroy winners)
- ADR-005: Tiered entry thresholds by minute (8@0.10%, 9@0.08%, 10@0.05%)
- ADR-006: Quarter-Kelly position sizing (0% ruin, <10% max DD at 99th pctl)
- ADR-007: Polymarket dynamic fee model (avg $0.88/trade, 3.1% of gross)
- ADR-008: Sigmoid YES price model (sensitivity 0.07 matches backtest)
