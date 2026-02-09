"""Run a backtest against generated BTC 15m data.

Loads CSV via DataLoader, constructs MarketState and OrderBookSnapshot
objects for each candle, uses a simple trend-following strategy, and
runs the BacktestEngine with a permissive RiskManager.  Results are
printed and saved to ``data/backtest_report.json``.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path so ``src`` package resolves.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtesting.data_loader import DataLoader
from src.backtesting.engine import BacktestEngine
from src.backtesting.reporter import BacktestReporter
from src.backtesting.validator import StatisticalValidator
from src.interfaces import RiskDecision
from src.models.market import (
    MarketState,
    OrderBookLevel,
    OrderBookSnapshot,
    Side,
)
from src.models.signal import Confidence, Signal, SignalType

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CSV_PATH = PROJECT_ROOT / "data" / "btc_15m_backtest.csv"
REPORT_PATH = PROJECT_ROOT / "data" / "backtest_report.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INITIAL_BALANCE = Decimal("10000")
CANDLE_SECONDS = 15 * 60  # 900 s per 15-minute candle
LOOKBACK = 5  # candles to detect trend


# ===================================================================
# Permissive Risk Manager — approves everything with a stop-loss
# ===================================================================
class PermissiveRiskManager:
    """Risk manager that approves all orders provided they have a stop-loss."""

    def has_stop_loss(self, signal: Signal) -> bool:
        return signal.stop_loss is not None

    def check_order(
        self,
        signal: Signal,
        position_size: Decimal,
        current_drawdown: Decimal,
    ) -> RiskDecision:
        if not self.has_stop_loss(signal):
            return RiskDecision(
                approved=False,
                reason="missing stop_loss",
            )
        # Approve with a fixed max size of 10 units
        return RiskDecision(
            approved=True,
            reason="permissive — approved",
            max_size=Decimal("10"),
        )


# ===================================================================
# Simple trend-following strategy
# ===================================================================
class SimpleTrendStrategy:
    """Lightweight strategy that generates ENTRY signals when a
    consecutive-candle trend is detected.

    * 3+ consecutive green candles  -> BUY (Side.YES)
    * 3+ consecutive red candles    -> SELL (Side.NO)
    * Otherwise                      -> no signal
    """

    strategy_id: str = "simple_trend"

    def generate_signals(
        self,
        market_state: MarketState,
        orderbook: OrderBookSnapshot,
        context: dict[str, Any],
    ) -> list[Signal]:
        candles = context.get("candles", [])
        if len(candles) < LOOKBACK:
            return []

        recent = candles[-LOOKBACK:]

        green_run = 0
        red_run = 0
        for c in recent:
            if c.close > c.open:
                green_run += 1
                red_run = 0
            elif c.close < c.open:
                red_run += 1
                green_run = 0
            else:
                green_run = 0
                red_run = 0

        entry_price = market_state.yes_price
        signals: list[Signal] = []

        if green_run >= 3:
            signals.append(
                Signal(
                    strategy_id=self.strategy_id,
                    market_id=market_state.market_id,
                    signal_type=SignalType.ENTRY,
                    direction=Side.YES,
                    strength=Decimal(str(round(green_run / LOOKBACK, 4))),
                    confidence=Confidence(
                        trend_strength=green_run / LOOKBACK,
                        overall=green_run / LOOKBACK,
                    ),
                    entry_price=entry_price,
                    stop_loss=entry_price - Decimal("0.04"),
                    take_profit=entry_price + Decimal("0.05"),
                    metadata={"trigger": f"{green_run}_green"},
                )
            )
        elif red_run >= 3:
            # Contrarian: bet YES after sell-off (mean reversion)
            signals.append(
                Signal(
                    strategy_id=self.strategy_id,
                    market_id=market_state.market_id,
                    signal_type=SignalType.ENTRY,
                    direction=Side.YES,
                    strength=Decimal(str(round(red_run / LOOKBACK, 4))),
                    confidence=Confidence(
                        trend_strength=red_run / LOOKBACK,
                        overall=red_run / LOOKBACK,
                    ),
                    entry_price=entry_price,
                    stop_loss=entry_price - Decimal("0.04"),
                    take_profit=entry_price + Decimal("0.05"),
                    metadata={"trigger": f"{red_run}_red_contrarian"},
                )
            )

        return signals


# ===================================================================
# Market-state / order-book generators
# ===================================================================

def _btc_trend_factor(candles: list[Any], idx: int, window: int = 10) -> float:
    """Compute a [-1, 1] trend factor from recent candle returns."""
    start = max(0, idx - window)
    subset = candles[start : idx + 1]
    if len(subset) < 2:
        return 0.0
    ret = float(subset[-1].close - subset[0].open) / float(subset[0].open)
    # Clamp to [-1, 1]
    return max(-1.0, min(1.0, ret * 50))


def build_market_states(
    candles: list[Any],
) -> list[MarketState]:
    """Create a MarketState for each candle.

    * yes_price correlates with BTC trend factor (range 0.40 – 0.80).
    * no_price ~ 1 - yes_price (with small noise).
    * time_remaining_seconds cycles 900 -> 0 each candle.
    """
    states: list[MarketState] = []
    for idx, candle in enumerate(candles):
        trend = _btc_trend_factor(candles, idx)
        # Map trend [-1,1] -> yes_price [0.40, 0.80]
        yes_price = Decimal(str(round(0.60 + 0.20 * trend, 4)))
        yes_price = max(Decimal("0.40"), min(Decimal("0.80"), yes_price))
        no_price = Decimal("1") - yes_price

        # Cycle time remaining: simulate lifecycle within candle
        time_remaining = CANDLE_SECONDS - (idx % (CANDLE_SECONDS // 15)) * 15
        time_remaining = max(0, min(CANDLE_SECONDS, time_remaining))

        states.append(
            MarketState(
                market_id="btc-15m-backtest",
                yes_price=yes_price,
                no_price=no_price,
                yes_bid=yes_price - Decimal("0.01"),
                yes_ask=yes_price + Decimal("0.01"),
                no_bid=no_price - Decimal("0.01"),
                no_ask=no_price + Decimal("0.01"),
                time_remaining_seconds=time_remaining,
                candle_start_time=candle.timestamp,
                question="Will BTC 15m candle close green?",
            )
        )
    return states


def build_orderbooks(
    candles: list[Any],
) -> list[OrderBookSnapshot]:
    """Create a synthetic OrderBookSnapshot per candle."""
    books: list[OrderBookSnapshot] = []
    for candle in candles:
        mid = float(candle.close)
        spread_pct = 0.001  # 10 bps spread
        bid_price = Decimal(str(round(mid * (1 - spread_pct / 2), 4)))
        ask_price = Decimal(str(round(mid * (1 + spread_pct / 2), 4)))
        depth = candle.volume * Decimal("0.3")  # 30% of candle volume

        books.append(
            OrderBookSnapshot(
                bids=[OrderBookLevel(price=bid_price, size=depth)],
                asks=[OrderBookLevel(price=ask_price, size=depth)],
                timestamp=candle.timestamp,
                market_id="btc-15m-backtest",
            )
        )
    return books


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    print("=" * 60)
    print("  MVHE Backtest Runner")
    print("=" * 60)

    # 1. Load data
    loader = DataLoader()
    candles = loader.load_csv(CSV_PATH)
    print(f"\nLoaded {len(candles)} candles from {CSV_PATH.name}")
    print(f"  Range: {candles[0].timestamp} -> {candles[-1].timestamp}")

    # 2. Build market states & order books
    market_states = build_market_states(candles)
    orderbooks = build_orderbooks(candles)
    print(f"  Market states: {len(market_states)}")
    print(f"  Order books  : {len(orderbooks)}")

    # 3. Create strategy and risk manager
    strategy = SimpleTrendStrategy()
    risk_manager = PermissiveRiskManager()

    # 4. Run backtest engine
    engine = BacktestEngine(initial_balance=INITIAL_BALANCE)
    print(f"\nRunning backtest with ${INITIAL_BALANCE} initial balance...")
    result = engine.run(
        candles=candles,
        market_states=market_states,
        orderbooks=orderbooks,
        strategy=strategy,
        risk_manager=risk_manager,
    )

    # 5. Statistical validation
    validator = StatisticalValidator(
        n_bootstrap=500,
        n_permutations=500,
        rng_seed=42,
    )
    validation = validator.validate(result.trades, result.equity_curve)

    # 6. Build report dict
    report: dict[str, Any] = {
        "metrics": {k: str(v) for k, v in result.metrics.items()},
        "validation": {
            "sharpe_ci_95": list(validation.sharpe_ci_95),
            "p_value": validation.p_value,
            "is_significant": validation.is_significant,
            "overfitting_warning": validation.overfitting_warning,
        },
        "total_trades": len(result.trades),
        "equity_start": str(result.equity_curve[0]) if result.equity_curve else "0",
        "equity_end": str(result.equity_curve[-1]) if result.equity_curve else "0",
        "trades": [
            {
                "market_id": t.get("market_id", ""),
                "side": t.get("side", ""),
                "entry_price": str(t.get("entry_price", "")),
                "exit_price": str(t.get("exit_price", "")),
                "pnl": str(t.get("pnl", "")),
                "exit_reason": t.get("exit_reason", ""),
            }
            for t in result.trades
        ],
    }

    # 7. Print summary
    reporter = BacktestReporter()
    summary = reporter.print_summary(report)
    print(summary)

    # 8. Save report
    reporter.save_report(report, REPORT_PATH)
    print(f"\nJSON report saved to {REPORT_PATH}")

    # 9. Print trade breakdown
    print(f"\n--- Trade Breakdown ({len(result.trades)} trades) ---")
    for i, t in enumerate(result.trades, 1):
        pnl = t.get("pnl", 0)
        side = t.get("side", "?")
        reason = t.get("exit_reason", "?")
        print(
            f"  #{i:3d}  {side:<4s}  "
            f"entry={float(t.get('entry_price', 0)):>10.4f}  "
            f"exit={float(t.get('exit_price', 0)):>10.4f}  "
            f"pnl={float(pnl):>+10.4f}  "
            f"reason={reason}"
        )


if __name__ == "__main__":
    main()
