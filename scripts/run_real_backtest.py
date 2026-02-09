"""Run backtests against REAL BTC 15m data across three months.

Loads three monthly CSV files, merges them, and runs two strategies:
  1. SimpleTrendStrategy — 3+ consecutive green/red candles
  2. MeanReversionStrategy — contrarian entry after 5+ same-direction candles

Each strategy gets full statistical validation (bootstrap + permutation)
and results are saved as JSON reports.
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
CSV_PATHS = [
    PROJECT_ROOT / "data" / "btc_15m_month1.csv",
    PROJECT_ROOT / "data" / "btc_15m_month2.csv",
    PROJECT_ROOT / "data" / "btc_15m_month3.csv",
]
REPORT_TREND_PATH = PROJECT_ROOT / "data" / "real_backtest_trend.json"
REPORT_MEANREV_PATH = PROJECT_ROOT / "data" / "real_backtest_meanrev.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INITIAL_BALANCE = Decimal("10000")
CANDLE_SECONDS = 15 * 60  # 900 s per 15-minute candle
TREND_LOOKBACK = 5  # candles to detect trend (SimpleTrend)
MEANREV_LOOKBACK = 7  # candles to detect overextension (MeanReversion)
TREND_WINDOW = 10  # window for BTC trend factor


# ===================================================================
# Permissive Risk Manager
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
        return RiskDecision(
            approved=True,
            reason="permissive - approved",
            max_size=Decimal("10"),
        )


# ===================================================================
# Strategy 1: Simple Trend Following
# ===================================================================
class SimpleTrendStrategy:
    """Generates ENTRY signals on 3+ consecutive green/red candles.

    * 3+ consecutive green candles  -> BUY (Side.YES)
    * 3+ consecutive red candles    -> contrarian BUY (Side.YES)
    """

    strategy_id: str = "simple_trend"

    def generate_signals(
        self,
        market_state: MarketState,
        orderbook: OrderBookSnapshot,
        context: dict[str, Any],
    ) -> list[Signal]:
        candles = context.get("candles", [])
        if len(candles) < TREND_LOOKBACK:
            return []

        recent = candles[-TREND_LOOKBACK:]

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
                    strength=Decimal(str(round(green_run / TREND_LOOKBACK, 4))),
                    confidence=Confidence(
                        trend_strength=green_run / TREND_LOOKBACK,
                        overall=green_run / TREND_LOOKBACK,
                    ),
                    entry_price=entry_price,
                    stop_loss=entry_price - Decimal("0.04"),
                    take_profit=entry_price + Decimal("0.05"),
                    metadata={"trigger": f"{green_run}_green"},
                )
            )
        elif red_run >= 3:
            signals.append(
                Signal(
                    strategy_id=self.strategy_id,
                    market_id=market_state.market_id,
                    signal_type=SignalType.ENTRY,
                    direction=Side.YES,
                    strength=Decimal(str(round(red_run / TREND_LOOKBACK, 4))),
                    confidence=Confidence(
                        trend_strength=red_run / TREND_LOOKBACK,
                        overall=red_run / TREND_LOOKBACK,
                    ),
                    entry_price=entry_price,
                    stop_loss=entry_price - Decimal("0.04"),
                    take_profit=entry_price + Decimal("0.05"),
                    metadata={"trigger": f"{red_run}_red_contrarian"},
                )
            )

        return signals


# ===================================================================
# Strategy 2: Mean Reversion
# ===================================================================
class MeanReversionStrategy:
    """Detects overextended moves (5+ candles same direction) and enters
    contrarian positions expecting a reversion to the mean.

    * 5+ consecutive green candles  -> SELL signal (Side.NO, expect pullback)
    * 5+ consecutive red candles    -> BUY signal (Side.YES, expect bounce)
    """

    strategy_id: str = "mean_reversion"

    def generate_signals(
        self,
        market_state: MarketState,
        orderbook: OrderBookSnapshot,
        context: dict[str, Any],
    ) -> list[Signal]:
        candles = context.get("candles", [])
        if len(candles) < MEANREV_LOOKBACK:
            return []

        recent = candles[-MEANREV_LOOKBACK:]

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

        if green_run >= 5:
            # Overextended up -> contrarian SELL (bet NO)
            signals.append(
                Signal(
                    strategy_id=self.strategy_id,
                    market_id=market_state.market_id,
                    signal_type=SignalType.ENTRY,
                    direction=Side.NO,
                    strength=Decimal(str(round(green_run / MEANREV_LOOKBACK, 4))),
                    confidence=Confidence(
                        trend_strength=green_run / MEANREV_LOOKBACK,
                        overall=green_run / MEANREV_LOOKBACK,
                    ),
                    entry_price=entry_price,
                    stop_loss=entry_price + Decimal("0.05"),
                    take_profit=entry_price - Decimal("0.04"),
                    metadata={"trigger": f"{green_run}_green_overextended"},
                )
            )
        elif red_run >= 5:
            # Overextended down -> contrarian BUY (bet YES)
            signals.append(
                Signal(
                    strategy_id=self.strategy_id,
                    market_id=market_state.market_id,
                    signal_type=SignalType.ENTRY,
                    direction=Side.YES,
                    strength=Decimal(str(round(red_run / MEANREV_LOOKBACK, 4))),
                    confidence=Confidence(
                        trend_strength=red_run / MEANREV_LOOKBACK,
                        overall=red_run / MEANREV_LOOKBACK,
                    ),
                    entry_price=entry_price,
                    stop_loss=entry_price - Decimal("0.05"),
                    take_profit=entry_price + Decimal("0.04"),
                    metadata={"trigger": f"{red_run}_red_overextended"},
                )
            )

        return signals


# ===================================================================
# Market-state / order-book generators
# ===================================================================

def _btc_trend_factor(candles: list[Any], idx: int, window: int = TREND_WINDOW) -> float:
    """Compute a [-1, 1] trend factor from recent candle returns."""
    start = max(0, idx - window)
    subset = candles[start : idx + 1]
    if len(subset) < 2:
        return 0.0
    ret = float(subset[-1].close - subset[0].open) / float(subset[0].open)
    # Clamp to [-1, 1]
    return max(-1.0, min(1.0, ret * 50))


def build_market_states(candles: list[Any]) -> list[MarketState]:
    """Create a MarketState for each candle.

    * yes_price correlates with BTC trend factor (range 0.40 - 0.80).
    * no_price = 1 - yes_price.
    * time_remaining_seconds cycles through candle lifetime.
    """
    states: list[MarketState] = []
    for idx, candle in enumerate(candles):
        trend = _btc_trend_factor(candles, idx)
        # Map trend [-1,1] -> yes_price [0.40, 0.80]
        yes_price = Decimal(str(round(0.60 + 0.20 * trend, 4)))
        yes_price = max(Decimal("0.40"), min(Decimal("0.80"), yes_price))
        no_price = Decimal("1") - yes_price

        # Cycle time remaining
        time_remaining = CANDLE_SECONDS - (idx % (CANDLE_SECONDS // 15)) * 15
        time_remaining = max(0, min(CANDLE_SECONDS, time_remaining))

        states.append(
            MarketState(
                market_id="btc-15m-real",
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


def build_orderbooks(candles: list[Any]) -> list[OrderBookSnapshot]:
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
                market_id="btc-15m-real",
            )
        )
    return books


# ===================================================================
# Report builder
# ===================================================================

def build_report(result: Any, validation: Any) -> dict[str, Any]:
    """Build a report dict from BacktestResult and ValidationResult."""
    return {
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


# ===================================================================
# Comparison table
# ===================================================================

def print_comparison(
    trend_report: dict[str, Any],
    meanrev_report: dict[str, Any],
) -> None:
    """Print a side-by-side comparison of two strategy reports."""
    print("\n" + "=" * 70)
    print("  STRATEGY COMPARISON")
    print("=" * 70)

    header = f"  {'Metric':<25} {'SimpleTrend':>18} {'MeanReversion':>18}"
    print(header)
    print("  " + "-" * 63)

    rows = [
        ("Total Trades", "total_trades", False, False),
        ("Total Return", "total_return_pct", True, True),
        ("Sharpe Ratio", "sharpe_ratio", False, True),
        ("Sortino Ratio", "sortino_ratio", False, True),
        ("Max Drawdown", "max_drawdown_pct", True, True),
        ("Calmar Ratio", "calmar_ratio", False, True),
        ("Win Rate", "win_rate", True, True),
        ("Profit Factor", "profit_factor", False, True),
    ]

    tm = trend_report.get("metrics", {})
    mm = meanrev_report.get("metrics", {})

    for label, key, as_pct, is_metric in rows:
        if is_metric:
            tv = tm.get(key)
            mv = mm.get(key)
            if tv is not None:
                tv_str = f"{float(tv) * 100:.2f}%" if as_pct else f"{float(tv):.4f}"
            else:
                tv_str = "N/A"
            if mv is not None:
                mv_str = f"{float(mv) * 100:.2f}%" if as_pct else f"{float(mv):.4f}"
            else:
                mv_str = "N/A"
        else:
            tv_str = str(trend_report.get(key, "N/A"))
            mv_str = str(meanrev_report.get(key, "N/A"))

        print(f"  {label:<25} {tv_str:>18} {mv_str:>18}")

    # Equity
    print(f"  {'Equity Start':<25} {'$' + trend_report.get('equity_start', '?'):>18} {'$' + meanrev_report.get('equity_start', '?'):>18}")
    print(f"  {'Equity End':<25} {'$' + trend_report.get('equity_end', '?'):>18} {'$' + meanrev_report.get('equity_end', '?'):>18}")

    # Validation
    print("\n  --- Validation ---")
    for name, report in [("SimpleTrend", trend_report), ("MeanReversion", meanrev_report)]:
        val = report.get("validation", {})
        ci = val.get("sharpe_ci_95", [0, 0])
        p_val = val.get("p_value", 1.0)
        sig = val.get("is_significant", False)
        print(f"  {name:<20} Sharpe CI=[{ci[0]:.4f}, {ci[1]:.4f}]  p={p_val:.4f}  significant={'Yes' if sig else 'No'}")

    print("=" * 70)


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    print("=" * 70)
    print("  MVHE Real-Data Backtest Runner")
    print("  3-Month BTC 15m Dataset")
    print("=" * 70)

    # 1. Load and merge data from all three CSVs
    loader = DataLoader()
    all_candles = []
    for csv_path in CSV_PATHS:
        candles = loader.load_csv(csv_path)
        print(f"  Loaded {len(candles):>5} candles from {csv_path.name}")
        all_candles.extend(candles)

    # Sort by timestamp and deduplicate
    all_candles.sort(key=lambda c: c.timestamp)
    seen_ts = set()
    unique_candles = []
    for c in all_candles:
        if c.timestamp not in seen_ts:
            seen_ts.add(c.timestamp)
            unique_candles.append(c)
    all_candles = unique_candles

    print(f"\n  Total candles: {len(all_candles)}")
    print(f"  Date range:   {all_candles[0].timestamp} -> {all_candles[-1].timestamp}")

    # 2. Build shared market states and order books
    print("\nBuilding market states and order books...")
    market_states = build_market_states(all_candles)
    orderbooks = build_orderbooks(all_candles)
    print(f"  Market states: {len(market_states)}")
    print(f"  Order books  : {len(orderbooks)}")

    # Shared risk manager and reporter
    risk_manager = PermissiveRiskManager()
    reporter = BacktestReporter()

    # ---------------------------------------------------------------
    # Strategy A: SimpleTrendStrategy
    # ---------------------------------------------------------------
    print("\n" + "-" * 70)
    print("  Running Strategy A: SimpleTrendStrategy")
    print("-" * 70)

    trend_strategy = SimpleTrendStrategy()
    trend_engine = BacktestEngine(initial_balance=INITIAL_BALANCE)
    trend_result = trend_engine.run(
        candles=all_candles,
        market_states=market_states,
        orderbooks=orderbooks,
        strategy=trend_strategy,
        risk_manager=risk_manager,
    )

    trend_validator = StatisticalValidator(
        n_bootstrap=1000,
        n_permutations=1000,
        rng_seed=42,
    )
    trend_validation = trend_validator.validate(trend_result.trades, trend_result.equity_curve)
    trend_report = build_report(trend_result, trend_validation)

    summary_a = reporter.print_summary(trend_report)
    print(summary_a)
    reporter.save_report(trend_report, REPORT_TREND_PATH)
    print(f"  Report saved to {REPORT_TREND_PATH}")
    print(f"  Trades: {len(trend_result.trades)}")

    # ---------------------------------------------------------------
    # Strategy B: MeanReversionStrategy
    # ---------------------------------------------------------------
    print("\n" + "-" * 70)
    print("  Running Strategy B: MeanReversionStrategy")
    print("-" * 70)

    meanrev_strategy = MeanReversionStrategy()
    meanrev_engine = BacktestEngine(initial_balance=INITIAL_BALANCE)
    meanrev_result = meanrev_engine.run(
        candles=all_candles,
        market_states=market_states,
        orderbooks=orderbooks,
        strategy=meanrev_strategy,
        risk_manager=risk_manager,
    )

    meanrev_validator = StatisticalValidator(
        n_bootstrap=1000,
        n_permutations=1000,
        rng_seed=42,
    )
    meanrev_validation = meanrev_validator.validate(meanrev_result.trades, meanrev_result.equity_curve)
    meanrev_report = build_report(meanrev_result, meanrev_validation)

    summary_b = reporter.print_summary(meanrev_report)
    print(summary_b)
    reporter.save_report(meanrev_report, REPORT_MEANREV_PATH)
    print(f"  Report saved to {REPORT_MEANREV_PATH}")
    print(f"  Trades: {len(meanrev_result.trades)}")

    # ---------------------------------------------------------------
    # Side-by-side comparison
    # ---------------------------------------------------------------
    print_comparison(trend_report, meanrev_report)

    print("\nDone.")


if __name__ == "__main__":
    main()
