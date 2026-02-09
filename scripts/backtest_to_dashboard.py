"""Run backtest and insert results into ClickHouse for dashboard display."""

from __future__ import annotations

import json
import math
import uuid
from datetime import datetime, timezone
from decimal import Decimal

import clickhouse_connect

from src.backtesting.data_loader import DataLoader
from src.backtesting.engine import BacktestEngine
from src.backtesting.reporter import BacktestReporter
from src.config.loader import ConfigLoader
from src.core.logging import get_logger
from src.models.market import MarketState, OrderBookSnapshot
from src.risk.kill_switch import KillSwitch
from src.risk.risk_manager import RiskManager
from src.strategies import registry

log = get_logger(__name__)


def main() -> int:
    import contextlib
    from pathlib import Path

    with contextlib.suppress(ImportError):
        import src.strategies.false_sentiment  # noqa: F401
    with contextlib.suppress(ImportError):
        import src.strategies.reversal_catcher  # noqa: F401

    strategy_name = "momentum_confirmation"
    config = ConfigLoader(config_dir="config")
    config.load()
    config.load_strategy(strategy_name)
    config.validate_ranges()

    # Load data
    loader = DataLoader()
    data_path = Path("data")
    csv_file = data_path / "btc_15m_backtest.csv"
    candles = loader.load_csv(csv_file)
    print(f"Loaded {len(candles)} candles from {csv_file}")

    # Synthesise market states
    market_states: list[MarketState] = []
    orderbooks: list[OrderBookSnapshot] = []
    sigmoid_sensitivity = 0.07

    for i, candle in enumerate(candles):
        cum_return = float((candle.close - candles[0].open) / candles[0].open)
        yes_prob = 1.0 / (1.0 + math.exp(-cum_return / sigmoid_sensitivity))
        yes_price = Decimal(str(round(yes_prob, 6)))
        no_price = Decimal("1") - yes_price

        market_states.append(
            MarketState(
                market_id="backtest-btc-15m",
                yes_price=yes_price,
                no_price=no_price,
                time_remaining_seconds=max(0, (14 - (i % 15)) * 60),
            )
        )
        orderbooks.append(
            OrderBookSnapshot(timestamp=candle.timestamp, market_id="backtest-btc-15m")
        )

    # Create strategy and risk manager
    strat = registry.create(strategy_name, config)
    max_position_pct = Decimal(str(config.get("risk.max_position_pct", 0.02)))
    max_daily_drawdown_pct = Decimal(str(config.get("risk.max_daily_drawdown_pct", 0.05)))
    kill_switch = KillSwitch(max_daily_drawdown_pct)
    risk_mgr = RiskManager(
        max_position_pct=max_position_pct,
        max_daily_drawdown_pct=max_daily_drawdown_pct,
        kill_switch=kill_switch,
    )

    initial_balance = Decimal(str(config.get("paper.initial_balance", 10000)))
    engine = BacktestEngine(initial_balance=initial_balance)

    # Run backtest
    result = engine.run(
        candles=candles,
        market_states=market_states,
        orderbooks=orderbooks,
        strategy=strat,
        risk_manager=risk_mgr,
    )

    # Print summary
    reporter = BacktestReporter()
    result_dict = result.model_dump()
    reporter.print_summary(result_dict)

    trades = result.trades
    print(f"\n{'='*60}")
    print(f"Inserting {len(trades)} trades into ClickHouse...")

    if not trades:
        print("No trades to insert.")
        return 0

    # Clear old backtest trades first
    ch = clickhouse_connect.get_client(host="localhost", port=8123, database="mvhe")
    ch.command("ALTER TABLE mvhe.trades DELETE WHERE strategy = 'momentum_confirmation' AND market_id = 'backtest-btc-15m'")

    # Insert trades into ClickHouse
    rows = []
    for t in trades:
        trade_id = f"bt-{uuid.uuid4().hex[:8]}"
        entry_time = t.get("entry_time")
        exit_time = t.get("exit_time", entry_time)
        entry_price = float(t.get("entry_price", 0))
        exit_price = float(t.get("exit_price", 0))
        quantity = float(t.get("quantity", 0))
        pnl = float(t.get("pnl", 0))
        entry_fee = float(t.get("entry_fee", 0))
        side = t.get("side", "YES")
        exit_reason = t.get("exit_reason", "unknown")
        # Window minute from entry time
        minute_in_window = entry_time.minute % 15 if entry_time else 0
        cum_return = float(t.get("slippage", 0))  # approximate

        rows.append([
            trade_id,
            "backtest-btc-15m",
            "momentum_confirmation",
            side,
            entry_price,
            exit_price,
            quantity,
            pnl,
            entry_fee,
            entry_time,
            exit_time or entry_time,
            exit_reason,
            minute_in_window,
            cum_return,
            0.75,  # default confidence
        ])

    ch.insert(
        "trades",
        rows,
        column_names=[
            "trade_id", "market_id", "strategy", "direction",
            "entry_price", "exit_price", "position_size", "pnl",
            "fee_cost", "entry_time", "exit_time", "exit_reason",
            "window_minute", "cum_return_pct", "confidence",
        ],
    )

    print(f"Inserted {len(rows)} trades into ClickHouse mvhe.trades")

    # Also insert audit events for the trades
    audit_rows = []
    for t in trades:
        entry_time = t.get("entry_time")
        exit_time = t.get("exit_time", entry_time)
        entry_price = float(t.get("entry_price", 0))
        exit_price = float(t.get("exit_price", 0))
        pnl = float(t.get("pnl", 0))
        side = t.get("side", "YES")

        order_id = f"bt-ord-{uuid.uuid4().hex[:8]}"
        audit_rows.append([
            f"bt-evt-{uuid.uuid4().hex[:8]}", order_id,
            "BACKTEST_ENTRY", "backtest-btc-15m", "momentum_confirmation",
            f"Entered {side} at {entry_price:.4f}", entry_time,
        ])
        audit_rows.append([
            f"bt-evt-{uuid.uuid4().hex[:8]}", order_id,
            "BACKTEST_EXIT", "backtest-btc-15m", "momentum_confirmation",
            f"Exited at {exit_price:.4f}, P&L: {pnl:+.4f}", exit_time or entry_time,
        ])

    ch.insert(
        "audit_events",
        audit_rows,
        column_names=[
            "event_id", "order_id", "event_type", "market_id",
            "strategy", "details", "timestamp",
        ],
    )

    print(f"Inserted {len(audit_rows)} audit events into ClickHouse mvhe.audit_events")
    print("\nDashboard will now show backtest results!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
