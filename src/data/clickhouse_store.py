"""ClickHouse analytics store for trade history and audit events."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any

import clickhouse_connect
from clickhouse_connect.driver.client import Client

from src.core.logging import get_logger

log = get_logger(__name__)

CREATE_DB = "CREATE DATABASE IF NOT EXISTS mvhe"

CREATE_TRADES_TABLE = """
CREATE TABLE IF NOT EXISTS mvhe.trades (
    trade_id String,
    market_id String,
    strategy String,
    direction String,
    entry_price Decimal64(6),
    exit_price Decimal64(6),
    position_size Decimal64(6),
    pnl Decimal64(6),
    fee_cost Decimal64(6),
    entry_time DateTime64(3),
    exit_time DateTime64(3),
    exit_reason String,
    window_minute UInt8,
    cum_return_pct Float64,
    confidence Float64
) ENGINE = MergeTree()
ORDER BY (strategy, entry_time)
PARTITION BY toYYYYMM(entry_time)
"""

CREATE_AUDIT_EVENTS_TABLE = """
CREATE TABLE IF NOT EXISTS mvhe.audit_events (
    event_id String,
    order_id String,
    event_type String,
    market_id String,
    strategy String,
    details String,
    timestamp DateTime64(3)
) ENGINE = MergeTree()
ORDER BY (order_id, timestamp)
PARTITION BY toYYYYMM(timestamp)
"""

CREATE_DAILY_SUMMARY_TABLE = """
CREATE TABLE IF NOT EXISTS mvhe.daily_summary (
    date Date,
    strategy String,
    trade_count UInt32,
    win_count UInt32,
    total_pnl Decimal64(6),
    total_fees Decimal64(6),
    max_drawdown Float64,
    avg_position_size Decimal64(6)
) ENGINE = SummingMergeTree()
ORDER BY (strategy, date)
"""

SELECT_TRADES = """
SELECT *
FROM mvhe.trades
WHERE entry_time >= %(start)s
  AND entry_time <= %(end)s
ORDER BY entry_time DESC
"""

SELECT_DAILY_PNL = """
SELECT
    strategy,
    count() AS trade_count,
    sum(pnl) AS total_pnl,
    sum(fee_cost) AS total_fees,
    countIf(pnl > 0) AS win_count
FROM mvhe.trades
WHERE toDate(entry_time) = %(date)s
GROUP BY strategy
"""

SELECT_STRATEGY_PERFORMANCE = """
SELECT
    strategy,
    count() AS trade_count,
    countIf(pnl > 0) AS win_count,
    sum(pnl) AS total_pnl,
    sum(fee_cost) AS total_fees,
    avg(pnl) AS avg_pnl,
    max(pnl) AS best_trade,
    min(pnl) AS worst_trade,
    avg(position_size) AS avg_position_size,
    avg(confidence) AS avg_confidence
FROM mvhe.trades
WHERE strategy = %(strategy)s
GROUP BY strategy
"""


class ClickHouseStore:
    """Async ClickHouse store for analytics and audit trail.

    Wraps the synchronous ``clickhouse-connect`` client with
    ``asyncio.to_thread`` for non-blocking usage.

    Reads connection config from environment:
      - CLICKHOUSE_HOST (default: localhost)
      - CLICKHOUSE_PORT (default: 8123)
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        self._host = host or os.environ.get("CLICKHOUSE_HOST", "localhost")
        self._port = port or int(os.environ.get("CLICKHOUSE_PORT", "8123"))
        self._client: Client | None = None

    async def connect(self) -> None:
        """Establish connection and create tables if they don't exist."""
        self._client = await asyncio.to_thread(
            clickhouse_connect.get_client,
            host=self._host,
            port=self._port,
        )
        await asyncio.to_thread(self._client.command, CREATE_DB)
        await asyncio.to_thread(self._client.command, CREATE_TRADES_TABLE)
        await asyncio.to_thread(self._client.command, CREATE_AUDIT_EVENTS_TABLE)
        await asyncio.to_thread(self._client.command, CREATE_DAILY_SUMMARY_TABLE)
        log.info("clickhouse.connected", host=self._host, port=self._port)

    async def disconnect(self) -> None:
        """Clean shutdown of the ClickHouse client."""
        if self._client is not None:
            self._client.close()
            self._client = None
            log.info("clickhouse.disconnected")

    async def insert_trade(self, trade_data: dict[str, Any]) -> None:
        """Insert a completed trade record.

        Args:
            trade_data: Dict with keys matching the trades table columns.
        """
        assert self._client is not None, "Call connect() first"
        row = [
            trade_data.get("trade_id", str(uuid.uuid4())),
            trade_data["market_id"],
            trade_data.get("strategy", ""),
            trade_data.get("direction", ""),
            Decimal(str(trade_data.get("entry_price", 0))),
            Decimal(str(trade_data.get("exit_price", 0))),
            Decimal(str(trade_data.get("position_size", 0))),
            Decimal(str(trade_data.get("pnl", 0))),
            Decimal(str(trade_data.get("fee_cost", 0))),
            trade_data.get("entry_time", datetime.now(tz=timezone.utc)),
            trade_data.get("exit_time", datetime.now(tz=timezone.utc)),
            trade_data.get("exit_reason", ""),
            trade_data.get("window_minute", 0),
            float(trade_data.get("cum_return_pct", 0.0)),
            float(trade_data.get("confidence", 0.0)),
        ]
        column_names = [
            "trade_id", "market_id", "strategy", "direction",
            "entry_price", "exit_price", "position_size", "pnl", "fee_cost",
            "entry_time", "exit_time", "exit_reason",
            "window_minute", "cum_return_pct", "confidence",
        ]
        await asyncio.to_thread(
            self._client.insert,
            "mvhe.trades",
            [row],
            column_names=column_names,
        )
        log.debug("clickhouse.trade_inserted", trade_id=row[0])

    async def insert_audit_event(self, event_data: dict[str, Any]) -> None:
        """Insert an order lifecycle audit event.

        Args:
            event_data: Dict with keys matching the audit_events table columns.
        """
        assert self._client is not None, "Call connect() first"
        details = event_data.get("details", {})
        if isinstance(details, dict):
            details = json.dumps(details)
        row = [
            event_data.get("event_id", str(uuid.uuid4())),
            event_data.get("order_id", ""),
            event_data.get("event_type", ""),
            event_data.get("market_id", ""),
            event_data.get("strategy", ""),
            details,
            event_data.get("timestamp", datetime.now(tz=timezone.utc)),
        ]
        column_names = [
            "event_id", "order_id", "event_type",
            "market_id", "strategy", "details", "timestamp",
        ]
        await asyncio.to_thread(
            self._client.insert,
            "mvhe.audit_events",
            [row],
            column_names=column_names,
        )
        log.debug("clickhouse.audit_event_inserted", event_id=row[0])

    async def get_trades(
        self,
        start: datetime,
        end: datetime,
    ) -> list[dict[str, Any]]:
        """Query trade history within a time range.

        Args:
            start: Start of time range (inclusive).
            end: End of time range (inclusive).

        Returns:
            List of trade records as dicts.
        """
        assert self._client is not None, "Call connect() first"
        result = await asyncio.to_thread(
            self._client.query,
            SELECT_TRADES,
            parameters={"start": start, "end": end},
        )
        columns = result.column_names
        return [dict(zip(columns, row)) for row in result.result_rows]

    async def get_daily_pnl(self, target_date: date) -> list[dict[str, Any]]:
        """Aggregate daily P&L by strategy.

        Args:
            target_date: The date to aggregate.

        Returns:
            List of dicts with strategy, trade_count, total_pnl, total_fees, win_count.
        """
        assert self._client is not None, "Call connect() first"
        result = await asyncio.to_thread(
            self._client.query,
            SELECT_DAILY_PNL,
            parameters={"date": target_date},
        )
        columns = result.column_names
        return [dict(zip(columns, row)) for row in result.result_rows]

    async def get_strategy_performance(
        self,
        strategy: str,
    ) -> dict[str, Any] | None:
        """Get aggregate performance metrics for a strategy.

        Args:
            strategy: Strategy identifier.

        Returns:
            Dict of performance metrics, or None if no trades found.
        """
        assert self._client is not None, "Call connect() first"
        result = await asyncio.to_thread(
            self._client.query,
            SELECT_STRATEGY_PERFORMANCE,
            parameters={"strategy": strategy},
        )
        if not result.result_rows:
            return None
        columns = result.column_names
        return dict(zip(columns, result.result_rows[0]))
