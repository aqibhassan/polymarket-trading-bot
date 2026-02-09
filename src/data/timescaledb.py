"""TimescaleDB storage for candle and market data."""

from __future__ import annotations

import os
from decimal import Decimal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datetime import datetime

import psycopg
import psycopg.rows
from psycopg_pool import AsyncConnectionPool

from src.core.logging import get_logger
from src.models.market import Candle

log = get_logger(__name__)

CREATE_CANDLES_TABLE = """
CREATE TABLE IF NOT EXISTS candles (
    time        TIMESTAMPTZ NOT NULL,
    exchange    TEXT        NOT NULL,
    symbol      TEXT        NOT NULL,
    interval    TEXT        NOT NULL DEFAULT '15m',
    open        NUMERIC     NOT NULL,
    high        NUMERIC     NOT NULL,
    low         NUMERIC     NOT NULL,
    close       NUMERIC     NOT NULL,
    volume      NUMERIC     NOT NULL
);
"""

CREATE_HYPERTABLE = """
SELECT create_hypertable('candles', 'time', if_not_exists => TRUE);
"""

INSERT_CANDLE = """
INSERT INTO candles (time, exchange, symbol, interval, open, high, low, close, volume)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT DO NOTHING;
"""

SELECT_CANDLES = """
SELECT time, exchange, symbol, interval, open, high, low, close, volume
FROM candles
WHERE symbol = %s AND time >= %s AND time <= %s
ORDER BY time DESC
LIMIT %s;
"""


class TimescaleDBStore:
    """Async TimescaleDB store for candle data.

    Reads connection URL from TIMESCALEDB_URL env var.
    """

    def __init__(self, dsn: str | None = None, min_pool: int = 2, max_pool: int = 10) -> None:
        self._dsn = dsn or os.environ.get("TIMESCALEDB_URL", "")
        self._min_pool = min_pool
        self._max_pool = max_pool
        self._pool: AsyncConnectionPool[psycopg.AsyncConnection[Any]] | None = None

    async def open(self) -> None:
        """Open connection pool."""
        self._pool = AsyncConnectionPool(
            conninfo=self._dsn,  # type: ignore[arg-type]
            min_size=self._min_pool,
            max_size=self._max_pool,
        )
        await self._pool.open()
        log.info("timescaledb.pool_opened")

    async def close(self) -> None:
        """Close connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            log.info("timescaledb.pool_closed")

    async def create_tables(self) -> None:
        """Create candles table and hypertable."""
        assert self._pool is not None
        async with self._pool.connection() as conn:
            await conn.execute(CREATE_CANDLES_TABLE)
            await conn.execute(CREATE_HYPERTABLE)
            await conn.commit()
        log.info("timescaledb.tables_created")

    async def insert_candle(self, candle: Candle) -> None:
        """Insert a single candle."""
        assert self._pool is not None
        async with self._pool.connection() as conn:
            await conn.execute(
                INSERT_CANDLE,
                (
                    candle.timestamp,
                    candle.exchange,
                    candle.symbol,
                    candle.interval,
                    candle.open,
                    candle.high,
                    candle.low,
                    candle.close,
                    candle.volume,
                ),
            )
            await conn.commit()

    async def get_candles(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        limit: int = 100,
    ) -> list[Candle]:
        """Fetch candles for a symbol in a time range."""
        assert self._pool is not None
        async with (
            self._pool.connection() as conn,
            conn.cursor(row_factory=psycopg.rows.dict_row) as cur,
        ):
            await cur.execute(SELECT_CANDLES, (symbol, start, end, limit))
            rows: list[dict[str, Any]] = await cur.fetchall()

        return [
            Candle(
                exchange=row["exchange"],
                symbol=row["symbol"],
                open=Decimal(str(row["open"])),
                high=Decimal(str(row["high"])),
                low=Decimal(str(row["low"])),
                close=Decimal(str(row["close"])),
                volume=Decimal(str(row["volume"])),
                timestamp=row["time"],
                interval=row.get("interval", "15m"),
            )
            for row in rows
        ]
