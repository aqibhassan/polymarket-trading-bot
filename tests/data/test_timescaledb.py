"""Tests for TimescaleDBStore."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.data.timescaledb import TimescaleDBStore
from src.models.market import Candle


@pytest.fixture()
def sample_candle() -> Candle:
    return Candle(
        exchange="binance",
        symbol="BTCUSDT",
        open=Decimal("50000"),
        high=Decimal("50500"),
        low=Decimal("49800"),
        close=Decimal("50300"),
        volume=Decimal("123.45"),
        timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        interval="15m",
    )


class TestTimescaleDBStoreInit:
    def test_default_dsn_from_env(self) -> None:
        with patch.dict("os.environ", {"TIMESCALEDB_URL": "postgres://test:5432/db"}):
            store = TimescaleDBStore()
            assert store._dsn == "postgres://test:5432/db"

    def test_explicit_dsn(self) -> None:
        store = TimescaleDBStore(dsn="postgres://custom/db")
        assert store._dsn == "postgres://custom/db"


class TestTimescaleDBStoreCreateTables:
    @pytest.mark.asyncio()
    async def test_create_tables(self) -> None:
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_conn.commit = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)

        mock_pool = AsyncMock()
        mock_pool.connection = MagicMock(return_value=mock_conn)

        store = TimescaleDBStore(dsn="postgres://test/db")
        store._pool = mock_pool

        await store.create_tables()

        assert mock_conn.execute.call_count == 2
        mock_conn.commit.assert_called_once()


class TestTimescaleDBStoreInsert:
    @pytest.mark.asyncio()
    async def test_insert_candle(self, sample_candle: Candle) -> None:
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_conn.commit = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)

        mock_pool = AsyncMock()
        mock_pool.connection = MagicMock(return_value=mock_conn)

        store = TimescaleDBStore(dsn="postgres://test/db")
        store._pool = mock_pool

        await store.insert_candle(sample_candle)

        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args
        params = call_args[0][1]
        assert params[0] == sample_candle.timestamp
        assert params[1] == "binance"
        assert params[2] == "BTCUSDT"
        assert params[4] == Decimal("50000")
        assert params[7] == Decimal("50300")


class TestTimescaleDBStoreGetCandles:
    @pytest.mark.asyncio()
    async def test_get_candles(self) -> None:
        rows = [
            {
                "time": datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                "exchange": "binance",
                "symbol": "BTCUSDT",
                "interval": "15m",
                "open": Decimal("50000"),
                "high": Decimal("50500"),
                "low": Decimal("49800"),
                "close": Decimal("50300"),
                "volume": Decimal("123.45"),
            },
        ]

        mock_cursor = AsyncMock()
        mock_cursor.execute = AsyncMock()
        mock_cursor.fetchall = AsyncMock(return_value=rows)
        mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
        mock_cursor.__aexit__ = AsyncMock(return_value=False)

        mock_conn = AsyncMock()
        mock_conn.cursor = MagicMock(return_value=mock_cursor)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)

        mock_pool = AsyncMock()
        mock_pool.connection = MagicMock(return_value=mock_conn)

        store = TimescaleDBStore(dsn="postgres://test/db")
        store._pool = mock_pool

        candles = await store.get_candles(
            symbol="BTCUSDT",
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 2, tzinfo=timezone.utc),
            limit=10,
        )

        assert len(candles) == 1
        assert candles[0].symbol == "BTCUSDT"
        assert candles[0].close == Decimal("50300")

    @pytest.mark.asyncio()
    async def test_get_candles_empty(self) -> None:
        mock_cursor = AsyncMock()
        mock_cursor.execute = AsyncMock()
        mock_cursor.fetchall = AsyncMock(return_value=[])
        mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
        mock_cursor.__aexit__ = AsyncMock(return_value=False)

        mock_conn = AsyncMock()
        mock_conn.cursor = MagicMock(return_value=mock_cursor)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)

        mock_pool = AsyncMock()
        mock_pool.connection = MagicMock(return_value=mock_conn)

        store = TimescaleDBStore(dsn="postgres://test/db")
        store._pool = mock_pool

        candles = await store.get_candles(
            symbol="BTCUSDT",
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )

        assert candles == []


class TestTimescaleDBStoreOpenClose:
    @pytest.mark.asyncio()
    async def test_close_when_pool_is_none(self) -> None:
        store = TimescaleDBStore(dsn="postgres://test/db")
        await store.close()
        assert store._pool is None

    @pytest.mark.asyncio()
    async def test_close_pool(self) -> None:
        mock_pool = AsyncMock()
        store = TimescaleDBStore(dsn="postgres://test/db")
        store._pool = mock_pool
        await store.close()
        mock_pool.close.assert_called_once()
        assert store._pool is None
