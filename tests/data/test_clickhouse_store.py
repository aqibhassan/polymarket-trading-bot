"""Tests for ClickHouseStore — table creation, inserts, queries."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from src.data.clickhouse_store import (
    CREATE_AUDIT_EVENTS_TABLE,
    CREATE_DAILY_SUMMARY_TABLE,
    CREATE_DB,
    CREATE_TRADES_TABLE,
    ClickHouseStore,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_client() -> MagicMock:
    """Create a mock synchronous Client with standard return values."""
    client = MagicMock()
    client.command = MagicMock()
    client.insert = MagicMock()
    client.close = MagicMock()

    # Default query result: empty
    result = MagicMock()
    result.column_names = []
    result.result_rows = []
    client.query = MagicMock(return_value=result)

    return client


@pytest.fixture()
def store() -> ClickHouseStore:
    return ClickHouseStore(host="localhost", port=8123)


@pytest.fixture()
def connected_store(store: ClickHouseStore, mock_client: MagicMock) -> ClickHouseStore:
    """A store with a pre-injected mock client (skips actual connect)."""
    store._client = mock_client
    return store


@pytest.fixture()
def sample_trade() -> dict:
    return {
        "trade_id": "t-001",
        "market_id": "market-abc",
        "strategy": "micro_vol",
        "direction": "YES",
        "entry_price": Decimal("0.55"),
        "exit_price": Decimal("0.62"),
        "position_size": Decimal("100"),
        "pnl": Decimal("7.00"),
        "fee_cost": Decimal("0.20"),
        "entry_time": datetime(2024, 6, 1, 10, 0, 0, tzinfo=timezone.utc),
        "exit_time": datetime(2024, 6, 1, 10, 15, 0, tzinfo=timezone.utc),
        "exit_reason": "PROFIT_TARGET",
        "window_minute": 8,
        "cum_return_pct": 12.7,
        "confidence": 0.85,
    }


@pytest.fixture()
def sample_audit_event() -> dict:
    return {
        "event_id": "evt-001",
        "order_id": "ord-123",
        "event_type": "order",
        "market_id": "market-abc",
        "strategy": "micro_vol",
        "details": {"action": "submit", "price": "0.55"},
        "timestamp": datetime(2024, 6, 1, 10, 0, 0, tzinfo=timezone.utc),
    }


# ---------------------------------------------------------------------------
# Helper: make asyncio.to_thread run synchronously
# ---------------------------------------------------------------------------


async def _fake_to_thread(fn, /, *args, **kwargs):
    """Run the sync function directly (no thread) for testing."""
    return fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# Init tests
# ---------------------------------------------------------------------------


class TestClickHouseStoreInit:
    def test_default_host_and_port(self) -> None:
        s = ClickHouseStore()
        assert s._host == "localhost"
        assert s._port == 8123

    def test_env_overrides(self) -> None:
        with patch.dict("os.environ", {"CLICKHOUSE_HOST": "ch-prod", "CLICKHOUSE_PORT": "9000"}):
            s = ClickHouseStore()
            assert s._host == "ch-prod"
            assert s._port == 9000

    def test_explicit_params(self) -> None:
        s = ClickHouseStore(host="custom-host", port=9999)
        assert s._host == "custom-host"
        assert s._port == 9999


# ---------------------------------------------------------------------------
# Connect / Disconnect
# ---------------------------------------------------------------------------


class TestClickHouseStoreConnect:
    @pytest.mark.asyncio()
    async def test_connect_creates_tables(self, store: ClickHouseStore, mock_client: MagicMock) -> None:
        with (
            patch("src.data.clickhouse_store.clickhouse_connect") as mock_cc,
            patch("src.data.clickhouse_store.asyncio.to_thread", side_effect=_fake_to_thread),
        ):
            mock_cc.get_client = MagicMock(return_value=mock_client)
            await store.connect()

        assert store._client is mock_client
        # Should create database + 3 tables
        assert mock_client.command.call_count == 4
        calls = [c.args[0] for c in mock_client.command.call_args_list]
        assert calls[0] == CREATE_DB
        assert calls[1] == CREATE_TRADES_TABLE
        assert calls[2] == CREATE_AUDIT_EVENTS_TABLE
        assert calls[3] == CREATE_DAILY_SUMMARY_TABLE

    @pytest.mark.asyncio()
    async def test_disconnect(self, connected_store: ClickHouseStore, mock_client: MagicMock) -> None:
        await connected_store.disconnect()
        mock_client.close.assert_called_once()
        assert connected_store._client is None

    @pytest.mark.asyncio()
    async def test_disconnect_when_no_client(self, store: ClickHouseStore) -> None:
        await store.disconnect()
        assert store._client is None


# ---------------------------------------------------------------------------
# Insert trade
# ---------------------------------------------------------------------------


class TestClickHouseStoreInsertTrade:
    @pytest.mark.asyncio()
    async def test_insert_trade(
        self,
        connected_store: ClickHouseStore,
        mock_client: MagicMock,
        sample_trade: dict,
    ) -> None:
        with patch("src.data.clickhouse_store.asyncio.to_thread", side_effect=_fake_to_thread):
            await connected_store.insert_trade(sample_trade)

        mock_client.insert.assert_called_once()
        call_args = mock_client.insert.call_args
        assert call_args.args[0] == "mvhe.trades"
        rows = call_args.args[1]
        assert len(rows) == 1
        row = rows[0]
        assert row[0] == "t-001"
        assert row[1] == "market-abc"
        assert row[2] == "micro_vol"
        assert row[3] == "YES"
        assert row[4] == Decimal("0.55")
        assert row[5] == Decimal("0.62")
        assert row[7] == Decimal("7.00")

    @pytest.mark.asyncio()
    async def test_insert_trade_generates_id(
        self,
        connected_store: ClickHouseStore,
        mock_client: MagicMock,
    ) -> None:
        trade = {"market_id": "m1", "strategy": "s1"}
        with patch("src.data.clickhouse_store.asyncio.to_thread", side_effect=_fake_to_thread):
            await connected_store.insert_trade(trade)
        row = mock_client.insert.call_args.args[1][0]
        # trade_id should be a UUID string
        assert len(row[0]) == 36  # UUID format


# ---------------------------------------------------------------------------
# Insert audit event
# ---------------------------------------------------------------------------


class TestClickHouseStoreInsertAuditEvent:
    @pytest.mark.asyncio()
    async def test_insert_audit_event(
        self,
        connected_store: ClickHouseStore,
        mock_client: MagicMock,
        sample_audit_event: dict,
    ) -> None:
        with patch("src.data.clickhouse_store.asyncio.to_thread", side_effect=_fake_to_thread):
            await connected_store.insert_audit_event(sample_audit_event)

        mock_client.insert.assert_called_once()
        call_args = mock_client.insert.call_args
        assert call_args.args[0] == "mvhe.audit_events"
        rows = call_args.args[1]
        assert len(rows) == 1
        row = rows[0]
        assert row[0] == "evt-001"
        assert row[1] == "ord-123"
        assert row[2] == "order"
        assert row[3] == "market-abc"
        assert row[4] == "micro_vol"
        # details dict should be serialized to JSON string
        assert '"action"' in row[5]
        assert '"submit"' in row[5]

    @pytest.mark.asyncio()
    async def test_insert_audit_event_string_details(
        self,
        connected_store: ClickHouseStore,
        mock_client: MagicMock,
    ) -> None:
        """When details is already a string, don't double-serialize."""
        event = {
            "event_id": "e1",
            "order_id": "o1",
            "event_type": "fill",
            "market_id": "m1",
            "strategy": "s1",
            "details": '{"price": "0.55"}',
            "timestamp": datetime(2024, 6, 1, tzinfo=timezone.utc),
        }
        with patch("src.data.clickhouse_store.asyncio.to_thread", side_effect=_fake_to_thread):
            await connected_store.insert_audit_event(event)
        row = mock_client.insert.call_args.args[1][0]
        assert row[5] == '{"price": "0.55"}'


# ---------------------------------------------------------------------------
# Query: get_trades
# ---------------------------------------------------------------------------


class TestClickHouseStoreGetTrades:
    @pytest.mark.asyncio()
    async def test_get_trades(
        self,
        connected_store: ClickHouseStore,
        mock_client: MagicMock,
    ) -> None:
        result = MagicMock()
        result.column_names = ["trade_id", "market_id", "pnl"]
        result.result_rows = [
            ("t-001", "m1", Decimal("7.00")),
            ("t-002", "m2", Decimal("-2.50")),
        ]
        mock_client.query = MagicMock(return_value=result)

        with patch("src.data.clickhouse_store.asyncio.to_thread", side_effect=_fake_to_thread):
            trades = await connected_store.get_trades(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 12, 31, tzinfo=timezone.utc),
            )

        assert len(trades) == 2
        assert trades[0]["trade_id"] == "t-001"
        assert trades[0]["pnl"] == Decimal("7.00")
        assert trades[1]["market_id"] == "m2"

    @pytest.mark.asyncio()
    async def test_get_trades_empty(
        self,
        connected_store: ClickHouseStore,
        mock_client: MagicMock,
    ) -> None:
        result = MagicMock()
        result.column_names = ["trade_id"]
        result.result_rows = []
        mock_client.query = MagicMock(return_value=result)

        with patch("src.data.clickhouse_store.asyncio.to_thread", side_effect=_fake_to_thread):
            trades = await connected_store.get_trades(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 2, tzinfo=timezone.utc),
            )
        assert trades == []


# ---------------------------------------------------------------------------
# Query: get_daily_pnl
# ---------------------------------------------------------------------------


class TestClickHouseStoreGetDailyPnl:
    @pytest.mark.asyncio()
    async def test_get_daily_pnl(
        self,
        connected_store: ClickHouseStore,
        mock_client: MagicMock,
    ) -> None:
        result = MagicMock()
        result.column_names = ["strategy", "trade_count", "total_pnl", "total_fees", "win_count"]
        result.result_rows = [
            ("micro_vol", 10, Decimal("42.50"), Decimal("2.10"), 7),
        ]
        mock_client.query = MagicMock(return_value=result)

        with patch("src.data.clickhouse_store.asyncio.to_thread", side_effect=_fake_to_thread):
            rows = await connected_store.get_daily_pnl(date(2024, 6, 1))

        assert len(rows) == 1
        assert rows[0]["strategy"] == "micro_vol"
        assert rows[0]["trade_count"] == 10
        assert rows[0]["total_pnl"] == Decimal("42.50")
        assert rows[0]["win_count"] == 7


# ---------------------------------------------------------------------------
# Query: get_strategy_performance
# ---------------------------------------------------------------------------


class TestClickHouseStoreGetStrategyPerformance:
    @pytest.mark.asyncio()
    async def test_get_strategy_performance(
        self,
        connected_store: ClickHouseStore,
        mock_client: MagicMock,
    ) -> None:
        result = MagicMock()
        result.column_names = [
            "strategy", "trade_count", "win_count", "total_pnl",
            "total_fees", "avg_pnl", "best_trade", "worst_trade",
            "avg_position_size", "avg_confidence",
        ]
        result.result_rows = [
            ("micro_vol", 50, 35, Decimal("120.00"), Decimal("5.00"),
             Decimal("2.40"), Decimal("15.00"), Decimal("-8.00"),
             Decimal("100"), 0.82),
        ]
        mock_client.query = MagicMock(return_value=result)

        with patch("src.data.clickhouse_store.asyncio.to_thread", side_effect=_fake_to_thread):
            perf = await connected_store.get_strategy_performance("micro_vol")

        assert perf is not None
        assert perf["strategy"] == "micro_vol"
        assert perf["trade_count"] == 50
        assert perf["win_count"] == 35
        assert perf["total_pnl"] == Decimal("120.00")

    @pytest.mark.asyncio()
    async def test_get_strategy_performance_no_data(
        self,
        connected_store: ClickHouseStore,
        mock_client: MagicMock,
    ) -> None:
        result = MagicMock()
        result.column_names = ["strategy"]
        result.result_rows = []
        mock_client.query = MagicMock(return_value=result)

        with patch("src.data.clickhouse_store.asyncio.to_thread", side_effect=_fake_to_thread):
            perf = await connected_store.get_strategy_performance("nonexistent")
        assert perf is None


# ---------------------------------------------------------------------------
# Graceful handling when ClickHouse is unavailable
# ---------------------------------------------------------------------------


class TestClickHouseStoreUnavailable:
    @pytest.mark.asyncio()
    async def test_connect_failure(self) -> None:
        s = ClickHouseStore(host="unreachable-host", port=8123)
        with (
            patch("src.data.clickhouse_store.clickhouse_connect") as mock_cc,
            patch("src.data.clickhouse_store.asyncio.to_thread", side_effect=_fake_to_thread),
        ):
            mock_cc.get_client = MagicMock(side_effect=ConnectionError("refused"))
            with pytest.raises(ConnectionError, match="refused"):
                await s.connect()

    @pytest.mark.asyncio()
    async def test_insert_before_connect_raises(self) -> None:
        s = ClickHouseStore()
        with pytest.raises(AssertionError, match="connect"):
            await s.insert_trade({"market_id": "m1"})

    @pytest.mark.asyncio()
    async def test_query_before_connect_raises(self) -> None:
        s = ClickHouseStore()
        with pytest.raises(AssertionError, match="connect"):
            await s.get_trades(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 2, tzinfo=timezone.utc),
            )
