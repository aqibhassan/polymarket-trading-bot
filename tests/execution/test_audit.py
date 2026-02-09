"""Tests for AuditLogger — file fallback, entry format."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path  # noqa: TCH003

import pytest

from src.execution.audit import AuditLogger
from src.models.market import Side
from src.models.order import Fill, Order, OrderSide, OrderType
from src.models.signal import Signal, SignalType


@pytest.fixture
def tmp_log(tmp_path: Path) -> Path:
    return tmp_path / "audit.jsonl"


@pytest.fixture
def audit(tmp_log: Path) -> AuditLogger:
    return AuditLogger(log_path=str(tmp_log))


def _make_order() -> Order:
    return Order(
        market_id="m1",
        token_id="t1",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=Decimal("0.55"),
        size=Decimal("100"),
        strategy_id="strat-1",
    )


def _make_signal() -> Signal:
    return Signal(
        strategy_id="strat-1",
        market_id="m1",
        signal_type=SignalType.ENTRY,
        direction=Side.YES,
    )


@pytest.mark.asyncio
async def test_log_order_creates_file(audit: AuditLogger, tmp_log: Path) -> None:
    order = _make_order()
    await audit.log_order(order, "submit")
    assert tmp_log.exists()
    lines = tmp_log.read_text().strip().split("\n")
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["event_type"] == "order"
    assert entry["action"] == "submit"
    assert entry["order_id"] == str(order.id)
    assert entry["market_id"] == "m1"
    assert entry["price"] == "0.55"


@pytest.mark.asyncio
async def test_log_fill(audit: AuditLogger, tmp_log: Path) -> None:
    order = _make_order()
    fill = Fill(order_id=order.id, price=Decimal("0.55"), size=Decimal("50"))
    await audit.log_fill(fill, order)
    lines = tmp_log.read_text().strip().split("\n")
    entry = json.loads(lines[0])
    assert entry["event_type"] == "fill"
    assert entry["fill_id"] == str(fill.id)
    assert entry["price"] == "0.55"
    assert entry["size"] == "50"


@pytest.mark.asyncio
async def test_log_rejection(audit: AuditLogger, tmp_log: Path) -> None:
    signal = _make_signal()
    await audit.log_rejection(signal, "max_drawdown_exceeded")
    lines = tmp_log.read_text().strip().split("\n")
    entry = json.loads(lines[0])
    assert entry["event_type"] == "rejection"
    assert entry["reason"] == "max_drawdown_exceeded"
    assert entry["strategy_id"] == "strat-1"


@pytest.mark.asyncio
async def test_append_only(audit: AuditLogger, tmp_log: Path) -> None:
    order = _make_order()
    await audit.log_order(order, "submit")
    await audit.log_order(order, "fill")
    lines = tmp_log.read_text().strip().split("\n")
    assert len(lines) == 2


@pytest.mark.asyncio
async def test_timestamp_present(audit: AuditLogger, tmp_log: Path) -> None:
    order = _make_order()
    await audit.log_order(order, "submit")
    entry = json.loads(tmp_log.read_text().strip())
    assert "timestamp" in entry
    assert "T" in entry["timestamp"]  # ISO format


@pytest.mark.asyncio
async def test_clickhouse_fallback(tmp_log: Path) -> None:
    """When ClickHouse write fails, still writes to file."""

    class FakeClickHouseStore:
        async def insert_audit_event(self, event_data: dict) -> None:
            raise ConnectionError("clickhouse down")

    audit = AuditLogger(log_path=str(tmp_log), clickhouse_store=FakeClickHouseStore())  # type: ignore[arg-type]
    order = _make_order()
    await audit.log_order(order, "submit")
    assert tmp_log.exists()
    lines = tmp_log.read_text().strip().split("\n")
    assert len(lines) == 1


@pytest.mark.asyncio
async def test_no_api_keys_in_log(audit: AuditLogger, tmp_log: Path) -> None:
    """Ensure no sensitive data leaks into audit logs."""
    order = _make_order()
    await audit.log_order(order, "submit")
    content = tmp_log.read_text()
    assert "api_key" not in content.lower()
    assert "secret" not in content.lower()
