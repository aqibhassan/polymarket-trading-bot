"""Tests for OrderManager — submit, cancel, get_order, state transitions."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest

from src.execution.order_manager import OrderManager
from src.models.order import Fill, OrderSide, OrderStatus, OrderType


@pytest.fixture
def mock_client() -> AsyncMock:
    client = AsyncMock()
    client.place_order = AsyncMock(return_value={"id": "exchange-123"})
    client.cancel_order = AsyncMock(return_value=True)
    client.get_order = AsyncMock(return_value={})
    return client


@pytest.fixture
def manager(mock_client: AsyncMock) -> OrderManager:
    return OrderManager(client=mock_client)


@pytest.mark.asyncio
async def test_submit_order_success(manager: OrderManager, mock_client: AsyncMock) -> None:
    order = await manager.submit_order(
        market_id="market-1",
        token_id="token-1",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=Decimal("0.55"),
        size=Decimal("100"),
        strategy_id="strat-1",
    )
    assert order.status == OrderStatus.SUBMITTED
    assert order.exchange_order_id == "exchange-123"
    assert order.market_id == "market-1"
    assert order.price == Decimal("0.55")
    mock_client.place_order.assert_awaited_once()


@pytest.mark.asyncio
async def test_submit_order_rejected(manager: OrderManager, mock_client: AsyncMock) -> None:
    mock_client.place_order.side_effect = RuntimeError("connection refused")
    order = await manager.submit_order(
        market_id="m1",
        token_id="t1",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        price=Decimal("0.60"),
        size=Decimal("50"),
    )
    assert order.status == OrderStatus.REJECTED


@pytest.mark.asyncio
async def test_cancel_order(manager: OrderManager, mock_client: AsyncMock) -> None:
    order = await manager.submit_order(
        market_id="m1", token_id="t1", side=OrderSide.BUY,
        order_type=OrderType.LIMIT, price=Decimal("0.50"), size=Decimal("10"),
    )
    oid = str(order.id)
    result = await manager.cancel_order(oid)
    assert result is True
    cancelled = await manager.get_order(oid)
    assert cancelled is not None
    assert cancelled.status == OrderStatus.CANCELLED


@pytest.mark.asyncio
async def test_cancel_nonexistent_order(manager: OrderManager) -> None:
    result = await manager.cancel_order("nonexistent")
    assert result is False


@pytest.mark.asyncio
async def test_cancel_terminal_order(manager: OrderManager, mock_client: AsyncMock) -> None:
    mock_client.place_order.side_effect = RuntimeError("fail")
    order = await manager.submit_order(
        market_id="m1", token_id="t1", side=OrderSide.BUY,
        order_type=OrderType.LIMIT, price=Decimal("0.50"), size=Decimal("10"),
    )
    result = await manager.cancel_order(str(order.id))
    assert result is False


@pytest.mark.asyncio
async def test_get_order(manager: OrderManager) -> None:
    order = await manager.submit_order(
        market_id="m1", token_id="t1", side=OrderSide.BUY,
        order_type=OrderType.LIMIT, price=Decimal("0.50"), size=Decimal("10"),
    )
    found = await manager.get_order(str(order.id))
    assert found is not None
    assert found.id == order.id


@pytest.mark.asyncio
async def test_get_order_not_found(manager: OrderManager) -> None:
    assert await manager.get_order("missing") is None


@pytest.mark.asyncio
async def test_pending_orders(manager: OrderManager) -> None:
    await manager.submit_order(
        market_id="m1", token_id="t1", side=OrderSide.BUY,
        order_type=OrderType.LIMIT, price=Decimal("0.50"), size=Decimal("10"),
    )
    await manager.submit_order(
        market_id="m2", token_id="t2", side=OrderSide.SELL,
        order_type=OrderType.LIMIT, price=Decimal("0.60"), size=Decimal("20"),
    )
    assert len(manager.pending_orders) == 2


@pytest.mark.asyncio
async def test_record_fill_full(manager: OrderManager) -> None:
    order = await manager.submit_order(
        market_id="m1", token_id="t1", side=OrderSide.BUY,
        order_type=OrderType.LIMIT, price=Decimal("0.50"), size=Decimal("10"),
    )
    oid = str(order.id)
    fill = Fill(order_id=order.id, price=Decimal("0.50"), size=Decimal("10"))
    updated = manager.record_fill(oid, fill)
    assert updated is not None
    assert updated.status == OrderStatus.FILLED
    assert updated.filled_size == Decimal("10")


@pytest.mark.asyncio
async def test_record_fill_partial(manager: OrderManager) -> None:
    order = await manager.submit_order(
        market_id="m1", token_id="t1", side=OrderSide.BUY,
        order_type=OrderType.LIMIT, price=Decimal("0.50"), size=Decimal("100"),
    )
    oid = str(order.id)
    fill = Fill(order_id=order.id, price=Decimal("0.50"), size=Decimal("30"))
    updated = manager.record_fill(oid, fill)
    assert updated is not None
    assert updated.status == OrderStatus.PARTIALLY_FILLED
    assert updated.filled_size == Decimal("30")


def test_mode(manager: OrderManager) -> None:
    assert manager.mode == "live"


@pytest.mark.asyncio
async def test_audit_logging_called(manager: OrderManager) -> None:
    with patch("src.execution.order_manager.log_order_event") as mock_log:
        await manager.submit_order(
            market_id="m1", token_id="t1", side=OrderSide.BUY,
            order_type=OrderType.LIMIT, price=Decimal("0.50"), size=Decimal("10"),
        )
        assert mock_log.call_count >= 2  # submit + submitted
