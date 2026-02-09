"""Tests for ExecutionBridge — routing to paper vs live mode."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from src.execution.bridge import ExecutionBridge
from src.execution.order_manager import OrderManager
from src.execution.paper_trader import PaperTrader
from src.models.order import OrderSide, OrderStatus, OrderType


@pytest.fixture
def paper_trader() -> PaperTrader:
    return PaperTrader(
        initial_balance=Decimal("10000"),
        slippage_bps=0,
        fill_delay=0,
        partial_fill_prob=0.0,
    )


@pytest.fixture
def mock_client() -> AsyncMock:
    client = AsyncMock()
    client.place_order = AsyncMock(return_value={"id": "ex-1"})
    client.cancel_order = AsyncMock(return_value=True)
    return client


@pytest.fixture
def order_manager(mock_client: AsyncMock) -> OrderManager:
    return OrderManager(client=mock_client)


def test_paper_mode(paper_trader: PaperTrader) -> None:
    bridge = ExecutionBridge(mode="paper", paper_trader=paper_trader)
    assert bridge.mode == "paper"


def test_live_mode(order_manager: OrderManager) -> None:
    bridge = ExecutionBridge(mode="live", order_manager=order_manager, skip_confirmation=True)
    assert bridge.mode == "live"


def test_paper_mode_requires_paper_trader() -> None:
    with pytest.raises(ValueError, match="paper_trader required"):
        ExecutionBridge(mode="paper")


def test_live_mode_requires_order_manager() -> None:
    with pytest.raises(ValueError, match="order_manager or live_trader required"):
        ExecutionBridge(mode="live", skip_confirmation=True)


@pytest.mark.asyncio
async def test_paper_submit_routes(paper_trader: PaperTrader) -> None:
    bridge = ExecutionBridge(mode="paper", paper_trader=paper_trader)
    order = await bridge.submit_order(
        market_id="m1", token_id="t1", side=OrderSide.BUY,
        order_type=OrderType.LIMIT, price=Decimal("0.50"), size=Decimal("10"),
    )
    assert order.status == OrderStatus.FILLED


@pytest.mark.asyncio
async def test_live_submit_routes(order_manager: OrderManager) -> None:
    bridge = ExecutionBridge(mode="live", order_manager=order_manager, skip_confirmation=True)
    order = await bridge.submit_order(
        market_id="m1", token_id="t1", side=OrderSide.BUY,
        order_type=OrderType.LIMIT, price=Decimal("0.50"), size=Decimal("10"),
    )
    assert order.status == OrderStatus.SUBMITTED


@pytest.mark.asyncio
async def test_paper_cancel_routes(paper_trader: PaperTrader) -> None:
    bridge = ExecutionBridge(mode="paper", paper_trader=paper_trader)
    result = await bridge.cancel_order("nonexistent")
    assert result is False


@pytest.mark.asyncio
async def test_paper_get_order_routes(paper_trader: PaperTrader) -> None:
    bridge = ExecutionBridge(mode="paper", paper_trader=paper_trader)
    order = await bridge.submit_order(
        market_id="m1", token_id="t1", side=OrderSide.BUY,
        order_type=OrderType.LIMIT, price=Decimal("0.50"), size=Decimal("10"),
    )
    found = await bridge.get_order(str(order.id))
    assert found is not None
    assert found.id == order.id


@pytest.mark.asyncio
async def test_live_cancel_routes(order_manager: OrderManager, mock_client: AsyncMock) -> None:
    bridge = ExecutionBridge(mode="live", order_manager=order_manager, skip_confirmation=True)
    order = await bridge.submit_order(
        market_id="m1", token_id="t1", side=OrderSide.BUY,
        order_type=OrderType.LIMIT, price=Decimal("0.50"), size=Decimal("10"),
    )
    result = await bridge.cancel_order(str(order.id))
    assert result is True
