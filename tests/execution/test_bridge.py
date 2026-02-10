"""Tests for ExecutionBridge — routing, circuit breaker, timeout, retry."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from src.execution.bridge import ExecutionBridge
from src.execution.circuit_breaker import CircuitBreaker
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


# --- Circuit breaker tests ---


@pytest.mark.asyncio
async def test_circuit_breaker_blocks_when_open(paper_trader: PaperTrader) -> None:
    cb = CircuitBreaker(max_failures=2, cooldown_seconds=60)
    bridge = ExecutionBridge(
        mode="paper", paper_trader=paper_trader, circuit_breaker=cb,
    )
    # Force breaker open
    cb.record_failure()
    cb.record_failure()
    assert cb.state == "OPEN"

    order = await bridge.submit_order(
        market_id="m1", token_id="t1", side=OrderSide.BUY,
        order_type=OrderType.LIMIT, price=Decimal("0.50"), size=Decimal("10"),
    )
    assert order.status == OrderStatus.REJECTED


@pytest.mark.asyncio
async def test_circuit_breaker_allows_when_closed(paper_trader: PaperTrader) -> None:
    cb = CircuitBreaker(max_failures=5, cooldown_seconds=60)
    bridge = ExecutionBridge(
        mode="paper", paper_trader=paper_trader, circuit_breaker=cb,
    )
    order = await bridge.submit_order(
        market_id="m1", token_id="t1", side=OrderSide.BUY,
        order_type=OrderType.LIMIT, price=Decimal("0.50"), size=Decimal("10"),
    )
    assert order.status == OrderStatus.FILLED
    assert cb.state == "CLOSED"


@pytest.mark.asyncio
async def test_circuit_breaker_records_success(paper_trader: PaperTrader) -> None:
    cb = CircuitBreaker(max_failures=5, cooldown_seconds=60)
    # Record some failures first
    cb.record_failure()
    cb.record_failure()
    bridge = ExecutionBridge(
        mode="paper", paper_trader=paper_trader, circuit_breaker=cb,
    )
    await bridge.submit_order(
        market_id="m1", token_id="t1", side=OrderSide.BUY,
        order_type=OrderType.LIMIT, price=Decimal("0.50"), size=Decimal("10"),
    )
    # Success should reset failure count
    assert cb.state == "CLOSED"


# --- Order timeout tests ---


@pytest.mark.asyncio
async def test_order_timeout_returns_rejected() -> None:
    """Backend that hangs should trigger timeout and return rejected order."""
    mock_trader = AsyncMock()

    async def slow_submit(*_args: object, **_kwargs: object) -> None:
        await asyncio.sleep(10)  # Hang forever

    mock_trader.submit_order = slow_submit
    bridge = ExecutionBridge(
        mode="live",
        live_trader=mock_trader,
        skip_confirmation=True,
        order_timeout_seconds=0.05,
        order_max_retries=0,
    )
    order = await bridge.submit_order(
        market_id="m1", token_id="t1", side=OrderSide.BUY,
        order_type=OrderType.LIMIT, price=Decimal("0.50"), size=Decimal("10"),
    )
    assert order.status == OrderStatus.REJECTED


@pytest.mark.asyncio
async def test_order_retry_succeeds_on_second_attempt(paper_trader: PaperTrader) -> None:
    """First attempt fails, retry succeeds."""
    call_count = 0

    async def flaky_submit(*args: object, **kwargs: object) -> object:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectionError("transient")
        return await PaperTrader.submit_order(paper_trader, *args, **kwargs)

    paper_trader.submit_order = flaky_submit  # type: ignore[assignment]
    bridge = ExecutionBridge(
        mode="paper",
        paper_trader=paper_trader,
        order_timeout_seconds=5.0,
        order_max_retries=1,
    )
    order = await bridge.submit_order(
        market_id="m1", token_id="t1", side=OrderSide.BUY,
        order_type=OrderType.LIMIT, price=Decimal("0.50"), size=Decimal("10"),
    )
    assert order.status == OrderStatus.FILLED
    assert call_count == 2


@pytest.mark.asyncio
async def test_all_retries_exhausted_records_failure() -> None:
    """All retries fail — circuit breaker records failure, returns rejected."""
    mock_trader = AsyncMock()
    mock_trader.submit_order = AsyncMock(side_effect=ConnectionError("down"))
    cb = CircuitBreaker(max_failures=3, cooldown_seconds=60)

    bridge = ExecutionBridge(
        mode="live",
        live_trader=mock_trader,
        circuit_breaker=cb,
        skip_confirmation=True,
        order_timeout_seconds=1.0,
        order_max_retries=1,
    )
    order = await bridge.submit_order(
        market_id="m1", token_id="t1", side=OrderSide.BUY,
        order_type=OrderType.LIMIT, price=Decimal("0.50"), size=Decimal("10"),
    )
    assert order.status == OrderStatus.REJECTED
    # Should have recorded one failure on the breaker
    assert cb._failure_count == 1


# --- Live trader routing test ---


@pytest.mark.asyncio
async def test_live_trader_routes_correctly() -> None:
    """Live trader receives order when passed to bridge."""
    mock_live = AsyncMock()
    mock_live.submit_order = AsyncMock(return_value=AsyncMock(
        status=OrderStatus.SUBMITTED,
    ))
    bridge = ExecutionBridge(
        mode="live", live_trader=mock_live, skip_confirmation=True,
    )
    await bridge.submit_order(
        market_id="m1", token_id="t1", side=OrderSide.BUY,
        order_type=OrderType.LIMIT, price=Decimal("0.50"), size=Decimal("10"),
    )
    mock_live.submit_order.assert_called_once()
