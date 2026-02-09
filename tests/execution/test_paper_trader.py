"""Tests for PaperTrader — fills, balance, slippage, partial fills."""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.execution.paper_trader import PaperTrader
from src.models.order import OrderSide, OrderStatus, OrderType


@pytest.fixture
def trader() -> PaperTrader:
    return PaperTrader(
        initial_balance=Decimal("10000"),
        slippage_bps=5,
        fill_delay=0,
        partial_fill_prob=0.0,  # Deterministic full fills
    )


@pytest.mark.asyncio
async def test_mode(trader: PaperTrader) -> None:
    assert trader.mode == "paper"


@pytest.mark.asyncio
async def test_initial_balance(trader: PaperTrader) -> None:
    assert trader.balance == Decimal("10000")


@pytest.mark.asyncio
async def test_submit_buy_order(trader: PaperTrader) -> None:
    order = await trader.submit_order(
        market_id="m1", token_id="t1", side=OrderSide.BUY,
        order_type=OrderType.LIMIT, price=Decimal("0.50"), size=Decimal("100"),
    )
    assert order.status == OrderStatus.FILLED
    assert order.filled_size == Decimal("100")
    # Balance should decrease
    assert trader.balance < Decimal("10000")


@pytest.mark.asyncio
async def test_submit_sell_order(trader: PaperTrader) -> None:
    order = await trader.submit_order(
        market_id="m1", token_id="t1", side=OrderSide.SELL,
        order_type=OrderType.LIMIT, price=Decimal("0.50"), size=Decimal("100"),
    )
    assert order.status == OrderStatus.FILLED
    # Balance should increase
    assert trader.balance > Decimal("10000")


@pytest.mark.asyncio
async def test_slippage_applied_buy(trader: PaperTrader) -> None:
    order = await trader.submit_order(
        market_id="m1", token_id="t1", side=OrderSide.BUY,
        order_type=OrderType.LIMIT, price=Decimal("1.00"), size=Decimal("10"),
    )
    # Buy slippage increases price: 1.00 + 1.00 * 5/10000 = 1.0005
    assert order.avg_fill_price is not None
    assert order.avg_fill_price > Decimal("1.00")


@pytest.mark.asyncio
async def test_slippage_applied_sell(trader: PaperTrader) -> None:
    order = await trader.submit_order(
        market_id="m1", token_id="t1", side=OrderSide.SELL,
        order_type=OrderType.LIMIT, price=Decimal("1.00"), size=Decimal("10"),
    )
    # Sell slippage decreases price
    assert order.avg_fill_price is not None
    assert order.avg_fill_price < Decimal("1.00")


@pytest.mark.asyncio
async def test_insufficient_balance_rejected() -> None:
    trader = PaperTrader(
        initial_balance=Decimal("1"),
        slippage_bps=0,
        fill_delay=0,
        partial_fill_prob=0.0,
    )
    order = await trader.submit_order(
        market_id="m1", token_id="t1", side=OrderSide.BUY,
        order_type=OrderType.LIMIT, price=Decimal("10.00"), size=Decimal("100"),
    )
    assert order.status == OrderStatus.REJECTED


@pytest.mark.asyncio
async def test_partial_fill() -> None:
    trader = PaperTrader(
        initial_balance=Decimal("10000"),
        slippage_bps=0,
        fill_delay=0,
        partial_fill_prob=1.0,  # Force partial fills
    )
    order = await trader.submit_order(
        market_id="m1", token_id="t1", side=OrderSide.BUY,
        order_type=OrderType.LIMIT, price=Decimal("0.50"), size=Decimal("100"),
    )
    assert order.status == OrderStatus.PARTIALLY_FILLED
    assert order.filled_size < Decimal("100")
    assert order.filled_size > Decimal("0")


@pytest.mark.asyncio
async def test_cancel_order(trader: PaperTrader) -> None:
    # Submit an order that will be filled, then try to cancel
    order = await trader.submit_order(
        market_id="m1", token_id="t1", side=OrderSide.BUY,
        order_type=OrderType.LIMIT, price=Decimal("0.50"), size=Decimal("10"),
    )
    # Filled orders can't be cancelled
    result = await trader.cancel_order(str(order.id))
    assert result is False


@pytest.mark.asyncio
async def test_cancel_nonexistent(trader: PaperTrader) -> None:
    result = await trader.cancel_order("nonexistent")
    assert result is False


@pytest.mark.asyncio
async def test_get_order(trader: PaperTrader) -> None:
    order = await trader.submit_order(
        market_id="m1", token_id="t1", side=OrderSide.BUY,
        order_type=OrderType.LIMIT, price=Decimal("0.50"), size=Decimal("10"),
    )
    found = await trader.get_order(str(order.id))
    assert found is not None
    assert found.id == order.id


@pytest.mark.asyncio
async def test_get_order_not_found(trader: PaperTrader) -> None:
    assert await trader.get_order("missing") is None


@pytest.mark.asyncio
async def test_pnl_tracking(trader: PaperTrader) -> None:
    assert trader.pnl == Decimal("0")
    await trader.submit_order(
        market_id="m1", token_id="t1", side=OrderSide.BUY,
        order_type=OrderType.LIMIT, price=Decimal("0.50"), size=Decimal("100"),
    )
    # After buying, balance decreased, pnl is negative
    assert trader.pnl < Decimal("0")
