"""Tests for order models."""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from src.models.order import Fill, Order, OrderSide, OrderStatus


class TestOrderStatus:
    def test_terminal_states(self) -> None:
        assert OrderStatus.FILLED.is_terminal is True
        assert OrderStatus.CANCELLED.is_terminal is True
        assert OrderStatus.REJECTED.is_terminal is True
        assert OrderStatus.EXPIRED.is_terminal is True

    def test_non_terminal_states(self) -> None:
        assert OrderStatus.PENDING.is_terminal is False
        assert OrderStatus.SUBMITTED.is_terminal is False
        assert OrderStatus.PARTIALLY_FILLED.is_terminal is False


class TestOrder:
    def test_default_order(self, sample_order: Order) -> None:
        assert sample_order.status == OrderStatus.PENDING
        assert sample_order.filled_size == Decimal("0")
        assert sample_order.remaining_size == Decimal("100")
        assert sample_order.is_complete is False

    def test_fill_pct(self, sample_order: Order) -> None:
        assert sample_order.fill_pct == Decimal("0")

    def test_partially_filled(self, sample_order: Order) -> None:
        updated = sample_order.model_copy(
            update={
                "filled_size": Decimal("50"),
                "status": OrderStatus.PARTIALLY_FILLED,
            }
        )
        assert updated.remaining_size == Decimal("50")
        assert updated.fill_pct == Decimal("0.5")
        assert updated.is_complete is False

    def test_fully_filled(self, sample_order: Order) -> None:
        updated = sample_order.model_copy(
            update={
                "filled_size": Decimal("100"),
                "status": OrderStatus.FILLED,
                "avg_fill_price": Decimal("0.40"),
            }
        )
        assert updated.remaining_size == Decimal("0")
        assert updated.fill_pct == Decimal("1")
        assert updated.is_complete is True

    def test_zero_size_order_fill_pct(self) -> None:
        order = Order(
            market_id="test",
            token_id="tok",
            side=OrderSide.BUY,
            price=Decimal("0.50"),
            size=Decimal("0"),
        )
        assert order.fill_pct == Decimal("0")


class TestFill:
    def test_fill_notional(self, sample_fill: Fill) -> None:
        assert sample_fill.notional == Decimal("20.00")

    def test_fill_net_cost(self, sample_fill: Fill) -> None:
        assert sample_fill.net_cost == Decimal("20.10")

    def test_fill_is_frozen(self, sample_fill: Fill) -> None:
        with pytest.raises(ValidationError):
            sample_fill.price = Decimal("99")  # type: ignore[misc]
