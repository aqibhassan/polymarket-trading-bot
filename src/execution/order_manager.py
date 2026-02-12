"""Live order manager — submits and tracks orders via an ExchangeClient."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from src.core.logging import get_logger, log_order_event
from src.models.order import Fill, Order, OrderSide, OrderStatus, OrderType

if TYPE_CHECKING:
    from decimal import Decimal

    from src.interfaces import ExchangeClient

logger = get_logger(__name__)


class OrderManager:
    """Manages order lifecycle against a live exchange.

    Implements the ExecutionEngine protocol with mode='live'.
    """

    def __init__(self, client: ExchangeClient) -> None:
        self._client = client
        self._orders: dict[str, Order] = {}

    @property
    def mode(self) -> str:
        return "live"

    @property
    def pending_orders(self) -> list[Order]:
        return [
            o for o in self._orders.values()
            if not o.status.is_terminal
        ]

    async def submit_order(
        self,
        market_id: str,
        token_id: str,
        side: OrderSide,
        order_type: OrderType,
        price: Decimal,
        size: Decimal,
        strategy_id: str = "",
        **kwargs: object,
    ) -> Order:
        order = Order(
            market_id=market_id,
            token_id=token_id,
            side=side,
            order_type=order_type,
            price=price,
            size=size,
            strategy_id=strategy_id,
        )
        oid = str(order.id)
        self._orders[oid] = order
        log_order_event(
            "submit", oid, market_id=market_id,
            side=side.value, price=str(price), size=str(size),
        )

        try:
            result = await self._client.place_order(token_id, side, price, size)
            exchange_id = result.get("id", "")
            order = order.model_copy(update={
                "status": OrderStatus.SUBMITTED,
                "exchange_order_id": exchange_id,
                "updated_at": datetime.now(tz=UTC),
            })
            self._orders[oid] = order
            log_order_event("submitted", oid, exchange_order_id=exchange_id)
        except Exception as exc:
            order = order.model_copy(update={
                "status": OrderStatus.REJECTED,
                "updated_at": datetime.now(tz=UTC),
            })
            self._orders[oid] = order
            log_order_event("rejected", oid, reason=str(exc))
            logger.error("order_rejected", order_id=oid, error=str(exc))

        return order

    async def cancel_order(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if order is None or order.status.is_terminal:
            return False

        log_order_event("cancel_request", order_id)
        try:
            success = await self._client.cancel_order(
                order.exchange_order_id or order_id,
            )
            if success:
                order = order.model_copy(update={
                    "status": OrderStatus.CANCELLED,
                    "updated_at": datetime.now(tz=UTC),
                })
                self._orders[order_id] = order
                log_order_event("cancelled", order_id)
            return success
        except Exception as exc:
            logger.error("cancel_failed", order_id=order_id, error=str(exc))
            return False

    async def get_order(self, order_id: str) -> Order | None:
        return self._orders.get(order_id)

    def record_fill(self, order_id: str, fill: Fill) -> Order | None:
        """Record an external fill against a tracked order."""
        order = self._orders.get(order_id)
        if order is None:
            return None

        new_filled = order.filled_size + fill.size
        new_status = (
            OrderStatus.FILLED if new_filled >= order.size
            else OrderStatus.PARTIALLY_FILLED
        )
        # Weighted average fill price
        if order.avg_fill_price is not None:
            total_cost = order.avg_fill_price * order.filled_size + fill.price * fill.size
            avg_price = total_cost / new_filled
        else:
            avg_price = fill.price

        order = order.model_copy(update={
            "filled_size": new_filled,
            "avg_fill_price": avg_price,
            "status": new_status,
            "updated_at": datetime.now(tz=UTC),
        })
        self._orders[order_id] = order
        log_order_event(
            "fill", order_id,
            fill_size=str(fill.size), fill_price=str(fill.price),
            total_filled=str(new_filled), status=new_status.value,
        )
        return order
