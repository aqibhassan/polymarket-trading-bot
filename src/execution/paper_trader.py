"""Paper trader — simulated fill execution for testing strategies."""

from __future__ import annotations

import asyncio
import random
from datetime import UTC, datetime
from decimal import Decimal

from src.core.logging import get_logger, log_order_event
from src.models.market import Position, Side
from src.models.order import Fill, Order, OrderSide, OrderStatus, OrderType

logger = get_logger(__name__)


class PaperTrader:
    """Simulated order execution for paper trading.

    Implements the ExecutionEngine protocol with mode='paper'.
    """

    def __init__(
        self,
        initial_balance: Decimal = Decimal("10000"),
        slippage_bps: int = 5,
        fill_delay: float = 0.05,
        partial_fill_prob: float = 0.1,
    ) -> None:
        self._balance = initial_balance
        self._initial_balance = initial_balance
        self._slippage_bps = slippage_bps
        self._fill_delay = fill_delay
        self._partial_fill_prob = partial_fill_prob
        self._orders: dict[str, Order] = {}
        self._positions: dict[str, Position] = {}
        self._fills: list[Fill] = []

    @property
    def mode(self) -> str:
        return "paper"

    @property
    def balance(self) -> Decimal:
        return self._balance

    @property
    def positions(self) -> list[Position]:
        return [p for p in self._positions.values() if p.is_open]

    @property
    def pnl(self) -> Decimal:
        return self._balance - self._initial_balance

    async def submit_order(
        self,
        market_id: str,
        token_id: str,
        side: OrderSide,
        order_type: OrderType,
        price: Decimal,
        size: Decimal,
        strategy_id: str = "",
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
            "paper_submit", oid, market_id=market_id,
            side=side.value, price=str(price), size=str(size),
        )

        # Simulate fill delay
        if self._fill_delay > 0:
            await asyncio.sleep(self._fill_delay)

        # Apply slippage
        slippage = price * Decimal(self._slippage_bps) / Decimal("10000")
        fill_price = price + slippage if side == OrderSide.BUY else price - slippage

        # Determine fill size (partial vs full)
        if random.random() < self._partial_fill_prob:
            fill_size = size * Decimal(str(round(random.uniform(0.3, 0.9), 2)))
        else:
            fill_size = size

        # Check balance for buys
        cost = fill_price * fill_size
        if side == OrderSide.BUY and cost > self._balance:
            order = order.model_copy(update={
                "status": OrderStatus.REJECTED,
                "updated_at": datetime.now(tz=UTC),
            })
            self._orders[oid] = order
            log_order_event("paper_rejected", oid, reason="insufficient_balance")
            return order

        # Create fill
        fill = Fill(
            order_id=order.id,
            price=fill_price,
            size=fill_size,
        )
        self._fills.append(fill)

        # Update balance
        if side == OrderSide.BUY:
            self._balance -= cost
        else:
            self._balance += fill_price * fill_size

        # Determine final status
        status = OrderStatus.FILLED if fill_size == size else OrderStatus.PARTIALLY_FILLED

        order = order.model_copy(update={
            "status": status,
            "filled_size": fill_size,
            "avg_fill_price": fill_price,
            "updated_at": datetime.now(tz=UTC),
        })
        self._orders[oid] = order

        log_order_event(
            "paper_fill", oid,
            fill_price=str(fill_price), fill_size=str(fill_size),
            status=status.value, balance=str(self._balance),
        )

        # Track position for buys
        if side == OrderSide.BUY and (
            not status.is_terminal or status == OrderStatus.FILLED
        ):
            # Infer position side from token_id suffix; default to YES
            pos_side = Side.NO if "no" in token_id.lower() else Side.YES
            self._positions[oid] = Position(
                market_id=market_id,
                side=pos_side,
                token_id=token_id,
                entry_price=fill_price,
                quantity=fill_size,
                entry_time=datetime.now(tz=UTC),
                stop_loss=fill_price * Decimal("0.96"),
                take_profit=fill_price * Decimal("1.05"),
            )

        return order

    async def cancel_order(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if order is None or order.status.is_terminal:
            return False

        order = order.model_copy(update={
            "status": OrderStatus.CANCELLED,
            "updated_at": datetime.now(tz=UTC),
        })
        self._orders[order_id] = order
        log_order_event("paper_cancel", order_id)
        return True

    async def get_order(self, order_id: str) -> Order | None:
        return self._orders.get(order_id)
