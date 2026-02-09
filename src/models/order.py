"""Order and fill models."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    FOK = "FOK"
    GTC = "GTC"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

    @property
    def is_terminal(self) -> bool:
        return self in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        )


class Order(BaseModel):
    """A trading order."""

    id: UUID = Field(default_factory=uuid4)
    market_id: str
    token_id: str
    side: OrderSide
    order_type: OrderType = OrderType.LIMIT
    price: Decimal
    size: Decimal
    status: OrderStatus = OrderStatus.PENDING
    filled_size: Decimal = Decimal("0")
    avg_fill_price: Decimal | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    exchange_order_id: str | None = None
    strategy_id: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)

    @property
    def remaining_size(self) -> Decimal:
        return self.size - self.filled_size

    @property
    def is_complete(self) -> bool:
        return self.status.is_terminal

    @property
    def fill_pct(self) -> Decimal:
        if self.size == 0:
            return Decimal("0")
        return self.filled_size / self.size


class Fill(BaseModel):
    """A single fill (partial or complete) of an order."""

    id: UUID = Field(default_factory=uuid4)
    order_id: UUID
    price: Decimal
    size: Decimal
    fee: Decimal = Decimal("0")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    trade_id: str = ""

    @property
    def notional(self) -> Decimal:
        return self.price * self.size

    @property
    def net_cost(self) -> Decimal:
        return self.notional + self.fee

    model_config = {"frozen": True}
