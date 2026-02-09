"""Protocol interfaces for MVHE components.

All teams code against these contracts, enabling parallel development.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from src.models.market import Candle, MarketState, OrderBookSnapshot
    from src.models.order import Fill, Order, OrderSide, OrderType
    from src.models.signal import Signal


@runtime_checkable
class DataFeed(Protocol):
    """Protocol for market data feeds (Binance WS, Polymarket WS)."""

    async def connect(self) -> None: ...

    async def disconnect(self) -> None: ...

    async def subscribe(self, symbol: str) -> None: ...

    @property
    def is_connected(self) -> bool: ...


@runtime_checkable
class CandleFeed(Protocol):
    """Protocol for candle data feeds."""

    async def get_candles(self, symbol: str, limit: int = 5) -> list[Candle]: ...


@runtime_checkable
class MarketFeed(Protocol):
    """Protocol for Polymarket state feeds."""

    async def get_market_state(self, market_id: str) -> MarketState: ...

    async def get_orderbook(self, market_id: str) -> OrderBookSnapshot: ...


class RiskDecision:
    """Result of a risk check."""

    def __init__(self, approved: bool, reason: str = "", max_size: Decimal = Decimal("0")) -> None:
        self.approved = approved
        self.reason = reason
        self.max_size = max_size


@runtime_checkable
class RiskGate(Protocol):
    """Protocol for pre-trade risk checks."""

    def check_order(
        self,
        signal: Signal,
        position_size: Decimal,
        current_drawdown: Decimal,
        balance: Decimal = ...,
        book_depth: Decimal = ...,
        estimated_fee: Decimal = ...,
    ) -> RiskDecision: ...

    def has_stop_loss(self, signal: Signal) -> bool: ...


@runtime_checkable
class ExecutionEngine(Protocol):
    """Protocol for order execution (paper or live)."""

    async def submit_order(
        self,
        market_id: str,
        token_id: str,
        side: OrderSide,
        order_type: OrderType,
        price: Decimal,
        size: Decimal,
        strategy_id: str = "",
    ) -> Order: ...

    async def cancel_order(self, order_id: str) -> bool: ...

    async def get_order(self, order_id: str) -> Order | None: ...

    @property
    def mode(self) -> str: ...


@runtime_checkable
class ExchangeClient(Protocol):
    """Protocol for exchange API clients (Polymarket CLOB)."""

    async def place_order(
        self,
        token_id: str,
        side: OrderSide,
        price: Decimal,
        size: Decimal,
    ) -> dict[str, Any]: ...

    async def cancel_order(self, order_id: str) -> bool: ...

    async def get_order(self, order_id: str) -> dict[str, Any]: ...

    async def get_balance(self) -> Decimal: ...


@runtime_checkable
class StateCache(Protocol):
    """Protocol for state caching (Redis)."""

    async def get(self, key: str) -> Any: ...

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None: ...

    async def delete(self, key: str) -> None: ...


@runtime_checkable
class AuditLog(Protocol):
    """Protocol for audit trail logging."""

    async def log_order(self, order: Order, action: str) -> None: ...

    async def log_fill(self, fill: Fill, order: Order) -> None: ...

    async def log_rejection(self, signal: Signal, reason: str) -> None: ...
