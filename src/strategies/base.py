"""Base strategy abstract class — all strategies must inherit from this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from src.config.loader import ConfigLoader
    from src.models.market import MarketState, OrderBookSnapshot, Position
    from src.models.order import Fill
    from src.models.signal import Signal


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies.

    Subclasses must define REQUIRED_PARAMS and implement generate_signals().
    """

    REQUIRED_PARAMS: ClassVar[list[str]] = []

    def __init__(self, config: ConfigLoader, strategy_id: str | None = None) -> None:
        self._config = config
        self.strategy_id = strategy_id or self.__class__.__name__
        self._positions: list[Position] = []
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate that all required params exist in config."""
        if self.REQUIRED_PARAMS:
            self._config.validate_keys(self.REQUIRED_PARAMS)

    @abstractmethod
    def generate_signals(
        self,
        market_state: MarketState,
        orderbook: OrderBookSnapshot,
        context: dict[str, Any],
    ) -> list[Signal]:
        """Generate trading signals given current market state.

        Args:
            market_state: Current Polymarket market state.
            orderbook: Current orderbook snapshot.
            context: Additional context (e.g., BTC candles, historical data).

        Returns:
            List of signals (may be empty if no opportunity).
        """
        ...

    @abstractmethod
    def on_fill(self, fill: Fill, position: Position) -> None:
        """Called when an order is filled.

        Args:
            fill: The fill details.
            position: The position that was filled.
        """
        ...

    @abstractmethod
    def on_cancel(self, order_id: str, reason: str) -> None:
        """Called when an order is cancelled.

        Args:
            order_id: The cancelled order ID.
            reason: Reason for cancellation.
        """
        ...

    def on_tick(self, market_state: MarketState) -> None:  # noqa: B027
        """Called on every market data update. Override for tick-level logic."""

    @property
    def positions(self) -> list[Position]:
        return list(self._positions)

    @property
    def open_positions(self) -> list[Position]:
        return [p for p in self._positions if p.is_open]

    def add_position(self, position: Position) -> None:
        self._positions.append(position)

    def get_config(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)
