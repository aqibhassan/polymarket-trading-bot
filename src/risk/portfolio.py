"""Portfolio tracker — open/closed positions, P&L, exposure, drawdown."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING

from src.core.logging import get_logger

if TYPE_CHECKING:
    from src.models.market import Position

log = get_logger(__name__)


class Portfolio:
    """Tracks all open and closed positions with real-time P&L calculations."""

    def __init__(self) -> None:
        self._open: list[Position] = []
        self._closed: list[Position] = []
        self._peak_equity: Decimal = Decimal("0")
        self._max_drawdown: Decimal = Decimal("0")

    @property
    def open_positions(self) -> list[Position]:
        """All currently open positions."""
        return list(self._open)

    @property
    def closed_positions(self) -> list[Position]:
        """All closed positions."""
        return list(self._closed)

    @property
    def max_drawdown(self) -> Decimal:
        """Peak-to-trough drawdown."""
        return self._max_drawdown

    def add_position(self, position: Position) -> None:
        """Add a new open position.

        Args:
            position: The position to track.
        """
        self._open.append(position)
        log.info(
            "position_opened",
            market=position.market_id,
            side=position.side.value,
            entry_price=str(position.entry_price),
            quantity=str(position.quantity),
        )

    def close_position(
        self,
        market_id: str,
        exit_price: Decimal,
        exit_time: datetime | None = None,
        exit_reason: str = "manual",
    ) -> Position | None:
        """Close a position by market_id.

        Args:
            market_id: The market ID of the position to close.
            exit_price: The exit price.
            exit_time: Time of exit (defaults to now).
            exit_reason: Reason for closing.

        Returns:
            The closed position, or None if not found.
        """
        exit_t = exit_time or datetime.now(tz=timezone.utc)

        for i, pos in enumerate(self._open):
            if pos.market_id == market_id:
                closed = pos.model_copy(
                    update={
                        "exit_price": exit_price,
                        "exit_time": exit_t,
                        "exit_reason": exit_reason,
                    }
                )
                self._open.pop(i)
                self._closed.append(closed)

                pnl = closed.realized_pnl()
                log.info(
                    "position_closed",
                    market=market_id,
                    exit_price=str(exit_price),
                    reason=exit_reason,
                    realized_pnl=str(pnl),
                )
                return closed

        log.warning("position_not_found", market=market_id)
        return None

    def daily_pnl(self, current_prices: dict[str, Decimal]) -> Decimal:
        """Calculate total daily P&L (realized + unrealized).

        Args:
            current_prices: Map of market_id -> current price.

        Returns:
            Total daily P&L.
        """
        realized = sum(
            (p.realized_pnl() or Decimal("0") for p in self._closed),
            Decimal("0"),
        )
        unrealized = sum(
            (
                p.unrealized_pnl(current_prices.get(p.market_id, p.entry_price))
                for p in self._open
            ),
            Decimal("0"),
        )
        total = realized + unrealized

        self._update_drawdown(total)

        return total

    def total_exposure(self, current_prices: dict[str, Decimal]) -> Decimal:
        """Calculate total notional exposure at risk.

        Args:
            current_prices: Map of market_id -> current price.

        Returns:
            Total notional value of open positions.
        """
        return sum(
            (
                current_prices.get(p.market_id, p.entry_price) * p.quantity
                for p in self._open
            ),
            Decimal("0"),
        )

    def daily_summary(self) -> dict[str, str | int]:
        """Generate summary statistics for the day.

        Returns:
            Dictionary with count and P&L stats.
        """
        realized_pnls: list[Decimal] = [
            pnl for p in self._closed if (pnl := p.realized_pnl()) is not None
        ]
        winners = [pnl for pnl in realized_pnls if pnl > 0]
        losers = [pnl for pnl in realized_pnls if pnl < 0]

        total_realized = sum(realized_pnls, Decimal("0"))

        return {
            "open_positions": len(self._open),
            "closed_positions": len(self._closed),
            "total_realized_pnl": str(total_realized),
            "winners": len(winners),
            "losers": len(losers),
            "max_drawdown": str(self._max_drawdown),
        }

    def update_equity(self, current_balance: Decimal) -> None:
        """Update drawdown tracking with the current account balance.

        Call this after every trade settlement to keep max_drawdown accurate.

        Args:
            current_balance: Current total account balance.
        """
        self._update_drawdown(current_balance)

    def _update_drawdown(self, equity: Decimal) -> None:
        """Track peak equity and max drawdown."""
        if equity > self._peak_equity:
            self._peak_equity = equity
        if self._peak_equity > 0:
            current_dd = (self._peak_equity - equity) / self._peak_equity
            if current_dd > self._max_drawdown:
                self._max_drawdown = current_dd
