"""Market speed tracker — measures price correction velocity."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from src.core.logging import get_logger

if TYPE_CHECKING:
    from datetime import datetime
    from decimal import Decimal

log = get_logger(__name__)


class MarketSpeedTracker:
    """Tracks how fast the market corrects after moves.

    Stores recent price changes with timestamps and calculates
    average reversal speed in cents per second.
    """

    def __init__(self, max_history: int = 100) -> None:
        self._prices: deque[tuple[Decimal, datetime]] = deque(maxlen=max_history)

    def add_price(self, price: Decimal, timestamp: datetime) -> None:
        """Record a price observation.

        Args:
            price: Observed price.
            timestamp: Time of observation.
        """
        self._prices.append((price, timestamp))

    def correction_speed(self) -> float:
        """Calculate average price reversal speed in cents/second.

        Returns:
            Average absolute price change per second, or 0.0 if insufficient data.
        """
        if len(self._prices) < 2:
            return 0.0

        total_change = 0.0
        total_seconds = 0.0

        prices = list(self._prices)
        for i in range(1, len(prices)):
            price_diff = abs(float(prices[i][0] - prices[i - 1][0]))
            time_diff = (prices[i][1] - prices[i - 1][1]).total_seconds()
            if time_diff > 0:
                total_change += price_diff
                total_seconds += time_diff

        if total_seconds <= 0:
            return 0.0

        speed = total_change / total_seconds
        log.debug("correction_speed_calculated", speed=speed)
        return speed

    def is_fast_market(self, threshold: float = 0.01) -> bool:
        """Determine if the market is moving faster than threshold.

        Args:
            threshold: Speed threshold in cents/second.

        Returns:
            True if market correction speed exceeds threshold.
        """
        return self.correction_speed() > threshold
