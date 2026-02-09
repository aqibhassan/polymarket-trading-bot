"""Dynamic entry threshold — time-scaled barrier for trade entry."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from src.core.logging import get_logger

if TYPE_CHECKING:
    from src.config.loader import ConfigLoader

log = get_logger(__name__)


class DynamicThreshold:
    """Entry threshold that increases with elapsed candle time.

    Formula: base + (time_scaling * minutes_elapsed / 15.0)
    """

    def __init__(self, config: ConfigLoader) -> None:
        self._base = Decimal(
            str(config.get("strategy.false_sentiment.entry_threshold_base", 0.59))
        )
        self._time_scaling = Decimal(
            str(config.get("strategy.false_sentiment.threshold_time_scaling", 0.15))
        )

    def calculate(self, minutes_elapsed: float) -> Decimal:
        """Calculate the entry threshold at a given time.

        Args:
            minutes_elapsed: Minutes since candle start.

        Returns:
            Dynamic threshold value.
        """
        elapsed = Decimal(str(minutes_elapsed))
        threshold = self._base + (self._time_scaling * elapsed / Decimal("15.0"))
        log.debug("threshold_calculated", minutes=minutes_elapsed, threshold=float(threshold))
        return threshold

    def should_enter(self, market_price: Decimal, minutes_elapsed: float) -> bool:
        """Check if market price exceeds the dynamic threshold.

        Args:
            market_price: Current market price.
            minutes_elapsed: Minutes since candle start.

        Returns:
            True if price exceeds threshold.
        """
        threshold = self.calculate(minutes_elapsed)
        return market_price >= threshold
