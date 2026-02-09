"""Liquidity filter — pre-trade market liquidity checks."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from src.core.logging import get_logger

if TYPE_CHECKING:
    from src.config.loader import ConfigLoader
    from src.models.market import OrderBookSnapshot

log = get_logger(__name__)


@dataclass(frozen=True)
class LiquidityResult:
    """Result of liquidity check."""

    passed: bool
    quality: float
    reason: str


class LiquidityFilter:
    """Pre-trade liquidity check: does the market have enough liquidity?"""

    def __init__(self, config: ConfigLoader) -> None:
        self._min_hourly_volume = Decimal(
            str(config.get("liquidity.min_hourly_volume", 100))
        )
        self._max_spread_cents = Decimal(
            str(config.get("liquidity.max_spread_cents", 0.05))
        )

    def check(
        self, orderbook: OrderBookSnapshot, hourly_volume: Decimal
    ) -> LiquidityResult:
        """Check if the market has sufficient liquidity for trading.

        Args:
            orderbook: Current order book snapshot.
            hourly_volume: Recent hourly trading volume.

        Returns:
            LiquidityResult indicating pass/fail with quality score.
        """
        spread = orderbook.spread
        if spread is None:
            log.warning("liquidity_check_failed", reason="no spread available")
            return LiquidityResult(passed=False, quality=0.0, reason="no spread available")

        if hourly_volume < self._min_hourly_volume:
            reason = (
                f"hourly volume {hourly_volume} below minimum {self._min_hourly_volume}"
            )
            log.info("liquidity_check_failed", reason=reason)
            return LiquidityResult(passed=False, quality=0.0, reason=reason)

        if spread > self._max_spread_cents:
            reason = f"spread {spread} exceeds max {self._max_spread_cents}"
            log.info("liquidity_check_failed", reason=reason)
            return LiquidityResult(passed=False, quality=0.0, reason=reason)

        # Quality score: blend of volume ratio and spread tightness
        volume_ratio = min(float(hourly_volume / self._min_hourly_volume), 2.0) / 2.0
        spread_score = max(
            0.0,
            1.0 - float(spread / self._max_spread_cents),
        )
        quality = (volume_ratio + spread_score) / 2.0

        log.debug("liquidity_check_passed", quality=quality)
        return LiquidityResult(passed=True, quality=quality, reason="")
