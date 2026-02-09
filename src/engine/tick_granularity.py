"""Tick granularity tracker — intra-candle early-move detection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from src.core.logging import get_logger

if TYPE_CHECKING:
    from datetime import datetime
    from decimal import Decimal

    from src.config.loader import ConfigLoader

log = get_logger(__name__)


class EarlyMoveResult(BaseModel):
    """Result of intra-candle early-move analysis."""

    early_half_return_pct: float
    direction: str  # "green" | "red" | "neutral"
    sustain_ratio: float
    confidence: float
    tick_count: int
    model_config = {"frozen": True}


class TickGranularityTracker:
    """Track tick-level OHLCV within each 1-minute candle.

    Detects early-move patterns by analyzing price action in the
    first N seconds of each candle to identify sustained directional
    moves before the candle closes.
    """

    def __init__(self, config: ConfigLoader) -> None:
        self._threshold_pct = float(
            config.get("strategy.singularity.early_move_threshold_pct", 0.08)
        )
        self._min_ticks = int(
            config.get("strategy.singularity.min_early_ticks", 15)
        )
        self._early_window_seconds = int(
            config.get("strategy.singularity.early_window_seconds", 30)
        )
        self._ticks: list[tuple[Decimal, datetime]] = []

    def on_tick(self, price: Decimal, timestamp: datetime) -> None:
        """Record a new tick price."""
        self._ticks.append((price, timestamp))

    def get_early_move(self) -> EarlyMoveResult:
        """Analyze early-half candle move at the configured window mark.

        Returns:
            EarlyMoveResult with direction, sustain ratio, and confidence.
            Neutral with zero confidence if insufficient ticks.
        """
        if len(self._ticks) < self._min_ticks:
            log.debug(
                "early_move_insufficient_ticks",
                tick_count=len(self._ticks),
                min_required=self._min_ticks,
            )
            return EarlyMoveResult(
                early_half_return_pct=0.0,
                direction="neutral",
                sustain_ratio=0.0,
                confidence=0.0,
                tick_count=len(self._ticks),
            )

        # Filter ticks within the early window
        candle_start = self._ticks[0][1]
        early_ticks = [
            t
            for t in self._ticks
            if (t[1] - candle_start).total_seconds() <= self._early_window_seconds
        ]

        if len(early_ticks) < self._min_ticks:
            return EarlyMoveResult(
                early_half_return_pct=0.0,
                direction="neutral",
                sustain_ratio=0.0,
                confidence=0.0,
                tick_count=len(early_ticks),
            )

        open_price = float(early_ticks[0][0])
        latest_price = float(early_ticks[-1][0])

        if open_price == 0.0:
            return EarlyMoveResult(
                early_half_return_pct=0.0,
                direction="neutral",
                sustain_ratio=0.0,
                confidence=0.0,
                tick_count=len(early_ticks),
            )

        early_return = (latest_price - open_price) / open_price * 100.0

        if early_return > 0:
            direction = "green"
        elif early_return < 0:
            direction = "red"
        else:
            direction = "neutral"

        # Compute sustain_ratio: fraction of consecutive ticks moving
        # in the same direction as the overall early move
        sustain_ratio = self._compute_sustain_ratio(early_ticks, direction)

        # Confidence = sustain_ratio * min(|early_return| / threshold, 1.0)
        if self._threshold_pct > 0:
            magnitude_factor = min(abs(early_return) / self._threshold_pct, 1.0)
        else:
            magnitude_factor = 1.0 if abs(early_return) > 0 else 0.0

        confidence = sustain_ratio * magnitude_factor

        log.debug(
            "early_move_detected",
            direction=direction,
            early_return_pct=early_return,
            sustain_ratio=sustain_ratio,
            confidence=confidence,
            tick_count=len(early_ticks),
        )

        return EarlyMoveResult(
            early_half_return_pct=early_return,
            direction=direction,
            sustain_ratio=sustain_ratio,
            confidence=confidence,
            tick_count=len(early_ticks),
        )

    def reset_minute(self) -> None:
        """Reset tracker for a new minute candle."""
        self._ticks.clear()

    def _compute_sustain_ratio(
        self,
        ticks: list[tuple[Decimal, datetime]],
        direction: str,
    ) -> float:
        """Compute fraction of tick-to-tick moves aligned with direction."""
        if len(ticks) < 2 or direction == "neutral":
            return 0.0

        aligned_count = 0
        total_moves = 0

        for i in range(1, len(ticks)):
            diff = float(ticks[i][0] - ticks[i - 1][0])
            if diff == 0.0:
                continue
            total_moves += 1
            if direction == "green" and diff > 0 or direction == "red" and diff < 0:
                aligned_count += 1

        if total_moves == 0:
            return 0.0

        return aligned_count / total_moves
