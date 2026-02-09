"""Trend analyzer — candle-based directional bias detection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.logging import get_logger
from src.models.market import Candle, CandleDirection
from src.models.signal import TrendDirection, TrendResult

if TYPE_CHECKING:
    from src.config.loader import ConfigLoader

log = get_logger(__name__)


class TrendAnalyzer:
    """Analyzes recent candles to determine market trend direction and strength."""

    def __init__(self, config: ConfigLoader) -> None:
        self._lookback = int(config.get("strategy.false_sentiment.lookback_candles", 5))

    def analyze(self, candles: list[Candle]) -> TrendResult:
        """Analyze the last N candles for trend direction.

        Args:
            candles: Recent candle history (newest last).

        Returns:
            TrendResult with direction, strength, and momentum.
        """
        window = candles[-self._lookback :] if len(candles) >= self._lookback else candles

        if not window:
            return TrendResult(direction=TrendDirection.NEUTRAL, strength=0.0)

        green_count = sum(1 for c in window if c.direction == CandleDirection.GREEN)
        red_count = sum(1 for c in window if c.direction == CandleDirection.RED)
        total = len(window)

        if green_count > red_count:
            direction = TrendDirection.UP
            dominant = green_count
        elif red_count > green_count:
            direction = TrendDirection.DOWN
            dominant = red_count
        else:
            direction = TrendDirection.NEUTRAL
            dominant = max(green_count, red_count)

        strength = dominant / total if total > 0 else 0.0

        # Momentum: average body_size / average range_size
        body_sizes: list[float] = []
        range_sizes: list[float] = []
        for c in window:
            body_sizes.append(float(c.body_size))
            rs = float(c.range_size)
            range_sizes.append(rs)

        avg_body = sum(body_sizes) / total
        avg_range = sum(range_sizes) / total
        momentum = avg_body / avg_range if avg_range > 0 else 0.0

        # Cumulative price move percentage
        first_open = float(window[0].open)
        last_close = float(window[-1].close)
        cumulative_move_pct = (
            ((last_close - first_open) / first_open * 100.0) if first_open != 0 else 0.0
        )

        log.debug(
            "trend_analyzed",
            direction=direction.value,
            strength=strength,
            green=green_count,
            red=red_count,
        )

        return TrendResult(
            direction=direction,
            strength=strength,
            momentum=momentum,
            green_count=green_count,
            red_count=red_count,
            cumulative_move_pct=cumulative_move_pct,
        )
