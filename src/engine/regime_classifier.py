"""Regime Classifier — real-time 3-state market regime detection.

Classifies current market into EXPANSION (trending), NEUTRAL, or
CONTRACTION (ranging) by comparing short-term vs long-term realized
volatility from 1-minute candles.
"""

from __future__ import annotations

import math
from typing import Any


class RegimeClassifier:
    """Real-time market regime detection."""

    EXPANSION = "expansion"
    NEUTRAL = "neutral"
    CONTRACTION = "contraction"

    def classify(
        self,
        candles_1m: list[Any],
        lookback_short: int = 20,
        lookback_long: int = 100,
    ) -> str:
        """Classify market regime from 1m candles."""
        if len(candles_1m) < lookback_short:
            return self.NEUTRAL

        short_vol = self._realized_vol(candles_1m[-lookback_short:])
        long_candles = candles_1m[-lookback_long:] if len(candles_1m) >= lookback_long else candles_1m
        long_vol = self._realized_vol(long_candles)

        if long_vol == 0:
            return self.NEUTRAL

        vol_ratio = short_vol / long_vol

        if vol_ratio > 1.3:
            return self.EXPANSION
        elif vol_ratio < 0.7:
            return self.CONTRACTION
        return self.NEUTRAL

    def _realized_vol(self, candles: list[Any]) -> float:
        """Compute realized volatility from log returns."""
        if len(candles) < 2:
            return 0.0
        returns: list[float] = []
        for i in range(1, len(candles)):
            c_prev = float(candles[i - 1].close)
            c_curr = float(candles[i].close)
            if c_prev > 0:
                returns.append(math.log(c_curr / c_prev))
        if len(returns) < 2:
            return 0.0
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        return math.sqrt(variance)
