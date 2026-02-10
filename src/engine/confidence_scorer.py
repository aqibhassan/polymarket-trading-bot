"""Confidence scorer — multi-factor weighted signal confidence."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.logging import get_logger
from src.models.signal import Confidence, TrendResult

if TYPE_CHECKING:
    from src.config.loader import ConfigLoader

log = get_logger(__name__)


class ConfidenceScorer:
    """Multi-factor weighted confidence scoring for trade signals.

    Weights (configurable):
    - trend: 0.35
    - book: 0.25
    - liquidity: 0.20
    - threshold: 0.20
    """

    def __init__(self, config: ConfigLoader) -> None:
        self._w_trend = float(config.get("confidence.weight_trend", 0.35))
        self._w_book = float(config.get("confidence.weight_book", 0.25))
        self._w_liquidity = float(config.get("confidence.weight_liquidity", 0.20))
        self._w_threshold = float(config.get("confidence.weight_threshold", 0.20))

    def score(
        self,
        trend: TrendResult,
        book_normality: float,
        liquidity_quality: float,
        threshold_exceedance: float,
    ) -> Confidence:
        """Calculate multi-factor confidence score.

        Args:
            trend: Trend analysis result.
            book_normality: Order book normality score (0-1).
            liquidity_quality: Liquidity quality score (0-1).
            threshold_exceedance: How much price exceeds threshold (0+).

        Returns:
            Confidence model with individual scores and overall.
        """
        trend_score = _clamp(trend.strength)
        book_score = _clamp(book_normality)
        liq_score = _clamp(liquidity_quality)
        thresh_score = _clamp(threshold_exceedance)

        overall = (
            self._w_trend * trend_score
            + self._w_book * book_score
            + self._w_liquidity * liq_score
            + self._w_threshold * thresh_score
        )
        overall = _clamp(overall)

        log.debug(
            "confidence_scored",
            trend=trend_score,
            book=book_score,
            liquidity=liq_score,
            threshold=thresh_score,
            overall=overall,
        )

        return Confidence(
            trend_strength=trend_score,
            book_normality=book_score,
            liquidity_quality=liq_score,
            threshold_exceedance=thresh_score,
            overall=overall,
        )


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a value between low and high."""
    return max(low, min(high, value))
