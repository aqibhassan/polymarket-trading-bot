"""Order book analyzer — spoofing detection and depth analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.core.logging import get_logger

if TYPE_CHECKING:
    from decimal import Decimal

    from src.config.loader import ConfigLoader
    from src.models.market import OrderBookSnapshot

log = get_logger(__name__)


@dataclass(frozen=True)
class BookAnalysis:
    """Result of order book analysis."""

    is_spoofed: bool
    normality_score: float
    bid_depth: Decimal
    ask_depth: Decimal
    imbalance_ratio: float


class BookAnalyzer:
    """Analyzes order book for manipulation signals and depth balance."""

    def __init__(self, config: ConfigLoader) -> None:
        self._heavy_multiplier = float(
            config.get("orderbook.heavy_book_multiplier", 3.0)
        )

    def analyze(self, orderbook: OrderBookSnapshot) -> BookAnalysis:
        """Analyze order book for spoofing and imbalance.

        Args:
            orderbook: Current order book snapshot.

        Returns:
            BookAnalysis with spoofing flag, normality score, and depth metrics.
        """
        bid_depth = orderbook.total_bid_depth
        ask_depth = orderbook.total_ask_depth

        bid_f = float(bid_depth)
        ask_f = float(ask_depth)
        total = bid_f + ask_f

        # Imbalance ratio: positive means bid-heavy, negative means ask-heavy
        imbalance_ratio = (bid_f - ask_f) / total if total > 0 else 0.0

        # Spoofing detection: one side has N times more depth than the other
        is_spoofed = False
        if (
            ask_f > 0 and bid_f / ask_f >= self._heavy_multiplier
            or bid_f > 0 and ask_f / bid_f >= self._heavy_multiplier
        ):
            is_spoofed = True

        # Normality score: 1.0 when balanced, 0.0 when extremely one-sided
        normality_score = 1.0 - abs(imbalance_ratio) if total > 0 else 0.0

        log.debug(
            "book_analyzed",
            bid_depth=float(bid_depth),
            ask_depth=float(ask_depth),
            imbalance=imbalance_ratio,
            spoofed=is_spoofed,
            normality=normality_score,
        )

        return BookAnalysis(
            is_spoofed=is_spoofed,
            normality_score=normality_score,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            imbalance_ratio=imbalance_ratio,
        )
