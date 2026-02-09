"""Signal generator — orchestrates analyzers into final trading signal."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from src.core.logging import get_logger
from src.engine.book_analyzer import BookAnalyzer
from src.engine.confidence_scorer import ConfidenceScorer
from src.engine.dynamic_threshold import DynamicThreshold
from src.engine.liquidity_filter import LiquidityFilter
from src.engine.trend_analyzer import TrendAnalyzer
from src.models.signal import Signal, SignalType

if TYPE_CHECKING:
    from src.config.loader import ConfigLoader
    from src.models.market import Candle, MarketState, OrderBookSnapshot

log = get_logger(__name__)


class FalseSentimentSignalGenerator:
    """Orchestrator: combines all analyzers into a final Signal.

    Flow:
    1. Analyze trend
    2. Check dynamic threshold
    3. Analyze order book
    4. Check liquidity
    5. Score confidence
    6. Generate ENTRY or SKIP signal
    """

    def __init__(self, config: ConfigLoader) -> None:
        self._config = config
        self._trend_analyzer = TrendAnalyzer(config)
        self._threshold = DynamicThreshold(config)
        self._book_analyzer = BookAnalyzer(config)
        self._liquidity_filter = LiquidityFilter(config)
        self._confidence_scorer = ConfidenceScorer(config)

        self._min_confidence = float(
            config.get("strategy.false_sentiment.min_confidence", 0.6)
        )
        self._no_entry_after = float(
            config.get("strategy.false_sentiment.no_entry_after_minute", 8)
        )
        self._profit_target_pct = Decimal(
            str(config.get("exit.profit_target_pct", 0.05))
        )
        self._hard_stop_loss_pct = Decimal(
            str(config.get("exit.hard_stop_loss_pct", 0.04))
        )

    def generate(
        self,
        candles: list[Candle],
        market_state: MarketState,
        orderbook: OrderBookSnapshot,
        hourly_volume: Decimal,
    ) -> Signal:
        """Generate a trading signal from market data.

        Args:
            candles: Recent candle history.
            market_state: Current market state.
            orderbook: Current order book snapshot.
            hourly_volume: Recent hourly volume.

        Returns:
            Signal with ENTRY or SKIP type.
        """
        market_id = market_state.market_id
        minutes_elapsed = market_state.minutes_elapsed
        dominant_price = market_state.dominant_price
        direction = market_state.dominant_side

        # Time gate: no entry after configured minute
        if minutes_elapsed > self._no_entry_after:
            log.info("signal_skip_time_gate", market_id=market_id, minutes=minutes_elapsed)
            return self._skip_signal(market_id, direction, "time gate exceeded")

        # Trend analysis
        trend = self._trend_analyzer.analyze(candles)

        # Dynamic threshold check
        threshold = self._threshold.calculate(minutes_elapsed)
        if dominant_price < threshold:
            log.info(
                "signal_skip_threshold",
                market_id=market_id,
                price=float(dominant_price),
                threshold=float(threshold),
            )
            return self._skip_signal(market_id, direction, "below threshold")

        max_expected_exceedance = float(
            self._config.get("confidence.max_expected_exceedance", 0.20)
        )
        raw_exceedance = float(dominant_price - threshold)
        threshold_exceedance = min(raw_exceedance / max_expected_exceedance, 1.0)

        # Book analysis
        book_analysis = self._book_analyzer.analyze(orderbook)

        # Liquidity check
        liquidity = self._liquidity_filter.check(orderbook, hourly_volume)
        if not liquidity.passed:
            log.info("signal_skip_liquidity", market_id=market_id, reason=liquidity.reason)
            return self._skip_signal(market_id, direction, f"liquidity: {liquidity.reason}")

        # Confidence scoring
        confidence = self._confidence_scorer.score(
            trend=trend,
            book_normality=book_analysis.normality_score,
            liquidity_quality=liquidity.quality,
            threshold_exceedance=threshold_exceedance,
        )

        if not confidence.meets_minimum(self._min_confidence):
            log.info(
                "signal_skip_confidence",
                market_id=market_id,
                confidence=confidence.overall,
                minimum=self._min_confidence,
            )
            return self._skip_signal(market_id, direction, "low confidence")

        # Generate ENTRY signal
        entry_price = dominant_price
        stop_loss = entry_price - self._hard_stop_loss_pct
        take_profit = entry_price + self._profit_target_pct

        log.info(
            "signal_entry",
            market_id=market_id,
            entry=float(entry_price),
            stop=float(stop_loss),
            take_profit=float(take_profit),
            confidence=confidence.overall,
        )

        return Signal(
            strategy_id="false_sentiment",
            market_id=market_id,
            signal_type=SignalType.ENTRY,
            direction=direction,
            strength=Decimal(str(round(confidence.overall, 4))),
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                "trend_direction": trend.direction.value,
                "trend_strength": str(trend.strength),
                "book_normality": str(book_analysis.normality_score),
                "threshold": str(float(threshold)),
            },
        )

    def _skip_signal(
        self,
        market_id: str,
        direction: object,
        reason: str,
    ) -> Signal:
        """Create a SKIP signal."""
        from src.models.market import Side

        side = direction if isinstance(direction, Side) else Side.YES
        return Signal(
            strategy_id="false_sentiment",
            market_id=market_id,
            signal_type=SignalType.SKIP,
            direction=side,
            metadata={"skip_reason": reason},
        )
