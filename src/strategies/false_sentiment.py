"""False Sentiment Strategy — BTC 15m candle contrarian signal detection."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any, ClassVar

from src.core.logging import get_logger
from src.models.signal import Signal, SignalType
from src.strategies.base import BaseStrategy
from src.strategies.registry import register

if TYPE_CHECKING:
    from src.config.loader import ConfigLoader
    from src.models.market import MarketState, OrderBookSnapshot, Position, Side
    from src.models.order import Fill

# Engine imports — other teams deliver these in parallel.
# Graceful fallback if engine modules are not yet available.
try:
    from src.engine.trend_analyzer import TrendAnalyzer
except ImportError:  # pragma: no cover
    TrendAnalyzer = None  # type: ignore[assignment,misc]

try:
    from src.engine.dynamic_threshold import DynamicThreshold
except ImportError:  # pragma: no cover
    DynamicThreshold = None  # type: ignore[assignment,misc]

try:
    from src.engine.book_analyzer import BookAnalyzer
except ImportError:  # pragma: no cover
    BookAnalyzer = None  # type: ignore[assignment,misc]

try:
    from src.engine.liquidity_filter import LiquidityFilter
except ImportError:  # pragma: no cover
    LiquidityFilter = None  # type: ignore[assignment,misc]

try:
    from src.engine.confidence_scorer import ConfidenceScorer
except ImportError:  # pragma: no cover
    ConfidenceScorer = None  # type: ignore[assignment,misc]

try:
    from src.engine.exit_manager import ExitManager
except ImportError:  # pragma: no cover
    ExitManager = None  # type: ignore[assignment,misc]

logger = get_logger(__name__)


@register("false_sentiment")
class FalseSentimentStrategy(BaseStrategy):
    """BTC 15-minute false sentiment detection strategy.

    Detects when Polymarket prices diverge from BTC candle trends,
    identifies manipulation via order book analysis, and enters
    contrarian positions with strict risk controls.
    """

    REQUIRED_PARAMS: ClassVar[list[str]] = [
        "strategy.false_sentiment.entry_threshold_base",
        "strategy.false_sentiment.threshold_time_scaling",
        "strategy.false_sentiment.lookback_candles",
        "strategy.false_sentiment.min_confidence",
        "strategy.false_sentiment.max_hold_minutes",
        "strategy.false_sentiment.force_exit_minute",
        "strategy.false_sentiment.no_entry_after_minute",
    ]

    def __init__(self, config: ConfigLoader, strategy_id: str | None = None) -> None:
        super().__init__(config, strategy_id or "false_sentiment")
        self._min_confidence = float(
            config.get("strategy.false_sentiment.min_confidence", 0.6)
        )
        self._no_entry_after = float(
            config.get("strategy.false_sentiment.no_entry_after_minute", 8)
        )
        self._force_exit_minute = int(
            config.get("strategy.false_sentiment.force_exit_minute", 14)
        )
        self._profit_target_pct = Decimal(
            str(config.get("exit.profit_target_pct", 0.05))
        )
        self._hard_stop_loss_pct = Decimal(
            str(config.get("exit.hard_stop_loss_pct", 0.04))
        )

        # Wire engine components (None if not yet available)
        self._trend_analyzer = TrendAnalyzer(config) if TrendAnalyzer is not None else None
        self._threshold = DynamicThreshold(config) if DynamicThreshold is not None else None
        self._book_analyzer = BookAnalyzer(config) if BookAnalyzer is not None else None
        self._liquidity_filter = LiquidityFilter(config) if LiquidityFilter is not None else None
        self._confidence_scorer = ConfidenceScorer(config) if ConfidenceScorer is not None else None
        self._exit_manager = ExitManager(config) if ExitManager is not None else None

    def generate_signals(
        self,
        market_state: MarketState,
        orderbook: OrderBookSnapshot,
        context: dict[str, Any],
    ) -> list[Signal]:
        """Generate trading signals for the false sentiment strategy.

        Pipeline:
        1. Check open positions for exit conditions
        2. Time-gate check
        3. Analyze BTC trend from candles
        4. Check dynamic entry threshold
        5. Analyze order book
        6. Check liquidity
        7. Score confidence
        8. Emit ENTRY or SKIP signal
        """
        signals: list[Signal] = []
        market_id = market_state.market_id
        current_price = market_state.dominant_price
        direction = market_state.dominant_side
        candles = context.get("candles", [])
        hourly_volume = Decimal(str(context.get("hourly_volume", 0)))

        # --- Exit checks for open positions ---
        for pos in self.open_positions:
            exit_signal = self._check_exit(pos, current_price, market_state)
            if exit_signal is not None:
                signals.append(exit_signal)

        # --- Entry pipeline ---
        minutes_elapsed = market_state.minutes_elapsed

        # Time gate
        if minutes_elapsed > self._no_entry_after:
            logger.info("skip_time_gate", market_id=market_id, minutes=minutes_elapsed)
            signals.append(self._skip(market_id, direction, "past no_entry_after"))
            return signals

        # Trend analysis
        if self._trend_analyzer is None:
            logger.warning("trend_analyzer_unavailable")
            signals.append(self._skip(market_id, direction, "engine not loaded"))
            return signals

        trend = self._trend_analyzer.analyze(candles)

        # Dynamic threshold
        if self._threshold is None or not self._threshold.should_enter(
            current_price, minutes_elapsed
        ):
            signals.append(self._skip(market_id, direction, "below threshold"))
            return signals

        threshold = self._threshold.calculate(minutes_elapsed)
        threshold_exceedance = float(current_price - threshold)

        # Book analysis
        if self._book_analyzer is None:
            signals.append(self._skip(market_id, direction, "book analyzer unavailable"))
            return signals

        book_result = self._book_analyzer.analyze(orderbook)

        # Liquidity
        if self._liquidity_filter is None:
            signals.append(self._skip(market_id, direction, "liquidity filter unavailable"))
            return signals

        liq_result = self._liquidity_filter.check(orderbook, hourly_volume)
        if not liq_result.passed:
            signals.append(self._skip(market_id, direction, f"liquidity: {liq_result.reason}"))
            return signals

        # Confidence
        if self._confidence_scorer is None:
            signals.append(self._skip(market_id, direction, "confidence scorer unavailable"))
            return signals

        confidence = self._confidence_scorer.score(
            trend=trend,
            book_normality=book_result.normality_score,
            liquidity_quality=liq_result.quality,
            threshold_exceedance=threshold_exceedance,
        )

        if not confidence.meets_minimum(self._min_confidence):
            logger.info("skip_low_confidence", confidence=confidence.overall)
            signals.append(self._skip(market_id, direction, "low confidence"))
            return signals

        # ENTRY signal
        entry_price = current_price
        stop_loss = entry_price - self._hard_stop_loss_pct
        take_profit = entry_price + self._profit_target_pct

        logger.info(
            "signal_entry",
            market_id=market_id,
            entry=float(entry_price),
            confidence=confidence.overall,
        )

        signals.append(
            Signal(
                strategy_id=self.strategy_id,
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
                },
            )
        )
        return signals

    def on_fill(self, fill: Fill, position: Position) -> None:
        """Log and track a fill event."""
        logger.info(
            "strategy_fill",
            strategy=self.strategy_id,
            order_id=str(fill.order_id),
            price=float(fill.price),
            size=float(fill.size),
            market_id=position.market_id,
        )

    def on_cancel(self, order_id: str, reason: str) -> None:
        """Log an order cancellation."""
        logger.info(
            "strategy_cancel",
            strategy=self.strategy_id,
            order_id=order_id,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_exit(
        self, position: Position, current_price: Decimal, market_state: MarketState
    ) -> Signal | None:
        """Check if a position should be exited.

        Resolution guard fires at force_exit_minute regardless of ExitManager
        availability — prevents holding past binary settlement.
        """
        from src.models.signal import ExitReason

        # Resolution guard — always active, independent of ExitManager
        minutes_elapsed = market_state.minutes_elapsed
        if minutes_elapsed >= self._force_exit_minute:
            logger.info(
                "signal_exit",
                market_id=position.market_id,
                reason="resolution_guard",
                minute=minutes_elapsed,
                price=float(current_price),
            )
            return Signal(
                strategy_id=self.strategy_id,
                market_id=position.market_id,
                signal_type=SignalType.EXIT,
                direction=position.side,
                exit_reason=ExitReason.RESOLUTION_GUARD,
                metadata={"exit_price": str(current_price), "minute": str(minutes_elapsed)},
            )

        if self._exit_manager is None:
            return None

        should_exit, reason = self._exit_manager.should_exit(
            position, current_price, market_state
        )
        if not should_exit or reason is None:
            return None

        logger.info(
            "signal_exit",
            market_id=position.market_id,
            reason=reason.value,
            price=float(current_price),
        )
        return Signal(
            strategy_id=self.strategy_id,
            market_id=position.market_id,
            signal_type=SignalType.EXIT,
            direction=position.side,
            exit_reason=reason,
            metadata={"exit_price": str(current_price)},
        )

    def _skip(self, market_id: str, direction: Side, reason: str) -> Signal:
        """Create a SKIP signal."""
        return Signal(
            strategy_id=self.strategy_id,
            market_id=market_id,
            signal_type=SignalType.SKIP,
            direction=direction,
            metadata={"skip_reason": reason},
        )
