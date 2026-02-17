"""Singularity Ensemble Strategy — combines 8 signal sources for defensible edge.

Signals:
  1. Momentum Confirmation (25% weight) — BTC 15m cumulative return direction
  2. Order Flow Imbalance (15% weight) — order book bid/ask imbalance
  3. Futures Lead-Lag (10% weight) — Binance perps leading spot/Polymarket
  4. Volatility Regime (5% weight) — realized vs implied vol mismatch
  5. Time-of-Day (5% weight) — hourly seasonality adjustment
  6. VPIN (20% weight) — Volume-Synchronized Probability of Informed Trading
  7. OI Delta (15% weight) — Open Interest change confirmation
  8. Window Memory (5% weight) — Multi-window serial correlation

Entry requires minimum 3 of 8 signals to agree on direction.
Exit on 2+ signal reversals or resolution guard at minute 12 (tighter).
Dynamic position sizing via Kelly * signal_count_mult * time_mult * vol_mult.
Regime-adaptive weighting: EXPANSION boosts momentum+VPIN, CONTRACTION boosts OFI+memory.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any, ClassVar

from src.core.logging import get_logger
from src.models.signal import Confidence, ExitReason, Signal, SignalType
from src.strategies.base import BaseStrategy
from src.strategies.registry import register

if TYPE_CHECKING:
    from src.config.loader import ConfigLoader
    from src.models.market import MarketState, OrderBookSnapshot, Position, Side
    from src.models.order import Fill

logger = get_logger(__name__)


class _SignalVote:
    """Internal: a single signal source's directional vote."""

    __slots__ = ("name", "direction", "strength", "weight")

    def __init__(
        self, name: str, direction: str, strength: float, weight: float
    ) -> None:
        self.name = name
        self.direction = direction  # "YES", "NO", or "neutral"
        self.strength = strength  # 0-1 confidence
        self.weight = weight


@register("singularity")
class SingularityStrategy(BaseStrategy):
    """Ensemble strategy combining 8 orthogonal signal sources.

    Gracefully degrades when signal sources are unavailable: if fewer
    than 8 sources produce a vote, the minimum agreement count and
    weights are adjusted proportionally. Supports regime-adaptive
    weighting (EXPANSION/NEUTRAL/CONTRACTION).
    """

    REQUIRED_PARAMS: ClassVar[list[str]] = [
        "strategy.singularity.min_confidence",
        "strategy.singularity.min_signals_agree",
    ]

    def __init__(self, config: ConfigLoader, strategy_id: str | None = None) -> None:
        super().__init__(config, strategy_id or "singularity")

        # --- Signal weights (must sum to 1.0) ---
        self._w_momentum = float(config.get("strategy.singularity.weight_momentum", 0.40))
        self._w_ofi = float(config.get("strategy.singularity.weight_ofi", 0.25))
        self._w_futures = float(config.get("strategy.singularity.weight_futures", 0.15))
        self._w_vol = float(config.get("strategy.singularity.weight_vol", 0.10))
        self._w_time = float(config.get("strategy.singularity.weight_time", 0.10))
        self._w_vpin = float(config.get("strategy.singularity.weight_vpin", 0.20))
        self._w_oi_delta = float(config.get("strategy.singularity.weight_oi_delta", 0.15))
        self._w_window_memory = float(config.get("strategy.singularity.weight_window_memory", 0.05))

        # --- Default weights snapshot (for regime reset) ---
        self._default_weights = {
            "momentum": self._w_momentum,
            "ofi": self._w_ofi,
            "futures": self._w_futures,
            "vol": self._w_vol,
            "time": self._w_time,
            "vpin": self._w_vpin,
            "oi_delta": self._w_oi_delta,
            "window_memory": self._w_window_memory,
        }

        # --- Entry parameters ---
        self._min_signals_agree = int(config.get("strategy.singularity.min_signals_agree", 3))
        self._min_confidence = float(config.get("strategy.singularity.min_confidence", 0.72))
        self._entry_minute_start = int(config.get("strategy.singularity.entry_minute_start", 6))
        self._entry_minute_end = int(config.get("strategy.singularity.entry_minute_end", 10))

        # --- Contrarian filter ---
        self._skip_contrarian = bool(
            config.get("strategy.singularity.skip_contrarian", True)
        )
        self._contrarian_threshold_pct = float(
            config.get("strategy.singularity.contrarian_threshold_pct", 0.0)
        )

        # --- Momentum veto ---
        self._momentum_veto_enabled = bool(
            config.get("strategy.singularity.momentum_veto_enabled", True)
        )
        self._momentum_veto_min_strength = float(
            config.get("strategy.singularity.momentum_veto_min_strength", 0.15)
        )

        # --- Momentum thresholds (tiered by minute) ---
        raw_tiers = config.get("strategy.singularity.entry_tiers", None)
        if raw_tiers and isinstance(raw_tiers, list):
            self._tier_thresholds: dict[int, float] = {
                int(t["minute"]): float(t["threshold_pct"]) for t in raw_tiers
            }
        else:
            self._tier_thresholds = {8: 0.14, 9: 0.12, 10: 0.09, 11: 0.07, 12: 0.05}

        # --- Exit parameters ---
        self._resolution_guard_minute = int(
            config.get("strategy.singularity.resolution_guard_minute", 12)
        )
        self._exit_reversal_count = int(
            config.get("strategy.singularity.exit_reversal_count", 2)
        )
        self._max_position_pct = float(
            config.get("strategy.singularity.max_position_pct", 0.035)
        )

        # --- Stop loss (for risk manager compatibility) ---
        self._stop_loss_pct = float(
            config.get("strategy.singularity.stop_loss_pct", 1.00)
        )
        self._profit_target_pct = float(
            config.get("strategy.singularity.profit_target_pct", 0.40)
        )

        # --- Lazy-loaded signal analyzers ---
        self._ofi_analyzer: Any = None
        self._futures_detector: Any = None
        self._tick_tracker: Any = None
        self._vol_detector: Any = None
        self._time_analyzer: Any = None
        self._init_analyzers(config)

        # --- Runtime state ---
        self._entry_minutes: dict[str, int] = {}
        self._entry_votes: dict[str, list[_SignalVote]] = {}
        self._last_evaluation: dict[str, Any] = {}

    def _init_analyzers(self, config: ConfigLoader) -> None:
        """Initialize signal analyzers, gracefully skipping unavailable ones."""
        try:
            from src.engine.order_flow_analyzer import OrderFlowAnalyzer
            self._ofi_analyzer = OrderFlowAnalyzer(config)
        except ImportError:
            logger.warning("singularity_missing_module", module="order_flow_analyzer")

        try:
            from src.data.binance_futures_ws import FuturesLeadLagDetector
            self._futures_detector = FuturesLeadLagDetector(config)
        except ImportError:
            logger.warning("singularity_missing_module", module="binance_futures_ws")

        try:
            from src.engine.tick_granularity import TickGranularityTracker
            self._tick_tracker = TickGranularityTracker(config)
        except ImportError:
            logger.warning("singularity_missing_module", module="tick_granularity")

        try:
            from src.engine.volatility_regime import VolatilityRegimeDetector
            self._vol_detector = VolatilityRegimeDetector(config)
        except ImportError:
            logger.warning("singularity_missing_module", module="volatility_regime")

        try:
            from src.engine.time_of_day import TimeOfDayAnalyzer
            self._time_analyzer = TimeOfDayAnalyzer(config)
        except ImportError:
            logger.warning("singularity_missing_module", module="time_of_day")

    def generate_signals(
        self,
        market_state: MarketState,
        orderbook: OrderBookSnapshot,
        context: dict[str, Any],
    ) -> list[Signal]:
        """Generate trading signals from the ensemble of 5 signal sources.

        Expected context keys:
            candles_1m: list of 1m Candle objects for the current 15m window
            window_open_price: BTC price at window start
            minute_in_window: current minute within the 15m window (0-14)
            yes_price: current Polymarket YES token price
            futures_price: current Binance futures price (optional)
            spot_price: current Binance spot price (optional)
            orderbook_history: list of recent OrderBookSnapshot (optional)
            recent_ticks: list of recent tick prices as Decimal (optional)
        """
        signals: list[Signal] = []
        market_id = market_state.market_id

        candles_1m = context.get("candles_1m", [])
        window_open_price = float(context.get("window_open_price", 0))
        minute_in_window = int(context.get("minute_in_window", 0))
        yes_price = float(context.get("yes_price", 0.5))

        # --- Exit checks for open positions ---
        for pos in self.open_positions:
            exit_signal = self._check_exit(
                pos, market_state, orderbook, context, minute_in_window,
            )
            if exit_signal is not None:
                signals.append(exit_signal)

        # --- Entry pipeline ---
        if minute_in_window < self._entry_minute_start:
            self._last_evaluation = {"outcome": "outside_window", "reason": "before_entry_window", "minute": minute_in_window, "market_id": market_id}
            return signals

        if minute_in_window > self._entry_minute_end:
            self._last_evaluation = {"outcome": "outside_window", "reason": "after_entry_window", "minute": minute_in_window, "market_id": market_id}
            return signals

        if not candles_1m or window_open_price <= 0:
            self._last_evaluation = {"outcome": "outside_window", "reason": "no_candle_data", "minute": minute_in_window, "market_id": market_id}
            return signals

        # --- Collect votes from all signal sources ---
        votes: list[_SignalVote] = []

        # Signal 1: Momentum Confirmation
        momentum_vote = self._vote_momentum(
            candles_1m, window_open_price, minute_in_window,
        )
        if momentum_vote is not None:
            votes.append(momentum_vote)

        # Signal 2: Order Flow Imbalance
        ofi_vote = self._vote_ofi(orderbook, context)
        if ofi_vote is not None:
            votes.append(ofi_vote)

        # Signal 3: Futures Lead-Lag
        futures_vote = self._vote_futures(context, yes_price)
        if futures_vote is not None:
            votes.append(futures_vote)

        # Signal 4: Volatility Regime
        vol_vote = self._vote_vol_regime(context, yes_price, market_state)
        if vol_vote is not None:
            votes.append(vol_vote)

        # Signal 5: Time-of-Day
        time_vote = self._vote_time_of_day()
        if time_vote is not None:
            votes.append(time_vote)

        # Signal 6: VPIN
        vpin_vote = self._vote_vpin(context)
        if vpin_vote is not None:
            votes.append(vpin_vote)

        # Signal 7: OI Delta
        oi_vote = self._vote_oi_delta(context)
        if oi_vote is not None:
            votes.append(oi_vote)

        # Signal 8: Window Memory
        memory_vote = self._vote_window_memory(context)
        if memory_vote is not None:
            votes.append(memory_vote)

        # --- Regime-adaptive weighting ---
        regime = context.get("market_regime", "neutral")
        self._apply_regime_weights(regime)

        # Build vote details for dashboard (all 8 slots, voted or not)
        _vote_results: list[dict[str, Any]] = []
        for _sv in [momentum_vote, ofi_vote, futures_vote, vol_vote, time_vote, vpin_vote, oi_vote, memory_vote]:
            if _sv is not None:
                _vote_results.append({"name": _sv.name, "direction": _sv.direction, "strength": round(_sv.strength, 4)})

        if not votes:
            self._last_evaluation = {"outcome": "skip", "reason": "no_votes", "minute": minute_in_window, "market_id": market_id, "vote_details": _vote_results}
            return signals

        # --- Aggregate votes ---
        yes_votes = [v for v in votes if v.direction == "YES"]
        no_votes = [v for v in votes if v.direction == "NO"]
        neutral_votes = [v for v in votes if v.direction == "neutral"]

        # Determine majority direction (neutral votes don't count for direction)
        yes_count = len(yes_votes)
        no_count = len(no_votes)

        # Dynamically adjust min agreement: neutral votes don't count toward
        # the directional agreement requirement (they boost confidence only).
        # Floor at 2 to always require multi-source confirmation.
        directional_count = yes_count + no_count
        available_sources = len(votes)
        effective_min = max(min(self._min_signals_agree, directional_count), 2)

        if yes_count >= effective_min:
            direction_str = "YES"
            # Neutral votes boost confidence for the majority side
            agreeing_votes = yes_votes + neutral_votes
        elif no_count >= effective_min:
            direction_str = "NO"
            agreeing_votes = no_votes + neutral_votes
        else:
            logger.debug(
                "singularity_insufficient_agreement",
                market_id=market_id,
                yes=yes_count,
                no=no_count,
                neutral=len(neutral_votes),
                required=effective_min,
            )
            self._last_evaluation = {
                "outcome": "skip", "reason": "insufficient_agreement",
                "minute": minute_in_window, "market_id": market_id,
                "detail": f"YES={yes_count} NO={no_count} neutral={len(neutral_votes)} need={effective_min}",
                "votes": {"yes": yes_count, "no": no_count, "neutral": len(neutral_votes), "required": effective_min},
                "vote_details": _vote_results,
            }
            return signals

        # --- Contrarian filter: skip bets against BTC trend ---
        if self._skip_contrarian and candles_1m and window_open_price > 0:
            current_close = float(candles_1m[-1].close)
            cum_return = (current_close - window_open_price) / window_open_price
            cum_return_pct = cum_return * 100.0  # signed

            is_contrarian = (
                (direction_str == "YES" and cum_return_pct < -self._contrarian_threshold_pct)
                or (direction_str == "NO" and cum_return_pct > self._contrarian_threshold_pct)
            )
            if is_contrarian:
                logger.info(
                    "singularity_skip_contrarian",
                    market_id=market_id,
                    direction=direction_str,
                    cum_return_pct=round(cum_return_pct, 4),
                    threshold=self._contrarian_threshold_pct,
                )
                self._last_evaluation = {
                    "outcome": "skip", "reason": "contrarian_filter",
                    "minute": minute_in_window, "market_id": market_id,
                    "direction": direction_str,
                    "detail": f"dir={direction_str} cum_return={cum_return_pct:+.4f}% threshold={self._contrarian_threshold_pct}%",
                    "vote_details": _vote_results,
                }
                return signals

        # --- Momentum veto: momentum actively disagrees with majority ---
        if (
            self._momentum_veto_enabled
            and momentum_vote is not None
            and momentum_vote.direction != "neutral"
            and momentum_vote.direction != direction_str
            and momentum_vote.strength >= self._momentum_veto_min_strength
        ):
            logger.info(
                "singularity_momentum_veto",
                market_id=market_id,
                direction=direction_str,
                momentum_direction=momentum_vote.direction,
                momentum_strength=round(momentum_vote.strength, 4),
                veto_threshold=self._momentum_veto_min_strength,
            )
            self._last_evaluation = {
                "outcome": "skip", "reason": "momentum_veto",
                "minute": minute_in_window, "market_id": market_id,
                "direction": direction_str,
                "detail": f"momentum={momentum_vote.direction}({momentum_vote.strength:.4f}) vetoes {direction_str}",
                "vote_details": _vote_results,
            }
            return signals

        # --- Compute weighted confidence ---
        total_weight = sum(v.weight for v in agreeing_votes)
        if total_weight > 0:
            weighted_confidence = sum(
                v.strength * v.weight for v in agreeing_votes
            ) / total_weight
        else:
            weighted_confidence = 0.0

        # Signal count bonus: more agreeing signals = higher confidence
        signal_count_mult = min(len(agreeing_votes) / 3.0, 1.5)
        overall_confidence = min(weighted_confidence * signal_count_mult, 1.0)

        if overall_confidence < self._min_confidence:
            logger.debug(
                "singularity_low_confidence",
                confidence=overall_confidence,
                min_required=self._min_confidence,
            )
            self._last_evaluation = {
                "outcome": "skip", "reason": "low_confidence",
                "minute": minute_in_window, "market_id": market_id,
                "direction": direction_str,
                "confidence": round(overall_confidence, 4),
                "detail": f"confidence={overall_confidence:.4f} < threshold={self._min_confidence}",
                "vote_details": _vote_results,
            }
            return signals

        # --- All gates passed: generate ENTRY signal ---
        from src.models.market import Side

        if direction_str == "YES":
            direction = Side.YES
            entry_price = Decimal(str(yes_price))
        else:
            direction = Side.NO
            entry_price = Decimal(str(1.0 - yes_price))

        stop_loss = entry_price * (Decimal("1") - Decimal(str(self._stop_loss_pct)))
        take_profit = entry_price * (Decimal("1") + Decimal(str(self._profit_target_pct)))

        self._entry_minutes[market_id] = minute_in_window
        self._entry_votes[market_id] = votes

        # Prevent unbounded memory growth: keep only last 200 entries
        if len(self._entry_minutes) > 200:
            oldest_keys = list(self._entry_minutes.keys())[:-100]
            for k in oldest_keys:
                self._entry_minutes.pop(k, None)
                self._entry_votes.pop(k, None)

        # Build metadata
        vote_summary = ", ".join(
            f"{v.name}={v.direction}({v.strength:.2f})" for v in votes
        )
        confidence_obj = Confidence(
            trend_strength=round(momentum_vote.strength if momentum_vote else 0.0, 4),
            threshold_exceedance=round(overall_confidence, 4),
            book_normality=round(ofi_vote.strength if ofi_vote else 0.0, 4),
            liquidity_quality=round(time_vote.strength if time_vote else 0.0, 4),
            overall=round(overall_confidence, 4),
        )

        self._last_evaluation = {
            "outcome": "entry", "reason": "entry_signal",
            "minute": minute_in_window, "market_id": market_id,
            "direction": direction_str,
            "confidence": round(overall_confidence, 4),
            "detail": f"{len(agreeing_votes)}/{available_sources} signals agree",
            "votes": {"yes": yes_count, "no": no_count, "neutral": len(neutral_votes)},
            "vote_details": _vote_results,
        }

        logger.info(
            "singularity_entry",
            market_id=market_id,
            direction=direction_str,
            confidence=overall_confidence,
            agreeing=len(agreeing_votes),
            total_sources=available_sources,
            votes=vote_summary,
        )

        signals.append(Signal(
            strategy_id=self.strategy_id,
            market_id=market_id,
            signal_type=SignalType.ENTRY,
            direction=direction,
            strength=Decimal(str(round(overall_confidence, 4))),
            confidence=confidence_obj,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                "direction": direction_str.lower(),
                "agreeing_signals": str(len(agreeing_votes)),
                "total_sources": str(available_sources),
                "votes": vote_summary,
                "entry_minute": str(minute_in_window),
            },
        ))
        return signals

    # ------------------------------------------------------------------
    # Individual signal voters
    # ------------------------------------------------------------------

    def _vote_momentum(
        self,
        candles_1m: list[Any],
        window_open_price: float,
        minute: int,
    ) -> _SignalVote | None:
        """Signal 1: BTC cumulative return direction."""
        threshold_pct = self._tier_thresholds.get(minute)
        if threshold_pct is None:
            return None

        current_close = float(candles_1m[-1].close)
        cum_return = (current_close - window_open_price) / window_open_price
        cum_return_pct = abs(cum_return) * 100.0

        if cum_return_pct < threshold_pct:
            return None

        # Skip if no directional signal (exactly flat)
        if cum_return == 0:
            return None

        direction = "YES" if cum_return > 0 else "NO"
        strength = min(cum_return_pct / 0.25, 1.0)

        return _SignalVote(
            name="momentum",
            direction=direction,
            strength=strength,
            weight=self._w_momentum,
        )

    def _vote_ofi(
        self,
        orderbook: OrderBookSnapshot,
        context: dict[str, Any],
    ) -> _SignalVote | None:
        """Signal 2: Order flow imbalance."""
        if self._ofi_analyzer is None:
            return None

        history = context.get("orderbook_history")
        result = self._ofi_analyzer.analyze(orderbook, history)

        if result.direction == "neutral":
            return None

        direction = "YES" if result.direction == "buy_pressure" else "NO"
        return _SignalVote(
            name="ofi",
            direction=direction,
            strength=result.signal_strength * result.confidence,
            weight=self._w_ofi,
        )

    def _vote_futures(
        self,
        context: dict[str, Any],
        yes_price: float,
    ) -> _SignalVote | None:
        """Signal 3: Binance futures lead-lag."""
        if self._futures_detector is None:
            return None

        futures_price = context.get("futures_price")
        spot_price = context.get("spot_price")
        futures_velocity = context.get("futures_velocity_pct_per_min", 0.0)

        if futures_price is None or spot_price is None:
            return None

        result = self._futures_detector.detect(
            futures_price=Decimal(str(futures_price)),
            spot_price=Decimal(str(spot_price)),
            polymarket_yes_price=Decimal(str(yes_price)),
            futures_velocity_pct_per_min=float(futures_velocity),
        )

        if result.direction == "neutral" or result.signal_strength == 0.0:
            return None

        direction = "YES" if result.direction == "long" else "NO"
        return _SignalVote(
            name="futures",
            direction=direction,
            strength=result.signal_strength,
            weight=self._w_futures,
        )

    def _vote_vol_regime(
        self,
        context: dict[str, Any],
        yes_price: float,
        market_state: MarketState,
    ) -> _SignalVote | None:
        """Signal 4: Volatility regime detection."""
        if self._vol_detector is None:
            return None

        recent_ticks = context.get("recent_ticks")
        if not recent_ticks or len(recent_ticks) < 2:
            return None

        result = self._vol_detector.detect_regime(
            recent_ticks=recent_ticks,
            yes_price=Decimal(str(yes_price)),
            time_remaining_seconds=market_state.time_remaining_seconds,
        )

        if result.signal_direction == "neutral":
            return None

        # long_vol means options are underpriced → buy whichever side is dominant
        # short_vol means don't trade (reduce confidence)
        if result.signal_direction == "long_vol":
            direction = str(market_state.dominant_side.value)
            strength = min(result.vol_ratio / 2.0, 1.0)
        else:
            # short_vol — signal to avoid trading
            return None

        return _SignalVote(
            name="vol_regime",
            direction=direction,
            strength=strength,
            weight=self._w_vol,
        )

    def _vote_time_of_day(self) -> _SignalVote | None:
        """Signal 5: Time-of-day seasonality.

        Time-of-day is a meta-signal — it doesn't vote on direction,
        but modulates confidence. Returns a neutral-direction vote with
        strength = position_size_multiplier / 1.25 (normalized).
        """
        if self._time_analyzer is None:
            return None

        adj = self._time_analyzer.get_current_adjustment()

        # In optimal window, vote with the dominant direction (agreed by others)
        # Since we don't know direction yet, we return a positive vote
        # that will be applied as a confidence modifier
        if adj.position_size_multiplier < 0.75:
            return None  # Don't trade in very low-edge hours

        strength = min(adj.position_size_multiplier / 1.25, 1.0)
        # Time-of-day is a meta-signal (confidence modifier), not directional.
        # It votes "neutral" and is counted as a boost for whichever side
        # has majority. We mark direction="neutral" so it doesn't distort
        # the directional vote count.
        direction = "neutral"

        return _SignalVote(
            name="time_of_day",
            direction=direction,
            strength=strength,
            weight=self._w_time,
        )

    def _vote_vpin(self, context: dict[str, Any]) -> _SignalVote | None:
        """Signal 6: VPIN-based informed flow direction."""
        vpin = context.get("vpin", 0.0)
        vpin_direction = context.get("vpin_direction", "neutral")
        vpin_strength = context.get("vpin_strength", 0.0)

        if vpin < 0.4 or vpin_direction == "neutral":
            return None

        return _SignalVote(
            name="vpin",
            direction=vpin_direction,
            strength=vpin_strength,
            weight=self._w_vpin,
        )

    def _vote_oi_delta(self, context: dict[str, Any]) -> _SignalVote | None:
        """Signal 7: Open Interest delta confirmation."""
        oi_delta = context.get("oi_delta", 0.0)
        oi_direction = context.get("oi_direction", "neutral")

        if oi_direction == "neutral" or abs(oi_delta) < 0.001:
            return None

        strength = min(abs(oi_delta) / 0.01, 1.0)
        return _SignalVote(
            name="oi_delta",
            direction=oi_direction,
            strength=strength,
            weight=self._w_oi_delta,
        )

    def _vote_window_memory(self, context: dict[str, Any]) -> _SignalVote | None:
        """Signal 8: Multi-window serial correlation."""
        mem_direction = context.get("window_memory_direction", "neutral")
        mem_strength = context.get("window_memory_strength", 0.0)

        if mem_direction == "neutral" or mem_strength < 0.2:
            return None

        return _SignalVote(
            name="window_memory",
            direction=mem_direction,
            strength=mem_strength,
            weight=self._w_window_memory,
        )

    def _apply_regime_weights(self, regime: str) -> None:
        """Apply regime-adaptive signal weights.

        EXPANSION: boost momentum + VPIN (trending signals)
        CONTRACTION: boost OFI + window_memory (mean-reversion signals)
        NEUTRAL: use default config weights
        """
        if regime == "expansion":
            self._w_momentum = 0.30
            self._w_ofi = 0.10
            self._w_futures = 0.10
            self._w_vol = 0.05
            self._w_time = 0.05
            self._w_vpin = 0.25
            self._w_oi_delta = 0.10
            self._w_window_memory = 0.05
        elif regime == "contraction":
            self._w_momentum = 0.15
            self._w_ofi = 0.25
            self._w_futures = 0.10
            self._w_vol = 0.05
            self._w_time = 0.05
            self._w_vpin = 0.15
            self._w_oi_delta = 0.10
            self._w_window_memory = 0.15
        else:
            # Reset to default config weights
            for k, v in self._default_weights.items():
                setattr(self, f"_w_{k}", v)

    # ------------------------------------------------------------------
    # Exit logic
    # ------------------------------------------------------------------

    def _check_exit(
        self,
        position: Position,
        market_state: MarketState,
        orderbook: OrderBookSnapshot,
        context: dict[str, Any],
        minute_in_window: int,
    ) -> Signal | None:
        """Check if a position should be exited.

        Exits:
        1. Resolution guard at minute 12 (tighter than momentum-only)
        2. Signal reversal: 2+ signals flip direction
        """

        # Resolution guard — always active
        if minute_in_window >= self._resolution_guard_minute:
            logger.info(
                "singularity_exit_resolution_guard",
                market_id=position.market_id,
                minute=minute_in_window,
            )
            return Signal(
                strategy_id=self.strategy_id,
                market_id=position.market_id,
                signal_type=SignalType.EXIT,
                direction=position.side,
                exit_reason=ExitReason.RESOLUTION_GUARD,
                metadata={"minute": str(minute_in_window)},
            )

        # Signal reversal check
        entry_votes = self._entry_votes.get(position.market_id)
        if entry_votes is None:
            return None

        position_dir = position.side.value  # "YES" or "NO"
        opposite_dir = "NO" if position_dir == "YES" else "YES"

        # Re-poll available signals to check for reversals
        reversed_count = 0

        # Check momentum reversal
        candles_1m = context.get("candles_1m", [])
        window_open = float(context.get("window_open_price", 0))
        if candles_1m and window_open > 0:
            cum_return = (float(candles_1m[-1].close) - window_open) / window_open
            if cum_return != 0:
                momentum_dir = "YES" if cum_return > 0 else "NO"
                if momentum_dir == opposite_dir:
                    reversed_count += 1

        # Check OFI reversal
        if self._ofi_analyzer is not None:
            ofi_result = self._ofi_analyzer.analyze(
                orderbook, context.get("orderbook_history"),
            )
            if ofi_result.direction != "neutral":
                ofi_dir = "YES" if ofi_result.direction == "buy_pressure" else "NO"
                if ofi_dir == opposite_dir:
                    reversed_count += 1

        # Check futures lead-lag reversal
        if self._futures_detector is not None:
            futures_price = context.get("futures_price")
            spot_price = context.get("spot_price")
            if futures_price is not None and spot_price is not None:
                futures_result = self._futures_detector.detect(
                    futures_price=Decimal(str(futures_price)),
                    spot_price=Decimal(str(spot_price)),
                    polymarket_yes_price=Decimal(str(context.get("yes_price", 0.5))),
                    futures_velocity_pct_per_min=float(
                        context.get("futures_velocity_pct_per_min", 0.0)
                    ),
                )
                if futures_result.direction not in ("neutral", "") and futures_result.signal_strength > 0:
                    futures_dir = "YES" if futures_result.direction == "long" else "NO"
                    if futures_dir == opposite_dir:
                        reversed_count += 1

        # Check volatility regime reversal (short_vol = signal to exit)
        if self._vol_detector is not None:
            recent_ticks = context.get("recent_ticks")
            if recent_ticks and len(recent_ticks) >= 2:
                vol_result = self._vol_detector.detect_regime(
                    recent_ticks=recent_ticks,
                    yes_price=Decimal(str(context.get("yes_price", 0.5))),
                    time_remaining_seconds=market_state.time_remaining_seconds,
                )
                if vol_result.signal_direction == "short_vol":
                    reversed_count += 1

        if reversed_count >= self._exit_reversal_count:
            logger.info(
                "singularity_exit_signal_reversal",
                market_id=position.market_id,
                reversed_count=reversed_count,
            )
            return Signal(
                strategy_id=self.strategy_id,
                market_id=position.market_id,
                signal_type=SignalType.EXIT,
                direction=position.side,
                exit_reason=ExitReason.TRAILING_STOP,
                metadata={
                    "reason": "signal_reversal",
                    "reversed_count": str(reversed_count),
                },
            )

        return None

    # ------------------------------------------------------------------
    # Position sizing helpers
    # ------------------------------------------------------------------

    def get_position_size_multiplier(
        self, agreeing_count: int, total_sources: int
    ) -> float:
        """Compute ensemble position size multiplier.

        3 signals agree: 1.0x (base)
        4-5 signals agree: 1.25x
        6-7 signals agree: 1.5x
        8 signals agree: 1.75x (max)
        """
        if agreeing_count <= 3:
            return 1.0
        if agreeing_count <= 5:
            return 1.25
        if agreeing_count <= 7:
            return 1.5
        return 1.75

    # ------------------------------------------------------------------
    # Lifecycle callbacks
    # ------------------------------------------------------------------

    def on_fill(self, fill: Fill, position: Position) -> None:
        """Log a fill event."""
        logger.info(
            "singularity_fill",
            strategy=self.strategy_id,
            order_id=str(fill.order_id),
            price=float(fill.price),
            size=float(fill.size),
            market_id=position.market_id,
        )

    def on_cancel(self, order_id: str, reason: str) -> None:
        """Log an order cancellation."""
        logger.info(
            "singularity_cancel",
            strategy=self.strategy_id,
            order_id=order_id,
            reason=reason,
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
