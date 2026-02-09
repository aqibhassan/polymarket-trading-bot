"""Exit manager — multi-reason position exit logic."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING

from src.core.logging import get_logger
from src.models.signal import ExitReason

if TYPE_CHECKING:
    from src.config.loader import ConfigLoader
    from src.models.market import MarketState, Position

log = get_logger(__name__)


class ExitManager:
    """Determines when to exit a position based on multiple criteria.

    Exit reasons:
    1. PROFIT_TARGET — price >= entry + profit_target_pct
    2. TRAILING_STOP — price drops below peak - trailing_stop_pct
    3. HARD_STOP_LOSS — price <= stop_loss
    4. MAX_TIME — held longer than max_hold_seconds
    5. RESOLUTION_GUARD — time_remaining < force_exit_minute threshold
    6. KILL_SWITCH — external kill signal
    """

    def __init__(self, config: ConfigLoader) -> None:
        self._profit_target_pct = Decimal(
            str(config.get("exit.profit_target_pct", 0.05))
        )
        self._trailing_stop_pct = Decimal(
            str(config.get("exit.trailing_stop_pct", 0.03))
        )
        self._max_hold_seconds = int(config.get("exit.max_hold_seconds", 420))
        self._force_exit_minute = int(
            config.get("strategy.false_sentiment.force_exit_minute", 11)
        )
        self._peak_prices: dict[str, Decimal] = {}

    def should_exit(
        self,
        position: Position,
        current_price: Decimal,
        market_state: MarketState,
        kill_active: bool = False,
    ) -> tuple[bool, ExitReason | None]:
        """Evaluate whether a position should be exited.

        Args:
            position: The open position.
            current_price: Current market price.
            market_state: Current market state with timing info.
            kill_active: Whether the kill switch is engaged.

        Returns:
            Tuple of (should_exit, exit_reason).
        """
        # Update peak price tracking for trailing stops
        market_id = position.market_id
        peak = self._peak_prices.get(market_id, position.entry_price)
        if current_price > peak:
            peak = current_price
            self._peak_prices[market_id] = peak

        # 6. Kill switch — highest priority
        if kill_active:
            log.warning("exit_kill_switch", market_id=market_id)
            return True, ExitReason.KILL_SWITCH

        # 3. Hard stop loss
        if current_price <= position.stop_loss:
            log.info(
                "exit_hard_stop",
                market_id=market_id,
                price=float(current_price),
                stop=float(position.stop_loss),
            )
            return True, ExitReason.HARD_STOP_LOSS

        # 1. Profit target
        target = position.entry_price + self._profit_target_pct
        if current_price >= target:
            log.info(
                "exit_profit_target",
                market_id=market_id,
                price=float(current_price),
                target=float(target),
            )
            return True, ExitReason.PROFIT_TARGET

        # 2. Trailing stop
        trail_floor = peak - self._trailing_stop_pct
        if current_price <= trail_floor and peak > position.entry_price:
            log.info(
                "exit_trailing_stop",
                market_id=market_id,
                price=float(current_price),
                peak=float(peak),
                floor=float(trail_floor),
            )
            return True, ExitReason.TRAILING_STOP

        # 4. Max hold time
        now = datetime.now(timezone.utc)
        held_seconds = (now - position.entry_time).total_seconds()
        if held_seconds >= self._max_hold_seconds:
            log.info(
                "exit_max_time",
                market_id=market_id,
                held_seconds=held_seconds,
            )
            return True, ExitReason.MAX_TIME

        # 5. Resolution guard — force exit before candle resolves
        minutes_remaining = market_state.time_remaining_seconds / 60.0
        force_exit_threshold = 15.0 - self._force_exit_minute
        if minutes_remaining <= force_exit_threshold:
            log.info(
                "exit_resolution_guard",
                market_id=market_id,
                minutes_remaining=minutes_remaining,
            )
            return True, ExitReason.RESOLUTION_GUARD

        return False, None

    def reset_peak(self, market_id: str) -> None:
        """Reset peak price tracking for a market.

        Args:
            market_id: Market identifier to reset.
        """
        self._peak_prices.pop(market_id, None)
