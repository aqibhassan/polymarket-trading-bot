"""Tests for ExitManager."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from src.engine.exit_manager import ExitManager
from src.models.market import MarketState, Position, Side
from src.models.signal import ExitReason


def _make_position(
    entry_price: str = "0.40",
    stop_loss: str = "0.36",
    take_profit: str = "0.50",
    entry_time: datetime | None = None,
) -> Position:
    return Position(
        market_id="test-market-123",
        side=Side.YES,
        token_id="token-yes-123",
        entry_price=Decimal(entry_price),
        quantity=Decimal("100"),
        entry_time=entry_time or datetime.now(UTC),
        stop_loss=Decimal(stop_loss),
        take_profit=Decimal(take_profit),
    )


def _make_market_state(time_remaining_seconds: int = 600) -> MarketState:
    return MarketState(
        market_id="test-market-123",
        yes_price=Decimal("0.62"),
        no_price=Decimal("0.38"),
        time_remaining_seconds=time_remaining_seconds,
    )


class TestExitManager:
    """Tests for ExitManager.should_exit()."""

    def test_profit_target_exit(self, config_loader) -> None:
        em = ExitManager(config_loader)
        position = _make_position()
        # profit_target_pct = 0.05, entry = 0.40, target = 0.45
        should, reason = em.should_exit(
            position, Decimal("0.46"), _make_market_state()
        )
        assert should is True
        assert reason == ExitReason.PROFIT_TARGET

    def test_trailing_stop_exit(self, config_loader) -> None:
        em = ExitManager(config_loader)
        # Use higher entry so peak of 0.44 doesn't trigger profit target
        # entry=0.40, profit target=0.40+0.05=0.45
        # Push peak to 0.44 (below profit target), then drop to 0.41 (below 0.44-0.03=0.41)
        position = _make_position()
        market = _make_market_state()
        # Set peak to 0.44 (below profit target of 0.45)
        em.should_exit(position, Decimal("0.44"), market)
        # Drop to 0.40 which is below 0.44 - 0.03 = 0.41
        should, reason = em.should_exit(position, Decimal("0.40"), market)
        assert should is True
        assert reason == ExitReason.TRAILING_STOP

    def test_hard_stop_loss_exit(self, config_loader) -> None:
        em = ExitManager(config_loader)
        position = _make_position(stop_loss="0.36")
        should, reason = em.should_exit(
            position, Decimal("0.35"), _make_market_state()
        )
        assert should is True
        assert reason == ExitReason.HARD_STOP_LOSS

    def test_max_time_exit(self, config_loader) -> None:
        em = ExitManager(config_loader)
        # max_hold_seconds = 420 (7 min)
        old_time = datetime.now(UTC) - timedelta(seconds=500)
        position = _make_position(entry_time=old_time)
        should, reason = em.should_exit(
            position, Decimal("0.41"), _make_market_state()
        )
        assert should is True
        assert reason == ExitReason.MAX_TIME

    def test_resolution_guard_exit(self, config_loader) -> None:
        em = ExitManager(config_loader)
        position = _make_position()
        # force_exit_minute = 11, so threshold = 15 - 11 = 4 minutes remaining
        # time_remaining_seconds = 200 => 3.33 min remaining < 4 min
        market = _make_market_state(time_remaining_seconds=200)
        should, reason = em.should_exit(
            position, Decimal("0.41"), market
        )
        assert should is True
        assert reason == ExitReason.RESOLUTION_GUARD

    def test_kill_switch_exit(self, config_loader) -> None:
        em = ExitManager(config_loader)
        position = _make_position()
        should, reason = em.should_exit(
            position, Decimal("0.41"), _make_market_state(), kill_active=True
        )
        assert should is True
        assert reason == ExitReason.KILL_SWITCH

    def test_no_exit_when_normal(self, config_loader) -> None:
        em = ExitManager(config_loader)
        # Entry just now, price between stop and target, plenty of time
        position = _make_position()
        should, reason = em.should_exit(
            position, Decimal("0.41"), _make_market_state()
        )
        assert should is False
        assert reason is None

    def test_kill_switch_highest_priority(self, config_loader) -> None:
        """Kill switch should trigger even when other exits would too."""
        em = ExitManager(config_loader)
        position = _make_position()
        # Price below stop loss AND kill active
        should, reason = em.should_exit(
            position, Decimal("0.30"), _make_market_state(), kill_active=True
        )
        assert should is True
        assert reason == ExitReason.KILL_SWITCH

    def test_hard_stop_at_exact_stop_loss(self, config_loader) -> None:
        em = ExitManager(config_loader)
        position = _make_position(stop_loss="0.36")
        should, reason = em.should_exit(
            position, Decimal("0.36"), _make_market_state()
        )
        assert should is True
        assert reason == ExitReason.HARD_STOP_LOSS

    def test_trailing_stop_not_triggered_without_gain(self, config_loader) -> None:
        """Trailing stop should not fire if price never rose above entry."""
        em = ExitManager(config_loader)
        position = _make_position()
        # Price is at entry, then drops a little
        em.should_exit(position, Decimal("0.40"), _make_market_state())
        should, reason = em.should_exit(
            position, Decimal("0.38"), _make_market_state()
        )
        # Hard stop at 0.36 so 0.38 won't trigger it, and trailing
        # should not fire because peak == entry
        assert should is False

    def test_reset_peak(self, config_loader) -> None:
        em = ExitManager(config_loader)
        position = _make_position()
        # Push peak to 0.44 (below profit target)
        em.should_exit(position, Decimal("0.44"), _make_market_state())
        em.reset_peak("test-market-123")
        # After reset, peak starts fresh from entry
        should, reason = em.should_exit(
            position, Decimal("0.41"), _make_market_state()
        )
        assert should is False
