"""Tests for TimeGuard — candle time entry/exit guard."""

from __future__ import annotations

from src.risk.time_guard import TimeGuard


class TestCanEnter:
    def test_early_candle_allowed(self) -> None:
        tg = TimeGuard(no_entry_after_minute=8.0)
        assert tg.can_enter(0.0) is True
        assert tg.can_enter(5.0) is True

    def test_at_cutoff_allowed(self) -> None:
        tg = TimeGuard(no_entry_after_minute=8.0)
        assert tg.can_enter(8.0) is True

    def test_after_cutoff_blocked(self) -> None:
        tg = TimeGuard(no_entry_after_minute=8.0)
        assert tg.can_enter(8.01) is False
        assert tg.can_enter(10.0) is False

    def test_custom_cutoff(self) -> None:
        tg = TimeGuard(no_entry_after_minute=5.0)
        assert tg.can_enter(4.9) is True
        assert tg.can_enter(5.1) is False


class TestMustExit:
    def test_early_candle_no_exit(self) -> None:
        tg = TimeGuard(force_exit_minute=11.0)
        assert tg.must_exit(0.0) is False
        assert tg.must_exit(10.9) is False

    def test_at_force_exit_must_exit(self) -> None:
        tg = TimeGuard(force_exit_minute=11.0)
        assert tg.must_exit(11.0) is True

    def test_past_force_exit(self) -> None:
        tg = TimeGuard(force_exit_minute=11.0)
        assert tg.must_exit(12.0) is True
        assert tg.must_exit(14.5) is True

    def test_custom_force_exit(self) -> None:
        tg = TimeGuard(force_exit_minute=9.0)
        assert tg.must_exit(8.9) is False
        assert tg.must_exit(9.0) is True


class TestTimeRemainingWarning:
    def test_no_warning_early(self) -> None:
        tg = TimeGuard(force_exit_minute=11.0, warning_threshold_minutes=3.0)
        assert tg.time_remaining_warning(0.0) is None
        assert tg.time_remaining_warning(7.0) is None

    def test_warning_at_threshold(self) -> None:
        tg = TimeGuard(force_exit_minute=11.0, warning_threshold_minutes=3.0)
        # At 8.0 minutes, 3.0 remaining — not yet warning (not strictly less)
        assert tg.time_remaining_warning(8.0) is None
        # At 8.1 minutes, 2.9 remaining — warning
        result = tg.time_remaining_warning(8.1)
        assert result is not None
        assert "2.9" in result

    def test_warning_one_minute_left(self) -> None:
        tg = TimeGuard(force_exit_minute=11.0, warning_threshold_minutes=3.0)
        result = tg.time_remaining_warning(10.0)
        assert result is not None
        assert "1.0" in result

    def test_warning_past_force_exit(self) -> None:
        tg = TimeGuard(force_exit_minute=11.0)
        result = tg.time_remaining_warning(11.5)
        assert result is not None
        assert "FORCE EXIT NOW" in result

    def test_warning_exactly_at_force_exit(self) -> None:
        tg = TimeGuard(force_exit_minute=11.0)
        result = tg.time_remaining_warning(11.0)
        assert result is not None
        assert "FORCE EXIT NOW" in result

    def test_custom_warning_threshold(self) -> None:
        tg = TimeGuard(
            force_exit_minute=11.0,
            warning_threshold_minutes=5.0,
        )
        # 6.5 minutes elapsed -> 4.5 remaining < 5.0 threshold
        result = tg.time_remaining_warning(6.5)
        assert result is not None
        assert "4.5" in result
