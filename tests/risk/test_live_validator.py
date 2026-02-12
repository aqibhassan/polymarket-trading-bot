"""Tests for LiveValidator — live performance auto-halt."""

from __future__ import annotations

from src.risk.live_validator import LiveValidator


class TestLiveValidator:
    """Test live trading validator."""

    def test_no_halt_before_min_trades(self) -> None:
        v = LiveValidator(min_trades=20)
        for i in range(19):
            v.record_outcome(f"t{i}", 0.6, won=True)
        result = v.should_halt()
        assert not result.should_halt
        assert result.trade_count == 19

    def test_halt_on_low_win_rate(self) -> None:
        v = LiveValidator(min_trades=20, halt_win_rate=0.45, check_interval=1)
        # 8 wins, 12 losses = 40% WR
        for i in range(8):
            v.record_outcome(f"w{i}", 0.6, won=True)
        for i in range(12):
            v.record_outcome(f"l{i}", 0.6, won=False)
        result = v.should_halt()
        assert result.should_halt
        assert result.reason is not None
        assert "Win rate" in result.reason
        assert v.is_halted

    def test_no_halt_on_acceptable_win_rate(self) -> None:
        v = LiveValidator(min_trades=20, halt_win_rate=0.45, check_interval=1)
        # 12 wins, 8 losses = 60% WR
        for i in range(12):
            v.record_outcome(f"w{i}", 0.6, won=True)
        for i in range(8):
            v.record_outcome(f"l{i}", 0.6, won=False)
        result = v.should_halt()
        assert not result.should_halt

    def test_halt_on_high_brier_score(self) -> None:
        v = LiveValidator(
            min_trades=20, halt_brier_score=0.30, halt_win_rate=0.0,
            check_interval=1,
        )
        # Predict 0.90 but lose — high Brier error
        for i in range(20):
            v.record_outcome(f"t{i}", 0.90, won=False)
        result = v.should_halt()
        assert result.should_halt
        assert result.reason is not None
        assert "Brier" in result.reason

    def test_halt_persists(self) -> None:
        v = LiveValidator(min_trades=5, halt_win_rate=0.45, check_interval=1)
        for i in range(5):
            v.record_outcome(f"l{i}", 0.6, won=False)
        v.should_halt()
        assert v.is_halted
        # Add more wins — halt still persists
        for i in range(10):
            v.record_outcome(f"w{i}", 0.6, won=True)
        result = v.should_halt()
        assert result.should_halt

    def test_reset_clears_halt(self) -> None:
        v = LiveValidator(min_trades=5, halt_win_rate=0.45, check_interval=1)
        for i in range(5):
            v.record_outcome(f"l{i}", 0.6, won=False)
        v.should_halt()
        assert v.is_halted
        v.reset()
        assert not v.is_halted

    def test_check_interval_respected(self) -> None:
        v = LiveValidator(min_trades=20, check_interval=5)
        # 20 losses
        for i in range(20):
            v.record_outcome(f"l{i}", 0.6, won=False)
        result = v.should_halt()
        assert result.should_halt  # 20 % 5 == 0 → check fires

    def test_check_interval_skips(self) -> None:
        v = LiveValidator(min_trades=20, check_interval=5, halt_win_rate=0.45)
        # 21 losses — 21 % 5 != 0
        for i in range(21):
            v.record_outcome(f"l{i}", 0.6, won=False)
        result = v.should_halt()
        assert not result.should_halt  # Skipped due to interval

    def test_get_stats(self) -> None:
        v = LiveValidator(min_trades=5)
        v.record_outcome("t1", 0.7, won=True, pnl=5.0)
        v.record_outcome("t2", 0.6, won=False, pnl=-3.0)
        stats = v.get_stats()
        assert stats["trade_count"] == 2
        assert stats["win_rate"] == 0.5
        assert stats["total_pnl"] == 2.0
        assert not stats["halted"]

    def test_empty_stats(self) -> None:
        v = LiveValidator()
        stats = v.get_stats()
        assert stats["trade_count"] == 0
