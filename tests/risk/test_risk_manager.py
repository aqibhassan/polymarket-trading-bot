"""Tests for RiskManager — pre-trade risk gate."""

from __future__ import annotations

from decimal import Decimal

from src.models.market import Side
from src.models.signal import Confidence, Signal, SignalType
from src.risk.kill_switch import KillSwitch
from src.risk.risk_manager import RiskManager


def _make_signal(
    stop_loss: Decimal | None = Decimal("0.45"),
    take_profit: Decimal | None = Decimal("0.65"),
    entry_price: Decimal = Decimal("0.55"),
    market_id: str = "test-market",
) -> Signal:
    return Signal(
        strategy_id="test",
        market_id=market_id,
        signal_type=SignalType.ENTRY,
        direction=Side.YES,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        confidence=Confidence(overall=0.7),
    )


class TestHasStopLoss:
    def test_has_stop_loss_true(self) -> None:
        signal = _make_signal(stop_loss=Decimal("0.45"))
        rm = RiskManager()
        assert rm.has_stop_loss(signal) is True

    def test_has_stop_loss_false(self) -> None:
        signal = _make_signal(stop_loss=None)
        rm = RiskManager()
        assert rm.has_stop_loss(signal) is False


class TestCheckOrder:
    def test_approval_for_valid_signal(self) -> None:
        rm = RiskManager()
        signal = _make_signal()
        decision = rm.check_order(
            signal=signal,
            position_size=Decimal("100"),
            current_drawdown=Decimal("0.01"),
            balance=Decimal("10000"),
            book_depth=Decimal("5000"),
        )
        assert decision.approved is True
        assert decision.reason == "all checks passed"

    def test_rejection_missing_stop_loss(self) -> None:
        rm = RiskManager()
        signal = _make_signal(stop_loss=None)
        decision = rm.check_order(
            signal=signal,
            position_size=Decimal("100"),
            current_drawdown=Decimal("0.01"),
            balance=Decimal("10000"),
        )
        assert decision.approved is False
        assert "stop_loss" in decision.reason

    def test_rejection_oversized_position(self) -> None:
        rm = RiskManager(max_position_pct=Decimal("0.02"))
        signal = _make_signal()
        decision = rm.check_order(
            signal=signal,
            position_size=Decimal("300"),
            current_drawdown=Decimal("0.01"),
            balance=Decimal("10000"),
        )
        assert decision.approved is False
        assert "exceeds max" in decision.reason
        assert decision.max_size == Decimal("200")

    def test_allowed_at_exact_drawdown_limit(self) -> None:
        """Trading should be allowed AT the limit (5%), only halted above it."""
        rm = RiskManager(max_daily_drawdown_pct=Decimal("0.05"))
        signal = _make_signal()
        decision = rm.check_order(
            signal=signal,
            position_size=Decimal("100"),
            current_drawdown=Decimal("0.05"),
            balance=Decimal("10000"),
        )
        assert decision.approved is True

    def test_rejection_drawdown_above_limit(self) -> None:
        rm = RiskManager(max_daily_drawdown_pct=Decimal("0.05"))
        signal = _make_signal()
        decision = rm.check_order(
            signal=signal,
            position_size=Decimal("100"),
            current_drawdown=Decimal("0.06"),
            balance=Decimal("10000"),
        )
        assert decision.approved is False

    def test_rejection_kill_switch_active(self) -> None:
        ks = KillSwitch()
        ks.trigger("test trigger")
        rm = RiskManager(kill_switch=ks)
        signal = _make_signal()
        decision = rm.check_order(
            signal=signal,
            position_size=Decimal("10"),
            current_drawdown=Decimal("0"),
            balance=Decimal("10000"),
        )
        assert decision.approved is False
        assert "kill switch" in decision.reason

    def test_rejection_order_book_impact(self) -> None:
        rm = RiskManager(max_order_book_pct=Decimal("0.10"))
        signal = _make_signal()
        decision = rm.check_order(
            signal=signal,
            position_size=Decimal("200"),
            current_drawdown=Decimal("0.01"),
            balance=Decimal("100000"),
            book_depth=Decimal("1000"),
        )
        assert decision.approved is False
        assert "order book impact" in decision.reason

    def test_approval_at_exact_position_limit(self) -> None:
        rm = RiskManager(max_position_pct=Decimal("0.02"))
        signal = _make_signal()
        decision = rm.check_order(
            signal=signal,
            position_size=Decimal("200"),
            current_drawdown=Decimal("0.01"),
            balance=Decimal("10000"),
        )
        assert decision.approved is True

    def test_rejection_zero_balance(self) -> None:
        """Zero balance now correctly rejects orders."""
        rm = RiskManager()
        signal = _make_signal()
        decision = rm.check_order(
            signal=signal,
            position_size=Decimal("999999"),
            current_drawdown=Decimal("0.01"),
            balance=Decimal("0"),
        )
        assert decision.approved is False
        assert "zero or negative balance" in decision.reason

    def test_approval_zero_book_depth_skips_impact_check(self) -> None:
        rm = RiskManager()
        signal = _make_signal()
        decision = rm.check_order(
            signal=signal,
            position_size=Decimal("100"),
            current_drawdown=Decimal("0.01"),
            balance=Decimal("100000"),
            book_depth=Decimal("0"),
        )
        assert decision.approved is True

    def test_drawdown_just_under_limit(self) -> None:
        rm = RiskManager(max_daily_drawdown_pct=Decimal("0.05"))
        signal = _make_signal()
        decision = rm.check_order(
            signal=signal,
            position_size=Decimal("100"),
            current_drawdown=Decimal("0.0499"),
            balance=Decimal("10000"),
        )
        assert decision.approved is True

    def test_order_book_impact_at_exact_limit(self) -> None:
        rm = RiskManager(max_order_book_pct=Decimal("0.10"))
        signal = _make_signal()
        decision = rm.check_order(
            signal=signal,
            position_size=Decimal("100"),
            current_drawdown=Decimal("0.01"),
            balance=Decimal("100000"),
            book_depth=Decimal("1000"),
        )
        assert decision.approved is True

    def test_kill_switch_priority_over_other_checks(self) -> None:
        """Kill switch should reject even if signal is valid."""
        ks = KillSwitch()
        ks.trigger("priority test")
        rm = RiskManager(kill_switch=ks)
        signal = _make_signal()
        decision = rm.check_order(
            signal=signal,
            position_size=Decimal("10"),
            current_drawdown=Decimal("0"),
            balance=Decimal("100000"),
        )
        assert decision.approved is False
        assert "kill switch" in decision.reason


class TestBalanceFloor:
    def test_rejects_below_floor(self) -> None:
        rm = RiskManager(
            balance_floor_pct=Decimal("0.50"),
            initial_balance=Decimal("10000"),
        )
        signal = _make_signal()
        decision = rm.check_order(
            signal=signal,
            position_size=Decimal("10"),
            current_drawdown=Decimal("0"),
            balance=Decimal("4000"),  # 40% of 10k — below 50% floor
        )
        assert decision.approved is False
        assert "below floor" in decision.reason

    def test_allows_above_floor(self) -> None:
        rm = RiskManager(
            balance_floor_pct=Decimal("0.50"),
            initial_balance=Decimal("10000"),
        )
        signal = _make_signal()
        decision = rm.check_order(
            signal=signal,
            position_size=Decimal("10"),
            current_drawdown=Decimal("0"),
            balance=Decimal("6000"),  # 60% of 10k — above 50% floor
        )
        assert decision.approved is True

    def test_allows_at_exact_floor(self) -> None:
        rm = RiskManager(
            balance_floor_pct=Decimal("0.50"),
            initial_balance=Decimal("10000"),
        )
        signal = _make_signal()
        decision = rm.check_order(
            signal=signal,
            position_size=Decimal("10"),
            current_drawdown=Decimal("0"),
            balance=Decimal("5000"),  # Exactly at floor
        )
        assert decision.approved is True

    def test_disabled_when_zero(self) -> None:
        """Balance floor is disabled when balance_floor_pct is 0."""
        rm = RiskManager(
            balance_floor_pct=Decimal("0"),
            initial_balance=Decimal("10000"),
        )
        signal = _make_signal()
        decision = rm.check_order(
            signal=signal,
            position_size=Decimal("10"),
            current_drawdown=Decimal("0"),
            balance=Decimal("10000"),
        )
        assert decision.approved is True
