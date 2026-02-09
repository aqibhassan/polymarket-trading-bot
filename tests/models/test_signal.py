"""Tests for signal models."""

from __future__ import annotations

from decimal import Decimal

from src.models.market import Side
from src.models.signal import (
    Confidence,
    ExitReason,
    Signal,
    SignalType,
    TrendDirection,
    TrendResult,
)


class TestConfidence:
    def test_meets_minimum(self) -> None:
        conf = Confidence(overall=0.7)
        assert conf.meets_minimum(0.6) is True
        assert conf.meets_minimum(0.8) is False

    def test_exact_threshold(self) -> None:
        conf = Confidence(overall=0.6)
        assert conf.meets_minimum(0.6) is True

    def test_default_values(self) -> None:
        conf = Confidence()
        assert conf.overall == 0.0
        assert conf.meets_minimum(0.0) is True


class TestSignal:
    def test_entry_signal(self) -> None:
        signal = Signal(
            strategy_id="false_sentiment",
            market_id="test-market",
            signal_type=SignalType.ENTRY,
            direction=Side.NO,
            strength=Decimal("0.8"),
            confidence=Confidence(overall=0.75),
            entry_price=Decimal("0.38"),
            stop_loss=Decimal("0.34"),
            take_profit=Decimal("0.50"),
        )
        assert signal.signal_type == SignalType.ENTRY
        assert signal.direction == Side.NO
        assert signal.confidence.meets_minimum(0.6)

    def test_exit_signal(self) -> None:
        signal = Signal(
            strategy_id="false_sentiment",
            market_id="test-market",
            signal_type=SignalType.EXIT,
            direction=Side.YES,
            exit_reason=ExitReason.PROFIT_TARGET,
        )
        assert signal.signal_type == SignalType.EXIT
        assert signal.exit_reason == ExitReason.PROFIT_TARGET

    def test_skip_signal(self) -> None:
        signal = Signal(
            strategy_id="false_sentiment",
            market_id="test-market",
            signal_type=SignalType.SKIP,
            direction=Side.YES,
            metadata={"reason": "heavy_book"},
        )
        assert signal.signal_type == SignalType.SKIP
        assert signal.metadata["reason"] == "heavy_book"


class TestTrendResult:
    def test_uptrend(self) -> None:
        trend = TrendResult(
            direction=TrendDirection.UP,
            strength=0.8,
            momentum=1.5,
            green_count=4,
            red_count=1,
            cumulative_move_pct=0.5,
        )
        assert trend.direction == TrendDirection.UP
        assert trend.strength == 0.8

    def test_neutral(self) -> None:
        trend = TrendResult(
            direction=TrendDirection.NEUTRAL,
            strength=0.2,
            green_count=2,
            red_count=3,
        )
        assert trend.direction == TrendDirection.NEUTRAL
