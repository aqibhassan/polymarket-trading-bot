"""Tests for PositionSizer — Half-Kelly with safety caps."""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from src.risk.position_sizer import PositionSizer


class TestPositionSizer:
    def test_basic_half_kelly(self) -> None:
        sizer = PositionSizer(
            max_position_pct=Decimal("1"),
            max_order_book_pct=Decimal("1"),
        )
        result = sizer.calculate(
            balance=Decimal("10000"),
            signal_confidence=1.0,
            win_rate=Decimal("0.6"),
            avg_win=Decimal("100"),
            avg_loss=Decimal("100"),
            book_depth=Decimal("1000000"),
        )
        # kelly = 0.6 - (0.4 / 1.0) = 0.2
        # half_kelly = 0.1
        # raw_size = 0.1 * 10000 = 1000
        assert result.kelly_fraction == Decimal("0.2")
        assert result.recommended_size == Decimal("1000.0")

    def test_capping_at_max_position_pct(self) -> None:
        sizer = PositionSizer(max_position_pct=Decimal("0.02"))
        result = sizer.calculate(
            balance=Decimal("10000"),
            signal_confidence=1.0,
            win_rate=Decimal("0.7"),
            avg_win=Decimal("100"),
            avg_loss=Decimal("50"),
        )
        # kelly = 0.7 - (0.3 / 2.0) = 0.55
        # half_kelly = 0.275
        # raw = 2750, max = 200
        assert result.recommended_size == Decimal("200")
        assert "max_position_pct" in (result.capped_reason or "")

    def test_capping_at_max_order_book_pct(self) -> None:
        sizer = PositionSizer(
            max_position_pct=Decimal("1"),
            max_order_book_pct=Decimal("0.10"),
        )
        result = sizer.calculate(
            balance=Decimal("100000"),
            signal_confidence=1.0,
            win_rate=Decimal("0.7"),
            avg_win=Decimal("100"),
            avg_loss=Decimal("50"),
            book_depth=Decimal("1000"),
        )
        # max_by_book = 1000 * 0.10 = 100
        assert result.recommended_size == Decimal("100")
        assert "max_order_book_pct" in (result.capped_reason or "")

    def test_zero_balance(self) -> None:
        sizer = PositionSizer()
        result = sizer.calculate(
            balance=Decimal("0"),
            signal_confidence=0.8,
            win_rate=Decimal("0.6"),
            avg_win=Decimal("100"),
            avg_loss=Decimal("100"),
        )
        assert result.recommended_size == Decimal("0")
        assert result.capped_reason is not None

    def test_zero_win_rate(self) -> None:
        sizer = PositionSizer()
        result = sizer.calculate(
            balance=Decimal("10000"),
            signal_confidence=0.8,
            win_rate=Decimal("0"),
            avg_win=Decimal("100"),
            avg_loss=Decimal("100"),
        )
        assert result.recommended_size == Decimal("0")

    def test_zero_avg_loss(self) -> None:
        sizer = PositionSizer()
        result = sizer.calculate(
            balance=Decimal("10000"),
            signal_confidence=0.8,
            win_rate=Decimal("0.6"),
            avg_win=Decimal("100"),
            avg_loss=Decimal("0"),
        )
        assert result.recommended_size == Decimal("0")

    def test_negative_kelly(self) -> None:
        sizer = PositionSizer()
        result = sizer.calculate(
            balance=Decimal("10000"),
            signal_confidence=0.8,
            win_rate=Decimal("0.3"),
            avg_win=Decimal("50"),
            avg_loss=Decimal("100"),
        )
        # kelly = 0.3 - (0.7 / 0.5) = 0.3 - 1.4 = -1.1
        assert result.recommended_size == Decimal("0")
        assert result.kelly_fraction < 0
        assert "negative kelly" in (result.capped_reason or "")

    def test_confidence_scaling(self) -> None:
        sizer = PositionSizer(
            max_position_pct=Decimal("1"),
            max_order_book_pct=Decimal("1"),
        )
        full = sizer.calculate(
            balance=Decimal("10000"),
            signal_confidence=1.0,
            win_rate=Decimal("0.6"),
            avg_win=Decimal("100"),
            avg_loss=Decimal("100"),
            book_depth=Decimal("1000000"),
        )
        half = sizer.calculate(
            balance=Decimal("10000"),
            signal_confidence=0.5,
            win_rate=Decimal("0.6"),
            avg_win=Decimal("100"),
            avg_loss=Decimal("100"),
            book_depth=Decimal("1000000"),
        )
        assert half.recommended_size == full.recommended_size * Decimal("0.5")

    def test_model_frozen(self) -> None:
        sizer = PositionSizer()
        result = sizer.calculate(
            balance=Decimal("10000"),
            signal_confidence=0.8,
            win_rate=Decimal("0.6"),
            avg_win=Decimal("100"),
            avg_loss=Decimal("100"),
        )
        with pytest.raises(ValidationError):
            result.recommended_size = Decimal("999")  # type: ignore[misc]

    def test_negative_balance(self) -> None:
        sizer = PositionSizer()
        result = sizer.calculate(
            balance=Decimal("-100"),
            signal_confidence=0.8,
            win_rate=Decimal("0.6"),
            avg_win=Decimal("100"),
            avg_loss=Decimal("100"),
        )
        assert result.recommended_size == Decimal("0")

    def test_no_book_depth_uses_balance_cap(self) -> None:
        sizer = PositionSizer(max_position_pct=Decimal("0.02"))
        result = sizer.calculate(
            balance=Decimal("10000"),
            signal_confidence=1.0,
            win_rate=Decimal("0.7"),
            avg_win=Decimal("100"),
            avg_loss=Decimal("50"),
            book_depth=Decimal("0"),
        )
        assert result.max_allowed == Decimal("200")

    def test_breakeven_edge(self) -> None:
        """Win rate = 0.5, equal win/loss -> kelly = 0 -> no bet."""
        sizer = PositionSizer()
        result = sizer.calculate(
            balance=Decimal("10000"),
            signal_confidence=1.0,
            win_rate=Decimal("0.5"),
            avg_win=Decimal("100"),
            avg_loss=Decimal("100"),
        )
        assert result.recommended_size == Decimal("0")

    def test_custom_kelly_multiplier(self) -> None:
        """Quarter-Kelly should give half the size of half-Kelly."""
        half = PositionSizer(
            max_position_pct=Decimal("1"),
            max_order_book_pct=Decimal("1"),
            kelly_multiplier=Decimal("0.5"),
        )
        quarter = PositionSizer(
            max_position_pct=Decimal("1"),
            max_order_book_pct=Decimal("1"),
            kelly_multiplier=Decimal("0.25"),
        )
        r_half = half.calculate(
            balance=Decimal("10000"),
            signal_confidence=1.0,
            win_rate=Decimal("0.6"),
            avg_win=Decimal("100"),
            avg_loss=Decimal("100"),
            book_depth=Decimal("1000000"),
        )
        r_quarter = quarter.calculate(
            balance=Decimal("10000"),
            signal_confidence=1.0,
            win_rate=Decimal("0.6"),
            avg_win=Decimal("100"),
            avg_loss=Decimal("100"),
            book_depth=Decimal("1000000"),
        )
        assert r_quarter.recommended_size == r_half.recommended_size / 2


class TestPositionSizerBinary:
    """Tests for binary prediction market Kelly sizing."""

    def test_basic_binary_kelly(self) -> None:
        """f* = (p - P) / (P * (1 - P)), then half-Kelly, then * balance."""
        sizer = PositionSizer(
            max_position_pct=Decimal("1"),
            kelly_multiplier=Decimal("0.5"),
        )
        # p=0.90, P=0.65 -> kelly = (0.90-0.65)/(0.65*0.35) = 0.25/0.2275 ≈ 1.0989
        result = sizer.calculate_binary(
            balance=Decimal("10000"),
            entry_price=Decimal("0.65"),
            estimated_win_prob=Decimal("0.90"),
        )
        # kelly ≈ 1.0989 (asymmetric binary payoff formula)
        # half-kelly ≈ 0.5495
        # raw = 10000 * 0.5495 ≈ 5495
        assert result.kelly_fraction > Decimal("1.09")
        assert result.kelly_fraction < Decimal("1.10")
        assert result.recommended_size > Decimal("5400")
        assert result.recommended_size < Decimal("5600")

    def test_binary_no_edge(self) -> None:
        """Win prob <= entry price -> no bet."""
        sizer = PositionSizer()
        result = sizer.calculate_binary(
            balance=Decimal("10000"),
            entry_price=Decimal("0.90"),
            estimated_win_prob=Decimal("0.85"),
        )
        assert result.recommended_size == Decimal("0")
        assert "no edge" in (result.capped_reason or "")

    def test_binary_capped_at_max_pct(self) -> None:
        """Large edge should still be capped at max_position_pct."""
        sizer = PositionSizer(max_position_pct=Decimal("0.02"))
        result = sizer.calculate_binary(
            balance=Decimal("10000"),
            entry_price=Decimal("0.55"),
            estimated_win_prob=Decimal("0.90"),
        )
        # kelly = (0.90-0.55)/(0.55*0.45) ≈ 1.4141
        # half ≈ 0.7071, raw ≈ 7071, cap = 200
        assert result.recommended_size == Decimal("200")
        assert "max_position_pct" in (result.capped_reason or "")

    def test_binary_invalid_inputs(self) -> None:
        sizer = PositionSizer()
        # Zero balance
        r = sizer.calculate_binary(Decimal("0"), Decimal("0.50"), Decimal("0.60"))
        assert r.recommended_size == Decimal("0")
        # Price out of range
        r = sizer.calculate_binary(Decimal("10000"), Decimal("0"), Decimal("0.60"))
        assert r.recommended_size == Decimal("0")
        r = sizer.calculate_binary(Decimal("10000"), Decimal("1"), Decimal("0.60"))
        assert r.recommended_size == Decimal("0")
        # Win prob out of range
        r = sizer.calculate_binary(Decimal("10000"), Decimal("0.50"), Decimal("0"))
        assert r.recommended_size == Decimal("0")

    def test_binary_kelly_at_even_odds(self) -> None:
        """At P=0.50 with p=0.90, f* = (0.90-0.50)/(0.50*0.50) = 1.60."""
        sizer = PositionSizer(
            max_position_pct=Decimal("1"),
            kelly_multiplier=Decimal("1"),  # full Kelly
        )
        result = sizer.calculate_binary(
            balance=Decimal("10000"),
            entry_price=Decimal("0.50"),
            estimated_win_prob=Decimal("0.90"),
        )
        # f* = (0.90-0.50)/(0.50*0.50) = 0.40/0.25 = 1.60
        assert result.kelly_fraction == Decimal("1.60")
        # full kelly * balance = 1.60 * 10000 = 16000 (capped at max_position_pct=100%)
        assert result.recommended_size == Decimal("10000")
        assert "max_position_pct" in (result.capped_reason or "")

    def test_binary_quarter_kelly_reduces_size(self) -> None:
        """Quarter-Kelly should size down appropriately."""
        half = PositionSizer(
            max_position_pct=Decimal("1"),
            kelly_multiplier=Decimal("0.5"),
        )
        quarter = PositionSizer(
            max_position_pct=Decimal("1"),
            kelly_multiplier=Decimal("0.25"),
        )
        r_half = half.calculate_binary(
            balance=Decimal("10000"),
            entry_price=Decimal("0.65"),
            estimated_win_prob=Decimal("0.90"),
        )
        r_quarter = quarter.calculate_binary(
            balance=Decimal("10000"),
            entry_price=Decimal("0.65"),
            estimated_win_prob=Decimal("0.90"),
        )
        assert r_quarter.recommended_size == r_half.recommended_size / 2

    def test_binary_book_depth_cap(self) -> None:
        sizer = PositionSizer(
            max_position_pct=Decimal("1"),
            max_order_book_pct=Decimal("0.10"),
        )
        result = sizer.calculate_binary(
            balance=Decimal("100000"),
            entry_price=Decimal("0.60"),
            estimated_win_prob=Decimal("0.90"),
            book_depth=Decimal("500"),
        )
        # max_by_book = 500 * 0.10 = 50
        assert result.recommended_size == Decimal("50")
        assert "max_order_book_pct" in (result.capped_reason or "")
