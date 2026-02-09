"""Tests for PositionSizer — Kelly with min/max caps."""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from src.risk.position_sizer import PositionSizer


class TestPositionSizer:
    def test_basic_quarter_kelly(self) -> None:
        """Default quarter-Kelly with no caps binding."""
        sizer = PositionSizer(
            max_position_pct=Decimal("1"),
            min_position_pct=Decimal("0"),
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
        # quarter_kelly = 0.05
        # raw_size = 0.05 * 10000 = 500
        assert result.kelly_fraction == Decimal("0.2")
        assert result.recommended_size == Decimal("500.0")

    def test_half_kelly_explicit(self) -> None:
        """Explicit half-Kelly multiplier."""
        sizer = PositionSizer(
            max_position_pct=Decimal("1"),
            min_position_pct=Decimal("0"),
            max_order_book_pct=Decimal("1"),
            kelly_multiplier=Decimal("0.5"),
        )
        result = sizer.calculate(
            balance=Decimal("10000"),
            signal_confidence=1.0,
            win_rate=Decimal("0.6"),
            avg_win=Decimal("100"),
            avg_loss=Decimal("100"),
            book_depth=Decimal("1000000"),
        )
        # kelly = 0.2, half = 0.1, raw = 1000
        assert result.kelly_fraction == Decimal("0.2")
        assert result.recommended_size == Decimal("1000.0")

    def test_capping_at_max_position_pct(self) -> None:
        sizer = PositionSizer(max_position_pct=Decimal("0.02"), min_position_pct=Decimal("0"))
        result = sizer.calculate(
            balance=Decimal("10000"),
            signal_confidence=1.0,
            win_rate=Decimal("0.7"),
            avg_win=Decimal("100"),
            avg_loss=Decimal("50"),
        )
        # kelly = 0.7 - (0.3 / 2.0) = 0.55
        # quarter_kelly = 0.1375
        # raw = 1375, max = 200
        assert result.recommended_size == Decimal("200")
        assert "max_position_pct" in (result.capped_reason or "")

    def test_capping_at_max_order_book_pct(self) -> None:
        sizer = PositionSizer(
            max_position_pct=Decimal("1"),
            min_position_pct=Decimal("0"),
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

    def test_min_position_floor(self) -> None:
        """Small Kelly should be floored at min_position_pct."""
        sizer = PositionSizer(
            max_position_pct=Decimal("0.10"),
            min_position_pct=Decimal("0.01"),
            kelly_multiplier=Decimal("0.25"),
        )
        result = sizer.calculate(
            balance=Decimal("10000"),
            signal_confidence=0.1,  # very low confidence → tiny raw size
            win_rate=Decimal("0.55"),
            avg_win=Decimal("100"),
            avg_loss=Decimal("100"),
        )
        # kelly = 0.55 - 0.45 = 0.10
        # quarter = 0.025, * confidence 0.1 = 0.0025
        # raw = 25 < min 100 → floored
        assert result.recommended_size == Decimal("100")
        assert "min_position_pct" in (result.capped_reason or "")

    def test_book_depth_overrides_min_floor(self) -> None:
        """Book depth cap must override the min floor (liquidity constraint)."""
        sizer = PositionSizer(
            max_position_pct=Decimal("1"),
            min_position_pct=Decimal("0.01"),
            max_order_book_pct=Decimal("0.10"),
        )
        result = sizer.calculate(
            balance=Decimal("100000"),
            signal_confidence=0.01,  # tiny
            win_rate=Decimal("0.55"),
            avg_win=Decimal("100"),
            avg_loss=Decimal("100"),
            book_depth=Decimal("500"),
        )
        # min_by_balance = 1000, but max_by_book = 50
        # Floor raises to 1000, then book depth cap overrides to 50
        assert result.recommended_size == Decimal("50")
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
            min_position_pct=Decimal("0"),
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
        sizer = PositionSizer(max_position_pct=Decimal("0.02"), min_position_pct=Decimal("0"))
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
            min_position_pct=Decimal("0"),
            max_order_book_pct=Decimal("1"),
            kelly_multiplier=Decimal("0.5"),
        )
        quarter = PositionSizer(
            max_position_pct=Decimal("1"),
            min_position_pct=Decimal("0"),
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
            min_position_pct=Decimal("0"),
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
        sizer = PositionSizer(max_position_pct=Decimal("0.02"), min_position_pct=Decimal("0"))
        result = sizer.calculate_binary(
            balance=Decimal("10000"),
            entry_price=Decimal("0.55"),
            estimated_win_prob=Decimal("0.90"),
        )
        # kelly = (0.90-0.55)/(0.55*0.45) ≈ 1.4141
        # quarter ≈ 0.3535, raw ≈ 3535, cap = 200
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
            min_position_pct=Decimal("0"),
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
            min_position_pct=Decimal("0"),
            kelly_multiplier=Decimal("0.5"),
        )
        quarter = PositionSizer(
            max_position_pct=Decimal("1"),
            min_position_pct=Decimal("0"),
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
            min_position_pct=Decimal("0"),
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

    def test_binary_min_floor(self) -> None:
        """Small edge should be floored at min_position_pct."""
        sizer = PositionSizer(
            max_position_pct=Decimal("0.05"),
            min_position_pct=Decimal("0.01"),
            kelly_multiplier=Decimal("0.25"),
        )
        # p=0.52, P=0.50 -> kelly = (0.52-0.50)/(0.50*0.50) = 0.02/0.25 = 0.08
        # quarter_kelly = 0.02, raw = 200
        # min = 10000 * 0.01 = 100, max = 500
        # 200 > 100, not floored
        result = sizer.calculate_binary(
            balance=Decimal("10000"),
            entry_price=Decimal("0.50"),
            estimated_win_prob=Decimal("0.52"),
        )
        assert result.recommended_size == Decimal("200")
        assert result.capped_reason is None

        # Now with tiny edge: p=0.505, P=0.50
        # kelly = (0.505-0.50)/(0.50*0.50) = 0.005/0.25 = 0.02
        # quarter = 0.005, raw = 50
        # min = 100 → floored
        result2 = sizer.calculate_binary(
            balance=Decimal("10000"),
            entry_price=Decimal("0.50"),
            estimated_win_prob=Decimal("0.505"),
        )
        assert result2.recommended_size == Decimal("100")
        assert "min_position_pct" in (result2.capped_reason or "")

    def test_binary_book_depth_overrides_min_floor(self) -> None:
        """Book depth cap is a hard liquidity limit that overrides min floor."""
        sizer = PositionSizer(
            max_position_pct=Decimal("1"),
            min_position_pct=Decimal("0.01"),
            max_order_book_pct=Decimal("0.10"),
        )
        result = sizer.calculate_binary(
            balance=Decimal("100000"),
            entry_price=Decimal("0.60"),
            estimated_win_prob=Decimal("0.90"),
            book_depth=Decimal("500"),
        )
        # min_by_balance = 1000, but max_by_book = 50
        # Floor raises to 1000, then book depth overrides to 50
        assert result.recommended_size == Decimal("50")
        assert "max_order_book_pct" in (result.capped_reason or "")

    def test_binary_default_params(self) -> None:
        """Default params: min=1%, max=5%, quarter-Kelly."""
        sizer = PositionSizer()
        # p=0.93, P=0.50 -> large edge
        # kelly = (0.93-0.50)/(0.50*0.50) = 0.43/0.25 = 1.72
        # quarter = 0.43, raw = 4300
        # max = 500, min = 100 → capped at 500
        result = sizer.calculate_binary(
            balance=Decimal("10000"),
            entry_price=Decimal("0.50"),
            estimated_win_prob=Decimal("0.93"),
        )
        assert result.recommended_size == Decimal("500")
        assert "max_position_pct" in (result.capped_reason or "")
