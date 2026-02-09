"""Tests for CostCalculator — trade friction calculation."""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from src.risk.cost_calculator import CostCalculator, TradeCost


class TestCostCalculator:
    def test_basic_calculation(self) -> None:
        calc = CostCalculator()
        result = calc.calculate(
            entry_price=Decimal("0.50"),
            size=Decimal("1000"),
            spread=Decimal("0.02"),
        )
        assert isinstance(result, TradeCost)
        # spread_cost = (0.02/2) * 1000 = 10
        assert result.spread_cost == Decimal("10")
        # fee_cost = 0.50 * 1000 * 0.002 = 1.0
        assert result.fee_cost == Decimal("1.000")
        # slippage_cost = 0.50 * 1000 * (5/10000) = 0.25
        assert result.slippage_cost == Decimal("0.25000")
        # total = 10 + 1 + 0.25 = 11.25
        assert result.total_cost == Decimal("11.25000")
        # min_profitable_move = 11.25 / 1000 = 0.01125
        assert result.min_profitable_move == Decimal("0.01125000")

    def test_custom_fee_rate(self) -> None:
        calc = CostCalculator()
        result = calc.calculate(
            entry_price=Decimal("0.50"),
            size=Decimal("1000"),
            spread=Decimal("0.02"),
            fee_rate=Decimal("0.001"),
        )
        # fee_cost = 0.50 * 1000 * 0.001 = 0.5
        assert result.fee_cost == Decimal("0.500")

    def test_zero_spread(self) -> None:
        calc = CostCalculator()
        result = calc.calculate(
            entry_price=Decimal("0.60"),
            size=Decimal("500"),
            spread=Decimal("0"),
        )
        assert result.spread_cost == Decimal("0")
        assert result.total_cost > 0  # still has fees and slippage

    def test_zero_size(self) -> None:
        calc = CostCalculator()
        result = calc.calculate(
            entry_price=Decimal("0.50"),
            size=Decimal("0"),
            spread=Decimal("0.02"),
        )
        assert result.total_cost == Decimal("0")
        assert result.min_profitable_move == Decimal("0")

    def test_large_trade(self) -> None:
        calc = CostCalculator()
        result = calc.calculate(
            entry_price=Decimal("0.80"),
            size=Decimal("10000"),
            spread=Decimal("0.01"),
        )
        # spread_cost = (0.01/2) * 10000 = 50
        assert result.spread_cost == Decimal("50")
        # fee_cost = 0.80 * 10000 * 0.002 = 16
        assert result.fee_cost == Decimal("16.000")
        assert result.total_cost > Decimal("66")  # plus slippage

    def test_custom_slippage_bps(self) -> None:
        calc = CostCalculator(slippage_bps=Decimal("10"))
        result = calc.calculate(
            entry_price=Decimal("0.50"),
            size=Decimal("1000"),
            spread=Decimal("0"),
        )
        # slippage = 0.50 * 1000 * (10/10000) = 0.50
        assert result.slippage_cost == Decimal("0.50000")

    def test_model_frozen(self) -> None:
        calc = CostCalculator()
        result = calc.calculate(
            entry_price=Decimal("0.50"),
            size=Decimal("1000"),
            spread=Decimal("0.02"),
        )
        with pytest.raises(ValidationError):
            result.total_cost = Decimal("999")  # type: ignore[misc]

    def test_min_profitable_move_covers_costs(self) -> None:
        calc = CostCalculator()
        result = calc.calculate(
            entry_price=Decimal("0.55"),
            size=Decimal("200"),
            spread=Decimal("0.03"),
        )
        # If price moves by min_profitable_move per unit, total profit = total_cost
        profit = result.min_profitable_move * Decimal("200")
        assert profit == result.total_cost

    def test_fee_rate_zero(self) -> None:
        calc = CostCalculator()
        result = calc.calculate(
            entry_price=Decimal("0.50"),
            size=Decimal("100"),
            spread=Decimal("0.02"),
            fee_rate=Decimal("0"),
        )
        assert result.fee_cost == Decimal("0")


class TestPolymarketFee:
    """Tests for Polymarket dynamic taker fee curve."""

    def test_fee_at_50_percent(self) -> None:
        """Fee peaks at 50/50 odds — $0.78 per 100 shares."""
        fee = CostCalculator.polymarket_fee(Decimal("50"), Decimal("0.50"))
        # fee = 50 * 0.25 * 0.25 * 0.25 = 0.78125
        assert fee > Decimal("0.78")
        assert fee < Decimal("0.79")

    def test_fee_at_extremes_is_low(self) -> None:
        """Fee drops toward zero at 10% and 90%."""
        fee_10 = CostCalculator.polymarket_fee(Decimal("10"), Decimal("0.10"))
        fee_90 = CostCalculator.polymarket_fee(Decimal("10"), Decimal("0.90"))
        # Both should be very small
        assert fee_10 < Decimal("0.03")
        assert fee_90 < Decimal("0.03")

    def test_fee_symmetry(self) -> None:
        """Fee at price P should equal fee at price (1-P) for same position."""
        fee_25 = CostCalculator.polymarket_fee(Decimal("100"), Decimal("0.25"))
        fee_75 = CostCalculator.polymarket_fee(Decimal("100"), Decimal("0.75"))
        # Not exactly equal because P^2*(1-P)^2 is symmetric but fee also has P^2 factor
        # Actually: P^2*(1-P)^2 vs (1-P)^2*P^2 — these ARE symmetric
        # But fee = N * 0.25 * P^2 * (1-P)^2, so yes, symmetric
        assert abs(fee_25 - fee_75) < Decimal("0.0001")

    def test_fee_boundary_zero(self) -> None:
        assert CostCalculator.polymarket_fee(Decimal("100"), Decimal("0")) == Decimal("0")
        assert CostCalculator.polymarket_fee(Decimal("100"), Decimal("1")) == Decimal("0")

    def test_fee_matches_published_table(self) -> None:
        """Verify against Polymarket's published fee table (per 100 shares)."""
        # At price 0.50: $0.78 per 100 shares (= $50 position)
        fee = CostCalculator.polymarket_fee(Decimal("50"), Decimal("0.50"))
        assert Decimal("0.77") < fee < Decimal("0.80")

        # At price 0.25: $0.22 per 100 shares (= $25 position)
        fee = CostCalculator.polymarket_fee(Decimal("25"), Decimal("0.25"))
        assert Decimal("0.21") < fee < Decimal("0.23")

        # At price 0.10: $0.02 per 100 shares (= $10 position)
        fee = CostCalculator.polymarket_fee(Decimal("10"), Decimal("0.10"))
        assert Decimal("0.019") < fee < Decimal("0.021")

    def test_fee_for_100_dollar_position(self) -> None:
        """For $100 position at 0.65 entry, fee should be ~$1.29."""
        fee = CostCalculator.polymarket_fee(Decimal("100"), Decimal("0.65"))
        assert Decimal("1.28") < fee < Decimal("1.31")


class TestBinaryCostCalculation:
    """Tests for binary market cost calculation."""

    def test_basic_binary_cost(self) -> None:
        calc = CostCalculator()
        result = calc.calculate_binary(
            entry_price=Decimal("0.65"),
            position_size=Decimal("100"),
        )
        assert result.fee_cost > Decimal("0")
        assert result.slippage_cost > Decimal("0")
        assert result.total_cost == result.spread_cost + result.fee_cost + result.slippage_cost

    def test_binary_cost_with_spread(self) -> None:
        calc = CostCalculator()
        result = calc.calculate_binary(
            entry_price=Decimal("0.65"),
            position_size=Decimal("100"),
            spread=Decimal("0.02"),
        )
        assert result.spread_cost > Decimal("0")

    def test_binary_min_profitable_move(self) -> None:
        calc = CostCalculator()
        result = calc.calculate_binary(
            entry_price=Decimal("0.50"),
            position_size=Decimal("100"),
        )
        # position_size is shares (100), not USDC
        # profit from min move = num_shares * min_profitable_move should = total_cost
        profit = result.min_profitable_move * Decimal("100")
        assert abs(profit - result.total_cost) < Decimal("0.0001")
