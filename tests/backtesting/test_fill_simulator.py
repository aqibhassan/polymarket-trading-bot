"""Tests for FillSimulator — slippage and fee modelling."""

from __future__ import annotations

import math
from decimal import Decimal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from src.backtesting.fill_simulator import FillSimulator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simulator() -> FillSimulator:
    return FillSimulator()


@pytest.fixture
def custom_simulator() -> FillSimulator:
    return FillSimulator(
        base_slippage=Decimal("0.001"),
        default_fee_rate=Decimal("0.003"),
    )


# ---------------------------------------------------------------------------
# Small order slippage (< 5% of book)
# ---------------------------------------------------------------------------

class TestSmallOrderSlippage:
    def test_slippage_linear_for_small_order(self, simulator: FillSimulator) -> None:
        """Small orders use linear slippage: size_ratio * base_slippage."""
        order_price = Decimal("100")
        order_size = Decimal("1")       # 1% of book
        book_depth = Decimal("100")

        fill = simulator.simulate_fill(order_price, order_size, book_depth)

        # size_ratio = 1/100 = 0.01 (< 0.05 threshold)
        # slippage_pct = 0.01 * 0.0005 = 0.000005
        # slippage_amount = 100 * 0.000005 = 0.0005
        expected_slippage = Decimal("100") * (Decimal("0.01") * Decimal("0.0005"))
        assert fill.slippage == expected_slippage
        assert fill.fill_price == order_price + expected_slippage

    def test_zero_size_order(self, simulator: FillSimulator) -> None:
        fill = simulator.simulate_fill(Decimal("50"), Decimal("0"), Decimal("100"))
        assert fill.slippage == Decimal("0")
        assert fill.fee == Decimal("0")


# ---------------------------------------------------------------------------
# Large order slippage (>= 5% of book)
# ---------------------------------------------------------------------------

class TestLargeOrderSlippage:
    def test_slippage_sqrt_for_large_order(self, simulator: FillSimulator) -> None:
        """Large orders use sqrt slippage: sqrt(size_ratio) * base_slippage."""
        order_price = Decimal("100")
        order_size = Decimal("10")       # 10% of book
        book_depth = Decimal("100")

        fill = simulator.simulate_fill(order_price, order_size, book_depth)

        # size_ratio = 0.10 (>= 0.05)
        sqrt_ratio = Decimal(str(math.sqrt(0.10)))
        expected_slippage = order_price * sqrt_ratio * Decimal("0.0005")
        assert fill.slippage == expected_slippage

    def test_very_large_order(self, simulator: FillSimulator) -> None:
        """Order equal to entire book depth."""
        fill = simulator.simulate_fill(
            Decimal("100"), Decimal("100"), Decimal("100"),
        )
        # size_ratio = 1.0, sqrt(1.0) = 1.0
        expected_slippage = Decimal("100") * Decimal("1.0") * Decimal("0.0005")
        assert fill.slippage == expected_slippage


# ---------------------------------------------------------------------------
# Fee calculation
# ---------------------------------------------------------------------------

class TestFeeCalculation:
    def test_default_fee_rate(self, simulator: FillSimulator) -> None:
        fill = simulator.simulate_fill(
            Decimal("100"), Decimal("10"), Decimal("1000"),
        )
        notional = fill.fill_price * Decimal("10")
        expected_fee = notional * Decimal("0.002")
        assert fill.fee == expected_fee

    def test_custom_fee_rate(self, simulator: FillSimulator) -> None:
        fill = simulator.simulate_fill(
            Decimal("100"), Decimal("10"), Decimal("1000"),
            fee_rate=Decimal("0.005"),
        )
        notional = fill.fill_price * Decimal("10")
        expected_fee = notional * Decimal("0.005")
        assert fill.fee == expected_fee

    def test_zero_fee(self, simulator: FillSimulator) -> None:
        fill = simulator.simulate_fill(
            Decimal("100"), Decimal("10"), Decimal("1000"),
            fee_rate=Decimal("0"),
        )
        assert fill.fee == Decimal("0")


# ---------------------------------------------------------------------------
# Net cost
# ---------------------------------------------------------------------------

class TestNetCost:
    def test_net_cost_equals_notional_plus_fee(self, simulator: FillSimulator) -> None:
        fill = simulator.simulate_fill(
            Decimal("100"), Decimal("5"), Decimal("1000"),
        )
        expected_notional = fill.fill_price * Decimal("5")
        assert fill.net_cost == expected_notional + fill.fee


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_book_depth_uses_base_slippage(self, simulator: FillSimulator) -> None:
        fill = simulator.simulate_fill(
            Decimal("100"), Decimal("5"), Decimal("0"),
        )
        expected_slippage = Decimal("100") * Decimal("0.0005")
        assert fill.slippage == expected_slippage

    def test_custom_simulator_params(self, custom_simulator: FillSimulator) -> None:
        fill = custom_simulator.simulate_fill(
            Decimal("100"), Decimal("1"), Decimal("100"),
        )
        # size_ratio = 0.01 < 0.05, so linear: 0.01 * 0.001 = 0.00001
        expected_slippage = Decimal("100") * Decimal("0.01") * Decimal("0.001")
        assert fill.slippage == expected_slippage


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

class TestProperties:
    @given(
        price=st.decimals(
            min_value=1, max_value=10000, places=2,
            allow_nan=False, allow_infinity=False,
        ),
        size=st.decimals(
            min_value=0, max_value=1000, places=2,
            allow_nan=False, allow_infinity=False,
        ),
        depth=st.decimals(
            min_value=1, max_value=100000, places=2,
            allow_nan=False, allow_infinity=False,
        ),
    )
    @settings(max_examples=50)
    def test_fill_price_gte_order_price(
        self,
        price: Decimal,
        size: Decimal,
        depth: Decimal,
    ) -> None:
        """Fill price should always be >= order price (slippage is non-negative)."""
        sim = FillSimulator()
        fill = sim.simulate_fill(price, size, depth)
        assert fill.fill_price >= price

    @given(
        price=st.decimals(
            min_value=1, max_value=10000, places=2,
            allow_nan=False, allow_infinity=False,
        ),
        size=st.decimals(
            min_value=1, max_value=1000, places=2,
            allow_nan=False, allow_infinity=False,
        ),
        depth=st.decimals(
            min_value=1, max_value=100000, places=2,
            allow_nan=False, allow_infinity=False,
        ),
    )
    @settings(max_examples=50)
    def test_net_cost_gte_zero(
        self,
        price: Decimal,
        size: Decimal,
        depth: Decimal,
    ) -> None:
        """Net cost should always be non-negative."""
        sim = FillSimulator()
        fill = sim.simulate_fill(price, size, depth)
        assert fill.net_cost >= Decimal("0")

    @given(
        price=st.decimals(
            min_value=1, max_value=10000, places=2,
            allow_nan=False, allow_infinity=False,
        ),
        size=st.decimals(
            min_value=1, max_value=1000, places=2,
            allow_nan=False, allow_infinity=False,
        ),
        depth=st.decimals(
            min_value=1, max_value=100000, places=2,
            allow_nan=False, allow_infinity=False,
        ),
    )
    @settings(max_examples=50)
    def test_simulated_fill_is_frozen(
        self,
        price: Decimal,
        size: Decimal,
        depth: Decimal,
    ) -> None:
        """SimulatedFill should be immutable."""
        sim = FillSimulator()
        fill = sim.simulate_fill(price, size, depth)
        with pytest.raises(ValidationError):
            fill.fill_price = Decimal("0")  # type: ignore[misc]
