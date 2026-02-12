"""Tests for the edge calculator."""

from __future__ import annotations

import pytest

from src.calibration.edge_calculator import EdgeCalculator, EdgeResult


class TestPolymarketFee:
    """Tests for the Polymarket fee formula."""

    def test_fee_at_050(self) -> None:
        """Fee peaks at price=0.50."""
        ec = EdgeCalculator(fee_constant=0.25)
        fee = ec.polymarket_fee(0.50)
        # 0.25 * 0.25 * 0.25 = 0.015625
        assert pytest.approx(fee, abs=1e-6) == 0.015625

    def test_fee_at_000(self) -> None:
        """Fee is zero at price=0."""
        ec = EdgeCalculator(fee_constant=0.25)
        fee = ec.polymarket_fee(0.0)
        assert fee == 0.0

    def test_fee_at_100(self) -> None:
        """Fee is zero at price=1."""
        ec = EdgeCalculator(fee_constant=0.25)
        fee = ec.polymarket_fee(1.0)
        assert fee == 0.0

    def test_fee_symmetric(self) -> None:
        """Fee at price p should equal fee at price (1-p)."""
        ec = EdgeCalculator(fee_constant=0.25)
        for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            assert pytest.approx(
                ec.polymarket_fee(p), abs=1e-10
            ) == ec.polymarket_fee(1.0 - p)

    def test_fee_at_025(self) -> None:
        ec = EdgeCalculator(fee_constant=0.25)
        fee = ec.polymarket_fee(0.25)
        # 0.25 * 0.0625 * 0.5625 = 0.25 * 0.03515625 = 0.00878906
        assert pytest.approx(fee, abs=1e-6) == 0.008789

    def test_custom_fee_constant(self) -> None:
        ec = EdgeCalculator(fee_constant=1.0)
        fee = ec.polymarket_fee(0.50)
        # 1.0 * 0.25 * 0.25 = 0.0625
        assert pytest.approx(fee, abs=1e-6) == 0.0625


class TestEdgeCalculation:
    """Tests for full edge calculation."""

    def test_basic_tradeable(self) -> None:
        ec = EdgeCalculator(min_edge=0.03, fee_constant=0.25)
        result = ec.calculate(
            posterior=0.70,
            entry_price=0.60,
            clob_mid=0.60,
        )
        assert isinstance(result, EdgeResult)
        assert result.raw_edge == pytest.approx(0.10, abs=1e-6)
        fee = 0.25 * (0.60 ** 2) * (0.40 ** 2)
        assert result.fee == pytest.approx(fee, abs=1e-6)
        # fee_adjusted_edge = raw_edge - fee * entry_price (per-share fee cost)
        assert result.fee_adjusted_edge == pytest.approx(
            0.10 - fee * 0.60, abs=1e-6
        )
        assert result.is_tradeable is True

    def test_not_tradeable_due_to_fee(self) -> None:
        ec = EdgeCalculator(min_edge=0.05, fee_constant=0.25)
        # Small raw edge that gets eaten by fees
        result = ec.calculate(
            posterior=0.55,
            entry_price=0.50,
            clob_mid=0.50,
        )
        # raw_edge = 0.05
        # fee at 0.50 = 0.015625
        # adjusted = 0.05 - 0.015625 * 0.50 = 0.05 - 0.0078125 = 0.0421875
        # 0.0421875 < 0.05 → not tradeable
        assert result.is_tradeable is False

    def test_negative_edge(self) -> None:
        """When posterior < entry_price, edge is negative."""
        ec = EdgeCalculator(min_edge=0.03, fee_constant=0.25)
        result = ec.calculate(
            posterior=0.55,
            entry_price=0.60,
            clob_mid=0.60,
        )
        assert result.raw_edge < 0
        assert result.fee_adjusted_edge < 0
        assert result.is_tradeable is False

    def test_edge_at_extreme_low_price(self) -> None:
        """Price near 0 has negligible fee."""
        ec = EdgeCalculator(min_edge=0.03, fee_constant=0.25)
        result = ec.calculate(
            posterior=0.10,
            entry_price=0.05,
            clob_mid=0.05,
        )
        # fee at 0.05 = 0.25 * 0.0025 * 0.9025 = 0.000564
        assert result.fee < 0.001
        assert result.raw_edge == pytest.approx(0.05, abs=1e-6)
        assert result.is_tradeable is True

    def test_edge_at_extreme_high_price(self) -> None:
        """Price near 1 has negligible fee."""
        ec = EdgeCalculator(min_edge=0.03, fee_constant=0.25)
        result = ec.calculate(
            posterior=0.99,
            entry_price=0.95,
            clob_mid=0.95,
        )
        # fee at 0.95 = 0.25 * 0.9025 * 0.0025 = 0.000564
        assert result.fee < 0.001
        assert result.raw_edge == pytest.approx(0.04, abs=1e-6)
        assert result.is_tradeable is True

    def test_exact_min_edge_is_tradeable(self) -> None:
        """fee_adjusted_edge == min_edge should be tradeable."""
        ec = EdgeCalculator(min_edge=0.03, fee_constant=0.25)
        # Find an entry price with fee such that adjusted edge = exactly 0.03
        # We'll use entry_price=0.0 (fee=0) for simplicity
        result = ec.calculate(
            posterior=0.03,
            entry_price=0.0,
            clob_mid=0.0,
        )
        assert result.fee == 0.0
        assert result.fee_adjusted_edge == pytest.approx(0.03, abs=1e-6)
        assert result.is_tradeable is True

    def test_result_fields(self) -> None:
        ec = EdgeCalculator(min_edge=0.03, fee_constant=0.25)
        result = ec.calculate(
            posterior=0.70,
            entry_price=0.60,
            clob_mid=0.58,
        )
        assert result.posterior == 0.70
        assert result.entry_price == 0.60
        assert result.clob_mid == 0.58
        assert result.min_edge == 0.03

    def test_frozen_dataclass(self) -> None:
        ec = EdgeCalculator()
        result = ec.calculate(
            posterior=0.70,
            entry_price=0.60,
            clob_mid=0.60,
        )
        with pytest.raises(AttributeError):
            result.posterior = 0.80  # type: ignore[misc]
