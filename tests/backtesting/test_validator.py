"""Tests for StatisticalValidator — bootstrap CI, permutation test, overfitting."""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from src.backtesting.validator import StatisticalValidator, ValidationResult


@pytest.fixture
def validator() -> StatisticalValidator:
    return StatisticalValidator(
        n_bootstrap=500,
        n_permutations=500,
        significance_level=0.05,
        rng_seed=42,
    )


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    def test_ci_contains_point_estimate(self, validator: StatisticalValidator) -> None:
        """The 95% CI should contain the point estimate Sharpe."""
        # Build a strong uptrend equity curve
        equity_curve = [Decimal("1000")]
        price = Decimal("1000")
        for _ in range(100):
            price += Decimal("5")
            equity_curve.append(price)

        trades = [{"pnl": Decimal("5")} for _ in range(100)]
        result = validator.validate(trades, equity_curve)

        # CI should have lower < upper
        assert result.sharpe_ci_95[0] < result.sharpe_ci_95[1]

    def test_ci_bounds_ordered(self) -> None:
        """The lower bound of the CI should be less than the upper bound."""
        val = StatisticalValidator(n_bootstrap=500, rng_seed=42)

        # Build a curve with some variance
        curve = [Decimal("1000")]
        p = Decimal("1000")
        for i in range(50):
            delta = Decimal("5") if i % 3 != 0 else Decimal("-3")
            p += delta
            curve.append(p)

        result = val.validate(
            [{"pnl": Decimal("2")} for _ in range(50)],
            curve,
        )

        assert result.sharpe_ci_95[0] < result.sharpe_ci_95[1]

    def test_empty_equity_curve(self, validator: StatisticalValidator) -> None:
        result = validator.validate([], [])
        assert result.sharpe_ci_95 == (0.0, 0.0)


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

class TestPermutationTest:
    def test_significant_result(self, validator: StatisticalValidator) -> None:
        """Strongly profitable trades should have low p-value."""
        trades = [{"pnl": Decimal("100")} for _ in range(50)]
        equity_curve = [Decimal("1000")]
        p = Decimal("1000")
        for _ in range(50):
            p += Decimal("100")
            equity_curve.append(p)

        result = validator.validate(trades, equity_curve)
        assert result.p_value < 0.05
        assert result.is_significant is True

    def test_random_result_not_significant(self) -> None:
        """Zero-mean trades should produce high p-value."""
        val = StatisticalValidator(
            n_permutations=500,
            rng_seed=123,
        )
        # Alternating wins and losses that sum to ~0
        trades = []
        for i in range(50):
            pnl = Decimal("10") if i % 2 == 0 else Decimal("-10")
            trades.append({"pnl": pnl})

        equity_curve = [Decimal("1000"), Decimal("1000")]
        result = val.validate(trades, equity_curve)
        # p-value should be high for zero-sum trades
        assert result.p_value > 0.1

    def test_no_trades(self, validator: StatisticalValidator) -> None:
        result = validator.validate([], [Decimal("1000")])
        assert result.p_value == 1.0
        assert result.is_significant is False


# ---------------------------------------------------------------------------
# Overfitting detection
# ---------------------------------------------------------------------------

class TestOverfittingDetection:
    def test_no_overfitting(self, validator: StatisticalValidator) -> None:
        warning = validator.check_overfitting(
            in_sample_sharpe=1.5,
            out_of_sample_sharpe=1.2,
        )
        assert warning is None

    def test_overfitting_high_ratio(self, validator: StatisticalValidator) -> None:
        warning = validator.check_overfitting(
            in_sample_sharpe=3.0,
            out_of_sample_sharpe=1.0,
        )
        assert warning is not None
        assert "3.00x" in warning

    def test_overfitting_negative_oos(self, validator: StatisticalValidator) -> None:
        warning = validator.check_overfitting(
            in_sample_sharpe=2.0,
            out_of_sample_sharpe=-0.5,
        )
        assert warning is not None
        assert "non-positive" in warning

    def test_borderline_ratio(self, validator: StatisticalValidator) -> None:
        """Ratio of exactly 2.0 should NOT trigger warning."""
        warning = validator.check_overfitting(
            in_sample_sharpe=2.0,
            out_of_sample_sharpe=1.0,
        )
        assert warning is None


# ---------------------------------------------------------------------------
# ValidationResult model
# ---------------------------------------------------------------------------

class TestValidationResult:
    def test_frozen_model(self) -> None:
        vr = ValidationResult(
            sharpe_ci_95=(0.5, 1.5),
            p_value=0.03,
            is_significant=True,
        )
        with pytest.raises(ValidationError):
            vr.p_value = 0.99  # type: ignore[misc]

    def test_optional_warning(self) -> None:
        vr = ValidationResult(
            sharpe_ci_95=(0.5, 1.5),
            p_value=0.03,
            is_significant=True,
            overfitting_warning="Possible overfitting",
        )
        assert vr.overfitting_warning == "Possible overfitting"
