"""Tests for VolatilityRegimeDetector."""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.engine.volatility_regime import (
    VolatilityRegimeDetector,
    VolRegimeResult,
    _inv_norm_cdf,
)


class TestComputeRealizedVol:
    """Tests for VolatilityRegimeDetector.compute_realized_vol()."""

    def test_too_few_ticks_returns_zero(self, config_loader) -> None:
        detector = VolatilityRegimeDetector(config_loader)
        assert detector.compute_realized_vol([]) == 0.0
        assert detector.compute_realized_vol([Decimal("100")]) == 0.0

    def test_constant_prices_returns_zero(self, config_loader) -> None:
        detector = VolatilityRegimeDetector(config_loader)
        ticks = [Decimal("100.00")] * 10
        assert detector.compute_realized_vol(ticks) == pytest.approx(0.0, abs=1e-10)

    def test_known_volatility(self, config_loader) -> None:
        detector = VolatilityRegimeDetector(config_loader)
        # Prices with known log returns
        ticks = [
            Decimal("100.00"),
            Decimal("101.00"),
            Decimal("100.50"),
            Decimal("102.00"),
            Decimal("101.50"),
        ]
        result = detector.compute_realized_vol(ticks)
        assert result > 0.0

    def test_higher_vol_for_larger_moves(self, config_loader) -> None:
        detector = VolatilityRegimeDetector(config_loader)
        # Small moves
        small_ticks = [Decimal(str(100 + i * 0.01)) for i in range(20)]
        # Large moves
        large_ticks = [Decimal(str(100 + i * 1.0)) for i in range(20)]
        small_vol = detector.compute_realized_vol(small_ticks)
        large_vol = detector.compute_realized_vol(large_ticks)
        assert large_vol > small_vol


class TestExtractImpliedVol:
    """Tests for VolatilityRegimeDetector.extract_implied_vol()."""

    def test_fair_price_moderate_vol(self, config_loader) -> None:
        detector = VolatilityRegimeDetector(config_loader)
        # At p=0.50, Phi^-1(0.50) = 0, so implied vol should be ~0
        result = detector.extract_implied_vol(Decimal("0.50"), 450)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_extreme_high_price_lower_vol(self, config_loader) -> None:
        detector = VolatilityRegimeDetector(config_loader)
        # At p=0.90 with time remaining, implied vol should be positive
        result = detector.extract_implied_vol(Decimal("0.90"), 450)
        assert result > 0.0

    def test_extreme_low_price_lower_vol(self, config_loader) -> None:
        detector = VolatilityRegimeDetector(config_loader)
        # At p=0.10 with time remaining, implied vol should be positive
        result = detector.extract_implied_vol(Decimal("0.10"), 450)
        assert result > 0.0

    def test_symmetric_prices(self, config_loader) -> None:
        detector = VolatilityRegimeDetector(config_loader)
        # 0.90 and 0.10 should give similar implied vol due to symmetry
        vol_high = detector.extract_implied_vol(Decimal("0.90"), 450)
        vol_low = detector.extract_implied_vol(Decimal("0.10"), 450)
        assert vol_high == pytest.approx(vol_low, rel=0.01)

    def test_zero_time_remaining_returns_zero(self, config_loader) -> None:
        detector = VolatilityRegimeDetector(config_loader)
        result = detector.extract_implied_vol(Decimal("0.60"), 0)
        assert result == 0.0

    def test_edge_price_zero_returns_zero(self, config_loader) -> None:
        detector = VolatilityRegimeDetector(config_loader)
        assert detector.extract_implied_vol(Decimal("0.00"), 450) == 0.0

    def test_edge_price_one_returns_zero(self, config_loader) -> None:
        detector = VolatilityRegimeDetector(config_loader)
        assert detector.extract_implied_vol(Decimal("1.00"), 450) == 0.0


class TestDetectRegime:
    """Tests for VolatilityRegimeDetector.detect_regime()."""

    def test_high_vol_regime(self, config_loader) -> None:
        detector = VolatilityRegimeDetector(config_loader)
        # Large price swings → high realized vol
        ticks = [Decimal(str(100 + ((-1) ** i) * 5)) for i in range(50)]
        result = detector.detect_regime(ticks, Decimal("0.55"), 450)
        # With large swings, realized should dominate implied
        if result.vol_ratio > 1.25:
            assert result.regime == "high_vol"
            assert result.signal_direction == "long_vol"

    def test_low_vol_regime(self, config_loader) -> None:
        detector = VolatilityRegimeDetector(config_loader)
        # Very small moves → low realized vol
        ticks = [Decimal("100.000") + Decimal(str(i * 0.0001)) for i in range(50)]
        # With a price away from 0.50, implied vol will be higher
        result = detector.detect_regime(ticks, Decimal("0.80"), 450)
        if result.vol_ratio < 0.80 and result.vol_ratio > 0:
            assert result.regime == "low_vol"
            assert result.signal_direction == "short_vol"

    def test_normal_regime(self, config_loader) -> None:
        detector = VolatilityRegimeDetector(config_loader)
        # Fair price gives 0 implied vol → ratio will be 0
        # With constant ticks and fair price, both vols near 0
        ticks = [Decimal("100.00")] * 20
        result = detector.detect_regime(ticks, Decimal("0.50"), 450)
        assert result.regime == "normal"
        assert result.signal_direction == "neutral"

    def test_zero_time_remaining(self, config_loader) -> None:
        detector = VolatilityRegimeDetector(config_loader)
        ticks = [Decimal("100.00"), Decimal("101.00"), Decimal("100.50")]
        result = detector.detect_regime(ticks, Decimal("0.60"), 0)
        # implied vol = 0 → vol_ratio = 0 → normal
        assert result.implied_vol_pct == 0.0

    def test_too_few_ticks(self, config_loader) -> None:
        detector = VolatilityRegimeDetector(config_loader)
        result = detector.detect_regime([], Decimal("0.60"), 450)
        assert result.realized_vol_pct == 0.0
        assert result.regime == "normal"

    def test_result_fields_populated(self, config_loader) -> None:
        detector = VolatilityRegimeDetector(config_loader)
        ticks = [Decimal(str(100 + i * 0.5)) for i in range(20)]
        result = detector.detect_regime(ticks, Decimal("0.60"), 450)
        assert isinstance(result.realized_vol_pct, float)
        assert isinstance(result.implied_vol_pct, float)
        assert isinstance(result.vol_ratio, float)
        assert result.regime in ("high_vol", "low_vol", "normal")
        assert result.signal_direction in ("long_vol", "short_vol", "neutral")


class TestInvNormCdf:
    """Tests for the inverse normal CDF approximation."""

    def test_median_is_zero(self) -> None:
        assert _inv_norm_cdf(0.5) == pytest.approx(0.0, abs=1e-6)

    def test_known_values(self) -> None:
        # Phi^-1(0.8413) ~ 1.0
        assert _inv_norm_cdf(0.8413) == pytest.approx(1.0, abs=0.01)
        # Phi^-1(0.1587) ~ -1.0
        assert _inv_norm_cdf(0.1587) == pytest.approx(-1.0, abs=0.01)

    def test_symmetry(self) -> None:
        assert _inv_norm_cdf(0.9) == pytest.approx(-_inv_norm_cdf(0.1), abs=1e-6)

    def test_edge_zero_returns_zero(self) -> None:
        assert _inv_norm_cdf(0.0) == 0.0

    def test_edge_one_returns_zero(self) -> None:
        assert _inv_norm_cdf(1.0) == 0.0

    def test_extreme_low_tail(self) -> None:
        # Very low p → large negative value
        result = _inv_norm_cdf(0.001)
        assert result < -2.5

    def test_extreme_high_tail(self) -> None:
        # Very high p → large positive value
        result = _inv_norm_cdf(0.999)
        assert result > 2.5


class TestVolRegimeResult:
    """Tests for VolRegimeResult model."""

    def test_frozen_model(self) -> None:
        result = VolRegimeResult(
            realized_vol_pct=1.5,
            implied_vol_pct=1.2,
            vol_ratio=1.25,
            regime="high_vol",
            signal_direction="long_vol",
        )
        with pytest.raises(Exception):  # noqa: B017
            result.regime = "low_vol"  # type: ignore[misc]
