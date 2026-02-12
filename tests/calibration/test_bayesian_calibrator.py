"""Tests for the Bayesian calibrator."""

from __future__ import annotations

import pytest

from src.calibration.bayesian_calibrator import (
    BayesianCalibrator,
    CalibrationResult,
    RollingWinProbability,
)
from src.calibration.calibration_tracker import CalibrationTracker


class TestCalibrationResult:
    """Tests for the CalibrationResult dataclass."""

    def test_frozen_dataclass(self) -> None:
        result = CalibrationResult(
            prior=0.60,
            likelihood_ratio=1.3,
            posterior=0.66,
            confidence=0.75,
            clob_mid=0.60,
            direction="YES",
            capped=False,
        )
        assert result.prior == 0.60
        assert result.direction == "YES"
        assert result.capped is False
        with pytest.raises(AttributeError):
            result.prior = 0.70  # type: ignore[misc]


class TestRollingWinProbability:
    """Tests for the Beta-distribution rolling win probability."""

    def test_initial_estimate(self) -> None:
        rwp = RollingWinProbability(alpha=83.8, beta=16.2)
        # 83.8 / (83.8 + 16.2) = 0.838
        assert pytest.approx(rwp.estimate(), abs=1e-4) == 0.838

    def test_update_win_increases_estimate(self) -> None:
        rwp = RollingWinProbability(alpha=50.0, beta=50.0)
        baseline = rwp.estimate()
        rwp.update(won=True)
        assert rwp.estimate() > baseline

    def test_update_loss_decreases_estimate(self) -> None:
        rwp = RollingWinProbability(alpha=50.0, beta=50.0)
        baseline = rwp.estimate()
        rwp.update(won=False)
        assert rwp.estimate() < baseline

    def test_decay_applied_on_update(self) -> None:
        rwp = RollingWinProbability(alpha=100.0, beta=0.0, decay=0.98)
        rwp.update(won=True)
        # After decay: alpha = 100 * 0.98 + 1 = 99, beta = 0
        assert pytest.approx(rwp.alpha, abs=1e-6) == 99.0
        assert pytest.approx(rwp.beta, abs=1e-6) == 0.0

    def test_multiple_losses_converge_down(self) -> None:
        rwp = RollingWinProbability(alpha=83.8, beta=16.2, decay=0.98)
        for _ in range(50):
            rwp.update(won=False)
        # After 50 consecutive losses, estimate should drop well below 0.838
        assert rwp.estimate() < 0.50

    def test_zero_total_returns_half(self) -> None:
        rwp = RollingWinProbability(alpha=0.0, beta=0.0, decay=0.98)
        assert rwp.estimate() == 0.5


class TestBayesianCalibrator:
    """Tests for the main Bayesian calibrator."""

    def test_basic_calibration_with_defaults(self) -> None:
        cal = BayesianCalibrator(
            max_edge=0.15,
            default_likelihood_ratio=1.3,
        )
        result = cal.calibrate(
            clob_mid=0.60,
            signal_confidence=0.75,
            direction="YES",
            minute=5,
        )
        assert isinstance(result, CalibrationResult)
        assert result.prior == 0.60
        assert result.likelihood_ratio == 1.3
        assert result.direction == "YES"
        # posterior = 1.3 * 0.60 / (1.3 * 0.60 + 0.40) = 0.78 / 1.18 ~ 0.6610
        expected = (1.3 * 0.60) / (1.3 * 0.60 + 0.40)
        assert pytest.approx(result.posterior, abs=1e-4) == expected
        assert result.capped is False

    def test_posterior_capping(self) -> None:
        """When posterior exceeds clob_mid + max_edge, it must be capped."""
        cal = BayesianCalibrator(
            max_edge=0.05,
            default_likelihood_ratio=3.0,
        )
        result = cal.calibrate(
            clob_mid=0.60,
            signal_confidence=0.90,
            direction="YES",
            minute=7,
        )
        assert result.capped is True
        assert result.posterior == pytest.approx(0.65, abs=1e-6)

    def test_posterior_clamped_to_099(self) -> None:
        """Posterior should never exceed 0.99."""
        cal = BayesianCalibrator(
            max_edge=0.50,
            default_likelihood_ratio=10.0,
        )
        result = cal.calibrate(
            clob_mid=0.95,
            signal_confidence=0.99,
            direction="YES",
            minute=10,
        )
        assert result.posterior <= 0.99

    def test_posterior_clamped_to_001(self) -> None:
        """Posterior should never go below 0.01."""
        cal = BayesianCalibrator(
            max_edge=0.15,
            default_likelihood_ratio=0.01,
        )
        result = cal.calibrate(
            clob_mid=0.02,
            signal_confidence=0.50,
            direction="YES",
            minute=3,
        )
        assert result.posterior >= 0.01

    def test_desert_clob_wider_max_edge(self) -> None:
        """CLOB mid ~0.50 (within 0.05) should use wider max_edge of 0.25."""
        cal = BayesianCalibrator(
            max_edge=0.15,
            default_likelihood_ratio=5.0,
        )
        # With mid=0.50 and LR=5.0:
        # posterior = 5.0 * 0.50 / (5.0 * 0.50 + 0.50) = 2.5 / 3.0 = 0.8333
        # normal cap = 0.50 + 0.15 = 0.65  (would be capped)
        # desert cap = 0.50 + 0.25 = 0.75  (still capped but at 0.75)
        result = cal.calibrate(
            clob_mid=0.50,
            signal_confidence=0.90,
            direction="YES",
            minute=5,
        )
        assert result.capped is True
        assert result.posterior == pytest.approx(0.75, abs=1e-6)

    def test_desert_boundary_at_045(self) -> None:
        """Mid=0.45 is exactly at desert boundary (|0.45 - 0.50| = 0.05)."""
        cal = BayesianCalibrator(
            max_edge=0.15,
            default_likelihood_ratio=5.0,
        )
        result = cal.calibrate(
            clob_mid=0.45,
            signal_confidence=0.90,
            direction="YES",
            minute=5,
        )
        # 0.45 is within desert zone (|0.45 - 0.50| = 0.05 <= 0.05)
        # desert cap = 0.45 + 0.25 = 0.70
        assert result.capped is True
        assert result.posterior == pytest.approx(0.70, abs=1e-6)

    def test_non_desert_at_056(self) -> None:
        """Mid=0.56 is outside desert (|0.56 - 0.50| = 0.06 > 0.05)."""
        cal = BayesianCalibrator(
            max_edge=0.10,
            default_likelihood_ratio=5.0,
        )
        result = cal.calibrate(
            clob_mid=0.56,
            signal_confidence=0.90,
            direction="YES",
            minute=5,
        )
        # normal cap = 0.56 + 0.10 = 0.66
        assert result.capped is True
        assert result.posterior == pytest.approx(0.66, abs=1e-6)

    def test_with_tracker_providing_likelihood_ratio(self) -> None:
        """When tracker has enough samples, use data-driven LR."""
        tracker = CalibrationTracker()
        # Add 35 predictions in the 0.75-0.85 bin with known outcomes
        for i in range(35):
            tid = f"trade_{i}"
            tracker.record_prediction(
                confidence=0.80,
                direction="YES",
                clob_mid=0.60,
                trade_id=tid,
            )
            # 30 wins, 5 losses → bin win rate = 30/35
            tracker.record_outcome(tid, won=(i < 30))

        cal = BayesianCalibrator(
            max_edge=0.30,
            default_likelihood_ratio=1.3,
            min_samples=30,
        )
        result = cal.calibrate(
            clob_mid=0.60,
            signal_confidence=0.80,
            direction="YES",
            minute=5,
            tracker=tracker,
        )
        # The LR should come from tracker, not the default 1.3
        assert result.likelihood_ratio != 1.3

    def test_with_tracker_insufficient_samples(self) -> None:
        """When tracker has too few samples, fall back to default LR."""
        tracker = CalibrationTracker()
        for i in range(5):
            tid = f"trade_{i}"
            tracker.record_prediction(
                confidence=0.80,
                direction="YES",
                clob_mid=0.60,
                trade_id=tid,
            )
            tracker.record_outcome(tid, won=True)

        cal = BayesianCalibrator(
            max_edge=0.30,
            default_likelihood_ratio=1.3,
            min_samples=30,
        )
        result = cal.calibrate(
            clob_mid=0.60,
            signal_confidence=0.80,
            direction="YES",
            minute=5,
            tracker=tracker,
        )
        assert result.likelihood_ratio == 1.3

    def test_no_direction(self) -> None:
        """Direction should be stored as-is in the result."""
        cal = BayesianCalibrator()
        result = cal.calibrate(
            clob_mid=0.40,
            signal_confidence=0.70,
            direction="NO",
            minute=8,
        )
        assert result.direction == "NO"

    def test_normalizer_zero_edge_case(self) -> None:
        """When prior is 1.0, normalizer edge case."""
        cal = BayesianCalibrator(max_edge=0.15, default_likelihood_ratio=1.3)
        result = cal.calibrate(
            clob_mid=0.99,
            signal_confidence=0.90,
            direction="YES",
            minute=5,
        )
        # prior=0.99, numerator=1.3*0.99=1.287, normalizer=1.287+0.01=1.297
        # posterior = 1.287/1.297 = 0.9923 → capped at 0.99+0.15=1.14 → clamped to 0.99
        assert result.posterior <= 0.99
