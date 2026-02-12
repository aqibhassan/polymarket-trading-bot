"""Tests for the calibration tracker."""

from __future__ import annotations

import pytest

from src.calibration.calibration_tracker import (
    BinStats,
    CalibrationTracker,
)


class TestRecordPrediction:
    """Tests for recording predictions."""

    def test_basic_record(self) -> None:
        tracker = CalibrationTracker()
        tracker.record_prediction(
            confidence=0.80,
            direction="YES",
            clob_mid=0.60,
            trade_id="t1",
        )
        assert len(tracker.predictions) == 1
        assert tracker.predictions[0]["trade_id"] == "t1"
        assert tracker.predictions[0]["confidence"] == 0.80
        assert tracker.predictions[0]["outcome"] is None

    def test_duplicate_prediction_ignored(self) -> None:
        tracker = CalibrationTracker()
        tracker.record_prediction(
            confidence=0.80,
            direction="YES",
            clob_mid=0.60,
            trade_id="t1",
        )
        tracker.record_prediction(
            confidence=0.90,
            direction="NO",
            clob_mid=0.50,
            trade_id="t1",
        )
        # Should still be just 1 record with original values
        assert len(tracker.predictions) == 1
        assert tracker.predictions[0]["confidence"] == 0.80

    def test_multiple_predictions(self) -> None:
        tracker = CalibrationTracker()
        for i in range(5):
            tracker.record_prediction(
                confidence=0.70 + i * 0.05,
                direction="YES",
                clob_mid=0.60,
                trade_id=f"t{i}",
            )
        assert len(tracker.predictions) == 5


class TestRecordOutcome:
    """Tests for recording outcomes."""

    def test_record_win(self) -> None:
        tracker = CalibrationTracker()
        tracker.record_prediction(
            confidence=0.80,
            direction="YES",
            clob_mid=0.60,
            trade_id="t1",
        )
        tracker.record_outcome("t1", won=True)
        assert tracker.predictions[0]["outcome"] is True

    def test_record_loss(self) -> None:
        tracker = CalibrationTracker()
        tracker.record_prediction(
            confidence=0.80,
            direction="YES",
            clob_mid=0.60,
            trade_id="t1",
        )
        tracker.record_outcome("t1", won=False)
        assert tracker.predictions[0]["outcome"] is False

    def test_outcome_for_unknown_trade(self) -> None:
        """Recording outcome for unknown trade should not raise."""
        tracker = CalibrationTracker()
        tracker.record_outcome("nonexistent", won=True)
        assert len(tracker.predictions) == 0


class TestBinAssignment:
    """Tests for confidence bin assignment."""

    def test_bins_cover_range(self) -> None:
        """Values from 0.45 to 1.00 should all have bins."""
        tracker = CalibrationTracker()
        test_values = [0.45, 0.50, 0.55, 0.65, 0.75, 0.85, 0.95, 1.00]
        for val in test_values:
            b = tracker._find_bin(val)
            assert b is not None, f"No bin for confidence={val}"

    def test_bin_boundaries(self) -> None:
        tracker = CalibrationTracker()
        # 0.55 should be in (0.55, 0.65), not (0.45, 0.55)
        assert tracker._find_bin(0.55) == (0.55, 0.65)
        # 0.549 should be in (0.45, 0.55)
        assert tracker._find_bin(0.549) == (0.45, 0.55)

    def test_value_below_bins(self) -> None:
        tracker = CalibrationTracker()
        assert tracker._find_bin(0.30) is None

    def test_last_bin_inclusive(self) -> None:
        """The last bin (0.95, 1.00) should include 1.00."""
        tracker = CalibrationTracker()
        assert tracker._find_bin(1.00) == (0.95, 1.00)
        assert tracker._find_bin(0.95) == (0.95, 1.00)
        assert tracker._find_bin(0.97) == (0.95, 1.00)


class TestGetLikelihoodRatio:
    """Tests for likelihood ratio computation."""

    def test_sufficient_samples(self) -> None:
        tracker = CalibrationTracker()
        # 35 predictions in 0.75-0.85 bin, all wins
        for i in range(35):
            tid = f"t{i}"
            tracker.record_prediction(
                confidence=0.80,
                direction="YES",
                clob_mid=0.60,
                trade_id=tid,
            )
            tracker.record_outcome(tid, won=True)

        lr = tracker.get_likelihood_ratio(0.80, min_samples=30)
        assert lr is not None
        # All wins → bin WR = 1.0, base rate = 1.0, LR = 1.0
        assert pytest.approx(lr, abs=1e-4) == 1.0

    def test_insufficient_samples(self) -> None:
        tracker = CalibrationTracker()
        for i in range(10):
            tid = f"t{i}"
            tracker.record_prediction(
                confidence=0.80,
                direction="YES",
                clob_mid=0.60,
                trade_id=tid,
            )
            tracker.record_outcome(tid, won=True)

        lr = tracker.get_likelihood_ratio(0.80, min_samples=30)
        assert lr is None

    def test_no_outcomes(self) -> None:
        tracker = CalibrationTracker()
        for i in range(35):
            tracker.record_prediction(
                confidence=0.80,
                direction="YES",
                clob_mid=0.60,
                trade_id=f"t{i}",
            )
        lr = tracker.get_likelihood_ratio(0.80, min_samples=30)
        assert lr is None

    def test_mixed_outcomes_across_bins(self) -> None:
        """LR should reflect relative win rate of bin vs base."""
        tracker = CalibrationTracker()
        # Bin 0.75-0.85: 30 wins out of 35
        for i in range(35):
            tid = f"high_{i}"
            tracker.record_prediction(
                confidence=0.80,
                direction="YES",
                clob_mid=0.60,
                trade_id=tid,
            )
            tracker.record_outcome(tid, won=(i < 30))

        # Bin 0.55-0.65: 15 wins out of 35
        for i in range(35):
            tid = f"low_{i}"
            tracker.record_prediction(
                confidence=0.60,
                direction="YES",
                clob_mid=0.55,
                trade_id=tid,
            )
            tracker.record_outcome(tid, won=(i < 15))

        # Overall base rate = (30+15) / (35+35) = 45/70 ~ 0.6429
        # High bin WR = 30/35 ~ 0.8571
        # LR for high bin = 0.8571 / 0.6429 ~ 1.333
        lr_high = tracker.get_likelihood_ratio(0.80, min_samples=30)
        assert lr_high is not None
        assert pytest.approx(lr_high, abs=0.01) == 1.333

        # Low bin WR = 15/35 ~ 0.4286
        # LR for low bin = 0.4286 / 0.6429 ~ 0.667
        lr_low = tracker.get_likelihood_ratio(0.60, min_samples=30)
        assert lr_low is not None
        assert pytest.approx(lr_low, abs=0.01) == 0.667

    def test_confidence_outside_bins(self) -> None:
        tracker = CalibrationTracker()
        lr = tracker.get_likelihood_ratio(0.20, min_samples=1)
        assert lr is None


class TestBrierScore:
    """Tests for Brier score computation."""

    def test_perfect_predictions(self) -> None:
        tracker = CalibrationTracker()
        # Confidence 1.0 → win, confidence 0.0 would be loss
        # Use confidence = 1.0 with win
        tracker.record_prediction(
            confidence=1.0,
            direction="YES",
            clob_mid=0.90,
            trade_id="t1",
        )
        tracker.record_outcome("t1", won=True)
        score = tracker.brier_score()
        assert score is not None
        assert pytest.approx(score, abs=1e-6) == 0.0

    def test_worst_predictions(self) -> None:
        tracker = CalibrationTracker()
        # Predict 1.0 but lose
        tracker.record_prediction(
            confidence=1.0,
            direction="YES",
            clob_mid=0.90,
            trade_id="t1",
        )
        tracker.record_outcome("t1", won=False)
        score = tracker.brier_score()
        assert score is not None
        assert pytest.approx(score, abs=1e-6) == 1.0

    def test_no_outcomes(self) -> None:
        tracker = CalibrationTracker()
        tracker.record_prediction(
            confidence=0.80,
            direction="YES",
            clob_mid=0.60,
            trade_id="t1",
        )
        assert tracker.brier_score() is None

    def test_empty_tracker(self) -> None:
        tracker = CalibrationTracker()
        assert tracker.brier_score() is None

    def test_mixed_outcomes(self) -> None:
        tracker = CalibrationTracker()
        # Predict 0.80, win → (0.80 - 1.0)^2 = 0.04
        tracker.record_prediction(
            confidence=0.80,
            direction="YES",
            clob_mid=0.60,
            trade_id="t1",
        )
        tracker.record_outcome("t1", won=True)

        # Predict 0.80, lose → (0.80 - 0.0)^2 = 0.64
        tracker.record_prediction(
            confidence=0.80,
            direction="YES",
            clob_mid=0.60,
            trade_id="t2",
        )
        tracker.record_outcome("t2", won=False)

        score = tracker.brier_score()
        assert score is not None
        # (0.04 + 0.64) / 2 = 0.34
        assert pytest.approx(score, abs=1e-6) == 0.34


class TestGetBinStats:
    """Tests for per-bin statistics."""

    def test_empty_tracker(self) -> None:
        tracker = CalibrationTracker()
        stats = tracker.get_bin_stats()
        assert len(stats) == 6
        for s in stats:
            assert s.predictions == 0
            assert s.wins == 0
            assert s.win_rate == 0.0

    def test_single_bin(self) -> None:
        tracker = CalibrationTracker()
        for i in range(10):
            tid = f"t{i}"
            tracker.record_prediction(
                confidence=0.80,
                direction="YES",
                clob_mid=0.60,
                trade_id=tid,
            )
            tracker.record_outcome(tid, won=(i < 7))

        stats = tracker.get_bin_stats()
        # Find the 0.75-0.85 bin
        bin_75 = [s for s in stats if s.bin_start == 0.75][0]
        assert bin_75.predictions == 10
        assert bin_75.wins == 7
        assert pytest.approx(bin_75.win_rate, abs=1e-4) == 0.70

    def test_unresolved_predictions_excluded(self) -> None:
        """Predictions without outcomes should not appear in bin stats."""
        tracker = CalibrationTracker()
        tracker.record_prediction(
            confidence=0.80,
            direction="YES",
            clob_mid=0.60,
            trade_id="t1",
        )
        # No outcome recorded
        stats = tracker.get_bin_stats()
        bin_75 = [s for s in stats if s.bin_start == 0.75][0]
        assert bin_75.predictions == 0

    def test_stats_are_binstats_instances(self) -> None:
        tracker = CalibrationTracker()
        stats = tracker.get_bin_stats()
        for s in stats:
            assert isinstance(s, BinStats)
