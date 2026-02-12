"""Calibration tracker for predicted-vs-actual analysis.

Records every prediction and its eventual outcome, bins predictions
by confidence level, and computes per-bin statistics and likelihood
ratios for use by the Bayesian calibrator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from src.core.logging import get_logger

logger = get_logger(__name__)


# Confidence bins: (lower_inclusive, upper_exclusive) except last bin is inclusive
_BINS: list[tuple[float, float]] = [
    (0.45, 0.55),
    (0.55, 0.65),
    (0.65, 0.75),
    (0.75, 0.85),
    (0.85, 0.95),
    (0.95, 1.00),
]


class _PredictionRecord(TypedDict):
    trade_id: str
    confidence: float
    direction: str
    clob_mid: float
    outcome: bool | None


@dataclass(frozen=True)
class BinStats:
    """Statistics for a single confidence bin."""

    bin_start: float
    bin_end: float
    predictions: int
    wins: int
    win_rate: float


class CalibrationTracker:
    """Track predicted confidence vs actual outcomes.

    Stores every prediction with its confidence level and eventual
    win/loss outcome.  Provides per-bin statistics, likelihood ratios,
    and Brier score computation.  Data is held in memory; persistence
    to Redis/ClickHouse can be added later.
    """

    def __init__(self) -> None:
        self._predictions: list[_PredictionRecord] = []
        self._id_index: dict[str, int] = {}

    @property
    def predictions(self) -> list[_PredictionRecord]:
        """Return a copy of all prediction records."""
        return list(self._predictions)

    def record_prediction(
        self,
        confidence: float,
        direction: str,
        clob_mid: float,
        trade_id: str,
    ) -> None:
        """Record a new prediction before the outcome is known.

        Args:
            confidence: Signal confidence (0-1).
            direction: Trade direction ("YES" or "NO").
            clob_mid: CLOB midpoint at time of prediction.
            trade_id: Unique identifier for this trade.
        """
        if trade_id in self._id_index:
            logger.warning(
                "duplicate_prediction",
                trade_id=trade_id,
            )
            return

        record: _PredictionRecord = {
            "trade_id": trade_id,
            "confidence": confidence,
            "direction": direction,
            "clob_mid": clob_mid,
            "outcome": None,
        }
        self._id_index[trade_id] = len(self._predictions)
        self._predictions.append(record)

        logger.debug(
            "prediction_recorded",
            trade_id=trade_id,
            confidence=round(confidence, 4),
            direction=direction,
            clob_mid=round(clob_mid, 4),
        )

    def record_outcome(self, trade_id: str, won: bool) -> None:
        """Record the actual outcome for a previously recorded prediction.

        Args:
            trade_id: The trade identifier from record_prediction.
            won: True if the trade was a win, False otherwise.
        """
        idx = self._id_index.get(trade_id)
        if idx is None:
            logger.warning(
                "outcome_for_unknown_prediction",
                trade_id=trade_id,
            )
            return

        self._predictions[idx]["outcome"] = won

        logger.debug(
            "outcome_recorded",
            trade_id=trade_id,
            won=won,
        )

    @staticmethod
    def _find_bin(confidence: float) -> tuple[float, float] | None:
        """Return the bin for a given confidence value, or None."""
        for bin_start, bin_end in _BINS:
            # Last bin is inclusive on both ends
            if bin_start == _BINS[-1][0]:
                if bin_start <= confidence <= bin_end:
                    return (bin_start, bin_end)
            else:
                if bin_start <= confidence < bin_end:
                    return (bin_start, bin_end)
        return None

    def get_likelihood_ratio(
        self,
        confidence: float,
        min_samples: int = 30,
    ) -> float | None:
        """Compute likelihood ratio for the confidence bin.

        The likelihood ratio is the observed win rate in this bin
        divided by the overall base rate (across all bins with
        outcomes).  Returns None if insufficient samples.

        Args:
            confidence: Signal confidence to look up.
            min_samples: Minimum completed predictions in the bin.

        Returns:
            Likelihood ratio or None if insufficient data.
        """
        target_bin = self._find_bin(confidence)
        if target_bin is None:
            return None

        # Gather outcomes for the target bin
        bin_outcomes: list[bool] = []
        all_outcomes: list[bool] = []

        for record in self._predictions:
            if record["outcome"] is None:
                continue
            all_outcomes.append(record["outcome"])
            rec_bin = self._find_bin(record["confidence"])
            if rec_bin == target_bin:
                bin_outcomes.append(record["outcome"])

        if len(bin_outcomes) < min_samples or len(all_outcomes) == 0:
            return None

        bin_win_rate = sum(bin_outcomes) / len(bin_outcomes)
        base_rate = sum(all_outcomes) / len(all_outcomes)

        if base_rate == 0.0:
            return None

        ratio = bin_win_rate / base_rate

        logger.debug(
            "likelihood_ratio_computed",
            bin=target_bin,
            bin_samples=len(bin_outcomes),
            bin_win_rate=round(bin_win_rate, 4),
            base_rate=round(base_rate, 4),
            ratio=round(ratio, 4),
        )

        return ratio

    def brier_score(self) -> float | None:
        """Compute Brier score over all predictions with outcomes.

        The Brier score measures the mean squared error between
        predicted probabilities and actual binary outcomes.
        Lower is better; 0.0 is perfect, 0.25 is random.

        Returns:
            Brier score or None if no outcomes recorded.
        """
        completed = [
            r for r in self._predictions if r["outcome"] is not None
        ]
        if not completed:
            return None

        total = 0.0
        for record in completed:
            predicted = record["confidence"]
            actual = 1.0 if record["outcome"] else 0.0
            total += (predicted - actual) ** 2

        return total / len(completed)

    def get_bin_stats(self) -> list[BinStats]:
        """Return per-bin statistics for all confidence bins.

        Returns:
            List of BinStats, one per bin, ordered by bin start.
        """
        stats: list[BinStats] = []

        for bin_start, bin_end in _BINS:
            bin_records = [
                r
                for r in self._predictions
                if r["outcome"] is not None
                and self._find_bin(r["confidence"]) == (bin_start, bin_end)
            ]
            predictions = len(bin_records)
            wins = sum(1 for r in bin_records if r["outcome"])
            win_rate = wins / predictions if predictions > 0 else 0.0

            stats.append(
                BinStats(
                    bin_start=bin_start,
                    bin_end=bin_end,
                    predictions=predictions,
                    wins=wins,
                    win_rate=win_rate,
                )
            )

        return stats
