"""Bayesian calibrator for MVHE signal confidence.

Uses the CLOB midpoint as a Bayesian prior and the signal confidence
to derive a likelihood ratio, producing a posterior probability that
is capped relative to the market price to prevent overconfidence.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.calibration.calibration_tracker import CalibrationTracker
from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class CalibrationResult:
    """Result of a Bayesian calibration pass."""

    prior: float
    likelihood_ratio: float
    posterior: float
    confidence: float
    clob_mid: float
    direction: str
    capped: bool


class RollingWinProbability:
    """Rolling estimate of win probability using a Beta distribution.

    Starts with an informative prior derived from paper trading results
    (alpha=83.8 wins, beta=16.2 losses ~ 83.8% win rate).  Each new
    trade outcome decays the existing evidence by ``decay`` and then
    increments the appropriate counter.
    """

    def __init__(
        self,
        alpha: float = 83.8,
        beta: float = 16.2,
        decay: float = 0.98,
    ) -> None:
        self._alpha = alpha
        self._beta = beta
        self._decay = decay

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def beta(self) -> float:
        return self._beta

    def update(self, won: bool) -> None:
        """Record a trade outcome, decaying old evidence first."""
        self._alpha *= self._decay
        self._beta *= self._decay
        if won:
            self._alpha += 1.0
        else:
            self._beta += 1.0

    def estimate(self) -> float:
        """Return current estimated win probability (Beta mean)."""
        total = self._alpha + self._beta
        if total == 0.0:
            return 0.5
        return self._alpha / total


# Desert CLOB detection constants
_DESERT_HALF_WIDTH: float = 0.05
_DESERT_CENTER: float = 0.50
_DESERT_MAX_EDGE: float = 0.25


class BayesianCalibrator:
    """Calibrate signal confidence against CLOB market pricing.

    The CLOB midpoint serves as the Bayesian prior (the market's best
    probability estimate).  The signal confidence is converted into a
    likelihood ratio --- either from tracked historical data or from
    a fixed default --- and applied via Bayes' rule.  The resulting
    posterior is capped so it never exceeds ``clob_mid + max_edge``
    to guard against overconfidence.

    When the CLOB mid is in the "desert" zone (~0.50), a wider max_edge
    is used because the market is uninformative (placeholder 0.01/0.99
    prices compress to ~0.50).
    """

    def __init__(
        self,
        max_edge: float = 0.15,
        default_likelihood_ratio: float = 1.3,
        min_samples: int = 30,
    ) -> None:
        self._max_edge = max_edge
        self._default_likelihood_ratio = default_likelihood_ratio
        self._min_samples = min_samples

    @property
    def max_edge(self) -> float:
        return self._max_edge

    @property
    def default_likelihood_ratio(self) -> float:
        return self._default_likelihood_ratio

    @property
    def min_samples(self) -> int:
        return self._min_samples

    def _is_desert(self, clob_mid: float) -> bool:
        """Return True if CLOB mid is in the uninformative desert zone."""
        return abs(clob_mid - _DESERT_CENTER) <= _DESERT_HALF_WIDTH

    def calibrate(
        self,
        clob_mid: float,
        signal_confidence: float,
        direction: str,
        minute: int,
        tracker: CalibrationTracker | None = None,
    ) -> CalibrationResult:
        """Produce a calibrated posterior from CLOB prior and signal.

        Args:
            clob_mid: Market midpoint price (0-1), used as Bayesian prior.
            signal_confidence: Raw signal confidence (0-1).
            direction: Trade direction ("YES" or "NO").
            minute: Current minute within the 15-minute window.
            tracker: Optional calibration tracker for data-driven
                likelihood ratios.

        Returns:
            CalibrationResult with the prior, likelihood ratio, posterior,
            and whether capping was applied.
        """
        # --- Prior ---
        prior = clob_mid

        # --- Likelihood ratio ---
        likelihood_ratio: float | None = None
        if tracker is not None:
            likelihood_ratio = tracker.get_likelihood_ratio(
                signal_confidence,
                min_samples=self._min_samples,
            )
        if likelihood_ratio is None:
            likelihood_ratio = self._default_likelihood_ratio

        # --- Posterior via Bayes' rule ---
        # P(win|signal) = P(signal|win) * P(win) / P(signal)
        # where normalizer P(signal) = lr * prior + (1 - prior)
        numerator = likelihood_ratio * prior
        normalizer = numerator + (1.0 - prior)
        if normalizer == 0.0:
            posterior = prior
        else:
            posterior = numerator / normalizer

        # --- Capping ---
        effective_max_edge = (
            _DESERT_MAX_EDGE if self._is_desert(clob_mid) else self._max_edge
        )
        cap = clob_mid + effective_max_edge
        capped = posterior > cap
        if capped:
            posterior = cap

        # Clamp to valid probability range
        posterior = max(0.01, min(0.99, posterior))

        logger.debug(
            "calibration_complete",
            prior=round(prior, 4),
            likelihood_ratio=round(likelihood_ratio, 4),
            posterior=round(posterior, 4),
            capped=capped,
            desert=self._is_desert(clob_mid),
            direction=direction,
            minute=minute,
        )

        return CalibrationResult(
            prior=prior,
            likelihood_ratio=likelihood_ratio,
            posterior=posterior,
            confidence=signal_confidence,
            clob_mid=clob_mid,
            direction=direction,
            capped=capped,
        )
