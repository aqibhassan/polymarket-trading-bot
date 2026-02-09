"""Statistical validation of backtest results."""

from __future__ import annotations

import math
import random
from decimal import Decimal
from typing import Any

from pydantic import BaseModel

from src.core.logging import get_logger

log = get_logger(__name__)

_ZERO = Decimal("0")


class ValidationResult(BaseModel):
    """Result of statistical validation."""

    sharpe_ci_95: tuple[float, float]
    p_value: float
    is_significant: bool
    overfitting_warning: str | None = None

    model_config = {"frozen": True}


class StatisticalValidator:
    """Validate that backtest results are statistically significant."""

    def __init__(
        self,
        n_bootstrap: int = 1000,
        n_permutations: int = 1000,
        significance_level: float = 0.05,
        rng_seed: int | None = None,
    ) -> None:
        self._n_bootstrap = n_bootstrap
        self._n_permutations = n_permutations
        self._significance_level = significance_level
        self._rng = random.Random(rng_seed)

    def validate(
        self,
        trades: list[dict[str, Any]],
        equity_curve: list[Decimal],
    ) -> ValidationResult:
        """Run all validation checks.

        Args:
            trades: List of trade dicts with 'pnl' key.
            equity_curve: Portfolio equity over time.

        Returns:
            ValidationResult with CI, p-value, and warnings.
        """
        returns = self._equity_to_returns(equity_curve)
        sharpe_ci = self._bootstrap_sharpe_ci(returns)
        p_value = self._permutation_test(trades)
        is_significant = p_value < self._significance_level

        # Overfitting check: split equity curve 70/30 and compare Sharpe
        overfitting_warning: str | None = None
        if len(equity_curve) >= 20:
            split_idx = int(len(equity_curve) * 0.7)
            is_returns = self._equity_to_returns(equity_curve[:split_idx])
            oos_returns = self._equity_to_returns(equity_curve[split_idx:])
            if len(is_returns) >= 2 and len(oos_returns) >= 2:
                is_sharpe = self._sharpe(is_returns)
                oos_sharpe = self._sharpe(oos_returns)
                overfitting_warning = self.check_overfitting(is_sharpe, oos_sharpe)

        log.info(
            "validation_complete",
            sharpe_ci_low=f"{sharpe_ci[0]:.4f}",
            sharpe_ci_high=f"{sharpe_ci[1]:.4f}",
            p_value=f"{p_value:.4f}",
            significant=is_significant,
            overfitting=overfitting_warning or "none",
        )

        return ValidationResult(
            sharpe_ci_95=sharpe_ci,
            p_value=p_value,
            is_significant=is_significant,
            overfitting_warning=overfitting_warning,
        )

    def check_overfitting(
        self,
        in_sample_sharpe: float,
        out_of_sample_sharpe: float,
    ) -> str | None:
        """Check for overfitting by comparing IS vs OOS Sharpe ratios.

        Args:
            in_sample_sharpe: Sharpe ratio on training data.
            out_of_sample_sharpe: Sharpe ratio on holdout data.

        Returns:
            Warning string if overfitting detected, else None.
        """
        if out_of_sample_sharpe <= 0:
            warning = (
                f"Out-of-sample Sharpe ({out_of_sample_sharpe:.2f}) is non-positive. "
                f"In-sample Sharpe was {in_sample_sharpe:.2f}. Likely overfitting."
            )
            log.warning("overfitting_detected", warning=warning)
            return warning

        ratio = in_sample_sharpe / out_of_sample_sharpe
        if ratio > 2.0:
            warning = (
                f"IS/OOS Sharpe ratio is {ratio:.2f}x "
                f"(IS={in_sample_sharpe:.2f}, OOS={out_of_sample_sharpe:.2f}). "
                f"Possible overfitting."
            )
            log.warning("overfitting_detected", warning=warning)
            return warning

        return None

    def _bootstrap_sharpe_ci(
        self,
        returns: list[float],
    ) -> tuple[float, float]:
        """Bootstrap 95% confidence interval on Sharpe ratio.

        Args:
            returns: Period returns.

        Returns:
            (lower, upper) bounds of 95% CI.
        """
        if len(returns) < 2:
            return (0.0, 0.0)

        n = len(returns)
        bootstrapped_sharpes: list[float] = []

        for _ in range(self._n_bootstrap):
            sample = [self._rng.choice(returns) for _ in range(n)]
            sharpe = self._sharpe(sample)
            bootstrapped_sharpes.append(sharpe)

        bootstrapped_sharpes.sort()
        lower_idx = int(self._n_bootstrap * 0.025)
        upper_idx = int(self._n_bootstrap * 0.975)
        return (bootstrapped_sharpes[lower_idx], bootstrapped_sharpes[upper_idx])

    def _permutation_test(
        self,
        trades: list[dict[str, Any]],
    ) -> float:
        """Monte Carlo permutation test on trade PnLs.

        Shuffles trade PnLs and computes how often the shuffled total
        exceeds the observed total. Returns a p-value.

        Args:
            trades: List of trade dicts with 'pnl' key.

        Returns:
            p-value (proportion of permutations >= observed).
        """
        if not trades:
            return 1.0

        pnls = [float(t["pnl"]) for t in trades]
        observed_total = sum(pnls)
        count_gte = 0

        for _ in range(self._n_permutations):
            shuffled = list(pnls)
            # Randomly flip signs to create null distribution
            shuffled = [p * self._rng.choice([1, -1]) for p in shuffled]
            if sum(shuffled) >= observed_total:
                count_gte += 1

        return count_gte / self._n_permutations

    @staticmethod
    def _equity_to_returns(equity_curve: list[Decimal]) -> list[float]:
        """Convert equity curve to period returns."""
        if len(equity_curve) < 2:
            return []
        returns: list[float] = []
        for i in range(1, len(equity_curve)):
            prev = equity_curve[i - 1]
            if prev == _ZERO:
                returns.append(0.0)
            else:
                returns.append(float((equity_curve[i] - prev) / prev))
        return returns

    @staticmethod
    def _sharpe(returns: list[float]) -> float:
        """Compute annualized Sharpe ratio from returns."""
        if len(returns) < 2:
            return 0.0
        mean_r = sum(returns) / len(returns)
        variance = sum((r - mean_r) ** 2 for r in returns) / len(returns)
        std_r = math.sqrt(variance)
        if std_r == 0.0:
            return 0.0
        return mean_r / std_r * math.sqrt(252)
