"""Volatility regime detector — realized vs implied vol analysis."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pydantic import BaseModel

from src.core.logging import get_logger

if TYPE_CHECKING:
    from decimal import Decimal

    from src.config.loader import ConfigLoader

log = get_logger(__name__)

# Ticks in a 15-minute window at ~1 tick/second
_TICKS_PER_15MIN = 900


class VolRegimeResult(BaseModel):
    """Result of volatility regime detection."""

    realized_vol_pct: float
    implied_vol_pct: float
    vol_ratio: float
    regime: str  # "high_vol" | "low_vol" | "normal"
    signal_direction: str  # "long_vol" | "short_vol" | "neutral"
    model_config = {"frozen": True}


def _inv_norm_cdf(p: float) -> float:
    """Rational approximation of inverse normal CDF (Beasley-Springer-Moro).

    Args:
        p: Probability value in (0, 1).

    Returns:
        Approximate inverse normal CDF value, or 0.0 for edge cases.
    """
    if p <= 0.0 or p >= 1.0:
        return 0.0

    a = [
        0,
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        0,
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        0,
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        0,
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    p_low = 0.02425
    p_high = 1 - p_low

    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (
            ((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]
        ) / (((( d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[1] * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * r + a[6]) * q
            / (((((b[1] * r + b[2]) * r + b[3]) * r + b[4]) * r + b[5]) * r + 1)
        )
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(
            ((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]
        ) / ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1)


class VolatilityRegimeDetector:
    """Detect volatility clustering and regime changes.

    Compares realized volatility from tick data against implied
    volatility extracted from Polymarket binary option prices to
    identify vol regime (high/low/normal) and generate directional
    signals (long_vol / short_vol / neutral).
    """

    def __init__(self, config: ConfigLoader) -> None:
        self._spike_threshold = float(
            config.get("strategy.singularity.vol_spike_threshold", 1.25)
        )
        self._low_threshold = float(
            config.get("strategy.singularity.vol_low_threshold", 0.80)
        )

    def compute_realized_vol(self, recent_ticks: list[Decimal]) -> float:
        """Compute realized volatility from tick prices.

        Uses log-return standard deviation, scaled to a 15-minute window.

        Args:
            recent_ticks: List of tick prices (at least 2 required).

        Returns:
            Realized volatility in percent per 15-min window, or 0.0
            if insufficient data.
        """
        if len(recent_ticks) < 2:
            return 0.0

        log_returns: list[float] = []
        for i in range(1, len(recent_ticks)):
            prev = float(recent_ticks[i - 1])
            curr = float(recent_ticks[i])
            if prev <= 0:
                continue
            log_returns.append(math.log(curr / prev))

        if len(log_returns) < 1:
            return 0.0

        mean = sum(log_returns) / len(log_returns)
        variance = sum((r - mean) ** 2 for r in log_returns) / len(log_returns)
        std = math.sqrt(variance)

        # Scale to 15-min window: std * sqrt(ticks_per_15min) * 100
        n_ticks = len(recent_ticks)
        scaling = math.sqrt(_TICKS_PER_15MIN / max(n_ticks, 1))
        realized = std * scaling * 100.0

        log.debug(
            "realized_vol_computed",
            std=std,
            n_returns=len(log_returns),
            realized_vol_pct=realized,
        )
        return realized

    def extract_implied_vol(
        self, yes_price: Decimal, time_remaining_seconds: int
    ) -> float:
        """Extract implied vol from Polymarket binary option price.

        Uses inverse normal CDF approximation.
        For binary: YES_price ~ Phi(d), where d ~ mu / (sigma * sqrt(T))
        So sigma ~ |Phi^-1(P)| / sqrt(T / 900)

        Args:
            yes_price: Current YES token price (0-1 range).
            time_remaining_seconds: Seconds until market resolution.

        Returns:
            Implied volatility in percent, or 0.0 for edge cases.
        """
        p = float(yes_price)
        if p <= 0.0 or p >= 1.0:
            return 0.0
        if time_remaining_seconds <= 0:
            return 0.0

        d = _inv_norm_cdf(p)
        t_fraction = time_remaining_seconds / _TICKS_PER_15MIN

        if t_fraction <= 0:
            return 0.0

        implied = abs(d) / math.sqrt(t_fraction) * 100.0

        log.debug(
            "implied_vol_extracted",
            yes_price=p,
            time_remaining=time_remaining_seconds,
            d=d,
            implied_vol_pct=implied,
        )
        return implied

    def detect_regime(
        self,
        recent_ticks: list[Decimal],
        yes_price: Decimal,
        time_remaining_seconds: int,
    ) -> VolRegimeResult:
        """Full regime detection combining realized and implied vol.

        Args:
            recent_ticks: Recent tick prices for realized vol computation.
            yes_price: Current YES token price.
            time_remaining_seconds: Seconds until market resolution.

        Returns:
            VolRegimeResult with regime classification and signal.
        """
        realized = self.compute_realized_vol(recent_ticks)
        implied = self.extract_implied_vol(yes_price, time_remaining_seconds)

        if implied <= 0:
            vol_ratio = 0.0
        else:
            vol_ratio = realized / implied

        if vol_ratio > self._spike_threshold:
            regime = "high_vol"
            signal = "long_vol"
        elif vol_ratio < self._low_threshold and vol_ratio > 0:
            regime = "low_vol"
            signal = "short_vol"
        else:
            regime = "normal"
            signal = "neutral"

        log.debug(
            "vol_regime_detected",
            realized_vol=realized,
            implied_vol=implied,
            vol_ratio=vol_ratio,
            regime=regime,
            signal=signal,
        )

        return VolRegimeResult(
            realized_vol_pct=realized,
            implied_vol_pct=implied,
            vol_ratio=vol_ratio,
            regime=regime,
            signal_direction=signal,
        )
