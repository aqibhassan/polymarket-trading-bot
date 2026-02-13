"""Edge calculator with Polymarket fee model.

Computes fee-adjusted edge between a calibrated posterior probability
and the entry price, gating tradeability on a minimum edge threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class EdgeResult:
    """Result of an edge calculation."""

    posterior: float
    entry_price: float
    clob_mid: float
    raw_edge: float
    fee: float
    fee_adjusted_edge: float
    is_tradeable: bool
    min_edge: float


class EdgeCalculator:
    """Calculate fee-adjusted edge and tradeability.

    The Polymarket fee formula is:
        fee = fee_constant * price^2 * (1 - price)^2

    This produces a bell-shaped fee curve peaking at price=0.50 and
    falling to zero at the extremes (0.0 and 1.0).

    A trade is considered tradeable when the fee-adjusted edge
    (posterior - entry_price - fee) meets or exceeds ``min_edge``.
    """

    def __init__(
        self,
        min_edge: float = 0.03,
        fee_constant: float = 0.25,
        dynamic_min_edge_enabled: bool = False,
        uncertainty_penalty_scale: float = 0.03,
    ) -> None:
        self._min_edge = min_edge
        self._fee_constant = fee_constant
        self._dynamic_min_edge_enabled = dynamic_min_edge_enabled
        self._uncertainty_penalty_scale = uncertainty_penalty_scale

    @property
    def min_edge(self) -> float:
        return self._min_edge

    @property
    def fee_constant(self) -> float:
        return self._fee_constant

    def _effective_min_edge(self, clob_mid: float) -> float:
        """Compute effective min edge, optionally scaled by market uncertainty.

        At clob_mid=0.50 (max uncertainty), penalty is full scale.
        At extremes (0.20 or 0.80), penalty is much smaller.

        Formula: base + scale * max(0, 1 - 2*|clob_mid - 0.50|)
        """
        if not self._dynamic_min_edge_enabled:
            return self._min_edge
        uncertainty = max(0.0, 1.0 - 2.0 * abs(clob_mid - 0.50))
        return self._min_edge + self._uncertainty_penalty_scale * uncertainty

    def polymarket_fee(self, price: float) -> float:
        """Compute the Polymarket fee for a given price.

        Args:
            price: Entry price (0-1).

        Returns:
            Fee amount.
        """
        return self._fee_constant * (price ** 2) * ((1.0 - price) ** 2)

    def calculate(
        self,
        posterior: float,
        entry_price: float,
        clob_mid: float,
    ) -> EdgeResult:
        """Calculate fee-adjusted edge and tradeability.

        Args:
            posterior: Calibrated posterior probability (0-1).
            entry_price: Actual entry price on CLOB (0-1).
            clob_mid: CLOB midpoint for reference.

        Returns:
            EdgeResult with raw edge, fee, adjusted edge, and
            tradeability flag.
        """
        raw_edge = posterior - entry_price
        fee = self.polymarket_fee(entry_price)
        # Fee is a rate on notional (shares * price), so per-share fee cost
        # is fee_rate * entry_price.  Deduct that from the per-share edge.
        fee_adjusted_edge = raw_edge - fee * entry_price

        # Dynamic min edge: scale with market uncertainty (distance from 0.50)
        effective_min_edge = self._effective_min_edge(clob_mid)
        is_tradeable = fee_adjusted_edge >= effective_min_edge

        logger.debug(
            "edge_calculated",
            posterior=round(posterior, 4),
            entry_price=round(entry_price, 4),
            clob_mid=round(clob_mid, 4),
            raw_edge=round(raw_edge, 4),
            fee=round(fee, 6),
            fee_adjusted_edge=round(fee_adjusted_edge, 4),
            min_edge=round(effective_min_edge, 4),
            is_tradeable=is_tradeable,
        )

        return EdgeResult(
            posterior=posterior,
            entry_price=entry_price,
            clob_mid=clob_mid,
            raw_edge=raw_edge,
            fee=fee,
            fee_adjusted_edge=fee_adjusted_edge,
            is_tradeable=is_tradeable,
            min_edge=effective_min_edge,
        )
