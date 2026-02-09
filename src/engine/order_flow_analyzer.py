"""Order flow imbalance analyzer — OFI computation from order book snapshots."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from src.core.logging import get_logger

if TYPE_CHECKING:
    from src.config.loader import ConfigLoader
    from src.models.market import OrderBookSnapshot

log = get_logger(__name__)


class OFIResult(BaseModel):
    """Order flow imbalance analysis result."""

    ofi_current: float
    ofi_trend: float
    signal_strength: float
    direction: str
    confidence: float

    model_config = {"frozen": True}


class OrderFlowAnalyzer:
    """Analyzes order flow imbalance from order book snapshots."""

    def __init__(self, config: ConfigLoader) -> None:
        self._levels = int(
            config.get("strategy.singularity.ofi_levels", 5)
        )
        self._signal_threshold = float(
            config.get("strategy.singularity.ofi_signal_threshold", 0.25)
        )
        self._saturation = float(
            config.get("strategy.singularity.ofi_saturation", 0.40)
        )

    def _compute_ofi(self, orderbook: OrderBookSnapshot) -> float:
        """Compute raw OFI from top N levels of the order book.

        OFI = (bid_vol - ask_vol) / total_vol, clamped to [-1, 1].
        Returns 0.0 when total volume is zero.
        """
        bid_vol = sum(
            float(lvl.size) for lvl in orderbook.bids[: self._levels]
        )
        ask_vol = sum(
            float(lvl.size) for lvl in orderbook.asks[: self._levels]
        )
        total = bid_vol + ask_vol
        if total == 0.0:
            return 0.0
        return (bid_vol - ask_vol) / total

    def analyze(
        self,
        orderbook: OrderBookSnapshot,
        history: list[OrderBookSnapshot] | None = None,
    ) -> OFIResult:
        """Compute OFI from current order book and optional history for trend.

        Args:
            orderbook: Current order book snapshot.
            history: Previous snapshots (oldest first) for trend calculation.

        Returns:
            OFIResult with current OFI, trend, signal strength, direction,
            and confidence.
        """
        ofi = self._compute_ofi(orderbook)
        abs_ofi = abs(ofi)

        # Signal strength: scales linearly from 0 at |OFI|=0 to 1.0 at saturation
        signal_strength = min(abs_ofi / self._saturation, 1.0) if self._saturation > 0 else 0.0

        # Direction classification
        if ofi > self._signal_threshold:
            direction = "buy_pressure"
        elif ofi < -self._signal_threshold:
            direction = "sell_pressure"
        else:
            direction = "neutral"

        # OFI trend: rate of change across history
        ofi_trend = 0.0
        if history and len(history) >= 2:
            historical_ofis = [self._compute_ofi(snap) for snap in history]
            historical_ofis.append(ofi)
            # Average change per step
            deltas = [
                historical_ofis[i] - historical_ofis[i - 1]
                for i in range(1, len(historical_ofis))
            ]
            ofi_trend = sum(deltas) / len(deltas)

        # Confidence: high when signal is strong and trend agrees with direction
        # Reduced when OFI is reversing (trend opposes current direction)
        if direction == "neutral":
            confidence = 1.0 - signal_strength
        else:
            trend_agrees = (ofi > 0 and ofi_trend >= 0) or (ofi < 0 and ofi_trend <= 0)
            confidence = 1.0 if trend_agrees else max(0.3, 1.0 - abs(ofi_trend) * 2)

        log.debug(
            "ofi_analyzed",
            ofi=ofi,
            trend=ofi_trend,
            signal_strength=signal_strength,
            direction=direction,
            confidence=confidence,
        )

        return OFIResult(
            ofi_current=ofi,
            ofi_trend=ofi_trend,
            signal_strength=signal_strength,
            direction=direction,
            confidence=confidence,
        )
