"""VPIN (Volume-Synchronized Probability of Informed Trading) analyzer.

Partitions Binance aggTrade volume into equal-size buckets, classifies
buy/sell flow, and computes order imbalance to predict price jumps.
Based on Easley, Lopez de Prado, O'Hara research.
"""

from __future__ import annotations

from collections import deque


class VPINAnalyzer:
    """Volume-Synchronized Probability of Informed Trading."""

    def __init__(self, bucket_size: float = 10.0, n_buckets: int = 50) -> None:
        self._bucket_size = bucket_size
        self._n_buckets = n_buckets
        self._buckets: deque[tuple[float, float]] = deque(maxlen=n_buckets)
        self._current_buy_vol = 0.0
        self._current_sell_vol = 0.0
        self._current_total_vol = 0.0

    def on_agg_trade(self, qty: float, is_buyer_maker: bool) -> None:
        """Process a single aggTrade tick."""
        if is_buyer_maker:
            self._current_sell_vol += qty
        else:
            self._current_buy_vol += qty
        self._current_total_vol += qty

        while self._current_total_vol >= self._bucket_size:
            total = self._current_buy_vol + self._current_sell_vol
            buy_ratio = self._current_buy_vol / total if total > 0 else 0.5
            imbalance = abs(self._current_buy_vol - self._current_sell_vol) / self._bucket_size
            self._buckets.append((imbalance, buy_ratio))
            overflow = self._current_total_vol - self._bucket_size
            ratio = overflow / self._current_total_vol if self._current_total_vol > 0 else 0
            self._current_buy_vol *= ratio
            self._current_sell_vol *= ratio
            self._current_total_vol = overflow

    def get_vpin(self) -> float:
        """Get current VPIN (0-1). Higher = more informed trading."""
        if not self._buckets:
            return 0.0
        return sum(b[0] for b in self._buckets) / len(self._buckets)

    def get_direction(self) -> str:
        """Get informed flow direction from recent buckets."""
        if len(self._buckets) < 5:
            return "neutral"
        recent = list(self._buckets)[-10:]
        avg_buy_ratio = sum(b[1] for b in recent) / len(recent)
        if avg_buy_ratio > 0.55:
            return "YES"
        elif avg_buy_ratio < 0.45:
            return "NO"
        return "neutral"

    def get_strength(self) -> float:
        """Signal strength from VPIN level (0-1)."""
        vpin = self.get_vpin()
        return min(max((vpin - 0.3) / 0.5, 0.0), 1.0)
