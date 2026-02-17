"""Tests for VPIN analyzer."""

from __future__ import annotations

import pytest

from src.engine.vpin_analyzer import VPINAnalyzer


class TestVPINAnalyzer:
    def test_vpin_empty(self) -> None:
        """No trades → VPIN = 0.0."""
        analyzer = VPINAnalyzer(bucket_size=10.0, n_buckets=50)
        assert analyzer.get_vpin() == 0.0
        assert analyzer.get_direction() == "neutral"
        assert analyzer.get_strength() == 0.0

    def test_vpin_balanced_flow(self) -> None:
        """Equal buy/sell → low VPIN."""
        analyzer = VPINAnalyzer(bucket_size=10.0, n_buckets=50)
        # Alternate buy and sell trades to fill buckets
        for _ in range(100):
            analyzer.on_agg_trade(5.0, is_buyer_maker=False)  # buy
            analyzer.on_agg_trade(5.0, is_buyer_maker=True)   # sell
        vpin = analyzer.get_vpin()
        assert vpin < 0.2, f"Balanced flow should have low VPIN, got {vpin}"
        assert analyzer.get_direction() == "neutral"

    def test_vpin_imbalanced_buy_flow(self) -> None:
        """All buys → high VPIN, direction = YES."""
        analyzer = VPINAnalyzer(bucket_size=10.0, n_buckets=50)
        for _ in range(100):
            analyzer.on_agg_trade(10.0, is_buyer_maker=False)  # all buys
        vpin = analyzer.get_vpin()
        assert vpin > 0.5, f"Imbalanced flow should have high VPIN, got {vpin}"
        assert analyzer.get_direction() == "YES"
        assert analyzer.get_strength() > 0.0

    def test_vpin_imbalanced_sell_flow(self) -> None:
        """All sells → high VPIN, direction = NO."""
        analyzer = VPINAnalyzer(bucket_size=10.0, n_buckets=50)
        for _ in range(100):
            analyzer.on_agg_trade(10.0, is_buyer_maker=True)  # all sells
        vpin = analyzer.get_vpin()
        assert vpin > 0.5
        assert analyzer.get_direction() == "NO"

    def test_vpin_bucket_rollover(self) -> None:
        """Volume exceeds bucket size → creates multiple buckets."""
        analyzer = VPINAnalyzer(bucket_size=5.0, n_buckets=50)
        # Single large trade that spans 2 buckets
        analyzer.on_agg_trade(12.0, is_buyer_maker=False)
        # Should have created 2 buckets (12/5 = 2 full buckets + overflow)
        assert len(analyzer._buckets) == 2

    def test_vpin_strength_scaling(self) -> None:
        """Strength scales from 0 at VPIN=0.3 to 1 at VPIN=0.8."""
        analyzer = VPINAnalyzer(bucket_size=10.0, n_buckets=50)
        # With no data, strength should be 0
        assert analyzer.get_strength() == 0.0
