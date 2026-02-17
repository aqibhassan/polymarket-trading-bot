"""Tests for Regime Classifier."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.engine.regime_classifier import RegimeClassifier


def _make_candles(prices: list[float]) -> list[SimpleNamespace]:
    """Create mock candle objects with a .close attribute."""
    return [SimpleNamespace(close=p) for p in prices]


class TestRegimeClassifier:
    def test_insufficient_data(self) -> None:
        """Fewer candles than lookback → NEUTRAL."""
        rc = RegimeClassifier()
        candles = _make_candles([100.0] * 5)
        assert rc.classify(candles, lookback_short=20) == "neutral"

    def test_expansion_regime(self) -> None:
        """High short-term vol relative to long-term → EXPANSION."""
        rc = RegimeClassifier()
        # Long stable period
        stable = [100.0] * 80
        # Short volatile period: big swings
        volatile = []
        for i in range(20):
            volatile.append(100.0 + (5.0 if i % 2 == 0 else -5.0))
        candles = _make_candles(stable + volatile)
        result = rc.classify(candles, lookback_short=20, lookback_long=100)
        assert result == "expansion"

    def test_contraction_regime(self) -> None:
        """Low short-term vol relative to long-term → CONTRACTION."""
        rc = RegimeClassifier()
        # Long volatile period
        volatile = []
        for i in range(80):
            volatile.append(100.0 + (3.0 if i % 2 == 0 else -3.0))
        # Short stable period
        stable = [100.0 + 0.01 * i for i in range(20)]
        candles = _make_candles(volatile + stable)
        result = rc.classify(candles, lookback_short=20, lookback_long=100)
        assert result == "contraction"

    def test_neutral_regime(self) -> None:
        """Normal vol ratio → NEUTRAL."""
        rc = RegimeClassifier()
        # Uniform moderate volatility
        prices = [100.0 + (1.0 if i % 2 == 0 else -1.0) for i in range(100)]
        candles = _make_candles(prices)
        result = rc.classify(candles, lookback_short=20, lookback_long=100)
        assert result == "neutral"

    def test_realized_vol_single_candle(self) -> None:
        """Single candle → vol = 0."""
        rc = RegimeClassifier()
        candles = _make_candles([100.0])
        assert rc._realized_vol(candles) == 0.0

    def test_realized_vol_flat(self) -> None:
        """Flat prices → vol = 0."""
        rc = RegimeClassifier()
        candles = _make_candles([100.0] * 10)
        assert rc._realized_vol(candles) == 0.0
