"""Tests for TrendAnalyzer."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.engine.trend_analyzer import TrendAnalyzer
from src.models.market import Candle
from src.models.signal import TrendDirection


class TestTrendAnalyzer:
    """Tests for TrendAnalyzer.analyze()."""

    def test_all_green_candles_uptrend(
        self, config_loader, green_candles
    ) -> None:
        analyzer = TrendAnalyzer(config_loader)
        result = analyzer.analyze(green_candles)
        assert result.direction == TrendDirection.UP
        assert result.green_count == 5
        assert result.red_count == 0
        assert result.strength == 1.0

    def test_all_red_candles_downtrend(
        self, config_loader, red_candles
    ) -> None:
        analyzer = TrendAnalyzer(config_loader)
        result = analyzer.analyze(red_candles)
        assert result.direction == TrendDirection.DOWN
        assert result.red_count == 5
        assert result.green_count == 0
        assert result.strength == 1.0

    def test_mixed_candles_neutral(self, config_loader) -> None:
        """2 green + 2 red + 1 neutral = NEUTRAL."""
        candles = []
        # 2 green
        for i in range(2):
            candles.append(
                Candle(
                    exchange="binance",
                    symbol="BTCUSDT",
                    open=Decimal("100"),
                    high=Decimal("110"),
                    low=Decimal("99"),
                    close=Decimal("105"),
                    volume=Decimal("10"),
                    timestamp=datetime(2024, 1, 1, 12, i * 15, 0),
                )
            )
        # 2 red
        for i in range(2, 4):
            candles.append(
                Candle(
                    exchange="binance",
                    symbol="BTCUSDT",
                    open=Decimal("105"),
                    high=Decimal("106"),
                    low=Decimal("99"),
                    close=Decimal("100"),
                    volume=Decimal("10"),
                    timestamp=datetime(2024, 1, 1, 12, i * 15, 0),
                )
            )
        # 1 neutral
        candles.append(
            Candle(
                exchange="binance",
                symbol="BTCUSDT",
                open=Decimal("100"),
                high=Decimal("101"),
                low=Decimal("99"),
                close=Decimal("100"),
                volume=Decimal("10"),
                timestamp=datetime(2024, 1, 1, 13, 0, 0),
            )
        )
        analyzer = TrendAnalyzer(config_loader)
        result = analyzer.analyze(candles)
        assert result.direction == TrendDirection.NEUTRAL
        assert result.green_count == 2
        assert result.red_count == 2

    def test_empty_candles_neutral(self, config_loader) -> None:
        analyzer = TrendAnalyzer(config_loader)
        result = analyzer.analyze([])
        assert result.direction == TrendDirection.NEUTRAL
        assert result.strength == 0.0

    def test_single_candle(self, config_loader, sample_candle) -> None:
        analyzer = TrendAnalyzer(config_loader)
        result = analyzer.analyze([sample_candle])
        assert result.direction == TrendDirection.UP  # close > open
        assert result.strength == 1.0  # 1/1

    def test_strength_is_ratio_of_dominant(self, config_loader) -> None:
        """3 green + 2 red => strength = 3/5 = 0.6."""
        candles = []
        start = datetime(2024, 1, 1, 12, 0, 0)
        for i in range(3):
            candles.append(
                Candle(
                    exchange="binance",
                    symbol="BTCUSDT",
                    open=Decimal("100"),
                    high=Decimal("110"),
                    low=Decimal("99"),
                    close=Decimal("105"),
                    volume=Decimal("10"),
                    timestamp=start + timedelta(minutes=i * 15),
                )
            )
        for i in range(3, 5):
            candles.append(
                Candle(
                    exchange="binance",
                    symbol="BTCUSDT",
                    open=Decimal("105"),
                    high=Decimal("106"),
                    low=Decimal("99"),
                    close=Decimal("100"),
                    volume=Decimal("10"),
                    timestamp=start + timedelta(minutes=i * 15),
                )
            )
        analyzer = TrendAnalyzer(config_loader)
        result = analyzer.analyze(candles)
        assert result.direction == TrendDirection.UP
        assert result.strength == pytest.approx(0.6)

    def test_cumulative_move_pct(self, config_loader, green_candles) -> None:
        analyzer = TrendAnalyzer(config_loader)
        result = analyzer.analyze(green_candles)
        assert result.cumulative_move_pct != 0.0

    def test_momentum_between_zero_and_one(self, config_loader, green_candles) -> None:
        analyzer = TrendAnalyzer(config_loader)
        result = analyzer.analyze(green_candles)
        assert 0.0 <= result.momentum <= 1.0

    def test_lookback_window_respected(self, config_loader, green_candles) -> None:
        """Only last 5 candles used even when given more."""
        extra_red = Candle(
            exchange="binance",
            symbol="BTCUSDT",
            open=Decimal("60000"),
            high=Decimal("60010"),
            low=Decimal("59000"),
            close=Decimal("59100"),
            volume=Decimal("100"),
            timestamp=datetime(2024, 1, 1, 11, 0, 0),
        )
        candles = [extra_red, extra_red, extra_red, *green_candles]
        analyzer = TrendAnalyzer(config_loader)
        result = analyzer.analyze(candles)
        # With lookback_candles=5, should only see last 5 (green)
        assert result.direction == TrendDirection.UP
        assert result.green_count == 5


class TestTrendAnalyzerHypothesis:
    """Property-based tests for TrendAnalyzer."""

    @given(
        n_green=st.integers(min_value=0, max_value=5),
        n_red=st.integers(min_value=0, max_value=5),
    )
    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_strength_bounded(self, config_loader, n_green: int, n_red: int) -> None:
        """Strength must always be in [0, 1]."""
        total = n_green + n_red
        if total == 0:
            return  # skip empty case
        candles = []
        for i in range(n_green):
            candles.append(
                Candle(
                    exchange="binance",
                    symbol="BTCUSDT",
                    open=Decimal("100"),
                    high=Decimal("110"),
                    low=Decimal("99"),
                    close=Decimal("105"),
                    volume=Decimal("10"),
                    timestamp=datetime(2024, 1, 1, 12, i, 0),
                )
            )
        for i in range(n_red):
            candles.append(
                Candle(
                    exchange="binance",
                    symbol="BTCUSDT",
                    open=Decimal("105"),
                    high=Decimal("106"),
                    low=Decimal("99"),
                    close=Decimal("100"),
                    volume=Decimal("10"),
                    timestamp=datetime(2024, 1, 1, 12, n_green + i, 0),
                )
            )
        analyzer = TrendAnalyzer(config_loader)
        result = analyzer.analyze(candles)
        assert 0.0 <= result.strength <= 1.0

    @given(
        n_green=st.integers(min_value=0, max_value=5),
        n_red=st.integers(min_value=0, max_value=5),
    )
    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_direction_consistent_with_counts(
        self, config_loader, n_green: int, n_red: int
    ) -> None:
        """Direction must match the majority candle color."""
        total = n_green + n_red
        if total == 0:
            return
        candles = []
        for i in range(n_green):
            candles.append(
                Candle(
                    exchange="binance",
                    symbol="BTCUSDT",
                    open=Decimal("100"),
                    high=Decimal("110"),
                    low=Decimal("99"),
                    close=Decimal("105"),
                    volume=Decimal("10"),
                    timestamp=datetime(2024, 1, 1, 12, i, 0),
                )
            )
        for i in range(n_red):
            candles.append(
                Candle(
                    exchange="binance",
                    symbol="BTCUSDT",
                    open=Decimal("105"),
                    high=Decimal("106"),
                    low=Decimal("99"),
                    close=Decimal("100"),
                    volume=Decimal("10"),
                    timestamp=datetime(2024, 1, 1, 12, n_green + i, 0),
                )
            )
        analyzer = TrendAnalyzer(config_loader)
        # Only last 5 considered
        result = analyzer.analyze(candles)
        if result.green_count > result.red_count:
            assert result.direction == TrendDirection.UP
        elif result.red_count > result.green_count:
            assert result.direction == TrendDirection.DOWN
        else:
            assert result.direction == TrendDirection.NEUTRAL
