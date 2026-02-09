"""Tests for CandleAggregator."""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from src.data.candle_aggregator import CandleAggregator
from src.models.market import Candle  # noqa: TCH001


@pytest.fixture()
def aggregator() -> CandleAggregator:
    return CandleAggregator(
        exchange="binance",
        symbol="BTCUSDT",
        interval_seconds=60,
        interval_label="1m",
    )


class TestCandleAggregatorBasic:
    def test_first_tick_returns_none(self, aggregator: CandleAggregator) -> None:
        result = aggregator.add_tick(
            price=Decimal("50000"),
            volume=Decimal("1"),
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
        )
        assert result is None

    def test_current_candle_after_first_tick(self, aggregator: CandleAggregator) -> None:
        aggregator.add_tick(
            price=Decimal("50000"),
            volume=Decimal("1"),
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
        )
        candle = aggregator.current_candle
        assert candle is not None
        assert candle.open == Decimal("50000")
        assert candle.high == Decimal("50000")
        assert candle.low == Decimal("50000")
        assert candle.close == Decimal("50000")
        assert candle.volume == Decimal("1")

    def test_current_candle_when_empty(self, aggregator: CandleAggregator) -> None:
        assert aggregator.current_candle is None


class TestCandleAggregatorOHLCV:
    def test_updates_high(self, aggregator: CandleAggregator) -> None:
        ts = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        aggregator.add_tick(Decimal("50000"), Decimal("1"), ts)
        aggregator.add_tick(Decimal("50500"), Decimal("1"), ts)
        candle = aggregator.current_candle
        assert candle is not None
        assert candle.high == Decimal("50500")

    def test_updates_low(self, aggregator: CandleAggregator) -> None:
        ts = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        aggregator.add_tick(Decimal("50000"), Decimal("1"), ts)
        aggregator.add_tick(Decimal("49500"), Decimal("1"), ts)
        candle = aggregator.current_candle
        assert candle is not None
        assert candle.low == Decimal("49500")

    def test_accumulates_volume(self, aggregator: CandleAggregator) -> None:
        ts = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        aggregator.add_tick(Decimal("50000"), Decimal("10"), ts)
        aggregator.add_tick(Decimal("50100"), Decimal("20"), ts)
        aggregator.add_tick(Decimal("50200"), Decimal("30"), ts)
        candle = aggregator.current_candle
        assert candle is not None
        assert candle.volume == Decimal("60")

    def test_close_is_last_price(self, aggregator: CandleAggregator) -> None:
        ts = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        aggregator.add_tick(Decimal("50000"), Decimal("1"), ts)
        aggregator.add_tick(Decimal("50300"), Decimal("1"), ts)
        candle = aggregator.current_candle
        assert candle is not None
        assert candle.close == Decimal("50300")


class TestCandleAggregatorFinalization:
    def test_interval_rollover_finalizes_candle(self, aggregator: CandleAggregator) -> None:
        t1 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        t2 = datetime(2024, 1, 1, 12, 0, 30, tzinfo=UTC)
        t3 = datetime(2024, 1, 1, 12, 1, 0, tzinfo=UTC)  # New minute

        aggregator.add_tick(Decimal("50000"), Decimal("10"), t1)
        aggregator.add_tick(Decimal("50500"), Decimal("5"), t2)

        finalized = aggregator.add_tick(Decimal("50100"), Decimal("8"), t3)
        assert finalized is not None
        assert finalized.open == Decimal("50000")
        assert finalized.high == Decimal("50500")
        assert finalized.low == Decimal("50000")
        assert finalized.close == Decimal("50500")
        assert finalized.volume == Decimal("15")
        assert finalized.exchange == "binance"
        assert finalized.symbol == "BTCUSDT"
        assert finalized.interval == "1m"

    def test_new_candle_starts_after_finalization(self, aggregator: CandleAggregator) -> None:
        t1 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        t2 = datetime(2024, 1, 1, 12, 1, 0, tzinfo=UTC)

        aggregator.add_tick(Decimal("50000"), Decimal("10"), t1)
        aggregator.add_tick(Decimal("50100"), Decimal("8"), t2)

        candle = aggregator.current_candle
        assert candle is not None
        assert candle.open == Decimal("50100")
        assert candle.volume == Decimal("8")

    def test_multiple_rollovers(self, aggregator: CandleAggregator) -> None:
        finalized: list[Candle] = []
        for minute in range(5):
            ts = datetime(2024, 1, 1, 12, minute, 0, tzinfo=UTC)
            result = aggregator.add_tick(Decimal(str(50000 + minute * 100)), Decimal("1"), ts)
            if result is not None:
                finalized.append(result)
        # 4 rollovers => 4 finalized candles (first tick starts candle 0)
        assert len(finalized) == 4


class TestCandleAggregatorThreadSafety:
    def test_concurrent_ticks(self) -> None:
        agg = CandleAggregator(
            exchange="test",
            symbol="BTCUSDT",
            interval_seconds=60,
            interval_label="1m",
        )
        errors: list[Exception] = []

        def add_ticks(start_price: int) -> None:
            try:
                ts = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
                for i in range(100):
                    agg.add_tick(
                        Decimal(str(start_price + i)),
                        Decimal("1"),
                        ts,
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_ticks, args=(50000 + t * 1000,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        candle = agg.current_candle
        assert candle is not None
        assert candle.volume == Decimal("400")
