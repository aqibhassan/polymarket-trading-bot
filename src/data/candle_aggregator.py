"""Tick-to-OHLCV candle aggregation."""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from decimal import Decimal

from src.core.logging import get_logger
from src.models.market import Candle

log = get_logger(__name__)


class CandleAggregator:
    """Aggregates raw ticks into OHLCV candles at a configurable interval.

    Thread-safe: all mutations are guarded by a lock.
    """

    def __init__(
        self,
        exchange: str,
        symbol: str,
        interval_seconds: int = 900,
        interval_label: str = "15m",
        server_time_offset_ms: int = 0,
    ) -> None:
        self._exchange = exchange
        self._symbol = symbol
        self._interval_seconds = interval_seconds
        self._interval_label = interval_label
        self._server_time_offset_ms = server_time_offset_ms
        self._lock = threading.Lock()
        self._open: Decimal | None = None
        self._high: Decimal | None = None
        self._low: Decimal | None = None
        self._close: Decimal | None = None
        self._volume: Decimal = Decimal("0")
        self._candle_start: datetime | None = None

    def _bucket_start(self, ts: datetime) -> datetime:
        """Compute the interval-aligned start time for a timestamp.

        Applies server_time_offset_ms to correct for clock skew between
        the local clock and the exchange server clock.
        """
        if ts.tzinfo is None:
            epoch = int(ts.replace(tzinfo=UTC).timestamp())
        else:
            epoch = int(ts.timestamp())
        # Apply clock skew correction (offset in ms -> s)
        epoch += self._server_time_offset_ms // 1000
        bucket = epoch - (epoch % self._interval_seconds)
        return datetime.fromtimestamp(bucket, tz=UTC)

    def add_tick(
        self,
        price: Decimal,
        volume: Decimal,
        timestamp: datetime,
    ) -> Candle | None:
        """Add a tick and return a finalized Candle if the interval rolled over.

        Args:
            price: Tick price.
            volume: Tick volume.
            timestamp: Tick timestamp.

        Returns:
            A finalized Candle if the interval boundary was crossed, else None.
        """
        bucket = self._bucket_start(timestamp)

        with self._lock:
            # First tick ever
            if self._candle_start is None:
                self._start_new_candle(price, volume, bucket)
                return None

            # Same interval — update in-progress candle
            if bucket == self._candle_start:
                self._update_candle(price, volume)
                return None

            # New interval — finalize current candle, start new one
            finalized = self._finalize_candle()
            self._start_new_candle(price, volume, bucket)
            return finalized

    @property
    def current_candle(self) -> Candle | None:
        """Return the in-progress (partial) candle, or None."""
        with self._lock:
            if self._open is None or self._candle_start is None:
                return None
            assert self._high is not None
            assert self._low is not None
            assert self._close is not None
            return Candle(
                exchange=self._exchange,
                symbol=self._symbol,
                open=self._open,
                high=self._high,
                low=self._low,
                close=self._close,
                volume=self._volume,
                timestamp=self._candle_start,
                interval=self._interval_label,
            )

    def _start_new_candle(self, price: Decimal, volume: Decimal, bucket: datetime) -> None:
        self._open = price
        self._high = price
        self._low = price
        self._close = price
        self._volume = volume
        self._candle_start = bucket

    def _update_candle(self, price: Decimal, volume: Decimal) -> None:
        assert self._high is not None
        assert self._low is not None
        if price > self._high:
            self._high = price
        if price < self._low:
            self._low = price
        self._close = price
        self._volume += volume

    def _finalize_candle(self) -> Candle:
        assert self._open is not None
        assert self._high is not None
        assert self._low is not None
        assert self._close is not None
        assert self._candle_start is not None
        candle = Candle(
            exchange=self._exchange,
            symbol=self._symbol,
            open=self._open,
            high=self._high,
            low=self._low,
            close=self._close,
            volume=self._volume,
            timestamp=self._candle_start,
            interval=self._interval_label,
        )
        log.debug(
            "aggregator.candle_finalized",
            symbol=self._symbol,
            close=str(candle.close),
        )
        return candle
