"""Historical data loader for backtesting — CSV and TimescaleDB sources."""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

from src.core.logging import get_logger
from src.models.market import Candle

log = get_logger(__name__)


class DataLoadError(Exception):
    """Raised when data loading or validation fails."""


class DataLoader:
    """Load, validate, and clean historical candle data."""

    def load_csv(self, path: str | Path) -> list[Candle]:
        """Load candles from a CSV file.

        Expected columns: timestamp, open, high, low, close, volume
        Optional columns: exchange, symbol, interval

        Args:
            path: Path to the CSV file.

        Returns:
            Sorted, validated list of Candle objects.

        Raises:
            DataLoadError: If the file is missing or malformed.
        """
        path = Path(path)
        if not path.exists():
            msg = f"CSV file not found: {path}"
            raise DataLoadError(msg)

        candles: list[Candle] = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                msg = f"CSV file has no header row: {path}"
                raise DataLoadError(msg)

            required = {"timestamp", "open", "high", "low", "close", "volume"}
            missing = required - set(reader.fieldnames)
            if missing:
                msg = f"CSV missing required columns: {missing}"
                raise DataLoadError(msg)

            for row_num, row in enumerate(reader, start=2):
                try:
                    ts = self._parse_timestamp(row["timestamp"])
                    candle = Candle(
                        exchange=row.get("exchange", "backtest"),
                        symbol=row.get("symbol", "UNKNOWN"),
                        open=Decimal(row["open"]),
                        high=Decimal(row["high"]),
                        low=Decimal(row["low"]),
                        close=Decimal(row["close"]),
                        volume=Decimal(row["volume"]),
                        timestamp=ts,
                        interval=row.get("interval", "15m"),
                    )
                    candles.append(candle)
                except (InvalidOperation, ValueError, KeyError, DataLoadError) as e:
                    log.warning("csv_row_skip", row=row_num, error=str(e))

        candles = self._remove_duplicates(candles)
        candles.sort(key=lambda c: c.timestamp)
        self._validate_ohlcv(candles)
        log.info("csv_loaded", path=str(path), candles=len(candles))
        return candles

    async def load_from_db(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        *,
        dsn: str = "",
    ) -> list[Candle]:
        """Load candles from TimescaleDB.

        Args:
            symbol: Trading pair symbol.
            start: Start of time range (inclusive).
            end: End of time range (inclusive).
            dsn: Database connection string.

        Returns:
            Sorted, validated list of Candle objects.
        """
        try:
            import asyncpg
        except ImportError:
            msg = "asyncpg is required for DB loading: pip install asyncpg"
            raise DataLoadError(msg)  # noqa: B904

        conn = await asyncpg.connect(dsn)
        try:
            rows = await conn.fetch(
                "SELECT timestamp, open, high, low, close, volume "
                "FROM candles WHERE symbol = $1 AND timestamp >= $2 AND timestamp <= $3 "
                "ORDER BY timestamp",
                symbol,
                start,
                end,
            )
            candles = [
                Candle(
                    exchange="timescaledb",
                    symbol=symbol,
                    open=Decimal(str(r["open"])),
                    high=Decimal(str(r["high"])),
                    low=Decimal(str(r["low"])),
                    close=Decimal(str(r["close"])),
                    volume=Decimal(str(r["volume"])),
                    timestamp=r["timestamp"],
                )
                for r in rows
            ]
        finally:
            await conn.close()

        candles = self._remove_duplicates(candles)
        self._validate_ohlcv(candles)
        log.info("db_loaded", symbol=symbol, candles=len(candles))
        return candles

    def detect_gaps(
        self,
        candles: list[Candle],
        expected_interval: timedelta | None = None,
    ) -> list[tuple[datetime, datetime]]:
        """Detect gaps (missing candles) in a time series.

        Args:
            candles: Sorted list of candles.
            expected_interval: Expected time between candles.
                Defaults to 15 minutes.

        Returns:
            List of (gap_start, gap_end) tuples.
        """
        if len(candles) < 2:
            return []

        if expected_interval is None:
            expected_interval = timedelta(minutes=15)

        tolerance = expected_interval * 1.5
        gaps: list[tuple[datetime, datetime]] = []

        for i in range(1, len(candles)):
            delta = candles[i].timestamp - candles[i - 1].timestamp
            if delta > tolerance:
                gaps.append((candles[i - 1].timestamp, candles[i].timestamp))

        if gaps:
            log.warning("gaps_detected", count=len(gaps))
        return gaps

    @staticmethod
    def _remove_duplicates(candles: list[Candle]) -> list[Candle]:
        """Remove candles with duplicate timestamps, keeping the first."""
        seen: set[datetime] = set()
        unique: list[Candle] = []
        for c in candles:
            if c.timestamp not in seen:
                seen.add(c.timestamp)
                unique.append(c)
        removed = len(candles) - len(unique)
        if removed:
            log.warning("duplicates_removed", count=removed)
        return unique

    @staticmethod
    def _validate_ohlcv(candles: list[Candle]) -> None:
        """Validate OHLCV relationships.

        Checks are already enforced by Candle.high_gte_low validator,
        but we additionally verify high >= open/close and low <= open/close.
        """
        for i, c in enumerate(candles):
            errors: list[str] = []
            if c.high < c.open or c.high < c.close:
                errors.append("high < open or close")
            if c.low > c.open or c.low > c.close:
                errors.append("low > open or close")
            if c.volume < 0:
                errors.append("negative volume")
            if errors:
                log.warning("ohlcv_invalid", index=i, ts=str(c.timestamp), issues=errors)

    @staticmethod
    def _parse_timestamp(raw: str) -> datetime:
        """Parse a timestamp string, supporting ISO 8601 and epoch seconds."""
        raw = raw.strip()
        # Python 3.9 fromisoformat doesn't handle 'Z' suffix — replace with +00:00
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            pass
        try:
            epoch = float(raw)
            return datetime.fromtimestamp(epoch, tz=timezone.utc)
        except (ValueError, OverflowError):
            pass
        msg = f"Unparseable timestamp: {raw}"
        raise DataLoadError(msg)
