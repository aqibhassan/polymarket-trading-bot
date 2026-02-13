"""Fast CSV loader for futures data files (BTC/ETH klines, funding rates).

Extends fast_loader.py pattern with FuturesCandle (10 fields) and alignment
functions for matching futures data to spot 15m windows by timestamp.
"""

from __future__ import annotations

import csv
from bisect import bisect_right
from collections import namedtuple
from datetime import datetime
from pathlib import Path
from typing import List

# Lightweight futures candle — includes taker buy volume
FuturesCandle = namedtuple("FuturesCandle", [
    "timestamp", "open", "high", "low", "close", "volume",
    "quote_volume", "num_trades", "taker_buy_volume", "taker_buy_quote_volume",
])

# Funding rate entry
FundingEntry = namedtuple("FundingEntry", ["timestamp", "funding_rate"])


def _parse_ts(raw: str) -> datetime | None:
    """Parse ISO timestamp, handling Z suffix."""
    raw = raw.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def load_futures_csv(path: Path) -> List[FuturesCandle]:
    """Load futures klines CSV into FuturesCandle namedtuples."""
    candles: list[FuturesCandle] = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = _parse_ts(row["timestamp"])
            if ts is None:
                continue
            candles.append(FuturesCandle(
                timestamp=ts,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                quote_volume=float(row.get("quote_volume", 0)),
                num_trades=int(float(row.get("num_trades", 0))),
                taker_buy_volume=float(row.get("taker_buy_volume", 0)),
                taker_buy_quote_volume=float(row.get("taker_buy_quote_volume", 0)),
            ))
    return candles


def build_futures_lookup(candles: List[FuturesCandle]) -> dict[datetime, int]:
    """Build timestamp -> index map for O(1) lookup."""
    return {c.timestamp: i for i, c in enumerate(candles)}


def load_funding_csv(path: Path) -> List[FundingEntry]:
    """Load funding rate CSV."""
    entries: list[FundingEntry] = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = _parse_ts(row["timestamp"])
            if ts is None:
                continue
            entries.append(FundingEntry(
                timestamp=ts,
                funding_rate=float(row["funding_rate"]),
            ))
    entries.sort(key=lambda e: e.timestamp)
    return entries


def get_latest_funding(
    entries: List[FundingEntry],
    timestamps: list[datetime],
    before: datetime,
) -> FundingEntry | None:
    """Get the most recent funding rate before a given timestamp using binary search.

    `timestamps` must be a sorted list of FundingEntry.timestamp values,
    matching the order of `entries`.
    """
    idx = bisect_right(timestamps, before) - 1
    if idx >= 0:
        return entries[idx]
    return None
