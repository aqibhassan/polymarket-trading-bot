"""Fast CSV loader for large BTC data files.

Bypasses Pydantic model creation for speed. Returns simple namedtuples.
Loads 1M+ rows in seconds instead of minutes.
"""

from __future__ import annotations

import csv
from collections import namedtuple
from datetime import datetime, timezone
from pathlib import Path
from typing import List

# Lightweight candle representation
FastCandle = namedtuple("FastCandle", ["timestamp", "open", "high", "low", "close", "volume"])


def load_csv_fast(path: Path) -> List[FastCandle]:
    """Load CSV into lightweight FastCandle namedtuples.

    ~10x faster than DataLoader for large files since we skip Pydantic.
    """
    candles: list[FastCandle] = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_ts = row["timestamp"].strip()
            if raw_ts.endswith("Z"):
                raw_ts = raw_ts[:-1] + "+00:00"
            try:
                ts = datetime.fromisoformat(raw_ts)
            except ValueError:
                continue

            candles.append(FastCandle(
                timestamp=ts,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            ))
    return candles


def group_into_15m_windows(candles: List[FastCandle]) -> List[List[FastCandle]]:
    """Group 1m candles into 15m windows."""
    if not candles:
        return []
    windows: list[list[FastCandle]] = []
    current_window: list[FastCandle] = []
    current_window_start: int | None = None

    for c in candles:
        ts = c.timestamp
        minute = ts.minute
        window_minute = (minute // 15) * 15

        if current_window_start is None:
            if minute == window_minute:
                current_window_start = window_minute
                current_window = [c]
        else:
            expected_minute = current_window_start
            offset = minute - expected_minute
            if offset < 0:
                offset += 60
            same_window = (
                len(current_window) < 15
                and 0 <= offset < 15
                and ts.hour == current_window[0].timestamp.hour
                and ts.date() == current_window[0].timestamp.date()
            )
            if same_window:
                current_window.append(c)
            else:
                if len(current_window) == 15:
                    windows.append(current_window)
                if minute == window_minute:
                    current_window = [c]
                    current_window_start = window_minute
                else:
                    current_window = []
                    current_window_start = None

    if len(current_window) == 15:
        windows.append(current_window)
    return windows
