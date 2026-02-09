"""Fetch BTCUSDT 1-minute klines from Binance: Nov 23 - Dec 8, 2025."""

import csv
import json
import time
import urllib.request
from datetime import datetime, timezone

SYMBOL = "BTCUSDT"
INTERVAL = "1m"
LIMIT = 1000
BASE_URL = "https://api.binance.com/api/v3/klines"

START_DT = datetime(2025, 11, 23, 0, 0, 0, tzinfo=timezone.utc)
END_DT = datetime(2025, 12, 8, 0, 0, 0, tzinfo=timezone.utc)

START_MS = int(START_DT.timestamp() * 1000)
END_MS = int(END_DT.timestamp() * 1000)

OUTPUT_PATH = "/Users/aqibhassan/Documents/polymarket-trading-bot/data/btc_1m_chunk2.csv"


def fetch_klines(start_ms: int, end_ms: int) -> list[list]:
    """Fetch all 1m klines between start_ms and end_ms via pagination."""
    all_klines: list[list] = []
    current_start = start_ms

    while current_start < end_ms:
        params = (
            f"symbol={SYMBOL}&interval={INTERVAL}&limit={LIMIT}"
            f"&startTime={current_start}&endTime={end_ms}"
        )
        url = f"{BASE_URL}?{params}"

        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())

        if not data:
            break

        all_klines.extend(data)
        last_open_time = data[-1][0]
        current_start = last_open_time + 60_000  # next minute

        print(f"  Fetched {len(data)} rows, total so far: {len(all_klines)}, "
              f"up to {datetime.fromtimestamp(last_open_time / 1000, tz=timezone.utc).isoformat()}")

        time.sleep(0.25)  # respect rate limits

    return all_klines


def main() -> None:
    print(f"Fetching {SYMBOL} 1m klines: {START_DT.isoformat()} -> {END_DT.isoformat()}")
    klines = fetch_klines(START_MS, END_MS)

    # Filter: only keep candles with open_time < END_MS
    klines = [k for k in klines if k[0] < END_MS]

    # Deduplicate by timestamp
    seen: set[int] = set()
    unique: list[list] = []
    for k in klines:
        if k[0] not in seen:
            seen.add(k[0])
            unique.append(k)
    klines = sorted(unique, key=lambda k: k[0])

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for k in klines:
            ts = datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc).isoformat()
            writer.writerow([ts, k[1], k[2], k[3], k[4], k[5]])

    print(f"\nSaved {len(klines)} rows to {OUTPUT_PATH}")

    # Verify expected count: 15 days * 24h * 60m = 21600
    expected = 15 * 24 * 60
    print(f"Expected ~{expected} rows, got {len(klines)}")

    if klines:
        first_ts = datetime.fromtimestamp(klines[0][0] / 1000, tz=timezone.utc).isoformat()
        last_ts = datetime.fromtimestamp(klines[-1][0] / 1000, tz=timezone.utc).isoformat()
        print(f"Range: {first_ts} -> {last_ts}")


if __name__ == "__main__":
    main()
