"""Fetch BTCUSDT 1-minute klines from Binance: Dec 8 - Dec 23, 2025."""

import csv
import json
import time
import urllib.request
from datetime import datetime, timezone

BASE_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1m"
LIMIT = 1000

START_DT = datetime(2025, 12, 8, 0, 0, 0, tzinfo=timezone.utc)
END_DT = datetime(2025, 12, 23, 0, 0, 0, tzinfo=timezone.utc)

START_MS = int(START_DT.timestamp() * 1000)
END_MS = int(END_DT.timestamp() * 1000)

OUTPUT_PATH = "data/btc_1m_chunk3.csv"


def fetch_klines(start_ms: int, end_ms: int) -> list[list]:
    """Fetch all 1m klines between start_ms and end_ms with pagination."""
    all_klines: list[list] = []
    current_start = start_ms

    while current_start < end_ms:
        url = (
            f"{BASE_URL}?symbol={SYMBOL}&interval={INTERVAL}"
            f"&limit={LIMIT}&startTime={current_start}&endTime={end_ms - 1}"
        )
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())

        if not data:
            break

        all_klines.extend(data)
        # Next batch starts 1ms after the last candle open time
        current_start = data[-1][0] + 60_000  # next minute
        print(f"  Fetched {len(data)} rows, total so far: {len(all_klines)}")
        time.sleep(0.25)  # rate-limit courtesy

    return all_klines


def main() -> None:
    print(f"Fetching {SYMBOL} 1m klines")
    print(f"  From: {START_DT.isoformat()}")
    print(f"  To:   {END_DT.isoformat()}")

    raw = fetch_klines(START_MS, END_MS)

    # Write CSV
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for k in raw:
            ts_iso = datetime.fromtimestamp(
                k[0] / 1000, tz=timezone.utc
            ).isoformat()
            writer.writerow([ts_iso, k[1], k[2], k[3], k[4], k[5]])

    print(f"\nSaved {len(raw)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
