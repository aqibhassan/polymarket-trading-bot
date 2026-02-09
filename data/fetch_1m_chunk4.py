"""Fetch BTCUSDT 1-minute klines from Binance: Dec 23, 2025 - Jan 8, 2026."""
import urllib.request
import json
import csv
import time
from datetime import datetime, timezone

SYMBOL = "BTCUSDT"
INTERVAL = "1m"
LIMIT = 1000
BASE_URL = "https://api.binance.com/api/v3/klines"
OUTPUT = "data/btc_1m_chunk4.csv"

# Date range in milliseconds
START_MS = int(datetime(2025, 12, 23, 0, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
END_MS = int(datetime(2026, 1, 8, 0, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)


def fetch_klines(start_time: int, end_time: int) -> list:
    """Fetch one batch of klines from Binance."""
    params = (
        f"symbol={SYMBOL}&interval={INTERVAL}&limit={LIMIT}"
        f"&startTime={start_time}&endTime={end_time}"
    )
    url = f"{BASE_URL}?{params}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def ms_to_iso(ms: int) -> str:
    """Convert millisecond epoch to ISO 8601 UTC string."""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main():
    all_rows = []
    current_start = START_MS
    batch = 0

    while current_start < END_MS:
        batch += 1
        data = fetch_klines(current_start, END_MS)
        if not data:
            break

        for candle in data:
            open_time = candle[0]
            # Skip candles at or beyond end time
            if open_time >= END_MS:
                break
            all_rows.append({
                "timestamp": ms_to_iso(open_time),
                "open": candle[1],
                "high": candle[2],
                "low": candle[3],
                "close": candle[4],
                "volume": candle[5],
            })

        last_close_time = data[-1][6]
        current_start = last_close_time + 1
        print(f"  Batch {batch}: fetched {len(data)} candles, total so far: {len(all_rows)}")
        time.sleep(0.25)  # rate limit courtesy

    # Write CSV
    with open(OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "open", "high", "low", "close", "volume"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nTotal rows: {len(all_rows)}")
    if all_rows:
        print(f"First row: {all_rows[0]}")
        print(f"Last row:  {all_rows[-1]}")
    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
