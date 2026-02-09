"""Fetch BTCUSDT 1-minute klines from Binance: Jan 23 - Feb 8, 2026."""

import csv
import json
import time
import urllib.request
from datetime import datetime, timezone

SYMBOL = "BTCUSDT"
INTERVAL = "1m"
LIMIT = 1000
BASE_URL = "https://api.binance.com/api/v3/klines"
OUTPUT = "data/btc_1m_chunk6.csv"

START = datetime(2026, 1, 23, 0, 0, 0, tzinfo=timezone.utc)
END = datetime(2026, 2, 8, 0, 0, 0, tzinfo=timezone.utc)

start_ms = int(START.timestamp() * 1000)
end_ms = int(END.timestamp() * 1000)

rows = []
current_start = start_ms

while current_start < end_ms:
    params = (
        f"symbol={SYMBOL}&interval={INTERVAL}&limit={LIMIT}"
        f"&startTime={current_start}&endTime={end_ms}"
    )
    url = f"{BASE_URL}?{params}"

    for attempt in range(5):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            break
        except Exception as e:
            if attempt < 4:
                wait = 2 ** attempt
                print(f"  Retry {attempt+1}/4 after error: {e} (wait {wait}s)")
                time.sleep(wait)
            else:
                raise

    if not data:
        break

    for k in data:
        open_time_ms = k[0]
        if open_time_ms >= end_ms:
            break
        ts = datetime.fromtimestamp(open_time_ms / 1000, tz=timezone.utc).isoformat()
        rows.append([ts, k[1], k[2], k[3], k[4], k[5]])

    last_open = data[-1][0]
    current_start = last_open + 60_000  # next minute

    print(f"  Fetched {len(data)} candles, total so far: {len(rows)}")
    time.sleep(0.25)

with open(OUTPUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
    writer.writerows(rows)

print(f"\nSaved {len(rows)} rows to {OUTPUT}")
