"""Fetch BTCUSDT 1-minute klines from Binance: Jan 8 - Jan 23, 2026."""

import csv
import json
import time
import urllib.request
from datetime import datetime, timezone

SYMBOL = "BTCUSDT"
INTERVAL = "1m"
LIMIT = 1000
BASE_URL = "https://api.binance.com/api/v3/klines"

START = datetime(2026, 1, 8, 0, 0, 0, tzinfo=timezone.utc)
END = datetime(2026, 1, 23, 0, 0, 0, tzinfo=timezone.utc)

start_ms = int(START.timestamp() * 1000)
end_ms = int(END.timestamp() * 1000)

OUTPUT = "data/btc_1m_chunk5.csv"

rows = []
current_start = start_ms

while current_start < end_ms:
    params = (
        f"symbol={SYMBOL}&interval={INTERVAL}&limit={LIMIT}"
        f"&startTime={current_start}&endTime={end_ms - 1}"
    )
    url = f"{BASE_URL}?{params}"

    for attempt in range(5):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            break
        except Exception as e:
            print(f"  Retry {attempt + 1}/5: {e}")
            time.sleep(2 ** attempt)
    else:
        print(f"Failed after 5 retries at startTime={current_start}")
        break

    if not data:
        break

    for k in data:
        open_time_ms = k[0]
        ts = datetime.fromtimestamp(open_time_ms / 1000, tz=timezone.utc)
        if open_time_ms >= end_ms:
            break
        rows.append({
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "open": k[1],
            "high": k[2],
            "low": k[3],
            "close": k[4],
            "volume": k[5],
        })

    last_open = data[-1][0]
    current_start = last_open + 60_000  # next minute

    print(f"  Fetched {len(data)} candles, total so far: {len(rows)}")
    time.sleep(0.25)

with open(OUTPUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["timestamp", "open", "high", "low", "close", "volume"])
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved {len(rows)} rows to {OUTPUT}")
