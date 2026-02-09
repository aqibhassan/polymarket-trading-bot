"""Download 2 years of BTC/USDT 1m candle data from Binance REST API.

Output: data/btc_1m_2y.csv (~1.05M rows)
Date range: 2024-02-08 to 2026-02-08
"""

from __future__ import annotations

import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

SYMBOL = "BTCUSDT"
INTERVAL = "1m"
LIMIT = 1000  # max per request
BASE_URL = "https://api.binance.com/api/v3/klines"

# 2 years: Feb 8 2024 -> Feb 8 2026
START_TS = int(datetime(2024, 2, 8, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
END_TS = int(datetime(2026, 2, 8, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "btc_1m_2y.csv"


def download_klines() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    total_expected = (END_TS - START_TS) // 60000  # ~1.05M
    print(f"Downloading {SYMBOL} {INTERVAL} candles: {total_expected:,} expected")
    print(f"  From: {datetime.fromtimestamp(START_TS/1000, tz=timezone.utc)}")
    print(f"  To:   {datetime.fromtimestamp(END_TS/1000, tz=timezone.utc)}")
    print(f"  Output: {OUTPUT}")

    rows_written = 0
    current_start = START_TS
    request_count = 0

    with open(OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])

        while current_start < END_TS:
            params = {
                "symbol": SYMBOL,
                "interval": INTERVAL,
                "startTime": current_start,
                "endTime": END_TS,
                "limit": LIMIT,
            }

            for attempt in range(5):
                try:
                    resp = requests.get(BASE_URL, params=params, timeout=30)
                    if resp.status_code == 429:
                        wait = int(resp.headers.get("Retry-After", 60))
                        print(f"  Rate limited, waiting {wait}s...")
                        time.sleep(wait)
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    break
                except (requests.RequestException, ValueError) as e:
                    print(f"  Request error (attempt {attempt+1}): {e}")
                    time.sleep(2 ** attempt)
            else:
                print(f"  FAILED after 5 attempts at {current_start}")
                sys.exit(1)

            if not data:
                break

            for kline in data:
                # kline format: [open_time, open, high, low, close, volume, close_time, ...]
                ts = datetime.fromtimestamp(kline[0] / 1000, tz=timezone.utc)
                writer.writerow([
                    ts.isoformat(),
                    kline[1],  # open
                    kline[2],  # high
                    kline[3],  # low
                    kline[4],  # close
                    kline[5],  # volume
                ])
                rows_written += 1

            # Move to next batch (start after last candle's open time)
            current_start = data[-1][0] + 60000  # +1 minute
            request_count += 1

            if request_count % 50 == 0:
                pct = rows_written / total_expected * 100
                print(f"  {rows_written:>8,} / {total_expected:,} ({pct:.1f}%) - {request_count} requests")

            # Rate limit: ~1200 req/min allowed, be conservative
            if request_count % 10 == 0:
                time.sleep(0.5)

    print(f"\nDone! {rows_written:,} candles written to {OUTPUT}")
    print(f"  Requests: {request_count}")
    print(f"  File size: {OUTPUT.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    download_klines()
