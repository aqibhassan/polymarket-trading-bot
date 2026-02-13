"""Download Deribit options/derivatives data for BWO v4 backtest.

Downloads:
1. DVOL (BTC Volatility Index) — 1-minute resolution, 2 years
2. BTC-PERPETUAL candles — 1-minute resolution, 2 years
3. BTC-PERPETUAL funding rate history — 2 years

Output: data/deribit_dvol_1m.csv, data/deribit_btc_perp_1m.csv, data/deribit_funding.csv
"""

from __future__ import annotations

import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# 2 years: Feb 8 2024 -> Feb 8 2026
START_TS = int(datetime(2024, 2, 8, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
END_TS = int(datetime(2026, 2, 8, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)

BASE_URL = "https://www.deribit.com/api/v2/public"


def _flush() -> None:
    sys.stdout.flush()


def _request_with_retry(url: str, params: dict, max_retries: int = 5) -> dict:
    """Make GET request with retry and rate-limit handling."""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 10))
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                err = data["error"]
                if err.get("code") == 10028:  # too_many_requests
                    print(f"  API credits exhausted, waiting 5s...")
                    time.sleep(5)
                    continue
                print(f"  API error: {err}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return {}
            return data
        except (requests.RequestException, ValueError) as e:
            print(f"  Request error (attempt {attempt + 1}): {e}")
            time.sleep(2 ** attempt)
    print(f"  FAILED after {max_retries} attempts")
    return {}


def download_dvol(output_path: Path) -> None:
    """Download DVOL (Deribit Volatility Index) at 1-hour resolution."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n  Downloading BTC DVOL (1h resolution)")
    print(f"    From: {datetime.fromtimestamp(START_TS / 1000, tz=timezone.utc)}")
    print(f"    To:   {datetime.fromtimestamp(END_TS / 1000, tz=timezone.utc)}")
    print(f"    Expected: ~17,520 hourly candles")
    print(f"    Output: {output_path}")
    _flush()

    url = f"{BASE_URL}/get_volatility_index_data"
    rows_written = 0
    request_count = 0
    current_start = START_TS
    # ~200 days per chunk at hourly resolution
    CHUNK_MS = 200 * 24 * 3600 * 1000

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "open", "high", "low", "close"])

        while current_start < END_TS:
            chunk_end = min(current_start + CHUNK_MS, END_TS)

            params = {
                "currency": "BTC",
                "start_timestamp": current_start,
                "end_timestamp": chunk_end,
                "resolution": 3600,
            }

            data = _request_with_retry(url, params)
            if not data:
                print(f"  ERROR: No response at {current_start}")
                break

            result = data.get("result", {})
            candles = result.get("data", [])
            continuation = result.get("continuation")

            if not candles:
                current_start = chunk_end
                request_count += 1
                time.sleep(0.3)
                continue

            for candle in candles:
                ts = datetime.fromtimestamp(candle[0] / 1000, tz=timezone.utc)
                writer.writerow([ts.isoformat(), candle[1], candle[2], candle[3], candle[4]])
                rows_written += 1

            request_count += 1
            # Advance past the chunk (not continuation — avoid re-fetching)
            current_start = chunk_end
            print(f"    {rows_written:>8,} rows - {request_count} requests")
            _flush()
            time.sleep(0.3)

    print(f"  Done! {rows_written:,} DVOL candles -> {output_path}")
    if output_path.exists():
        print(f"    Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    _flush()


def download_btc_perpetual(output_path: Path) -> None:
    """Download BTC-PERPETUAL 1-hour candles from Deribit."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n  Downloading BTC-PERPETUAL 1h candles")
    print(f"    From: {datetime.fromtimestamp(START_TS / 1000, tz=timezone.utc)}")
    print(f"    To:   {datetime.fromtimestamp(END_TS / 1000, tz=timezone.utc)}")
    print(f"    Expected: ~17,520 candles")
    print(f"    Output: {output_path}")
    _flush()

    url = f"{BASE_URL}/get_tradingview_chart_data"
    rows_written = 0
    request_count = 0
    current_start = START_TS
    CHUNK_MS = 200 * 24 * 3600 * 1000  # ~200 days per chunk

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])

        while current_start < END_TS:
            chunk_end = min(current_start + CHUNK_MS, END_TS)

            params = {
                "instrument_name": "BTC-PERPETUAL",
                "start_timestamp": current_start,
                "end_timestamp": chunk_end,
                "resolution": "60",  # 60-minute (1 hour) resolution
            }

            data = _request_with_retry(url, params)
            if not data:
                print(f"  ERROR: No response at {current_start}")
                break

            result = data.get("result", {})
            ticks = result.get("ticks", [])
            opens = result.get("open", [])
            highs = result.get("high", [])
            lows = result.get("low", [])
            closes = result.get("close", [])
            volumes = result.get("volume", [])

            if not ticks:
                current_start = chunk_end
                request_count += 1
                time.sleep(0.3)
                continue

            for i in range(len(ticks)):
                ts = datetime.fromtimestamp(ticks[i] / 1000, tz=timezone.utc)
                writer.writerow([
                    ts.isoformat(),
                    opens[i] if i < len(opens) else 0,
                    highs[i] if i < len(highs) else 0,
                    lows[i] if i < len(lows) else 0,
                    closes[i] if i < len(closes) else 0,
                    volumes[i] if i < len(volumes) else 0,
                ])
                rows_written += 1

            request_count += 1
            current_start = chunk_end
            print(f"    {rows_written:>8,} rows - {request_count} requests")
            _flush()
            time.sleep(0.3)

    print(f"  Done! {rows_written:,} candles -> {output_path}")
    if output_path.exists():
        print(f"    Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    _flush()


def download_funding_rate(output_path: Path) -> None:
    """Download BTC-PERPETUAL funding rate history from Deribit."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n  Downloading BTC-PERPETUAL funding rate history")
    print(f"    Output: {output_path}")
    _flush()

    url = f"{BASE_URL}/get_funding_rate_history"
    rows_written = 0
    request_count = 0
    current_start = START_TS

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "index_price", "interest_8h", "prev_index_price"])

        while current_start < END_TS:
            # Funding rates are every 8 hours — chunk by ~90 days
            chunk_end = min(current_start + 90 * 24 * 3600 * 1000, END_TS)

            params = {
                "instrument_name": "BTC-PERPETUAL",
                "start_timestamp": current_start,
                "end_timestamp": chunk_end,
            }

            data = _request_with_retry(url, params)
            if not data:
                print(f"  ERROR: No response at {current_start}")
                break

            result = data.get("result", [])

            if not result:
                current_start = chunk_end
                request_count += 1
                time.sleep(0.5)
                continue

            for entry in result:
                ts = datetime.fromtimestamp(entry["timestamp"] / 1000, tz=timezone.utc)
                writer.writerow([
                    ts.isoformat(),
                    entry.get("index_price", 0),
                    entry.get("interest_8h", 0),
                    entry.get("prev_index_price", 0),
                ])
                rows_written += 1

            last_ts = result[-1]["timestamp"]
            current_start = last_ts + 1
            request_count += 1

            time.sleep(0.5)

    print(f"  Done! {rows_written:,} entries -> {output_path}")
    _flush()


def main() -> None:
    print("=" * 100)
    print("  DERIBIT DATA DOWNLOAD")
    print("  DVOL + BTC-PERPETUAL candles + Funding rates")
    print("=" * 100)
    _flush()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. DVOL (BTC Volatility Index) — 1-minute resolution
    download_dvol(DATA_DIR / "deribit_dvol_1m.csv")

    # 2. BTC-PERPETUAL 1m candles
    download_btc_perpetual(DATA_DIR / "deribit_btc_perp_1m.csv")

    # 3. Funding rate history
    download_funding_rate(DATA_DIR / "deribit_funding.csv")

    print(f"\n{'=' * 100}")
    print("  ALL DERIBIT DOWNLOADS COMPLETE")
    print(f"{'=' * 100}")

    for name in ["deribit_dvol_1m.csv", "deribit_btc_perp_1m.csv", "deribit_funding.csv"]:
        p = DATA_DIR / name
        if p.exists():
            sz = p.stat().st_size / 1024 / 1024
            # Count rows
            with open(p) as f:
                row_count = sum(1 for _ in f) - 1  # minus header
            print(f"  {name:<30} {sz:>8.1f} MB  {row_count:>10,} rows")
        else:
            print(f"  {name:<30} MISSING")
    _flush()


if __name__ == "__main__":
    main()
