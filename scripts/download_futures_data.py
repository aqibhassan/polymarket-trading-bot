"""Download BTC/ETH futures data from Binance Futures REST API.

Downloads 6 datasets:
1. BTC futures 1m klines (with taker_buy_volume) — 2 years
2. ETH futures 1m klines (with taker_buy_volume) — 2 years
3. BTC funding rate history — 2 years
4. BTC open interest (5m) — limited (~30-180 days)
5. BTC long/short ratio (5m) — limited (~30 days)
6. BTC taker long/short ratio (5m) — limited (~30 days)

Output: data/btc_futures_1m_2y.csv, data/eth_futures_1m_2y.csv, etc.
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

FAPI_BASE = "https://fapi.binance.com"
DAPI_BASE = "https://fapi.binance.com"

LIMIT_KLINES = 1500  # futures klines max per request
LIMIT_FUNDING = 1000
LIMIT_OI = 500
LIMIT_RATIO = 500


def _flush() -> None:
    sys.stdout.flush()


def _request_with_retry(url: str, params: dict, max_retries: int = 5) -> list | dict:
    """Make GET request with retry and rate-limit handling."""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 60))
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as e:
            print(f"  Request error (attempt {attempt + 1}): {e}")
            time.sleep(2 ** attempt)
    print(f"  FAILED after {max_retries} attempts")
    sys.exit(1)


def download_futures_klines(symbol: str, output_path: Path) -> None:
    """Download futures 1m klines with all columns including taker_buy_volume."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_expected = (END_TS - START_TS) // 60000
    print(f"\n  Downloading {symbol} futures 1m klines: ~{total_expected:,} expected")
    print(f"    From: {datetime.fromtimestamp(START_TS / 1000, tz=timezone.utc)}")
    print(f"    To:   {datetime.fromtimestamp(END_TS / 1000, tz=timezone.utc)}")
    print(f"    Output: {output_path}")
    _flush()

    url = f"{FAPI_BASE}/fapi/v1/klines"
    rows_written = 0
    current_start = START_TS
    request_count = 0

    header = [
        "timestamp", "open", "high", "low", "close", "volume",
        "quote_volume", "num_trades", "taker_buy_volume", "taker_buy_quote_volume",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        while current_start < END_TS:
            params = {
                "symbol": symbol,
                "interval": "1m",
                "startTime": current_start,
                "endTime": END_TS,
                "limit": LIMIT_KLINES,
            }
            data = _request_with_retry(url, params)

            if not data:
                break

            for kline in data:
                # kline: [open_time, O, H, L, C, vol, close_time, quote_vol,
                #          num_trades, taker_buy_base_vol, taker_buy_quote_vol, ignore]
                ts = datetime.fromtimestamp(kline[0] / 1000, tz=timezone.utc)
                writer.writerow([
                    ts.isoformat(),
                    kline[1],   # open
                    kline[2],   # high
                    kline[3],   # low
                    kline[4],   # close
                    kline[5],   # volume
                    kline[7],   # quote_asset_volume
                    kline[8],   # num_trades
                    kline[9],   # taker_buy_base_asset_volume
                    kline[10],  # taker_buy_quote_asset_volume
                ])
                rows_written += 1

            current_start = data[-1][0] + 60000
            request_count += 1

            if request_count % 50 == 0:
                pct = rows_written / total_expected * 100
                print(f"    {rows_written:>8,} / {total_expected:,} ({pct:.1f}%) - {request_count} requests")
                _flush()

            if request_count % 10 == 0:
                time.sleep(0.3)

    print(f"  Done! {rows_written:,} candles -> {output_path}")
    print(f"    Requests: {request_count}, Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    _flush()


def download_funding_rate(symbol: str, output_path: Path) -> None:
    """Download funding rate history."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n  Downloading {symbol} funding rate history")
    print(f"    Output: {output_path}")
    _flush()

    url = f"{FAPI_BASE}/fapi/v1/fundingRate"
    rows_written = 0
    current_start = START_TS
    request_count = 0

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "funding_rate"])

        while current_start < END_TS:
            params = {
                "symbol": symbol,
                "startTime": current_start,
                "endTime": END_TS,
                "limit": LIMIT_FUNDING,
            }
            data = _request_with_retry(url, params)

            if not data:
                break

            for entry in data:
                ts = datetime.fromtimestamp(entry["fundingTime"] / 1000, tz=timezone.utc)
                writer.writerow([ts.isoformat(), entry["fundingRate"]])
                rows_written += 1

            current_start = data[-1]["fundingTime"] + 1
            request_count += 1

            if request_count % 20 == 0:
                time.sleep(0.3)

    print(f"  Done! {rows_written:,} rates -> {output_path}")
    _flush()


def download_open_interest(symbol: str, output_path: Path) -> None:
    """Download open interest history (5m, limited availability)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n  Downloading {symbol} open interest (5m)")
    print(f"    Output: {output_path}")
    _flush()

    url = f"{FAPI_BASE}/futures/data/openInterestHist"
    rows_written = 0
    current_start = START_TS
    request_count = 0

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "sum_open_interest", "sum_open_interest_value"])

        while current_start < END_TS:
            params = {
                "symbol": symbol,
                "period": "5m",
                "startTime": current_start,
                "endTime": END_TS,
                "limit": LIMIT_OI,
            }
            try:
                data = _request_with_retry(url, params)
            except SystemExit:
                print("  WARNING: OI data may have limited availability, continuing...")
                break

            if not data or not isinstance(data, list):
                break

            for entry in data:
                ts = datetime.fromtimestamp(entry["timestamp"] / 1000, tz=timezone.utc)
                writer.writerow([
                    ts.isoformat(),
                    entry.get("sumOpenInterest", 0),
                    entry.get("sumOpenInterestValue", 0),
                ])
                rows_written += 1

            current_start = data[-1]["timestamp"] + 1
            request_count += 1

            if request_count % 20 == 0:
                time.sleep(0.5)

    print(f"  Done! {rows_written:,} records -> {output_path}")
    _flush()


def download_ratio_data(
    endpoint: str, symbol: str, output_path: Path, label: str,
) -> None:
    """Download long/short or taker ratio data (5m, limited availability)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n  Downloading {symbol} {label} (5m)")
    print(f"    Output: {output_path}")
    _flush()

    url = f"{FAPI_BASE}/{endpoint}"
    rows_written = 0
    current_start = START_TS
    request_count = 0

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "long_account", "short_account", "long_short_ratio"])

        while current_start < END_TS:
            params = {
                "symbol": symbol,
                "period": "5m",
                "startTime": current_start,
                "endTime": END_TS,
                "limit": LIMIT_RATIO,
            }
            try:
                data = _request_with_retry(url, params)
            except SystemExit:
                print(f"  WARNING: {label} data may have limited availability, continuing...")
                break

            if not data or not isinstance(data, list):
                break

            for entry in data:
                ts = datetime.fromtimestamp(entry["timestamp"] / 1000, tz=timezone.utc)
                writer.writerow([
                    ts.isoformat(),
                    entry.get("longAccount", entry.get("buyVol", 0)),
                    entry.get("shortAccount", entry.get("sellVol", 0)),
                    entry.get("longShortRatio", entry.get("buySellRatio", 0)),
                ])
                rows_written += 1

            current_start = data[-1]["timestamp"] + 1
            request_count += 1

            if request_count % 20 == 0:
                time.sleep(0.5)

    print(f"  Done! {rows_written:,} records -> {output_path}")
    _flush()


def main() -> None:
    print("=" * 100)
    print("  BINANCE FUTURES DATA DOWNLOAD")
    print("  BTC/ETH klines + funding + OI + ratios")
    print("=" * 100)
    _flush()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. BTC futures 1m klines (2 years)
    download_futures_klines("BTCUSDT", DATA_DIR / "btc_futures_1m_2y.csv")

    # 2. ETH futures 1m klines (2 years)
    download_futures_klines("ETHUSDT", DATA_DIR / "eth_futures_1m_2y.csv")

    # 3. BTC funding rate (2 years)
    download_funding_rate("BTCUSDT", DATA_DIR / "btc_funding_rate.csv")

    # 4. BTC open interest (limited)
    download_open_interest("BTCUSDT", DATA_DIR / "btc_open_interest_5m.csv")

    # 5. BTC global long/short account ratio (limited)
    download_ratio_data(
        "futures/data/globalLongShortAccountRatio",
        "BTCUSDT",
        DATA_DIR / "btc_long_short_ratio.csv",
        "long/short ratio",
    )

    # 6. BTC taker long/short ratio (limited)
    download_ratio_data(
        "futures/data/takerlongshortRatio",
        "BTCUSDT",
        DATA_DIR / "btc_taker_ls_ratio.csv",
        "taker L/S ratio",
    )

    print(f"\n{'=' * 100}")
    print("  ALL DOWNLOADS COMPLETE")
    print(f"{'=' * 100}")

    # Print file sizes
    for name in [
        "btc_futures_1m_2y.csv", "eth_futures_1m_2y.csv",
        "btc_funding_rate.csv", "btc_open_interest_5m.csv",
        "btc_long_short_ratio.csv", "btc_taker_ls_ratio.csv",
    ]:
        p = DATA_DIR / name
        if p.exists():
            sz = p.stat().st_size / 1024 / 1024
            print(f"  {name:<30} {sz:>8.1f} MB")
        else:
            print(f"  {name:<30} MISSING")
    _flush()


if __name__ == "__main__":
    main()
