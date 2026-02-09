"""Download resolved BTC 15-minute prediction market data from Polymarket.

Uses Gamma API to fetch closed Bitcoin events (Oct 2025 - Feb 2026).
Extracts market metadata: outcomes, resolution, volume, timestamps.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data"
OUTPUT_CSV = OUTPUT_DIR / "polymarket_btc_15m.csv"

GAMMA_URL = "https://gamma-api.polymarket.com/events"
TAG_ID_BITCOIN = 235
PAGE_SIZE = 100
RATE_LIMIT_DELAY = 0.5  # seconds between requests

# 5 months: Oct 2025 - Feb 2026
START_DATE = "2025-10-01T00:00:00Z"
END_DATE = "2026-02-08T23:59:59Z"


def is_15m_btc_market(title: str) -> bool:
    """Check if event title matches 15-minute BTC up/down pattern."""
    title_lower = title.lower()
    # Match patterns like "Bitcoin: Up or Down? February 7, 6:45PM-7:00PM ET"
    if "bitcoin" not in title_lower:
        return False
    if "up or down" not in title_lower:
        return False
    # 15-min markets have time ranges like "6:45PM-7:00PM" (15 min apart)
    # Exclude hourly markets (which have wider ranges)
    # Check for common 15-min patterns
    if any(x in title_lower for x in ["15 minute", "15-minute", "15m"]):
        return True
    # Most 15-min markets have format with :00, :15, :30, :45 time boundaries
    import re
    time_pattern = re.search(r'(\d{1,2}):(\d{2})\s*(am|pm)\s*[-–]\s*(\d{1,2}):(\d{2})\s*(am|pm)', title_lower)
    if time_pattern:
        h1, m1, ap1, h2, m2, ap2 = time_pattern.groups()
        h1, m1, h2, m2 = int(h1), int(m1), int(h2), int(m2)
        if ap1 == ap2:
            diff = (h2 * 60 + m2) - (h1 * 60 + m1)
            if diff == 15 or diff == -45:  # 15 min or wrapping around hour
                return True
        elif ap1 == "am" and ap2 == "pm" and h1 == 11 and m1 == 45 and h2 == 12 and m2 == 0:
            return True  # 11:45AM-12:00PM
    return False


def parse_resolution(market: dict) -> str | None:
    """Parse market resolution (Up/Down/None)."""
    outcome_prices = market.get("outcomePrices")
    outcomes = market.get("outcomes")
    if not outcome_prices or not outcomes:
        return None

    try:
        if isinstance(outcome_prices, str):
            prices = json.loads(outcome_prices)
        else:
            prices = outcome_prices
        if isinstance(outcomes, str):
            names = json.loads(outcomes)
        else:
            names = outcomes
    except (json.JSONDecodeError, TypeError):
        return None

    if len(prices) < 2 or len(names) < 2:
        return None

    # Resolved markets have prices [1, 0] or [0, 1]
    try:
        p0, p1 = float(prices[0]), float(prices[1])
    except (ValueError, TypeError):
        return None

    if p0 >= 0.99 and p1 <= 0.01:
        return str(names[0])  # First outcome won
    elif p1 >= 0.99 and p0 <= 0.01:
        return str(names[1])  # Second outcome won
    return None


def download_events() -> list[dict]:
    """Download all closed Bitcoin events from Gamma API."""
    all_events: list[dict] = []
    offset = 0
    total_fetched = 0

    print(f"  Downloading closed Bitcoin events from {START_DATE[:10]} to {END_DATE[:10]}...")
    print(f"  Using Gamma API: {GAMMA_URL}")

    while True:
        params = {
            "tag_id": TAG_ID_BITCOIN,
            "closed": "true",
            "limit": PAGE_SIZE,
            "offset": offset,
            "order": "endDate",
            "ascending": "true",
            "end_date_min": START_DATE,
            "end_date_max": END_DATE,
        }

        try:
            resp = requests.get(GAMMA_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(f"  ERROR at offset {offset}: {e}")
            time.sleep(2)
            continue

        if not data:
            break

        all_events.extend(data)
        total_fetched += len(data)
        offset += PAGE_SIZE

        if total_fetched % 500 == 0:
            print(f"    ... fetched {total_fetched:,} events")

        time.sleep(RATE_LIMIT_DELAY)

    print(f"  Total events fetched: {total_fetched:,}")
    return all_events


def extract_markets(events: list[dict]) -> list[dict]:
    """Extract 15-minute BTC markets from events."""
    markets: list[dict] = []
    skipped_non_15m = 0
    skipped_no_resolution = 0

    for event in events:
        title = event.get("title", "")
        if not is_15m_btc_market(title):
            skipped_non_15m += 1
            continue

        for market in event.get("markets", []):
            resolution = parse_resolution(market)
            if resolution is None:
                skipped_no_resolution += 1
                continue

            end_date = market.get("endDate", "")
            closed_time = market.get("closedTime", "")
            volume = market.get("volume", 0)

            # Extract token IDs
            clob_token_ids = market.get("clobTokenIds", "[]")
            try:
                if isinstance(clob_token_ids, str):
                    token_ids = json.loads(clob_token_ids)
                else:
                    token_ids = clob_token_ids
            except (json.JSONDecodeError, TypeError):
                token_ids = []

            markets.append({
                "event_id": event.get("id", ""),
                "event_title": title,
                "market_id": market.get("id", ""),
                "condition_id": market.get("conditionId", ""),
                "token_up": token_ids[0] if len(token_ids) > 0 else "",
                "token_down": token_ids[1] if len(token_ids) > 1 else "",
                "end_date": end_date,
                "closed_time": closed_time,
                "volume": volume,
                "resolution": resolution,
                "outcomes": json.dumps(market.get("outcomes", [])),
                "outcome_prices": json.dumps(market.get("outcomePrices", [])),
            })

    print(f"  15-min BTC markets found: {len(markets):,}")
    print(f"  Skipped (not 15-min): {skipped_non_15m:,}")
    print(f"  Skipped (no resolution): {skipped_no_resolution:,}")
    return markets


def save_csv(markets: list[dict], path: Path) -> None:
    """Save markets to CSV."""
    if not markets:
        print("  No markets to save!")
        return

    fieldnames = list(markets[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(markets)

    print(f"  Saved {len(markets):,} markets to {path}")
    print(f"  File size: {path.stat().st_size / 1024:.1f} KB")


def print_summary(markets: list[dict]) -> None:
    """Print summary statistics."""
    if not markets:
        return

    # Count by month
    by_month: dict[str, int] = {}
    by_resolution: dict[str, int] = {}
    volumes: list[float] = []

    for m in markets:
        end_date = m["end_date"]
        try:
            dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            month = dt.strftime("%Y-%m")
        except (ValueError, AttributeError):
            month = "unknown"
        by_month[month] = by_month.get(month, 0) + 1

        res = m["resolution"]
        by_resolution[res] = by_resolution.get(res, 0) + 1

        try:
            volumes.append(float(m["volume"]))
        except (ValueError, TypeError):
            pass

    print(f"\n  {'='*60}")
    print(f"  POLYMARKET DATA SUMMARY")
    print(f"  {'='*60}")

    print(f"\n  Markets per month:")
    for month in sorted(by_month):
        print(f"    {month}: {by_month[month]:,}")

    print(f"\n  Resolution distribution:")
    total = sum(by_resolution.values())
    for res in sorted(by_resolution):
        pct = by_resolution[res] / total * 100
        print(f"    {res}: {by_resolution[res]:,} ({pct:.1f}%)")

    if volumes:
        print(f"\n  Volume stats:")
        print(f"    Total volume: ${sum(volumes):,.0f}")
        print(f"    Avg per market: ${sum(volumes)/len(volumes):,.0f}")
        print(f"    Median: ${sorted(volumes)[len(volumes)//2]:,.0f}")

    # Markets per day
    total_days = len(set(m["end_date"][:10] for m in markets if m.get("end_date")))
    if total_days > 0:
        print(f"\n  Coverage: {total_days} trading days, ~{len(markets)//total_days} markets/day")


def main() -> None:
    print("=" * 80)
    print("  POLYMARKET BTC 15-MIN MARKET DATA DOWNLOAD")
    print(f"  Period: {START_DATE[:10]} to {END_DATE[:10]} (5 months)")
    print("=" * 80)

    events = download_events()
    if not events:
        print("  No events found!")
        sys.exit(1)

    markets = extract_markets(events)
    if not markets:
        print("  No 15-min BTC markets found!")
        sys.exit(1)

    save_csv(markets, OUTPUT_CSV)
    print_summary(markets)

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
