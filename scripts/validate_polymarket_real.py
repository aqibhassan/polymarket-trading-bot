"""Validate strategy against REAL Polymarket resolved BTC 15-min markets.

Cross-references our Binance candle-based predictions with actual
Polymarket market resolutions. Includes realistic fee modeling.

This is the ultimate real-world validation — if our Binance-based prediction
matches Polymarket's resolution, the strategy works in production.
"""

from __future__ import annotations

import csv
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fast_loader import FastCandle, load_csv_fast

POLYMARKET_CSV = PROJECT_ROOT / "data" / "polymarket_btc_15m.csv"
BTC_CSV = PROJECT_ROOT / "data" / "btc_1m_2y.csv"
POSITION_SIZE = 100.0
SENSITIVITY = 0.07
TIERS = [(8, 0.0010), (9, 0.0008), (10, 0.0005)]
FEE_CONSTANT = 0.25
SLIPPAGE_BPS = 5


def _sigmoid(cum_ret: float) -> float:
    x = SENSITIVITY * cum_ret * 10000
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def polymarket_fee(position_size: float, entry_price: float) -> float:
    if entry_price <= 0 or entry_price >= 1:
        return 0.0
    return position_size * FEE_CONSTANT * (entry_price ** 2) * ((1.0 - entry_price) ** 2)


def parse_et_time(title: str) -> tuple[datetime, datetime] | None:
    """Parse the ET time range from a Polymarket BTC event title.

    Example: 'Bitcoin Up or Down - February 7, 6:45PM-7:00PM ET'
    Returns (window_start_utc, window_end_utc) or None.
    """
    # Match patterns like "January 15, 3:30PM-3:45PM ET" or "December 31, 11:45AM-12:00PM ET"
    pattern = r'(\w+ \d+),?\s+(\d{1,2}):(\d{2})(AM|PM)\s*[-–]\s*(\d{1,2}):(\d{2})(AM|PM)\s*ET'
    match = re.search(pattern, title, re.IGNORECASE)
    if not match:
        return None

    date_str, h1, m1, ap1, h2, m2, ap2 = match.groups()
    h1, m1, h2, m2 = int(h1), int(m1), int(h2), int(m2)
    ap1, ap2 = ap1.upper(), ap2.upper()

    # Convert to 24h
    if ap1 == "PM" and h1 != 12:
        h1 += 12
    elif ap1 == "AM" and h1 == 12:
        h1 = 0
    if ap2 == "PM" and h2 != 12:
        h2 += 12
    elif ap2 == "AM" and h2 == 12:
        h2 = 0

    # Parse the date (try multiple year options)
    for year in [2026, 2025, 2024]:
        try:
            dt = datetime.strptime(f"{date_str} {year}", "%B %d %Y")
            break
        except ValueError:
            continue
    else:
        return None

    # ET = UTC-5 (EST) or UTC-4 (EDT)
    # Approximate: Nov-Mar = EST (UTC-5), Apr-Oct = EDT (UTC-4)
    month = dt.month
    if month >= 4 and month <= 10:
        utc_offset = timedelta(hours=4)  # EDT
    else:
        utc_offset = timedelta(hours=5)  # EST

    start_utc = datetime(dt.year, dt.month, dt.day, h1, m1, tzinfo=timezone.utc) + utc_offset
    end_utc = datetime(dt.year, dt.month, dt.day, h2, m2, tzinfo=timezone.utc) + utc_offset

    # Handle overnight wrap (e.g., 11:45PM-12:00AM)
    if end_utc <= start_utc:
        end_utc += timedelta(days=1)

    return start_utc, end_utc


@dataclass
class PolymarketMatch:
    event_title: str
    window_start: datetime
    window_end: datetime
    poly_resolution: str  # "Up" or "Down"
    binance_direction: str  # "Up" or "Down" based on candle data
    resolution_match: bool  # Do Polymarket and Binance agree?
    our_prediction: str | None  # "Up" or "Down" or None (no trade)
    entry_minute: int | None
    entry_price: float | None
    cum_ret: float | None
    correct: bool | None  # Did we predict correctly?
    pnl_gross: float | None
    pnl_net: float | None
    fee: float | None
    volume: float


def main() -> None:
    print("=" * 120)
    print("  REAL POLYMARKET DATA VALIDATION")
    print("  Cross-referencing Binance candles with Polymarket resolutions + fees")
    print("=" * 120)

    if not POLYMARKET_CSV.exists():
        print(f"  ERROR: {POLYMARKET_CSV} not found. Run download_polymarket_btc.py first.")
        sys.exit(1)
    if not BTC_CSV.exists():
        print(f"  ERROR: {BTC_CSV} not found.")
        sys.exit(1)

    # Load Polymarket data
    poly_markets: list[dict] = []
    with open(POLYMARKET_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            poly_markets.append(row)
    print(f"\n  Loaded {len(poly_markets):,} Polymarket markets")

    # Load Binance candles and index by timestamp
    print("  Loading Binance candle data...")
    all_candles = load_csv_fast(BTC_CSV)
    all_candles.sort(key=lambda c: c.timestamp)

    # Deduplicate
    seen: set[datetime] = set()
    unique = [c for c in all_candles if c.timestamp not in seen and not seen.add(c.timestamp)]  # type: ignore
    all_candles = unique
    print(f"  Loaded {len(all_candles):,} Binance 1m candles")

    # Build index: timestamp -> candle for O(1) lookup
    candle_index: dict[datetime, FastCandle] = {}
    for c in all_candles:
        # Normalize: strip timezone info for matching
        ts = c.timestamp.replace(tzinfo=None) if c.timestamp.tzinfo else c.timestamp
        candle_index[ts] = c

    # Match each Polymarket market with Binance candles
    matches: list[PolymarketMatch] = []
    parse_failures = 0
    no_candle_data = 0
    insufficient_candles = 0

    for pm in poly_markets:
        title = pm["event_title"]
        resolution = pm["resolution"]
        volume = float(pm.get("volume", 0))

        # Parse time window from title
        time_range = parse_et_time(title)
        if time_range is None:
            parse_failures += 1
            continue

        window_start, window_end = time_range

        # Get the 15 candles for this window from Binance
        window_candles: list[FastCandle] = []
        for minute_offset in range(15):
            ts = (window_start + timedelta(minutes=minute_offset)).replace(tzinfo=None)
            candle = candle_index.get(ts)
            if candle is not None:
                window_candles.append(candle)

        if len(window_candles) == 0:
            no_candle_data += 1
            continue

        if len(window_candles) < 11:  # Need at least through minute 10
            insufficient_candles += 1
            continue

        # Determine Binance direction
        window_open = float(window_candles[0].open)
        final_close = float(window_candles[-1].close)
        if window_open == 0:
            continue
        binance_green = final_close > window_open
        binance_direction = "Up" if binance_green else "Down"
        resolution_match = (binance_direction == resolution)

        # Run our strategy on this window
        our_prediction = None
        entry_minute = None
        entry_price = None
        cum_ret_val = None
        correct = None
        pnl_gross = None
        pnl_net = None
        fee = None

        for entry_min, threshold in TIERS:
            if entry_min >= len(window_candles):
                continue
            current_close = float(window_candles[entry_min].close)
            cum_ret = (current_close - window_open) / window_open
            if abs(cum_ret) < threshold:
                continue

            predict_green = cum_ret > 0
            entry_yes = _sigmoid(cum_ret)
            ep = entry_yes if predict_green else 1.0 - entry_yes
            if ep <= 0.01:
                continue

            our_prediction = "Up" if predict_green else "Down"
            entry_minute = entry_min
            entry_price = ep
            cum_ret_val = cum_ret

            # Check against POLYMARKET resolution (not Binance!)
            correct = (our_prediction == resolution)

            # Settlement based on Polymarket resolution
            if predict_green:
                settlement = 1.0 if resolution == "Up" else 0.0
            else:
                settlement = 1.0 if resolution == "Down" else 0.0

            pnl_gross = ((settlement - ep) / ep) * POSITION_SIZE
            f = polymarket_fee(POSITION_SIZE, ep)
            slip = POSITION_SIZE * SLIPPAGE_BPS / 10000
            fee = f + slip
            pnl_net = pnl_gross - fee
            break

        matches.append(PolymarketMatch(
            event_title=title,
            window_start=window_start,
            window_end=window_end,
            poly_resolution=resolution,
            binance_direction=binance_direction,
            resolution_match=resolution_match,
            our_prediction=our_prediction,
            entry_minute=entry_minute,
            entry_price=entry_price,
            cum_ret=cum_ret_val,
            correct=correct,
            pnl_gross=pnl_gross,
            pnl_net=pnl_net,
            fee=fee,
            volume=volume,
        ))

    # ===== RESULTS =====
    print(f"\n  {'='*110}")
    print(f"  MATCHING RESULTS")
    print(f"  {'='*110}")
    print(f"  Total Polymarket markets: {len(poly_markets):,}")
    print(f"  Title parse failures: {parse_failures:,}")
    print(f"  No Binance data available: {no_candle_data:,}")
    print(f"  Insufficient candles (<11): {insufficient_candles:,}")
    print(f"  Successfully matched: {len(matches):,}")

    # ===== Resolution agreement =====
    matched_with_data = [m for m in matches if m.binance_direction is not None]
    resolution_agrees = sum(1 for m in matched_with_data if m.resolution_match)
    resolution_disagrees = len(matched_with_data) - resolution_agrees

    print(f"\n  {'='*110}")
    print(f"  BINANCE vs POLYMARKET RESOLUTION AGREEMENT")
    print(f"  {'='*110}")
    print(f"  Markets compared: {len(matched_with_data):,}")
    print(f"  Agreement: {resolution_agrees:,} ({resolution_agrees/len(matched_with_data)*100:.1f}%)")
    print(f"  Disagreement: {resolution_disagrees:,} ({resolution_disagrees/len(matched_with_data)*100:.1f}%)")
    if resolution_disagrees > 0:
        print(f"  NOTE: Disagreements may be due to timezone parsing or rounding at resolution time")

    # ===== Strategy performance =====
    traded = [m for m in matches if m.our_prediction is not None]
    not_traded = [m for m in matches if m.our_prediction is None]

    print(f"\n  {'='*110}")
    print(f"  STRATEGY PERFORMANCE (vs Polymarket resolution)")
    print(f"  {'='*110}")
    print(f"  Markets matched: {len(matches):,}")
    print(f"  Strategy traded: {len(traded):,} ({len(traded)/len(matches)*100:.1f}%)")
    print(f"  No trade (below threshold): {len(not_traded):,}")

    if traded:
        correct_count = sum(1 for t in traded if t.correct)
        accuracy = correct_count / len(traded)

        gross_pnls = [t.pnl_gross for t in traded if t.pnl_gross is not None]
        net_pnls = [t.pnl_net for t in traded if t.pnl_net is not None]
        fees = [t.fee for t in traded if t.fee is not None]

        gross_total = sum(gross_pnls)
        net_total = sum(net_pnls)
        total_fees = sum(fees)
        avg_gross = sum(gross_pnls) / len(gross_pnls) if gross_pnls else 0
        avg_net = sum(net_pnls) / len(net_pnls) if net_pnls else 0

        # Sharpe
        if net_pnls:
            mean_net = sum(p / POSITION_SIZE for p in net_pnls) / len(net_pnls)
            var_net = sum((p / POSITION_SIZE - mean_net) ** 2 for p in net_pnls) / len(net_pnls)
            std_net = math.sqrt(var_net) if var_net > 0 else 0.001
            sharpe = (mean_net / std_net) * math.sqrt(35040)
        else:
            sharpe = 0

        print(f"\n  Trades: {len(traded):,}")
        print(f"  Correct: {correct_count:,}")
        print(f"  Accuracy: {accuracy*100:.1f}%")
        print(f"\n  {'Metric':<25} {'Gross':>15} {'Net (after fees)':>15}")
        print(f"  {'-'*55}")
        print(f"  {'Avg P&L per trade':<25} ${avg_gross:>+13.2f} ${avg_net:>+13.2f}")
        print(f"  {'Total P&L':<25} ${gross_total:>+13,.0f} ${net_total:>+13,.0f}")
        print(f"  {'Total fees':<25} {'':>15} ${total_fees:>+13,.0f}")
        print(f"  {'Sharpe ratio':<25} {'':>15} {sharpe:>15.1f}")

        # Monthly breakdown
        print(f"\n  {'='*110}")
        print(f"  MONTHLY BREAKDOWN (Real Polymarket)")
        print(f"  {'='*110}")
        by_month: dict[str, list[PolymarketMatch]] = defaultdict(list)
        for t in traded:
            month = t.window_start.strftime("%Y-%m")
            by_month[month].append(t)

        print(f"  {'Month':<10} {'Trades':>7} {'Correct':>8} {'Acc%':>7} {'Gross':>12} {'Net':>12} {'Fees':>10}")
        print(f"  {'-'*70}")

        net_losing_months = 0
        for month in sorted(by_month):
            trades = by_month[month]
            n = len(trades)
            corr = sum(1 for t in trades if t.correct)
            acc = corr / n
            g = sum(t.pnl_gross for t in trades if t.pnl_gross is not None)
            nt = sum(t.pnl_net for t in trades if t.pnl_net is not None)
            f = sum(t.fee for t in trades if t.fee is not None)
            is_loss = nt < 0
            if is_loss:
                net_losing_months += 1
            marker = " *** LOSS" if is_loss else ""
            print(f"  {month:<10} {n:>7} {corr:>8} {acc*100:>6.1f}% "
                  f"${g:>+10,.0f} ${nt:>+10,.0f} ${f:>8,.0f}{marker}")

        print(f"\n  Net profitable months: {len(by_month) - net_losing_months}/{len(by_month)}")

        # Resolution disagreement impact
        traded_disagree = [t for t in traded if not t.resolution_match]
        if traded_disagree:
            disagree_correct = sum(1 for t in traded_disagree if t.correct)
            print(f"\n  {'='*110}")
            print(f"  IMPACT OF BINANCE/POLYMARKET DISAGREEMENTS")
            print(f"  {'='*110}")
            print(f"  Trades where Binance != Polymarket resolution: {len(traded_disagree)}")
            print(f"  Of those, our prediction was correct: {disagree_correct} ({disagree_correct/len(traded_disagree)*100:.1f}%)")
            # These are cases where our strategy would have been right on Binance but wrong on Polymarket (or vice versa)

        # By entry price
        print(f"\n  {'='*110}")
        print(f"  BY ENTRY PRICE")
        print(f"  {'='*110}")
        price_bins: dict[str, list[PolymarketMatch]] = defaultdict(list)
        for t in traded:
            if t.entry_price is not None:
                if t.entry_price < 0.60:
                    price_bins["<0.60"].append(t)
                elif t.entry_price < 0.70:
                    price_bins["0.60-0.70"].append(t)
                else:
                    price_bins[">=0.70"].append(t)

        print(f"  {'Price':<12} {'Trades':>7} {'Acc%':>7} {'Net PnL':>12}")
        print(f"  {'-'*40}")
        for label in ["<0.60", "0.60-0.70", ">=0.70"]:
            trades = price_bins.get(label, [])
            if trades:
                n = len(trades)
                acc = sum(1 for t in trades if t.correct) / n
                nt = sum(t.pnl_net for t in trades if t.pnl_net is not None)
                print(f"  {label:<12} {n:>7} {acc*100:>6.1f}% ${nt:>+10,.0f}")

    # Final verdict
    print(f"\n  {'='*110}")
    print(f"  VERDICT")
    print(f"  {'='*110}")
    if traded:
        if accuracy >= 0.85 and net_total > 0:
            print(f"  STRATEGY VALIDATED on real Polymarket data")
            print(f"  Accuracy {accuracy*100:.1f}% with net P&L ${net_total:+,.0f} after fees")
        elif accuracy >= 0.80 and net_total > 0:
            print(f"  STRATEGY VIABLE but edge is thinner on real data")
            print(f"  Accuracy {accuracy*100:.1f}% with net P&L ${net_total:+,.0f} after fees")
        else:
            print(f"  STRATEGY NEEDS REVIEW — accuracy {accuracy*100:.1f}%, net P&L ${net_total:+,.0f}")

    print(f"\n{'='*120}")


if __name__ == "__main__":
    main()
