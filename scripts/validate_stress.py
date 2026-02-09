"""Stress testing and edge case analysis over 2 years.

Tests: flash crashes, extreme vol, low vol, parameter sensitivity,
time-of-day, day-of-week, gap analysis, worst-case stretches.
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fast_loader import load_csv_fast, group_into_15m_windows

CSV_2Y = PROJECT_ROOT / "data" / "btc_1m_2y.csv"
MINUTES_PER_WINDOW = 15
POSITION_SIZE = 100.0
SENSITIVITY = 0.07
FIXED_TIERS = [(8, 0.0010), (9, 0.0008), (10, 0.0005)]


def _sigmoid(cum_ret: float) -> float:
    x = SENSITIVITY * cum_ret * 10000
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class Trade:
    window_start: str
    entry_minute: int
    direction: str
    entry_price: float
    settlement: float
    pnl: float
    pnl_dollar: float
    correct: bool
    cum_return: float
    window_range_pct: float  # (high-low)/open of the full 15m window


def run_strategy(windows: list[list[Any]], tiers: list[tuple[int, float]] | None = None) -> list[Trade]:
    if tiers is None:
        tiers = FIXED_TIERS
    trades: list[Trade] = []
    for window in windows:
        window_open = float(window[0].open)
        if window_open == 0:
            continue
        final_close = float(window[-1].close)
        actual_green = final_close > window_open

        # Window range
        window_high = max(float(c.high) for c in window)
        window_low = min(float(c.low) for c in window)
        window_range_pct = (window_high - window_low) / window_open

        for entry_min, threshold in tiers:
            if entry_min >= MINUTES_PER_WINDOW:
                continue
            current_close = float(window[entry_min].close)
            cum_ret = (current_close - window_open) / window_open
            if abs(cum_ret) < threshold:
                continue
            predict_green = cum_ret > 0
            entry_yes = _sigmoid(cum_ret)
            entry_price = entry_yes if predict_green else 1.0 - entry_yes
            if entry_price <= 0.01:
                continue
            settlement = (1.0 if actual_green else 0.0) if predict_green else (1.0 if not actual_green else 0.0)
            correct = predict_green == actual_green
            pnl = (settlement - entry_price) / entry_price
            trades.append(Trade(
                window_start=str(window[0].timestamp),
                entry_minute=entry_min,
                direction="YES" if predict_green else "NO",
                entry_price=round(entry_price, 6),
                settlement=settlement,
                pnl=round(pnl, 6),
                pnl_dollar=round(pnl * POSITION_SIZE, 2),
                correct=correct,
                cum_return=round(cum_ret, 6),
                window_range_pct=round(window_range_pct, 6),
            ))
            break
    return trades


def _print_group(label: str, trades: list[Trade]) -> None:
    if not trades:
        print(f"  {label:<25} no trades")
        return
    n = len(trades)
    acc = sum(1 for t in trades if t.correct) / n
    ev = sum(t.pnl for t in trades) / n * 100
    total = sum(t.pnl_dollar for t in trades)
    print(f"  {label:<25} {n:>6} trades  {acc*100:>6.1f}% acc  ${ev:>+6.2f} EV  ${total:>+9,.0f}")


def main() -> None:
    print("=" * 100)
    print("  STRESS TESTS & EDGE CASE ANALYSIS (2 Years)")
    print("=" * 100)

    if not CSV_2Y.exists():
        print(f"  ERROR: {CSV_2Y} not found.")
        sys.exit(1)

    all_candles = load_csv_fast(CSV_2Y)
    all_candles.sort(key=lambda c: c.timestamp)
    seen: set[datetime] = set()
    unique = [c for c in all_candles if c.timestamp not in seen and not seen.add(c.timestamp)]  # type: ignore
    all_candles = unique
    print(f"\n  Loaded {len(all_candles):,} candles")

    windows = group_into_15m_windows(all_candles)
    trades = run_strategy(windows)
    print(f"  {len(windows):,} windows, {len(trades):,} trades")

    # ========== 1. Flash crash / extreme move windows ==========
    print(f"\n  {'='*90}")
    print(f"  1. EXTREME MOVE ANALYSIS")
    print(f"  {'='*90}")

    big_moves = [t for t in trades if abs(t.cum_return) > 0.005]  # >0.5% move
    small_moves = [t for t in trades if abs(t.cum_return) <= 0.002]  # <0.2% move
    medium_moves = [t for t in trades if 0.002 < abs(t.cum_return) <= 0.005]

    _print_group("Big moves (>0.5%)", big_moves)
    _print_group("Medium moves (0.2-0.5%)", medium_moves)
    _print_group("Small moves (<0.2%)", small_moves)

    # Window range analysis
    print(f"\n  By 15m window range:")
    high_vol = [t for t in trades if t.window_range_pct > 0.01]  # >1% range
    med_vol = [t for t in trades if 0.005 <= t.window_range_pct <= 0.01]
    low_vol = [t for t in trades if t.window_range_pct < 0.005]

    _print_group("High vol windows (>1%)", high_vol)
    _print_group("Med vol windows (0.5-1%)", med_vol)
    _print_group("Low vol windows (<0.5%)", low_vol)

    # ========== 2. Consecutive loss analysis ==========
    print(f"\n  {'='*90}")
    print(f"  2. CONSECUTIVE LOSS ANALYSIS")
    print(f"  {'='*90}")

    max_streak = 0
    current_streak = 0
    worst_streak_end = 0
    for i, t in enumerate(trades):
        if not t.correct:
            current_streak += 1
            if current_streak > max_streak:
                max_streak = current_streak
                worst_streak_end = i
        else:
            current_streak = 0

    print(f"  Max consecutive losses: {max_streak}")
    if max_streak > 0:
        streak_start = worst_streak_end - max_streak + 1
        print(f"  Streak trades: #{streak_start} to #{worst_streak_end}")
        print(f"  First: {trades[streak_start].window_start}")
        print(f"  Last:  {trades[worst_streak_end].window_start}")
        streak_loss = sum(trades[j].pnl_dollar for j in range(streak_start, worst_streak_end + 1))
        print(f"  Streak loss: ${streak_loss:,.2f}")

    # Max drawdown over entire 2 years
    equity = [POSITION_SIZE]
    for t in trades:
        equity.append(equity[-1] + t.pnl_dollar)
    peak = equity[0]
    max_dd = 0.0
    max_dd_dollar = 0.0
    max_dd_idx = 0
    for i, val in enumerate(equity):
        if val > peak:
            peak = val
        dd = (peak - val) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
            max_dd_dollar = peak - val
            max_dd_idx = i

    print(f"\n  Max drawdown: {max_dd*100:.1f}% (${max_dd_dollar:,.0f})")
    print(f"  At trade #{max_dd_idx}")
    print(f"  Final equity: ${equity[-1]:,.0f}")

    # Worst 100-trade stretch
    worst_100_pnl = float("inf")
    worst_100_start = 0
    for i in range(len(trades) - 99):
        stretch_pnl = sum(trades[j].pnl_dollar for j in range(i, i + 100))
        if stretch_pnl < worst_100_pnl:
            worst_100_pnl = stretch_pnl
            worst_100_start = i

    worst_100_acc = sum(1 for j in range(worst_100_start, worst_100_start + 100) if trades[j].correct) / 100
    print(f"\n  Worst 100-trade stretch: ${worst_100_pnl:,.0f} ({worst_100_acc*100:.0f}% acc)")
    print(f"  Starting at: {trades[worst_100_start].window_start}")

    # ========== 3. Parameter sensitivity ==========
    print(f"\n  {'='*90}")
    print(f"  3. PARAMETER SENSITIVITY")
    print(f"  {'='*90}")

    sensitivity_configs = [
        ("Fixed (optimal)", FIXED_TIERS),
        ("Tighter: 8@0.12, 9@0.10, 10@0.08", [(8, 0.0012), (9, 0.0010), (10, 0.0008)]),
        ("Looser: 8@0.08, 9@0.06, 10@0.04", [(8, 0.0008), (9, 0.0006), (10, 0.0004)]),
        ("Earlier: 7@0.10, 8@0.08, 9@0.05", [(7, 0.0010), (8, 0.0008), (9, 0.0005)]),
        ("Later: 9@0.10, 10@0.08, 11@0.05", [(9, 0.0010), (10, 0.0008), (11, 0.0005)]),
        ("Single: 8@0.10 only", [(8, 0.0010)]),
        ("Single: 10@0.05 only", [(10, 0.0005)]),
        ("Wide: 7@0.12, 8@0.10, 9@0.08, 10@0.05", [(7, 0.0012), (8, 0.0010), (9, 0.0008), (10, 0.0005)]),
        ("Ultra-tight: 8@0.15, 9@0.12, 10@0.10", [(8, 0.0015), (9, 0.0012), (10, 0.0010)]),
    ]

    print(f"  {'Config':<40} {'Trades':>7} {'Acc%':>7} {'EV/100':>8} {'PnL$':>12} {'Sharpe':>7}")
    print(f"  {'-'*85}")

    for name, tiers in sensitivity_configs:
        t = run_strategy(windows, tiers)
        if t:
            n = len(t)
            acc = sum(1 for x in t if x.correct) / n
            ev = sum(x.pnl for x in t) / n * 100
            total = sum(x.pnl_dollar for x in t)
            pnls = [x.pnl for x in t]
            mean = sum(pnls) / n
            var = sum((p - mean) ** 2 for p in pnls) / n
            std = math.sqrt(var) if var > 0 else 0.001
            sharpe = (mean / std) * math.sqrt(35040)
            print(f"  {name:<40} {n:>7} {acc*100:>6.1f}% ${ev:>+6.2f} ${total:>+11,.0f} {sharpe:>6.1f}")

    # ========== 4. Time-of-day analysis ==========
    print(f"\n  {'='*90}")
    print(f"  4. TIME-OF-DAY ANALYSIS (UTC)")
    print(f"  {'='*90}")

    by_hour: dict[int, list[Trade]] = defaultdict(list)
    for t in trades:
        hour = int(t.window_start[11:13]) if len(t.window_start) > 13 else 0
        by_hour[hour].append(t)

    print(f"  {'Hour':>6} {'Trades':>7} {'Acc%':>7} {'EV/100':>8} {'PnL$':>10}")
    print(f"  {'-'*42}")
    for hour in range(24):
        group = by_hour.get(hour, [])
        if group:
            n = len(group)
            acc = sum(1 for t in group if t.correct) / n
            ev = sum(t.pnl for t in group) / n * 100
            total = sum(t.pnl_dollar for t in group)
            marker = " ***" if acc < 0.80 else ""
            print(f"  {hour:>6} {n:>7} {acc*100:>6.1f}% ${ev:>+6.2f} ${total:>+9,.0f}{marker}")

    # ========== 5. Day-of-week analysis ==========
    print(f"\n  {'='*90}")
    print(f"  5. DAY-OF-WEEK ANALYSIS")
    print(f"  {'='*90}")

    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    by_dow: dict[int, list[Trade]] = defaultdict(list)
    for t in trades:
        try:
            dt = datetime.fromisoformat(t.window_start.replace("Z", "+00:00") if "Z" in t.window_start else t.window_start)
            by_dow[dt.weekday()].append(t)
        except ValueError:
            pass

    print(f"  {'Day':>6} {'Trades':>7} {'Acc%':>7} {'EV/100':>8} {'PnL$':>10}")
    print(f"  {'-'*42}")
    for dow in range(7):
        group = by_dow.get(dow, [])
        if group:
            n = len(group)
            acc = sum(1 for t in group if t.correct) / n
            ev = sum(t.pnl for t in group) / n * 100
            total = sum(t.pnl_dollar for t in group)
            marker = " ***" if acc < 0.80 else ""
            print(f"  {days[dow]:>6} {n:>7} {acc*100:>6.1f}% ${ev:>+6.2f} ${total:>+9,.0f}{marker}")

    # ========== 6. Gap analysis ==========
    print(f"\n  {'='*90}")
    print(f"  6. GAP ANALYSIS (window open vs previous close)")
    print(f"  {'='*90}")

    gap_trades: dict[str, list[Trade]] = {"big_gap": [], "small_gap": [], "no_gap": []}
    for i, w in enumerate(windows):
        if i == 0:
            continue
        prev_close = float(windows[i - 1][-1].close)
        curr_open = float(w[0].open)
        if prev_close > 0:
            gap_pct = abs(curr_open - prev_close) / prev_close
        else:
            gap_pct = 0

        # Find matching trade
        ws = str(w[0].timestamp)
        matching = [t for t in trades if t.window_start == ws]
        for t in matching:
            if gap_pct > 0.002:
                gap_trades["big_gap"].append(t)
            elif gap_pct > 0.0005:
                gap_trades["small_gap"].append(t)
            else:
                gap_trades["no_gap"].append(t)

    _print_group("Big gap (>0.2%)", gap_trades["big_gap"])
    _print_group("Small gap (0.05-0.2%)", gap_trades["small_gap"])
    _print_group("No gap (<0.05%)", gap_trades["no_gap"])

    # ========== Summary ==========
    print(f"\n  {'='*90}")
    print(f"  STRESS TEST SUMMARY")
    print(f"  {'='*90}")
    overall_acc = sum(1 for t in trades if t.correct) / len(trades)
    overall_ev = sum(t.pnl for t in trades) / len(trades) * 100
    print(f"  Total trades (2 years): {len(trades):,}")
    print(f"  Overall accuracy: {overall_acc*100:.1f}%")
    print(f"  Overall EV: ${overall_ev:+.2f}")
    print(f"  Max consecutive losses: {max_streak}")
    print(f"  Max drawdown: {max_dd*100:.1f}%")
    print(f"  Worst 100-trade stretch: ${worst_100_pnl:,.0f}")
    print(f"  Final equity (from $100): ${equity[-1]:,.0f}")

    # Flag any concerning results
    concerns = []
    if overall_acc < 0.80:
        concerns.append(f"Overall accuracy {overall_acc*100:.1f}% < 80%")
    if max_streak > 10:
        concerns.append(f"Max consecutive losses {max_streak} > 10")
    if max_dd > 0.50:
        concerns.append(f"Max drawdown {max_dd*100:.1f}% > 50%")
    bad_hours = [h for h, g in by_hour.items() if len(g) > 20 and sum(1 for t in g if t.correct)/len(g) < 0.75]
    if bad_hours:
        concerns.append(f"Low accuracy hours: {bad_hours}")

    if concerns:
        print(f"\n  CONCERNS:")
        for c in concerns:
            print(f"    - {c}")
    else:
        print(f"\n  No major concerns identified.")

    print(f"\n{'='*100}")


if __name__ == "__main__":
    main()
