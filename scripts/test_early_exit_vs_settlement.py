"""Early Exit vs Hold-to-Settlement comparison.

Key question: If we're already up 40-50%, should we take profit
to avoid a late reversal? Or does holding to settlement always win?

Tests a HYBRID approach:
  - If token price reaches +X% profit at any minute, EXIT EARLY
  - Otherwise, hold to settlement (binary $1.00 or $0.00)

Compares against pure hold-to-settlement.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtesting.data_loader import DataLoader

CSV_PATHS = [PROJECT_ROOT / "data" / f"btc_1m_chunk{i}.csv" for i in range(1, 7)]
MINUTES_PER_WINDOW = 15
POSITION_SIZE = 100.0
SENSITIVITY = 0.07


def _sigmoid(cum_ret: float) -> float:
    x = SENSITIVITY * cum_ret * 10000
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _group_into_15m_windows(candles: list[Any]) -> list[list[Any]]:
    if not candles:
        return []
    windows: list[list[Any]] = []
    current_window: list[Any] = []
    current_window_start: int | None = None
    for c in candles:
        ts: datetime = c.timestamp
        minute = ts.minute
        window_minute = (minute // 15) * 15
        if current_window_start is None:
            if minute == window_minute:
                current_window_start = window_minute
                current_window = [c]
        else:
            expected_minute = current_window_start
            offset = minute - expected_minute
            if offset < 0:
                offset += 60
            same_window = (
                len(current_window) < MINUTES_PER_WINDOW
                and 0 <= offset < MINUTES_PER_WINDOW
                and ts.hour == current_window[0].timestamp.hour
                and ts.date() == current_window[0].timestamp.date()
            )
            if same_window:
                current_window.append(c)
            else:
                if len(current_window) == MINUTES_PER_WINDOW:
                    windows.append(current_window)
                if minute == window_minute:
                    current_window = [c]
                    current_window_start = window_minute
                else:
                    current_window = []
                    current_window_start = None
    if len(current_window) == MINUTES_PER_WINDOW:
        windows.append(current_window)
    return windows


@dataclass
class TradeResult:
    window_start: str
    entry_minute: int
    direction: str
    entry_price: float
    # Hold-to-settlement outcome
    settlement: float
    settlement_pnl: float
    correct: bool
    # Early exit tracking
    max_profit_pct: float        # highest unrealized profit during hold
    max_profit_minute: int       # minute when max profit occurred
    early_exit_price: float      # price at early exit (if triggered)
    early_exit_minute: int       # minute of early exit (-1 if held to settle)
    early_exit_pnl: float        # P&L if exiting early
    # What actually happened with the hybrid strategy
    hybrid_pnl: float
    hybrid_exit_reason: str      # "early_tp" or "settlement"


def run_comparison(
    windows: list[list[Any]],
    *,
    threshold: float,
    entry_minutes: list[int],
    tp_pct: float,  # early exit take-profit threshold
) -> list[TradeResult]:
    results: list[TradeResult] = []

    for window in windows:
        window_open = float(window[0].open)
        if window_open == 0:
            continue

        final_close = float(window[-1].close)
        actual_green = final_close > window_open

        for entry_min in entry_minutes:
            if entry_min >= MINUTES_PER_WINDOW:
                continue

            current_close = float(window[entry_min].close)
            cum_ret = (current_close - window_open) / window_open

            if abs(cum_ret) < threshold:
                continue

            predict_green = cum_ret > 0
            entry_yes = _sigmoid(cum_ret)

            if predict_green:
                entry_price = entry_yes
            else:
                entry_price = 1.0 - entry_yes

            if entry_price <= 0.01:
                continue

            # Settlement outcome
            if predict_green:
                settlement = 1.0 if actual_green else 0.0
            else:
                settlement = 1.0 if not actual_green else 0.0

            correct = (predict_green == actual_green)
            settlement_pnl = (settlement - entry_price) / entry_price

            # Track token price minute-by-minute for early exit
            max_profit_pct = 0.0
            max_profit_minute = entry_min
            early_exit_price = -1.0
            early_exit_minute = -1
            early_exit_pnl = 0.0
            hybrid_pnl = settlement_pnl
            hybrid_exit_reason = "settlement"

            for m in range(entry_min + 1, MINUTES_PER_WINDOW):
                m_close = float(window[m].close)
                m_cum_ret = (m_close - window_open) / window_open
                m_yes = _sigmoid(m_cum_ret)

                if predict_green:
                    m_price = m_yes
                else:
                    m_price = 1.0 - m_yes

                m_pnl = (m_price - entry_price) / entry_price

                if m_pnl > max_profit_pct:
                    max_profit_pct = m_pnl
                    max_profit_minute = m

                # Check early exit
                if early_exit_minute == -1 and m_pnl >= tp_pct:
                    early_exit_price = m_price
                    early_exit_minute = m
                    early_exit_pnl = m_pnl
                    hybrid_pnl = m_pnl
                    hybrid_exit_reason = "early_tp"

            results.append(TradeResult(
                window_start=str(window[0].timestamp),
                entry_minute=entry_min,
                direction="YES" if predict_green else "NO",
                entry_price=round(entry_price, 6),
                settlement=settlement,
                settlement_pnl=round(settlement_pnl, 6),
                correct=correct,
                max_profit_pct=round(max_profit_pct, 6),
                max_profit_minute=max_profit_minute,
                early_exit_price=round(early_exit_price, 6) if early_exit_price >= 0 else -1,
                early_exit_minute=early_exit_minute,
                early_exit_pnl=round(early_exit_pnl, 6),
                hybrid_pnl=round(hybrid_pnl, 6),
                hybrid_exit_reason=hybrid_exit_reason,
            ))
            break

    return results


def print_comparison(results: list[TradeResult], name: str, tp_pct: float) -> None:
    n = len(results)
    if n == 0:
        return

    # Pure settlement stats
    settle_wins = sum(1 for r in results if r.settlement_pnl > 0)
    settle_pnls = [r.settlement_pnl for r in results]
    settle_avg = sum(settle_pnls) / n
    settle_total = sum(r.settlement_pnl * POSITION_SIZE for r in results)

    # Hybrid stats
    hybrid_wins = sum(1 for r in results if r.hybrid_pnl > 0)
    hybrid_pnls = [r.hybrid_pnl for r in results]
    hybrid_avg = sum(hybrid_pnls) / n
    hybrid_total = sum(r.hybrid_pnl * POSITION_SIZE for r in results)

    # How many trades hit the TP
    tp_triggered = sum(1 for r in results if r.hybrid_exit_reason == "early_tp")
    tp_pct_of_total = tp_triggered / n * 100

    # Of trades that HIT TP, how many would have been winners at settlement?
    tp_trades = [r for r in results if r.hybrid_exit_reason == "early_tp"]
    tp_would_win = sum(1 for r in tp_trades if r.correct)
    tp_would_lose = len(tp_trades) - tp_would_win

    # Of trades that DIDN'T hit TP, what's their settlement result?
    no_tp_trades = [r for r in results if r.hybrid_exit_reason == "settlement"]
    no_tp_wins = sum(1 for r in no_tp_trades if r.correct)

    # Trades that hit TP but would have LOST at settlement (saved by early exit)
    saved_by_tp = sum(1 for r in tp_trades if not r.correct)
    # Trades that hit TP but would have WON MORE at settlement (left on table)
    left_on_table = [r for r in tp_trades if r.correct]
    if left_on_table:
        avg_left = sum(r.settlement_pnl - r.early_exit_pnl for r in left_on_table) / len(left_on_table)
    else:
        avg_left = 0.0

    print(f"\n{'='*90}")
    print(f"  {name} — TP at +{tp_pct*100:.0f}%")
    print(f"{'='*90}")
    print(f"  Total trades: {n}")
    print()
    print(f"  {'Metric':<35} {'Settlement':>15} {'Hybrid':>15} {'Delta':>12}")
    print(f"  {'-'*78}")
    print(f"  {'Win Rate':<35} {settle_wins/n*100:>14.1f}% {hybrid_wins/n*100:>14.1f}% {(hybrid_wins-settle_wins)/n*100:>+11.1f}%")
    print(f"  {'Avg P&L per trade':<35} {settle_avg*100:>+14.2f}% {hybrid_avg*100:>+14.2f}% {(hybrid_avg-settle_avg)*100:>+11.2f}%")
    print(f"  {'Total P&L ($100/trade)':<35} ${settle_total:>+13,.0f} ${hybrid_total:>+13,.0f} ${hybrid_total-settle_total:>+10,.0f}")
    print()
    print(f"  Early exit breakdown:")
    print(f"    TP triggered: {tp_triggered} trades ({tp_pct_of_total:.1f}% of all trades)")
    print(f"    Of those {tp_triggered} early exits:")
    print(f"      Would have WON at settlement:  {tp_would_win} (took +{tp_pct*100:.0f}% instead of avg +{sum(r.settlement_pnl for r in left_on_table)/max(1,len(left_on_table))*100:.1f}%)")
    print(f"      Would have LOST at settlement: {tp_would_lose} (SAVED! took +{tp_pct*100:.0f}% instead of -100%)")
    print(f"    Avg profit LEFT on table (per winning TP trade): {avg_left*100:+.1f}%")
    print(f"    Trades held to settlement: {len(no_tp_trades)} ({no_tp_wins} wins / {len(no_tp_trades)-no_tp_wins} losses)")

    # The key insight: value of each saved trade vs cost of leaving profit on table
    if tp_trades:
        saved_value = saved_by_tp * (tp_pct + 1.0)  # gained tp_pct instead of losing 100%
        cost = sum(r.settlement_pnl - r.early_exit_pnl for r in left_on_table)
        print(f"\n  VALUE ANALYSIS:")
        print(f"    Saved from -100% loss: {saved_by_tp} trades × ${(tp_pct+1.0)*POSITION_SIZE:.0f} saved = ${saved_value*POSITION_SIZE:+,.0f}")
        print(f"    Left on table:         {len(left_on_table)} trades × avg ${avg_left*POSITION_SIZE:.1f} = ${cost*POSITION_SIZE:+,.0f}")
        print(f"    Net impact of early exit: ${(hybrid_total-settle_total):+,.0f}")


def main() -> None:
    print("=" * 90)
    print("  EARLY EXIT vs HOLD-TO-SETTLEMENT COMPARISON")
    print("  Should we take profit at +40-50% to avoid late reversals?")
    print("=" * 90)

    loader = DataLoader()
    all_candles: list[Any] = []
    for csv_path in CSV_PATHS:
        if not csv_path.exists():
            continue
        candles = loader.load_csv(csv_path)
        all_candles.extend(candles)

    all_candles.sort(key=lambda c: c.timestamp)
    seen: set[datetime] = set()
    unique = [c for c in all_candles if c.timestamp not in seen and not seen.add(c.timestamp)]  # type: ignore[func-returns-value]
    all_candles = unique
    print(f"\n  Loaded {len(all_candles)} candles")

    windows = _group_into_15m_windows(all_candles)
    print(f"  {len(windows)} 15m windows")

    # Test multiple entry configs × multiple TP levels
    entry_configs = [
        ("Min5 / 0.10%", 0.0010, [5]),
        ("Min5 / 0.05%", 0.0005, [5]),
        ("Min8 / 0.10%", 0.0010, [8]),
        ("Min6 / 0.10%", 0.0010, [6]),
    ]

    tp_levels = [0.25, 0.30, 0.35, 0.40, 0.50]

    # Summary table
    print(f"\n{'='*110}")
    print(f"  SUMMARY: Hybrid (TP + Settlement) vs Pure Settlement")
    print(f"{'='*110}")
    print(f"  {'Config':<20} {'TP%':>5} {'Trades':>7} {'Settle$':>12} {'Hybrid$':>12} "
          f"{'Delta$':>10} {'TP Hit%':>8} {'Saved':>6} {'Left$':>8}")
    print(f"  {'-'*100}")

    for cfg_name, thresh, entry_mins in entry_configs:
        for tp in tp_levels:
            results = run_comparison(windows, threshold=thresh, entry_minutes=entry_mins, tp_pct=tp)
            if not results:
                continue

            n = len(results)
            settle_total = sum(r.settlement_pnl * POSITION_SIZE for r in results)
            hybrid_total = sum(r.hybrid_pnl * POSITION_SIZE for r in results)
            delta = hybrid_total - settle_total

            tp_hit = sum(1 for r in results if r.hybrid_exit_reason == "early_tp")
            tp_pct_str = f"{tp_hit/n*100:.0f}%"
            saved = sum(1 for r in results if r.hybrid_exit_reason == "early_tp" and not r.correct)
            left = sum(
                (r.settlement_pnl - r.early_exit_pnl) * POSITION_SIZE
                for r in results if r.hybrid_exit_reason == "early_tp" and r.correct
            )

            marker = " <-- BETTER" if delta > 0 else ""
            print(f"  {cfg_name:<20} {tp*100:>4.0f}% {n:>7} ${settle_total:>+10,.0f} ${hybrid_total:>+10,.0f} "
                  f"${delta:>+9,.0f} {tp_pct_str:>8} {saved:>6} ${left:>+7,.0f}{marker}")

        print()  # blank line between configs

    # Detailed breakdown for the most interesting configs
    for cfg_name, thresh, entry_mins in [("Min5 / 0.05%", 0.0005, [5]), ("Min5 / 0.10%", 0.0010, [5])]:
        for tp in [0.30, 0.40, 0.50]:
            results = run_comparison(windows, threshold=thresh, entry_minutes=entry_mins, tp_pct=tp)
            print_comparison(results, cfg_name, tp)

    # Max unrealized profit analysis
    print(f"\n{'='*90}")
    print(f"  MAX UNREALIZED PROFIT ANALYSIS (Min5 / 0.10%)")
    print(f"  How high does the token price go before settlement?")
    print(f"{'='*90}")

    results = run_comparison(windows, threshold=0.0010, entry_minutes=[5], tp_pct=99.0)
    buckets = [
        ("0-10%", 0.0, 0.10),
        ("10-20%", 0.10, 0.20),
        ("20-30%", 0.20, 0.30),
        ("30-40%", 0.30, 0.40),
        ("40-50%", 0.40, 0.50),
        ("50-75%", 0.50, 0.75),
        ("75%+", 0.75, 99.0),
    ]

    print(f"\n  {'Max Profit Bucket':<20} {'Trades':>7} {'% Total':>8} {'Correct%':>9} {'Avg Settle PnL':>15}")
    print(f"  {'-'*62}")
    for bname, lo, hi in buckets:
        group = [r for r in results if lo <= r.max_profit_pct < hi]
        if group:
            pct = len(group) / len(results) * 100
            correct = sum(1 for r in group if r.correct) / len(group) * 100
            avg_pnl = sum(r.settlement_pnl for r in group) / len(group) * 100
            print(f"  {bname:<20} {len(group):>7} {pct:>7.1f}% {correct:>8.1f}% {avg_pnl:>+14.1f}%")

    print(f"\n{'='*90}")
    print(f"  Analysis complete.")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
