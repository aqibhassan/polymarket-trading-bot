"""Walk-forward validation: train/test split to detect overfitting.

Rolling windows: 3-month train / 1-month test, sliding 1 month.
Tests both optimized-per-window and fixed configs.
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

# Grid for optimization
OPT_MINUTES = [7, 8, 9, 10, 11]
OPT_THRESHOLDS = [0.0003, 0.0005, 0.0008, 0.0010, 0.0012, 0.0015]


def _sigmoid(cum_ret: float) -> float:
    x = SENSITIVITY * cum_ret * 10000
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class Trade:
    entry_minute: int
    pnl: float
    correct: bool


def run_tiers(windows: list[list[Any]], tiers: list[tuple[int, float]]) -> list[Trade]:
    trades: list[Trade] = []
    for window in windows:
        window_open = float(window[0].open)
        if window_open == 0:
            continue
        final_close = float(window[-1].close)
        actual_green = final_close > window_open

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
            trades.append(Trade(entry_minute=entry_min, pnl=round(pnl, 6), correct=correct))
            break
    return trades


def score(trades: list[Trade]) -> dict[str, float]:
    if not trades:
        return {"n": 0, "acc": 0, "ev": 0, "total": 0, "sharpe": 0}
    n = len(trades)
    correct = sum(1 for t in trades if t.correct)
    pnls = [t.pnl for t in trades]
    mean = sum(pnls) / n
    total = sum(t.pnl * POSITION_SIZE for t in trades)
    var = sum((p - mean) ** 2 for p in pnls) / n
    std = math.sqrt(var) if var > 0 else 0.001
    sharpe = (mean / std) * math.sqrt(35040)
    return {"n": n, "acc": round(correct / n, 4), "ev": round(mean * 100, 2),
            "total": round(total, 0), "sharpe": round(sharpe, 1)}


def optimize_tiers(windows: list[list[Any]]) -> list[tuple[int, float]]:
    """Find the best 3-tier config on training data."""
    best_total = -999999.0
    best_tiers: list[tuple[int, float]] = FIXED_TIERS

    # Test all 3-tier combos where minutes are ascending
    for m1 in OPT_MINUTES:
        for t1 in OPT_THRESHOLDS:
            for m2 in OPT_MINUTES:
                if m2 <= m1:
                    continue
                for t2 in OPT_THRESHOLDS:
                    if t2 >= t1:
                        continue  # later minutes should have lower thresholds
                    for m3 in OPT_MINUTES:
                        if m3 <= m2:
                            continue
                        for t3 in OPT_THRESHOLDS:
                            if t3 >= t2:
                                continue
                            tiers = [(m1, t1), (m2, t2), (m3, t3)]
                            trades = run_tiers(windows, tiers)
                            if not trades:
                                continue
                            s = score(trades)
                            if s["total"] > best_total and s["acc"] > 0.80:
                                best_total = s["total"]
                                best_tiers = tiers

    return best_tiers


def main() -> None:
    print("=" * 110)
    print("  WALK-FORWARD VALIDATION (2 Years)")
    print("  3-month train / 1-month test, rolling monthly")
    print("=" * 110)

    if not CSV_2Y.exists():
        print(f"  ERROR: {CSV_2Y} not found.")
        sys.exit(1)

    all_candles = load_csv_fast(CSV_2Y)
    all_candles.sort(key=lambda c: c.timestamp)
    seen: set[datetime] = set()
    unique = [c for c in all_candles if c.timestamp not in seen and not seen.add(c.timestamp)]  # type: ignore
    all_candles = unique
    print(f"\n  Loaded {len(all_candles):,} candles")

    # Group by month
    by_month: dict[str, list[Any]] = defaultdict(list)
    for c in all_candles:
        by_month[c.timestamp.strftime("%Y-%m")].append(c)

    months = sorted(by_month.keys())
    print(f"  {len(months)} months: {months[0]} to {months[-1]}")

    # Walk-forward: 3-month train, 1-month test
    print(f"\n  PART 1: FIXED CONFIG (8@0.10, 9@0.08, 10@0.05) — no optimization")
    print(f"  {'Test Month':<12} {'Windows':>8} {'Trades':>7} {'Acc%':>7} {'EV/100':>8} {'PnL$':>10} {'Sharpe':>7}")
    print(f"  {'-'*65}")

    fixed_oos_results: list[dict[str, float]] = []
    for i in range(len(months)):
        test_month = months[i]
        test_candles = by_month[test_month]
        test_windows = group_into_15m_windows(test_candles)
        trades = run_tiers(test_windows, FIXED_TIERS)
        s = score(trades)
        fixed_oos_results.append(s)
        if s["n"] > 0:
            print(f"  {test_month:<12} {len(test_windows):>8} {s['n']:>7} {s['acc']*100:>6.1f}% "
                  f"${s['ev']:>+6.2f} ${s['total']:>+9,.0f} {s['sharpe']:>6.1f}")

    # Walk-forward with optimization
    print(f"\n  PART 2: OPTIMIZED PER WINDOW (3-month train → 1-month test)")
    print(f"  {'Test Month':<12} {'Train':>12} {'OptTiers':>30} {'IS Acc':>7} {'OOS Acc':>8} {'OOS EV':>8} {'OOS PnL':>10} {'Gap':>6}")
    print(f"  {'-'*100}")

    opt_oos_results: list[dict[str, float]] = []
    degradation_total = 0.0
    degradation_count = 0

    for i in range(3, len(months)):
        test_month = months[i]
        train_months = months[i - 3 : i]
        train_label = f"{train_months[0]}..{train_months[-1]}"

        # Train data
        train_candles: list[Any] = []
        for m in train_months:
            train_candles.extend(by_month[m])
        train_windows = group_into_15m_windows(train_candles)

        # Optimize on train
        opt_tiers = optimize_tiers(train_windows)
        is_trades = run_tiers(train_windows, opt_tiers)
        is_score = score(is_trades)

        # Test on OOS
        test_candles = by_month[test_month]
        test_windows = group_into_15m_windows(test_candles)
        oos_trades = run_tiers(test_windows, opt_tiers)
        oos_score = score(oos_trades)
        opt_oos_results.append(oos_score)

        tier_str = " > ".join(f"{m}@{t*100:.2f}%" for m, t in opt_tiers)

        gap = 0.0
        if is_score["n"] > 0 and oos_score["n"] > 0:
            gap = (oos_score["acc"] - is_score["acc"]) * 100
            degradation_total += gap
            degradation_count += 1

        print(f"  {test_month:<12} {train_label:>12} {tier_str:>30} "
              f"{is_score['acc']*100:>6.1f}% {oos_score['acc']*100:>7.1f}% "
              f"${oos_score['ev']:>+6.2f} ${oos_score['total']:>+9,.0f} {gap:>+5.1f}%")

    # Summary
    print(f"\n  {'='*100}")
    print(f"  SUMMARY")
    print(f"  {'='*100}")

    # Fixed config summary
    fixed_accs = [r["acc"] for r in fixed_oos_results if r["n"] > 0]
    fixed_evs = [r["ev"] for r in fixed_oos_results if r["n"] > 0]
    fixed_totals = [r["total"] for r in fixed_oos_results if r["n"] > 0]
    if fixed_accs:
        print(f"\n  FIXED CONFIG across all months:")
        print(f"    Avg accuracy: {sum(fixed_accs)/len(fixed_accs)*100:.1f}%")
        print(f"    Accuracy range: {min(fixed_accs)*100:.1f}% - {max(fixed_accs)*100:.1f}%")
        print(f"    Avg EV: ${sum(fixed_evs)/len(fixed_evs):+.2f}")
        print(f"    Total P&L: ${sum(fixed_totals):+,.0f}")
        print(f"    Profitable months: {sum(1 for t in fixed_totals if t > 0)}/{len(fixed_totals)}")

    # Optimized config summary
    opt_accs = [r["acc"] for r in opt_oos_results if r["n"] > 0]
    opt_totals = [r["total"] for r in opt_oos_results if r["n"] > 0]
    if opt_accs:
        print(f"\n  OPTIMIZED CONFIG (OOS):")
        print(f"    Avg OOS accuracy: {sum(opt_accs)/len(opt_accs)*100:.1f}%")
        print(f"    OOS accuracy range: {min(opt_accs)*100:.1f}% - {max(opt_accs)*100:.1f}%")
        print(f"    Total OOS P&L: ${sum(opt_totals):+,.0f}")

    if degradation_count > 0:
        avg_deg = degradation_total / degradation_count
        print(f"\n  OVERFITTING ANALYSIS:")
        print(f"    Avg IS→OOS accuracy gap: {avg_deg:+.1f}%")
        if avg_deg < -5:
            print(f"    WARNING: Significant overfitting detected (gap > 5%)")
        elif avg_deg < -2:
            print(f"    CAUTION: Mild overfitting detected (gap 2-5%)")
        else:
            print(f"    OK: No significant overfitting (gap < 2%)")

    print(f"\n{'='*110}")


if __name__ == "__main__":
    main()
