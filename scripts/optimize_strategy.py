"""Find THE optimal strategy configuration.

Tests tiered entries: different thresholds at different minutes to
maximize total P&L while maintaining high accuracy.
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
class Trade:
    entry_minute: int
    direction: str
    entry_price: float
    settlement: float
    pnl: float
    correct: bool
    cum_return: float


def run_tiered_strategy(
    windows: list[list[Any]],
    tiers: list[tuple[int, float]],  # [(minute, threshold), ...]
) -> list[Trade]:
    """Run a tiered entry strategy. First matching tier wins per window."""
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

            if predict_green:
                entry_price = entry_yes
            else:
                entry_price = 1.0 - entry_yes

            if entry_price <= 0.01:
                continue

            if predict_green:
                settlement = 1.0 if actual_green else 0.0
            else:
                settlement = 1.0 if not actual_green else 0.0

            correct = (predict_green == actual_green)
            pnl = (settlement - entry_price) / entry_price

            trades.append(Trade(
                entry_minute=entry_min,
                direction="YES" if predict_green else "NO",
                entry_price=round(entry_price, 6),
                settlement=settlement,
                pnl=round(pnl, 6),
                correct=correct,
                cum_return=round(cum_ret, 6),
            ))
            break  # one entry per window

    return trades


def score(trades: list[Trade]) -> dict[str, Any]:
    if not trades:
        return {"trades": 0, "accuracy": 0, "ev": 0, "total_pnl": 0, "sharpe": 0}
    n = len(trades)
    correct = sum(1 for t in trades if t.correct)
    pnls = [t.pnl for t in trades]
    total = sum(t.pnl * POSITION_SIZE for t in trades)
    ev = sum(pnls) / n
    mean = ev
    var = sum((p - mean) ** 2 for p in pnls) / n
    std = math.sqrt(var) if var > 0 else 0.001
    sharpe = (mean / std) * math.sqrt(35040)
    return {
        "trades": n,
        "accuracy": round(correct / n, 4),
        "ev": round(ev * 100, 2),
        "total_pnl": round(total, 0),
        "sharpe": round(sharpe, 1),
        "avg_entry": round(sum(t.entry_price for t in trades) / n, 4),
    }


def main() -> None:
    print("=" * 100)
    print("  STRATEGY OPTIMIZER — Find the best tiered entry config")
    print("=" * 100)

    loader = DataLoader()
    all_candles: list[Any] = []
    for csv_path in CSV_PATHS:
        if not csv_path.exists():
            continue
        all_candles.extend(loader.load_csv(csv_path))

    all_candles.sort(key=lambda c: c.timestamp)
    seen: set[datetime] = set()
    unique = [c for c in all_candles if c.timestamp not in seen and not seen.add(c.timestamp)]  # type: ignore[func-returns-value]
    all_candles = unique
    windows = _group_into_15m_windows(all_candles)
    print(f"  {len(all_candles)} candles, {len(windows)} windows\n")

    # ======================================================================
    # PART 1: Single-minute configs (baseline)
    # ======================================================================
    print("  PART 1: Single-minute baselines")
    print(f"  {'Config':<30} {'Trades':>7} {'Acc%':>7} {'EV/trade':>9} {'Total$':>12} {'Sharpe':>7} {'AvgEntry':>9}")
    print(f"  {'-'*83}")

    single_configs = [
        ("Min5 / 0.05%", [(5, 0.0005)]),
        ("Min5 / 0.10%", [(5, 0.0010)]),
        ("Min5 / 0.15%", [(5, 0.0015)]),
        ("Min6 / 0.10%", [(6, 0.0010)]),
        ("Min6 / 0.15%", [(6, 0.0015)]),
        ("Min7 / 0.10%", [(7, 0.0010)]),
        ("Min7 / 0.15%", [(7, 0.0015)]),
        ("Min8 / 0.10%", [(8, 0.0010)]),
        ("Min8 / 0.15%", [(8, 0.0015)]),
        ("Min10 / 0.10%", [(10, 0.0010)]),
        ("Min12 / 0.10%", [(12, 0.0010)]),
    ]

    for name, tiers in single_configs:
        trades = run_tiered_strategy(windows, tiers)
        s = score(trades)
        print(f"  {name:<30} {s['trades']:>7} {s['accuracy']*100:>6.1f}% ${s['ev']:>+7.2f} ${s['total_pnl']:>+10,.0f} {s['sharpe']:>7.1f} {s['avg_entry']:>9.4f}")

    # ======================================================================
    # PART 2: Tiered entry configs
    # ======================================================================
    print(f"\n  PART 2: Tiered entry strategies")
    print(f"  {'Config':<40} {'Trades':>7} {'Acc%':>7} {'EV/trade':>9} {'Total$':>12} {'Sharpe':>7}")
    print(f"  {'-'*85}")

    tiered_configs = [
        # Aggressive: low thresholds, early entry
        ("Aggressive: 5@0.05%",
         [(5, 0.0005)]),

        # Standard: cascade from min5 to min8
        ("Cascade: 5@0.15 > 6@0.10 > 7-8@0.10",
         [(5, 0.0015), (6, 0.0010), (7, 0.0010), (8, 0.0010)]),

        ("Cascade: 5@0.10 > 6-8@0.10",
         [(5, 0.0010), (6, 0.0010), (7, 0.0010), (8, 0.0010)]),

        ("Cascade: 5@0.10 > 6@0.10 > 7-8@0.05",
         [(5, 0.0010), (6, 0.0010), (7, 0.0005), (8, 0.0005)]),

        # Early aggressive + late conservative
        ("Early: 4@0.15 > 5@0.10 > 6-8@0.05",
         [(4, 0.0015), (5, 0.0010), (6, 0.0005), (7, 0.0005), (8, 0.0005)]),

        ("Wide: 3@0.20 > 4@0.15 > 5@0.10 > 6-8@0.05",
         [(3, 0.0020), (4, 0.0015), (5, 0.0010), (6, 0.0005), (7, 0.0005), (8, 0.0005)]),

        # Sweet spot: minutes 5-8 with cascading thresholds
        ("Tiered-A: 5@0.15 > 6@0.12 > 7@0.10 > 8@0.08",
         [(5, 0.0015), (6, 0.0012), (7, 0.0010), (8, 0.0008)]),

        ("Tiered-B: 5@0.12 > 6@0.10 > 7@0.08 > 8@0.05",
         [(5, 0.0012), (6, 0.0010), (7, 0.0008), (8, 0.0005)]),

        ("Tiered-C: 5@0.10 > 6@0.08 > 7@0.05 > 8@0.05",
         [(5, 0.0010), (6, 0.0008), (7, 0.0005), (8, 0.0005)]),

        # Extended range: include minute 3-4 for very strong signals
        ("Extended: 3@0.25 > 4@0.20 > 5@0.15 > 6-8@0.10",
         [(3, 0.0025), (4, 0.0020), (5, 0.0015), (6, 0.0010), (7, 0.0010), (8, 0.0010)]),

        ("Extended-B: 3@0.20 > 4@0.15 > 5@0.10 > 6-8@0.08",
         [(3, 0.0020), (4, 0.0015), (5, 0.0010), (6, 0.0008), (7, 0.0008), (8, 0.0008)]),

        # Late-only high accuracy
        ("Late: 8@0.10 > 9@0.08 > 10@0.05",
         [(8, 0.0010), (9, 0.0008), (10, 0.0005)]),

        ("Late-B: 7@0.10 > 8@0.08 > 9@0.05 > 10@0.05",
         [(7, 0.0010), (8, 0.0008), (9, 0.0005), (10, 0.0005)]),

        # Ultra-wide: maximize coverage
        ("Ultra: 3@0.20 > 4@0.12 > 5@0.08 > 6@0.05 > 7-10@0.05",
         [(3, 0.0020), (4, 0.0012), (5, 0.0008), (6, 0.0005), (7, 0.0005),
          (8, 0.0005), (9, 0.0005), (10, 0.0005)]),
    ]

    best_name = ""
    best_total = -999999.0

    for name, tiers in tiered_configs:
        trades = run_tiered_strategy(windows, tiers)
        s = score(trades)
        marker = ""
        if s["total_pnl"] > best_total:
            best_total = s["total_pnl"]
            best_name = name
            marker = " ***"
        print(f"  {name:<40} {s['trades']:>7} {s['accuracy']*100:>6.1f}% ${s['ev']:>+7.2f} ${s['total_pnl']:>+10,.0f} {s['sharpe']:>7.1f}{marker}")

    # ======================================================================
    # PART 3: Detailed breakdown of best strategy
    # ======================================================================
    # Re-run the top contenders for detailed analysis
    top_configs = [
        ("Ultra: 3@0.20 > 4@0.12 > 5@0.08 > 6-10@0.05",
         [(3, 0.0020), (4, 0.0012), (5, 0.0008), (6, 0.0005), (7, 0.0005),
          (8, 0.0005), (9, 0.0005), (10, 0.0005)]),
        ("Tiered-C: 5@0.10 > 6@0.08 > 7@0.05 > 8@0.05",
         [(5, 0.0010), (6, 0.0008), (7, 0.0005), (8, 0.0005)]),
        ("Min5 / 0.05% (baseline)",
         [(5, 0.0005)]),
    ]

    for name, tiers in top_configs:
        trades = run_tiered_strategy(windows, tiers)
        s = score(trades)

        print(f"\n{'='*90}")
        print(f"  DETAILED: {name}")
        print(f"{'='*90}")
        print(f"  Trades: {s['trades']}  |  Accuracy: {s['accuracy']*100:.1f}%  |  EV: ${s['ev']:+.2f}  |  Total: ${s['total_pnl']:+,.0f}  |  Sharpe: {s['sharpe']:.1f}")

        # By entry minute
        by_min: dict[int, list[Trade]] = {}
        for t in trades:
            by_min.setdefault(t.entry_minute, []).append(t)

        print(f"\n  {'Min':>5} {'Trades':>8} {'Acc%':>7} {'AvgEntry':>9} {'EV/trade':>9} {'Total$':>10}")
        print(f"  {'-'*52}")
        for m in sorted(by_min):
            group = by_min[m]
            g_n = len(group)
            g_acc = sum(1 for t in group if t.correct) / g_n
            g_ev = sum(t.pnl for t in group) / g_n
            g_total = sum(t.pnl * POSITION_SIZE for t in group)
            g_entry = sum(t.entry_price for t in group) / g_n
            print(f"  {m:>5} {g_n:>8} {g_acc*100:>6.1f}% {g_entry:>9.4f} ${g_ev*100:>+7.2f} ${g_total:>+9,.0f}")

        # Win/loss breakdown
        wins = [t for t in trades if t.correct]
        losses = [t for t in trades if not t.correct]
        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0

        print(f"\n  Wins: {len(wins)} ({len(wins)/len(trades)*100:.1f}%) avg +{avg_win*100:.1f}%")
        print(f"  Losses: {len(losses)} ({len(losses)/len(trades)*100:.1f}%) avg {avg_loss*100:.1f}%")

        # Equity curve stats
        equity = POSITION_SIZE
        peak = equity
        max_dd = 0.0
        max_consecutive_losses = 0
        current_streak = 0
        for t in trades:
            equity += t.pnl * POSITION_SIZE
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
            if t.pnl < 0:
                current_streak += 1
                max_consecutive_losses = max(max_consecutive_losses, current_streak)
            else:
                current_streak = 0

        print(f"  Max drawdown: {max_dd*100:.1f}%")
        print(f"  Max consecutive losses: {max_consecutive_losses}")
        print(f"  Final equity: ${equity:,.0f} (started at ${POSITION_SIZE:.0f})")

    print(f"\n{'='*90}")
    print(f"  WINNER: {best_name}")
    print(f"  Total P&L: ${best_total:+,.0f}")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
