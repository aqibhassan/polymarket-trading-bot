"""Directional accuracy analysis — the REAL win rate metric.

The key question: when we predict "BTC 15m candle will close GREEN" at minute 5-8,
how often is the final candle actually green?

This is the TRUE edge — Polymarket pays ~$1.00 for correct direction, ~$0.00 for wrong.
If we buy YES at $0.60 and direction is correct, we profit $0.40 (67% return).
If wrong, we lose $0.60 (100% loss).

For 80%+ profitability: 0.80 * 0.40 - 0.20 * 0.60 = +0.20 per dollar risked.
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
SENSITIVITY = 0.07


def _sigmoid(cum_ret: float, sensitivity: float = SENSITIVITY) -> float:
    x = sensitivity * cum_ret * 10000
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
class DirectionalPrediction:
    window_start: str
    entry_minute: int
    cum_return_at_entry: float
    predicted_direction: str  # "green" or "red"
    actual_direction: str     # "green" or "red"
    correct: bool
    entry_yes_price: float    # sigmoid price at entry
    final_yes_price: float    # sigmoid price at settlement (minute 14)
    settlement_price: float   # 1.0 if green, 0.0 if red
    pnl_hold_to_settlement: float  # P&L if held to full settlement


def analyze_windows(
    windows: list[list[Any]],
    threshold: float,
    entry_minutes: list[int],
    require_last3: bool = False,
    require_no_reversal: bool = False,
) -> list[DirectionalPrediction]:
    """Analyze directional accuracy for each qualifying window."""
    predictions: list[DirectionalPrediction] = []

    for window in windows:
        window_open = float(window[0].open)
        if window_open == 0:
            continue

        # Determine actual final direction
        final_close = float(window[-1].close)
        actual_direction = "green" if final_close > window_open else "red"

        for entry_min in entry_minutes:
            if entry_min >= MINUTES_PER_WINDOW:
                continue

            current_close = float(window[entry_min].close)
            cum_ret = (current_close - window_open) / window_open

            if abs(cum_ret) < threshold:
                continue

            # Optional filters
            if require_last3:
                if entry_min < 2:
                    continue
                cum_dir = "green" if cum_ret > 0 else "red"
                ok = True
                for i in range(entry_min - 2, entry_min + 1):
                    c = window[i]
                    o, cl = float(c.open), float(c.close)
                    if cum_dir == "green" and cl <= o:
                        ok = False
                        break
                    if cum_dir == "red" and cl >= o:
                        ok = False
                        break
                if not ok:
                    continue

            if require_no_reversal:
                max_cr, min_cr = 0.0, 0.0
                max_idx, min_idx = 0, 0
                for i in range(1, entry_min + 1):
                    cr = (float(window[i].close) - window_open) / window_open
                    if cr > max_cr:
                        max_cr = cr
                        max_idx = i
                    if cr < min_cr:
                        min_cr = cr
                        min_idx = i
                skip = False
                if max_cr > 0.001:
                    retrace = max_cr - cum_ret
                    if retrace > max_cr * 0.3 and max_idx < entry_min:
                        skip = True
                if not skip and abs(min_cr) > 0.001:
                    retrace_down = cum_ret - min_cr
                    if retrace_down > abs(min_cr) * 0.3 and min_idx < entry_min:
                        skip = True
                if skip:
                    continue

            # Predicted direction = follow momentum
            predicted_direction = "green" if cum_ret > 0 else "red"
            correct = predicted_direction == actual_direction

            # Token prices
            entry_yes_price = _sigmoid(cum_ret)
            final_cum_ret = (final_close - window_open) / window_open
            final_yes_price = _sigmoid(final_cum_ret)

            # Settlement: YES pays 1.0 if green, 0.0 if red
            settlement_price = 1.0 if actual_direction == "green" else 0.0

            # P&L if we bought YES when predicting green, NO when predicting red
            if predicted_direction == "green":
                entry_price = entry_yes_price
                exit_price = settlement_price  # 1.0 if correct, 0.0 if wrong
            else:
                entry_price = 1.0 - entry_yes_price
                exit_price = 1.0 - settlement_price  # 1.0 if correct, 0.0 if wrong

            pnl = (exit_price - entry_price) / entry_price if entry_price > 0 else 0.0

            predictions.append(DirectionalPrediction(
                window_start=str(window[0].timestamp),
                entry_minute=entry_min,
                cum_return_at_entry=round(cum_ret, 6),
                predicted_direction=predicted_direction,
                actual_direction=actual_direction,
                correct=correct,
                entry_yes_price=round(entry_yes_price, 4),
                final_yes_price=round(final_yes_price, 4),
                settlement_price=settlement_price,
                pnl_hold_to_settlement=round(pnl, 4),
            ))
            break  # one trade per window

    return predictions


def main() -> None:
    print("=" * 90)
    print("  DIRECTIONAL ACCURACY ANALYSIS")
    print("  Does BTC momentum at minute 5-8 predict the final 15m candle direction?")
    print("=" * 90)

    loader = DataLoader()
    all_candles: list[Any] = []
    for csv_path in CSV_PATHS:
        if not csv_path.exists():
            continue
        candles = loader.load_csv(csv_path)
        print(f"  Loaded {len(candles):>6} candles from {csv_path.name}")
        all_candles.extend(candles)

    all_candles.sort(key=lambda c: c.timestamp)
    seen: set[datetime] = set()
    unique = []
    for c in all_candles:
        if c.timestamp not in seen:
            seen.add(c.timestamp)
            unique.append(c)
    all_candles = unique

    print(f"\n  Total 1m candles: {len(all_candles)}")
    windows = _group_into_15m_windows(all_candles)
    print(f"  15m windows: {len(windows)}")

    # Run multiple threshold/entry combos
    configs = [
        ("Min5 | 0.05%", 0.0005, [5], False, False),
        ("Min5 | 0.10%", 0.0010, [5], False, False),
        ("Min5 | 0.15%", 0.0015, [5], False, False),
        ("Min5 | 0.20%", 0.0020, [5], False, False),
        ("Min6 | 0.10%", 0.0010, [6], False, False),
        ("Min6 | 0.15%", 0.0015, [6], False, False),
        ("Min7 | 0.10%", 0.0010, [7], False, False),
        ("Min7 | 0.15%", 0.0015, [7], False, False),
        ("Min8 | 0.10%", 0.0010, [8], False, False),
        ("Min8 | 0.15%", 0.0015, [8], False, False),
        ("Min8 | 0.20%", 0.0020, [8], False, False),
        ("Min10 | 0.10%", 0.0010, [10], False, False),
        ("Min10 | 0.15%", 0.0015, [10], False, False),
        ("Min10 | 0.20%", 0.0020, [10], False, False),
        ("Min12 | 0.10%", 0.0010, [12], False, False),
        ("Min12 | 0.15%", 0.0015, [12], False, False),
        ("Min5-8 | 0.10%", 0.0010, [5, 6, 7, 8], False, False),
        ("Min5-8 | 0.15%", 0.0015, [5, 6, 7, 8], False, False),
        ("Min5-8 | 0.10% +filt", 0.0010, [5, 6, 7, 8], True, True),
    ]

    print(f"\n  {'Config':<25} {'Trades':>7} {'Correct':>8} {'Accuracy':>9} "
          f"{'AvgEntry':>9} {'AvgPnL%':>9} {'EV/trade':>9}")
    print(f"  {'-'*80}")

    for name, thresh, entry_mins, l3, nr in configs:
        preds = analyze_windows(windows, thresh, entry_mins, l3, nr)
        if not preds:
            print(f"  {name:<25} {'N/A':>7}")
            continue

        n = len(preds)
        correct = sum(1 for p in preds if p.correct)
        accuracy = correct / n
        avg_entry = sum(p.entry_yes_price for p in preds if p.predicted_direction == "green") / max(1, sum(1 for p in preds if p.predicted_direction == "green"))
        avg_pnl = sum(p.pnl_hold_to_settlement for p in preds) / n
        ev_per_trade = avg_pnl * 100  # as dollars per $100

        print(f"  {name:<25} {n:>7} {correct:>8} {accuracy*100:>8.1f}% "
              f"{avg_entry:>9.4f} {avg_pnl*100:>+8.2f}% ${ev_per_trade:>+7.2f}")

    # Detailed analysis for best config
    print("\n" + "=" * 90)
    print("  DETAILED ANALYSIS — Min8 | 0.20% threshold")
    print("=" * 90)

    preds = analyze_windows(windows, 0.0020, [8], False, False)
    if preds:
        correct = sum(1 for p in preds if p.correct)
        wrong = len(preds) - correct
        wins = [p for p in preds if p.pnl_hold_to_settlement > 0]
        losses = [p for p in preds if p.pnl_hold_to_settlement <= 0]

        print(f"  Total predictions: {len(preds)}")
        print(f"  Correct direction: {correct} ({correct/len(preds)*100:.1f}%)")
        print(f"  Wrong direction:   {wrong} ({wrong/len(preds)*100:.1f}%)")
        print()

        if wins:
            avg_win = sum(p.pnl_hold_to_settlement for p in wins) / len(wins)
            print(f"  Winning trades:    {len(wins)}")
            print(f"  Avg win P&L:       {avg_win*100:+.2f}%")
        if losses:
            avg_loss = sum(p.pnl_hold_to_settlement for p in losses) / len(losses)
            print(f"  Losing trades:     {len(losses)}")
            print(f"  Avg loss P&L:      {avg_loss*100:+.2f}%")

        avg_entry_prices = {}
        for d in ["green", "red"]:
            group = [p for p in preds if p.predicted_direction == d]
            if group:
                if d == "green":
                    avg_entry_prices[d] = sum(p.entry_yes_price for p in group) / len(group)
                else:
                    avg_entry_prices[d] = sum(1.0 - p.entry_yes_price for p in group) / len(group)

        print(f"\n  Avg YES entry price (green predictions): {avg_entry_prices.get('green', 0):.4f}")
        print(f"  Avg NO entry price (red predictions):    {avg_entry_prices.get('red', 0):.4f}")

        # By |cum_return| bucket
        print(f"\n  By |cum_return| bucket:")
        print(f"  {'Bucket':<15} {'Trades':>7} {'Accuracy':>9} {'AvgPnL%':>9} {'AvgEntry':>9}")
        print(f"  {'-'*50}")
        buckets = [
            ("0.20-0.30%", 0.002, 0.003),
            ("0.30-0.50%", 0.003, 0.005),
            ("0.50-1.00%", 0.005, 0.010),
            ("1.00%+", 0.010, 999.0),
        ]
        for bname, lo, hi in buckets:
            group = [p for p in preds if lo <= abs(p.cum_return_at_entry) < hi]
            if group:
                acc = sum(1 for p in group if p.correct) / len(group)
                avg_pnl = sum(p.pnl_hold_to_settlement for p in group) / len(group)
                avg_ep = sum(
                    p.entry_yes_price if p.predicted_direction == "green" else 1.0 - p.entry_yes_price
                    for p in group
                ) / len(group)
                print(f"  {bname:<15} {len(group):>7} {acc*100:>8.1f}% {avg_pnl*100:>+8.2f}% {avg_ep:>9.4f}")

    # Expected value calculation for different accuracy levels
    print("\n" + "=" * 90)
    print("  EXPECTED VALUE TABLE (per $100 invested)")
    print("  Assumes: buy at entry_price, settle at $1.00 (correct) or $0.00 (wrong)")
    print("=" * 90)
    print(f"  {'Entry Price':>12} {'Break-Even Acc':>15} {'@50%':>8} {'@55%':>8} {'@60%':>8} "
          f"{'@65%':>8} {'@70%':>8} {'@75%':>8} {'@80%':>8}")
    print(f"  {'-'*100}")

    for entry_p in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        breakeven = entry_p  # at entry_p accuracy, EV = 0
        evs = []
        for acc in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            win_pnl = (1.0 - entry_p) / entry_p  # return on winning trade
            loss_pnl = -1.0  # lose entire position
            ev = acc * win_pnl + (1 - acc) * loss_pnl
            evs.append(ev * 100)
        print(f"  ${entry_p:.2f}          {breakeven*100:>13.0f}% "
              f"{''.join(f'{e:>+8.1f}' for e in evs)}")

    print("\n" + "=" * 90)
    print("  Analysis complete.")
    print("=" * 90)


if __name__ == "__main__":
    main()
