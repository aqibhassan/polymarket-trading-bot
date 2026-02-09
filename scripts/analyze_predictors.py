"""Analyze predictive features for 15m candle color prediction.

Loads 1m BTC data, groups into 15m windows, and evaluates which features
at minutes 3-8 best predict the final 15m candle color (green/red).
"""

from __future__ import annotations

import sys
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from itertools import combinations
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtesting.data_loader import DataLoader

CSV_PATHS = [PROJECT_ROOT / "data" / f"btc_1m_chunk{i}.csv" for i in range(1, 7)]
MINUTES_PER_WINDOW = 15


def group_into_15m_windows(candles: list[Any]) -> list[list[Any]]:
    """Group 1m candles into 15m windows aligned to :00, :15, :30, :45."""
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


def compute_features(window: list[Any], minute_idx: int) -> dict[str, Any]:
    """Compute all features at a given minute index within a 15m window."""
    window_open = float(window[0].open)
    current_close = float(window[minute_idx].close)

    # a) Cumulative return from window open
    cum_return = (current_close - window_open) / window_open if window_open != 0 else 0.0

    # b) Cumulative return direction
    cum_return_direction = "green" if cum_return > 0 else ("red" if cum_return < 0 else "neutral")

    # c) Last 3 candles direction
    if minute_idx >= 2:
        last_3_dirs = []
        for i in range(minute_idx - 2, minute_idx + 1):
            c = window[i]
            if float(c.close) > float(c.open):
                last_3_dirs.append("green")
            elif float(c.close) < float(c.open):
                last_3_dirs.append("red")
            else:
                last_3_dirs.append("neutral")
        last_3_same = len(set(last_3_dirs)) == 1 and last_3_dirs[0] != "neutral"
        last_3_direction = last_3_dirs[0] if last_3_same else None
    else:
        last_3_same = False
        last_3_direction = None

    # d) Volume trend (comparing last 3 candles' volumes)
    if minute_idx >= 2:
        vols = [float(window[i].volume) for i in range(minute_idx - 2, minute_idx + 1)]
        volume_increasing = vols[2] > vols[1] > vols[0]
        volume_decreasing = vols[2] < vols[1] < vols[0]
        if volume_increasing:
            volume_trend = "increasing"
        elif volume_decreasing:
            volume_trend = "decreasing"
        else:
            volume_trend = "mixed"
    else:
        volume_trend = "mixed"

    # e) Reversal detection: was there a move >0.1% then reversal?
    reversal_detected = False
    reversal_retrace_pct = 0.0
    initial_move_direction = None

    # Track max/min cum return up to this point
    max_cum_ret = 0.0
    min_cum_ret = 0.0
    max_idx = 0
    min_idx = 0
    for i in range(1, minute_idx + 1):
        cr = (float(window[i].close) - window_open) / window_open if window_open != 0 else 0.0
        if cr > max_cum_ret:
            max_cum_ret = cr
            max_idx = i
        if cr < min_cum_ret:
            min_cum_ret = cr
            min_idx = i

    # Check if there was a significant move that then reversed
    if max_cum_ret > 0.001:  # 0.1% up move
        # Check if current return is below the max (reversal from up)
        retrace = max_cum_ret - cum_return
        if retrace > max_cum_ret * 0.3 and max_idx < minute_idx:
            reversal_detected = True
            initial_move_direction = "up"
            reversal_retrace_pct = retrace / max_cum_ret if max_cum_ret != 0 else 0.0

    if abs(min_cum_ret) > 0.001:  # 0.1% down move
        retrace_down = cum_return - min_cum_ret
        if retrace_down > abs(min_cum_ret) * 0.3 and min_idx < minute_idx:
            if not reversal_detected or abs(min_cum_ret) > max_cum_ret:
                reversal_detected = True
                initial_move_direction = "down"
                reversal_retrace_pct = retrace_down / abs(min_cum_ret) if min_cum_ret != 0 else 0.0

    # f) Move magnitude
    move_magnitude = abs(cum_return)

    # g) Candle shrinking (exhaustion) - are consecutive 1m candle bodies getting smaller?
    candle_shrinking = False
    if minute_idx >= 2:
        bodies = []
        for i in range(minute_idx - 2, minute_idx + 1):
            bodies.append(abs(float(window[i].close) - float(window[i].open)))
        candle_shrinking = bodies[2] < bodies[1] < bodies[0] and bodies[0] > 0

    return {
        "cum_return": cum_return,
        "cum_return_direction": cum_return_direction,
        "last_3_same": last_3_same,
        "last_3_direction": last_3_direction,
        "volume_trend": volume_trend,
        "reversal_detected": reversal_detected,
        "initial_move_direction": initial_move_direction,
        "reversal_retrace_pct": reversal_retrace_pct,
        "move_magnitude": move_magnitude,
        "candle_shrinking": candle_shrinking,
    }


def get_final_color(window: list[Any]) -> str:
    """Determine the final 15m candle color."""
    window_open = float(window[0].open)
    window_close = float(window[-1].close)
    if window_close > window_open:
        return "green"
    elif window_close < window_open:
        return "red"
    return "neutral"


def main() -> None:
    print("=" * 90)
    print("  MVHE Predictor Analysis — 15m Candle Color Prediction")
    print("=" * 90)

    # 1. Load all chunks
    loader = DataLoader()
    all_candles: list[Any] = []
    for csv_path in CSV_PATHS:
        if not csv_path.exists():
            print(f"  [SKIP] {csv_path.name} not found")
            continue
        candles = loader.load_csv(csv_path)
        print(f"  Loaded {len(candles):>6} candles from {csv_path.name}")
        all_candles.extend(candles)

    if not all_candles:
        print("ERROR: No data loaded.")
        sys.exit(1)

    # Sort and deduplicate
    all_candles.sort(key=lambda c: c.timestamp)
    seen_ts: set[datetime] = set()
    unique: list[Any] = []
    for c in all_candles:
        if c.timestamp not in seen_ts:
            seen_ts.add(c.timestamp)
            unique.append(c)
    all_candles = unique

    print(f"\n  Total 1m candles: {len(all_candles)}")
    print(f"  Date range: {all_candles[0].timestamp} -> {all_candles[-1].timestamp}")

    # 2. Group into 15m windows
    windows = group_into_15m_windows(all_candles)
    print(f"  15m windows: {len(windows)}")

    # Filter out neutral windows
    non_neutral = [(w, get_final_color(w)) for w in windows if get_final_color(w) != "neutral"]
    print(f"  Non-neutral windows: {len(non_neutral)}")

    green_count = sum(1 for _, c in non_neutral if c == "green")
    red_count = sum(1 for _, c in non_neutral if c == "red")
    print(f"  Green: {green_count} ({green_count/len(non_neutral)*100:.1f}%), Red: {red_count} ({red_count/len(non_neutral)*100:.1f}%)")

    # 3. Analyze individual features at each minute mark
    check_minutes = [3, 4, 5, 6, 7, 8]
    cum_return_thresholds = [0.0005, 0.0010, 0.0015, 0.0020, 0.0025]  # 0.05% to 0.25%

    print("\n" + "=" * 90)
    print("  SECTION 1: Individual Feature Analysis")
    print("=" * 90)

    for minute in check_minutes:
        print(f"\n  --- Minute {minute} ---")
        print(f"  {'Feature':<40} {'Accuracy':>10} {'Coverage':>10} {'Samples':>10}")
        print(f"  {'-'*70}")

        # Feature 1: cum_return direction matches final color
        for thresh in cum_return_thresholds:
            correct = 0
            total_signal = 0
            for window, final_color in non_neutral:
                feat = compute_features(window, minute)
                if feat["move_magnitude"] >= thresh:
                    total_signal += 1
                    predicted = feat["cum_return_direction"]
                    if predicted == final_color:
                        correct += 1
            if total_signal > 0:
                acc = correct / total_signal * 100
                cov = total_signal / len(non_neutral) * 100
                print(f"  cum_return > {thresh*100:.2f}%{'':<22} {acc:>9.1f}% {cov:>9.1f}% {total_signal:>10}")

        # Feature 2: last_3_same direction matches final color
        correct = 0
        total_signal = 0
        for window, final_color in non_neutral:
            feat = compute_features(window, minute)
            if feat["last_3_same"] and feat["last_3_direction"] is not None:
                total_signal += 1
                if feat["last_3_direction"] == final_color:
                    correct += 1
        if total_signal > 0:
            acc = correct / total_signal * 100
            cov = total_signal / len(non_neutral) * 100
            print(f"  last_3_candles_same_dir                  {acc:>9.1f}% {cov:>9.1f}% {total_signal:>10}")

        # Feature 3: volume_increasing + cum_return direction
        for trend_type in ["increasing", "decreasing"]:
            correct = 0
            total_signal = 0
            for window, final_color in non_neutral:
                feat = compute_features(window, minute)
                if feat["volume_trend"] == trend_type and feat["move_magnitude"] > 0.0005:
                    total_signal += 1
                    if feat["cum_return_direction"] == final_color:
                        correct += 1
            if total_signal > 0:
                acc = correct / total_signal * 100
                cov = total_signal / len(non_neutral) * 100
                print(f"  vol_{trend_type} + cum_ret>0.05%{'':<{12 - len(trend_type)}} {acc:>9.1f}% {cov:>9.1f}% {total_signal:>10}")

        # Feature 4: reversal_detected
        correct = 0
        total_signal = 0
        for window, final_color in non_neutral:
            feat = compute_features(window, minute)
            if feat["reversal_detected"]:
                total_signal += 1
                # If initial move was up and reversal, predict red; vice versa
                predicted = "red" if feat["initial_move_direction"] == "up" else "green"
                if predicted == final_color:
                    correct += 1
        if total_signal > 0:
            acc = correct / total_signal * 100
            cov = total_signal / len(non_neutral) * 100
            print(f"  reversal_detected (contrarian)            {acc:>9.1f}% {cov:>9.1f}% {total_signal:>10}")

        # Feature 5: candle_shrinking + direction
        correct = 0
        total_signal = 0
        for window, final_color in non_neutral:
            feat = compute_features(window, minute)
            if feat["candle_shrinking"] and feat["move_magnitude"] > 0.0005:
                total_signal += 1
                # Exhaustion suggests reversal
                predicted = "red" if feat["cum_return_direction"] == "green" else "green"
                if predicted == final_color:
                    correct += 1
        if total_signal > 0:
            acc = correct / total_signal * 100
            cov = total_signal / len(non_neutral) * 100
            print(f"  candle_shrinking (contrarian)             {acc:>9.1f}% {cov:>9.1f}% {total_signal:>10}")

        # Feature 5b: candle_shrinking used as momentum confirmation (same direction)
        correct = 0
        total_signal = 0
        for window, final_color in non_neutral:
            feat = compute_features(window, minute)
            if feat["candle_shrinking"] and feat["move_magnitude"] > 0.0005:
                total_signal += 1
                predicted = feat["cum_return_direction"]
                if predicted == final_color:
                    correct += 1
        if total_signal > 0:
            acc = correct / total_signal * 100
            cov = total_signal / len(non_neutral) * 100
            print(f"  candle_shrinking (momentum)               {acc:>9.1f}% {cov:>9.1f}% {total_signal:>10}")

    # 4. COMBINATION analysis
    print("\n" + "=" * 90)
    print("  SECTION 2: Feature Combination Analysis (Target: 80%+ accuracy)")
    print("=" * 90)

    best_combos: list[dict[str, Any]] = []

    for minute in check_minutes:
        print(f"\n  === Minute {minute} ===")
        print(f"  {'Combo':<55} {'Acc':>7} {'Cov':>7} {'N':>7}")
        print(f"  {'-'*76}")

        # Pre-compute all features for this minute
        window_features: list[tuple[dict[str, Any], str]] = []
        for window, final_color in non_neutral:
            feat = compute_features(window, minute)
            window_features.append((feat, final_color))

        # Combo 1: cum_return threshold + momentum (follow direction)
        for thresh in cum_return_thresholds:
            correct = 0
            total = 0
            for feat, final_color in window_features:
                if feat["move_magnitude"] >= thresh:
                    total += 1
                    if feat["cum_return_direction"] == final_color:
                        correct += 1
            if total >= 20:
                acc = correct / total * 100
                cov = total / len(non_neutral) * 100
                label = f"cum_ret > {thresh*100:.2f}% (momentum)"
                print(f"  {label:<55} {acc:>6.1f}% {cov:>6.1f}% {total:>7}")
                if acc >= 70:
                    best_combos.append({"minute": minute, "combo": label, "accuracy": acc, "coverage": cov, "samples": total})

        # Combo 2: cum_return + last_3_same (both agree on direction)
        for thresh in cum_return_thresholds:
            correct = 0
            total = 0
            for feat, final_color in window_features:
                if (feat["move_magnitude"] >= thresh
                    and feat["last_3_same"]
                    and feat["last_3_direction"] == feat["cum_return_direction"]):
                    total += 1
                    if feat["cum_return_direction"] == final_color:
                        correct += 1
            if total >= 20:
                acc = correct / total * 100
                cov = total / len(non_neutral) * 100
                label = f"cum_ret > {thresh*100:.2f}% + last3_agree"
                print(f"  {label:<55} {acc:>6.1f}% {cov:>6.1f}% {total:>7}")
                if acc >= 70:
                    best_combos.append({"minute": minute, "combo": label, "accuracy": acc, "coverage": cov, "samples": total})

        # Combo 3: cum_return + volume increasing
        for thresh in cum_return_thresholds:
            correct = 0
            total = 0
            for feat, final_color in window_features:
                if feat["move_magnitude"] >= thresh and feat["volume_trend"] == "increasing":
                    total += 1
                    if feat["cum_return_direction"] == final_color:
                        correct += 1
            if total >= 20:
                acc = correct / total * 100
                cov = total / len(non_neutral) * 100
                label = f"cum_ret > {thresh*100:.2f}% + vol_increasing"
                print(f"  {label:<55} {acc:>6.1f}% {cov:>6.1f}% {total:>7}")
                if acc >= 70:
                    best_combos.append({"minute": minute, "combo": label, "accuracy": acc, "coverage": cov, "samples": total})

        # Combo 4: cum_return + candle_shrinking (exhaustion confirmation -> momentum)
        for thresh in cum_return_thresholds:
            correct = 0
            total = 0
            for feat, final_color in window_features:
                if feat["move_magnitude"] >= thresh and feat["candle_shrinking"]:
                    total += 1
                    if feat["cum_return_direction"] == final_color:
                        correct += 1
            if total >= 20:
                acc = correct / total * 100
                cov = total / len(non_neutral) * 100
                label = f"cum_ret > {thresh*100:.2f}% + candle_shrinking (momentum)"
                print(f"  {label:<55} {acc:>6.1f}% {cov:>6.1f}% {total:>7}")
                if acc >= 70:
                    best_combos.append({"minute": minute, "combo": label, "accuracy": acc, "coverage": cov, "samples": total})

        # Combo 5: cum_return + candle_shrinking (exhaustion -> reversal)
        for thresh in cum_return_thresholds:
            correct = 0
            total = 0
            for feat, final_color in window_features:
                if feat["move_magnitude"] >= thresh and feat["candle_shrinking"]:
                    total += 1
                    predicted = "red" if feat["cum_return_direction"] == "green" else "green"
                    if predicted == final_color:
                        correct += 1
            if total >= 20:
                acc = correct / total * 100
                cov = total / len(non_neutral) * 100
                label = f"cum_ret > {thresh*100:.2f}% + candle_shrinking (reversal)"
                print(f"  {label:<55} {acc:>6.1f}% {cov:>6.1f}% {total:>7}")
                if acc >= 70:
                    best_combos.append({"minute": minute, "combo": label, "accuracy": acc, "coverage": cov, "samples": total})

        # Combo 6: reversal_detected + retrace > 50%
        for retrace_thresh in [0.3, 0.5, 0.7, 0.9]:
            correct = 0
            total = 0
            for feat, final_color in window_features:
                if feat["reversal_detected"] and feat["reversal_retrace_pct"] >= retrace_thresh:
                    total += 1
                    predicted = "red" if feat["initial_move_direction"] == "up" else "green"
                    if predicted == final_color:
                        correct += 1
            if total >= 20:
                acc = correct / total * 100
                cov = total / len(non_neutral) * 100
                label = f"reversal + retrace > {retrace_thresh*100:.0f}%"
                print(f"  {label:<55} {acc:>6.1f}% {cov:>6.1f}% {total:>7}")
                if acc >= 70:
                    best_combos.append({"minute": minute, "combo": label, "accuracy": acc, "coverage": cov, "samples": total})

        # Combo 7: cum_return + last_3 + volume
        for thresh in cum_return_thresholds:
            correct = 0
            total = 0
            for feat, final_color in window_features:
                if (feat["move_magnitude"] >= thresh
                    and feat["last_3_same"]
                    and feat["last_3_direction"] == feat["cum_return_direction"]
                    and feat["volume_trend"] == "increasing"):
                    total += 1
                    if feat["cum_return_direction"] == final_color:
                        correct += 1
            if total >= 10:
                acc = correct / total * 100
                cov = total / len(non_neutral) * 100
                label = f"cum_ret > {thresh*100:.2f}% + last3 + vol_inc"
                print(f"  {label:<55} {acc:>6.1f}% {cov:>6.1f}% {total:>7}")
                if acc >= 70:
                    best_combos.append({"minute": minute, "combo": label, "accuracy": acc, "coverage": cov, "samples": total})

        # Combo 8: cum_return + no reversal (clean trend)
        for thresh in cum_return_thresholds:
            correct = 0
            total = 0
            for feat, final_color in window_features:
                if feat["move_magnitude"] >= thresh and not feat["reversal_detected"]:
                    total += 1
                    if feat["cum_return_direction"] == final_color:
                        correct += 1
            if total >= 20:
                acc = correct / total * 100
                cov = total / len(non_neutral) * 100
                label = f"cum_ret > {thresh*100:.2f}% + no_reversal (clean)"
                print(f"  {label:<55} {acc:>6.1f}% {cov:>6.1f}% {total:>7}")
                if acc >= 70:
                    best_combos.append({"minute": minute, "combo": label, "accuracy": acc, "coverage": cov, "samples": total})

        # Combo 9: large move + last3 agree + no reversal
        for thresh in [0.0015, 0.0020, 0.0025]:
            correct = 0
            total = 0
            for feat, final_color in window_features:
                if (feat["move_magnitude"] >= thresh
                    and feat["last_3_same"]
                    and feat["last_3_direction"] == feat["cum_return_direction"]
                    and not feat["reversal_detected"]):
                    total += 1
                    if feat["cum_return_direction"] == final_color:
                        correct += 1
            if total >= 10:
                acc = correct / total * 100
                cov = total / len(non_neutral) * 100
                label = f"cum_ret > {thresh*100:.2f}% + last3 + no_rev"
                print(f"  {label:<55} {acc:>6.1f}% {cov:>6.1f}% {total:>7}")
                if acc >= 70:
                    best_combos.append({"minute": minute, "combo": label, "accuracy": acc, "coverage": cov, "samples": total})

        # Combo 10: reversal + exhaustion (shrinking candles during reversal)
        correct = 0
        total = 0
        for feat, final_color in window_features:
            if feat["reversal_detected"] and feat["candle_shrinking"]:
                total += 1
                predicted = "red" if feat["initial_move_direction"] == "up" else "green"
                if predicted == final_color:
                    correct += 1
        if total >= 10:
            acc = correct / total * 100
            cov = total / len(non_neutral) * 100
            label = "reversal + candle_shrinking"
            print(f"  {label:<55} {acc:>6.1f}% {cov:>6.1f}% {total:>7}")
            if acc >= 70:
                best_combos.append({"minute": minute, "combo": label, "accuracy": acc, "coverage": cov, "samples": total})

    # 5. Summary of best combos
    print("\n" + "=" * 90)
    print("  SECTION 3: Best Feature Combinations (sorted by accuracy)")
    print("=" * 90)

    best_combos.sort(key=lambda x: (-x["accuracy"], -x["samples"]))

    print(f"\n  {'Min':>3} {'Combo':<55} {'Acc':>7} {'Cov':>7} {'N':>7}")
    print(f"  {'-'*79}")
    for combo in best_combos[:40]:
        marker = " ***" if combo["accuracy"] >= 80 else ""
        print(f"  {combo['minute']:>3} {combo['combo']:<55} {combo['accuracy']:>6.1f}% {combo['coverage']:>6.1f}% {combo['samples']:>7}{marker}")

    # 6. Deep dive on combos with 80%+ accuracy
    high_acc = [c for c in best_combos if c["accuracy"] >= 80]
    print(f"\n  Combos with 80%+ accuracy: {len(high_acc)}")

    if not high_acc:
        print("  No combos reached 80%. Showing top 75%+ combos:")
        high_acc = [c for c in best_combos if c["accuracy"] >= 75]
        print(f"  Combos with 75%+ accuracy: {len(high_acc)}")

    if not high_acc:
        print("  No combos reached 75%. Showing top 70%+ combos:")
        high_acc = [c for c in best_combos if c["accuracy"] >= 70]
        print(f"  Combos with 70%+ accuracy: {len(high_acc)}")

    # 7. Additional analysis: by direction (green vs red separately)
    print("\n" + "=" * 90)
    print("  SECTION 4: Direction-Specific Analysis (top combos)")
    print("=" * 90)

    for minute in [5, 6, 7, 8]:
        print(f"\n  === Minute {minute} — Direction Split ===")
        print(f"  {'Combo':<45} {'Green Acc':>10} {'Red Acc':>10} {'G_N':>6} {'R_N':>6}")
        print(f"  {'-'*77}")

        window_features = []
        for window, final_color in non_neutral:
            feat = compute_features(window, minute)
            window_features.append((feat, final_color))

        for thresh in cum_return_thresholds:
            # Green predictions
            g_correct = 0
            g_total = 0
            r_correct = 0
            r_total = 0
            for feat, final_color in window_features:
                if feat["move_magnitude"] >= thresh:
                    if feat["cum_return_direction"] == "green":
                        g_total += 1
                        if final_color == "green":
                            g_correct += 1
                    elif feat["cum_return_direction"] == "red":
                        r_total += 1
                        if final_color == "red":
                            r_correct += 1
            if g_total >= 20 and r_total >= 20:
                g_acc = g_correct / g_total * 100
                r_acc = r_correct / r_total * 100
                label = f"cum_ret > {thresh*100:.2f}%"
                print(f"  {label:<45} {g_acc:>9.1f}% {r_acc:>9.1f}% {g_total:>6} {r_total:>6}")

    # 8. Time progression analysis
    print("\n" + "=" * 90)
    print("  SECTION 5: Accuracy Progression Over Time (cum_ret thresholds)")
    print("=" * 90)
    print(f"\n  {'Threshold':<15}", end="")
    for m in check_minutes:
        print(f"  Min{m:>2}", end="")
    print("    (accuracy% @ coverage%)")

    for thresh in cum_return_thresholds:
        print(f"  {thresh*100:.2f}%{'':<10}", end="")
        for minute in check_minutes:
            correct = 0
            total = 0
            for window, final_color in non_neutral:
                feat = compute_features(window, minute)
                if feat["move_magnitude"] >= thresh:
                    total += 1
                    if feat["cum_return_direction"] == final_color:
                        correct += 1
            if total > 0:
                acc = correct / total * 100
                cov = total / len(non_neutral) * 100
                print(f"  {acc:.0f}@{cov:.0f}", end="")
            else:
                print(f"  {'N/A':>6}", end="")
        print()

    print("\n" + "=" * 90)
    print("  Analysis complete.")
    print("=" * 90)


if __name__ == "__main__":
    main()
