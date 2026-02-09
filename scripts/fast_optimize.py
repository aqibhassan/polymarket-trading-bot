#!/usr/bin/env python3
"""Fast vectorized singularity parameter optimization.

Instead of running the full strategy with all signal analyzers (slow),
this directly computes momentum signals + binary settlement outcomes
over all 15m windows, then sweeps parameters with numpy vectorization.

Usage:
    python scripts/fast_optimize.py
    python scripts/fast_optimize.py --top 30
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Data loading (lightweight — no Pydantic overhead)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class MiniCandle:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def load_1m_fast(path: Path) -> list[MiniCandle]:
    """Load 1m candles as lightweight dataclass (10x faster than Pydantic)."""
    candles: list[MiniCandle] = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_ts = row["timestamp"].strip()
            if raw_ts.endswith("Z"):
                raw_ts = raw_ts[:-1] + "+00:00"
            ts = datetime.fromisoformat(raw_ts)
            candles.append(MiniCandle(
                ts=ts,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            ))
    candles.sort(key=lambda c: c.ts)
    return candles


def group_windows(candles: list[MiniCandle]) -> list[list[MiniCandle]]:
    windows: list[list[MiniCandle]] = []
    current: list[MiniCandle] = []
    for c in candles:
        if c.ts.minute % 15 == 0 and current:
            windows.append(current)
            current = []
        current.append(c)
    if current:
        windows.append(current)
    return windows


# ---------------------------------------------------------------------------
# Pre-compute per-window data vectors
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class WindowData:
    """Pre-computed data for one 15m window."""
    idx: int
    open_price: float
    close_price: float
    true_dir: int           # 1 = up, -1 = down
    hour_utc: int
    # Per-minute cum returns (up to 15 values)
    cum_returns_pct: list[float]   # |cum_return| * 100 at each minute
    cum_directions: list[int]       # 1 = up, -1 = down
    n_minutes: int
    # Confirmation: last-3-candle agreement at each minute
    last3_agree: list[bool]


def precompute_windows(windows: list[list[MiniCandle]]) -> list[WindowData]:
    """Vectorize all window data upfront."""
    data: list[WindowData] = []
    for idx, window in enumerate(windows):
        if len(window) < 5:
            continue
        w_open = window[0].open
        w_close = window[-1].close
        if w_open == 0:
            continue

        true_dir = 1 if w_close >= w_open else -1
        hour_utc = window[0].ts.hour

        cum_returns_pct: list[float] = []
        cum_directions: list[int] = []
        last3_agree: list[bool] = []

        for i, candle in enumerate(window):
            cr = (candle.close - w_open) / w_open
            cum_returns_pct.append(abs(cr) * 100.0)
            cum_directions.append(1 if cr >= 0 else -1)

            # Last-3 agreement
            if i >= 3:
                closes = [window[j].close for j in range(i - 2, i + 1)]
                opens = [window[j].open for j in range(i - 2, i + 1)]
                changes = [1 if c >= o else -1 for c, o in zip(closes, opens)]
                agree = all(d == changes[0] for d in changes) and changes[0] == cum_directions[-1]
                last3_agree.append(agree)
            else:
                last3_agree.append(False)

        data.append(WindowData(
            idx=idx,
            open_price=w_open,
            close_price=w_close,
            true_dir=true_dir,
            hour_utc=hour_utc,
            cum_returns_pct=cum_returns_pct,
            cum_directions=cum_directions,
            n_minutes=len(window),
            last3_agree=last3_agree,
        ))
    return data


# ---------------------------------------------------------------------------
# Hour-of-day multipliers (from time_of_day.py)
# ---------------------------------------------------------------------------

HOUR_MULTIPLIERS = {
    0: 0.95, 1: 0.68, 2: 0.80, 3: 0.85, 4: 0.90, 5: 1.10,
    6: 1.05, 7: 1.05, 8: 0.95, 9: 1.10, 10: 1.15, 11: 1.10,
    12: 0.90, 13: 0.80, 14: 1.25, 15: 1.15, 16: 1.10, 17: 1.05,
    18: 0.95, 19: 0.90, 20: 0.88, 21: 0.90, 22: 0.85, 23: 0.92,
}

HOUR_WIN_RATES = {
    0: 0.87, 1: 0.83, 2: 0.84, 3: 0.85, 4: 0.86, 5: 0.88,
    6: 0.87, 7: 0.87, 8: 0.86, 9: 0.88, 10: 0.89, 11: 0.88,
    12: 0.86, 13: 0.84, 14: 0.89, 15: 0.88, 16: 0.87, 17: 0.87,
    18: 0.86, 19: 0.85, 20: 0.85, 21: 0.86, 22: 0.85, 23: 0.87,
}


# ---------------------------------------------------------------------------
# Fast parameter sweep
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    params: dict[str, Any]
    total_trades: int
    wins: int
    win_rate: float
    net_pnl: float
    sharpe: float
    max_drawdown: float
    profit_factor: float
    avg_pnl: float


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x / 0.07))


def polymarket_fee(pos_size: float, yes_price: float) -> float:
    p = yes_price
    return pos_size * 0.25 * p * p * (1 - p) * (1 - p)


def run_trial(
    windows_data: list[WindowData],
    entry_start: int,
    entry_end: int,
    tier_thresholds: dict[int, float],
    min_confidence: float,
    min_signals: int,
    weight_momentum: float,
    use_time_filter: bool,
    use_last3_bonus: bool,
    initial_balance: float = 10000.0,
    position_pct: float = 0.02,
    no_compound: bool = False,
) -> TrialResult:
    """Run a single trial over pre-computed window data.

    Updated to match corrected ensemble logic:
    - Time-of-day is a neutral meta-signal (boosts confidence, not directional count)
    - effective_min has floor of 2 (always require multi-source confirmation)
    - cum_return == 0 skipped (no directional signal)
    """
    balance = initial_balance
    trades = 0
    wins = 0
    gross_profit = 0.0
    gross_loss = 0.0
    total_fees = 0.0
    equity_curve: list[float] = [balance]

    for wd in windows_data:
        traded = False
        for minute in range(entry_start, min(entry_end + 1, wd.n_minutes)):
            if traded:
                break

            threshold = tier_thresholds.get(minute)
            if threshold is None:
                continue

            cum_pct = wd.cum_returns_pct[minute]
            direction = wd.cum_directions[minute]

            if cum_pct < threshold:
                continue

            # Skip exactly flat (cum_return == 0) — no directional signal
            if cum_pct == 0.0:
                continue

            # Count DIRECTIONAL signals only (neutral votes excluded)
            n_directional = 0
            # Total signal strength includes neutral boost for confidence
            total_signal_strength = 0.0
            neutral_strength = 0.0

            # Signal 1: Momentum (always available, directional)
            momentum_strength = min(cum_pct / 0.25, 1.0)
            n_directional += 1
            total_signal_strength += momentum_strength * weight_momentum

            # Signal 2: OFI proxy (synthetic — matches direction in our backtest)
            ofi_strength = min(cum_pct / 0.15, 1.0)
            n_directional += 1
            total_signal_strength += ofi_strength * (1 - weight_momentum) * 0.4

            # Signal 3: Time-of-day — NEUTRAL meta-signal
            # Boosts confidence for majority direction but does NOT count
            # toward directional agreement (matching corrected ensemble logic)
            if use_time_filter:
                hour_mult = HOUR_MULTIPLIERS.get(wd.hour_utc, 1.0)
                if hour_mult >= 0.75:
                    time_strength = min(hour_mult / 1.25, 1.0)
                    # Add to strength but NOT to n_directional
                    neutral_strength += time_strength * (1 - weight_momentum) * 0.3

            # Signal 4: Last-3 agreement bonus (directional confirmation)
            if use_last3_bonus and minute < len(wd.last3_agree) and wd.last3_agree[minute]:
                n_directional += 1
                total_signal_strength += 0.8 * (1 - weight_momentum) * 0.3

            # Effective min: floor at 2 to always require multi-source confirmation
            effective_min = max(min(min_signals, n_directional), 2)
            if n_directional < effective_min:
                continue

            # Include neutral strength in confidence (boosts majority side)
            total_signal_strength += neutral_strength
            n_total = n_directional + (1 if neutral_strength > 0 else 0)

            # Compute confidence
            w_total = weight_momentum + (1 - weight_momentum) * min(n_total / 4, 1.0)
            confidence = total_signal_strength / w_total if w_total > 0 else 0
            signal_mult = min(n_total / 3.0, 1.5)
            confidence = min(confidence * signal_mult, 1.0)

            if confidence < min_confidence:
                continue

            # Execute trade
            cum_return = wd.cum_returns_pct[minute] / 100.0 * direction
            yes_price = sigmoid(cum_return)
            entry_price = yes_price if direction == 1 else (1.0 - yes_price)
            if entry_price <= 0 or entry_price >= 1:
                continue

            if no_compound:
                pos_size = initial_balance * position_pct
            else:
                pos_size = balance * position_pct
            if pos_size <= 0:
                continue

            fee = polymarket_fee(pos_size, entry_price)
            quantity = pos_size / entry_price

            # Settlement: direction matches true_dir → win
            is_win = direction == wd.true_dir
            settlement = 1.0 if is_win else 0.0
            pnl = (settlement - entry_price) * quantity - fee

            trades += 1
            if is_win:
                wins += 1
                gross_profit += pnl + fee
            else:
                gross_loss += abs(pnl + fee)
            total_fees += fee
            balance += pnl
            traded = True

        equity_curve.append(balance)

    # Compute Sharpe
    sharpe = 0.0
    if len(equity_curve) > 2:
        returns = []
        for i in range(1, len(equity_curve)):
            prev = equity_curve[i - 1]
            if prev > 0:
                returns.append((equity_curve[i] - prev) / prev)
        if len(returns) >= 2:
            mean_r = sum(returns) / len(returns)
            var_r = sum((r - mean_r) ** 2 for r in returns) / len(returns)
            std_r = math.sqrt(var_r) if var_r > 0 else 0
            if std_r > 0:
                sharpe = (mean_r / std_r) * math.sqrt(252)

    # Max drawdown
    max_dd = 0.0
    peak = equity_curve[0]
    for v in equity_curve:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (peak - v) / peak
            if dd > max_dd:
                max_dd = dd

    net_pnl = balance - initial_balance
    wr = wins / trades if trades > 0 else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return TrialResult(
        params={
            "entry_start": entry_start,
            "entry_end": entry_end,
            "min_confidence": min_confidence,
            "min_signals": min_signals,
            "weight_momentum": weight_momentum,
            "use_time_filter": use_time_filter,
            "use_last3_bonus": use_last3_bonus,
        },
        total_trades=trades,
        wins=wins,
        win_rate=wr,
        net_pnl=net_pnl,
        sharpe=sharpe,
        max_drawdown=max_dd,
        profit_factor=pf,
        avg_pnl=net_pnl / trades if trades > 0 else 0,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--data", type=str, default="data/btc_1m_2y.csv")
    parser.add_argument("--output", type=str, default="data/singularity_optimal.json")
    parser.add_argument("--no-compound", action="store_true",
                        help="Use fixed position size (no compounding)")
    args = parser.parse_args()

    print("=" * 70)
    print("  SINGULARITY FAST PARAMETER OPTIMISATION")
    print("=" * 70)

    t0 = time.time()
    print(f"  Loading candles from {args.data}...")
    candles = load_1m_fast(Path(args.data))
    print(f"  Loaded {len(candles):,} candles in {time.time() - t0:.1f}s")
    print(f"  Range: {candles[0].ts} to {candles[-1].ts}")

    t1 = time.time()
    windows = group_windows(candles)
    print(f"  {len(windows):,} windows in {time.time() - t1:.1f}s")

    t2 = time.time()
    wd = precompute_windows(windows)
    print(f"  Pre-computed {len(wd):,} valid windows in {time.time() - t2:.1f}s")

    # Parameter grid — updated for corrected ensemble (neutral time-of-day,
    # effective_min floor of 2). Key changes:
    # - min_signals 1 → effectively 2 due to floor, kept for completeness
    # - Finer min_confidence grid (0.20-0.70) since confidence calc changed
    # - Finer weight_momentum grid (0.30-0.80) for precision
    param_grid = {
        "entry_start": [5, 6, 7, 8],
        "entry_end": [9, 10, 11, 12],
        "min_confidence": [0.20, 0.30, 0.40, 0.50, 0.60, 0.70],
        "min_signals": [2, 3],
        "weight_momentum": [0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
        "use_time_filter": [True, False],
        "use_last3_bonus": [True, False],
    }

    # Build tier thresholds for each start/end combo
    def make_tiers(start: int, end: int) -> dict[int, float]:
        tiers: dict[int, float] = {}
        span = max(1, end - start)
        for m in range(start, end + 1):
            pos = (m - start) / span
            tiers[m] = round(0.14 - pos * 0.09, 3)  # 0.14 down to 0.05
        return tiers

    # Count total combos
    import itertools
    all_starts = param_grid["entry_start"]
    all_ends = param_grid["entry_end"]
    all_conf = param_grid["min_confidence"]
    all_signals = param_grid["min_signals"]
    all_wmom = param_grid["weight_momentum"]
    all_time = param_grid["use_time_filter"]
    all_last3 = param_grid["use_last3_bonus"]

    total = 0
    valid_combos = []
    for s, e, c, ns, wm, tf, l3 in itertools.product(
        all_starts, all_ends, all_conf, all_signals, all_wmom, all_time, all_last3,
    ):
        if s >= e:
            continue
        valid_combos.append((s, e, c, ns, wm, tf, l3))
        total += 1

    print(f"\n  Total valid combinations: {total:,}")
    print(f"  Running sweep...")

    t3 = time.time()
    results: list[TrialResult] = []

    for i, (s, e, c, ns, wm, tf, l3) in enumerate(valid_combos):
        tiers = make_tiers(s, e)
        result = run_trial(
            wd, entry_start=s, entry_end=e,
            tier_thresholds=tiers, min_confidence=c,
            min_signals=ns, weight_momentum=wm,
            use_time_filter=tf, use_last3_bonus=l3,
            no_compound=args.no_compound,
        )
        if result.total_trades >= 100:
            results.append(result)

        if (i + 1) % 500 == 0:
            print(f"    {i + 1}/{total} trials ({len(results)} valid)...")

    elapsed = time.time() - t3
    print(f"  Sweep completed: {total:,} trials in {elapsed:.1f}s "
          f"({total / elapsed:.0f} trials/sec)")
    print(f"  {len(results):,} trials with >= 100 trades")

    # Sort by multiple objectives
    # Primary: Sharpe, Secondary: win_rate, Tertiary: net_pnl
    results.sort(key=lambda r: (r.sharpe, r.win_rate, r.net_pnl), reverse=True)

    # Print top results
    n = min(args.top, len(results))
    print(f"\n  TOP {n} CONFIGURATIONS (sorted by Sharpe):")
    print("  " + "-" * 120)
    print(f"  {'#':>3}  {'Sharpe':>7}  {'WR%':>6}  {'Trades':>7}  {'NetPnL':>12}  "
          f"{'PF':>7}  {'DD%':>6}  {'AvgPnL':>8}  "
          f"{'Start':>5}  {'End':>3}  {'MinConf':>7}  {'MinSig':>6}  {'WtMom':>5}  "
          f"{'Time':>4}  {'L3':>3}")
    print("  " + "-" * 120)

    for i, r in enumerate(results[:n]):
        p = r.params
        print(f"  {i+1:>3}  {r.sharpe:>7.2f}  {r.win_rate*100:>5.1f}  {r.total_trades:>7,}  "
              f"${r.net_pnl:>11,.2f}  {r.profit_factor:>7.2f}  {r.max_drawdown*100:>5.1f}  "
              f"${r.avg_pnl:>7,.2f}  "
              f"{p['entry_start']:>5}  {p['entry_end']:>3}  {p['min_confidence']:>7.2f}  "
              f"{p['min_signals']:>6}  {p['weight_momentum']:>5.2f}  "
              f"{'Y' if p['use_time_filter'] else 'N':>4}  "
              f"{'Y' if p['use_last3_bonus'] else 'N':>3}")

    # Also show top configs by win rate (different objective)
    results_by_wr = sorted(results, key=lambda r: (r.win_rate, r.sharpe), reverse=True)
    top_wr = [r for r in results_by_wr if r.total_trades >= 500][:10]
    if top_wr:
        print(f"\n  TOP 10 BY WIN RATE (min 500 trades):")
        print("  " + "-" * 120)
        print(f"  {'#':>3}  {'WR%':>6}  {'Sharpe':>7}  {'Trades':>7}  {'NetPnL':>12}  "
              f"{'PF':>7}  {'DD%':>6}  "
              f"{'Start':>5}  {'End':>3}  {'MinConf':>7}  {'MinSig':>6}  {'WtMom':>5}  "
              f"{'Time':>4}  {'L3':>3}")
        print("  " + "-" * 120)
        for i, r in enumerate(top_wr):
            p = r.params
            print(f"  {i+1:>3}  {r.win_rate*100:>5.1f}  {r.sharpe:>7.2f}  {r.total_trades:>7,}  "
                  f"${r.net_pnl:>11,.2f}  {r.profit_factor:>7.2f}  {r.max_drawdown*100:>5.1f}  "
                  f"{p['entry_start']:>5}  {p['entry_end']:>3}  {p['min_confidence']:>7.2f}  "
                  f"{p['min_signals']:>6}  {p['weight_momentum']:>5.2f}  "
                  f"{'Y' if p['use_time_filter'] else 'N':>4}  "
                  f"{'Y' if p['use_last3_bonus'] else 'N':>3}")

    # Find best balanced config (Sharpe * sqrt(trades))
    results_balanced = sorted(
        [r for r in results if r.total_trades >= 200],
        key=lambda r: r.sharpe * math.sqrt(r.total_trades),
        reverse=True,
    )
    if results_balanced:
        best = results_balanced[0]
        print(f"\n{'=' * 70}")
        print(f"  OPTIMAL SINGULARITY CONFIG (balanced: Sharpe * sqrt(trades))")
        print(f"{'=' * 70}")
        print(f"  entry_minute_start    = {best.params['entry_start']}")
        print(f"  entry_minute_end      = {best.params['entry_end']}")
        print(f"  min_confidence        = {best.params['min_confidence']}")
        print(f"  min_signals_agree     = {best.params['min_signals']}")
        print(f"  weight_momentum       = {best.params['weight_momentum']}")
        print(f"  use_time_filter       = {best.params['use_time_filter']}")
        print(f"  use_last3_bonus       = {best.params['use_last3_bonus']}")
        print(f"  ---")
        print(f"  Sharpe Ratio          = {best.sharpe:.2f}")
        print(f"  Win Rate              = {best.win_rate:.1%}")
        print(f"  Total Trades          = {best.total_trades:,}")
        print(f"  Net P&L               = ${best.net_pnl:,.2f}")
        print(f"  Profit Factor         = {best.profit_factor:.4f}")
        print(f"  Max Drawdown          = {best.max_drawdown:.1%}")
        print(f"  Avg P&L/Trade         = ${best.avg_pnl:.2f}")
        print(f"{'=' * 70}")

    # Save results
    output = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "data_file": args.data,
        "total_windows": len(wd),
        "total_trials": total,
        "valid_trials": len(results),
        "trials_per_sec": round(total / elapsed, 1),
        "best_by_sharpe": {
            "params": results[0].params if results else {},
            "sharpe": round(results[0].sharpe, 4) if results else 0,
            "win_rate": round(results[0].win_rate, 4) if results else 0,
            "trades": results[0].total_trades if results else 0,
            "net_pnl": round(results[0].net_pnl, 2) if results else 0,
        },
        "best_balanced": {
            "params": results_balanced[0].params if results_balanced else {},
            "sharpe": round(results_balanced[0].sharpe, 4) if results_balanced else 0,
            "win_rate": round(results_balanced[0].win_rate, 4) if results_balanced else 0,
            "trades": results_balanced[0].total_trades if results_balanced else 0,
            "net_pnl": round(results_balanced[0].net_pnl, 2) if results_balanced else 0,
        },
        "top_20": [
            {
                "params": r.params,
                "sharpe": round(r.sharpe, 4),
                "win_rate": round(r.win_rate, 4),
                "trades": r.total_trades,
                "net_pnl": round(r.net_pnl, 2),
                "profit_factor": round(r.profit_factor, 4),
                "max_drawdown": round(r.max_drawdown, 4),
            }
            for r in results[:20]
        ],
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {args.output}")
    print(f"  Total time: {time.time() - t0:.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
