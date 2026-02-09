#!/usr/bin/env python3
"""Walk-forward validation for MVHE singularity strategy.

Rolling walk-forward: 6-month train window, 1-month test window,
roll forward by 1 month. For each fold, optimize on train then
evaluate out-of-sample on test.

Also evaluates two fixed configs (BALANCED, HIGH_ACCURACY) across
all test folds without re-optimizing.

Usage:
    python scripts/validation/walk_forward.py
"""

from __future__ import annotations

import csv
import itertools
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Suppress all logging (structlog, stdlib)
logging.disable(logging.CRITICAL)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Suppress structlog before any src/ import could trigger it
try:
    import structlog
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL),
    )
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Data loading (copied from fast_optimize.py — no src/ imports)
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
    """Load 1m candles as lightweight dataclass."""
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
    ts: datetime            # timestamp of first candle
    cum_returns_pct: list[float]
    cum_directions: list[int]
    n_minutes: int
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
            ts=window[0].ts,
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


# ---------------------------------------------------------------------------
# Trial runner (copied from fast_optimize.py)
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


def make_tiers(start: int, end: int) -> dict[int, float]:
    tiers: dict[int, float] = {}
    span = max(1, end - start)
    for m in range(start, end + 1):
        pos = (m - start) / span
        tiers[m] = round(0.14 - pos * 0.09, 3)
    return tiers


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
) -> TrialResult:
    """Run a single trial with corrected ensemble logic and fixed sizing."""
    balance = initial_balance
    fixed_pos_size = initial_balance * position_pct  # fixed sizing
    trades = 0
    wins = 0
    gross_profit = 0.0
    gross_loss = 0.0
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

            # Skip flat candles
            if cum_pct == 0.0:
                continue

            # Count DIRECTIONAL signals only
            n_directional = 0
            total_signal_strength = 0.0
            neutral_strength = 0.0

            momentum_strength = min(cum_pct / 0.25, 1.0)
            n_directional += 1
            total_signal_strength += momentum_strength * weight_momentum

            ofi_strength = min(cum_pct / 0.15, 1.0)
            n_directional += 1
            total_signal_strength += ofi_strength * (1 - weight_momentum) * 0.4

            # Time-of-day: NEUTRAL meta-signal
            if use_time_filter:
                hour_mult = HOUR_MULTIPLIERS.get(wd.hour_utc, 1.0)
                if hour_mult >= 0.75:
                    time_strength = min(hour_mult / 1.25, 1.0)
                    neutral_strength += time_strength * (1 - weight_momentum) * 0.3

            # Last-3 agreement: directional
            if use_last3_bonus and minute < len(wd.last3_agree) and wd.last3_agree[minute]:
                n_directional += 1
                total_signal_strength += 0.8 * (1 - weight_momentum) * 0.3

            # Effective min with floor of 2
            effective_min = max(min(min_signals, n_directional), 2)
            if n_directional < effective_min:
                continue

            # Include neutral in confidence
            total_signal_strength += neutral_strength
            n_total = n_directional + (1 if neutral_strength > 0 else 0)

            w_total = weight_momentum + (1 - weight_momentum) * min(n_total / 4, 1.0)
            confidence = total_signal_strength / w_total if w_total > 0 else 0
            signal_mult = min(n_total / 3.0, 1.5)
            confidence = min(confidence * signal_mult, 1.0)

            if confidence < min_confidence:
                continue

            cum_return = wd.cum_returns_pct[minute] / 100.0 * direction
            yes_price = sigmoid(cum_return)
            entry_price = yes_price if direction == 1 else (1.0 - yes_price)
            if entry_price <= 0 or entry_price >= 1:
                continue

            pos_size = fixed_pos_size
            if pos_size <= 0:
                continue

            fee = polymarket_fee(pos_size, entry_price)
            quantity = pos_size / entry_price

            is_win = direction == wd.true_dir
            settlement = 1.0 if is_win else 0.0
            pnl = (settlement - entry_price) * quantity - fee

            trades += 1
            if is_win:
                wins += 1
                gross_profit += pnl + fee
            else:
                gross_loss += abs(pnl + fee)
            balance += pnl
            traded = True

        equity_curve.append(balance)

    # Sharpe ratio
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


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

# Fixed configs to test across all folds
BALANCED_CONFIG = {
    "entry_start": 8,
    "entry_end": 12,
    "min_confidence": 0.50,
    "min_signals": 2,
    "weight_momentum": 0.30,
    "use_time_filter": True,
    "use_last3_bonus": True,
}

HIGH_ACCURACY_CONFIG = {
    "entry_start": 8,
    "entry_end": 12,
    "min_confidence": 0.70,
    "min_signals": 2,
    "weight_momentum": 0.80,
    "use_time_filter": False,
    "use_last3_bonus": True,
}

# Reduced parameter grid for walk-forward (keep it fast)
WF_PARAM_GRID = {
    "entry_start": [7, 8],
    "entry_end": [11, 12],
    "min_confidence": [0.30, 0.50, 0.70],
    "min_signals": [2, 3],
    "weight_momentum": [0.30, 0.50, 0.80],
    "use_time_filter": [True, False],
    "use_last3_bonus": [True, False],
}


def build_param_combos() -> list[tuple[int, int, float, int, float, bool, bool]]:
    """Build valid parameter combinations from the grid."""
    combos = []
    for s, e, c, ns, wm, tf, l3 in itertools.product(
        WF_PARAM_GRID["entry_start"],
        WF_PARAM_GRID["entry_end"],
        WF_PARAM_GRID["min_confidence"],
        WF_PARAM_GRID["min_signals"],
        WF_PARAM_GRID["weight_momentum"],
        WF_PARAM_GRID["use_time_filter"],
        WF_PARAM_GRID["use_last3_bonus"],
    ):
        if s >= e:
            continue
        combos.append((s, e, c, ns, wm, tf, l3))
    return combos


def eval_config(
    windows_data: list[WindowData],
    config: dict[str, Any],
) -> TrialResult:
    """Evaluate a single config on a slice of window data."""
    tiers = make_tiers(config["entry_start"], config["entry_end"])
    return run_trial(
        windows_data,
        entry_start=config["entry_start"],
        entry_end=config["entry_end"],
        tier_thresholds=tiers,
        min_confidence=config["min_confidence"],
        min_signals=config["min_signals"],
        weight_momentum=config["weight_momentum"],
        use_time_filter=config["use_time_filter"],
        use_last3_bonus=config["use_last3_bonus"],
    )


def optimize_on_slice(
    windows_data: list[WindowData],
    combos: list[tuple[int, int, float, int, float, bool, bool]],
) -> tuple[TrialResult, dict[str, Any]]:
    """Run param sweep on a data slice and return best result by balanced score."""
    best_result: TrialResult | None = None
    best_score = -float("inf")
    best_params: dict[str, Any] = {}

    for s, e, c, ns, wm, tf, l3 in combos:
        tiers = make_tiers(s, e)
        result = run_trial(
            windows_data,
            entry_start=s,
            entry_end=e,
            tier_thresholds=tiers,
            min_confidence=c,
            min_signals=ns,
            weight_momentum=wm,
            use_time_filter=tf,
            use_last3_bonus=l3,
        )
        if result.total_trades < 20:
            continue
        # Balanced score: Sharpe * sqrt(trades)
        score = result.sharpe * math.sqrt(result.total_trades)
        if score > best_score:
            best_score = score
            best_result = result
            best_params = result.params

    if best_result is None:
        # Fallback: return BALANCED config result
        best_result = eval_config(windows_data, BALANCED_CONFIG)
        best_params = BALANCED_CONFIG

    return best_result, best_params


def generate_monthly_boundaries(all_windows: list[WindowData]) -> list[datetime]:
    """Generate month-start boundaries from the data range."""
    if not all_windows:
        return []

    first_ts = all_windows[0].ts
    last_ts = all_windows[-1].ts

    boundaries: list[datetime] = []
    year = first_ts.year
    month = first_ts.month

    while True:
        boundary = datetime(year, month, 1, tzinfo=timezone.utc)
        if boundary > last_ts:
            break
        boundaries.append(boundary)
        month += 1
        if month > 12:
            month = 1
            year += 1

    # Add one more month past the end to capture the last fold
    boundaries.append(datetime(year, month, 1, tzinfo=timezone.utc))
    return boundaries


def slice_windows(
    all_windows: list[WindowData],
    start: datetime,
    end: datetime,
) -> list[WindowData]:
    """Return windows with ts in [start, end)."""
    return [w for w in all_windows if start <= w.ts < end]


@dataclass
class FoldResult:
    fold_num: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    is_sharpe: float
    oos_sharpe: float
    is_wr: float
    oos_wr: float
    oos_trades: int
    oos_pnl: float
    oos_max_dd: float
    best_params: dict[str, Any]


@dataclass
class FixedConfigFoldResult:
    fold_num: int
    test_start: str
    test_end: str
    sharpe: float
    win_rate: float
    trades: int
    pnl: float
    max_dd: float


def run_walk_forward(
    all_windows: list[WindowData],
    train_months: int = 6,
    test_months: int = 1,
) -> list[FoldResult]:
    """Run rolling walk-forward validation."""
    boundaries = generate_monthly_boundaries(all_windows)
    combos = build_param_combos()

    print(f"\n  Parameter grid: {len(combos)} combinations per fold")
    print(f"  Train window: {train_months} months, Test window: {test_months} month(s)")
    print(f"  Month boundaries: {len(boundaries)} ({boundaries[0].strftime('%Y-%m')} to {boundaries[-1].strftime('%Y-%m')})")

    folds: list[FoldResult] = []
    fold_num = 0

    # We need at least train_months + test_months months of data
    min_idx = train_months
    max_idx = len(boundaries) - test_months

    for test_start_idx in range(min_idx, max_idx):
        train_start = boundaries[test_start_idx - train_months]
        train_end = boundaries[test_start_idx]
        test_start = boundaries[test_start_idx]
        test_end = boundaries[test_start_idx + test_months]

        train_data = slice_windows(all_windows, train_start, train_end)
        test_data = slice_windows(all_windows, test_start, test_end)

        if len(train_data) < 100 or len(test_data) < 20:
            continue

        fold_num += 1

        # Optimize on train set
        is_result, best_params = optimize_on_slice(train_data, combos)

        # Evaluate on test set with best params
        oos_result = eval_config(test_data, best_params)

        folds.append(FoldResult(
            fold_num=fold_num,
            train_start=train_start.strftime("%Y-%m-%d"),
            train_end=train_end.strftime("%Y-%m-%d"),
            test_start=test_start.strftime("%Y-%m-%d"),
            test_end=test_end.strftime("%Y-%m-%d"),
            is_sharpe=is_result.sharpe,
            oos_sharpe=oos_result.sharpe,
            is_wr=is_result.win_rate,
            oos_wr=oos_result.win_rate,
            oos_trades=oos_result.total_trades,
            oos_pnl=oos_result.net_pnl,
            oos_max_dd=oos_result.max_drawdown,
            best_params=best_params,
        ))

        print(f"    Fold {fold_num:>2}: train {train_start.strftime('%Y-%m')} - {train_end.strftime('%Y-%m')} | "
              f"test {test_start.strftime('%Y-%m')} - {test_end.strftime('%Y-%m')} | "
              f"IS Sharpe {is_result.sharpe:>6.2f} | OOS Sharpe {oos_result.sharpe:>6.2f} | "
              f"OOS WR {oos_result.win_rate:.1%} | {oos_result.total_trades} trades")

    return folds


def eval_fixed_config_walk_forward(
    all_windows: list[WindowData],
    config: dict[str, Any],
    config_name: str,
    train_months: int = 6,
    test_months: int = 1,
) -> list[FixedConfigFoldResult]:
    """Evaluate a fixed config across all test folds (no optimization)."""
    boundaries = generate_monthly_boundaries(all_windows)
    results: list[FixedConfigFoldResult] = []

    min_idx = train_months
    max_idx = len(boundaries) - test_months
    fold_num = 0

    for test_start_idx in range(min_idx, max_idx):
        test_start = boundaries[test_start_idx]
        test_end = boundaries[test_start_idx + test_months]
        test_data = slice_windows(all_windows, test_start, test_end)

        if len(test_data) < 20:
            continue

        fold_num += 1
        result = eval_config(test_data, config)

        results.append(FixedConfigFoldResult(
            fold_num=fold_num,
            test_start=test_start.strftime("%Y-%m-%d"),
            test_end=test_end.strftime("%Y-%m-%d"),
            sharpe=result.sharpe,
            win_rate=result.win_rate,
            trades=result.total_trades,
            pnl=result.net_pnl,
            max_dd=result.max_drawdown,
        ))

    return results


def print_fold_table(folds: list[FoldResult]) -> None:
    """Print formatted walk-forward results table."""
    print(f"\n  {'Fold':>4}  {'Train Period':>23}  {'Test Period':>23}  "
          f"{'IS Sharpe':>10}  {'OOS Sharpe':>10}  {'IS WR':>7}  {'OOS WR':>7}  "
          f"{'OOS Trades':>10}  {'OOS PnL':>10}  {'OOS DD%':>7}")
    print("  " + "-" * 140)

    for f in folds:
        print(f"  {f.fold_num:>4}  {f.train_start} - {f.train_end}  "
              f"{f.test_start} - {f.test_end}  "
              f"{f.is_sharpe:>10.2f}  {f.oos_sharpe:>10.2f}  "
              f"{f.is_wr * 100:>6.1f}%  {f.oos_wr * 100:>6.1f}%  "
              f"{f.oos_trades:>10,}  ${f.oos_pnl:>9.2f}  "
              f"{f.oos_max_dd * 100:>6.2f}%")


def print_fixed_config_table(
    results: list[FixedConfigFoldResult],
    config_name: str,
) -> None:
    """Print formatted fixed config results table."""
    print(f"\n  {'Fold':>4}  {'Test Period':>23}  {'Sharpe':>8}  {'WR':>7}  "
          f"{'Trades':>8}  {'PnL':>10}  {'DD%':>7}")
    print("  " + "-" * 90)

    for r in results:
        print(f"  {r.fold_num:>4}  {r.test_start} - {r.test_end}  "
              f"{r.sharpe:>8.2f}  {r.win_rate * 100:>6.1f}%  "
              f"{r.trades:>8,}  ${r.pnl:>9.2f}  {r.max_dd * 100:>6.2f}%")


def print_aggregate_metrics(folds: list[FoldResult]) -> None:
    """Print aggregate walk-forward metrics."""
    if not folds:
        print("  No folds to aggregate.")
        return

    n = len(folds)
    mean_is_sharpe = sum(f.is_sharpe for f in folds) / n
    mean_oos_sharpe = sum(f.oos_sharpe for f in folds) / n
    mean_is_wr = sum(f.is_wr for f in folds) / n
    mean_oos_wr = sum(f.oos_wr for f in folds) / n
    total_oos_trades = sum(f.oos_trades for f in folds)
    total_oos_pnl = sum(f.oos_pnl for f in folds)
    max_oos_dd = max(f.oos_max_dd for f in folds)
    overfit_ratio = mean_is_sharpe / mean_oos_sharpe if mean_oos_sharpe > 0 else float("inf")

    # Sharpe std dev
    if n >= 2:
        var_oos = sum((f.oos_sharpe - mean_oos_sharpe) ** 2 for f in folds) / (n - 1)
        std_oos_sharpe = math.sqrt(var_oos)
    else:
        std_oos_sharpe = 0.0

    print(f"\n  AGGREGATE WALK-FORWARD METRICS ({n} folds)")
    print("  " + "=" * 50)
    print(f"  Mean IS Sharpe:        {mean_is_sharpe:>8.2f}")
    print(f"  Mean OOS Sharpe:       {mean_oos_sharpe:>8.2f}  (std: {std_oos_sharpe:.2f})")
    print(f"  Mean IS Win Rate:      {mean_is_wr * 100:>7.1f}%")
    print(f"  Mean OOS Win Rate:     {mean_oos_wr * 100:>7.1f}%")
    print(f"  Total OOS Trades:      {total_oos_trades:>8,}")
    print(f"  Total OOS PnL:        ${total_oos_pnl:>8.2f}")
    print(f"  Worst OOS Drawdown:    {max_oos_dd * 100:>7.2f}%")
    print(f"  Overfitting Ratio:     {overfit_ratio:>8.2f}  (IS/OOS Sharpe; closer to 1.0 = less overfit)")


def print_fixed_aggregate(
    results: list[FixedConfigFoldResult],
    config_name: str,
) -> None:
    """Print aggregate metrics for a fixed config."""
    if not results:
        print(f"  No results for {config_name}.")
        return

    n = len(results)
    mean_sharpe = sum(r.sharpe for r in results) / n
    mean_wr = sum(r.win_rate for r in results) / n
    total_trades = sum(r.trades for r in results)
    total_pnl = sum(r.pnl for r in results)
    max_dd = max(r.max_dd for r in results)

    if n >= 2:
        var_s = sum((r.sharpe - mean_sharpe) ** 2 for r in results) / (n - 1)
        std_sharpe = math.sqrt(var_s)
    else:
        std_sharpe = 0.0

    print(f"\n  {config_name} AGGREGATE ({n} folds)")
    print("  " + "=" * 50)
    print(f"  Mean Sharpe:           {mean_sharpe:>8.2f}  (std: {std_sharpe:.2f})")
    print(f"  Mean Win Rate:         {mean_wr * 100:>7.1f}%")
    print(f"  Total Trades:          {total_trades:>8,}")
    print(f"  Total PnL:            ${total_pnl:>8.2f}")
    print(f"  Worst Drawdown:        {max_dd * 100:>7.2f}%")


def build_output(
    folds: list[FoldResult],
    balanced_results: list[FixedConfigFoldResult],
    highaccuracy_results: list[FixedConfigFoldResult],
    elapsed: float,
    total_windows: int,
) -> dict[str, Any]:
    """Build JSON-serializable output dict."""
    n = len(folds)
    mean_is_sharpe = sum(f.is_sharpe for f in folds) / n if n > 0 else 0
    mean_oos_sharpe = sum(f.oos_sharpe for f in folds) / n if n > 0 else 0
    mean_oos_wr = sum(f.oos_wr for f in folds) / n if n > 0 else 0
    overfit_ratio = mean_is_sharpe / mean_oos_sharpe if mean_oos_sharpe > 0 else float("inf")

    def fixed_agg(results: list[FixedConfigFoldResult]) -> dict[str, Any]:
        if not results:
            return {}
        rn = len(results)
        return {
            "mean_sharpe": round(sum(r.sharpe for r in results) / rn, 4),
            "mean_win_rate": round(sum(r.win_rate for r in results) / rn, 4),
            "total_trades": sum(r.trades for r in results),
            "total_pnl": round(sum(r.pnl for r in results), 2),
            "worst_drawdown": round(max(r.max_dd for r in results), 4),
        }

    return {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "total_windows": total_windows,
        "train_months": 6,
        "test_months": 1,
        "total_folds": n,
        "elapsed_seconds": round(elapsed, 1),
        "aggregate": {
            "mean_is_sharpe": round(mean_is_sharpe, 4),
            "mean_oos_sharpe": round(mean_oos_sharpe, 4),
            "mean_oos_win_rate": round(mean_oos_wr, 4),
            "overfitting_ratio": round(overfit_ratio, 4) if overfit_ratio != float("inf") else "inf",
            "total_oos_trades": sum(f.oos_trades for f in folds),
            "total_oos_pnl": round(sum(f.oos_pnl for f in folds), 2),
        },
        "folds": [
            {
                "fold": f.fold_num,
                "train_period": f"{f.train_start} to {f.train_end}",
                "test_period": f"{f.test_start} to {f.test_end}",
                "is_sharpe": round(f.is_sharpe, 4),
                "oos_sharpe": round(f.oos_sharpe, 4),
                "is_win_rate": round(f.is_wr, 4),
                "oos_win_rate": round(f.oos_wr, 4),
                "oos_trades": f.oos_trades,
                "oos_pnl": round(f.oos_pnl, 2),
                "oos_max_drawdown": round(f.oos_max_dd, 4),
                "best_params": f.best_params,
            }
            for f in folds
        ],
        "fixed_configs": {
            "balanced": {
                "config": BALANCED_CONFIG,
                "aggregate": fixed_agg(balanced_results),
                "folds": [
                    {
                        "fold": r.fold_num,
                        "test_period": f"{r.test_start} to {r.test_end}",
                        "sharpe": round(r.sharpe, 4),
                        "win_rate": round(r.win_rate, 4),
                        "trades": r.trades,
                        "pnl": round(r.pnl, 2),
                        "max_drawdown": round(r.max_dd, 4),
                    }
                    for r in balanced_results
                ],
            },
            "high_accuracy": {
                "config": HIGH_ACCURACY_CONFIG,
                "aggregate": fixed_agg(highaccuracy_results),
                "folds": [
                    {
                        "fold": r.fold_num,
                        "test_period": f"{r.test_start} to {r.test_end}",
                        "sharpe": round(r.sharpe, 4),
                        "win_rate": round(r.win_rate, 4),
                        "trades": r.trades,
                        "pnl": round(r.pnl, 2),
                        "max_drawdown": round(r.max_dd, 4),
                    }
                    for r in highaccuracy_results
                ],
            },
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    t0 = time.time()

    data_path = PROJECT_ROOT / "data" / "btc_1m_2y.csv"
    output_path = PROJECT_ROOT / "data" / "validation" / "walk_forward_results.json"

    print("=" * 70)
    print("  WALK-FORWARD VALIDATION")
    print("=" * 70)

    # Load data
    print(f"\n  Loading candles from {data_path}...")
    candles = load_1m_fast(data_path)
    print(f"  Loaded {len(candles):,} candles in {time.time() - t0:.1f}s")
    print(f"  Range: {candles[0].ts} to {candles[-1].ts}")

    t1 = time.time()
    windows = group_windows(candles)
    print(f"  {len(windows):,} windows in {time.time() - t1:.1f}s")

    t2 = time.time()
    all_wd = precompute_windows(windows)
    print(f"  Pre-computed {len(all_wd):,} valid windows in {time.time() - t2:.1f}s")

    # --- Part 1: Walk-forward with optimization ---
    print("\n" + "=" * 70)
    print("  PART 1: WALK-FORWARD WITH ROLLING OPTIMIZATION")
    print("=" * 70)

    t3 = time.time()
    folds = run_walk_forward(all_wd, train_months=6, test_months=1)
    wf_elapsed = time.time() - t3

    print_fold_table(folds)
    print_aggregate_metrics(folds)
    print(f"\n  Walk-forward optimization completed in {wf_elapsed:.1f}s")

    # --- Part 2: Fixed configs across all test folds ---
    print("\n" + "=" * 70)
    print("  PART 2: FIXED CONFIGS (NO RE-OPTIMIZATION)")
    print("=" * 70)

    print("\n  --- BALANCED CONFIG ---")
    print(f"  {BALANCED_CONFIG}")
    balanced_results = eval_fixed_config_walk_forward(all_wd, BALANCED_CONFIG, "BALANCED")
    print_fixed_config_table(balanced_results, "BALANCED")
    print_fixed_aggregate(balanced_results, "BALANCED")

    print("\n  --- HIGH ACCURACY CONFIG ---")
    print(f"  {HIGH_ACCURACY_CONFIG}")
    ha_results = eval_fixed_config_walk_forward(all_wd, HIGH_ACCURACY_CONFIG, "HIGH_ACCURACY")
    print_fixed_config_table(ha_results, "HIGH_ACCURACY")
    print_fixed_aggregate(ha_results, "HIGH_ACCURACY")

    # --- Save results ---
    total_elapsed = time.time() - t0
    output = build_output(folds, balanced_results, ha_results, total_elapsed, len(all_wd))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"  Results saved to {output_path}")
    print(f"  Total time: {total_elapsed:.1f}s")
    print(f"{'=' * 70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
