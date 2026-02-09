#!/usr/bin/env python3
"""Monte Carlo & bootstrap statistical validation for MVHE strategies.

Implements:
  1. Bootstrap confidence intervals (1000 resamples)
  2. Permutation test for significance (1000 permutations)
  3. Random entry baseline with z-score
  4. Wald-Wolfowitz runs test for independence
  5. Risk of ruin Monte Carlo (10,000 paths)
  6. Kelly criterion validation via Monte Carlo

Usage:
    python scripts/validation/monte_carlo.py
"""

from __future__ import annotations

import csv
import json
import logging
import math
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Suppress all logging (structlog etc.)
logging.disable(logging.CRITICAL)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Reproducibility
random.seed(42)


# ---------------------------------------------------------------------------
# Data structures (copied from fast_optimize.py — no src/ imports)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class MiniCandle:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(slots=True)
class WindowData:
    idx: int
    open_price: float
    close_price: float
    true_dir: int
    hour_utc: int
    cum_returns_pct: list[float]
    cum_directions: list[int]
    n_minutes: int
    last3_agree: list[bool]


@dataclass(slots=True)
class TradeResult:
    """Single trade outcome."""
    pnl: float
    is_win: bool
    entry_price: float
    settlement: float
    fee: float
    pos_size: float


# ---------------------------------------------------------------------------
# Data loading / window prep (from fast_optimize.py)
# ---------------------------------------------------------------------------

HOUR_MULTIPLIERS = {
    0: 0.95, 1: 0.68, 2: 0.80, 3: 0.85, 4: 0.90, 5: 1.10,
    6: 1.05, 7: 1.05, 8: 0.95, 9: 1.10, 10: 1.15, 11: 1.10,
    12: 0.90, 13: 0.80, 14: 1.25, 15: 1.15, 16: 1.10, 17: 1.05,
    18: 0.95, 19: 0.90, 20: 0.88, 21: 0.90, 22: 0.85, 23: 0.92,
}


def load_1m_fast(path: Path) -> list[MiniCandle]:
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


def precompute_windows(windows: list[list[MiniCandle]]) -> list[WindowData]:
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
            idx=idx, open_price=w_open, close_price=w_close,
            true_dir=true_dir, hour_utc=hour_utc,
            cum_returns_pct=cum_returns_pct, cum_directions=cum_directions,
            n_minutes=len(window), last3_agree=last3_agree,
        ))
    return data


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


# ---------------------------------------------------------------------------
# Run strategy on windows → list of TradeResult
# ---------------------------------------------------------------------------

def run_strategy_trades(
    windows_data: list[WindowData],
    entry_start: int,
    entry_end: int,
    min_confidence: float,
    min_signals: int,
    weight_momentum: float,
    use_time_filter: bool,
    use_last3_bonus: bool,
    position_pct: float = 0.02,
) -> list[TradeResult]:
    """Run strategy and return individual trade results.

    Uses FIXED position sizing and corrected ensemble logic:
    - Time-of-day is neutral (boosts confidence, not directional count)
    - effective_min floor of 2 (always require multi-source confirmation)
    - cum_pct == 0 skipped
    """
    tier_thresholds = make_tiers(entry_start, entry_end)
    initial_balance = 10000.0
    fixed_pos_size = initial_balance * position_pct  # fixed $200 per trade
    balance = initial_balance
    trades: list[TradeResult] = []

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

            momentum_str = min(cum_pct / 0.25, 1.0)
            n_directional += 1
            total_signal_strength += momentum_str * weight_momentum

            ofi_str = min(cum_pct / 0.15, 1.0)
            n_directional += 1
            total_signal_strength += ofi_str * (1 - weight_momentum) * 0.4

            # Time-of-day: NEUTRAL meta-signal
            if use_time_filter:
                hour_mult = HOUR_MULTIPLIERS.get(wd.hour_utc, 1.0)
                if hour_mult >= 0.75:
                    time_str = min(hour_mult / 1.25, 1.0)
                    neutral_strength += time_str * (1 - weight_momentum) * 0.3

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

            trades.append(TradeResult(
                pnl=pnl, is_win=is_win, entry_price=entry_price,
                settlement=settlement, fee=fee, pos_size=pos_size,
            ))
            balance += pnl
            traded = True

    return trades


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_sharpe(pnls: list[float]) -> float:
    if len(pnls) < 2:
        return 0.0
    mean_r = statistics.mean(pnls)
    std_r = statistics.pstdev(pnls)
    if std_r == 0:
        return 0.0
    return (mean_r / std_r) * math.sqrt(252)


def compute_win_rate(trades: list[TradeResult]) -> float:
    if not trades:
        return 0.0
    return sum(1 for t in trades if t.is_win) / len(trades)


def compute_profit_factor(trades: list[TradeResult]) -> float:
    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = sum(abs(t.pnl) for t in trades if t.pnl < 0)
    if gross_loss == 0:
        return float("inf")
    return gross_profit / gross_loss


def compute_avg_pnl(trades: list[TradeResult]) -> float:
    if not trades:
        return 0.0
    return statistics.mean(t.pnl for t in trades)


# ---------------------------------------------------------------------------
# 1. Bootstrap Confidence Intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    trades: list[TradeResult], n_resamples: int = 1000, ci: float = 0.95,
) -> dict[str, dict[str, float]]:
    """Bootstrap 95% CI for win rate, Sharpe, profit factor, avg P&L."""
    n = len(trades)
    if n == 0:
        return {}

    alpha = (1 - ci) / 2
    lo_idx = int(n_resamples * alpha)
    hi_idx = int(n_resamples * (1 - alpha))

    win_rates: list[float] = []
    sharpes: list[float] = []
    profit_factors: list[float] = []
    avg_pnls: list[float] = []

    for _ in range(n_resamples):
        sample = random.choices(trades, k=n)
        win_rates.append(compute_win_rate(sample))
        sharpes.append(compute_sharpe([t.pnl for t in sample]))
        profit_factors.append(compute_profit_factor(sample))
        avg_pnls.append(compute_avg_pnl(sample))

    win_rates.sort()
    sharpes.sort()
    profit_factors.sort()
    avg_pnls.sort()

    return {
        "win_rate": {
            "mean": statistics.mean(win_rates),
            "ci_lower": win_rates[lo_idx],
            "ci_upper": win_rates[hi_idx],
        },
        "sharpe": {
            "mean": statistics.mean(sharpes),
            "ci_lower": sharpes[lo_idx],
            "ci_upper": sharpes[hi_idx],
        },
        "profit_factor": {
            "mean": statistics.mean(profit_factors),
            "ci_lower": profit_factors[lo_idx],
            "ci_upper": profit_factors[hi_idx],
        },
        "avg_pnl": {
            "mean": statistics.mean(avg_pnls),
            "ci_lower": avg_pnls[lo_idx],
            "ci_upper": avg_pnls[hi_idx],
        },
    }


# ---------------------------------------------------------------------------
# 2. Permutation Test
# ---------------------------------------------------------------------------

def permutation_test(
    trades: list[TradeResult], n_permutations: int = 1000,
) -> dict[str, float]:
    """Shuffle win/loss labels, compute p-value for Sharpe."""
    actual_sharpe = compute_sharpe([t.pnl for t in trades])
    pnl_magnitudes = [abs(t.pnl) for t in trades]
    signs = [1 if t.is_win else -1 for t in trades]

    count_ge = 0
    for _ in range(n_permutations):
        shuffled_signs = signs[:]
        random.shuffle(shuffled_signs)
        shuffled_pnls = [s * m for s, m in zip(shuffled_signs, pnl_magnitudes)]
        perm_sharpe = compute_sharpe(shuffled_pnls)
        if perm_sharpe >= actual_sharpe:
            count_ge += 1

    p_value = count_ge / n_permutations
    return {
        "actual_sharpe": actual_sharpe,
        "p_value": p_value,
        "significant_at_005": p_value < 0.05,
        "permutations": n_permutations,
    }


# ---------------------------------------------------------------------------
# 3. Random Entry Baseline
# ---------------------------------------------------------------------------

def random_entry_baseline(
    windows_data: list[WindowData],
    n_random: int = 1000,
    entry_start: int = 8,
    entry_end: int = 11,
) -> dict[str, float]:
    """Run random entry strategies as baseline comparison.

    Uses stratified sampling (10K windows per trial) for speed while
    maintaining statistical validity via CLT.
    """
    # Pre-compute per-window random trade outcomes for all possible
    # minutes/directions to avoid repeated sigmoid calls
    sample_size = min(10000, len(windows_data))
    random_win_rates: list[float] = []
    random_sharpes: list[float] = []

    for _ in range(n_random):
        sampled = random.sample(windows_data, sample_size)
        wins = 0
        total = 0
        pnls: list[float] = []
        balance = 10000.0

        fixed_pos = 10000.0 * 0.02  # fixed $200 per trade
        for wd in sampled:
            minute = random.randint(entry_start, min(entry_end, wd.n_minutes - 1))
            if minute >= wd.n_minutes:
                continue
            direction = random.choice([1, -1])

            cum_return = wd.cum_returns_pct[minute] / 100.0 * wd.cum_directions[minute]
            yes_price = sigmoid(cum_return)
            entry_price = yes_price if direction == 1 else (1.0 - yes_price)
            if entry_price <= 0 or entry_price >= 1:
                continue

            pos_size = fixed_pos
            if pos_size <= 0:
                continue
            fee = polymarket_fee(pos_size, entry_price)
            quantity = pos_size / entry_price
            is_win = direction == wd.true_dir
            settlement = 1.0 if is_win else 0.0
            pnl = (settlement - entry_price) * quantity - fee

            total += 1
            if is_win:
                wins += 1
            pnls.append(pnl)
            balance += pnl

        wr = wins / total if total > 0 else 0
        random_win_rates.append(wr)
        random_sharpes.append(compute_sharpe(pnls))

    mean_wr = statistics.mean(random_win_rates)
    std_wr = statistics.pstdev(random_win_rates) if len(random_win_rates) > 1 else 1e-9
    mean_sharpe = statistics.mean(random_sharpes)
    std_sharpe = statistics.pstdev(random_sharpes) if len(random_sharpes) > 1 else 1e-9

    return {
        "random_mean_win_rate": mean_wr,
        "random_std_win_rate": std_wr,
        "random_mean_sharpe": mean_sharpe,
        "random_std_sharpe": std_sharpe,
        "n_random_strategies": n_random,
        "sample_size_per_trial": sample_size,
    }


# ---------------------------------------------------------------------------
# 4. Wald-Wolfowitz Runs Test
# ---------------------------------------------------------------------------

def runs_test(trades: list[TradeResult]) -> dict[str, Any]:
    """Wald-Wolfowitz runs test for independence of outcomes."""
    if len(trades) < 10:
        return {"error": "Too few trades for runs test"}

    sequence = [1 if t.is_win else 0 for t in trades]
    n = len(sequence)
    n1 = sum(sequence)       # wins
    n0 = n - n1              # losses

    if n0 == 0 or n1 == 0:
        return {"error": "All trades same outcome — cannot run test"}

    # Count runs
    runs = 1
    for i in range(1, n):
        if sequence[i] != sequence[i - 1]:
            runs += 1

    # Expected runs and variance under independence
    expected_runs = 1 + (2 * n0 * n1) / n
    var_runs = (2 * n0 * n1 * (2 * n0 * n1 - n)) / (n * n * (n - 1))

    if var_runs <= 0:
        return {"error": "Variance is zero — degenerate case"}

    z_score = (runs - expected_runs) / math.sqrt(var_runs)

    # Two-tailed p-value approximation using normal CDF
    p_value = 2 * (1 - _norm_cdf(abs(z_score)))

    return {
        "total_trades": n,
        "wins": n1,
        "losses": n0,
        "observed_runs": runs,
        "expected_runs": round(expected_runs, 2),
        "z_score": round(z_score, 4),
        "p_value": round(p_value, 4),
        "independent_at_005": p_value > 0.05,
    }


def _norm_cdf(x: float) -> float:
    """Approximate standard normal CDF (Abramowitz & Stegun)."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ---------------------------------------------------------------------------
# 5. Risk of Ruin Monte Carlo
# ---------------------------------------------------------------------------

def risk_of_ruin(
    trades: list[TradeResult],
    n_paths: int = 10000,
    trades_per_path: int = 1000,
    starting_balance: float = 10000.0,
    position_pct: float = 0.02,
) -> dict[str, Any]:
    """Simulate Monte Carlo paths to estimate drawdown probabilities."""
    win_rate = compute_win_rate(trades)
    win_pnls = [t.pnl / t.pos_size for t in trades if t.is_win and t.pos_size > 0]
    loss_pnls = [t.pnl / t.pos_size for t in trades if not t.is_win and t.pos_size > 0]

    if not win_pnls or not loss_pnls:
        return {"error": "Need both winning and losing trades"}

    avg_win_pct = statistics.mean(win_pnls)
    avg_loss_pct = statistics.mean(loss_pnls)

    final_balances: list[float] = []
    max_drawdowns: list[float] = []
    ruin_50_count = 0
    ruin_90_count = 0

    for _ in range(n_paths):
        balance = starting_balance
        peak = balance
        max_dd = 0.0

        for _ in range(trades_per_path):
            pos_size = balance * position_pct
            if pos_size <= 0:
                break

            if random.random() < win_rate:
                pnl = pos_size * random.choice(win_pnls)
            else:
                pnl = pos_size * random.choice(loss_pnls)

            balance += pnl

            if balance > peak:
                peak = balance
            if peak > 0:
                dd = (peak - balance) / peak
                if dd > max_dd:
                    max_dd = dd

            if balance <= 0:
                balance = 0
                max_dd = 1.0
                break

        final_balances.append(balance)
        max_drawdowns.append(max_dd)
        if max_dd >= 0.50:
            ruin_50_count += 1
        if max_dd >= 0.90:
            ruin_90_count += 1

    final_balances.sort()
    return {
        "n_paths": n_paths,
        "trades_per_path": trades_per_path,
        "win_rate_used": round(win_rate, 4),
        "avg_win_return_pct": round(avg_win_pct * 100, 4),
        "avg_loss_return_pct": round(avg_loss_pct * 100, 4),
        "prob_50pct_drawdown": round(ruin_50_count / n_paths, 4),
        "prob_90pct_drawdown": round(ruin_90_count / n_paths, 4),
        "median_final_balance": round(final_balances[n_paths // 2], 2),
        "p5_final_balance": round(final_balances[int(n_paths * 0.05)], 2),
        "p95_final_balance": round(final_balances[int(n_paths * 0.95)], 2),
        "mean_max_drawdown": round(statistics.mean(max_drawdowns) * 100, 2),
    }


# ---------------------------------------------------------------------------
# 6. Kelly Criterion Validation
# ---------------------------------------------------------------------------

def kelly_validation(
    trades: list[TradeResult],
    n_paths: int = 5000,
    trades_per_path: int = 1000,
    starting_balance: float = 10000.0,
) -> dict[str, Any]:
    """Compare quarter-Kelly, half-Kelly, full-Kelly via Monte Carlo."""
    win_rate = compute_win_rate(trades)
    win_pnls = [t.pnl / t.pos_size for t in trades if t.is_win and t.pos_size > 0]
    loss_pnls = [t.pnl / t.pos_size for t in trades if not t.is_win and t.pos_size > 0]

    if not win_pnls or not loss_pnls:
        return {"error": "Need both winning and losing trades"}

    avg_win = statistics.mean(win_pnls)
    avg_loss = abs(statistics.mean(loss_pnls))

    # Optimal Kelly fraction for binary: f* = (p * b - q) / b
    # where b = avg_win / avg_loss, p = win_rate, q = 1 - p
    if avg_loss == 0:
        optimal_kelly = 1.0
    else:
        b = avg_win / avg_loss
        optimal_kelly = (win_rate * b - (1 - win_rate)) / b
    optimal_kelly = max(0, min(optimal_kelly, 1.0))

    kelly_fractions = {
        "quarter_kelly": optimal_kelly * 0.25,
        "half_kelly": optimal_kelly * 0.50,
        "full_kelly": optimal_kelly * 1.00,
    }

    results: dict[str, Any] = {
        "optimal_kelly_fraction": round(optimal_kelly, 4),
        "win_rate": round(win_rate, 4),
        "avg_win_return": round(avg_win, 6),
        "avg_loss_return": round(avg_loss, 6),
    }

    for label, frac in kelly_fractions.items():
        final_balances: list[float] = []
        ruin_count = 0

        for _ in range(n_paths):
            balance = starting_balance
            peak = balance
            max_dd = 0.0

            for _ in range(trades_per_path):
                pos_size = balance * frac
                if pos_size <= 0:
                    break

                if random.random() < win_rate:
                    pnl = pos_size * random.choice(win_pnls)
                else:
                    pnl = pos_size * random.choice(loss_pnls)

                balance += pnl
                if balance > peak:
                    peak = balance
                if peak > 0:
                    dd = (peak - balance) / peak
                    if dd > max_dd:
                        max_dd = dd
                if balance <= 0:
                    balance = 0
                    break

            final_balances.append(balance)
            if max_dd >= 0.50:
                ruin_count += 1

        final_balances.sort()
        results[label] = {
            "fraction": round(frac, 4),
            "median_final_balance": round(final_balances[n_paths // 2], 2),
            "p5_final_balance": round(final_balances[int(n_paths * 0.05)], 2),
            "p95_final_balance": round(final_balances[int(n_paths * 0.95)], 2),
            "prob_50pct_drawdown": round(ruin_count / n_paths, 4),
            "mean_final_balance": round(statistics.mean(final_balances), 2),
        }

    return results


# ---------------------------------------------------------------------------
# Strategy configs
# ---------------------------------------------------------------------------

CONFIGS: dict[str, dict[str, Any]] = {
    "BALANCED": {
        "entry_start": 8,
        "entry_end": 12,
        "min_confidence": 0.50,
        "min_signals": 2,
        "weight_momentum": 0.30,
        "use_time_filter": True,
        "use_last3_bonus": True,
    },
    "HIGH_ACCURACY": {
        "entry_start": 8,
        "entry_end": 12,
        "min_confidence": 0.70,
        "min_signals": 2,
        "weight_momentum": 0.80,
        "use_time_filter": False,
        "use_last3_bonus": True,
    },
}


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _print(msg: str = "") -> None:
    print(msg, flush=True)


def print_header(text: str) -> None:
    _print(f"\n{'=' * 70}")
    _print(f"  {text}")
    _print(f"{'=' * 70}")


def print_section(text: str) -> None:
    _print(f"\n  --- {text} ---")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    t0 = time.time()

    data_path = PROJECT_ROOT / "data" / "btc_1m_2y.csv"
    output_dir = PROJECT_ROOT / "data" / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    print_header("MONTE CARLO STATISTICAL VALIDATION")
    _print(f"  Data: {data_path}")
    _print(f"  Seed: 42 (deterministic)")

    # Load data
    _print(f"\n  Loading candles...")
    candles = load_1m_fast(data_path)
    _print(f"  Loaded {len(candles):,} candles")
    _print(f"  Range: {candles[0].ts} to {candles[-1].ts}")

    windows = group_windows(candles)
    windows_data = precompute_windows(windows)
    _print(f"  {len(windows_data):,} valid 15m windows")

    all_results: dict[str, Any] = {}

    for config_name, config in CONFIGS.items():
        print_header(f"CONFIG: {config_name}")
        for k, v in config.items():
            _print(f"    {k}: {v}")

        # Run strategy
        t1 = time.time()
        trades = run_strategy_trades(windows_data, **config)
        run_time = time.time() - t1
        _print(f"\n  Strategy produced {len(trades):,} trades in {run_time:.1f}s")

        if not trades:
            _print("  WARNING: No trades generated, skipping validation")
            all_results[config_name] = {"error": "No trades"}
            continue

        actual_wr = compute_win_rate(trades)
        actual_sharpe = compute_sharpe([t.pnl for t in trades])
        actual_pf = compute_profit_factor(trades)
        actual_avg_pnl = compute_avg_pnl(trades)

        _print(f"  Win Rate:       {actual_wr:.4%}")
        _print(f"  Sharpe Ratio:   {actual_sharpe:.4f}")
        _print(f"  Profit Factor:  {actual_pf:.4f}")
        _print(f"  Avg P&L/Trade:  ${actual_avg_pnl:.4f}")

        config_results: dict[str, Any] = {
            "actual_metrics": {
                "trades": len(trades),
                "win_rate": round(actual_wr, 6),
                "sharpe": round(actual_sharpe, 4),
                "profit_factor": round(actual_pf, 4),
                "avg_pnl": round(actual_avg_pnl, 6),
            },
        }

        # 1. Bootstrap CI
        print_section("1. Bootstrap Confidence Intervals (1000 resamples)")
        t1 = time.time()
        boot = bootstrap_ci(trades, n_resamples=1000)
        _print(f"     Completed in {time.time() - t1:.1f}s")
        for metric, vals in boot.items():
            _print(f"     {metric:>15s}: {vals['mean']:.6f}  "
                   f"95% CI [{vals['ci_lower']:.6f}, {vals['ci_upper']:.6f}]")
        config_results["bootstrap_ci"] = boot

        # 2. Permutation Test
        print_section("2. Permutation Test (1000 permutations)")
        t1 = time.time()
        perm = permutation_test(trades, n_permutations=1000)
        _print(f"     Completed in {time.time() - t1:.1f}s")
        _print(f"     Actual Sharpe:  {perm['actual_sharpe']:.4f}")
        _print(f"     p-value:        {perm['p_value']:.4f}")
        _print(f"     Significant:    {'YES (p < 0.05)' if perm['significant_at_005'] else 'NO'}")
        config_results["permutation_test"] = perm

        # 3. Random Entry Baseline
        print_section("3. Random Entry Baseline (1000 random strategies)")
        t1 = time.time()
        rand = random_entry_baseline(
            windows_data, n_random=1000,
            entry_start=config["entry_start"],
            entry_end=config["entry_end"],
        )
        _print(f"     Completed in {time.time() - t1:.1f}s")

        wr_z = (actual_wr - rand["random_mean_win_rate"]) / rand["random_std_win_rate"] if rand["random_std_win_rate"] > 0 else 0
        sharpe_z = (actual_sharpe - rand["random_mean_sharpe"]) / rand["random_std_sharpe"] if rand["random_std_sharpe"] > 0 else 0

        _print(f"     Random Mean WR:     {rand['random_mean_win_rate']:.4%}")
        _print(f"     Random Std WR:      {rand['random_std_win_rate']:.4%}")
        _print(f"     Actual WR:          {actual_wr:.4%}")
        _print(f"     WR z-score:         {wr_z:.2f}")
        _print(f"     Random Mean Sharpe: {rand['random_mean_sharpe']:.4f}")
        _print(f"     Actual Sharpe:      {actual_sharpe:.4f}")
        _print(f"     Sharpe z-score:     {sharpe_z:.2f}")

        rand["wr_z_score"] = round(wr_z, 4)
        rand["sharpe_z_score"] = round(sharpe_z, 4)
        config_results["random_baseline"] = rand

        # 4. Runs Test
        print_section("4. Wald-Wolfowitz Runs Test (Independence)")
        runs = runs_test(trades)
        if "error" in runs:
            _print(f"     Error: {runs['error']}")
        else:
            _print(f"     Total Trades:    {runs['total_trades']:,}")
            _print(f"     Wins / Losses:   {runs['wins']:,} / {runs['losses']:,}")
            _print(f"     Observed Runs:   {runs['observed_runs']:,}")
            _print(f"     Expected Runs:   {runs['expected_runs']:.2f}")
            _print(f"     z-score:         {runs['z_score']:.4f}")
            _print(f"     p-value:         {runs['p_value']:.4f}")
            _print(f"     Independent:     {'YES (p > 0.05)' if runs['independent_at_005'] else 'NO (clustered)'}")
        config_results["runs_test"] = runs

        # 5. Risk of Ruin
        print_section("5. Risk of Ruin (10,000 Monte Carlo paths x 1,000 trades)")
        t1 = time.time()
        ruin = risk_of_ruin(trades, n_paths=10000, trades_per_path=1000)
        _print(f"     Completed in {time.time() - t1:.1f}s")
        if "error" in ruin:
            _print(f"     Error: {ruin['error']}")
        else:
            _print(f"     Win Rate Used:      {ruin['win_rate_used']:.4%}")
            _print(f"     Avg Win Return:     {ruin['avg_win_return_pct']:.4f}%")
            _print(f"     Avg Loss Return:    {ruin['avg_loss_return_pct']:.4f}%")
            _print(f"     P(50% drawdown):    {ruin['prob_50pct_drawdown']:.4%}")
            _print(f"     P(90% drawdown):    {ruin['prob_90pct_drawdown']:.4%}")
            _print(f"     Median Final Bal:   ${ruin['median_final_balance']:,.2f}")
            _print(f"     5th Pctl Bal:       ${ruin['p5_final_balance']:,.2f}")
            _print(f"     95th Pctl Bal:      ${ruin['p95_final_balance']:,.2f}")
            _print(f"     Mean Max DD:        {ruin['mean_max_drawdown']:.2f}%")
        config_results["risk_of_ruin"] = ruin

        # 6. Kelly Criterion Validation
        print_section("6. Kelly Criterion Validation (5,000 paths x 1,000 trades)")
        t1 = time.time()
        kelly = kelly_validation(trades, n_paths=5000, trades_per_path=1000)
        _print(f"     Completed in {time.time() - t1:.1f}s")
        if "error" in kelly:
            _print(f"     Error: {kelly['error']}")
        else:
            _print(f"     Optimal Kelly f*:   {kelly['optimal_kelly_fraction']:.4f}")
            _print(f"     Win Rate:           {kelly['win_rate']:.4%}")
            for label in ["quarter_kelly", "half_kelly", "full_kelly"]:
                k = kelly[label]
                _print(f"     {label:>15s} (f={k['fraction']:.4f}):")
                _print(f"       Median Final:     ${k['median_final_balance']:,.2f}")
                _print(f"       Mean Final:       ${k['mean_final_balance']:,.2f}")
                _print(f"       P(50% DD):        {k['prob_50pct_drawdown']:.4%}")
                _print(f"       5th Pctl:         ${k['p5_final_balance']:,.2f}")
                _print(f"       95th Pctl:        ${k['p95_final_balance']:,.2f}")
        config_results["kelly_validation"] = kelly

        all_results[config_name] = config_results

    # Save JSON
    output_path = output_dir / "monte_carlo_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    elapsed = time.time() - t0
    print_header("VALIDATION COMPLETE")
    _print(f"  Total runtime:  {elapsed:.1f}s")
    _print(f"  Results saved:  {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
