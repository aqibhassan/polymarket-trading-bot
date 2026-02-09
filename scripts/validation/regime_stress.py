#!/usr/bin/env python3
"""Market regime classification and stress testing for MVHE strategies.

Analyses two strategy configs (BALANCED, HIGH_ACCURACY) across:
  1. Market regimes (trending, sideways, volatility)
  2. Monthly breakdown
  3. Hourly analysis
  4. Drawdown analysis
  5. Consecutive loss streaks
  6. Worst period analysis
  7. Day-of-week analysis

Usage:
    python scripts/validation/regime_stress.py
"""

from __future__ import annotations

import csv
import json
import logging
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# Suppress all structlog / logging noise
logging.disable(logging.CRITICAL)
for name in ("structlog", "uvicorn", "httpx", "httpcore"):
    logging.getLogger(name).setLevel(logging.CRITICAL)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Data loading (copied from fast_optimize.py -- lightweight, no src imports)
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
# Pre-compute per-window data (copied from fast_optimize.py)
# ---------------------------------------------------------------------------

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
    ts: datetime  # added: timestamp for regime/date analysis


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
            idx=idx,
            open_price=w_open,
            close_price=w_close,
            true_dir=true_dir,
            hour_utc=hour_utc,
            cum_returns_pct=cum_returns_pct,
            cum_directions=cum_directions,
            n_minutes=len(window),
            last3_agree=last3_agree,
            ts=window[0].ts,
        ))
    return data


# ---------------------------------------------------------------------------
# Hour-of-day multipliers (from fast_optimize.py)
# ---------------------------------------------------------------------------

HOUR_MULTIPLIERS = {
    0: 0.95, 1: 0.68, 2: 0.80, 3: 0.85, 4: 0.90, 5: 1.10,
    6: 1.05, 7: 1.05, 8: 0.95, 9: 1.10, 10: 1.15, 11: 1.10,
    12: 0.90, 13: 0.80, 14: 1.25, 15: 1.15, 16: 1.10, 17: 1.05,
    18: 0.95, 19: 0.90, 20: 0.88, 21: 0.90, 22: 0.85, 23: 0.92,
}


# ---------------------------------------------------------------------------
# Trade simulation (copied from fast_optimize.py)
# ---------------------------------------------------------------------------

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x / 0.07))


def polymarket_fee(pos_size: float, yes_price: float) -> float:
    p = yes_price
    return pos_size * 0.25 * p * p * (1 - p) * (1 - p)


@dataclass(slots=True)
class TradeRecord:
    window_idx: int
    ts: datetime
    direction: int
    entry_price: float
    is_win: bool
    pnl: float
    balance_after: float


def run_strategy(
    windows_data: list[WindowData],
    entry_start: int,
    entry_end: int,
    min_confidence: float,
    min_signals: int,
    weight_momentum: float,
    use_time_filter: bool,
    use_last3_bonus: bool,
    initial_balance: float = 10000.0,
    position_pct: float = 0.02,
) -> tuple[list[TradeRecord], list[float]]:
    """Run strategy and return individual trade records + equity curve.

    Uses FIXED position sizing (fraction of initial balance) to keep
    P&L in a realistic, comparable range across regimes/periods.
    """
    balance = initial_balance
    fixed_pos_size = initial_balance * position_pct  # fixed $200 per trade
    trades: list[TradeRecord] = []
    equity_curve: list[float] = [balance]

    # Build tier thresholds
    tiers: dict[int, float] = {}
    span = max(1, entry_end - entry_start)
    for m in range(entry_start, entry_end + 1):
        pos = (m - entry_start) / span
        tiers[m] = round(0.14 - pos * 0.09, 3)

    for wd in windows_data:
        traded = False
        for minute in range(entry_start, min(entry_end + 1, wd.n_minutes)):
            if traded:
                break

            threshold = tiers.get(minute)
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
            fee = polymarket_fee(pos_size, entry_price)
            quantity = pos_size / entry_price

            is_win = direction == wd.true_dir
            settlement = 1.0 if is_win else 0.0
            pnl = (settlement - entry_price) * quantity - fee

            balance += pnl
            traded = True

            trades.append(TradeRecord(
                window_idx=wd.idx,
                ts=wd.ts,
                direction=direction,
                entry_price=entry_price,
                is_win=is_win,
                pnl=pnl,
                balance_after=balance,
            ))

        equity_curve.append(balance)

    return trades, equity_curve


# ---------------------------------------------------------------------------
# Strategy configs
# ---------------------------------------------------------------------------

STRATEGIES = {
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
# Analysis helpers
# ---------------------------------------------------------------------------

def compute_sharpe(pnls: list[float]) -> float:
    if len(pnls) < 2:
        return 0.0
    mean_r = sum(pnls) / len(pnls)
    var_r = sum((r - mean_r) ** 2 for r in pnls) / len(pnls)
    std_r = math.sqrt(var_r) if var_r > 0 else 0
    if std_r == 0:
        return 0.0
    return (mean_r / std_r) * math.sqrt(252 * 96)  # annualized for 15m windows


def classify_day_regime(candles_for_day: list[MiniCandle]) -> list[str]:
    """Classify a day into regime labels (can have multiple)."""
    if not candles_for_day:
        return ["unknown"]
    day_open = candles_for_day[0].open
    day_close = candles_for_day[-1].close
    day_high = max(c.high for c in candles_for_day)
    day_low = min(c.low for c in candles_for_day)

    if day_open == 0:
        return ["unknown"]

    daily_return = (day_close - day_open) / day_open * 100.0
    daily_range = (day_high - day_low) / day_open * 100.0

    regimes: list[str] = []

    # Trend classification
    if daily_return > 1.0:
        regimes.append("trending_up")
    elif daily_return < -1.0:
        regimes.append("trending_down")
    elif abs(daily_return) < 0.3:
        regimes.append("sideways")
    else:
        regimes.append("moderate")

    # Volatility classification
    if daily_range > 3.0:
        regimes.append("high_vol")
    elif daily_range < 1.0:
        regimes.append("low_vol")

    return regimes


# ---------------------------------------------------------------------------
# 1. Regime Classification
# ---------------------------------------------------------------------------

def regime_analysis(
    trades: list[TradeRecord],
    candles: list[MiniCandle],
) -> dict[str, dict[str, Any]]:
    # Group candles by date
    candles_by_date: dict[str, list[MiniCandle]] = defaultdict(list)
    for c in candles:
        candles_by_date[c.ts.strftime("%Y-%m-%d")].append(c)

    # Map date -> regimes
    date_regimes: dict[str, list[str]] = {}
    for date_str, day_candles in candles_by_date.items():
        date_regimes[date_str] = classify_day_regime(day_candles)

    # Group trades by regime
    regime_trades: dict[str, list[TradeRecord]] = defaultdict(list)
    for t in trades:
        date_str = t.ts.strftime("%Y-%m-%d")
        regimes = date_regimes.get(date_str, ["unknown"])
        for regime in regimes:
            regime_trades[regime].append(t)

    results: dict[str, dict[str, Any]] = {}
    for regime in ["trending_up", "trending_down", "sideways", "moderate", "high_vol", "low_vol"]:
        rt = regime_trades.get(regime, [])
        wins = sum(1 for t in rt if t.is_win)
        pnls = [t.pnl for t in rt]
        results[regime] = {
            "trades": len(rt),
            "wins": wins,
            "win_rate": wins / len(rt) if rt else 0.0,
            "net_pnl": sum(pnls),
            "sharpe": compute_sharpe(pnls),
        }
    return results


# ---------------------------------------------------------------------------
# 2. Monthly Breakdown
# ---------------------------------------------------------------------------

def monthly_analysis(trades: list[TradeRecord]) -> dict[str, dict[str, Any]]:
    monthly: dict[str, list[TradeRecord]] = defaultdict(list)
    for t in trades:
        key = t.ts.strftime("%Y-%m")
        monthly[key].append(t)

    results: dict[str, dict[str, Any]] = {}
    for month in sorted(monthly.keys()):
        mt = monthly[month]
        wins = sum(1 for t in mt if t.is_win)
        pnls = [t.pnl for t in mt]
        results[month] = {
            "trades": len(mt),
            "wins": wins,
            "win_rate": wins / len(mt) if mt else 0.0,
            "net_pnl": sum(pnls),
            "sharpe": compute_sharpe(pnls),
        }
    return results


# ---------------------------------------------------------------------------
# 3. Hourly Analysis
# ---------------------------------------------------------------------------

def hourly_analysis(trades: list[TradeRecord]) -> dict[int, dict[str, Any]]:
    hourly: dict[int, list[TradeRecord]] = defaultdict(list)
    for t in trades:
        hourly[t.ts.hour].append(t)

    results: dict[int, dict[str, Any]] = {}
    for hour in range(24):
        ht = hourly.get(hour, [])
        wins = sum(1 for t in ht if t.is_win)
        results[hour] = {
            "trades": len(ht),
            "wins": wins,
            "win_rate": wins / len(ht) if ht else 0.0,
        }
    return results


# ---------------------------------------------------------------------------
# 4. Drawdown Analysis
# ---------------------------------------------------------------------------

@dataclass
class DrawdownEvent:
    start_idx: int
    trough_idx: int
    end_idx: int  # recovery index (-1 if not recovered)
    peak_equity: float
    trough_equity: float
    drawdown_pct: float
    duration_windows: int
    start_date: str
    trough_date: str


def drawdown_analysis(
    trades: list[TradeRecord],
    equity_curve: list[float],
) -> dict[str, Any]:
    if not equity_curve:
        return {"max_drawdown_pct": 0, "events": []}

    # Track drawdown events
    peak = equity_curve[0]
    peak_idx = 0
    in_drawdown = False
    current_trough = peak
    current_trough_idx = 0
    events: list[DrawdownEvent] = []
    current_start_idx = 0

    for i, eq in enumerate(equity_curve):
        if eq > peak:
            if in_drawdown:
                # Recovered
                events.append(DrawdownEvent(
                    start_idx=current_start_idx,
                    trough_idx=current_trough_idx,
                    end_idx=i,
                    peak_equity=peak,
                    trough_equity=current_trough,
                    drawdown_pct=(peak - current_trough) / peak * 100 if peak > 0 else 0,
                    duration_windows=i - current_start_idx,
                    start_date=_idx_to_date(trades, current_start_idx),
                    trough_date=_idx_to_date(trades, current_trough_idx),
                ))
                in_drawdown = False
            peak = eq
            peak_idx = i
        elif eq < peak:
            if not in_drawdown:
                in_drawdown = True
                current_start_idx = peak_idx
                current_trough = eq
                current_trough_idx = i
            elif eq < current_trough:
                current_trough = eq
                current_trough_idx = i

    # Handle ongoing drawdown at end
    if in_drawdown:
        events.append(DrawdownEvent(
            start_idx=current_start_idx,
            trough_idx=current_trough_idx,
            end_idx=-1,
            peak_equity=peak,
            trough_equity=current_trough,
            drawdown_pct=(peak - current_trough) / peak * 100 if peak > 0 else 0,
            duration_windows=len(equity_curve) - current_start_idx,
            start_date=_idx_to_date(trades, current_start_idx),
            trough_date=_idx_to_date(trades, current_trough_idx),
        ))

    events.sort(key=lambda e: e.drawdown_pct, reverse=True)

    max_dd = events[0].drawdown_pct if events else 0.0
    max_duration = max((e.duration_windows for e in events), default=0)

    return {
        "max_drawdown_pct": max_dd,
        "max_drawdown_duration_windows": max_duration,
        "top_5_events": [
            {
                "drawdown_pct": round(e.drawdown_pct, 4),
                "duration_windows": e.duration_windows,
                "start_date": e.start_date,
                "trough_date": e.trough_date,
                "peak_equity": round(e.peak_equity, 2),
                "trough_equity": round(e.trough_equity, 2),
                "recovered": e.end_idx != -1,
            }
            for e in events[:5]
        ],
    }


def _idx_to_date(trades: list[TradeRecord], idx: int) -> str:
    """Map equity curve index to approximate date from trades."""
    if not trades:
        return "N/A"
    # equity_curve[0] is initial, trades grow from idx 0
    # approximate: use trade closest to this index
    trade_idx = min(max(idx - 1, 0), len(trades) - 1)
    return trades[trade_idx].ts.strftime("%Y-%m-%d %H:%M")


# ---------------------------------------------------------------------------
# 5. Consecutive Loss Analysis
# ---------------------------------------------------------------------------

def consecutive_loss_analysis(trades: list[TradeRecord]) -> dict[str, Any]:
    if not trades:
        return {"max_streak": 0, "distribution": {}, "prob_5_plus": 0.0}

    streaks: list[int] = []
    current_streak = 0
    for t in trades:
        if not t.is_win:
            current_streak += 1
        else:
            if current_streak > 0:
                streaks.append(current_streak)
            current_streak = 0
    if current_streak > 0:
        streaks.append(current_streak)

    max_streak = max(streaks) if streaks else 0

    distribution: dict[str, int] = {"1": 0, "2": 0, "3": 0, "4": 0, "5+": 0}
    for s in streaks:
        if s == 1:
            distribution["1"] += 1
        elif s == 2:
            distribution["2"] += 1
        elif s == 3:
            distribution["3"] += 1
        elif s == 4:
            distribution["4"] += 1
        else:
            distribution["5+"] += 1

    total_streaks = len(streaks)
    prob_5_plus = distribution["5+"] / total_streaks if total_streaks > 0 else 0.0

    return {
        "max_streak": max_streak,
        "distribution": distribution,
        "total_loss_streaks": total_streaks,
        "prob_5_plus": prob_5_plus,
    }


# ---------------------------------------------------------------------------
# 6. Worst Period Analysis
# ---------------------------------------------------------------------------

def worst_period_analysis(trades: list[TradeRecord]) -> dict[str, Any]:
    if not trades:
        return {}

    # Group trades by date for daily P&L
    daily_pnl: dict[str, float] = defaultdict(float)
    for t in trades:
        daily_pnl[t.ts.strftime("%Y-%m-%d")] += t.pnl
    sorted_days = sorted(daily_pnl.keys())
    day_pnls = [(d, daily_pnl[d]) for d in sorted_days]

    # Worst 1-day
    worst_day = min(day_pnls, key=lambda x: x[1]) if day_pnls else ("N/A", 0)

    # Worst 1-week (7 consecutive days)
    worst_week_pnl = float("inf")
    worst_week_start = "N/A"
    for i in range(len(day_pnls)):
        # Find 7-day window
        start_date = datetime.strptime(day_pnls[i][0], "%Y-%m-%d")
        end_date = start_date + timedelta(days=7)
        window_pnl = sum(
            pnl for d, pnl in day_pnls
            if start_date <= datetime.strptime(d, "%Y-%m-%d") < end_date
        )
        if window_pnl < worst_week_pnl:
            worst_week_pnl = window_pnl
            worst_week_start = day_pnls[i][0]

    # Worst 1-month (30 consecutive days)
    worst_month_pnl = float("inf")
    worst_month_start = "N/A"
    for i in range(len(day_pnls)):
        start_date = datetime.strptime(day_pnls[i][0], "%Y-%m-%d")
        end_date = start_date + timedelta(days=30)
        window_pnl = sum(
            pnl for d, pnl in day_pnls
            if start_date <= datetime.strptime(d, "%Y-%m-%d") < end_date
        )
        if window_pnl < worst_month_pnl:
            worst_month_pnl = window_pnl
            worst_month_start = day_pnls[i][0]

    return {
        "worst_1_day": {
            "date": worst_day[0],
            "pnl": round(worst_day[1], 2),
            "profitable": worst_day[1] > 0,
        },
        "worst_1_week": {
            "start_date": worst_week_start,
            "pnl": round(worst_week_pnl, 2) if worst_week_pnl != float("inf") else 0,
            "profitable": worst_week_pnl > 0 if worst_week_pnl != float("inf") else True,
        },
        "worst_1_month": {
            "start_date": worst_month_start,
            "pnl": round(worst_month_pnl, 2) if worst_month_pnl != float("inf") else 0,
            "profitable": worst_month_pnl > 0 if worst_month_pnl != float("inf") else True,
        },
    }


# ---------------------------------------------------------------------------
# 7. Day-of-Week Analysis
# ---------------------------------------------------------------------------

def day_of_week_analysis(trades: list[TradeRecord]) -> dict[str, dict[str, Any]]:
    DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_trades: dict[int, list[TradeRecord]] = defaultdict(list)
    for t in trades:
        dow_trades[t.ts.weekday()].append(t)

    results: dict[str, dict[str, Any]] = {}
    for i, name in enumerate(DOW_NAMES):
        dt = dow_trades.get(i, [])
        wins = sum(1 for t in dt if t.is_win)
        results[name] = {
            "trades": len(dt),
            "wins": wins,
            "win_rate": wins / len(dt) if dt else 0.0,
            "net_pnl": round(sum(t.pnl for t in dt), 2),
        }
    return results


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_regime(results: dict[str, dict[str, Any]], name: str) -> None:
    print(f"\n  {'REGIME':15s}  {'Trades':>7s}  {'Wins':>6s}  {'WR%':>7s}  {'Net PnL':>12s}  {'Sharpe':>8s}")
    print("  " + "-" * 62)
    for regime, data in results.items():
        print(f"  {regime:15s}  {data['trades']:>7,}  {data['wins']:>6,}  "
              f"{data['win_rate']*100:>6.1f}%  ${data['net_pnl']:>11,.2f}  "
              f"{data['sharpe']:>8.2f}")


def print_monthly(results: dict[str, dict[str, Any]], name: str) -> None:
    print(f"\n  {'Month':10s}  {'Trades':>7s}  {'Wins':>6s}  {'WR%':>7s}  {'Net PnL':>12s}  {'Sharpe':>8s}")
    print("  " + "-" * 57)
    negative_months = []
    best_month = ("", -float("inf"))
    worst_month = ("", float("inf"))
    for month, data in results.items():
        flag = ""
        if data["net_pnl"] < 0:
            flag = " <-- NEGATIVE"
            negative_months.append(month)
        print(f"  {month:10s}  {data['trades']:>7,}  {data['wins']:>6,}  "
              f"{data['win_rate']*100:>6.1f}%  ${data['net_pnl']:>11,.2f}  "
              f"{data['sharpe']:>8.2f}{flag}")
        if data["net_pnl"] > best_month[1]:
            best_month = (month, data["net_pnl"])
        if data["net_pnl"] < worst_month[1]:
            worst_month = (month, data["net_pnl"])

    print(f"\n  Best month:  {best_month[0]} (${best_month[1]:,.2f})")
    print(f"  Worst month: {worst_month[0]} (${worst_month[1]:,.2f})")
    if negative_months:
        print(f"  Negative months: {', '.join(negative_months)}")
    else:
        print(f"  Consistency: ALL months profitable")


def print_hourly(results: dict[int, dict[str, Any]], name: str) -> None:
    print(f"\n  {'Hour':>4s}  {'Trades':>7s}  {'Wins':>6s}  {'WR%':>7s}")
    print("  " + "-" * 30)
    best_hour = (0, 0.0)
    worst_hour = (0, 1.0)
    for hour in range(24):
        data = results[hour]
        wr = data["win_rate"]
        print(f"  {hour:>4d}  {data['trades']:>7,}  {data['wins']:>6,}  {wr*100:>6.1f}%")
        if data["trades"] > 0:
            if wr > best_hour[1]:
                best_hour = (hour, wr)
            if wr < worst_hour[1]:
                worst_hour = (hour, wr)
    print(f"\n  Best hour:  {best_hour[0]:02d}:00 UTC ({best_hour[1]*100:.1f}%)")
    print(f"  Worst hour: {worst_hour[0]:02d}:00 UTC ({worst_hour[1]*100:.1f}%)")


def print_drawdown(results: dict[str, Any], name: str) -> None:
    print(f"\n  Max Drawdown: {results['max_drawdown_pct']:.2f}%")
    print(f"  Max Drawdown Duration: {results['max_drawdown_duration_windows']:,} windows")
    print(f"\n  Top 5 Drawdown Events:")
    print(f"  {'#':>3s}  {'DD%':>7s}  {'Duration':>10s}  {'Recovered':>9s}  {'Start':>18s}  {'Trough':>18s}")
    print("  " + "-" * 72)
    for i, e in enumerate(results["top_5_events"]):
        print(f"  {i+1:>3d}  {e['drawdown_pct']:>6.2f}%  {e['duration_windows']:>10,}  "
              f"{'Yes' if e['recovered'] else 'No':>9s}  {e['start_date']:>18s}  {e['trough_date']:>18s}")


def print_consecutive(results: dict[str, Any], name: str) -> None:
    print(f"\n  Max Consecutive Losses: {results['max_streak']}")
    print(f"  Total Loss Streaks:    {results['total_loss_streaks']}")
    print(f"  P(5+ consecutive):     {results['prob_5_plus']*100:.2f}%")
    print(f"\n  Distribution:")
    print(f"  {'Length':>8s}  {'Count':>7s}")
    print("  " + "-" * 18)
    for length, count in results["distribution"].items():
        print(f"  {length:>8s}  {count:>7,}")


def print_worst_period(results: dict[str, Any], name: str) -> None:
    for period, data in results.items():
        label = period.replace("_", " ").title()
        status = "PROFITABLE" if data.get("profitable") else "LOSING"
        date_key = "date" if "date" in data else "start_date"
        print(f"  {label}: ${data['pnl']:,.2f} ({status}) — {data[date_key]}")


def print_day_of_week(results: dict[str, dict[str, Any]], name: str) -> None:
    print(f"\n  {'Day':>5s}  {'Trades':>7s}  {'Wins':>6s}  {'WR%':>7s}  {'Net PnL':>12s}")
    print("  " + "-" * 44)
    for day, data in results.items():
        print(f"  {day:>5s}  {data['trades']:>7,}  {data['wins']:>6,}  "
              f"{data['win_rate']*100:>6.1f}%  ${data['net_pnl']:>11,.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    data_path = PROJECT_ROOT / "data" / "btc_1m_2y.csv"
    output_path = PROJECT_ROOT / "data" / "validation" / "regime_stress_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  MVHE REGIME CLASSIFICATION & STRESS TESTING")
    print("=" * 78)

    t0 = time.time()
    print(f"\n  Loading candles from {data_path.name}...")
    candles = load_1m_fast(data_path)
    print(f"  Loaded {len(candles):,} candles in {time.time() - t0:.1f}s")
    print(f"  Range: {candles[0].ts} to {candles[-1].ts}")

    t1 = time.time()
    windows = group_windows(candles)
    wd = precompute_windows(windows)
    print(f"  {len(wd):,} valid 15m windows in {time.time() - t1:.1f}s")

    all_results: dict[str, Any] = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "data_range": f"{candles[0].ts} to {candles[-1].ts}",
        "total_windows": len(wd),
        "strategies": {},
    }

    for strat_name, params in STRATEGIES.items():
        print(f"\n{'=' * 78}")
        print(f"  STRATEGY: {strat_name}")
        print(f"  Config: {params}")
        print(f"{'=' * 78}")

        t2 = time.time()
        trades, equity_curve = run_strategy(wd, **params)
        elapsed = time.time() - t2

        total_trades = len(trades)
        total_wins = sum(1 for t in trades if t.is_win)
        total_pnl = sum(t.pnl for t in trades)
        win_rate = total_wins / total_trades if total_trades > 0 else 0

        print(f"\n  Summary: {total_trades:,} trades | {win_rate:.1%} win rate | "
              f"${total_pnl:,.2f} net P&L | {elapsed:.1f}s")

        # 1. Regime
        print(f"\n  --- 1. REGIME CLASSIFICATION ---")
        regime_res = regime_analysis(trades, candles)
        print_regime(regime_res, strat_name)

        # 2. Monthly
        print(f"\n  --- 2. MONTHLY BREAKDOWN ---")
        monthly_res = monthly_analysis(trades)
        print_monthly(monthly_res, strat_name)

        # 3. Hourly
        print(f"\n  --- 3. HOURLY ANALYSIS ---")
        hourly_res = hourly_analysis(trades)
        print_hourly(hourly_res, strat_name)

        # 4. Drawdown
        print(f"\n  --- 4. DRAWDOWN ANALYSIS ---")
        dd_res = drawdown_analysis(trades, equity_curve)
        print_drawdown(dd_res, strat_name)

        # 5. Consecutive losses
        print(f"\n  --- 5. CONSECUTIVE LOSS ANALYSIS ---")
        consec_res = consecutive_loss_analysis(trades)
        print_consecutive(consec_res, strat_name)

        # 6. Worst periods
        print(f"\n  --- 6. WORST PERIOD ANALYSIS ---")
        worst_res = worst_period_analysis(trades)
        print_worst_period(worst_res, strat_name)

        # 7. Day of week
        print(f"\n  --- 7. DAY-OF-WEEK ANALYSIS ---")
        dow_res = day_of_week_analysis(trades)
        print_day_of_week(dow_res, strat_name)

        # Collect for JSON
        all_results["strategies"][strat_name] = {
            "config": params,
            "summary": {
                "total_trades": total_trades,
                "wins": total_wins,
                "win_rate": round(win_rate, 4),
                "net_pnl": round(total_pnl, 2),
            },
            "regime": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in regime_res.items()},
            "monthly": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in monthly_res.items()},
            "hourly": {str(k): {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in hourly_res.items()},
            "drawdown": dd_res,
            "consecutive_losses": consec_res,
            "worst_periods": worst_res,
            "day_of_week": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in dow_res.items()},
        }

    # Save JSON
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n{'=' * 78}")
    print(f"  Results saved to {output_path}")
    print(f"  Total time: {time.time() - t0:.1f}s")
    print(f"{'=' * 78}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
