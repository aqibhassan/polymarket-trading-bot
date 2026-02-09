#!/usr/bin/env python3
"""Parameter sensitivity and robustness analysis for MVHE strategies.

Tests stability of BALANCED and HIGH_ACCURACY configs under perturbation,
transaction cost changes, position sizing, capital scaling, temporal splits,
and overfitting metrics.

Usage:
    python scripts/validation/sensitivity.py
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Suppress all structlog / logging noise
# ---------------------------------------------------------------------------
logging.disable(logging.CRITICAL)
os.environ["STRUCTLOG_LEVEL"] = "CRITICAL"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Data loading (copied from fast_optimize — lightweight, no src/ imports)
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
    ts: datetime  # timestamp of first candle (for temporal splits)


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
# Hour-of-day tables (from fast_optimize)
# ---------------------------------------------------------------------------

HOUR_MULTIPLIERS = {
    0: 0.95, 1: 0.68, 2: 0.80, 3: 0.85, 4: 0.90, 5: 1.10,
    6: 1.05, 7: 1.05, 8: 0.95, 9: 1.10, 10: 1.15, 11: 1.10,
    12: 0.90, 13: 0.80, 14: 1.25, 15: 1.15, 16: 1.10, 17: 1.05,
    18: 0.95, 19: 0.90, 20: 0.88, 21: 0.90, 22: 0.85, 23: 0.92,
}


# ---------------------------------------------------------------------------
# Core trial runner (copied from fast_optimize with fee/slippage overrides)
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
    final_balance: float
    return_pct: float = 0.0  # total return as percentage


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x / 0.07))


def polymarket_fee(pos_size: float, yes_price: float) -> float:
    p = yes_price
    return pos_size * 0.25 * p * p * (1 - p) * (1 - p)


def make_tiers(start: int, end: int, base_high: float = 0.14, base_low: float = 0.05) -> dict[int, float]:
    tiers: dict[int, float] = {}
    span = max(1, end - start)
    for m in range(start, end + 1):
        pos = (m - start) / span
        tiers[m] = round(base_high - pos * (base_high - base_low), 3)
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
    fee_multiplier: float = 1.0,
    slippage_bps: float = 0.0,
) -> TrialResult:
    """Run a single trial. Uses per-trade log returns to avoid float overflow."""
    balance = initial_balance
    trades = 0
    wins = 0
    per_trade_returns: list[float] = []  # per-trade % return on risked capital
    cum_log_return = 0.0  # sum of log(1 + r_trade * position_pct) for compounding
    max_dd = 0.0
    peak_cum = 0.0

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

            # Apply slippage to entry price
            if slippage_bps > 0:
                slip = slippage_bps / 10000.0
                entry_price = entry_price * (1 + slip)
                if entry_price >= 1:
                    continue

            # Compute per-unit PnL on the risked capital
            fee_rate = 0.25 * entry_price * entry_price * (1 - entry_price) * (1 - entry_price) * fee_multiplier
            is_win = direction == wd.true_dir
            settlement = 1.0 if is_win else 0.0
            # Return per dollar risked: (settlement - entry) / entry - fee_rate
            trade_return = (settlement - entry_price) / entry_price - fee_rate

            trades += 1
            if is_win:
                wins += 1
            per_trade_returns.append(trade_return)

            # Track compounding via log returns for drawdown
            portfolio_return = trade_return * position_pct
            # Clamp to avoid log(0) or log(negative)
            portfolio_return = max(portfolio_return, -0.999)
            cum_log_return += math.log(1 + portfolio_return)
            if cum_log_return > peak_cum:
                peak_cum = cum_log_return
            dd = 1 - math.exp(cum_log_return - peak_cum)
            if dd > max_dd:
                max_dd = dd

            traded = True

    # Compute Sharpe from per-trade returns (portfolio-level)
    sharpe = 0.0
    if len(per_trade_returns) >= 2:
        port_returns = [r * position_pct for r in per_trade_returns]
        mean_r = sum(port_returns) / len(port_returns)
        var_r = sum((r - mean_r) ** 2 for r in port_returns) / len(port_returns)
        std_r = math.sqrt(var_r) if var_r > 0 else 0
        if std_r > 0:
            # Annualize: ~96 windows/day * 252 trading days
            sharpe = (mean_r / std_r) * math.sqrt(252)

    # Final balance from log return (cap to avoid overflow)
    MAX_LOG = 700  # exp(700) ~ 1e304, near float64 max
    capped_log = min(cum_log_return, MAX_LOG)
    final_balance = initial_balance * math.exp(capped_log)
    net_pnl = final_balance - initial_balance
    wr = wins / trades if trades > 0 else 0

    # Profit factor from per-trade returns
    gross_profit = sum(r for r in per_trade_returns if r > 0)
    gross_loss_abs = sum(abs(r) for r in per_trade_returns if r < 0)
    pf = gross_profit / gross_loss_abs if gross_loss_abs > 0 else float("inf")

    return_pct = (math.exp(capped_log) - 1) * 100

    return TrialResult(
        params={},
        total_trades=trades,
        wins=wins,
        win_rate=wr,
        net_pnl=net_pnl,
        sharpe=sharpe,
        max_drawdown=max_dd,
        profit_factor=pf,
        avg_pnl=net_pnl / trades if trades > 0 else 0,
        final_balance=final_balance,
        return_pct=return_pct,
    )


# ---------------------------------------------------------------------------
# Strategy configs
# ---------------------------------------------------------------------------

BALANCED = {
    "entry_start": 8,
    "entry_end": 12,
    "min_confidence": 0.50,
    "min_signals": 2,
    "weight_momentum": 0.30,
    "use_time_filter": True,
    "use_last3_bonus": True,
}

HIGH_ACCURACY = {
    "entry_start": 8,
    "entry_end": 12,
    "min_confidence": 0.70,
    "min_signals": 2,
    "weight_momentum": 0.80,
    "use_time_filter": False,
    "use_last3_bonus": True,
}


def run_config(
    wd: list[WindowData],
    cfg: dict[str, Any],
    initial_balance: float = 10000.0,
    position_pct: float = 0.02,
    fee_multiplier: float = 1.0,
    slippage_bps: float = 0.0,
) -> TrialResult:
    tiers = make_tiers(cfg["entry_start"], cfg["entry_end"])
    return run_trial(
        wd,
        entry_start=cfg["entry_start"],
        entry_end=cfg["entry_end"],
        tier_thresholds=tiers,
        min_confidence=cfg["min_confidence"],
        min_signals=cfg["min_signals"],
        weight_momentum=cfg["weight_momentum"],
        use_time_filter=cfg["use_time_filter"],
        use_last3_bonus=cfg["use_last3_bonus"],
        initial_balance=initial_balance,
        position_pct=position_pct,
        fee_multiplier=fee_multiplier,
        slippage_bps=slippage_bps,
    )


# ---------------------------------------------------------------------------
# 1. Parameter Perturbation Analysis
# ---------------------------------------------------------------------------

def parameter_perturbation(wd: list[WindowData]) -> dict[str, Any]:
    print("\n" + "=" * 70)
    print("  1. PARAMETER PERTURBATION ANALYSIS (BALANCED config)")
    print("=" * 70)

    results: dict[str, Any] = {}

    # --- min_confidence ---
    conf_values = [0.25, 0.40, 0.45, 0.50, 0.55, 0.60, 0.75]
    print(f"\n  min_confidence perturbation (base=0.50):")
    print(f"  {'Value':>8}  {'Pct':>6}  {'Sharpe':>8}  {'WR%':>7}  {'Trades':>7}  {'MaxDD%':>7}  {'PF':>7}")
    print(f"  {'-'*62}")
    conf_results = []
    base_sharpe_conf = None
    for val in conf_values:
        cfg = {**BALANCED, "min_confidence": val}
        r = run_config(wd, cfg)
        pct_change = (val - 0.50) / 0.50 * 100
        marker = " <-- base" if val == 0.50 else ""
        print(f"  {val:>8.2f}  {pct_change:>+5.0f}%  {r.sharpe:>8.2f}  {r.win_rate*100:>6.1f}  "
              f"{r.total_trades:>7,}  {r.max_drawdown*100:>6.1f}  {r.profit_factor:>7.2f}{marker}")
        conf_results.append({"value": val, "sharpe": round(r.sharpe, 4),
                             "win_rate": round(r.win_rate, 4), "trades": r.total_trades,
                             "max_drawdown": round(r.max_drawdown, 4),
                             "profit_factor": round(r.profit_factor, 4)})
        if val == 0.50:
            base_sharpe_conf = r.sharpe

    # Stability score: avg Sharpe drop at +/-20%
    sharpe_at_040 = next(x["sharpe"] for x in conf_results if x["value"] == 0.40)
    sharpe_at_060 = next(x["sharpe"] for x in conf_results if x["value"] == 0.60)
    stability_conf = 0.0
    if base_sharpe_conf and base_sharpe_conf > 0:
        drop_lo = (base_sharpe_conf - sharpe_at_040) / base_sharpe_conf
        drop_hi = (base_sharpe_conf - sharpe_at_060) / base_sharpe_conf
        stability_conf = (abs(drop_lo) + abs(drop_hi)) / 2
    print(f"  Stability score (avg |Sharpe drop| at +/-20%): {stability_conf:.4f}")

    results["min_confidence"] = {
        "values": conf_results,
        "stability_score": round(stability_conf, 4),
    }

    # --- weight_momentum ---
    wm_values = [0.15, 0.24, 0.27, 0.30, 0.33, 0.36, 0.50]
    print(f"\n  weight_momentum perturbation (base=0.30):")
    print(f"  {'Value':>8}  {'Pct':>6}  {'Sharpe':>8}  {'WR%':>7}  {'Trades':>7}  {'MaxDD%':>7}  {'PF':>7}")
    print(f"  {'-'*62}")
    wm_results = []
    base_sharpe_wm = None
    for val in wm_values:
        cfg = {**BALANCED, "weight_momentum": val}
        r = run_config(wd, cfg)
        pct_change = (val - 0.30) / 0.30 * 100
        marker = " <-- base" if val == 0.30 else ""
        print(f"  {val:>8.2f}  {pct_change:>+5.0f}%  {r.sharpe:>8.2f}  {r.win_rate*100:>6.1f}  "
              f"{r.total_trades:>7,}  {r.max_drawdown*100:>6.1f}  {r.profit_factor:>7.2f}{marker}")
        wm_results.append({"value": val, "sharpe": round(r.sharpe, 4),
                           "win_rate": round(r.win_rate, 4), "trades": r.total_trades,
                           "max_drawdown": round(r.max_drawdown, 4),
                           "profit_factor": round(r.profit_factor, 4)})
        if val == 0.30:
            base_sharpe_wm = r.sharpe

    sharpe_at_024 = next(x["sharpe"] for x in wm_results if x["value"] == 0.24)
    sharpe_at_036 = next(x["sharpe"] for x in wm_results if x["value"] == 0.36)
    stability_wm = 0.0
    if base_sharpe_wm and base_sharpe_wm > 0:
        drop_lo = (base_sharpe_wm - sharpe_at_024) / base_sharpe_wm
        drop_hi = (base_sharpe_wm - sharpe_at_036) / base_sharpe_wm
        stability_wm = (abs(drop_lo) + abs(drop_hi)) / 2
    print(f"  Stability score (avg |Sharpe drop| at +/-20%): {stability_wm:.4f}")

    results["weight_momentum"] = {
        "values": wm_results,
        "stability_score": round(stability_wm, 4),
    }

    # --- threshold at minute 8 ---
    # Default tier at minute 8 with start=8, end=11 is 0.14 (the high end)
    threshold_values = [0.07, 0.09, 0.11, 0.14, 0.17, 0.19, 0.21]
    print(f"\n  threshold_at_min8 perturbation (base=0.14):")
    print(f"  {'Value':>8}  {'Pct':>6}  {'Sharpe':>8}  {'WR%':>7}  {'Trades':>7}  {'MaxDD%':>7}  {'PF':>7}")
    print(f"  {'-'*62}")
    thr_results = []
    base_sharpe_thr = None
    for val in threshold_values:
        # Scale all tiers proportionally: ratio = val / 0.14
        ratio = val / 0.14
        base_tiers = make_tiers(8, 11)
        custom_tiers = {m: round(t * ratio, 4) for m, t in base_tiers.items()}
        r = run_trial(
            wd,
            entry_start=BALANCED["entry_start"],
            entry_end=BALANCED["entry_end"],
            tier_thresholds=custom_tiers,
            min_confidence=BALANCED["min_confidence"],
            min_signals=BALANCED["min_signals"],
            weight_momentum=BALANCED["weight_momentum"],
            use_time_filter=BALANCED["use_time_filter"],
            use_last3_bonus=BALANCED["use_last3_bonus"],
        )
        pct_change = (val - 0.14) / 0.14 * 100
        marker = " <-- base" if val == 0.14 else ""
        print(f"  {val:>8.2f}  {pct_change:>+5.0f}%  {r.sharpe:>8.2f}  {r.win_rate*100:>6.1f}  "
              f"{r.total_trades:>7,}  {r.max_drawdown*100:>6.1f}  {r.profit_factor:>7.2f}{marker}")
        thr_results.append({"value": val, "sharpe": round(r.sharpe, 4),
                            "win_rate": round(r.win_rate, 4), "trades": r.total_trades,
                            "max_drawdown": round(r.max_drawdown, 4),
                            "profit_factor": round(r.profit_factor, 4)})
        if val == 0.14:
            base_sharpe_thr = r.sharpe

    sharpe_at_011 = next(x["sharpe"] for x in thr_results if x["value"] == 0.11)
    sharpe_at_017 = next(x["sharpe"] for x in thr_results if x["value"] == 0.17)
    stability_thr = 0.0
    if base_sharpe_thr and base_sharpe_thr > 0:
        drop_lo = (base_sharpe_thr - sharpe_at_011) / base_sharpe_thr
        drop_hi = (base_sharpe_thr - sharpe_at_017) / base_sharpe_thr
        stability_thr = (abs(drop_lo) + abs(drop_hi)) / 2
    print(f"  Stability score (avg |Sharpe drop| at ~+/-20%): {stability_thr:.4f}")

    results["threshold_at_min8"] = {
        "values": thr_results,
        "stability_score": round(stability_thr, 4),
    }

    return results


# ---------------------------------------------------------------------------
# 2. Transaction Cost Sensitivity
# ---------------------------------------------------------------------------

def transaction_cost_sensitivity(wd: list[WindowData]) -> dict[str, Any]:
    print("\n" + "=" * 70)
    print("  2. TRANSACTION COST SENSITIVITY")
    print("=" * 70)

    results: dict[str, Any] = {}

    # Fee multiplier sweep
    fee_mults = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    print(f"\n  Fee multiplier sweep (BALANCED):")
    print(f"  {'FeeMult':>8}  {'Sharpe':>8}  {'WR%':>7}  {'Trades':>7}  {'MaxDD%':>7}  {'PF':>7}")
    print(f"  {'-'*54}")
    fee_results = []
    for fm in fee_mults:
        r = run_config(wd, BALANCED, fee_multiplier=fm)
        marker = " <-- base" if fm == 1.0 else ""
        print(f"  {fm:>7.1f}x  {r.sharpe:>8.2f}  {r.win_rate*100:>6.1f}  "
              f"{r.total_trades:>7,}  {r.max_drawdown*100:>6.1f}  {r.profit_factor:>7.2f}{marker}")
        fee_results.append({"fee_mult": fm, "sharpe": round(r.sharpe, 4),
                            "win_rate": round(r.win_rate, 4), "trades": r.total_trades,
                            "max_drawdown": round(r.max_drawdown, 4),
                            "profit_factor": round(r.profit_factor, 4)})
    results["fee_multiplier"] = fee_results

    # Slippage sweep
    slip_values = [0, 5, 10, 20, 50]
    print(f"\n  Slippage sweep in bps (BALANCED):")
    print(f"  {'Slip(bps)':>10}  {'Sharpe':>8}  {'WR%':>7}  {'Trades':>7}  {'MaxDD%':>7}  {'PF':>7}")
    print(f"  {'-'*56}")
    slip_results = []
    for sl in slip_values:
        r = run_config(wd, BALANCED, slippage_bps=float(sl))
        marker = " <-- base" if sl == 0 else ""
        print(f"  {sl:>9}  {r.sharpe:>8.2f}  {r.win_rate*100:>6.1f}  "
              f"{r.total_trades:>7,}  {r.max_drawdown*100:>6.1f}  {r.profit_factor:>7.2f}{marker}")
        slip_results.append({"slippage_bps": sl, "sharpe": round(r.sharpe, 4),
                             "win_rate": round(r.win_rate, 4), "trades": r.total_trades,
                             "max_drawdown": round(r.max_drawdown, 4),
                             "profit_factor": round(r.profit_factor, 4)})
    results["slippage"] = slip_results

    return results


# ---------------------------------------------------------------------------
# 3. Position Size Sensitivity
# ---------------------------------------------------------------------------

def position_size_sensitivity(wd: list[WindowData]) -> dict[str, Any]:
    print("\n" + "=" * 70)
    print("  3. POSITION SIZE SENSITIVITY (BALANCED)")
    print("=" * 70)

    sizes = [0.005, 0.01, 0.02, 0.03, 0.05, 0.10]
    print(f"\n  {'Size%':>6}  {'Sharpe':>8}  {'WR%':>7}  {'Trades':>7}  "
          f"{'LogRet':>8}  {'MaxDD%':>7}  {'PF':>7}")
    print(f"  {'-'*62}")
    results_list = []
    for sz in sizes:
        r = run_config(wd, BALANCED, position_pct=sz)
        marker = " <-- base" if sz == 0.02 else ""
        # Use log10 of multiplier for display: log10(final/initial)
        if r.final_balance > 0 and r.final_balance > r.params.get("_ib", 10000):
            log_mult = math.log10(r.final_balance / 10000) if r.final_balance > 10000 else 0
        else:
            log_mult = 0
        # Recompute from cum_log_return stored in return_pct
        # return_pct = (exp(cum_log) - 1) * 100, so cum_log = log(return_pct/100 + 1)
        # But for display, just show the log10 of the multiplier
        log10_mult = math.log10(1 + r.return_pct / 100) if r.return_pct > 0 else 0
        print(f"  {sz*100:>5.1f}%  {r.sharpe:>8.2f}  {r.win_rate*100:>6.1f}  "
              f"{r.total_trades:>7,}  10^{log10_mult:>.0f}  "
              f"{r.max_drawdown*100:>6.1f}%  {r.profit_factor:>7.2f}{marker}")
        results_list.append({
            "position_pct": sz, "sharpe": round(r.sharpe, 4),
            "win_rate": round(r.win_rate, 4), "trades": r.total_trades,
            "log10_multiplier": round(log10_mult, 1),
            "max_drawdown": round(r.max_drawdown, 4),
            "profit_factor": round(r.profit_factor, 4),
        })

    # Find optimal by Sharpe
    best = max(results_list, key=lambda x: x["sharpe"])
    print(f"\n  Optimal position size by Sharpe: {best['position_pct']*100:.1f}% "
          f"(Sharpe={best['sharpe']:.2f}, MaxDD={best['max_drawdown']*100:.1f}%)")

    return {"results": results_list, "optimal_by_sharpe": best["position_pct"]}


# ---------------------------------------------------------------------------
# 4. Starting Capital Sensitivity
# ---------------------------------------------------------------------------

def starting_capital_sensitivity(wd: list[WindowData]) -> dict[str, Any]:
    print("\n" + "=" * 70)
    print("  4. STARTING CAPITAL SENSITIVITY (BALANCED)")
    print("=" * 70)

    capitals = [1_000, 5_000, 10_000, 50_000, 100_000]
    print(f"\n  {'Capital':>10}  {'Sharpe':>8}  {'WR%':>7}  {'Trades':>7}  "
          f"{'MaxDD%':>7}  {'PF':>7}")
    print(f"  {'-'*55}")
    results_list = []
    for cap in capitals:
        r = run_config(wd, BALANCED, initial_balance=float(cap))
        marker = " <-- base" if cap == 10_000 else ""
        print(f"  ${cap:>9,}  {r.sharpe:>8.2f}  {r.win_rate*100:>6.1f}  "
              f"{r.total_trades:>7,}  {r.max_drawdown*100:>6.1f}%  {r.profit_factor:>7.2f}{marker}")
        results_list.append({
            "capital": cap, "sharpe": round(r.sharpe, 4),
            "win_rate": round(r.win_rate, 4), "trades": r.total_trades,
            "max_drawdown": round(r.max_drawdown, 4),
            "profit_factor": round(r.profit_factor, 4),
        })

    # Check scale invariance
    sharpes = [x["sharpe"] for x in results_list]
    sharpe_range = max(sharpes) - min(sharpes) if sharpes else 0
    wr_vals = [x["win_rate"] for x in results_list]
    wr_range = max(wr_vals) - min(wr_vals) if wr_vals else 0
    is_scale_invariant = sharpe_range < 0.5 and wr_range < 0.01

    print(f"\n  Scale invariance check:")
    print(f"    Sharpe range across capitals: {sharpe_range:.4f} {'(PASS)' if sharpe_range < 0.5 else '(FAIL)'}")
    print(f"    Win rate range: {wr_range:.4f} {'(PASS)' if wr_range < 0.01 else '(FAIL)'}")
    print(f"    Scale invariant: {'YES' if is_scale_invariant else 'NO'}")

    return {
        "results": results_list,
        "scale_invariant": is_scale_invariant,
        "sharpe_range": round(sharpe_range, 4),
    }


# ---------------------------------------------------------------------------
# 5. Data Recency Bias Test (4 x 6-month quarters)
# ---------------------------------------------------------------------------

def data_recency_test(wd: list[WindowData]) -> dict[str, Any]:
    print("\n" + "=" * 70)
    print("  5. DATA RECENCY BIAS TEST (4 x 6-month quarters)")
    print("=" * 70)

    # Sort by timestamp and split into 4 equal quarters
    sorted_wd = sorted(wd, key=lambda w: w.ts)
    n = len(sorted_wd)
    q_size = n // 4
    quarters = [
        sorted_wd[:q_size],
        sorted_wd[q_size:2*q_size],
        sorted_wd[2*q_size:3*q_size],
        sorted_wd[3*q_size:],
    ]

    results: dict[str, Any] = {}
    for config_name, cfg in [("BALANCED", BALANCED), ("HIGH_ACCURACY", HIGH_ACCURACY)]:
        print(f"\n  {config_name}:")
        print(f"  {'Quarter':>10}  {'Period':>30}  {'Sharpe':>8}  {'WR%':>7}  "
              f"{'Trades':>7}  {'MaxDD%':>7}  {'PF':>7}")
        print(f"  {'-'*84}")
        q_results = []
        for i, q in enumerate(quarters):
            r = run_config(q, cfg)
            period = f"{q[0].ts.strftime('%Y-%m')} to {q[-1].ts.strftime('%Y-%m')}"
            print(f"  {'Q'+str(i+1):>10}  {period:>30}  {r.sharpe:>8.2f}  {r.win_rate*100:>6.1f}  "
                  f"{r.total_trades:>7,}  {r.max_drawdown*100:>6.1f}  {r.profit_factor:>7.2f}")
            q_results.append({
                "quarter": f"Q{i+1}",
                "period": period,
                "windows": len(q),
                "sharpe": round(r.sharpe, 4),
                "win_rate": round(r.win_rate, 4),
                "trades": r.total_trades,
                "max_drawdown": round(r.max_drawdown, 4),
                "profit_factor": round(r.profit_factor, 4),
            })

        sharpes = [q["sharpe"] for q in q_results]
        wrs = [q["win_rate"] for q in q_results]

        # Check for degradation trend (linear regression on quarter index vs sharpe)
        n_q = len(sharpes)
        x_mean = (n_q - 1) / 2
        y_mean = sum(sharpes) / n_q
        num = sum((i - x_mean) * (s - y_mean) for i, s in enumerate(sharpes))
        den = sum((i - x_mean) ** 2 for i in range(n_q))
        slope = num / den if den > 0 else 0

        trend = "IMPROVING" if slope > 0.5 else ("DEGRADING" if slope < -0.5 else "STABLE")
        print(f"  Sharpe trend slope: {slope:+.4f} ({trend})")
        print(f"  Win rate range: {min(wrs):.4f} - {max(wrs):.4f}")

        results[config_name] = {
            "quarters": q_results,
            "sharpe_trend_slope": round(slope, 4),
            "trend": trend,
            "sharpe_std": round(
                math.sqrt(sum((s - y_mean) ** 2 for s in sharpes) / n_q), 4
            ),
        }

    return results


# ---------------------------------------------------------------------------
# 6. Overfitting Score
# ---------------------------------------------------------------------------

def overfitting_analysis(wd: list[WindowData]) -> dict[str, Any]:
    print("\n" + "=" * 70)
    print("  6. OVERFITTING ANALYSIS")
    print("=" * 70)

    sorted_wd = sorted(wd, key=lambda w: w.ts)
    n = len(sorted_wd)
    split_idx = int(n * 0.7)
    is_data = sorted_wd[:split_idx]
    oos_data = sorted_wd[split_idx:]

    print(f"\n  In-sample:  {len(is_data):,} windows "
          f"({is_data[0].ts.strftime('%Y-%m')} to {is_data[-1].ts.strftime('%Y-%m')})")
    print(f"  Out-of-sample: {len(oos_data):,} windows "
          f"({oos_data[0].ts.strftime('%Y-%m')} to {oos_data[-1].ts.strftime('%Y-%m')})")

    results: dict[str, Any] = {}

    for config_name, cfg in [("BALANCED", BALANCED), ("HIGH_ACCURACY", HIGH_ACCURACY)]:
        r_is = run_config(is_data, cfg)
        r_oos = run_config(oos_data, cfg)

        overfit_ratio = r_is.sharpe / r_oos.sharpe if r_oos.sharpe > 0 else float("inf")

        # Deflated Sharpe Ratio approximation
        # Accounts for multiple testing: DSR = Sharpe_OOS * (1 - gamma * ln(N_trials))
        # We use N_trials from the optimization grid in fast_optimize (~1,296 combos)
        n_trials = 4608
        # Bailey & Lopez de Prado (2014) deflation factor
        # DSR ≈ OOS_Sharpe - sqrt(variance_of_sharpe) * expected_max_of_N_normals
        # E[max(N normals)] ≈ sqrt(2 * ln(N))
        expected_max_normal = math.sqrt(2 * math.log(n_trials)) if n_trials > 1 else 0
        # Variance of Sharpe estimator ≈ 1/T + (Sharpe^2)/(2T) where T = n_trades
        T = r_oos.total_trades
        if T > 0:
            var_sharpe = (1 / T) + (r_oos.sharpe ** 2) / (2 * T)
            std_sharpe = math.sqrt(var_sharpe)
        else:
            std_sharpe = 0
        deflated_sharpe = r_oos.sharpe - std_sharpe * expected_max_normal

        pass_overfit = overfit_ratio < 1.5
        pass_dsr = deflated_sharpe > 0

        print(f"\n  {config_name}:")
        print(f"    IS  Sharpe: {r_is.sharpe:>8.2f}  WR: {r_is.win_rate*100:>5.1f}%  Trades: {r_is.total_trades:,}")
        print(f"    OOS Sharpe: {r_oos.sharpe:>8.2f}  WR: {r_oos.win_rate*100:>5.1f}%  Trades: {r_oos.total_trades:,}")
        print(f"    Overfit Ratio (IS/OOS): {overfit_ratio:.4f} {'(PASS < 1.5)' if pass_overfit else '(FAIL >= 1.5)'}")
        print(f"    Deflated Sharpe Ratio:  {deflated_sharpe:.4f} {'(PASS > 0)' if pass_dsr else '(FAIL <= 0)'}")
        print(f"    (N_trials={n_trials}, E[max]={expected_max_normal:.2f}, std_sharpe={std_sharpe:.4f})")

        results[config_name] = {
            "is_sharpe": round(r_is.sharpe, 4),
            "is_win_rate": round(r_is.win_rate, 4),
            "is_trades": r_is.total_trades,
            "oos_sharpe": round(r_oos.sharpe, 4),
            "oos_win_rate": round(r_oos.win_rate, 4),
            "oos_trades": r_oos.total_trades,
            "overfit_ratio": round(overfit_ratio, 4),
            "deflated_sharpe": round(deflated_sharpe, 4),
            "n_trials_tested": n_trials,
            "pass_overfit": pass_overfit,
            "pass_deflated_sharpe": pass_dsr,
        }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    data_path = PROJECT_ROOT / "data" / "btc_1m_2y.csv"
    output_dir = PROJECT_ROOT / "data" / "validation"
    output_path = output_dir / "sensitivity_results.json"

    print("=" * 70)
    print("  MVHE PARAMETER SENSITIVITY & ROBUSTNESS ANALYSIS")
    print("=" * 70)

    t0 = time.time()
    print(f"\n  Loading candles from {data_path}...")
    candles = load_1m_fast(data_path)
    print(f"  Loaded {len(candles):,} candles in {time.time() - t0:.1f}s")
    print(f"  Range: {candles[0].ts} to {candles[-1].ts}")

    t1 = time.time()
    windows = group_windows(candles)
    wd = precompute_windows(windows)
    print(f"  Pre-computed {len(wd):,} valid windows in {time.time() - t1:.1f}s")

    # Run baseline for both configs
    print(f"\n  BASELINE RESULTS:")
    print(f"  {'-'*50}")
    for name, cfg in [("BALANCED", BALANCED), ("HIGH_ACCURACY", HIGH_ACCURACY)]:
        r = run_config(wd, cfg)
        print(f"  {name:>15}: Sharpe={r.sharpe:.2f}  WR={r.win_rate*100:.1f}%  "
              f"Trades={r.total_trades:,}  MaxDD={r.max_drawdown*100:.1f}%  PF={r.profit_factor:.2f}")

    all_results: dict[str, Any] = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "data_file": str(data_path),
        "total_windows": len(wd),
        "configs": {
            "BALANCED": BALANCED,
            "HIGH_ACCURACY": HIGH_ACCURACY,
        },
    }

    # 1. Parameter perturbation
    t2 = time.time()
    all_results["parameter_perturbation"] = parameter_perturbation(wd)
    print(f"  [Section 1 completed in {time.time() - t2:.1f}s]")

    # 2. Transaction cost sensitivity
    t3 = time.time()
    all_results["transaction_cost"] = transaction_cost_sensitivity(wd)
    print(f"  [Section 2 completed in {time.time() - t3:.1f}s]")

    # 3. Position size sensitivity
    t4 = time.time()
    all_results["position_size"] = position_size_sensitivity(wd)
    print(f"  [Section 3 completed in {time.time() - t4:.1f}s]")

    # 4. Starting capital sensitivity
    t5 = time.time()
    all_results["starting_capital"] = starting_capital_sensitivity(wd)
    print(f"  [Section 4 completed in {time.time() - t5:.1f}s]")

    # 5. Data recency bias
    t6 = time.time()
    all_results["data_recency"] = data_recency_test(wd)
    print(f"  [Section 5 completed in {time.time() - t6:.1f}s]")

    # 6. Overfitting analysis
    t7 = time.time()
    all_results["overfitting"] = overfitting_analysis(wd)
    print(f"  [Section 6 completed in {time.time() - t7:.1f}s]")

    # Summary
    total_time = time.time() - t0
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    pp = all_results["parameter_perturbation"]
    stability_scores = [
        pp["min_confidence"]["stability_score"],
        pp["weight_momentum"]["stability_score"],
        pp["threshold_at_min8"]["stability_score"],
    ]
    avg_stability = sum(stability_scores) / len(stability_scores)

    of = all_results["overfitting"]
    print(f"\n  Avg parameter stability score: {avg_stability:.4f} "
          f"({'ROBUST' if avg_stability < 0.15 else 'SENSITIVE'})")
    print(f"  Scale invariant: {all_results['starting_capital']['scale_invariant']}")
    for name in ["BALANCED", "HIGH_ACCURACY"]:
        ofd = of[name]
        print(f"  {name}: Overfit ratio={ofd['overfit_ratio']:.2f} "
              f"{'PASS' if ofd['pass_overfit'] else 'FAIL'}, "
              f"DSR={ofd['deflated_sharpe']:.2f} "
              f"{'PASS' if ofd['pass_deflated_sharpe'] else 'FAIL'}")

    dr = all_results["data_recency"]
    for name in ["BALANCED", "HIGH_ACCURACY"]:
        print(f"  {name} temporal trend: {dr[name]['trend']} "
              f"(slope={dr[name]['sharpe_trend_slope']:+.4f})")

    all_results["summary"] = {
        "avg_parameter_stability": round(avg_stability, 4),
        "scale_invariant": all_results["starting_capital"]["scale_invariant"],
        "total_time_seconds": round(total_time, 1),
    }

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")
    print(f"  Total time: {total_time:.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
