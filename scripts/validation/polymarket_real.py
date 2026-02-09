#!/usr/bin/env python3
"""Validate trading strategies against REAL Polymarket resolution data.

Loads real Polymarket BTC 15m market resolutions, matches them to BTC 1m candle
data, runs strategy logic, and compares predicted vs actual outcomes.

Usage:
    python scripts/validation/polymarket_real.py
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# Suppress all logging (structlog, etc.)
logging.disable(logging.CRITICAL)
os.environ["STRUCTLOG_LEVEL"] = "CRITICAL"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Lightweight data classes (no Pydantic, no src/ imports)
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
class PolymarketMarket:
    event_id: int
    event_title: str
    market_id: int
    end_date: datetime
    closed_time: datetime
    volume: float
    resolution: str  # "Up" or "Down"


@dataclass
class TradeResult:
    market_id: int
    end_date: datetime
    resolution: str
    predicted_direction: str  # "Up" or "Down"
    is_win: bool
    entry_price: float
    pnl: float
    fee: float
    confidence: float
    entry_minute: int
    hour_utc: int
    volume: float
    cum_return_pct: float


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_btc_candles(path: Path) -> dict[str, MiniCandle]:
    """Load BTC 1m candles into a dict keyed by ISO timestamp string for fast lookup."""
    candles: dict[str, MiniCandle] = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_ts = row["timestamp"].strip()
            if raw_ts.endswith("Z"):
                raw_ts = raw_ts[:-1] + "+00:00"
            ts = datetime.fromisoformat(raw_ts)
            key = ts.strftime("%Y-%m-%dT%H:%M")
            candles[key] = MiniCandle(
                ts=ts,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
    return candles


def load_polymarket_markets(path: Path) -> list[PolymarketMarket]:
    """Load Polymarket BTC 15m resolution data."""
    markets: list[PolymarketMarket] = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            resolution = row["resolution"].strip()
            if resolution not in ("Up", "Down"):
                continue

            # Parse end_date
            raw_end = row["end_date"].strip()
            if raw_end.endswith("Z"):
                raw_end = raw_end[:-1] + "+00:00"
            end_date = datetime.fromisoformat(raw_end)

            # Parse closed_time
            raw_closed = row["closed_time"].strip()
            if raw_closed.endswith("Z"):
                raw_closed = raw_closed[:-1] + "+00:00"
            try:
                closed_time = datetime.fromisoformat(raw_closed)
            except ValueError:
                closed_time = end_date

            # Parse volume
            try:
                volume = float(row["volume"])
            except (ValueError, KeyError):
                volume = 0.0

            markets.append(PolymarketMarket(
                event_id=int(row["event_id"]),
                event_title=row["event_title"],
                market_id=int(row["market_id"]),
                end_date=end_date,
                closed_time=closed_time,
                volume=volume,
                resolution=resolution,
            ))
    return markets


def get_window_candles(
    candle_map: dict[str, MiniCandle],
    window_end: datetime,
) -> list[MiniCandle]:
    """Get 15 one-minute candles for the window ending at window_end.

    The window covers [end - 15min, end - 1min] (minutes 0-14).
    """
    candles: list[MiniCandle] = []
    window_start = window_end - timedelta(minutes=15)
    for i in range(15):
        ts = window_start + timedelta(minutes=i)
        key = ts.strftime("%Y-%m-%dT%H:%M")
        if key in candle_map:
            candles.append(candle_map[key])
    return candles


# ---------------------------------------------------------------------------
# Strategy logic (copied from fast_optimize.py — no src/ imports)
# ---------------------------------------------------------------------------

HOUR_MULTIPLIERS = {
    0: 0.95, 1: 0.68, 2: 0.80, 3: 0.85, 4: 0.90, 5: 1.10,
    6: 1.05, 7: 1.05, 8: 0.95, 9: 1.10, 10: 1.15, 11: 1.10,
    12: 0.90, 13: 0.80, 14: 1.25, 15: 1.15, 16: 1.10, 17: 1.05,
    18: 0.95, 19: 0.90, 20: 0.88, 21: 0.90, 22: 0.85, 23: 0.92,
}


def sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x / 0.07))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


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


@dataclass
class StrategyConfig:
    name: str
    entry_start: int
    entry_end: int
    min_confidence: float
    min_signals: int
    weight_momentum: float
    use_time_filter: bool
    use_last3_bonus: bool


BALANCED = StrategyConfig(
    name="BALANCED",
    entry_start=8, entry_end=12,
    min_confidence=0.50, min_signals=2,
    weight_momentum=0.30, use_time_filter=True,
    use_last3_bonus=True,
)

HIGH_ACCURACY = StrategyConfig(
    name="HIGH_ACCURACY",
    entry_start=8, entry_end=12,
    min_confidence=0.70, min_signals=2,
    weight_momentum=0.80, use_time_filter=False,
    use_last3_bonus=True,
)


def run_strategy_on_window(
    candles: list[MiniCandle],
    config: StrategyConfig,
    balance: float = 10000.0,
    position_pct: float = 0.02,
) -> TradeResult | None:
    """Run strategy on a single 15m window. Returns TradeResult or None if no trade."""
    if len(candles) < 5:
        return None

    w_open = candles[0].open
    if w_open == 0:
        return None

    hour_utc = candles[0].ts.hour
    tiers = make_tiers(config.entry_start, config.entry_end)

    # Precompute cumulative returns, directions, last3 agreement
    cum_returns_pct: list[float] = []
    cum_directions: list[int] = []
    last3_agree: list[bool] = []

    for i, candle in enumerate(candles):
        cr = (candle.close - w_open) / w_open
        cum_returns_pct.append(abs(cr) * 100.0)
        cum_directions.append(1 if cr >= 0 else -1)

        if i >= 3:
            closes = [candles[j].close for j in range(i - 2, i + 1)]
            opens = [candles[j].open for j in range(i - 2, i + 1)]
            changes = [1 if c >= o else -1 for c, o in zip(closes, opens)]
            agree = (
                all(d == changes[0] for d in changes)
                and changes[0] == cum_directions[-1]
            )
            last3_agree.append(agree)
        else:
            last3_agree.append(False)

    n_minutes = len(candles)

    for minute in range(config.entry_start, min(config.entry_end + 1, n_minutes)):
        threshold = tiers.get(minute)
        if threshold is None:
            continue

        cum_pct = cum_returns_pct[minute]
        direction = cum_directions[minute]

        if cum_pct < threshold:
            continue

        # Skip flat candles
        if cum_pct == 0.0:
            continue

        # Count DIRECTIONAL signals only
        n_directional = 0
        total_signal_strength = 0.0
        neutral_strength = 0.0

        # Signal 1: Momentum (directional)
        momentum_strength = min(cum_pct / 0.25, 1.0)
        n_directional += 1
        total_signal_strength += momentum_strength * config.weight_momentum

        # Signal 2: OFI proxy (directional)
        ofi_strength = min(cum_pct / 0.15, 1.0)
        n_directional += 1
        total_signal_strength += ofi_strength * (1 - config.weight_momentum) * 0.4

        # Signal 3: Time-of-day — NEUTRAL meta-signal
        if config.use_time_filter:
            hour_mult = HOUR_MULTIPLIERS.get(hour_utc, 1.0)
            if hour_mult >= 0.75:
                time_strength = min(hour_mult / 1.25, 1.0)
                neutral_strength += time_strength * (1 - config.weight_momentum) * 0.3

        # Signal 4: Last-3 agreement (directional)
        if config.use_last3_bonus and minute < len(last3_agree) and last3_agree[minute]:
            n_directional += 1
            total_signal_strength += 0.8 * (1 - config.weight_momentum) * 0.3

        # Effective min with floor of 2
        effective_min = max(min(config.min_signals, n_directional), 2)
        if n_directional < effective_min:
            continue

        # Include neutral in confidence
        total_signal_strength += neutral_strength
        n_total = n_directional + (1 if neutral_strength > 0 else 0)

        # Compute confidence
        w_total = config.weight_momentum + (1 - config.weight_momentum) * min(n_total / 4, 1.0)
        confidence = total_signal_strength / w_total if w_total > 0 else 0
        signal_mult = min(n_total / 3.0, 1.5)
        confidence = min(confidence * signal_mult, 1.0)

        if confidence < config.min_confidence:
            continue

        # Execute trade
        cum_return = cum_returns_pct[minute] / 100.0 * direction
        yes_price = sigmoid(cum_return)
        entry_price = yes_price if direction == 1 else (1.0 - yes_price)
        if entry_price <= 0 or entry_price >= 1:
            continue

        pos_size = balance * position_pct
        if pos_size <= 0:
            continue

        fee = polymarket_fee(pos_size, entry_price)
        quantity = pos_size / entry_price

        predicted_direction = "Up" if direction == 1 else "Down"

        return TradeResult(
            market_id=0,  # filled by caller
            end_date=candles[-1].ts,
            resolution="",  # filled by caller
            predicted_direction=predicted_direction,
            is_win=False,  # filled by caller
            entry_price=entry_price,
            pnl=0.0,  # filled by caller
            fee=fee,
            confidence=confidence,
            entry_minute=minute,
            hour_utc=hour_utc,
            volume=0.0,  # filled by caller
            cum_return_pct=cum_pct,
        )

    return None


# ---------------------------------------------------------------------------
# Validation engine
# ---------------------------------------------------------------------------

def run_validation(
    markets: list[PolymarketMarket],
    candle_map: dict[str, MiniCandle],
    config: StrategyConfig,
    initial_balance: float = 10000.0,
    fixed_pos_size: float = 200.0,
) -> dict[str, Any]:
    """Run strategy on all Polymarket markets and compare to real resolutions.

    Uses FIXED position size ($200) per trade to produce interpretable P&L.
    """
    trades: list[TradeResult] = []
    skipped_no_data = 0
    skipped_no_trade = 0
    balance = initial_balance
    equity_curve: list[float] = [balance]

    for market in markets:
        candles = get_window_candles(candle_map, market.end_date)
        if len(candles) < 5:
            skipped_no_data += 1
            continue

        # Use fixed position size for strategy signal detection
        result = run_strategy_on_window(candles, config, initial_balance, 0.02)
        if result is None:
            skipped_no_trade += 1
            continue

        # Fill in real data
        result.market_id = market.market_id
        result.end_date = market.end_date
        result.resolution = market.resolution
        result.volume = market.volume
        result.is_win = (result.predicted_direction == market.resolution)

        # Compute P&L with FIXED position size and real settlement
        pos_size = fixed_pos_size
        fee = polymarket_fee(pos_size, result.entry_price)
        result.fee = fee
        quantity = pos_size / result.entry_price
        settlement = 1.0 if result.is_win else 0.0
        pnl = (settlement - result.entry_price) * quantity - fee
        result.pnl = pnl

        balance += pnl
        trades.append(result)
        equity_curve.append(balance)

    # Compute metrics
    total_trades = len(trades)
    wins = sum(1 for t in trades if t.is_win)
    losses = total_trades - wins
    win_rate = wins / total_trades if total_trades > 0 else 0.0
    net_pnl = balance - initial_balance
    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    total_fees = sum(t.fee for t in trades)

    # Sharpe
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

    # --- Breakdown analyses ---

    # Win rate by hour
    hour_wins: dict[int, int] = defaultdict(int)
    hour_total: dict[int, int] = defaultdict(int)
    for t in trades:
        hour_total[t.hour_utc] += 1
        if t.is_win:
            hour_wins[t.hour_utc] += 1
    win_rate_by_hour = {
        h: round(hour_wins[h] / hour_total[h], 4) if hour_total[h] > 0 else 0
        for h in sorted(hour_total.keys())
    }
    trades_by_hour = {h: hour_total[h] for h in sorted(hour_total.keys())}

    # Win rate by month
    month_wins: dict[str, int] = defaultdict(int)
    month_total: dict[str, int] = defaultdict(int)
    for t in trades:
        key = t.end_date.strftime("%Y-%m")
        month_total[key] += 1
        if t.is_win:
            month_wins[key] += 1
    win_rate_by_month = {
        m: round(month_wins[m] / month_total[m], 4) if month_total[m] > 0 else 0
        for m in sorted(month_total.keys())
    }
    trades_by_month = {m: month_total[m] for m in sorted(month_total.keys())}

    # Volume quartile analysis
    volumes = sorted([t.volume for t in trades])
    if len(volumes) >= 4:
        q1 = volumes[len(volumes) // 4]
        q2 = volumes[len(volumes) // 2]
        q3 = volumes[3 * len(volumes) // 4]
    else:
        q1 = q2 = q3 = 0.0

    volume_tiers = {
        "Q1_low": {"min": 0, "max": q1, "wins": 0, "total": 0, "pnl": 0.0},
        "Q2": {"min": q1, "max": q2, "wins": 0, "total": 0, "pnl": 0.0},
        "Q3": {"min": q2, "max": q3, "wins": 0, "total": 0, "pnl": 0.0},
        "Q4_high": {"min": q3, "max": float("inf"), "wins": 0, "total": 0, "pnl": 0.0},
    }
    for t in trades:
        if t.volume <= q1:
            tier = "Q1_low"
        elif t.volume <= q2:
            tier = "Q2"
        elif t.volume <= q3:
            tier = "Q3"
        else:
            tier = "Q4_high"
        volume_tiers[tier]["total"] += 1
        volume_tiers[tier]["pnl"] += t.pnl
        if t.is_win:
            volume_tiers[tier]["wins"] += 1

    for tier_data in volume_tiers.values():
        tier_data["win_rate"] = round(
            tier_data["wins"] / tier_data["total"], 4
        ) if tier_data["total"] > 0 else 0.0
        tier_data["pnl"] = round(tier_data["pnl"], 2)
        if tier_data["max"] == float("inf"):
            tier_data["max"] = round(max(volumes), 2) if volumes else 0
        else:
            tier_data["max"] = round(tier_data["max"], 2)
        tier_data["min"] = round(tier_data["min"], 2)

    # Losing trades analysis
    losing_trades = [t for t in trades if not t.is_win]
    losing_analysis: list[dict[str, Any]] = []
    for t in losing_trades:
        losing_analysis.append({
            "market_id": t.market_id,
            "end_date": t.end_date.isoformat(),
            "resolution": t.resolution,
            "predicted": t.predicted_direction,
            "entry_minute": t.entry_minute,
            "cum_return_pct": round(t.cum_return_pct, 4),
            "confidence": round(t.confidence, 4),
            "pnl": round(t.pnl, 4),
            "volume": round(t.volume, 2),
            "hour_utc": t.hour_utc,
        })

    # Entry minute distribution
    minute_wins: dict[int, int] = defaultdict(int)
    minute_total: dict[int, int] = defaultdict(int)
    for t in trades:
        minute_total[t.entry_minute] += 1
        if t.is_win:
            minute_wins[t.entry_minute] += 1
    win_rate_by_minute = {
        m: round(minute_wins[m] / minute_total[m], 4) if minute_total[m] > 0 else 0
        for m in sorted(minute_total.keys())
    }
    trades_by_minute = {m: minute_total[m] for m in sorted(minute_total.keys())}

    return {
        "config": config.name,
        "params": {
            "entry_start": config.entry_start,
            "entry_end": config.entry_end,
            "min_confidence": config.min_confidence,
            "min_signals": config.min_signals,
            "weight_momentum": config.weight_momentum,
            "use_time_filter": config.use_time_filter,
            "use_last3_bonus": config.use_last3_bonus,
        },
        "summary": {
            "total_markets": len(markets),
            "skipped_no_data": skipped_no_data,
            "skipped_no_trade": skipped_no_trade,
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 4),
            "net_pnl": round(net_pnl, 2),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else "inf",
            "total_fees": round(total_fees, 2),
            "sharpe": round(sharpe, 4),
            "max_drawdown": round(max_dd, 4),
            "final_balance": round(balance, 2),
            "avg_pnl_per_trade": round(net_pnl / total_trades, 4) if total_trades > 0 else 0,
        },
        "breakdowns": {
            "win_rate_by_hour": win_rate_by_hour,
            "trades_by_hour": trades_by_hour,
            "win_rate_by_month": win_rate_by_month,
            "trades_by_month": trades_by_month,
            "win_rate_by_entry_minute": win_rate_by_minute,
            "trades_by_entry_minute": trades_by_minute,
            "volume_quartiles": volume_tiers,
        },
        "losing_trades": losing_analysis[:50],  # cap at 50 for output
        "total_losing_trades": len(losing_analysis),
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_results(results: dict[str, Any], backtest_wr: float | None = None) -> None:
    cfg = results["config"]
    s = results["summary"]
    bd = results["breakdowns"]

    print(f"\n{'=' * 72}")
    print(f"  {cfg} STRATEGY — REAL POLYMARKET VALIDATION")
    print(f"{'=' * 72}")
    print(f"  Total Polymarket markets    : {s['total_markets']:,}")
    print(f"  Skipped (no BTC data)       : {s['skipped_no_data']:,}")
    print(f"  Skipped (no trade signal)   : {s['skipped_no_trade']:,}")
    print(f"  Trades executed             : {s['total_trades']:,}")
    print(f"  Wins / Losses               : {s['wins']:,} / {s['losses']:,}")
    print(f"  REAL Win Rate               : {s['win_rate']:.2%}")
    if backtest_wr is not None:
        delta = s["win_rate"] - backtest_wr
        print(f"  Backtest Win Rate           : {backtest_wr:.2%}")
        print(f"  Delta (Real - Backtest)     : {delta:+.2%}")
    print(f"  Net P&L                     : ${s['net_pnl']:,.2f}")
    print(f"  Sharpe Ratio                : {s['sharpe']:.2f}")
    print(f"  Profit Factor               : {s['profit_factor']}")
    print(f"  Max Drawdown                : {s['max_drawdown']:.2%}")
    print(f"  Total Fees                  : ${s['total_fees']:,.2f}")
    print(f"  Avg P&L / Trade             : ${s['avg_pnl_per_trade']:.4f}")
    print(f"  Final Balance               : ${s['final_balance']:,.2f}")

    # Win rate by hour
    print(f"\n  {'--- Win Rate by Hour (UTC) ---':^50}")
    print(f"  {'Hour':>6}  {'Trades':>7}  {'WinRate':>8}")
    for h in sorted(bd["win_rate_by_hour"].keys(), key=int):
        wr = bd["win_rate_by_hour"][h]
        tr = bd["trades_by_hour"].get(h, 0)
        bar = "#" * int(wr * 40)
        print(f"  {h:>6}  {tr:>7}  {wr:>7.1%}  {bar}")

    # Win rate by month
    print(f"\n  {'--- Win Rate by Month ---':^50}")
    print(f"  {'Month':>8}  {'Trades':>7}  {'WinRate':>8}")
    for m in sorted(bd["win_rate_by_month"].keys()):
        wr = bd["win_rate_by_month"][m]
        tr = bd["trades_by_month"].get(m, 0)
        print(f"  {m:>8}  {tr:>7}  {wr:>7.1%}")

    # Win rate by entry minute
    print(f"\n  {'--- Win Rate by Entry Minute ---':^50}")
    print(f"  {'Minute':>8}  {'Trades':>7}  {'WinRate':>8}")
    for m in sorted(bd["win_rate_by_entry_minute"].keys(), key=int):
        wr = bd["win_rate_by_entry_minute"][m]
        tr = bd["trades_by_entry_minute"].get(m, 0)
        print(f"  {m:>8}  {tr:>7}  {wr:>7.1%}")

    # Volume analysis
    print(f"\n  {'--- Win Rate by Volume Quartile ---':^50}")
    print(f"  {'Tier':>10}  {'Range':>22}  {'Trades':>7}  {'WinRate':>8}  {'P&L':>10}")
    for tier_name, td in bd["volume_quartiles"].items():
        wr_str = f"{td['win_rate']:.1%}" if td["total"] > 0 else "N/A"
        rng = f"${td['min']:.0f}-${td['max']:.0f}"
        print(f"  {tier_name:>10}  {rng:>22}  {td['total']:>7}  {wr_str:>8}  ${td['pnl']:>9,.2f}")

    # Losing trades sample
    losers = results["losing_trades"]
    total_losers = results["total_losing_trades"]
    if losers:
        print(f"\n  {'--- Sample Losing Trades ---':^50}")
        print(f"  (showing {min(len(losers), 15)} of {total_losers} losses)")
        print(f"  {'Date':>20}  {'Pred':>5}  {'Real':>5}  {'Min':>4}  {'CumRet%':>8}  {'Conf':>6}  {'P&L':>9}  {'Vol':>10}")
        for t in losers[:15]:
            print(
                f"  {t['end_date'][:19]:>20}  {t['predicted']:>5}  {t['resolution']:>5}  "
                f"{t['entry_minute']:>4}  {t['cum_return_pct']:>7.3f}  {t['confidence']:>5.3f}  "
                f"${t['pnl']:>8.4f}  ${t['volume']:>9.2f}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 72)
    print("  POLYMARKET REAL DATA VALIDATION")
    print("  Comparing strategy predictions vs actual market resolutions")
    print("=" * 72)

    btc_path = PROJECT_ROOT / "data" / "btc_1m_2y.csv"
    poly_path = PROJECT_ROOT / "data" / "polymarket_btc_15m.csv"
    output_dir = PROJECT_ROOT / "data" / "validation"
    output_path = output_dir / "polymarket_real_results.json"

    if not btc_path.exists():
        print(f"  ERROR: BTC data not found at {btc_path}")
        return 1
    if not poly_path.exists():
        print(f"  ERROR: Polymarket data not found at {poly_path}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load BTC candles
    t0 = time.time()
    print(f"\n  Loading BTC 1m candles from {btc_path.name}...")
    candle_map = load_btc_candles(btc_path)
    print(f"  Loaded {len(candle_map):,} candles in {time.time() - t0:.1f}s")

    # Load Polymarket markets
    t1 = time.time()
    print(f"  Loading Polymarket markets from {poly_path.name}...")
    markets = load_polymarket_markets(poly_path)
    print(f"  Loaded {len(markets):,} markets in {time.time() - t1:.1f}s")

    # Date range info
    if markets:
        dates = [m.end_date for m in markets]
        print(f"  Polymarket range: {min(dates).date()} to {max(dates).date()}")

    # Resolution distribution
    up_count = sum(1 for m in markets if m.resolution == "Up")
    down_count = sum(1 for m in markets if m.resolution == "Down")
    print(f"  Resolution distribution: {up_count:,} Up ({up_count/len(markets):.1%}), "
          f"{down_count:,} Down ({down_count/len(markets):.1%})")

    # Volume stats
    volumes = [m.volume for m in markets]
    zero_vol = sum(1 for v in volumes if v == 0)
    non_zero = [v for v in volumes if v > 0]
    if non_zero:
        avg_vol = sum(non_zero) / len(non_zero)
        med_vol = sorted(non_zero)[len(non_zero) // 2]
        print(f"  Volume: avg=${avg_vol:,.1f}, median=${med_vol:,.1f}, "
              f"zero-volume={zero_vol:,} ({zero_vol/len(markets):.1%})")

    # Run both strategy configs
    # Backtest reference win rates from memory
    backtest_refs = {
        "BALANCED": 0.930,       # 93.0% from re-optimized singularity balanced
        "HIGH_ACCURACY": 0.971,  # 97.1% from re-optimized singularity high-accuracy
    }

    all_results: dict[str, Any] = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "btc_data": str(btc_path),
        "polymarket_data": str(poly_path),
        "total_markets": len(markets),
        "resolution_distribution": {"Up": up_count, "Down": down_count},
    }

    for config in [BALANCED, HIGH_ACCURACY]:
        print(f"\n  Running {config.name} strategy...")
        t2 = time.time()
        result = run_validation(markets, candle_map, config)
        elapsed = time.time() - t2
        print(f"  Completed in {elapsed:.1f}s")

        bt_wr = backtest_refs.get(config.name)
        print_results(result, backtest_wr=bt_wr)

        all_results[config.name.lower()] = result

    # Comparison summary
    print(f"\n{'=' * 72}")
    print(f"  SUMMARY COMPARISON")
    print(f"{'=' * 72}")
    print(f"  {'Metric':<28}  {'BALANCED':>14}  {'HIGH_ACCURACY':>14}  {'Backtest BAL':>14}  {'Backtest HA':>14}")
    print(f"  {'-'*28}  {'-'*14}  {'-'*14}  {'-'*14}  {'-'*14}")

    for key, label in [
        ("total_trades", "Trades"),
        ("win_rate", "Win Rate"),
        ("net_pnl", "Net P&L"),
        ("sharpe", "Sharpe"),
        ("max_drawdown", "Max Drawdown"),
        ("profit_factor", "Profit Factor"),
    ]:
        bal = all_results.get("balanced", {}).get("summary", {}).get(key, "N/A")
        ha = all_results.get("high_accuracy", {}).get("summary", {}).get(key, "N/A")

        if key == "win_rate":
            bal_s = f"{bal:.2%}" if isinstance(bal, (int, float)) else str(bal)
            ha_s = f"{ha:.2%}" if isinstance(ha, (int, float)) else str(ha)
            bt_bal = f"{backtest_refs['BALANCED']:.2%}"
            bt_ha = f"{backtest_refs['HIGH_ACCURACY']:.2%}"
        elif key in ("net_pnl",):
            bal_s = f"${bal:,.2f}" if isinstance(bal, (int, float)) else str(bal)
            ha_s = f"${ha:,.2f}" if isinstance(ha, (int, float)) else str(ha)
            bt_bal = "N/A"
            bt_ha = "N/A"
        elif key == "max_drawdown":
            bal_s = f"{bal:.2%}" if isinstance(bal, (int, float)) else str(bal)
            ha_s = f"{ha:.2%}" if isinstance(ha, (int, float)) else str(ha)
            bt_bal = "8.90%"
            bt_ha = "4.50%"
        elif key == "total_trades":
            bal_s = f"{bal:,}" if isinstance(bal, int) else str(bal)
            ha_s = f"{ha:,}" if isinstance(ha, int) else str(ha)
            bt_bal = "43,567"
            bt_ha = "13,860"
        else:
            bal_s = f"{bal}" if bal is not None else "N/A"
            ha_s = f"{ha}" if ha is not None else "N/A"
            bt_bal = "N/A"
            bt_ha = "N/A"

        print(f"  {label:<28}  {bal_s:>14}  {ha_s:>14}  {bt_bal:>14}  {bt_ha:>14}")

    # Save JSON
    # Convert any non-serializable types
    def default_serializer(obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, float) and math.isinf(obj):
            return "inf"
        return str(obj)

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=default_serializer)
    print(f"\n  Results saved to {output_path}")
    print(f"  Total time: {time.time() - t0:.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
