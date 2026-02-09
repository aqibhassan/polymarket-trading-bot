"""Regime-based strategy validation over 2 years of BTC data.

Classifies each month into market regimes and tests strategy per regime.
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fast_loader import load_csv_fast, group_into_15m_windows as _group_15m

CSV_2Y = PROJECT_ROOT / "data" / "btc_1m_2y.csv"
MINUTES_PER_WINDOW = 15
POSITION_SIZE = 100.0
SENSITIVITY = 0.07
TIERS = [(8, 0.0010), (9, 0.0008), (10, 0.0005)]


def _sigmoid(cum_ret: float) -> float:
    x = SENSITIVITY * cum_ret * 10000
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class Trade:
    window_start: str
    entry_minute: int
    direction: str
    entry_price: float
    settlement: float
    pnl: float
    pnl_dollar: float
    correct: bool


def run_strategy(windows: list[list[Any]]) -> list[Trade]:
    trades: list[Trade] = []
    for window in windows:
        window_open = float(window[0].open)
        if window_open == 0:
            continue
        final_close = float(window[-1].close)
        actual_green = final_close > window_open

        for entry_min, threshold in TIERS:
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
            trades.append(Trade(
                window_start=str(window[0].timestamp),
                entry_minute=entry_min,
                direction="YES" if predict_green else "NO",
                entry_price=round(entry_price, 6),
                settlement=settlement,
                pnl=round(pnl, 6),
                pnl_dollar=round(pnl * POSITION_SIZE, 2),
                correct=correct,
            ))
            break
    return trades


def classify_regime(monthly_candles: list[Any]) -> str:
    """Classify a month's candles into a market regime."""
    if not monthly_candles:
        return "UNKNOWN"

    first_open = float(monthly_candles[0].open)
    last_close = float(monthly_candles[-1].close)
    monthly_return = (last_close - first_open) / first_open

    # Calculate daily ranges
    daily_ranges: list[float] = []
    day_candles: dict[str, list[Any]] = defaultdict(list)
    for c in monthly_candles:
        day_key = str(c.timestamp.date())
        day_candles[day_key].append(c)

    for day_key, candles in day_candles.items():
        day_high = max(float(c.high) for c in candles)
        day_low = min(float(c.low) for c in candles)
        day_open = float(candles[0].open)
        if day_open > 0:
            daily_ranges.append((day_high - day_low) / day_open)

    avg_daily_range = sum(daily_ranges) / len(daily_ranges) if daily_ranges else 0

    if monthly_return > 0.10:
        return "BULL"
    elif monthly_return < -0.10:
        return "BEAR"
    elif avg_daily_range > 0.03:
        return "HIGH_VOL"
    elif avg_daily_range < 0.01:
        return "LOW_VOL"
    else:
        return "SIDEWAYS"


def score_trades(trades: list[Trade], label: str) -> dict[str, Any]:
    if not trades:
        return {"label": label, "trades": 0}
    n = len(trades)
    correct = sum(1 for t in trades if t.correct)
    pnls = [t.pnl for t in trades]
    total = sum(t.pnl_dollar for t in trades)
    mean = sum(pnls) / n
    var = sum((p - mean) ** 2 for p in pnls) / n
    std = math.sqrt(var) if var > 0 else 0.001
    sharpe = (mean / std) * math.sqrt(35040)

    # Max drawdown
    equity = [POSITION_SIZE]
    for t in trades:
        equity.append(equity[-1] + t.pnl_dollar)
    peak = equity[0]
    max_dd = 0.0
    for val in equity:
        if val > peak:
            peak = val
        dd = (peak - val) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    return {
        "label": label,
        "trades": n,
        "accuracy": round(correct / n, 4),
        "ev_per_100": round(mean * 100, 2),
        "total_pnl": round(total, 2),
        "sharpe": round(sharpe, 1),
        "max_dd_pct": round(max_dd * 100, 1),
    }


def main() -> None:
    print("=" * 100)
    print("  REGIME-BASED STRATEGY VALIDATION (2 Years)")
    print("=" * 100)

    if not CSV_2Y.exists():
        print(f"  ERROR: {CSV_2Y} not found. Run download_btc_2y.py first.")
        sys.exit(1)

    all_candles = load_csv_fast(CSV_2Y)
    all_candles.sort(key=lambda c: c.timestamp)

    # Deduplicate
    seen: set[datetime] = set()
    unique = [c for c in all_candles if c.timestamp not in seen and not seen.add(c.timestamp)]  # type: ignore
    all_candles = unique

    print(f"\n  Loaded {len(all_candles):,} candles")
    if all_candles:
        print(f"  Range: {all_candles[0].timestamp} to {all_candles[-1].timestamp}")

    # Group candles by month
    by_month: dict[str, list[Any]] = defaultdict(list)
    for c in all_candles:
        month_key = c.timestamp.strftime("%Y-%m")
        by_month[month_key].append(c)

    # Classify regimes and run strategy per month
    print(f"\n  {'Month':<10} {'Regime':<10} {'Candles':>8} {'Windows':>8} {'Trades':>7} {'Acc%':>7} {'EV/100':>8} {'PnL$':>10} {'Sharpe':>7} {'MaxDD':>7}")
    print(f"  {'-'*90}")

    regime_trades: dict[str, list[Trade]] = defaultdict(list)
    all_monthly_results: list[dict[str, Any]] = []
    losing_months = 0

    for month in sorted(by_month):
        candles = by_month[month]
        regime = classify_regime(candles)
        windows = _group_15m(candles)
        trades = run_strategy(windows)

        regime_trades[regime].extend(trades)
        s = score_trades(trades, month)
        all_monthly_results.append(s)

        if s["trades"] > 0:
            is_loss = s["total_pnl"] < 0
            if is_loss:
                losing_months += 1
            marker = " *** LOSS" if is_loss else ""
            print(f"  {month:<10} {regime:<10} {len(candles):>8,} {len(windows):>8} {s['trades']:>7} "
                  f"{s['accuracy']*100:>6.1f}% ${s['ev_per_100']:>+6.2f} ${s['total_pnl']:>+9,.0f} "
                  f"{s['sharpe']:>6.1f} {s['max_dd_pct']:>5.1f}%{marker}")
        else:
            print(f"  {month:<10} {regime:<10} {len(candles):>8,} {len(windows):>8}   NO TRADES")

    # Per-regime summary
    print(f"\n  {'='*90}")
    print(f"  PER-REGIME SUMMARY")
    print(f"  {'='*90}")
    print(f"  {'Regime':<12} {'Trades':>7} {'Acc%':>7} {'EV/100':>8} {'PnL$':>12} {'Sharpe':>7} {'MaxDD':>7}")
    print(f"  {'-'*65}")

    for regime in ["BULL", "BEAR", "HIGH_VOL", "LOW_VOL", "SIDEWAYS"]:
        trades = regime_trades.get(regime, [])
        if trades:
            s = score_trades(trades, regime)
            print(f"  {regime:<12} {s['trades']:>7} {s['accuracy']*100:>6.1f}% ${s['ev_per_100']:>+6.2f} "
                  f"${s['total_pnl']:>+11,.0f} {s['sharpe']:>6.1f} {s['max_dd_pct']:>5.1f}%")

    # Overall
    all_trades: list[Trade] = []
    for trades in regime_trades.values():
        all_trades.extend(trades)
    all_trades.sort(key=lambda t: t.window_start)
    overall = score_trades(all_trades, "OVERALL")

    print(f"\n  {'OVERALL':<12} {overall['trades']:>7} {overall['accuracy']*100:>6.1f}% ${overall['ev_per_100']:>+6.2f} "
          f"${overall['total_pnl']:>+11,.0f} {overall['sharpe']:>6.1f} {overall['max_dd_pct']:>5.1f}%")

    total_months = len([r for r in all_monthly_results if r["trades"] > 0])
    print(f"\n  Monthly consistency: {total_months - losing_months}/{total_months} profitable months "
          f"({losing_months} losing)")

    # Worst month
    worst = min((r for r in all_monthly_results if r["trades"] > 0), key=lambda r: r["total_pnl"])
    print(f"  Worst month: {worst['label']} — ${worst['total_pnl']:+,.0f} ({worst['accuracy']*100:.1f}% acc)")
    best = max((r for r in all_monthly_results if r["trades"] > 0), key=lambda r: r["total_pnl"])
    print(f"  Best month:  {best['label']} — ${best['total_pnl']:+,.0f} ({best['accuracy']*100:.1f}% acc)")

    print(f"\n{'='*100}")


if __name__ == "__main__":
    main()
