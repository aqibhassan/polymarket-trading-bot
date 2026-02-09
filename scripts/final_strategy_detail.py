"""Final detailed analysis of the winning strategy: Late 8@0.10 > 9@0.08 > 10@0.05"""

from __future__ import annotations

import json
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
    window_start: str
    entry_minute: int
    direction: str
    entry_price: float
    settlement: float
    pnl: float
    pnl_dollar: float
    correct: bool
    cum_return: float
    threshold_used: float


TIERS = [(8, 0.0010), (9, 0.0008), (10, 0.0005)]


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
                cum_return=round(cum_ret, 6),
                threshold_used=threshold,
            ))
            break
    return trades


def main() -> None:
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

    trades = run_strategy(windows)
    n = len(trades)

    print("=" * 90)
    print("  FINAL STRATEGY: Late Tiered (8@0.10% > 9@0.08% > 10@0.05%)")
    print("  Hold to settlement | 132,480 real BTC 1m candles | 3 months")
    print("=" * 90)

    correct = sum(1 for t in trades if t.correct)
    pnls = [t.pnl for t in trades]
    total_dollar = sum(t.pnl_dollar for t in trades)

    print(f"\n  Trades: {n}")
    print(f"  Accuracy: {correct/n*100:.1f}% ({correct} correct / {n-correct} wrong)")
    print(f"  Avg entry price: ${sum(t.entry_price for t in trades)/n:.4f}")
    print(f"  EV per $100 trade: ${sum(pnls)/n*100:+.2f}")
    print(f"  Total P&L: ${total_dollar:+,.2f}")

    # Sharpe
    mean = sum(pnls) / n
    var = sum((p - mean) ** 2 for p in pnls) / n
    std = math.sqrt(var)
    sharpe = (mean / std) * math.sqrt(35040) if std > 0 else 0
    print(f"  Sharpe ratio: {sharpe:.1f}")

    # By entry minute
    print(f"\n  BY ENTRY MINUTE:")
    print(f"  {'Min':>5} {'Thresh':>8} {'Trades':>8} {'Acc%':>7} {'AvgEntry':>9} {'EV/trade':>9} {'Total$':>10}")
    print(f"  {'-'*60}")
    for m, th in TIERS:
        group = [t for t in trades if t.entry_minute == m]
        if not group:
            continue
        g_n = len(group)
        g_acc = sum(1 for t in group if t.correct) / g_n
        g_ev = sum(t.pnl for t in group) / g_n
        g_total = sum(t.pnl_dollar for t in group)
        g_entry = sum(t.entry_price for t in group) / g_n
        print(f"  {m:>5} {th*100:>7.2f}% {g_n:>8} {g_acc*100:>6.1f}% {g_entry:>9.4f} ${g_ev*100:>+7.2f} ${g_total:>+9,.0f}")

    # Equity curve
    equity = [POSITION_SIZE]
    for t in trades:
        equity.append(equity[-1] + t.pnl_dollar)
    peak = equity[0]
    max_dd = 0.0
    max_dd_dollar = 0.0
    for val in equity:
        if val > peak:
            peak = val
        dd = (peak - val) / peak if peak > 0 else 0
        dd_dollar = peak - val
        if dd > max_dd:
            max_dd = dd
            max_dd_dollar = dd_dollar

    # Consecutive losses
    max_streak = 0
    cur = 0
    for t in trades:
        if not t.correct:
            cur += 1
            max_streak = max(max_streak, cur)
        else:
            cur = 0

    print(f"\n  RISK METRICS:")
    print(f"  Max drawdown: {max_dd*100:.1f}% (${max_dd_dollar:,.0f})")
    print(f"  Max consecutive losses: {max_streak}")
    print(f"  Final equity: ${equity[-1]:,.0f} (from ${POSITION_SIZE:.0f})")

    # Win/loss stats
    wins = [t for t in trades if t.correct]
    losses = [t for t in trades if not t.correct]
    avg_win = sum(t.pnl for t in wins) / len(wins)
    avg_loss = sum(t.pnl for t in losses) / len(losses)

    print(f"\n  WIN/LOSS BREAKDOWN:")
    print(f"  Wins:   {len(wins):>5} ({len(wins)/n*100:.1f}%)  avg return: +{avg_win*100:.1f}%  avg $: +${avg_win*POSITION_SIZE:.2f}")
    print(f"  Losses: {len(losses):>5} ({len(losses)/n*100:.1f}%)  avg return: {avg_loss*100:.1f}%  avg $: ${avg_loss*POSITION_SIZE:.2f}")

    # Profit factor
    gross_profit = sum(t.pnl_dollar for t in wins)
    gross_loss = abs(sum(t.pnl_dollar for t in losses))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    print(f"  Profit factor: {pf:.2f}")

    # By direction
    print(f"\n  BY DIRECTION:")
    for d in ["YES", "NO"]:
        group = [t for t in trades if t.direction == d]
        if group:
            g_acc = sum(1 for t in group if t.correct) / len(group)
            g_ev = sum(t.pnl for t in group) / len(group)
            print(f"  {d:>5}: {len(group):>5} trades, {g_acc*100:.1f}% accuracy, ${g_ev*100:+.2f} EV")

    # Monthly breakdown
    print(f"\n  MONTHLY BREAKDOWN:")
    by_month: dict[str, list[Trade]] = {}
    for t in trades:
        month = t.window_start[:7]
        by_month.setdefault(month, []).append(t)

    print(f"  {'Month':<12} {'Trades':>8} {'Acc%':>7} {'P&L$':>10} {'Avg EV':>8}")
    print(f"  {'-'*48}")
    for month in sorted(by_month):
        group = by_month[month]
        g_acc = sum(1 for t in group if t.correct) / len(group)
        g_pnl = sum(t.pnl_dollar for t in group)
        g_ev = sum(t.pnl for t in group) / len(group)
        print(f"  {month:<12} {len(group):>8} {g_acc*100:>6.1f}% ${g_pnl:>+9,.0f} ${g_ev*100:>+6.2f}")

    # Trades per day
    trades_per_day = n / 92  # ~92 days in 3 months
    print(f"\n  TRADING FREQUENCY:")
    print(f"  Avg trades/day: {trades_per_day:.1f}")
    print(f"  Avg trades/hour: {trades_per_day/24:.1f}")
    print(f"  Coverage: {n/len(windows)*100:.1f}% of 15m windows traded")

    # Sample trades
    print(f"\n  SAMPLE TRADES (first 20):")
    print(f"  {'Window':<28} {'Min':>3} {'Dir':>4} {'Entry':>7} {'Settle':>7} {'PnL%':>8} {'$':>7} {'OK':>4}")
    print(f"  {'-'*75}")
    for t in trades[:20]:
        print(f"  {t.window_start:<28} {t.entry_minute:>3} {t.direction:>4} "
              f"{t.entry_price:>7.4f} {t.settlement:>7.1f} {t.pnl*100:>+7.1f}% "
              f"${t.pnl_dollar:>+6.1f} {'Y' if t.correct else 'N':>4}")

    # Save optimal config
    config = {
        "strategy": "momentum_confirmation",
        "mode": "hold_to_settlement",
        "tiers": [
            {"minute": 8, "threshold_pct": 0.10},
            {"minute": 9, "threshold_pct": 0.08},
            {"minute": 10, "threshold_pct": 0.05},
        ],
        "backtest_results": {
            "period": "2025-11-08 to 2026-02-07",
            "candles": len(all_candles),
            "windows": len(windows),
            "trades": n,
            "accuracy": round(correct / n, 4),
            "ev_per_100": round(sum(pnls) / n * 100, 2),
            "total_pnl": round(total_dollar, 2),
            "sharpe": round(sharpe, 1),
            "max_drawdown_pct": round(max_dd * 100, 1),
            "profit_factor": round(pf, 2),
        },
    }
    config_path = PROJECT_ROOT / "data" / "optimal_strategy.json"
    config_path.write_text(json.dumps(config, indent=2))
    print(f"\n  Config saved: {config_path}")

    print(f"\n{'='*90}")
    print(f"  STRATEGY: Enter at min 8 (0.10%), min 9 (0.08%), min 10 (0.05%)")
    print(f"  EXIT: Hold to settlement (binary $1/$0)")
    print(f"  RESULT: {correct/n*100:.1f}% accuracy, ${sum(pnls)/n*100:+.2f} EV, ${total_dollar:+,.0f} total")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
