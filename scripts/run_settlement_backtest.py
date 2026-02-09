"""Settlement-based backtest — the CORRECT Polymarket economics.

On Polymarket:
  - YES token settles to $1.00 if candle closes green, $0.00 if red
  - NO token settles to $1.00 if candle closes red, $0.00 if green
  - We buy the token aligned with BTC momentum direction
  - We can either hold to settlement or sell early at market price

This backtest tests two exit strategies:
  1. HOLD_TO_SETTLE: Buy token, hold to settlement -> binary $1.00 or $0.00
  2. EARLY_EXIT: Buy token, sell at minute 13 at market price -> continuous P&L

Both are tested across multiple entry configs.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtesting.data_loader import DataLoader

CSV_PATHS = [PROJECT_ROOT / "data" / f"btc_1m_chunk{i}.csv" for i in range(1, 7)]
MINUTES_PER_WINDOW = 15
POSITION_SIZE = 100.0  # dollars per trade
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
class SettlementTrade:
    window_start: str
    entry_minute: int
    direction: str       # "YES" or "NO"
    entry_price: float   # token price we bought at
    settlement: float    # 1.0 or 0.0
    pnl_pct: float
    pnl_dollar: float
    correct: bool
    cum_return_at_entry: float
    exit_strategy: str   # "settlement" or "early_exit"
    early_exit_price: float  # price at minute 13 if early exit


def run_settlement_strategy(
    windows: list[list[Any]],
    *,
    threshold: float,
    entry_minutes: list[int],
    exit_strategy: str = "settlement",
    early_exit_minute: int = 13,
) -> list[SettlementTrade]:
    trades: list[SettlementTrade] = []

    for window in windows:
        window_open = float(window[0].open)
        if window_open == 0:
            continue

        final_close = float(window[-1].close)
        actual_green = final_close > window_open

        for entry_min in entry_minutes:
            if entry_min >= MINUTES_PER_WINDOW:
                continue

            current_close = float(window[entry_min].close)
            cum_ret = (current_close - window_open) / window_open

            if abs(cum_ret) < threshold:
                continue

            # Follow momentum
            predict_green = cum_ret > 0

            # Token prices via sigmoid
            entry_yes_price = _sigmoid(cum_ret)

            if predict_green:
                direction = "YES"
                entry_price = entry_yes_price
            else:
                direction = "NO"
                entry_price = 1.0 - entry_yes_price

            if entry_price <= 0.01:  # skip extreme prices
                continue

            # Settlement outcome
            if predict_green:
                settlement = 1.0 if actual_green else 0.0
            else:
                settlement = 1.0 if not actual_green else 0.0

            correct = (predict_green == actual_green)

            # Early exit price at minute 13
            if early_exit_minute < MINUTES_PER_WINDOW:
                exit_cum_ret = (float(window[early_exit_minute].close) - window_open) / window_open
                exit_yes_price = _sigmoid(exit_cum_ret)
                if predict_green:
                    early_exit_price = exit_yes_price
                else:
                    early_exit_price = 1.0 - exit_yes_price
            else:
                early_exit_price = settlement

            # Choose exit based on strategy
            if exit_strategy == "settlement":
                exit_price = settlement
            else:
                exit_price = early_exit_price

            pnl_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0.0

            trades.append(SettlementTrade(
                window_start=str(window[0].timestamp),
                entry_minute=entry_min,
                direction=direction,
                entry_price=round(entry_price, 6),
                settlement=settlement,
                pnl_pct=round(pnl_pct, 6),
                pnl_dollar=round(pnl_pct * POSITION_SIZE, 4),
                correct=correct,
                cum_return_at_entry=round(cum_ret, 6),
                exit_strategy=exit_strategy,
                early_exit_price=round(early_exit_price, 6),
            ))
            break  # one trade per window

    return trades


def compute_metrics(trades: list[SettlementTrade]) -> dict[str, Any]:
    if not trades:
        return {"trades": 0}

    pnls = [t.pnl_pct for t in trades]
    dollar_pnls = [t.pnl_dollar for t in trades]
    n = len(trades)

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    n_wins = len(wins)
    n_losses = len(losses)

    correct = sum(1 for t in trades if t.correct)
    accuracy = correct / n

    avg_entry = sum(t.entry_price for t in trades) / n
    avg_win = sum(wins) / n_wins if n_wins else 0.0
    avg_loss = sum(losses) / n_losses if n_losses else 0.0

    # Equity curve
    equity = [POSITION_SIZE]
    for dp in dollar_pnls:
        equity.append(equity[-1] + dp)
    total_return = (equity[-1] - equity[0]) / equity[0]

    # Sharpe
    if n >= 2:
        mean_ret = sum(pnls) / n
        var = sum((r - mean_ret) ** 2 for r in pnls) / n
        std = math.sqrt(var)
        sharpe = (mean_ret / std) * math.sqrt(35040) if std > 0 else 0.0
    else:
        sharpe = 0.0

    # Max drawdown
    peak = equity[0]
    max_dd = 0.0
    for val in equity:
        if val > peak:
            peak = val
        dd = (peak - val) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    # Profit factor
    gp = sum(d for d in dollar_pnls if d > 0)
    gl = abs(sum(d for d in dollar_pnls if d < 0))
    pf = gp / gl if gl > 0 else float("inf")

    return {
        "trades": n,
        "wins": n_wins,
        "losses": n_losses,
        "win_rate": round(n_wins / n, 4),
        "accuracy": round(accuracy, 4),
        "avg_entry_price": round(avg_entry, 4),
        "avg_win_pct": round(avg_win * 100, 2),
        "avg_loss_pct": round(avg_loss * 100, 2),
        "total_return_pct": round(total_return * 100, 2),
        "total_dollar_pnl": round(equity[-1] - equity[0], 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "profit_factor": round(pf, 2) if pf != float("inf") else "inf",
        "ev_per_trade": round(sum(pnls) / n * 100, 2),
    }


def main() -> None:
    print("=" * 100)
    print("  MVHE Settlement Backtest — Real Polymarket Economics")
    print("  Binary settlement: correct direction -> $1.00, wrong -> $0.00")
    print("=" * 100)

    loader = DataLoader()
    all_candles: list[Any] = []
    for csv_path in CSV_PATHS:
        if not csv_path.exists():
            continue
        candles = loader.load_csv(csv_path)
        print(f"  Loaded {len(candles):>6} from {csv_path.name}")
        all_candles.extend(candles)

    all_candles.sort(key=lambda c: c.timestamp)
    seen: set[datetime] = set()
    unique = [c for c in all_candles if c.timestamp not in seen and not seen.add(c.timestamp)]  # type: ignore[func-returns-value]
    all_candles = unique

    print(f"\n  Total candles: {len(all_candles)}")
    windows = _group_into_15m_windows(all_candles)
    print(f"  15m windows: {len(windows)}")

    # Strategy configs to test
    configs = [
        # (name, threshold, entry_minutes, exit_strategy)
        ("Min5 0.05% Settle", 0.0005, [5], "settlement"),
        ("Min5 0.10% Settle", 0.0010, [5], "settlement"),
        ("Min5 0.10% Early", 0.0010, [5], "early_exit"),
        ("Min6 0.10% Settle", 0.0010, [6], "settlement"),
        ("Min6 0.15% Settle", 0.0015, [6], "settlement"),
        ("Min7 0.10% Settle", 0.0010, [7], "settlement"),
        ("Min7 0.15% Settle", 0.0015, [7], "settlement"),
        ("Min8 0.10% Settle", 0.0010, [8], "settlement"),
        ("Min8 0.10% Early", 0.0010, [8], "early_exit"),
        ("Min8 0.15% Settle", 0.0015, [8], "settlement"),
        ("Min10 0.10% Settle", 0.0010, [10], "settlement"),
        ("Min10 0.10% Early", 0.0010, [10], "early_exit"),
        ("Min12 0.10% Settle", 0.0010, [12], "settlement"),
    ]

    all_results: dict[str, dict[str, Any]] = {}
    all_trades: dict[str, list[SettlementTrade]] = {}

    for name, thresh, entry_mins, exit_strat in configs:
        trades = run_settlement_strategy(
            windows, threshold=thresh, entry_minutes=entry_mins, exit_strategy=exit_strat,
        )
        metrics = compute_metrics(trades)
        all_results[name] = metrics
        all_trades[name] = trades

    # Print comparison
    print("\n" + "=" * 100)
    print("  STRATEGY COMPARISON — Settlement vs Early Exit")
    print("=" * 100)
    print(f"  {'Config':<25} {'Trades':>7} {'Acc%':>7} {'WinR%':>7} {'AvgEntry':>9} "
          f"{'AvgWin%':>8} {'AvgLoss%':>9} {'EV/trade':>9} {'TotalRet%':>10} {'Sharpe':>7} {'PF':>6}")
    print(f"  {'-'*108}")

    for name, m in all_results.items():
        if m["trades"] == 0:
            continue
        print(f"  {name:<25} {m['trades']:>7} {m['accuracy']*100:>6.1f}% {m['win_rate']*100:>6.1f}% "
              f"{m['avg_entry_price']:>9.4f} {m['avg_win_pct']:>+7.1f}% {m['avg_loss_pct']:>+8.1f}% "
              f"${m['ev_per_trade']:>+7.2f} {m['total_return_pct']:>+9.1f}% {m['sharpe']:>7.1f} "
              f"{m['profit_factor']:>6}")

    # Show THE optimal config
    best_name = max(all_results, key=lambda n: all_results[n].get("total_dollar_pnl", 0))
    best = all_results[best_name]
    print(f"\n  BEST: {best_name}")
    print(f"    Total P&L: ${best['total_dollar_pnl']:+,.2f} on ${POSITION_SIZE:.0f}/trade")
    print(f"    Accuracy: {best['accuracy']*100:.1f}%")
    print(f"    Win rate: {best['win_rate']*100:.1f}%")
    print(f"    EV per trade: ${best['ev_per_trade']:+.2f}")

    # Sample trades from best
    print(f"\n  SAMPLE TRADES — {best_name} (first 20)")
    print(f"  {'Window':<28} {'Min':>3} {'Dir':>4} {'Entry':>7} {'Settle':>7} "
          f"{'PnL%':>8} {'$PnL':>7} {'Correct':>8}")
    print(f"  {'-'*85}")
    for t in all_trades[best_name][:20]:
        print(f"  {t.window_start:<28} {t.entry_minute:>3} {t.direction:>4} "
              f"{t.entry_price:>7.4f} {t.settlement:>7.1f} "
              f"{t.pnl_pct*100:>+7.2f}% ${t.pnl_dollar:>+6.1f} {'YES' if t.correct else 'NO':>8}")

    # Save results
    report_path = PROJECT_ROOT / "data" / "settlement_backtest_report.json"
    report = {
        "summary": {
            "total_candles": len(all_candles),
            "total_windows": len(windows),
            "best_strategy": best_name,
        },
        "results": all_results,
    }
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n  Report saved: {report_path}")

    print("\n" + "=" * 100)
    print("  KEY TAKEAWAY:")
    print(f"    At Min8 | 0.10% threshold: {all_results['Min8 0.10% Settle']['accuracy']*100:.1f}% directional accuracy")
    print(f"    Average entry price: ${all_results['Min8 0.10% Settle']['avg_entry_price']:.4f}")
    print(f"    EV per $100 trade: ${all_results['Min8 0.10% Settle']['ev_per_trade']:+.2f}")
    print(f"    Over {all_results['Min8 0.10% Settle']['trades']} trades: ${all_results['Min8 0.10% Settle']['total_dollar_pnl']:+,.2f} total P&L")
    print("=" * 100)


if __name__ == "__main__":
    main()
