"""Momentum Confirmation backtest on 1m BTC candle data.

Validates whether the momentum approach (follow cum_return direction)
achieves 80%+ win rate in a full backtest simulation with sigmoid
token pricing, take-profit, stop-loss, and time-based exits.

Three variants are tested:
  1. Balanced:      threshold 0.10%, entry min 5-8
  2. Conservative:  threshold 0.15%, entry min 6-8
  3. Filtered:      threshold 0.10%, entry min 5-8, require last_3_agree AND no_reversal
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtesting.data_loader import DataLoader

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CSV_PATHS = [PROJECT_ROOT / "data" / f"btc_1m_chunk{i}.csv" for i in range(1, 7)]
MINUTES_PER_WINDOW = 15
POSITION_SIZE = 100.0
SENSITIVITY = 0.07


# ---------------------------------------------------------------------------
# Sigmoid helper
# ---------------------------------------------------------------------------
def _sigmoid(cum_ret: float, sensitivity: float = SENSITIVITY) -> float:
    """Map cumulative BTC return to YES token price via sigmoid."""
    x = sensitivity * cum_ret * 10000
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------------
# Window grouping (same logic as analyze_predictors.py)
# ---------------------------------------------------------------------------
def _group_into_15m_windows(candles: list[Any]) -> list[list[Any]]:
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


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------
def _compute_cum_return(window: list[Any], minute_idx: int) -> float:
    """Cumulative return from window open to minute_idx close."""
    window_open = float(window[0].open)
    current_close = float(window[minute_idx].close)
    return (current_close - window_open) / window_open if window_open != 0 else 0.0


def _last_3_agree(window: list[Any], minute_idx: int, cum_return: float) -> bool:
    """Check if last 3 1m candles all match the cum_return direction."""
    if minute_idx < 2:
        return False
    cum_dir = "green" if cum_return > 0 else "red"
    for i in range(minute_idx - 2, minute_idx + 1):
        c = window[i]
        o = float(c.open)
        cl = float(c.close)
        if cum_dir == "green" and cl <= o:
            return False
        if cum_dir == "red" and cl >= o:
            return False
    return True


def _no_reversal(window: list[Any], minute_idx: int) -> bool:
    """Check that there was no significant reversal (>0.1% move then 30%+ retrace)."""
    window_open = float(window[0].open)
    if window_open == 0:
        return True

    cum_return = _compute_cum_return(window, minute_idx)

    max_cum_ret = 0.0
    min_cum_ret = 0.0
    max_idx = 0
    min_idx = 0
    for i in range(1, minute_idx + 1):
        cr = (float(window[i].close) - window_open) / window_open
        if cr > max_cum_ret:
            max_cum_ret = cr
            max_idx = i
        if cr < min_cum_ret:
            min_cum_ret = cr
            min_idx = i

    # Check upward reversal
    if max_cum_ret > 0.001:
        retrace = max_cum_ret - cum_return
        if retrace > max_cum_ret * 0.3 and max_idx < minute_idx:
            return False

    # Check downward reversal
    if abs(min_cum_ret) > 0.001:
        retrace_down = cum_return - min_cum_ret
        if retrace_down > abs(min_cum_ret) * 0.3 and min_idx < minute_idx:
            return False

    return True


# ---------------------------------------------------------------------------
# Trade dataclass
# ---------------------------------------------------------------------------
@dataclass
class Trade:
    window_start: str
    entry_minute: int
    exit_minute: int
    direction: str      # "YES" or "NO"
    entry_price: float
    exit_price: float
    pnl_pct: float
    pnl_dollar: float
    exit_reason: str    # "profit_target", "stop_loss", "time_exit"
    cum_return_at_entry: float


# ---------------------------------------------------------------------------
# Strategy runner
# ---------------------------------------------------------------------------
def run_strategy(
    windows: list[list[Any]],
    *,
    threshold: float,
    entry_minutes: list[int],
    require_last3: bool = False,
    require_no_reversal: bool = False,
    tp_pct: float = 0.25,
    sl_pct: float = -0.15,
    exit_minute: int = 13,
) -> list[Trade]:
    """Run momentum strategy on all windows and return trade list."""
    trades: list[Trade] = []

    for window in windows:
        trade = _evaluate_window(
            window,
            threshold=threshold,
            entry_minutes=entry_minutes,
            require_last3=require_last3,
            require_no_reversal=require_no_reversal,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            exit_minute=exit_minute,
        )
        if trade is not None:
            trades.append(trade)

    return trades


def _evaluate_window(
    window: list[Any],
    *,
    threshold: float,
    entry_minutes: list[int],
    require_last3: bool,
    require_no_reversal: bool,
    tp_pct: float,
    sl_pct: float,
    exit_minute: int,
) -> Trade | None:
    """Evaluate a single window for a momentum entry."""
    window_open = float(window[0].open)
    if window_open == 0:
        return None

    for entry_min in entry_minutes:
        if entry_min >= MINUTES_PER_WINDOW:
            continue

        cum_ret = _compute_cum_return(window, entry_min)

        # Check threshold
        if abs(cum_ret) < threshold:
            continue

        # Check optional filters
        if require_last3 and not _last_3_agree(window, entry_min, cum_ret):
            continue
        if require_no_reversal and not _no_reversal(window, entry_min):
            continue

        # Determine direction: follow cumulative return
        if cum_ret > 0:
            direction = "YES"
        else:
            direction = "NO"

        # Compute YES prices at each minute using sigmoid
        yes_prices: list[float] = []
        for i in range(MINUTES_PER_WINDOW):
            cr = _compute_cum_return(window, i)
            yes_prices.append(_sigmoid(cr))

        # Entry price
        if direction == "YES":
            entry_price = yes_prices[entry_min]
        else:
            entry_price = 1.0 - yes_prices[entry_min]

        if entry_price <= 0.0:
            continue

        # Simulate hold until exit
        final_exit_min = entry_min
        final_exit_price = entry_price
        final_exit_reason = "time_exit"

        for hold_min in range(entry_min + 1, min(exit_minute + 1, MINUTES_PER_WINDOW)):
            if direction == "YES":
                current_price = yes_prices[hold_min]
            else:
                current_price = 1.0 - yes_prices[hold_min]

            pnl = (current_price - entry_price) / entry_price

            if pnl >= tp_pct:
                final_exit_min = hold_min
                final_exit_price = current_price
                final_exit_reason = "profit_target"
                break
            if pnl <= sl_pct:
                final_exit_min = hold_min
                final_exit_price = current_price
                final_exit_reason = "stop_loss"
                break

            final_exit_min = hold_min
            final_exit_price = current_price

        final_pnl_pct = (
            (final_exit_price - entry_price) / entry_price
            if entry_price != 0.0
            else 0.0
        )

        return Trade(
            window_start=str(window[0].timestamp),
            entry_minute=entry_min,
            exit_minute=final_exit_min,
            direction=direction,
            entry_price=round(entry_price, 6),
            exit_price=round(final_exit_price, 6),
            pnl_pct=round(final_pnl_pct, 6),
            pnl_dollar=round(final_pnl_pct * POSITION_SIZE, 4),
            exit_reason=final_exit_reason,
            cum_return_at_entry=round(cum_ret, 6),
        )

    return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(trades: list[Trade]) -> dict[str, Any]:
    """Compute full metrics for a list of trades."""
    if not trades:
        return {
            "trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
            "avg_win_pct": 0.0, "avg_loss_pct": 0.0,
            "total_return_pct": 0.0, "sharpe": 0.0,
            "max_drawdown_pct": 0.0, "profit_factor": 0.0,
            "avg_hold_min": 0.0, "best_pct": 0.0, "worst_pct": 0.0,
        }

    pnls = [t.pnl_pct for t in trades]
    dollar_pnls = [t.pnl_dollar for t in trades]
    hold_times = [t.exit_minute - t.entry_minute for t in trades]

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    n = len(trades)
    n_wins = len(wins)
    n_losses = len(losses)
    win_rate = n_wins / n if n > 0 else 0.0
    avg_win = sum(wins) / n_wins if n_wins > 0 else 0.0
    avg_loss = sum(losses) / n_losses if n_losses > 0 else 0.0

    # Equity curve
    equity = [POSITION_SIZE]
    for dp in dollar_pnls:
        equity.append(equity[-1] + dp)
    total_return = (equity[-1] - equity[0]) / equity[0] if equity[0] != 0 else 0.0

    # Sharpe (annualized, ~35040 trades/year at 4/hr)
    if n >= 2:
        mean_ret = sum(pnls) / n
        variance = sum((r - mean_ret) ** 2 for r in pnls) / n
        std_ret = math.sqrt(variance)
        if std_ret > 0:
            sharpe = (mean_ret / std_ret) * math.sqrt(35040)
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    # Max drawdown
    peak = equity[0]
    max_dd = 0.0
    for val in equity:
        if val > peak:
            peak = val
        dd = (peak - val) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    # Profit factor
    gross_profit = sum(d for d in dollar_pnls if d > 0)
    gross_loss = abs(sum(d for d in dollar_pnls if d < 0))
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    avg_hold = sum(hold_times) / n if n > 0 else 0.0

    return {
        "trades": n,
        "wins": n_wins,
        "losses": n_losses,
        "win_rate": round(win_rate, 4),
        "avg_win_pct": round(avg_win * 100, 4),
        "avg_loss_pct": round(avg_loss * 100, 4),
        "total_return_pct": round(total_return * 100, 4),
        "sharpe": round(sharpe, 4),
        "max_drawdown_pct": round(max_dd * 100, 4),
        "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else "inf",
        "avg_hold_min": round(avg_hold, 2),
        "best_pct": round(max(pnls) * 100, 4),
        "worst_pct": round(min(pnls) * 100, 4),
    }


# ---------------------------------------------------------------------------
# Breakdown printers
# ---------------------------------------------------------------------------
def print_exit_breakdown(trades: list[Trade], name: str) -> None:
    """Print breakdown by exit reason."""
    reasons: dict[str, list[Trade]] = {}
    for t in trades:
        reasons.setdefault(t.exit_reason, []).append(t)

    print(f"\n  Exit Breakdown — {name}")
    print(f"  {'Reason':<18} {'Count':>7} {'Pct':>7} {'Avg PnL%':>10}")
    print(f"  {'-'*44}")
    for reason in ["profit_target", "stop_loss", "time_exit"]:
        group = reasons.get(reason, [])
        if group:
            avg_pnl = sum(t.pnl_pct for t in group) / len(group) * 100
            pct = len(group) / len(trades) * 100
            print(f"  {reason:<18} {len(group):>7} {pct:>6.1f}% {avg_pnl:>+9.2f}%")


def print_entry_minute_breakdown(trades: list[Trade], name: str) -> None:
    """Print breakdown by entry minute."""
    by_minute: dict[int, list[Trade]] = {}
    for t in trades:
        by_minute.setdefault(t.entry_minute, []).append(t)

    print(f"\n  Entry Minute Breakdown — {name}")
    print(f"  {'Min':>5} {'Trades':>8} {'WinRate':>9} {'AvgPnL%':>10}")
    print(f"  {'-'*34}")
    for minute in sorted(by_minute.keys()):
        group = by_minute[minute]
        w = sum(1 for t in group if t.pnl_pct > 0)
        wr = w / len(group) * 100
        avg_pnl = sum(t.pnl_pct for t in group) / len(group) * 100
        print(f"  {minute:>5} {len(group):>8} {wr:>8.1f}% {avg_pnl:>+9.2f}%")


def print_direction_breakdown(trades: list[Trade], name: str) -> None:
    """Print breakdown by direction (YES/NO)."""
    by_dir: dict[str, list[Trade]] = {}
    for t in trades:
        by_dir.setdefault(t.direction, []).append(t)

    print(f"\n  Direction Breakdown — {name}")
    print(f"  {'Dir':>5} {'Trades':>8} {'WinRate':>9} {'AvgPnL%':>10}")
    print(f"  {'-'*34}")
    for d in ["YES", "NO"]:
        group = by_dir.get(d, [])
        if group:
            w = sum(1 for t in group if t.pnl_pct > 0)
            wr = w / len(group) * 100
            avg_pnl = sum(t.pnl_pct for t in group) / len(group) * 100
            print(f"  {d:>5} {len(group):>8} {wr:>8.1f}% {avg_pnl:>+9.2f}%")


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
def print_comparison(all_results: dict[str, dict[str, Any]]) -> None:
    """Print a side-by-side comparison of all strategy variants."""
    names = list(all_results.keys())
    col_width = 22

    print("\n" + "=" * 90)
    print("  MOMENTUM STRATEGY COMPARISON")
    print("=" * 90)

    header = f"  {'Metric':<22}"
    for name in names:
        header += f" {name:>{col_width}}"
    print(header)
    print("  " + "-" * (22 + (col_width + 1) * len(names)))

    rows = [
        ("Trades", "trades", False),
        ("Wins", "wins", False),
        ("Losses", "losses", False),
        ("Win Rate", "win_rate", True),
        ("Avg Win %", "avg_win_pct", False),
        ("Avg Loss %", "avg_loss_pct", False),
        ("Total Return %", "total_return_pct", False),
        ("Sharpe Ratio", "sharpe", False),
        ("Max Drawdown %", "max_drawdown_pct", False),
        ("Profit Factor", "profit_factor", False),
        ("Avg Hold (min)", "avg_hold_min", False),
        ("Best Trade %", "best_pct", False),
        ("Worst Trade %", "worst_pct", False),
    ]

    for label, key, as_pct in rows:
        line = f"  {label:<22}"
        for name in names:
            val = all_results[name].get(key, "N/A")
            if as_pct and isinstance(val, (int, float)):
                line += f" {val * 100:>{col_width}.2f}%"
            elif isinstance(val, float):
                line += f" {val:>{col_width}.4f}"
            else:
                line += f" {str(val):>{col_width}}"
        print(line)

    print("=" * 90)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 90)
    print("  MVHE Momentum Confirmation Backtest")
    print("  Validates momentum approach on 132K 1m BTC candles")
    print("=" * 90)

    # 1. Load all CSV chunks
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
    windows = _group_into_15m_windows(all_candles)
    print(f"  15m windows: {len(windows)}")

    # 3. Define strategy variants
    #
    # KEY INSIGHT: The original 3 variants had stop_loss=-0.15 which kills
    # winning trades. With 85%+ directional accuracy, intra-candle BTC noise
    # causes sigmoid token price to swing enough to trigger tight stops on
    # trades that ultimately prove correct. Solution: remove stop loss and
    # hold to near-settlement (minute 13-14).
    variants = {
        "Balanced": {
            "threshold": 0.0010,
            "entry_minutes": [5, 6, 7, 8],
            "require_last3": False,
            "require_no_reversal": False,
        },
        "Conservative": {
            "threshold": 0.0015,
            "entry_minutes": [6, 7, 8],
            "require_last3": False,
            "require_no_reversal": False,
        },
        "Filtered": {
            "threshold": 0.0010,
            "entry_minutes": [5, 6, 7, 8],
            "require_last3": True,
            "require_no_reversal": True,
        },
        # --- NEW: No stop loss variants that let directional accuracy work ---
        "Hold_Balanced": {
            "threshold": 0.0010,
            "entry_minutes": [5, 6, 7, 8],
            "require_last3": False,
            "require_no_reversal": False,
            "tp_pct": 0.40,          # generous TP
            "sl_pct": -999.0,        # effectively no stop loss
            "exit_minute": 14,       # hold to near-settlement
        },
        "Hold_Conservative": {
            "threshold": 0.0015,
            "entry_minutes": [6, 7, 8],
            "require_last3": False,
            "require_no_reversal": False,
            "tp_pct": 0.40,
            "sl_pct": -999.0,
            "exit_minute": 14,
        },
        "Hold_Filtered": {
            "threshold": 0.0010,
            "entry_minutes": [5, 6, 7, 8],
            "require_last3": True,
            "require_no_reversal": True,
            "tp_pct": 0.40,
            "sl_pct": -999.0,
            "exit_minute": 14,
        },
    }

    # 4. Run each variant
    all_results: dict[str, dict[str, Any]] = {}
    all_trades: dict[str, list[Trade]] = {}

    for name, params in variants.items():
        tp = params.get("tp_pct", 0.25)
        sl = params.get("sl_pct", -0.15)
        em = params.get("exit_minute", 13)
        print(f"\n  Running {name}...")
        print(f"    threshold={params['threshold']*100:.2f}%, "
              f"entry_min={params['entry_minutes']}, "
              f"last3={params['require_last3']}, "
              f"no_rev={params['require_no_reversal']}, "
              f"tp={tp*100:.0f}%, sl={sl*100:.0f}%, exit_min={em}")

        trades = run_strategy(
            windows,
            threshold=params["threshold"],
            entry_minutes=params["entry_minutes"],
            require_last3=params["require_last3"],
            require_no_reversal=params["require_no_reversal"],
            tp_pct=tp,
            sl_pct=sl,
            exit_minute=em,
        )

        metrics = compute_metrics(trades)
        all_results[name] = metrics
        all_trades[name] = trades

        coverage = len(trades) / len(windows) * 100 if windows else 0
        print(f"    Trades: {len(trades)} ({coverage:.1f}% of windows)")
        print(f"    Win rate: {metrics['win_rate']*100:.1f}%")

    # 5. Print comparison table
    print_comparison(all_results)

    # 6. Print detailed breakdowns for each variant
    for name, trades in all_trades.items():
        print_exit_breakdown(trades, name)
        print_entry_minute_breakdown(trades, name)
        print_direction_breakdown(trades, name)

    # 7. Find best variant by win rate and show sample trades
    best_name = max(all_results, key=lambda n: all_results[n].get("win_rate", 0))
    print("\n" + "=" * 90)
    print(f"  SAMPLE TRADES — {best_name} (first 15)")
    print("=" * 90)
    print(f"  {'Window Start':<28} {'Min':>4} {'Dir':>4} {'Entry':>8} {'Exit':>8} "
          f"{'PnL%':>8} {'$PnL':>8} {'Reason':<15}")
    print(f"  {'-'*90}")

    for t in all_trades[best_name][:15]:
        print(f"  {t.window_start:<28} {t.entry_minute:>4} {t.direction:>4} "
              f"{t.entry_price:>8.4f} {t.exit_price:>8.4f} "
              f"{t.pnl_pct*100:>+7.2f}% {t.pnl_dollar:>+7.2f} {t.exit_reason:<15}")

    print("\n" + "=" * 90)
    print("  Backtest complete.")
    print("=" * 90)


if __name__ == "__main__":
    main()
