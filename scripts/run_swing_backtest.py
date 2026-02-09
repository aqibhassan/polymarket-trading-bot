"""Intra-15-minute swing trading backtest on 1m BTC candle data.

Loads 1m BTC CSV chunks, groups into 15m windows, and simulates
Polymarket YES/NO token pricing using a sigmoid model.  Three
strategies are evaluated:

  1. EarlyMomentum   — enter after minute 3 on directional move
  2. ReversalCatcher — contrarian entry after sharp initial move reverses
  3. ConfirmationTrader — enter after 4+ consecutive same-direction 1m candles

Each strategy gets full statistical validation (bootstrap + permutation)
and results are saved to data/swing_backtest_report.json.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path so ``src`` package resolves.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtesting.data_loader import DataLoader
from src.backtesting.reporter import BacktestReporter
from src.backtesting.validator import StatisticalValidator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CSV_PATHS = [
    PROJECT_ROOT / "data" / f"btc_1m_chunk{i}.csv" for i in range(1, 7)
]
REPORT_PATH = PROJECT_ROOT / "data" / "swing_backtest_report.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
POSITION_SIZE = 100.0          # $100 per trade
# Sigmoid sensitivity calibrated so that:
#   0.1% BTC move -> YES ~0.67 (significant shift from 0.50)
#   0.2% BTC move -> YES ~0.80
#   0.3% BTC move -> YES ~0.89 (near certainty)
SENSITIVITY = 0.07
MINUTES_PER_WINDOW = 15


# ---------------------------------------------------------------------------
# Sigmoid helper
# ---------------------------------------------------------------------------
def _sigmoid(cum_ret: float, sensitivity: float = SENSITIVITY) -> float:
    """Map cumulative BTC return to YES token price via sigmoid."""
    x = sensitivity * cum_ret * 10000
    # Clamp to avoid overflow
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------------
# Trade dataclass
# ---------------------------------------------------------------------------
@dataclass
class SwingTrade:
    strategy: str
    window_start: str
    entry_minute: int
    exit_minute: int
    direction: str          # "YES" or "NO"
    entry_price: float
    exit_price: float
    pnl_pct: float
    pnl_dollar: float
    exit_reason: str


# ---------------------------------------------------------------------------
# Strategy base
# ---------------------------------------------------------------------------
@dataclass
class WindowState:
    """Pre-computed state for a single 15m window."""
    window_open: float                # BTC open price at minute 0
    cum_rets: list[float]             # cumulative returns per minute (len 15)
    yes_prices: list[float]           # YES token prices per minute
    no_prices: list[float]            # NO token prices per minute
    candle_directions: list[str]      # "green", "red", "neutral" per 1m candle
    candle_returns: list[float]       # per-minute BTC return from previous close
    window_start_ts: str


def _compute_window_state(candles_1m: list[Any]) -> WindowState:
    """Build WindowState from exactly 15 one-minute candles."""
    window_open = float(candles_1m[0].open)
    cum_rets: list[float] = []
    yes_prices: list[float] = []
    no_prices: list[float] = []
    directions: list[str] = []
    candle_returns: list[float] = []

    for i, c in enumerate(candles_1m):
        close_f = float(c.close)
        cum_ret = (close_f - window_open) / window_open if window_open != 0.0 else 0.0
        cum_rets.append(cum_ret)
        yp = _sigmoid(cum_ret)
        yes_prices.append(yp)
        no_prices.append(1.0 - yp)

        if c.close > c.open:
            directions.append("green")
        elif c.close < c.open:
            directions.append("red")
        else:
            directions.append("neutral")

        if i == 0:
            candle_returns.append(0.0)
        else:
            prev_close = float(candles_1m[i - 1].close)
            candle_returns.append(
                (close_f - prev_close) / prev_close if prev_close != 0.0 else 0.0,
            )

    return WindowState(
        window_open=window_open,
        cum_rets=cum_rets,
        yes_prices=yes_prices,
        no_prices=no_prices,
        candle_directions=directions,
        candle_returns=candle_returns,
        window_start_ts=str(candles_1m[0].timestamp),
    )


# ---------------------------------------------------------------------------
# Strategy 1: EarlyMomentum
# ---------------------------------------------------------------------------
def _run_early_momentum(ws: WindowState) -> SwingTrade | None:
    """After minute 3, if cumulative BTC return > +0.05% buy YES,
    if < -0.05% buy NO.  Exit at minute 12 or profit >= 30%.
    Stop loss at -15%."""
    for entry_min in range(3, 13):  # minutes 3..12
        cum_ret = ws.cum_rets[entry_min]
        if abs(cum_ret) < 0.0005:  # 0.05%
            continue

        if cum_ret > 0.0005:
            direction = "YES"
            entry_price = ws.yes_prices[entry_min]
        else:
            direction = "NO"
            entry_price = ws.no_prices[entry_min]

        if entry_price <= 0.0:
            continue

        # Simulate hold
        exit_min = entry_min
        exit_price = entry_price
        exit_reason = "time_exit"

        for hold_min in range(entry_min + 1, min(13, MINUTES_PER_WINDOW)):
            if direction == "YES":
                current_price = ws.yes_prices[hold_min]
            else:
                current_price = ws.no_prices[hold_min]

            pnl_pct = (current_price - entry_price) / entry_price

            if pnl_pct >= 0.30:
                exit_min = hold_min
                exit_price = current_price
                exit_reason = "profit_target"
                break
            if pnl_pct <= -0.15:
                exit_min = hold_min
                exit_price = current_price
                exit_reason = "stop_loss"
                break

            exit_min = hold_min
            exit_price = current_price

        final_pnl_pct = (exit_price - entry_price) / entry_price if entry_price != 0.0 else 0.0
        return SwingTrade(
            strategy="EarlyMomentum",
            window_start=ws.window_start_ts,
            entry_minute=entry_min,
            exit_minute=exit_min,
            direction=direction,
            entry_price=round(entry_price, 6),
            exit_price=round(exit_price, 6),
            pnl_pct=round(final_pnl_pct, 6),
            pnl_dollar=round(final_pnl_pct * POSITION_SIZE, 4),
            exit_reason=exit_reason,
        )

    return None


# ---------------------------------------------------------------------------
# Strategy 2: ReversalCatcher
# ---------------------------------------------------------------------------
def _run_reversal_catcher(ws: WindowState) -> SwingTrade | None:
    """Monitor first 5 minutes. If BTC moves >0.15% in one direction then
    starts reversing (1m candle goes opposite), enter contrarian at minute 6-7.
    Exit when profit >= 25% or at minute 13. Stop loss at -20%."""
    # Check if there was a strong move in first 5 minutes
    cum_ret_5 = ws.cum_rets[4] if len(ws.cum_rets) > 4 else 0.0
    if abs(cum_ret_5) < 0.0015:  # 0.15%
        return None

    initial_direction = "up" if cum_ret_5 > 0 else "down"

    # Look for reversal signal in minutes 5-7
    entry_min = None
    for check_min in range(5, min(8, MINUTES_PER_WINDOW)):
        candle_dir = ws.candle_directions[check_min]
        if initial_direction == "up" and candle_dir == "red":
            entry_min = check_min
            break
        if initial_direction == "down" and candle_dir == "green":
            entry_min = check_min
            break

    if entry_min is None:
        return None

    # Contrarian entry: if initial move was up, bet NO (expect reversal down)
    if initial_direction == "up":
        direction = "NO"
        entry_price = ws.no_prices[entry_min]
    else:
        direction = "YES"
        entry_price = ws.yes_prices[entry_min]

    if entry_price <= 0.0:
        return None

    exit_min = entry_min
    exit_price = entry_price
    exit_reason = "time_exit"

    for hold_min in range(entry_min + 1, min(14, MINUTES_PER_WINDOW)):
        if direction == "YES":
            current_price = ws.yes_prices[hold_min]
        else:
            current_price = ws.no_prices[hold_min]

        pnl_pct = (current_price - entry_price) / entry_price

        if pnl_pct >= 0.25:
            exit_min = hold_min
            exit_price = current_price
            exit_reason = "profit_target"
            break
        if pnl_pct <= -0.20:
            exit_min = hold_min
            exit_price = current_price
            exit_reason = "stop_loss"
            break

        exit_min = hold_min
        exit_price = current_price

    final_pnl_pct = (exit_price - entry_price) / entry_price if entry_price != 0.0 else 0.0
    return SwingTrade(
        strategy="ReversalCatcher",
        window_start=ws.window_start_ts,
        entry_minute=entry_min,
        exit_minute=exit_min,
        direction=direction,
        entry_price=round(entry_price, 6),
        exit_price=round(exit_price, 6),
        pnl_pct=round(final_pnl_pct, 6),
        pnl_dollar=round(final_pnl_pct * POSITION_SIZE, 4),
        exit_reason=exit_reason,
    )


# ---------------------------------------------------------------------------
# Strategy 3: ConfirmationTrader
# ---------------------------------------------------------------------------
def _run_confirmation_trader(ws: WindowState) -> SwingTrade | None:
    """Count consecutive 1m candles in same direction.
    When 4+ consecutive (e.g., 4 green), enter in that direction.
    Exit at minute 11 or profit >= 35%. Stop loss at -15%."""
    entry_min = None
    direction = None
    consecutive = 0
    prev_dir = None

    for i in range(MINUTES_PER_WINDOW):
        d = ws.candle_directions[i]
        if d == "neutral":
            consecutive = 0
            prev_dir = None
            continue
        if d == prev_dir:
            consecutive += 1
        else:
            consecutive = 1
            prev_dir = d

        if consecutive >= 4 and entry_min is None:
            entry_min = i
            if d == "green":
                direction = "YES"
            else:
                direction = "NO"
            break

    if entry_min is None or direction is None:
        return None

    if direction == "YES":
        entry_price = ws.yes_prices[entry_min]
    else:
        entry_price = ws.no_prices[entry_min]

    if entry_price <= 0.0:
        return None

    exit_min = entry_min
    exit_price = entry_price
    exit_reason = "time_exit"

    for hold_min in range(entry_min + 1, min(12, MINUTES_PER_WINDOW)):
        if direction == "YES":
            current_price = ws.yes_prices[hold_min]
        else:
            current_price = ws.no_prices[hold_min]

        pnl_pct = (current_price - entry_price) / entry_price

        if pnl_pct >= 0.35:
            exit_min = hold_min
            exit_price = current_price
            exit_reason = "profit_target"
            break
        if pnl_pct <= -0.15:
            exit_min = hold_min
            exit_price = current_price
            exit_reason = "stop_loss"
            break

        exit_min = hold_min
        exit_price = current_price

    final_pnl_pct = (exit_price - entry_price) / entry_price if entry_price != 0.0 else 0.0
    return SwingTrade(
        strategy="ConfirmationTrader",
        window_start=ws.window_start_ts,
        entry_minute=entry_min,
        exit_minute=exit_min,
        direction=direction,
        entry_price=round(entry_price, 6),
        exit_price=round(exit_price, 6),
        pnl_pct=round(final_pnl_pct, 6),
        pnl_dollar=round(final_pnl_pct * POSITION_SIZE, 4),
        exit_reason=exit_reason,
    )


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------
def _compute_strategy_metrics(trades: list[SwingTrade]) -> dict[str, Any]:
    """Compute full metrics for a list of trades."""
    if not trades:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
            "total_return_pct": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "profit_factor": 0.0,
            "avg_hold_time_min": 0.0,
            "best_trade_pct": 0.0,
            "worst_trade_pct": 0.0,
        }

    pnls = [t.pnl_pct for t in trades]
    dollar_pnls = [t.pnl_dollar for t in trades]
    hold_times = [t.exit_minute - t.entry_minute for t in trades]

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_trades = len(trades)
    n_wins = len(wins)
    n_losses = len(losses)
    win_rate = n_wins / total_trades if total_trades > 0 else 0.0
    avg_win = sum(wins) / n_wins if n_wins > 0 else 0.0
    avg_loss = sum(losses) / n_losses if n_losses > 0 else 0.0

    # Total return on equity curve
    equity = [POSITION_SIZE]  # start with initial capital per-trade basis
    for dp in dollar_pnls:
        equity.append(equity[-1] + dp)

    total_return = (equity[-1] - equity[0]) / equity[0] if equity[0] != 0.0 else 0.0

    # Sharpe ratio (annualized) — each trade is one observation
    # Annualize assuming ~4 trades/hour * 24 * 365 = ~35040 trades/year
    if len(pnls) >= 2:
        mean_ret = sum(pnls) / len(pnls)
        variance = sum((r - mean_ret) ** 2 for r in pnls) / len(pnls)
        std_ret = math.sqrt(variance)
        if std_ret > 0:
            # Approximate: trades happen every ~15 min = 35040/year
            ann_factor = math.sqrt(35040)
            sharpe = (mean_ret / std_ret) * ann_factor
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    # Max drawdown on equity curve
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
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (
        float("inf") if gross_profit > 0 else 0.0
    )

    avg_hold = sum(hold_times) / len(hold_times) if hold_times else 0.0

    return {
        "total_trades": total_trades,
        "wins": n_wins,
        "losses": n_losses,
        "win_rate": round(win_rate, 4),
        "avg_win_pct": round(avg_win * 100, 4),
        "avg_loss_pct": round(avg_loss * 100, 4),
        "total_return_pct": round(total_return * 100, 4),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown_pct": round(max_dd * 100, 4),
        "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else "inf",
        "avg_hold_time_min": round(avg_hold, 2),
        "best_trade_pct": round(max(pnls) * 100, 4),
        "worst_trade_pct": round(min(pnls) * 100, 4),
    }


# ---------------------------------------------------------------------------
# Validation adapter — convert SwingTrades to the format StatisticalValidator
# expects (list of dicts with 'pnl' key and Decimal equity curve)
# ---------------------------------------------------------------------------
def _build_validation_inputs(
    trades: list[SwingTrade],
) -> tuple[list[dict[str, Any]], list[Decimal]]:
    """Convert SwingTrades to validator-compatible format."""
    trade_dicts = [{"pnl": t.pnl_dollar} for t in trades]
    equity = [Decimal(str(POSITION_SIZE))]
    for t in trades:
        equity.append(equity[-1] + Decimal(str(t.pnl_dollar)))
    return trade_dicts, equity


# ---------------------------------------------------------------------------
# Group 1m candles into 15m windows
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
        # Determine the 15m boundary: 0, 15, 30, 45
        window_minute = (minute // 15) * 15

        if current_window_start is None:
            # First candle — start a new window only if it aligns
            if minute == window_minute:
                current_window_start = window_minute
                current_window = [c]
            # else skip candles until we find a window boundary
        else:
            # Check if this candle belongs to the current window
            expected_minute = current_window_start
            offset = minute - expected_minute
            if offset < 0:
                offset += 60
            # Same hour and within the 15-minute range, and same window
            same_window = (
                len(current_window) < MINUTES_PER_WINDOW
                and 0 <= offset < MINUTES_PER_WINDOW
                and ts.hour == current_window[0].timestamp.hour
                and ts.date() == current_window[0].timestamp.date()
            )

            if same_window:
                current_window.append(c)
            else:
                # Save completed window if it has exactly 15 candles
                if len(current_window) == MINUTES_PER_WINDOW:
                    windows.append(current_window)
                # Start new window
                if minute == window_minute:
                    current_window = [c]
                    current_window_start = window_minute
                else:
                    current_window = []
                    current_window_start = None

    # Don't forget the last window
    if len(current_window) == MINUTES_PER_WINDOW:
        windows.append(current_window)

    return windows


# ---------------------------------------------------------------------------
# Comparison table printer
# ---------------------------------------------------------------------------
def _print_comparison(
    results: dict[str, dict[str, Any]],
) -> None:
    """Print a side-by-side comparison of all strategies."""
    names = list(results.keys())
    col_width = 20

    print("\n" + "=" * 80)
    print("  SWING STRATEGY COMPARISON")
    print("=" * 80)

    header = f"  {'Metric':<25}"
    for name in names:
        header += f" {name:>{col_width}}"
    print(header)
    print("  " + "-" * (25 + (col_width + 1) * len(names)))

    rows = [
        ("Total Trades", "total_trades", False),
        ("Wins", "wins", False),
        ("Losses", "losses", False),
        ("Win Rate", "win_rate", True),
        ("Avg Win %", "avg_win_pct", False),
        ("Avg Loss %", "avg_loss_pct", False),
        ("Total Return %", "total_return_pct", False),
        ("Sharpe Ratio", "sharpe_ratio", False),
        ("Max Drawdown %", "max_drawdown_pct", False),
        ("Profit Factor", "profit_factor", False),
        ("Avg Hold (min)", "avg_hold_time_min", False),
        ("Best Trade %", "best_trade_pct", False),
        ("Worst Trade %", "worst_trade_pct", False),
    ]

    for label, key, as_pct in rows:
        line = f"  {label:<25}"
        for name in names:
            val = results[name].get("metrics", {}).get(key, "N/A")
            if as_pct and isinstance(val, (int, float)):
                line += f" {val * 100:>{col_width}.2f}%"
            elif isinstance(val, float):
                line += f" {val:>{col_width}.4f}"
            else:
                line += f" {str(val):>{col_width}}"
        print(line)

    # Validation section
    print("\n  --- Statistical Validation ---")
    for name in names:
        val = results[name].get("validation", {})
        ci = val.get("sharpe_ci_95", (0, 0))
        p_val = val.get("p_value", 1.0)
        sig = val.get("is_significant", False)
        print(
            f"  {name:<20} Sharpe CI=[{ci[0]:.4f}, {ci[1]:.4f}]  "
            f"p={p_val:.4f}  significant={'Yes' if sig else 'No'}"
        )

    print("=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 80)
    print("  MVHE Intra-15m Swing Trading Backtest")
    print("  1-Minute BTC Data -> 15m Windows -> Polymarket Token Simulation")
    print("=" * 80)

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
        print("ERROR: No data loaded. Ensure at least one btc_1m_chunk*.csv exists.")
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
    print("\nGrouping into 15-minute windows...")
    windows = _group_into_15m_windows(all_candles)
    print(f"  15m windows: {len(windows)}")

    if not windows:
        print("ERROR: No complete 15m windows could be formed.")
        sys.exit(1)

    # 3. Run all three strategies on every window
    print("\nRunning strategies on all windows...")

    strategy_trades: dict[str, list[SwingTrade]] = {
        "EarlyMomentum": [],
        "ReversalCatcher": [],
        "ConfirmationTrader": [],
    }

    strategy_fns = {
        "EarlyMomentum": _run_early_momentum,
        "ReversalCatcher": _run_reversal_catcher,
        "ConfirmationTrader": _run_confirmation_trader,
    }

    for window in windows:
        ws = _compute_window_state(window)
        for name, fn in strategy_fns.items():
            trade = fn(ws)
            if trade is not None:
                strategy_trades[name].append(trade)

    for name, trades in strategy_trades.items():
        print(f"  {name}: {len(trades)} trades")

    # 4. Compute metrics for each strategy
    print("\nComputing metrics...")
    all_results: dict[str, dict[str, Any]] = {}

    for name, trades in strategy_trades.items():
        metrics = _compute_strategy_metrics(trades)
        print(f"\n  --- {name} ---")
        for k, v in metrics.items():
            print(f"    {k}: {v}")

        all_results[name] = {"metrics": metrics, "trades_count": len(trades)}

    # 5. Statistical validation (500 bootstrap, 500 permutations)
    print("\nRunning statistical validation (500 bootstrap, 500 permutation)...")
    validator = StatisticalValidator(
        n_bootstrap=500,
        n_permutations=500,
        rng_seed=42,
    )

    for name, trades in strategy_trades.items():
        trade_dicts, equity_curve = _build_validation_inputs(trades)
        val_result = validator.validate(trade_dicts, equity_curve)
        all_results[name]["validation"] = {
            "sharpe_ci_95": list(val_result.sharpe_ci_95),
            "p_value": val_result.p_value,
            "is_significant": val_result.is_significant,
            "overfitting_warning": val_result.overfitting_warning,
        }
        print(
            f"  {name}: CI=[{val_result.sharpe_ci_95[0]:.4f}, "
            f"{val_result.sharpe_ci_95[1]:.4f}]  p={val_result.p_value:.4f}  "
            f"sig={'Yes' if val_result.is_significant else 'No'}"
        )

    # 6. Print comparison table
    _print_comparison(all_results)

    # 7. Build and save JSON report
    report: dict[str, Any] = {
        "backtest_type": "intra_15m_swing",
        "data_source": "btc_1m_candles",
        "total_1m_candles": len(all_candles),
        "total_15m_windows": len(windows),
        "date_range": {
            "start": str(all_candles[0].timestamp),
            "end": str(all_candles[-1].timestamp),
        },
        "position_size_usd": POSITION_SIZE,
        "sigmoid_sensitivity": SENSITIVITY,
        "strategies": {},
    }

    for name in strategy_trades:
        trades = strategy_trades[name]
        report["strategies"][name] = {
            "metrics": all_results[name]["metrics"],
            "validation": all_results[name].get("validation", {}),
            "sample_trades": [
                {
                    "window_start": t.window_start,
                    "entry_minute": t.entry_minute,
                    "exit_minute": t.exit_minute,
                    "direction": t.direction,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "pnl_pct": t.pnl_pct,
                    "pnl_dollar": t.pnl_dollar,
                    "exit_reason": t.exit_reason,
                }
                for t in trades[:20]  # first 20 as samples
            ],
            "total_trades": len(trades),
        }

    # Save report
    reporter = BacktestReporter()
    reporter.save_report(report, REPORT_PATH)
    print(f"\n  Report saved to {REPORT_PATH}")

    print("\nDone.")


if __name__ == "__main__":
    main()
