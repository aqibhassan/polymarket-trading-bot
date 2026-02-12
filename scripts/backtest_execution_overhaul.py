"""Execution Overhaul Backtest — simulates the full GTC-first + FAK pipeline.

Tests the new execution strategy on 132K 1m BTC candles:
  1. GTC-first at minutes 2-4 (passive, no price offset)
  2. Price ladder escalation at minutes 5-10 (+3c/+7c/+10c)
  3. FAK taker sweep at minutes 11+ (crosses spread at best_ask)
  4. Max price caps: GTC max 0.65, FAK max 0.85
  5. Binary settlement: correct → $1, wrong → $0

Compares OLD execution (single GTC at minute 8-10) vs NEW execution
(GTC-first at minute 2 + escalating ladder + late FAK sweep).
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

CSV_PATHS = [PROJECT_ROOT / "data" / f"btc_1m_chunk{i}.csv" for i in range(1, 7)]
MINUTES_PER_WINDOW = 15
POSITION_SIZE = 100.0
SENSITIVITY = 0.07

# --- New execution config (from config/default.toml) ---
MAX_CLOB_ENTRY_PRICE = 0.65  # GTC/GTD max
FAK_MAX_ENTRY_PRICE = 0.85   # FAK taker max
PRICE_LADDER = {2: 0.00, 5: 0.03, 8: 0.07, 10: 0.10}  # minute → offset
MIN_ENTRY_MINUTE = 2
FAK_LATE_MINUTE = 11
MIN_THRESHOLD_PCT = 0.05  # Minimum momentum threshold for entry

# Polymarket fee model
def _polymarket_fee(price: float) -> float:
    """Dynamic taker fee: 0.25 * p^2 * (1-p)^2. Maker = 0."""
    return 0.25 * price * price * (1.0 - price) * (1.0 - price)


def _sigmoid(cum_ret: float) -> float:
    x = SENSITIVITY * cum_ret * 10000
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _get_ladder_offset(minute: int) -> float:
    """Time-escalating price ladder offset."""
    if minute >= 10:
        return PRICE_LADDER[10]
    if minute >= 8:
        return PRICE_LADDER[8]
    if minute >= 5:
        return PRICE_LADDER[5]
    return PRICE_LADDER[2]


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
class ExecutionTrade:
    window_start: str
    entry_minute: int
    direction: str  # YES or NO
    entry_price: float
    entry_method: str  # GTC, GTC+LADDER, FAK
    settlement: float  # 1.0 or 0.0
    pnl_pct: float
    pnl_dollar: float
    fee: float
    correct: bool


def _simulate_old_execution(windows: list[list[Any]]) -> list[ExecutionTrade]:
    """Old execution: single GTC at minute 8-10, FOK at 11+, max 0.80."""
    trades: list[ExecutionTrade] = []
    old_max = 0.80

    for window in windows:
        if len(window) < MINUTES_PER_WINDOW:
            continue
        open_price = float(window[0].open)
        if open_price <= 0:
            continue

        # Try entry at minutes 8, 9, 10, then FOK at 11, 12
        for entry_min in [8, 9, 10, 11, 12]:
            candle = window[entry_min]
            cum_return = (float(candle.close) - open_price) / open_price
            abs_return = abs(cum_return)

            # Threshold check
            threshold = {8: 0.0010, 9: 0.0008, 10: 0.0006, 11: 0.0004, 12: 0.0002}.get(entry_min, 0.001)
            if abs_return < threshold:
                continue

            direction = "YES" if cum_return > 0 else "NO"
            yes_price = _sigmoid(cum_return)
            entry_price = yes_price if direction == "YES" else (1.0 - yes_price)

            if entry_price > old_max:
                continue

            # Settlement
            final_cum_ret = (float(window[-1].close) - open_price) / open_price
            correct = (final_cum_ret > 0 and direction == "YES") or (final_cum_ret < 0 and direction == "NO")
            settlement = 1.0 if correct else 0.0
            fee = _polymarket_fee(entry_price) * POSITION_SIZE
            shares = POSITION_SIZE / entry_price
            pnl = shares * (settlement - entry_price) - fee

            trades.append(ExecutionTrade(
                window_start=str(window[0].timestamp),
                entry_minute=entry_min,
                direction=direction,
                entry_price=entry_price,
                entry_method="GTC" if entry_min <= 10 else "FOK",
                settlement=settlement,
                pnl_pct=(settlement / entry_price - 1.0) * 100,
                pnl_dollar=pnl,
                fee=fee,
                correct=correct,
            ))
            break  # One entry per window

    return trades


def _simulate_new_execution(windows: list[list[Any]]) -> list[ExecutionTrade]:
    """New execution: GTC-first at min 2 + price ladder + FAK at min 11+.

    Simulates the REALISTIC fill scenario:
    - GTC at minutes 2-4: LOW fill probability on desert CLOB (10% fill rate)
    - GTC+ladder at minutes 5-10: MEDIUM fill probability (20-40% per escalation)
    - FAK at minutes 11+: HIGH fill probability (90% if within price cap)

    Uses a fill probability model based on observed 17% GTC fill rate
    from live monitoring data (9 windows, 1/6 GTC fills).
    """
    trades: list[ExecutionTrade] = []

    for window in windows:
        if len(window) < MINUTES_PER_WINDOW:
            continue
        open_price = float(window[0].open)
        if open_price <= 0:
            continue

        filled = False

        # Phase 1: GTC at minutes 2-4 (passive, low fill rate on desert CLOB)
        for entry_min in [2, 3, 4]:
            candle = window[entry_min]
            cum_return = (float(candle.close) - open_price) / open_price
            abs_return = abs(cum_return)

            # Early entry requires stronger signal
            threshold = {2: 0.0030, 3: 0.0025, 4: 0.0020}.get(entry_min, 0.003)
            if abs_return < threshold:
                continue

            direction = "YES" if cum_return > 0 else "NO"
            yes_price = _sigmoid(cum_return)
            entry_price = yes_price if direction == "YES" else (1.0 - yes_price)

            if entry_price > MAX_CLOB_ENTRY_PRICE:
                continue

            # GTC fill probability: ~10% on desert book (from live data)
            # Use deterministic model: fill if entry price <= 0.55 (very aggressive limit)
            # In practice, GTC only fills if counterparty appears — rare on 15m BTC
            # Simulate 10% fill rate using hash of window start
            fill_hash = hash(str(window[0].timestamp) + str(entry_min)) % 100
            if fill_hash >= 10:  # 90% skip (desert book)
                continue

            # Settlement
            final_cum_ret = (float(window[-1].close) - open_price) / open_price
            correct = (final_cum_ret > 0 and direction == "YES") or (final_cum_ret < 0 and direction == "NO")
            settlement = 1.0 if correct else 0.0
            fee = 0.0  # GTC = maker, 0% fee
            shares = POSITION_SIZE / entry_price
            pnl = shares * (settlement - entry_price)

            trades.append(ExecutionTrade(
                window_start=str(window[0].timestamp),
                entry_minute=entry_min,
                direction=direction,
                entry_price=entry_price,
                entry_method="GTC",
                settlement=settlement,
                pnl_pct=(settlement / entry_price - 1.0) * 100,
                pnl_dollar=pnl,
                fee=fee,
                correct=correct,
            ))
            filled = True
            break

        if filled:
            continue

        # Phase 2: GTC+Ladder at minutes 5-10 (escalating, medium fill rate)
        for entry_min in [5, 7, 9, 10]:
            candle = window[entry_min]
            cum_return = (float(candle.close) - open_price) / open_price
            abs_return = abs(cum_return)

            threshold = {5: 0.0010, 7: 0.0008, 9: 0.0006, 10: 0.0005}.get(entry_min, 0.001)
            if abs_return < threshold:
                continue

            direction = "YES" if cum_return > 0 else "NO"
            yes_price = _sigmoid(cum_return)
            base_price = yes_price if direction == "YES" else (1.0 - yes_price)
            ladder_offset = _get_ladder_offset(entry_min)
            entry_price = min(base_price + ladder_offset, MAX_CLOB_ENTRY_PRICE)

            if entry_price > MAX_CLOB_ENTRY_PRICE:
                continue

            # Ladder fill probability: higher offset → higher fill chance
            # min5: 20%, min7: 30%, min9: 40%, min10: 50%
            fill_rates = {5: 20, 7: 30, 9: 40, 10: 50}
            fill_hash = hash(str(window[0].timestamp) + f"L{entry_min}") % 100
            if fill_hash >= fill_rates.get(entry_min, 30):
                continue

            final_cum_ret = (float(window[-1].close) - open_price) / open_price
            correct = (final_cum_ret > 0 and direction == "YES") or (final_cum_ret < 0 and direction == "NO")
            settlement = 1.0 if correct else 0.0
            fee = 0.0  # GTC = maker, 0% fee
            shares = POSITION_SIZE / entry_price
            pnl = shares * (settlement - entry_price)

            trades.append(ExecutionTrade(
                window_start=str(window[0].timestamp),
                entry_minute=entry_min,
                direction=direction,
                entry_price=entry_price,
                entry_method=f"GTC+LADDER(+{ladder_offset:.0%})",
                settlement=settlement,
                pnl_pct=(settlement / entry_price - 1.0) * 100,
                pnl_dollar=pnl,
                fee=fee,
                correct=correct,
            ))
            filled = True
            break

        if filled:
            continue

        # Phase 3: FAK at minutes 11-12 (taker, high fill rate)
        for entry_min in [11, 12]:
            candle = window[entry_min]
            cum_return = (float(candle.close) - open_price) / open_price
            abs_return = abs(cum_return)

            threshold = {11: 0.0004, 12: 0.0002}.get(entry_min, 0.0004)
            if abs_return < threshold:
                continue

            direction = "YES" if cum_return > 0 else "NO"
            yes_price = _sigmoid(cum_return)
            # FAK crosses spread: use best_ask (slightly worse than model price)
            entry_price = yes_price if direction == "YES" else (1.0 - yes_price)
            # Simulate taker slippage: +1-2c spread crossing
            entry_price = min(entry_price + 0.02, FAK_MAX_ENTRY_PRICE)

            if entry_price > FAK_MAX_ENTRY_PRICE:
                continue

            # FAK fill rate: 90% (fills whatever is available)
            fill_hash = hash(str(window[0].timestamp) + f"FAK{entry_min}") % 100
            if fill_hash >= 90:
                continue

            final_cum_ret = (float(window[-1].close) - open_price) / open_price
            correct = (final_cum_ret > 0 and direction == "YES") or (final_cum_ret < 0 and direction == "NO")
            settlement = 1.0 if correct else 0.0
            fee = _polymarket_fee(entry_price) * POSITION_SIZE
            shares = POSITION_SIZE / entry_price
            pnl = shares * (settlement - entry_price) - fee

            trades.append(ExecutionTrade(
                window_start=str(window[0].timestamp),
                entry_minute=entry_min,
                direction=direction,
                entry_price=entry_price,
                entry_method="FAK",
                settlement=settlement,
                pnl_pct=(settlement / entry_price - 1.0) * 100,
                pnl_dollar=pnl,
                fee=fee,
                correct=correct,
            ))
            filled = True
            break

    return trades


def _print_results(label: str, trades: list[ExecutionTrade]) -> None:
    if not trades:
        print(f"\n  {label}: NO TRADES")
        return

    total_pnl = sum(t.pnl_dollar for t in trades)
    total_fees = sum(t.fee for t in trades)
    wins = sum(1 for t in trades if t.correct)
    losses = len(trades) - wins
    win_rate = wins / len(trades) * 100
    avg_entry = sum(t.entry_price for t in trades) / len(trades)
    ev_per_trade = total_pnl / len(trades)
    avg_win_pnl = sum(t.pnl_dollar for t in trades if t.correct) / max(wins, 1)
    avg_loss_pnl = sum(t.pnl_dollar for t in trades if not t.correct) / max(losses, 1)

    # Entry method breakdown
    methods: dict[str, list[ExecutionTrade]] = {}
    for t in trades:
        methods.setdefault(t.entry_method, []).append(t)

    # Entry minute breakdown
    minutes: dict[int, list[ExecutionTrade]] = {}
    for t in trades:
        minutes.setdefault(t.entry_minute, []).append(t)

    print(f"\n  {'='*90}")
    print(f"  {label}")
    print(f"  {'='*90}")
    print(f"  Trades:       {len(trades):>8}")
    print(f"  Win Rate:     {win_rate:>7.1f}%  ({wins}W / {losses}L)")
    print(f"  Avg Entry:    ${avg_entry:>7.4f}")
    print(f"  EV/trade:     ${ev_per_trade:>+7.2f}")
    print(f"  Total P&L:    ${total_pnl:>+12.2f}")
    print(f"  Total Fees:   ${total_fees:>+12.2f}")
    print(f"  Avg Win:      ${avg_win_pnl:>+7.2f}")
    print(f"  Avg Loss:     ${avg_loss_pnl:>+7.2f}")

    print(f"\n  Entry Method Breakdown:")
    print(f"  {'Method':<25} {'Trades':>7} {'WinR%':>7} {'AvgEntry':>9} {'EV/trade':>10} {'TotalPnL':>12}")
    print(f"  {'-'*73}")
    for method, mtrades in sorted(methods.items()):
        mwins = sum(1 for t in mtrades if t.correct)
        mwr = mwins / len(mtrades) * 100
        mavg = sum(t.entry_price for t in mtrades) / len(mtrades)
        mpnl = sum(t.pnl_dollar for t in mtrades)
        mev = mpnl / len(mtrades)
        print(f"  {method:<25} {len(mtrades):>7} {mwr:>6.1f}% ${mavg:>8.4f} ${mev:>+9.2f} ${mpnl:>+11.2f}")

    print(f"\n  Entry Minute Breakdown:")
    print(f"  {'Min':>5} {'Trades':>7} {'WinR%':>7} {'AvgEntry':>9} {'EV/trade':>10} {'TotalPnL':>12}")
    print(f"  {'-'*55}")
    for minute, mtrades in sorted(minutes.items()):
        mwins = sum(1 for t in mtrades if t.correct)
        mwr = mwins / len(mtrades) * 100
        mavg = sum(t.entry_price for t in mtrades) / len(mtrades)
        mpnl = sum(t.pnl_dollar for t in mtrades)
        mev = mpnl / len(mtrades)
        print(f"  {minute:>5} {len(mtrades):>7} {mwr:>6.1f}% ${mavg:>8.4f} ${mev:>+9.2f} ${mpnl:>+11.2f}")


def main() -> None:
    print("=" * 100)
    print("  EXECUTION OVERHAUL BACKTEST")
    print("  GTC-First + Price Ladder + FAK Sweep vs Old Single-Entry")
    print("  132K 1m BTC candles, binary settlement, realistic fill rates")
    print("=" * 100)

    # Load data
    loader = DataLoader()
    all_candles: list[Any] = []
    for csv_path in CSV_PATHS:
        if csv_path.exists():
            candles = loader.load_csv(csv_path)
            print(f"  Loaded {len(candles):>6} from {csv_path.name}")
            all_candles.extend(candles)

    print(f"\n  Total candles: {len(all_candles)}")
    windows = _group_into_15m_windows(all_candles)
    print(f"  15m windows:  {len(windows)}")

    # Run both execution strategies
    old_trades = _simulate_old_execution(windows)
    new_trades = _simulate_new_execution(windows)

    _print_results("OLD EXECUTION: GTC at min 8-10, FOK at 11+ (max $0.80)", old_trades)
    _print_results("NEW EXECUTION: GTC-first min 2 + Ladder + FAK (max GTC $0.65, FAK $0.85)", new_trades)

    # Comparison
    old_pnl = sum(t.pnl_dollar for t in old_trades)
    new_pnl = sum(t.pnl_dollar for t in new_trades)
    old_trades_n = len(old_trades)
    new_trades_n = len(new_trades)
    old_wr = sum(1 for t in old_trades if t.correct) / max(old_trades_n, 1) * 100
    new_wr = sum(1 for t in new_trades if t.correct) / max(new_trades_n, 1) * 100
    old_ev = old_pnl / max(old_trades_n, 1)
    new_ev = new_pnl / max(new_trades_n, 1)
    old_fees = sum(t.fee for t in old_trades)
    new_fees = sum(t.fee for t in new_trades)

    print(f"\n  {'='*90}")
    print(f"  HEAD-TO-HEAD COMPARISON")
    print(f"  {'='*90}")
    print(f"  {'Metric':<25} {'OLD':>15} {'NEW':>15} {'Delta':>15}")
    print(f"  {'-'*70}")
    print(f"  {'Trades':<25} {old_trades_n:>15} {new_trades_n:>15} {new_trades_n - old_trades_n:>+15}")
    print(f"  {'Win Rate':<25} {old_wr:>14.1f}% {new_wr:>14.1f}% {new_wr - old_wr:>+14.1f}%")
    print(f"  {'EV/trade':<25} ${old_ev:>+13.2f} ${new_ev:>+13.2f} ${new_ev - old_ev:>+13.2f}")
    print(f"  {'Total P&L':<25} ${old_pnl:>+13.2f} ${new_pnl:>+13.2f} ${new_pnl - old_pnl:>+13.2f}")
    print(f"  {'Total Fees':<25} ${old_fees:>+13.2f} ${new_fees:>+13.2f} ${new_fees - old_fees:>+13.2f}")
    print(f"  {'Avg Entry Price':<25} ${sum(t.entry_price for t in old_trades)/max(old_trades_n,1):>13.4f} ${sum(t.entry_price for t in new_trades)/max(new_trades_n,1):>13.4f}")

    # Verdict
    print(f"\n  {'='*90}")
    if new_pnl > old_pnl:
        improvement = (new_pnl - old_pnl) / abs(old_pnl) * 100 if old_pnl != 0 else float("inf")
        print(f"  VERDICT: NEW execution WINS by ${new_pnl - old_pnl:+,.2f} ({improvement:+.1f}%)")
    else:
        print(f"  VERDICT: OLD execution wins by ${old_pnl - new_pnl:+,.2f}")
    print(f"  {'='*90}")


if __name__ == "__main__":
    main()
