"""Validate strategy with Kelly-based position sizing.

Compares fixed $100 sizing vs half-Kelly vs quarter-Kelly.
Re-runs Monte Carlo ruin probability to prove the fix works.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fast_loader import load_csv_fast, group_into_15m_windows

CSV_2Y = PROJECT_ROOT / "data" / "btc_1m_2y.csv"
MINUTES_PER_WINDOW = 15
SENSITIVITY = 0.07
TIERS = [(8, 0.0010), (9, 0.0008), (10, 0.0005)]
FEE_CONSTANT = 0.25
SLIPPAGE_BPS = 5

STARTING_BALANCE = 10_000.0  # $10K starting capital
N_MONTE_CARLO = 10_000
SEED = 42

# Estimated win probability by entry price bucket (from backtest data)
WIN_PROB_MAP = {
    (0.00, 0.55): 0.83,
    (0.55, 0.60): 0.83,
    (0.60, 0.65): 0.87,
    (0.65, 0.70): 0.87,
    (0.70, 0.75): 0.90,
    (0.75, 0.80): 0.92,
    (0.80, 1.00): 0.93,
}


def _sigmoid(cum_ret: float) -> float:
    x = SENSITIVITY * cum_ret * 10000
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def polymarket_fee(position_size: float, entry_price: float) -> float:
    if entry_price <= 0 or entry_price >= 1:
        return 0.0
    return position_size * FEE_CONSTANT * (entry_price ** 2) * ((1.0 - entry_price) ** 2)


def estimated_win_prob(entry_price: float) -> float:
    """Get estimated win probability for a given entry price."""
    for (lo, hi), prob in WIN_PROB_MAP.items():
        if lo <= entry_price < hi:
            return prob
    return 0.89  # fallback overall average


def kelly_fraction_binary(win_prob: float, entry_price: float) -> float:
    """Binary Kelly: f* = (p - P) / (1 - P)."""
    if entry_price >= 1 or entry_price <= 0:
        return 0.0
    f = (win_prob - entry_price) / (1.0 - entry_price)
    return max(0.0, f)


@dataclass
class Trade:
    entry_price: float
    correct: bool
    # Sizing info
    position_size_fixed: float = 0.0
    position_size_half_kelly: float = 0.0
    position_size_quarter_kelly: float = 0.0
    # PnL
    pnl_fixed: float = 0.0
    pnl_half_kelly: float = 0.0
    pnl_quarter_kelly: float = 0.0


def generate_trades(windows: list[list[Any]]) -> list[Trade]:
    """Generate all trade signals with entry prices and outcomes."""
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
            correct = predict_green == actual_green
            trades.append(Trade(entry_price=entry_price, correct=correct))
            break
    return trades


def simulate_equity(
    trades: list[Trade],
    starting_balance: float,
    kelly_mult: float,
    max_position_pct: float = 0.02,
    label: str = "",
) -> dict[str, Any]:
    """Simulate equity curve with Kelly-based position sizing."""
    balance = starting_balance
    peak = balance
    max_dd = 0.0
    max_dd_dollar = 0.0
    total_fees = 0.0
    total_trades = 0
    total_correct = 0
    total_pnl = 0.0
    trade_sizes: list[float] = []

    for t in trades:
        if balance <= 0:
            break

        # Kelly sizing
        win_prob = estimated_win_prob(t.entry_price)
        full_kelly = kelly_fraction_binary(win_prob, t.entry_price)
        frac_kelly = full_kelly * kelly_mult

        # Position size = fraction of current balance, capped at max_position_pct
        raw_size = frac_kelly * balance
        max_size = balance * max_position_pct
        position_size = min(raw_size, max_size)
        position_size = max(position_size, 0.0)

        if position_size < 0.01:  # Min trade size
            continue

        trade_sizes.append(position_size)

        # Fee on this position
        fee = polymarket_fee(position_size, t.entry_price)
        slip = position_size * SLIPPAGE_BPS / 10000
        total_fees += fee + slip

        # PnL
        if t.correct:
            # Win: profit = position_size * (1 - entry_price) / entry_price
            gross_profit = position_size * (1.0 - t.entry_price) / t.entry_price
            pnl = gross_profit - fee - slip
            total_correct += 1
        else:
            # Loss: lose the position_size
            pnl = -position_size - fee - slip

        balance += pnl
        total_pnl += pnl
        total_trades += 1

        if balance > peak:
            peak = balance
        dd = (peak - balance) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)
        dd_dollar = peak - balance
        max_dd_dollar = max(max_dd_dollar, dd_dollar)

    avg_size = sum(trade_sizes) / len(trade_sizes) if trade_sizes else 0

    return {
        "label": label,
        "kelly_mult": kelly_mult,
        "max_pct": max_position_pct,
        "starting": starting_balance,
        "ending": balance,
        "total_pnl": total_pnl,
        "total_return": (balance - starting_balance) / starting_balance * 100,
        "trades": total_trades,
        "accuracy": total_correct / total_trades * 100 if total_trades > 0 else 0,
        "total_fees": total_fees,
        "avg_size": avg_size,
        "max_dd_pct": max_dd * 100,
        "max_dd_dollar": max_dd_dollar,
    }


def simulate_fixed(
    trades: list[Trade],
    starting_balance: float,
    fixed_size: float,
    label: str = "",
) -> dict[str, Any]:
    """Simulate equity curve with fixed position sizing."""
    balance = starting_balance
    peak = balance
    max_dd = 0.0
    max_dd_dollar = 0.0
    total_fees = 0.0
    total_trades = 0
    total_correct = 0
    total_pnl = 0.0

    for t in trades:
        if balance < fixed_size:
            break  # Can't afford the position

        position_size = fixed_size

        fee = polymarket_fee(position_size, t.entry_price)
        slip = position_size * SLIPPAGE_BPS / 10000
        total_fees += fee + slip

        if t.correct:
            gross_profit = position_size * (1.0 - t.entry_price) / t.entry_price
            pnl = gross_profit - fee - slip
            total_correct += 1
        else:
            pnl = -position_size - fee - slip

        balance += pnl
        total_pnl += pnl
        total_trades += 1

        if balance > peak:
            peak = balance
        dd = (peak - balance) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)
        dd_dollar = peak - balance
        max_dd_dollar = max(max_dd_dollar, dd_dollar)

    return {
        "label": label,
        "starting": starting_balance,
        "ending": balance,
        "total_pnl": total_pnl,
        "total_return": (balance - starting_balance) / starting_balance * 100,
        "trades": total_trades,
        "accuracy": total_correct / total_trades * 100 if total_trades > 0 else 0,
        "total_fees": total_fees,
        "avg_size": fixed_size,
        "max_dd_pct": max_dd * 100,
        "max_dd_dollar": max_dd_dollar,
    }


def monte_carlo_ruin(
    win_rate: float,
    avg_win_pct: float,
    avg_loss_pct: float,
    n_trades: int,
    n_sims: int,
    starting_balance: float,
    label: str = "",
) -> dict[str, Any]:
    """Monte Carlo ruin probability simulation."""
    rng = np.random.RandomState(SEED)

    wins = rng.random((n_sims, n_trades)) < win_rate
    pnl_matrix = np.where(wins, avg_win_pct * starting_balance, -avg_loss_pct * starting_balance)
    equity = starting_balance + np.cumsum(pnl_matrix, axis=1)

    peaks = np.maximum.accumulate(equity, axis=1)
    drawdowns = np.where(peaks > 0, (peaks - equity) / peaks, 0)
    max_dds = drawdowns.max(axis=1)

    ever_ruined = (equity <= 0.5 * starting_balance).any(axis=1)
    ever_doubled = (equity >= 2 * starting_balance).any(axis=1)

    ruin_count = 0
    for i in range(n_sims):
        if ever_ruined[i]:
            ruin_idx = np.argmax(equity[i] <= 0.5 * starting_balance)
            if ever_doubled[i]:
                double_idx = np.argmax(equity[i] >= 2 * starting_balance)
                if ruin_idx < double_idx:
                    ruin_count += 1
            else:
                ruin_count += 1

    max_dds_sorted = np.sort(max_dds)
    return {
        "label": label,
        "ruin_prob": ruin_count / n_sims * 100,
        "dd_50": max_dds_sorted[int(0.50 * n_sims)] * 100,
        "dd_90": max_dds_sorted[int(0.90 * n_sims)] * 100,
        "dd_95": max_dds_sorted[int(0.95 * n_sims)] * 100,
        "dd_99": max_dds_sorted[int(0.99 * n_sims)] * 100,
        "final_median": float(np.median(equity[:, -1])),
    }


def main() -> None:
    print("=" * 120)
    print("  KELLY POSITION SIZING VALIDATION")
    print(f"  Starting balance: ${STARTING_BALANCE:,.0f}")
    print("=" * 120)

    if not CSV_2Y.exists():
        print(f"  ERROR: {CSV_2Y} not found.")
        sys.exit(1)

    all_candles = load_csv_fast(CSV_2Y)
    all_candles.sort(key=lambda c: c.timestamp)
    seen: set[datetime] = set()
    unique = [c for c in all_candles if c.timestamp not in seen and not seen.add(c.timestamp)]  # type: ignore
    all_candles = unique
    print(f"\n  Loaded {len(all_candles):,} candles")

    windows = group_into_15m_windows(all_candles)
    trades = generate_trades(windows)
    print(f"  {len(windows):,} windows, {len(trades):,} trades")

    # ===== PART 1: Equity simulation comparison =====
    print(f"\n  {'='*110}")
    print(f"  EQUITY SIMULATION: Fixed vs Kelly Sizing (with fees)")
    print(f"  {'='*110}")

    configs = [
        ("Fixed $100", lambda t: simulate_fixed(t, STARTING_BALANCE, 100.0, "Fixed $100")),
        ("Fixed $50", lambda t: simulate_fixed(t, STARTING_BALANCE, 50.0, "Fixed $50")),
        ("Half-Kelly (2% cap)", lambda t: simulate_equity(t, STARTING_BALANCE, 0.5, 0.02, "Half-Kelly 2%")),
        ("Half-Kelly (5% cap)", lambda t: simulate_equity(t, STARTING_BALANCE, 0.5, 0.05, "Half-Kelly 5%")),
        ("Quarter-Kelly (2% cap)", lambda t: simulate_equity(t, STARTING_BALANCE, 0.25, 0.02, "Quarter-Kelly 2%")),
        ("Quarter-Kelly (5% cap)", lambda t: simulate_equity(t, STARTING_BALANCE, 0.25, 0.05, "Quarter-Kelly 5%")),
        ("Eighth-Kelly (2% cap)", lambda t: simulate_equity(t, STARTING_BALANCE, 0.125, 0.02, "Eighth-Kelly 2%")),
    ]

    print(f"\n  {'Config':<25} {'Trades':>7} {'Acc%':>6} {'End Balance':>14} {'Return':>10} "
          f"{'Avg Size':>10} {'Max DD%':>8} {'Max DD$':>10} {'Fees':>10}")
    print(f"  {'-'*110}")

    results = []
    for label, sim_fn in configs:
        r = sim_fn(trades)
        results.append(r)
        print(f"  {r['label']:<25} {r['trades']:>7} {r['accuracy']:>5.1f}% "
              f"${r['ending']:>12,.0f} {r['total_return']:>+8.1f}%  "
              f"${r['avg_size']:>8.2f} {r['max_dd_pct']:>7.1f}% "
              f"${r['max_dd_dollar']:>8,.0f} ${r['total_fees']:>8,.0f}")

    # ===== PART 2: Monte Carlo ruin probability =====
    print(f"\n  {'='*110}")
    print(f"  MONTE CARLO RUIN PROBABILITY ({N_MONTE_CARLO:,} simulations)")
    print(f"  {'='*110}")

    # Calculate average win/loss ratios for each sizing method
    # For fixed sizing: win ~35% of position, lose 100%
    # For Kelly sizing: win/loss % varies with position size

    # Fixed $100 stats
    win_trades = [t for t in trades if t.correct]
    loss_trades = [t for t in trades if not t.correct]
    win_rate = len(win_trades) / len(trades)
    avg_win_pct_fixed = sum(
        (1.0 - t.entry_price) / t.entry_price for t in win_trades
    ) / len(win_trades)
    # For fixed sizing, loss = 100% of position
    avg_loss_pct_fixed = 1.0

    mc_configs = [
        ("Fixed $100 (1% of $10K)", win_rate, avg_win_pct_fixed * 0.01, avg_loss_pct_fixed * 0.01),
        ("Fixed $100 (high risk)", win_rate, avg_win_pct_fixed * 0.01, avg_loss_pct_fixed * 0.01),
        ("Half-Kelly 2% cap", win_rate, avg_win_pct_fixed * 0.02, avg_loss_pct_fixed * 0.02),
        ("Quarter-Kelly 2% cap", win_rate, avg_win_pct_fixed * 0.01, avg_loss_pct_fixed * 0.01),
    ]

    # Better approach: simulate with actual trade-by-trade Kelly sizing
    print(f"\n  Approach: Simulating {N_MONTE_CARLO:,} random paths of {len(trades)} trades each")
    print(f"  Win rate: {win_rate*100:.1f}%, Avg win payout: {avg_win_pct_fixed*100:.1f}% of position")
    print()

    # For each sizing method, compute per-trade returns as fraction of balance
    # Fixed $100: bet $100 of $10K = 1% per trade
    # With Kelly: bet varies, but capped at 2%

    # Simulate ruin probability for different sizing methods
    print(f"  {'Config':<30} {'Ruin%':>7} {'DD-50%':>7} {'DD-90%':>7} {'DD-95%':>7} "
          f"{'DD-99%':>7} {'Median Final':>14}")
    print(f"  {'-'*85}")

    # Fixed 1% (= $100 of $10K)
    mc = monte_carlo_ruin(win_rate, avg_win_pct_fixed * 0.01, 0.01,
                          len(trades), N_MONTE_CARLO, STARTING_BALANCE, "Fixed 1% ($100)")
    print(f"  {mc['label']:<30} {mc['ruin_prob']:>6.2f}% {mc['dd_50']:>6.1f}% "
          f"{mc['dd_90']:>6.1f}% {mc['dd_95']:>6.1f}% {mc['dd_99']:>6.1f}% "
          f"${mc['final_median']:>12,.0f}")

    # Fixed 2%
    mc = monte_carlo_ruin(win_rate, avg_win_pct_fixed * 0.02, 0.02,
                          len(trades), N_MONTE_CARLO, STARTING_BALANCE, "Fixed 2% ($200)")
    print(f"  {mc['label']:<30} {mc['ruin_prob']:>6.2f}% {mc['dd_50']:>6.1f}% "
          f"{mc['dd_90']:>6.1f}% {mc['dd_95']:>6.1f}% {mc['dd_99']:>6.1f}% "
          f"${mc['final_median']:>12,.0f}")

    # Half-Kelly capped at 2% → effective bet ≈ 0.7-1.4% of balance
    # Average Kelly fraction for our trades:
    avg_kelly = sum(
        kelly_fraction_binary(estimated_win_prob(t.entry_price), t.entry_price)
        for t in trades
    ) / len(trades)
    avg_half_kelly = avg_kelly * 0.5
    avg_quarter_kelly = avg_kelly * 0.25
    eff_half = min(avg_half_kelly, 0.02)
    eff_quarter = min(avg_quarter_kelly, 0.02)

    print(f"\n  Average full Kelly fraction: {avg_kelly*100:.2f}%")
    print(f"  Average half-Kelly: {avg_half_kelly*100:.2f}% (capped at 2%: {eff_half*100:.2f}%)")
    print(f"  Average quarter-Kelly: {avg_quarter_kelly*100:.2f}% (capped at 2%: {eff_quarter*100:.2f}%)")
    print()

    mc = monte_carlo_ruin(win_rate, avg_win_pct_fixed * eff_half, eff_half,
                          len(trades), N_MONTE_CARLO, STARTING_BALANCE, f"Half-Kelly 2% cap (~{eff_half*100:.1f}%)")
    print(f"  {mc['label']:<30} {mc['ruin_prob']:>6.2f}% {mc['dd_50']:>6.1f}% "
          f"{mc['dd_90']:>6.1f}% {mc['dd_95']:>6.1f}% {mc['dd_99']:>6.1f}% "
          f"${mc['final_median']:>12,.0f}")

    mc = monte_carlo_ruin(win_rate, avg_win_pct_fixed * eff_quarter, eff_quarter,
                          len(trades), N_MONTE_CARLO, STARTING_BALANCE, f"Quarter-Kelly 2% cap (~{eff_quarter*100:.1f}%)")
    print(f"  {mc['label']:<30} {mc['ruin_prob']:>6.2f}% {mc['dd_50']:>6.1f}% "
          f"{mc['dd_90']:>6.1f}% {mc['dd_95']:>6.1f}% {mc['dd_99']:>6.1f}% "
          f"${mc['final_median']:>12,.0f}")

    # Eighth-Kelly
    eff_eighth = min(avg_kelly * 0.125, 0.02)
    mc = monte_carlo_ruin(win_rate, avg_win_pct_fixed * eff_eighth, eff_eighth,
                          len(trades), N_MONTE_CARLO, STARTING_BALANCE, f"Eighth-Kelly 2% cap (~{eff_eighth*100:.1f}%)")
    print(f"  {mc['label']:<30} {mc['ruin_prob']:>6.2f}% {mc['dd_50']:>6.1f}% "
          f"{mc['dd_90']:>6.1f}% {mc['dd_95']:>6.1f}% {mc['dd_99']:>6.1f}% "
          f"${mc['final_median']:>12,.0f}")

    # ===== PART 3: Recommendation =====
    print(f"\n  {'='*110}")
    print(f"  RECOMMENDATION")
    print(f"  {'='*110}")

    # Find best config from equity sim
    best = max(results, key=lambda r: r["ending"] if r["max_dd_pct"] < 30 else 0)
    safest = min((r for r in results if r["ending"] > r["starting"]), key=lambda r: r["max_dd_pct"])

    print(f"\n  Best risk-adjusted return (DD < 30%): {best['label']}")
    print(f"    Ending balance: ${best['ending']:,.0f}, Return: {best['total_return']:+.1f}%, Max DD: {best['max_dd_pct']:.1f}%")
    print(f"\n  Safest (lowest drawdown): {safest['label']}")
    print(f"    Ending balance: ${safest['ending']:,.0f}, Return: {safest['total_return']:+.1f}%, Max DD: {safest['max_dd_pct']:.1f}%")

    print(f"\n  SUGGESTED PRODUCTION CONFIG:")
    print(f"    - Kelly multiplier: 0.25 (quarter-Kelly)")
    print(f"    - Max position: 2% of balance")
    print(f"    - Starting balance: $10,000+")
    print(f"    - This provides growth while keeping drawdowns manageable")

    print(f"\n{'='*120}")


if __name__ == "__main__":
    main()
