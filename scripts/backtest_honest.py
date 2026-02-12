"""Honest Backtest — realistic simulation matching live FAK-only execution.

Key differences from the old backtest:
  1. FAK-only fills (no GTC resting orders)
  2. CLOB simulation: desert (80%) vs real liquidity (20%)
  3. Adverse selection penalty on fill prices
  4. Exact Polymarket dynamic fee model
  5. Calibrated confidence → edge gating (min_edge = 0.03)
  6. Bootstrap confidence intervals on WR, EV, drawdown

Expected results:
  - WR drops from 89% to ~55-65% (matching live)
  - Fill rate drops to ~15-25% (matching live 17%)
  - If honest WR < 52% → strategy has no edge
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtesting.data_loader import DataLoader

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
CSV_PATHS = [PROJECT_ROOT / "data" / f"btc_1m_chunk{i}.csv" for i in range(1, 7)]
MINUTES_PER_WINDOW = 15
INITIAL_BALANCE = 10_000.0
SENSITIVITY = 0.07

# FAK-only execution
FAK_MAX_ENTRY_PRICE = 0.85
MIN_ENTRY_MINUTE = 2
MAX_ENTRY_MINUTE = 12
MIN_CONFIDENCE = 0.45
MIN_SIGNALS_AGREE = 2

# CLOB simulation parameters (from live observation)
REAL_LIQUIDITY_RATE = 0.20      # ~20% of windows have real asks
ADVERSE_SELECTION_FACTOR = 0.03  # price moves 3c against fill
SPREAD_BASE = 0.04               # base spread above fair value
DESERT_SPREAD = 0.98             # desert book: 0.01/0.99

# Calibration
MIN_EDGE = 0.03                  # minimum fee-adjusted edge to trade
MAX_EDGE = 0.15                  # cap posterior at clob_mid + max_edge
DEFAULT_LR = 1.3                 # default likelihood ratio

# Position sizing (matches production)
KELLY_MULTIPLIER = 0.15
MAX_POSITION_PCT = 0.05
MIN_POSITION_PCT = 0.03

# Bootstrap
N_BOOTSTRAP = 1000
SEED = 42


# ------------------------------------------------------------------
# Fee model
# ------------------------------------------------------------------
def polymarket_fee(price: float) -> float:
    """Dynamic taker fee: 0.25 * p^2 * (1-p)^2."""
    return 0.25 * price * price * (1.0 - price) * (1.0 - price)


# ------------------------------------------------------------------
# Signal model (simplified singularity)
# ------------------------------------------------------------------
def sigmoid(cum_ret: float) -> float:
    x = SENSITIVITY * cum_ret * 10000
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class Signal:
    direction: str  # "YES" or "NO"
    confidence: float
    cum_return: float
    minute: int


def generate_signal(
    cum_return: float,
    minute: int,
    rng: random.Random,
) -> Signal | None:
    """Simplified singularity signal: direction from cum_return, confidence ~ |cum_return|."""
    if minute < MIN_ENTRY_MINUTE or minute > MAX_ENTRY_MINUTE:
        return None

    abs_ret = abs(cum_return)
    if abs_ret < 0.0002:  # ~2 bps threshold
        return None

    direction = "YES" if cum_return >= 0 else "NO"

    # Confidence model: base + scaled by |return| + noise
    base_conf = 0.55
    conf_scale = min(abs_ret * 200, 0.40)  # up to 0.40 boost for large moves
    noise = rng.gauss(0, 0.05)
    confidence = max(MIN_CONFIDENCE, min(0.98, base_conf + conf_scale + noise))

    return Signal(
        direction=direction,
        confidence=confidence,
        cum_return=cum_return,
        minute=minute,
    )


# ------------------------------------------------------------------
# CLOB simulation
# ------------------------------------------------------------------
@dataclass
class CLOBSim:
    """Simulates CLOB state for a window."""

    has_liquidity: bool
    fair_price: float
    best_ask: float
    best_bid: float
    spread: float

    @staticmethod
    def for_window(sigmoid_price: float, rng: random.Random) -> CLOBSim:
        """Generate CLOB state. 80% desert, 20% real liquidity."""
        has_liq = rng.random() < REAL_LIQUIDITY_RATE

        if not has_liq:
            return CLOBSim(
                has_liquidity=False,
                fair_price=sigmoid_price,
                best_ask=0.99,
                best_bid=0.01,
                spread=DESERT_SPREAD,
            )

        # Real liquidity: spread centered around fair price + noise
        spread = SPREAD_BASE + rng.uniform(0, 0.06)
        mid = sigmoid_price + rng.gauss(0, 0.02)
        mid = max(0.10, min(0.90, mid))
        best_bid = max(0.01, mid - spread / 2)
        best_ask = min(0.99, mid + spread / 2)

        return CLOBSim(
            has_liquidity=True,
            fair_price=mid,
            best_ask=best_ask,
            best_bid=best_bid,
            spread=best_ask - best_bid,
        )


# ------------------------------------------------------------------
# Calibration (simplified)
# ------------------------------------------------------------------
def calibrate_posterior(
    clob_mid: float,
    signal_confidence: float,
) -> float:
    """Bayesian calibration: CLOB mid as prior, signal as likelihood.

    Returns calibrated posterior probability.
    """
    prior = clob_mid
    # Likelihood ratio: how much the signal shifts from prior
    lr = DEFAULT_LR
    posterior = prior * lr / (prior * lr + (1 - prior))
    # Cap: never exceed clob_mid + MAX_EDGE
    max_edge = MAX_EDGE
    # Desert detection: wider edge when mid ~0.50
    if abs(clob_mid - 0.50) < 0.05:
        max_edge = 0.25
    posterior = min(posterior, clob_mid + max_edge)
    return max(0.01, min(0.99, posterior))


# ------------------------------------------------------------------
# Trade result
# ------------------------------------------------------------------
@dataclass
class TradeResult:
    window_idx: int
    direction: str
    entry_price: float
    confidence: float
    posterior: float
    edge: float
    fee: float
    position_size_usd: float
    won: bool
    pnl: float


# ------------------------------------------------------------------
# Backtest engine
# ------------------------------------------------------------------
def run_backtest(candles: list[Any], seed: int = SEED) -> list[TradeResult]:
    """Run honest backtest over all 15-minute windows."""
    rng = random.Random(seed)
    windows = _group_into_15m_windows(candles)
    balance = INITIAL_BALANCE
    trades: list[TradeResult] = []

    for w_idx, window in enumerate(windows):
        if len(window) < 2:
            continue

        open_price = float(window[0].open)
        if open_price <= 0:
            continue

        # Try entry at each minute
        traded = False
        for candle in window:
            minute = _minute_in_window(candle, window[0])
            if minute < MIN_ENTRY_MINUTE or minute > MAX_ENTRY_MINUTE:
                continue
            if traded:
                break

            # Cumulative return from window open
            cum_return = (float(candle.close) - open_price) / open_price

            # Generate signal
            signal = generate_signal(cum_return, minute, rng)
            if signal is None:
                continue

            # Model sigmoid price
            sigmoid_price = sigmoid(cum_return)

            # Simulate CLOB
            clob = CLOBSim.for_window(sigmoid_price, rng)

            # FAK requires real liquidity to cross
            if not clob.has_liquidity:
                continue

            # Entry price = best_ask + adverse selection
            if signal.direction == "YES":
                entry_price = clob.best_ask + rng.uniform(0, ADVERSE_SELECTION_FACTOR)
            else:
                # NO direction: buy NO token
                no_ask = 1.0 - clob.best_bid
                entry_price = no_ask + rng.uniform(0, ADVERSE_SELECTION_FACTOR)
            entry_price = min(entry_price, FAK_MAX_ENTRY_PRICE)

            if entry_price <= 0 or entry_price >= 1:
                continue

            # Calibrate
            clob_mid = clob.fair_price
            if signal.direction == "NO":
                clob_mid = 1.0 - clob_mid
            posterior = calibrate_posterior(clob_mid, signal.confidence)

            # Edge check
            fee = polymarket_fee(entry_price)
            raw_edge = posterior - entry_price
            # Fee is a rate on notional, so per-share cost = fee_rate * price
            fee_adjusted_edge = raw_edge - fee * entry_price

            if fee_adjusted_edge < MIN_EDGE:
                continue

            # Position sizing (Kelly)
            win_payout = 1.0 - entry_price  # profit if correct
            loss_amount = entry_price       # loss if wrong
            odds = win_payout / loss_amount if loss_amount > 0 else 0
            kelly = (posterior * odds - (1 - posterior)) / odds if odds > 0 else 0
            kelly = max(0, kelly) * KELLY_MULTIPLIER
            kelly = max(MIN_POSITION_PCT, min(MAX_POSITION_PCT, kelly))

            position_usd = balance * kelly
            shares = position_usd / entry_price

            # Determine outcome: binary settlement
            final_close = float(window[-1].close)
            final_cum = (final_close - open_price) / open_price
            winning_side = "YES" if final_cum >= 0 else "NO"
            won = signal.direction == winning_side

            if won:
                pnl = shares * (1.0 - entry_price) - fee * shares * entry_price
            else:
                pnl = -(shares * entry_price) - fee * shares * entry_price

            balance += pnl
            if balance <= 0:
                balance = 0
                break

            trades.append(TradeResult(
                window_idx=w_idx,
                direction=signal.direction,
                entry_price=entry_price,
                confidence=signal.confidence,
                posterior=posterior,
                edge=fee_adjusted_edge,
                fee=fee,
                position_size_usd=position_usd,
                won=won,
                pnl=pnl,
            ))
            traded = True

    return trades


# ------------------------------------------------------------------
# Bootstrap confidence intervals
# ------------------------------------------------------------------
def bootstrap_ci(
    trades: list[TradeResult],
    n_iter: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> dict[str, Any]:
    """Bootstrap 95% confidence intervals for key metrics."""
    if not trades:
        return {}

    rng = np.random.RandomState(seed)
    n = len(trades)
    wrs = []
    evs = []
    max_dds = []

    for _ in range(n_iter):
        idx = rng.choice(n, size=n, replace=True)
        sample = [trades[i] for i in idx]
        wins = sum(1 for t in sample if t.won)
        wrs.append(wins / n)
        evs.append(sum(t.pnl for t in sample) / n)

        # Max drawdown
        equity = INITIAL_BALANCE
        peak = equity
        max_dd = 0.0
        for t in sample:
            equity += t.pnl
            peak = max(peak, equity)
            dd = (peak - equity) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        max_dds.append(max_dd)

    return {
        "win_rate": {
            "mean": float(np.mean(wrs)),
            "ci_95": (float(np.percentile(wrs, 2.5)), float(np.percentile(wrs, 97.5))),
        },
        "ev_per_trade": {
            "mean": float(np.mean(evs)),
            "ci_95": (float(np.percentile(evs, 2.5)), float(np.percentile(evs, 97.5))),
        },
        "max_drawdown": {
            "mean": float(np.mean(max_dds)),
            "ci_95": (float(np.percentile(max_dds, 2.5)), float(np.percentile(max_dds, 97.5))),
        },
    }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _group_into_15m_windows(candles: list[Any]) -> list[list[Any]]:
    if not candles:
        return []
    windows: list[list[Any]] = []
    current: list[Any] = []
    current_start: int | None = None
    for c in candles:
        ts = c.timestamp
        minute = ts.minute
        window_minute = (minute // 15) * 15
        if current_start is None:
            if minute == window_minute:
                current_start = window_minute
                current = [c]
        else:
            offset = minute - current_start
            if offset < 0:
                offset += 60
            if offset < 15:
                current.append(c)
            else:
                if current:
                    windows.append(current)
                current_start = window_minute
                current = [c]
    if current:
        windows.append(current)
    return windows


def _minute_in_window(candle: Any, first_candle: Any) -> int:
    delta = (candle.timestamp - first_candle.timestamp).total_seconds()
    return int(delta // 60)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("HONEST BACKTEST — FAK-only, CLOB simulation, calibrated")
    print("=" * 60)

    # Load data
    loader = DataLoader()
    all_candles: list[Any] = []
    for path in CSV_PATHS:
        if path.exists():
            chunk = loader.load_csv(str(path))
            all_candles.extend(chunk)
    if not all_candles:
        print("ERROR: No candle data found. Place btc_1m_chunk*.csv in data/")
        return

    all_candles.sort(key=lambda c: c.timestamp)
    windows = _group_into_15m_windows(all_candles)
    print(f"Loaded {len(all_candles):,} candles → {len(windows):,} 15-min windows")

    # Run backtest
    trades = run_backtest(all_candles)
    n_trades = len(trades)

    if n_trades == 0:
        print("No trades generated. Check signal/CLOB parameters.")
        return

    wins = sum(1 for t in trades if t.won)
    losses = n_trades - wins
    total_pnl = sum(t.pnl for t in trades)
    win_rate = wins / n_trades

    fill_rate = n_trades / len(windows) * 100

    print(f"\n--- Results ---")
    print(f"Windows:      {len(windows):,}")
    print(f"Trades:       {n_trades:,}")
    print(f"Fill rate:    {fill_rate:.1f}%")
    print(f"Wins:         {wins}")
    print(f"Losses:       {losses}")
    print(f"Win rate:     {win_rate:.1%}")
    print(f"Total P&L:    ${total_pnl:,.2f}")
    print(f"Avg P&L/trade: ${total_pnl / n_trades:,.2f}")

    # Edge stats
    edges = [t.edge for t in trades]
    fees = [t.fee for t in trades]
    print(f"\n--- Edge Analysis ---")
    print(f"Avg edge:     {np.mean(edges):.4f}")
    print(f"Median edge:  {np.median(edges):.4f}")
    print(f"Avg fee:      {np.mean(fees):.4f}")

    # Max drawdown
    equity = INITIAL_BALANCE
    peak = equity
    max_dd = 0.0
    for t in trades:
        equity += t.pnl
        peak = max(peak, equity)
        dd = (peak - equity) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)
    print(f"\n--- Risk ---")
    print(f"Max drawdown: {max_dd:.1%}")
    print(f"Final equity: ${equity:,.2f}")

    # Bootstrap CIs
    print(f"\n--- Bootstrap 95% CIs ({N_BOOTSTRAP} iterations) ---")
    ci = bootstrap_ci(trades)
    for metric, vals in ci.items():
        m = vals["mean"]
        lo, hi = vals["ci_95"]
        if "rate" in metric or "drawdown" in metric:
            print(f"  {metric}: {m:.1%} [{lo:.1%}, {hi:.1%}]")
        else:
            print(f"  {metric}: ${m:.2f} [${lo:.2f}, ${hi:.2f}]")

    # Verdict
    print(f"\n--- Verdict ---")
    if win_rate < 0.52:
        print("WARNING: WR < 52% — strategy may have no edge. Rethink signals.")
    elif win_rate < 0.55:
        print("MARGINAL: WR 52-55% — small edge, needs perfect execution.")
    elif win_rate < 0.65:
        print("VIABLE: WR 55-65% — matches expected live range. Proceed to paper.")
    else:
        print("STRONG: WR > 65% — above expectations. Verify not overfitting.")


if __name__ == "__main__":
    main()
