#!/usr/bin/env python3
"""Comprehensive backtest & singularity parameter optimisation.

Loads 1-minute BTC candles, groups them into 15-minute windows, runs
each window through the strategies with proper context, and tracks
binary settlement outcomes (correct direction = $1, wrong = $0).

Usage:
    python scripts/backtest_optimize.py                    # full run
    python scripts/backtest_optimize.py --strategy singularity  # single strat
    python scripts/backtest_optimize.py --optimize          # param sweep
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Suppress all structlog output for speed — must be done before any src imports
import os
os.environ["MVHE_LOG_LEVEL"] = "CRITICAL"
import logging
logging.disable(logging.CRITICAL)

# Monkey-patch structlog to suppress all output before any module uses get_logger()
import structlog
_devnull = open(os.devnull, "w")
structlog.configure(
    processors=[structlog.dev.ConsoleRenderer()],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(file=_devnull),
    cache_logger_on_first_use=False,
)
# Also patch the core logging module so get_logger() returns silent loggers
import src.core.logging as _core_log
_core_log._CONFIGURED = True

from src.backtesting.data_loader import DataLoader
from src.config.loader import ConfigLoader
from src.models.market import (
    Candle,
    MarketState,
    OrderBookSnapshot,
    OrderBookLevel,
    Side,
)
from src.models.signal import SignalType
from src.strategies import registry

# Ensure strategies are registered
import contextlib

with contextlib.suppress(ImportError):
    import src.strategies.reversal_catcher  # noqa: F401
with contextlib.suppress(ImportError):
    import src.strategies.false_sentiment  # noqa: F401
with contextlib.suppress(ImportError):
    import src.strategies.singularity  # noqa: F401


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIGMOID_SENSITIVITY = 0.07
POLYMARKET_FEE_CONSTANT = 0.25
SLIPPAGE_BPS = 5
WINDOW_BARS = 15


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    """A single completed trade."""
    window_idx: int
    entry_minute: int
    direction: str        # "YES" or "NO"
    entry_price: float
    settlement_price: float  # 1.0 or 0.0
    pnl: float
    fee: float
    is_win: bool
    entry_time: datetime | None = None


@dataclass
class BacktestStats:
    """Aggregated backtest statistics."""
    strategy: str
    total_windows: int = 0
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    total_fees: float = 0.0
    net_pnl: float = 0.0
    equity_curve: list[float] = field(default_factory=list)
    trades: list[TradeRecord] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.wins / self.total_trades if self.total_trades > 0 else 0.0

    @property
    def profit_factor(self) -> float:
        return self.gross_profit / self.gross_loss if self.gross_loss > 0 else float("inf")

    @property
    def avg_pnl_per_trade(self) -> float:
        return self.net_pnl / self.total_trades if self.total_trades > 0 else 0.0

    @property
    def sharpe_ratio(self) -> float:
        if len(self.equity_curve) < 2:
            return 0.0
        returns = []
        for i in range(1, len(self.equity_curve)):
            prev = self.equity_curve[i - 1]
            if prev > 0:
                returns.append((self.equity_curve[i] - prev) / prev)
        if not returns or len(returns) < 2:
            return 0.0
        mean_r = sum(returns) / len(returns)
        var_r = sum((r - mean_r) ** 2 for r in returns) / len(returns)
        std_r = math.sqrt(var_r) if var_r > 0 else 0.0
        if std_r == 0:
            return 0.0
        return (mean_r / std_r) * math.sqrt(252)

    @property
    def max_drawdown(self) -> float:
        if not self.equity_curve:
            return 0.0
        peak = self.equity_curve[0]
        max_dd = 0.0
        for v in self.equity_curve:
            if v > peak:
                peak = v
            if peak > 0:
                dd = (peak - v) / peak
                if dd > max_dd:
                    max_dd = dd
        return max_dd

    def summary(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "total_windows": self.total_windows,
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate * 100, 2),
            "gross_profit": round(self.gross_profit, 2),
            "gross_loss": round(self.gross_loss, 2),
            "total_fees": round(self.total_fees, 2),
            "net_pnl": round(self.net_pnl, 2),
            "avg_pnl_per_trade": round(self.avg_pnl_per_trade, 2),
            "profit_factor": round(self.profit_factor, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "max_drawdown_pct": round(self.max_drawdown * 100, 2),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def polymarket_fee(position_size: float, yes_price: float) -> float:
    """Dynamic Polymarket taker fee."""
    p = yes_price
    return position_size * POLYMARKET_FEE_CONSTANT * (p ** 2) * ((1 - p) ** 2)


def sigmoid_yes_price(cum_return: float) -> float:
    """Map cumulative return to YES probability via sigmoid."""
    return 1.0 / (1.0 + math.exp(-cum_return / SIGMOID_SENSITIVITY))


def load_1m_candles(data_dir: Path) -> list[Candle]:
    """Load 1-minute candles from the 2-year dataset."""
    csv_path = data_dir / "btc_1m_2y.csv"
    if not csv_path.exists():
        # Fall back to chunks
        chunks = sorted(data_dir.glob("btc_1m_chunk*.csv"))
        if not chunks:
            print("ERROR: No 1m candle data found")
            sys.exit(1)
        loader = DataLoader()
        all_candles: list[Candle] = []
        for chunk in chunks:
            all_candles.extend(loader.load_csv(chunk))
        all_candles.sort(key=lambda c: c.timestamp)
        return all_candles

    loader = DataLoader()
    return loader.load_csv(csv_path)


def group_into_windows(candles: list[Candle]) -> list[list[Candle]]:
    """Group 1m candles into 15-minute windows.

    Uses timestamp alignment: each window starts when minute % 15 == 0.
    """
    if not candles:
        return []

    windows: list[list[Candle]] = []
    current_window: list[Candle] = []

    for candle in candles:
        minute = candle.timestamp.minute
        # Detect window boundary
        if minute % 15 == 0 and current_window:
            windows.append(current_window)
            current_window = []
        current_window.append(candle)

    if current_window:
        windows.append(current_window)

    return windows


def build_orderbook(yes_price: float, bullish: bool) -> OrderBookSnapshot:
    """Build a synthetic order book for backtesting.

    Creates a book with mild imbalance in the direction of the move.
    """
    p = Decimal(str(round(yes_price, 4)))
    spread = Decimal("0.01")

    if bullish:
        bid_sizes = [Decimal("300"), Decimal("200"), Decimal("150"), Decimal("100"), Decimal("50")]
        ask_sizes = [Decimal("80"), Decimal("60"), Decimal("40"), Decimal("30"), Decimal("20")]
    else:
        bid_sizes = [Decimal("80"), Decimal("60"), Decimal("40"), Decimal("30"), Decimal("20")]
        ask_sizes = [Decimal("300"), Decimal("200"), Decimal("150"), Decimal("100"), Decimal("50")]

    bids = []
    asks = []
    for i in range(5):
        bids.append(OrderBookLevel(
            price=p - spread * Decimal(str(i + 1)),
            size=bid_sizes[i],
        ))
        asks.append(OrderBookLevel(
            price=p + spread * Decimal(str(i)),
            size=ask_sizes[i],
        ))

    return OrderBookSnapshot(
        timestamp=datetime.now(tz=timezone.utc),
        market_id="backtest",
        bids=bids,
        asks=asks,
    )


# ---------------------------------------------------------------------------
# Core backtest runner
# ---------------------------------------------------------------------------

def run_strategy_backtest(
    strategy_name: str,
    windows: list[list[Candle]],
    config: ConfigLoader,
    initial_balance: float = 10000.0,
    position_size_pct: float = 0.02,
    verbose: bool = False,
    no_compound: bool = False,
) -> BacktestStats:
    """Run a strategy over all 15m windows with proper context.

    For each window:
    1. Determine true outcome (is the 15m close > open?)
    2. Step through minutes 0-14, building the context the strategy expects
    3. If the strategy emits an ENTRY signal, record the trade
    4. Settle at window end: correct direction = $1, wrong = $0
    """
    strategy = registry.create(strategy_name, config)
    stats = BacktestStats(strategy=strategy_name)
    balance = initial_balance

    for w_idx, window in enumerate(windows):
        if len(window) < 5:
            continue

        stats.total_windows += 1
        window_open = float(window[0].open)
        window_close = float(window[-1].close)
        true_direction = "YES" if window_close >= window_open else "NO"

        traded_this_window = False

        for minute_idx, candle in enumerate(window):
            if traded_this_window:
                break

            # Compute current state
            current_close = float(candle.close)
            cum_return = (current_close - window_open) / window_open if window_open > 0 else 0.0
            yes_price = sigmoid_yes_price(cum_return)

            # Time remaining in seconds (each candle = 1 minute)
            remaining_minutes = max(0, len(window) - 1 - minute_idx)
            time_remaining_s = remaining_minutes * 60

            market_state = MarketState(
                market_id=f"backtest_w{w_idx}",
                yes_price=Decimal(str(round(yes_price, 6))),
                no_price=Decimal(str(round(1.0 - yes_price, 6))),
                time_remaining_seconds=time_remaining_s,
            )

            is_bullish = cum_return > 0
            orderbook = build_orderbook(yes_price, is_bullish)

            # Build context matching what the strategies expect
            context: dict[str, Any] = {
                "candles_1m": window[: minute_idx + 1],
                "window_open_price": window_open,
                "minute_in_window": minute_idx,
                "yes_price": yes_price,
                # Singularity extras (backtest synthetic data)
                "recent_ticks": [c.close for c in window[max(0, minute_idx - 10) : minute_idx + 1]],
            }

            # Generate signals
            try:
                signals = strategy.generate_signals(market_state, orderbook, context)
            except Exception as e:
                if verbose:
                    print(f"  Signal error w={w_idx} m={minute_idx}: {e}")
                continue

            for sig in signals:
                if sig.signal_type != SignalType.ENTRY:
                    continue

                # Determine entry price and direction
                direction_str = sig.direction.value  # "YES" or "NO"
                entry_price = float(sig.entry_price) if sig.entry_price else yes_price

                # Compute position size
                if no_compound:
                    pos_size = initial_balance * position_size_pct
                else:
                    pos_size = min(balance * position_size_pct, balance * 0.05)
                if pos_size <= 0 or balance <= 0:
                    continue

                # Fee calculation
                fee = polymarket_fee(pos_size, entry_price)

                # Settlement: correct direction = $1.00, wrong = $0.00
                is_win = direction_str == true_direction
                settlement_price = 1.0 if is_win else 0.0

                # P&L: buy at entry_price, settle at settlement_price
                # For binary: pnl = (settlement - entry) * quantity
                quantity = pos_size / entry_price if entry_price > 0 else 0
                pnl = (settlement_price - entry_price) * quantity - fee

                trade = TradeRecord(
                    window_idx=w_idx,
                    entry_minute=minute_idx,
                    direction=direction_str,
                    entry_price=entry_price,
                    settlement_price=settlement_price,
                    pnl=pnl,
                    fee=fee,
                    is_win=is_win,
                    entry_time=candle.timestamp,
                )
                stats.trades.append(trade)
                stats.total_trades += 1

                if is_win:
                    stats.wins += 1
                    stats.gross_profit += pnl + fee
                else:
                    stats.losses += 1
                    stats.gross_loss += abs(pnl + fee)

                stats.total_fees += fee
                balance += pnl
                stats.net_pnl += pnl
                traded_this_window = True
                break

        # Record equity after each window
        stats.equity_curve.append(balance)

        if verbose and w_idx > 0 and w_idx % 5000 == 0:
            print(f"  Window {w_idx}/{len(windows)}: {stats.total_trades} trades, "
                  f"win_rate={stats.win_rate:.1%}, balance=${balance:.2f}")

    return stats


# ---------------------------------------------------------------------------
# Parameter optimisation
# ---------------------------------------------------------------------------

@dataclass
class OptimResult:
    """Result of a single optimisation trial."""
    params: dict[str, Any]
    win_rate: float
    net_pnl: float
    sharpe: float
    total_trades: int
    profit_factor: float
    max_drawdown: float


def create_config_with_overrides(
    base_config: ConfigLoader,
    overrides: dict[str, Any],
) -> ConfigLoader:
    """Create a new ConfigLoader with parameter overrides applied."""
    import copy
    new_config = ConfigLoader.__new__(ConfigLoader)
    new_config._config_dir = base_config._config_dir
    new_config._env = base_config._env
    new_config._config = copy.deepcopy(base_config._config)

    # Apply overrides via dotted key paths
    for key, value in overrides.items():
        parts = key.split(".")
        target = new_config._config
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value

    return new_config


def run_optimization(
    windows: list[list[Candle]],
    base_config: ConfigLoader,
    max_trials: int = 200,
) -> list[OptimResult]:
    """Sweep singularity parameters to find optimal configuration.

    Optimises:
    - min_signals_agree: [1, 2, 3]
    - min_confidence: [0.40, 0.50, 0.60, 0.72]
    - entry_minute_start: [5, 6, 7, 8]
    - entry_minute_end: [9, 10, 11]
    - weight_momentum: [0.30, 0.40, 0.50, 0.60]
    - momentum tier thresholds (scaled)
    """
    print("\n" + "=" * 60)
    print("  SINGULARITY PARAMETER OPTIMISATION")
    print("=" * 60)

    param_grid = {
        "min_signals_agree": [1, 2, 3],
        "min_confidence": [0.40, 0.50, 0.60, 0.72],
        "entry_minute_start": [5, 6, 7, 8],
        "entry_minute_end": [9, 10, 11],
        "weight_momentum": [0.30, 0.40, 0.50, 0.60],
    }

    # Generate all combinations
    keys = list(param_grid.keys())
    all_combos = list(itertools.product(*[param_grid[k] for k in keys]))
    total = min(len(all_combos), max_trials)

    print(f"  Total parameter combinations: {len(all_combos)}")
    print(f"  Running up to {total} trials")
    print()

    results: list[OptimResult] = []
    best_sharpe = -999.0
    best_params: dict[str, Any] = {}

    for trial_idx, combo in enumerate(all_combos[:total]):
        params = dict(zip(keys, combo))

        # Skip invalid combos
        if params["entry_minute_start"] >= params["entry_minute_end"]:
            continue

        # Build tier thresholds based on entry minutes
        start = params["entry_minute_start"]
        end = params["entry_minute_end"]
        tiers = []
        for m in range(start, end + 1):
            # Wider threshold for earlier minutes, tighter for later
            position = (m - start) / max(1, end - start)
            threshold = round(0.14 - position * 0.09, 2)  # 0.14 down to 0.05
            tiers.append({"minute": m, "threshold_pct": threshold})

        # Build config overrides
        overrides = {
            "strategy.singularity.min_signals_agree": params["min_signals_agree"],
            "strategy.singularity.min_confidence": params["min_confidence"],
            "strategy.singularity.entry_minute_start": params["entry_minute_start"],
            "strategy.singularity.entry_minute_end": params["entry_minute_end"],
            "strategy.singularity.weight_momentum": params["weight_momentum"],
            "strategy.singularity.weight_ofi": round((1.0 - params["weight_momentum"]) * 0.35, 3),
            "strategy.singularity.weight_futures": round((1.0 - params["weight_momentum"]) * 0.25, 3),
            "strategy.singularity.weight_vol": round((1.0 - params["weight_momentum"]) * 0.20, 3),
            "strategy.singularity.weight_time": round((1.0 - params["weight_momentum"]) * 0.20, 3),
            "strategy.singularity.entry_tiers": tiers,
        }

        trial_config = create_config_with_overrides(base_config, overrides)

        # Run backtest
        stats = run_strategy_backtest(
            "singularity", windows, trial_config, verbose=False,
        )

        if stats.total_trades < 50:
            continue

        result = OptimResult(
            params=params,
            win_rate=stats.win_rate,
            net_pnl=stats.net_pnl,
            sharpe=stats.sharpe_ratio,
            total_trades=stats.total_trades,
            profit_factor=stats.profit_factor,
            max_drawdown=stats.max_drawdown,
        )
        results.append(result)

        if stats.sharpe_ratio > best_sharpe:
            best_sharpe = stats.sharpe_ratio
            best_params = params.copy()

        if (trial_idx + 1) % 20 == 0:
            print(f"  Trial {trial_idx + 1}/{total}: best_sharpe={best_sharpe:.2f}, "
                  f"current trades={stats.total_trades}, wr={stats.win_rate:.1%}")

    # Sort by Sharpe
    results.sort(key=lambda r: r.sharpe, reverse=True)
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results(stats: BacktestStats) -> None:
    """Print formatted backtest results."""
    s = stats.summary()
    print()
    print("=" * 60)
    print(f"  BACKTEST: {s['strategy']}")
    print("=" * 60)
    print(f"  Windows processed     {s['total_windows']:>12,}")
    print(f"  Total trades          {s['total_trades']:>12,}")
    print(f"  Wins                  {s['wins']:>12,}")
    print(f"  Losses                {s['losses']:>12,}")
    print(f"  Win Rate              {s['win_rate']:>11.2f}%")
    print(f"  Gross Profit          ${s['gross_profit']:>11,.2f}")
    print(f"  Gross Loss            ${s['gross_loss']:>11,.2f}")
    print(f"  Total Fees            ${s['total_fees']:>11,.2f}")
    print(f"  Net P&L               ${s['net_pnl']:>11,.2f}")
    print(f"  Avg P&L per Trade     ${s['avg_pnl_per_trade']:>11,.2f}")
    print(f"  Profit Factor         {s['profit_factor']:>12.4f}")
    print(f"  Sharpe Ratio          {s['sharpe_ratio']:>12.2f}")
    print(f"  Max Drawdown          {s['max_drawdown_pct']:>11.2f}%")
    print("=" * 60)

    # Win rate by entry minute
    if stats.trades:
        print("\n  Win Rate by Entry Minute:")
        from collections import defaultdict
        by_minute: dict[int, list[bool]] = defaultdict(list)
        for t in stats.trades:
            by_minute[t.entry_minute].append(t.is_win)
        for m in sorted(by_minute):
            wins = sum(by_minute[m])
            total = len(by_minute[m])
            wr = wins / total if total > 0 else 0
            print(f"    Minute {m:>2d}: {wr:>6.1%}  ({wins}/{total})")

    # Win rate by hour of day
    if stats.trades:
        print("\n  Win Rate by Hour (UTC):")
        by_hour: dict[int, list[bool]] = defaultdict(list)
        for t in stats.trades:
            if t.entry_time:
                by_hour[t.entry_time.hour].append(t.is_win)
        for h in sorted(by_hour):
            wins = sum(by_hour[h])
            total = len(by_hour[h])
            wr = wins / total if total > 0 else 0
            print(f"    {h:02d}:00 UTC: {wr:>6.1%}  ({wins}/{total})")


def print_optimization_results(results: list[OptimResult], top_n: int = 20) -> None:
    """Print top optimisation results."""
    print(f"\n  TOP {top_n} CONFIGURATIONS:")
    print("  " + "-" * 100)
    print(f"  {'#':>3}  {'Sharpe':>7}  {'WinRate':>7}  {'Trades':>7}  {'NetPnL':>10}  "
          f"{'PF':>7}  {'MaxDD':>6}  {'MinAgree':>8}  {'MinConf':>7}  "
          f"{'Start':>5}  {'End':>3}  {'WtMom':>5}")
    print("  " + "-" * 100)

    for i, r in enumerate(results[:top_n]):
        print(f"  {i+1:>3}  {r.sharpe:>7.2f}  {r.win_rate:>6.1%}  {r.total_trades:>7}  "
              f"${r.net_pnl:>9,.2f}  {r.profit_factor:>7.2f}  {r.max_drawdown:>5.1%}  "
              f"{r.params['min_signals_agree']:>8}  {r.params['min_confidence']:>7.2f}  "
              f"{r.params['entry_minute_start']:>5}  {r.params['entry_minute_end']:>3}  "
              f"{r.params['weight_momentum']:>5.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="MVHE Backtest & Optimisation")
    parser.add_argument("--strategy", type=str, default="all",
                        help="Strategy to backtest (momentum_confirmation, singularity, all)")
    parser.add_argument("--optimize", action="store_true",
                        help="Run parameter optimisation for singularity")
    parser.add_argument("--max-trials", type=int, default=200,
                        help="Max optimisation trials")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output", type=str, default="data/optimization_results.json",
                        help="Output file for results")
    parser.add_argument("--no-compound", action="store_true",
                        help="Use fixed position size (no compounding)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print("=" * 60)
    print("  MVHE BACKTEST ENGINE")
    print("=" * 60)
    print(f"  Loading 1-minute candles from {data_dir}...")
    t0 = time.time()

    candles = load_1m_candles(data_dir)
    load_time = time.time() - t0
    print(f"  Loaded {len(candles):,} candles in {load_time:.1f}s")
    print(f"  Date range: {candles[0].timestamp} to {candles[-1].timestamp}")

    print(f"  Grouping into 15-minute windows...")
    windows = group_into_windows(candles)
    print(f"  {len(windows):,} windows created")

    # Load config
    config = ConfigLoader(config_dir="config")
    config.load()

    all_results: dict[str, Any] = {}

    # --- Run momentum_confirmation ---
    if args.strategy in ("all", "momentum_confirmation"):
        config_mc = ConfigLoader(config_dir="config")
        config_mc.load()
        config_mc.load_strategy("momentum_confirmation")

        print(f"\n  Running momentum_confirmation backtest...")
        t0 = time.time()
        mc_stats = run_strategy_backtest(
            "momentum_confirmation", windows, config_mc, verbose=args.verbose,
            no_compound=args.no_compound,
        )
        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s")
        print_results(mc_stats)
        all_results["momentum_confirmation"] = mc_stats.summary()

    # --- Run singularity ---
    if args.strategy in ("all", "singularity"):
        config_sg = ConfigLoader(config_dir="config")
        config_sg.load()
        config_sg.load_strategy("singularity")

        print(f"\n  Running singularity backtest...")
        t0 = time.time()
        sg_stats = run_strategy_backtest(
            "singularity", windows, config_sg, verbose=args.verbose,
            no_compound=args.no_compound,
        )
        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s")
        print_results(sg_stats)
        all_results["singularity"] = sg_stats.summary()

    # --- Optimisation ---
    if args.optimize:
        config_opt = ConfigLoader(config_dir="config")
        config_opt.load()
        config_opt.load_strategy("singularity")

        t0 = time.time()
        opt_results = run_optimization(
            windows, config_opt, max_trials=args.max_trials,
        )
        elapsed = time.time() - t0
        print(f"\n  Optimisation completed in {elapsed:.1f}s ({len(opt_results)} valid trials)")

        if opt_results:
            print_optimization_results(opt_results)

            # Save best config
            best = opt_results[0]
            all_results["optimization"] = {
                "best_params": best.params,
                "best_sharpe": best.sharpe,
                "best_win_rate": round(best.win_rate * 100, 2),
                "best_net_pnl": round(best.net_pnl, 2),
                "best_trades": best.total_trades,
                "best_profit_factor": round(best.profit_factor, 4),
                "best_max_drawdown": round(best.max_drawdown * 100, 2),
                "top_10": [
                    {
                        "params": r.params,
                        "sharpe": round(r.sharpe, 2),
                        "win_rate": round(r.win_rate * 100, 2),
                        "net_pnl": round(r.net_pnl, 2),
                        "trades": r.total_trades,
                    }
                    for r in opt_results[:10]
                ],
            }

            # Print optimal singularity config
            print("\n" + "=" * 60)
            print("  OPTIMAL SINGULARITY CONFIGURATION")
            print("=" * 60)
            print(f"  min_signals_agree = {best.params['min_signals_agree']}")
            print(f"  min_confidence    = {best.params['min_confidence']}")
            print(f"  entry_minute_start = {best.params['entry_minute_start']}")
            print(f"  entry_minute_end   = {best.params['entry_minute_end']}")
            print(f"  weight_momentum    = {best.params['weight_momentum']}")
            print(f"  ---")
            print(f"  Sharpe Ratio       = {best.sharpe:.2f}")
            print(f"  Win Rate           = {best.win_rate:.1%}")
            print(f"  Net P&L            = ${best.net_pnl:,.2f}")
            print(f"  Total Trades       = {best.total_trades:,}")
            print(f"  Profit Factor      = {best.profit_factor:.4f}")
            print(f"  Max Drawdown       = {best.max_drawdown:.1%}")
            print("=" * 60)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
