"""BWO v6 — Hybrid Strategy Backtest (REALISTIC POLYMARKET PRICING).

STRATEGY:
  1. ALWAYS buy YES at $0.50 BEFORE window opens (cheap BWO entry)
  2. At minute N (1-3), observe BTC price action + run continuation filter
  3. Use filter to MANAGE position (hold/sell), NOT to decide entry

DECISION MATRIX:
  Position favorable (BTC UP) + filter says "continue"  → HOLD to settlement
  Position adverse   (BTC DOWN) + filter says "continue" → SELL early (cut loss)
  Position favorable + filter says "no continue"         → variant-dependent
  Position adverse   + filter says "no continue"         → variant-dependent

REALISTIC PRICING:
  - Entry: $0.50 + slippage before window
  - Exit: estimated from empirical P(UP | BTC return at min N) - spread
  - Settlement: $1.00 or $0.00
  - Fees: Polymarket fee model on entry AND exit
  - Bid-ask spread on early exits: 2-3 cents

OUTPUT: comprehensive risk/reward metrics (win rate, avg win/loss, R:R, Sharpe,
max drawdown, Kelly, total PnL), walk-forward validated.
"""

from __future__ import annotations

import json
import math
import sys
import time
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest_before_window import (
    FEE_CONSTANT,
    SLIPPAGE_BPS,
    WindowData,
    compute_all_features,
)
from scripts.fast_loader import FastCandle, group_into_15m_windows, load_csv_fast
from scripts.bwo_continuation_backtest import (
    compute_signal_quality_features,
    compute_taker_features_simple,
    compute_mi_bits,
    _parse_ts,
)

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BTC_SPOT_CSV = PROJECT_ROOT / "data" / "btc_1m_2y.csv"
BTC_FUTURES_CSV = PROJECT_ROOT / "data" / "btc_futures_1m_2y.csv"
ETH_FUTURES_CSV = PROJECT_ROOT / "data" / "eth_futures_1m_2y.csv"
DVOL_CSV = PROJECT_ROOT / "data" / "deribit_dvol_1m.csv"
DERIBIT_PERP_CSV = PROJECT_ROOT / "data" / "deribit_btc_perp_1m.csv"
REPORT_JSON = PROJECT_ROOT / "data" / "bwo_hybrid_report.json"

# Trade parameters
POSITION_SIZE = 100.0       # $ per trade
ENTRY_PRICE = 0.50          # Pre-window YES price
ENTRY_SLIPPAGE = 0.005      # Buy at 0.505 (0.5 cents slippage)
EXIT_SPREAD = 0.02          # Sell at fair_value - 2 cents (bid-ask spread)
EXIT_SLIPPAGE = 0.005       # Additional slippage on exit

# Walk-forward
WF_TRAIN_MONTHS = 3
WF_TEST_MONTHS = 1

ENTRY_MINUTES = [1, 2, 3]

# Continuation filter confidence thresholds to sweep
CONT_THRESHOLDS = [round(0.50 + i * 0.01, 2) for i in range(46)]


def _flush() -> None:
    sys.stdout.flush()


def polymarket_fee(position_size: float, price: float) -> float:
    if price <= 0 or price >= 1:
        return 0.0
    return position_size * FEE_CONSTANT * (price ** 2) * ((1.0 - price) ** 2)


# ---------------------------------------------------------------------------
# Polymarket Pricing Model
# ---------------------------------------------------------------------------

class PolymarketPricingModel:
    """Estimates YES token fair value at minute N given BTC return.

    Uses logistic regression trained on historical data:
    P(BTC finishes UP | BTC_return_at_minute_N) → YES fair value.

    When selling YES: sell_price = fair_value - EXIT_SPREAD - EXIT_SLIPPAGE
    """

    def __init__(self) -> None:
        self.model: LogisticRegression | None = None
        self._fitted = False

    def fit(
        self,
        btc_returns_at_n: np.ndarray,
        final_up: np.ndarray,
    ) -> None:
        """Train pricing model from historical data."""
        X = btc_returns_at_n.reshape(-1, 1)
        self.model = LogisticRegression(C=1.0, max_iter=1000)
        self.model.fit(X, final_up)
        self._fitted = True

    def fair_value(self, btc_return: float) -> float:
        """Estimate YES token fair value given BTC return at minute N."""
        if not self._fitted or self.model is None:
            return 0.50
        prob = self.model.predict_proba(np.array([[btc_return]]))[0, 1]
        return float(np.clip(prob, 0.01, 0.99))

    def sell_price(self, btc_return: float) -> float:
        """Price we can sell YES at (fair value minus spread and slippage)."""
        fv = self.fair_value(btc_return)
        return max(0.01, fv - EXIT_SPREAD - EXIT_SLIPPAGE)

    def stats(self, btc_returns: np.ndarray) -> dict[str, float]:
        """Return pricing model statistics for reporting."""
        if not self._fitted or self.model is None:
            return {}
        probs = self.model.predict_proba(btc_returns.reshape(-1, 1))[:, 1]
        return {
            "coef": float(self.model.coef_[0][0]),
            "intercept": float(self.model.intercept_[0]),
            "mean_prob": float(probs.mean()),
            "std_prob": float(probs.std()),
            "min_prob": float(probs.min()),
            "max_prob": float(probs.max()),
        }


# ---------------------------------------------------------------------------
# Trade result dataclass
# ---------------------------------------------------------------------------

@dataclass
class HybridTradeResult:
    window_idx: int
    action: str  # "hold_win", "hold_lose", "sell_early_cut_loss", "sell_early_take_profit", "sell_early_uncertain"
    entry_price: float       # Effective entry price (0.50 + slippage)
    exit_price: float        # Settlement ($1/$0) or early exit price
    pnl_gross: float
    pnl_net: float
    fee_entry: float
    fee_exit: float
    correct: bool            # Did we end up profitable?
    btc_return_at_n: float   # BTC return at decision minute
    continuation_prob: float # Model's continuation probability
    early_direction: str     # "Up" or "Down" (first N candles)
    resolution: str          # Final 15m resolution


# ---------------------------------------------------------------------------
# Strategy Variants
# ---------------------------------------------------------------------------

def simulate_hybrid_trade(
    wd: WindowData,
    windows: list[list[FastCandle]],
    entry_minute: int,
    continuation_prob: float,
    cont_threshold: float,
    btc_return_at_n: float,
    pricing_model: PolymarketPricingModel,
    variant: str,
) -> HybridTradeResult:
    """Simulate a single hybrid trade with realistic Polymarket pricing.

    Always buys YES at $0.50 before window.
    Uses continuation filter + pricing model for exit decisions.
    """
    w = windows[wd.window_idx]

    # Entry: always buy YES at $0.50 + slippage
    effective_entry = ENTRY_PRICE + ENTRY_SLIPPAGE  # 0.505
    fee_entry = polymarket_fee(POSITION_SIZE, effective_entry)
    n_tokens = POSITION_SIZE / effective_entry  # ~198 tokens

    # Early direction at minute N
    early_up = btc_return_at_n > 0
    early_down = btc_return_at_n < 0
    early_flat = not early_up and not early_down

    # Filter prediction
    filter_continue = continuation_prob >= cont_threshold
    filter_no_continue = not filter_continue

    # Position status: we bought YES, so BTC UP = favorable
    position_favorable = early_up
    position_adverse = early_down

    # Determine action based on variant
    action = "hold"  # default

    if variant == "A":
        # CONSERVATIVE: Only sell when adverse + continuation confirmed
        if position_adverse and filter_continue:
            action = "sell_early"
        else:
            action = "hold"

    elif variant == "B":
        # BALANCED: Sell on adverse continuation, take profit on favorable no-continue
        if position_adverse and filter_continue:
            action = "sell_early"  # Cut loss
        elif position_favorable and filter_no_continue:
            action = "sell_early"  # Take small profit before potential reversal
        else:
            action = "hold"

    elif variant == "C":
        # AGGRESSIVE: Only hold when favorable + continuation confirmed
        if position_favorable and filter_continue:
            action = "hold"
        else:
            action = "sell_early"

    # Execute action
    if action == "sell_early":
        # Sell YES at minute N market price
        exit_price = pricing_model.sell_price(btc_return_at_n)
        fee_exit = polymarket_fee(POSITION_SIZE, exit_price)

        pnl_gross = n_tokens * (exit_price - effective_entry)
        pnl_net = pnl_gross - fee_entry - fee_exit

        # Categorize the sell
        if position_adverse and filter_continue:
            action_label = "sell_early_cut_loss"
        elif position_favorable and filter_no_continue:
            action_label = "sell_early_take_profit"
        else:
            action_label = "sell_early_uncertain"

        return HybridTradeResult(
            window_idx=wd.window_idx,
            action=action_label,
            entry_price=effective_entry,
            exit_price=exit_price,
            pnl_gross=pnl_gross,
            pnl_net=pnl_net,
            fee_entry=fee_entry,
            fee_exit=fee_exit,
            correct=pnl_net > 0,
            btc_return_at_n=btc_return_at_n,
            continuation_prob=continuation_prob,
            early_direction="Up" if early_up else ("Down" if early_down else "Flat"),
            resolution=wd.resolution,
        )

    else:  # HOLD to settlement
        # Settlement: YES pays $1 if BTC finished UP, $0 if DOWN
        settlement = 1.0 if wd.resolution == "Up" else 0.0
        fee_exit = polymarket_fee(POSITION_SIZE, settlement) if settlement > 0 else 0.0

        pnl_gross = n_tokens * (settlement - effective_entry)
        pnl_net = pnl_gross - fee_entry - fee_exit

        if settlement > 0:
            action_label = "hold_win"
        else:
            action_label = "hold_lose"

        return HybridTradeResult(
            window_idx=wd.window_idx,
            action=action_label,
            entry_price=effective_entry,
            exit_price=settlement,
            pnl_gross=pnl_gross,
            pnl_net=pnl_net,
            fee_entry=fee_entry,
            fee_exit=fee_exit,
            correct=pnl_net > 0,
            btc_return_at_n=btc_return_at_n,
            continuation_prob=continuation_prob,
            early_direction="Up" if early_up else ("Down" if early_down else "Flat"),
            resolution=wd.resolution,
        )


# ---------------------------------------------------------------------------
# Risk Metrics
# ---------------------------------------------------------------------------

@dataclass
class RiskMetrics:
    variant: str = ""
    entry_minute: int = 0
    cont_threshold: float = 0.50
    total_trades: int = 0
    # Actions breakdown
    n_hold_win: int = 0
    n_hold_lose: int = 0
    n_sell_cut_loss: int = 0
    n_sell_take_profit: int = 0
    n_sell_uncertain: int = 0
    # Win/loss
    win_rate: float = 0.0
    wins: int = 0
    losses: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    risk_reward_ratio: float = 0.0  # avg_win / avg_loss
    # PnL
    total_pnl: float = 0.0
    total_fees: float = 0.0
    ev_per_trade: float = 0.0
    # Risk
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    kelly_fraction: float = 0.0
    # Comparison
    baseline_pnl: float = 0.0  # Always buy YES, hold to settlement
    edge_vs_baseline: float = 0.0


def compute_risk_metrics(
    trades: list[HybridTradeResult],
    variant: str,
    entry_minute: int,
    cont_threshold: float,
    baseline_trades: list[HybridTradeResult] | None = None,
) -> RiskMetrics:
    m = RiskMetrics(variant=variant, entry_minute=entry_minute, cont_threshold=cont_threshold)

    if not trades:
        return m

    m.total_trades = len(trades)
    m.n_hold_win = sum(1 for t in trades if t.action == "hold_win")
    m.n_hold_lose = sum(1 for t in trades if t.action == "hold_lose")
    m.n_sell_cut_loss = sum(1 for t in trades if t.action == "sell_early_cut_loss")
    m.n_sell_take_profit = sum(1 for t in trades if t.action == "sell_early_take_profit")
    m.n_sell_uncertain = sum(1 for t in trades if t.action == "sell_early_uncertain")

    pnls = [t.pnl_net for t in trades]
    m.total_pnl = sum(pnls)
    m.total_fees = sum(t.fee_entry + t.fee_exit for t in trades)
    m.ev_per_trade = m.total_pnl / m.total_trades

    # Win/loss
    wins_list = [p for p in pnls if p > 0]
    losses_list = [p for p in pnls if p <= 0]
    m.wins = len(wins_list)
    m.losses = len(losses_list)
    m.win_rate = m.wins / m.total_trades if m.total_trades > 0 else 0

    m.avg_win = sum(wins_list) / len(wins_list) if wins_list else 0
    m.avg_loss = abs(sum(losses_list) / len(losses_list)) if losses_list else 0.001
    m.risk_reward_ratio = m.avg_win / m.avg_loss if m.avg_loss > 0 else 999

    # Sharpe (annualized — 35040 15-min windows per year)
    returns = [p / POSITION_SIZE for p in pnls]
    mean_r = sum(returns) / len(returns)
    var_r = sum((r - mean_r) ** 2 for r in returns) / len(returns)
    std_r = math.sqrt(var_r) if var_r > 0 else 0.001
    m.sharpe = (mean_r / std_r) * math.sqrt(35040)

    # Max drawdown
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        cum += p
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd
    m.max_drawdown = max_dd
    m.max_drawdown_pct = (max_dd / peak * 100) if peak > 0 else 0

    # Kelly criterion: f* = (p * b - q) / b  where p=win_rate, q=1-p, b=avg_win/avg_loss
    if m.avg_loss > 0 and m.risk_reward_ratio > 0:
        p = m.win_rate
        q = 1.0 - p
        b = m.risk_reward_ratio
        m.kelly_fraction = max(0, (p * b - q) / b)

    # Baseline comparison
    if baseline_trades:
        m.baseline_pnl = sum(t.pnl_net for t in baseline_trades)
        m.edge_vs_baseline = m.total_pnl - m.baseline_pnl

    return m


def print_metrics(m: RiskMetrics) -> None:
    print(f"\n  {'='*70}")
    print(f"  VARIANT {m.variant} | Entry Min {m.entry_minute} | Threshold {m.cont_threshold:.2f}")
    print(f"  {'='*70}")
    print(f"  Total trades:     {m.total_trades:,}")
    print(f"")
    print(f"  --- Actions Breakdown ---")
    print(f"  Hold → Win:       {m.n_hold_win:>6,}  ({m.n_hold_win/m.total_trades*100:>5.1f}%)")
    print(f"  Hold → Lose:      {m.n_hold_lose:>6,}  ({m.n_hold_lose/m.total_trades*100:>5.1f}%)")
    print(f"  Sell: Cut Loss:   {m.n_sell_cut_loss:>6,}  ({m.n_sell_cut_loss/m.total_trades*100:>5.1f}%)")
    print(f"  Sell: Take Profit:{m.n_sell_take_profit:>6,}  ({m.n_sell_take_profit/m.total_trades*100:>5.1f}%)")
    print(f"  Sell: Uncertain:  {m.n_sell_uncertain:>6,}  ({m.n_sell_uncertain/m.total_trades*100:>5.1f}%)")
    print(f"")
    print(f"  --- Win / Loss ---")
    print(f"  Win rate:         {m.win_rate*100:.1f}%")
    print(f"  Avg win:          ${m.avg_win:>8.2f}")
    print(f"  Avg loss:         ${m.avg_loss:>8.2f}")
    print(f"  Risk/Reward:      {m.risk_reward_ratio:.2f}x")
    print(f"")
    print(f"  --- P&L ---")
    print(f"  Total PnL:        ${m.total_pnl:>12,.2f}")
    print(f"  Total fees:       ${m.total_fees:>12,.2f}")
    print(f"  EV per trade:     ${m.ev_per_trade:>8.2f}")
    print(f"  Baseline PnL:     ${m.baseline_pnl:>12,.2f}  (always hold)")
    print(f"  Edge vs baseline: ${m.edge_vs_baseline:>12,.2f}")
    print(f"")
    print(f"  --- Risk ---")
    print(f"  Sharpe (ann):     {m.sharpe:.2f}")
    print(f"  Max drawdown:     ${m.max_drawdown:>10,.2f} ({m.max_drawdown_pct:.1f}%)")
    print(f"  Kelly fraction:   {m.kelly_fraction*100:.1f}%")
    _flush()


# ---------------------------------------------------------------------------
# Walk-Forward Validation
# ---------------------------------------------------------------------------

def walk_forward_hybrid(
    window_data: list[WindowData],
    windows: list[list[FastCandle]],
    all_feature_dicts: list[dict[str, float]],
    feature_names: list[str],
    continuation_targets: list[int],
    valid_mask: list[bool],
    btc_returns_at_n: list[float],
    entry_minute: int,
    cont_threshold: float,
    variant: str,
) -> tuple[list[dict[str, Any]], RiskMetrics]:
    """Walk-forward validation of hybrid strategy with realistic pricing."""

    by_month: dict[str, list[int]] = defaultdict(list)
    for i, wd in enumerate(window_data):
        if valid_mask[i]:
            by_month[wd.timestamp.strftime("%Y-%m")].append(i)
    months = sorted(by_month.keys())

    if len(months) < WF_TRAIN_MONTHS + WF_TEST_MONTHS:
        return [], RiskMetrics()

    all_trades: list[HybridTradeResult] = []
    all_baseline: list[HybridTradeResult] = []
    period_results: list[dict[str, Any]] = []

    for mi in range(WF_TRAIN_MONTHS, len(months), WF_TEST_MONTHS):
        if mi >= len(months):
            break

        test_month = months[mi]
        train_months = months[max(0, mi - WF_TRAIN_MONTHS):mi]

        train_indices: list[int] = []
        for m in train_months:
            train_indices.extend(by_month[m])
        test_indices = by_month[test_month]

        if len(train_indices) < 100 or len(test_indices) < 20:
            continue

        # Train continuation filter
        X_train = np.array([[all_feature_dicts[i].get(f, 0.0) for f in feature_names] for i in train_indices])
        y_train = np.array([continuation_targets[i] for i in train_indices])

        model = HistGradientBoostingClassifier(
            max_iter=300, max_depth=4, learning_rate=0.05,
            min_samples_leaf=80, l2_regularization=1.0,
            max_bins=128, random_state=42,
        )
        model.fit(X_train, y_train)

        # Train pricing model on same training data
        train_returns = np.array([btc_returns_at_n[i] for i in train_indices])
        train_final_up = np.array([1 if window_data[i].resolution == "Up" else 0 for i in train_indices])
        pricing = PolymarketPricingModel()
        pricing.fit(train_returns, train_final_up)

        # Test
        X_test = np.array([[all_feature_dicts[i].get(f, 0.0) for f in feature_names] for i in test_indices])
        test_probas = model.predict_proba(X_test)[:, 1]

        period_trades: list[HybridTradeResult] = []
        period_baseline: list[HybridTradeResult] = []

        for j, idx in enumerate(test_indices):
            wd = window_data[idx]
            ret_n = btc_returns_at_n[idx]
            cont_prob = float(test_probas[j])

            # Hybrid trade
            trade = simulate_hybrid_trade(
                wd, windows, entry_minute, cont_prob, cont_threshold,
                ret_n, pricing, variant,
            )
            period_trades.append(trade)

            # Baseline: always hold YES to settlement
            baseline = simulate_hybrid_trade(
                wd, windows, entry_minute, 0.0, 999.0,  # threshold=999 → always hold
                ret_n, pricing, "BASELINE",
            )
            period_baseline.append(baseline)

        all_trades.extend(period_trades)
        all_baseline.extend(period_baseline)

        # Period metrics
        period_pnl = sum(t.pnl_net for t in period_trades)
        baseline_pnl = sum(t.pnl_net for t in period_baseline)
        period_wins = sum(1 for t in period_trades if t.pnl_net > 0)
        period_win_rate = period_wins / len(period_trades) if period_trades else 0

        # Avg win/loss for period
        p_wins = [t.pnl_net for t in period_trades if t.pnl_net > 0]
        p_losses = [t.pnl_net for t in period_trades if t.pnl_net <= 0]
        avg_w = sum(p_wins) / len(p_wins) if p_wins else 0
        avg_l = abs(sum(p_losses) / len(p_losses)) if p_losses else 0.001
        rr = avg_w / avg_l if avg_l > 0 else 999

        n_sells = sum(1 for t in period_trades if "sell_early" in t.action)

        period_results.append({
            "period": test_month,
            "trades": len(period_trades),
            "win_rate": round(period_win_rate * 100, 1),
            "pnl": round(period_pnl, 2),
            "baseline_pnl": round(baseline_pnl, 2),
            "edge": round(period_pnl - baseline_pnl, 2),
            "avg_win": round(avg_w, 2),
            "avg_loss": round(avg_l, 2),
            "rr": round(rr, 2),
            "early_exits": n_sells,
            "profitable": period_pnl > 0,
        })

    overall = compute_risk_metrics(all_trades, variant, entry_minute, cont_threshold, all_baseline)
    return period_results, overall


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 100)
    print("  BWO v6 — HYBRID STRATEGY BACKTEST")
    print("  Entry: ALWAYS buy YES at $0.50 BEFORE window (cheap BWO entry)")
    print("  Management: Continuation filter decides hold/sell at minute N")
    print("  Pricing: Realistic Polymarket pricing model for early exits")
    print("=" * 100)
    _flush()

    # ------------------------------------------------------------------
    # Load data (same as v5)
    # ------------------------------------------------------------------
    print("\n  Loading data...")
    _flush()

    if not BTC_SPOT_CSV.exists():
        print(f"  ERROR: {BTC_SPOT_CSV} not found.")
        sys.exit(1)

    all_candles = load_csv_fast(BTC_SPOT_CSV)
    all_candles.sort(key=lambda c: c.timestamp)
    seen: set[datetime] = set()
    unique: list[FastCandle] = []
    for c in all_candles:
        if c.timestamp not in seen:
            seen.add(c.timestamp)
            unique.append(c)
    all_candles = unique
    print(f"  Spot BTC: {len(all_candles):,} candles")

    # Futures
    futures_data: list[dict] = []
    futures_lookup: dict[datetime, int] = {}
    if BTC_FUTURES_CSV.exists():
        import csv
        with open(BTC_FUTURES_CSV) as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = _parse_ts(row["timestamp"])
                if ts is None:
                    continue
                futures_lookup[ts] = len(futures_data)
                futures_data.append({
                    "timestamp": ts,
                    "volume": float(row.get("volume", 0)),
                    "taker_buy_volume": float(row.get("taker_buy_volume", 0)),
                    "num_trades": int(float(row.get("num_trades", 0))),
                })
        print(f"  BTC futures: {len(futures_data):,} candles")

    # ETH futures
    eth_data: list[dict] = []
    eth_lookup: dict[datetime, int] = {}
    if ETH_FUTURES_CSV.exists():
        import csv
        with open(ETH_FUTURES_CSV) as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = _parse_ts(row["timestamp"])
                if ts is None:
                    continue
                eth_lookup[ts] = len(eth_data)
                eth_data.append({
                    "timestamp": ts,
                    "close": float(row.get("close", 0)),
                    "volume": float(row.get("volume", 0)),
                    "taker_buy_volume": float(row.get("taker_buy_volume", 0)),
                })
        print(f"  ETH futures: {len(eth_data):,} candles")

    # DVOL
    dvol_data: list[dict] = []
    dvol_timestamps: list[datetime] = []
    if DVOL_CSV.exists():
        import csv
        with open(DVOL_CSV) as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = _parse_ts(row["timestamp"])
                if ts is None:
                    continue
                dvol_data.append({"timestamp": ts, "close": float(row.get("close", 0))})
                dvol_timestamps.append(ts)
        print(f"  DVOL: {len(dvol_data):,} candles")

    # Deribit perpetual
    deribit_data: list[dict] = []
    deribit_lookup: dict[datetime, int] = {}
    if DERIBIT_PERP_CSV.exists():
        import csv
        with open(DERIBIT_PERP_CSV) as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = _parse_ts(row["timestamp"])
                if ts is None:
                    continue
                deribit_lookup[ts] = len(deribit_data)
                deribit_data.append({
                    "timestamp": ts,
                    "close": float(row.get("close", 0)),
                    "volume": float(row.get("volume", 0)),
                })
        print(f"  Deribit perp: {len(deribit_data):,} candles")
    _flush()

    # ------------------------------------------------------------------
    # Build windows
    # ------------------------------------------------------------------
    print("\n  Building 15m windows...", end=" ")
    _flush()
    windows = group_into_15m_windows(all_candles)
    candle_by_ts = {c.timestamp: i for i, c in enumerate(all_candles)}

    window_data: list[WindowData] = []
    for wi, w in enumerate(windows):
        ts = w[0].timestamp
        idx = candle_by_ts.get(ts)
        if idx is None:
            continue
        resolution = "Up" if w[-1].close > w[0].open else "Down"
        window_data.append(WindowData(
            window_idx=wi, candle_idx=idx, resolution=resolution, timestamp=ts,
        ))
    print(f"{len(window_data):,} windows")

    up_count = sum(1 for wd in window_data if wd.resolution == "Up")
    print(f"  Base rate (direction): Up {up_count/len(window_data)*100:.1f}%")
    _flush()

    # ------------------------------------------------------------------
    # Baseline: always buy YES at $0.50, hold to settlement
    # ------------------------------------------------------------------
    print(f"\n  --- BASELINE: Always buy YES at $0.505, hold to settlement ---")
    effective_entry = ENTRY_PRICE + ENTRY_SLIPPAGE
    n_tokens = POSITION_SIZE / effective_entry
    fee_per_trade = polymarket_fee(POSITION_SIZE, effective_entry)

    baseline_wins = 0
    baseline_pnl = 0.0
    for wd in window_data:
        settlement = 1.0 if wd.resolution == "Up" else 0.0
        fee_exit = polymarket_fee(POSITION_SIZE, settlement) if settlement > 0 else 0.0
        pnl = n_tokens * (settlement - effective_entry) - fee_per_trade - fee_exit
        baseline_pnl += pnl
        if pnl > 0:
            baseline_wins += 1

    baseline_win_rate = baseline_wins / len(window_data)
    baseline_ev = baseline_pnl / len(window_data)
    print(f"  Trades:   {len(window_data):,}")
    print(f"  Win rate: {baseline_win_rate*100:.1f}%")
    print(f"  Total PnL: ${baseline_pnl:,.2f}")
    print(f"  EV/trade:  ${baseline_ev:.2f}")
    print(f"  (Slightly negative due to fees + slippage on 50/50 outcome)")
    _flush()

    # ------------------------------------------------------------------
    # Per entry-minute + variant analysis
    # ------------------------------------------------------------------
    report: dict[str, Any] = {
        "meta": {
            "total_windows": len(window_data),
            "base_rate_up_pct": round(up_count / len(window_data) * 100, 2),
            "entry_price": effective_entry,
            "exit_spread": EXIT_SPREAD,
            "exit_slippage": EXIT_SLIPPAGE,
            "position_size": POSITION_SIZE,
            "baseline_pnl": round(baseline_pnl, 2),
            "baseline_ev_per_trade": round(baseline_ev, 2),
        },
        "results": {},
    }

    VARIANTS = ["A", "B", "C"]

    for entry_minute in ENTRY_MINUTES:
        print(f"\n{'='*100}")
        print(f"  ENTRY MINUTE {entry_minute}")
        print(f"{'='*100}")
        _flush()

        # ------------------------------------------------------------------
        # Compute features (same as v5)
        # ------------------------------------------------------------------
        print(f"  Computing features (entry_minute={entry_minute})...")
        _flush()
        t0 = time.time()

        all_feature_dicts: list[dict[str, float]] = []
        continuation_targets: list[int] = []
        valid_mask: list[bool] = []
        btc_returns_at_n: list[float] = []

        for i, wd in enumerate(window_data):
            ci = wd.candle_idx
            w = windows[wd.window_idx]

            # BTC return at minute N (for pricing model)
            if entry_minute <= len(w) and w[0].open > 0:
                ret_n = (w[entry_minute - 1].close - w[0].open) / w[0].open
            else:
                ret_n = 0.0
            btc_returns_at_n.append(ret_n)

            # Original pre-window features
            original = compute_all_features(wd, windows, all_candles, entry_minute=entry_minute)

            # Signal quality features
            signal = compute_signal_quality_features(w, entry_minute, all_candles, ci)

            # Taker features
            taker = compute_taker_features_simple(ci, futures_data, futures_lookup, all_candles)

            # DVOL
            dvol_feats: dict[str, float] = {}
            if dvol_timestamps:
                di = bisect_right(dvol_timestamps, wd.timestamp) - 1
                if di >= 5:
                    dvol_now = dvol_data[di]["close"]
                    dvol_feats["dvol_level"] = dvol_now
                    dvol_feats["dvol_change_5"] = (dvol_now - dvol_data[di - 5]["close"]) / dvol_data[di - 5]["close"] if dvol_data[di - 5]["close"] > 0 else 0.0
                    start_z = max(0, di - 24)
                    recent = [dvol_data[j]["close"] for j in range(start_z, di + 1)]
                    if len(recent) >= 5:
                        m_val = sum(recent) / len(recent)
                        v_val = sum((x - m_val) ** 2 for x in recent) / len(recent)
                        s_val = math.sqrt(v_val) if v_val > 0 else 0.001
                        dvol_feats["dvol_z"] = (dvol_now - m_val) / s_val
                    else:
                        dvol_feats["dvol_z"] = 0.0
                else:
                    dvol_feats = {"dvol_level": 0.0, "dvol_change_5": 0.0, "dvol_z": 0.0}
            else:
                dvol_feats = {"dvol_level": 0.0, "dvol_change_5": 0.0, "dvol_z": 0.0}

            # ETH
            eth_feats: dict[str, float] = {}
            ei = eth_lookup.get(wd.timestamp)
            bi_f = futures_lookup.get(wd.timestamp)
            if ei is not None and ei >= 15 and bi_f is not None and bi_f >= 15:
                eth_close = eth_data[ei - 1]["close"]
                eth_open = eth_data[ei - 15]["close"]
                eth_ret = (eth_close - eth_open) / eth_open if eth_open > 0 else 0.0
                eth_feats["eth_momentum_15"] = eth_ret
                eth_tv = sum(eth_data[j].get("volume", 0) for j in range(ei - 15, ei))
                eth_bv = sum(eth_data[j].get("taker_buy_volume", 0) for j in range(ei - 15, ei))
                eth_imb = (2 * eth_bv - eth_tv) / eth_tv if eth_tv > 0 else 0.0
                btc_imb = taker.get("taker_imbalance_15", 0.0)
                btc_sign = 1.0 if btc_imb > 0 else (-1.0 if btc_imb < 0 else 0.0)
                eth_sign = 1.0 if eth_imb > 0 else (-1.0 if eth_imb < 0 else 0.0)
                eth_feats["eth_taker_alignment"] = 1.0 if btc_sign == eth_sign and btc_sign != 0 else 0.0
            else:
                eth_feats = {"eth_momentum_15": 0.0, "eth_taker_alignment": 0.0}

            # Deribit cross-exchange basis
            cross_feats: dict[str, float] = {}
            target_hour = wd.timestamp.replace(minute=0, second=0, microsecond=0)
            dbi = deribit_lookup.get(target_hour)
            if dbi is not None and ci > 0:
                deribit_close = deribit_data[dbi]["close"]
                spot_close = all_candles[ci - 1].close
                if spot_close > 0:
                    cross_feats["deribit_basis_bps"] = (deribit_close - spot_close) / spot_close * 10000
                else:
                    cross_feats["deribit_basis_bps"] = 0.0
            else:
                cross_feats["deribit_basis_bps"] = 0.0

            combined = {**original, **signal, **taker, **dvol_feats, **eth_feats, **cross_feats}
            all_feature_dicts.append(combined)

            # Continuation target
            early_dir = original.get("early_direction", 0.0)
            if early_dir == 0.0:
                continuation_targets.append(0)
                valid_mask.append(False)
            else:
                continued = (
                    (wd.resolution == "Up" and early_dir > 0) or
                    (wd.resolution == "Down" and early_dir < 0)
                )
                continuation_targets.append(1 if continued else 0)
                valid_mask.append(True)

            if (i + 1) % 10000 == 0:
                elapsed = time.time() - t0
                print(f"    {i+1:>8,} / {len(window_data):,} ({(i+1)/len(window_data)*100:.1f}%) - {elapsed:.0f}s")
                _flush()

        elapsed = time.time() - t0
        print(f"  Done: {elapsed:.1f}s")

        n_valid = sum(valid_mask)
        n_cont = sum(1 for i in range(len(window_data)) if valid_mask[i] and continuation_targets[i] == 1)
        cont_rate = n_cont / n_valid if n_valid > 0 else 0
        print(f"  Valid windows: {n_valid:,} | Continuation base rate: {cont_rate*100:.1f}%")
        _flush()

        # ------------------------------------------------------------------
        # Pricing model sanity check (on full data)
        # ------------------------------------------------------------------
        valid_returns = np.array([btc_returns_at_n[i] for i in range(len(window_data)) if valid_mask[i]])
        valid_final_up = np.array([1 if window_data[i].resolution == "Up" else 0
                                   for i in range(len(window_data)) if valid_mask[i]])
        full_pricing = PolymarketPricingModel()
        full_pricing.fit(valid_returns, valid_final_up)
        pstats = full_pricing.stats(valid_returns)
        print(f"\n  Pricing model (logistic): coef={pstats.get('coef', 0):.1f}, "
              f"P(UP) range=[{pstats.get('min_prob', 0):.3f}, {pstats.get('max_prob', 0):.3f}]")

        # Show example prices at different BTC returns
        print(f"  Example YES prices at minute {entry_minute}:")
        for ret_pct in [-0.5, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5]:
            ret = ret_pct / 100
            fv = full_pricing.fair_value(ret)
            sp = full_pricing.sell_price(ret)
            print(f"    BTC {ret_pct:+.2f}% → YES fair={fv:.3f}, sell@{sp:.3f}")
        _flush()

        # ------------------------------------------------------------------
        # Feature selection (use same top MI features as v5)
        # ------------------------------------------------------------------
        all_feature_names = sorted(set().union(*(d.keys() for d in all_feature_dicts[:100])))
        valid_indices = [i for i in range(len(window_data)) if valid_mask[i]]
        target_arr = np.array([continuation_targets[i] for i in valid_indices])

        mi_results: dict[str, float] = {}
        for fname in all_feature_names:
            vals = np.array([all_feature_dicts[i].get(fname, 0.0) for i in valid_indices])
            if vals.std() == 0:
                mi = 0.0
            else:
                mi = compute_mi_bits(vals, target_arr)
            mi_results[fname] = mi

        sorted_mi = sorted(mi_results.items(), key=lambda x: x[1], reverse=True)
        top_features = [f for f, mi in sorted_mi if mi >= 0.0005][:40]
        if len(top_features) < 5:
            top_features = [f for f, _ in sorted_mi[:20]]
        print(f"\n  Using {len(top_features)} features for continuation filter")
        _flush()

        # ------------------------------------------------------------------
        # Walk-forward for each variant + threshold sweep
        # ------------------------------------------------------------------
        entry_report: dict[str, Any] = {
            "entry_minute": entry_minute,
            "continuation_base_rate": round(cont_rate, 4),
            "n_valid": n_valid,
            "pricing_model": pstats,
            "variants": {},
        }

        for variant in VARIANTS:
            print(f"\n  {'─'*70}")
            print(f"  VARIANT {variant}: ", end="")
            if variant == "A":
                print("Conservative (only cut losses on adverse continuation)")
            elif variant == "B":
                print("Balanced (cut losses + take profits)")
            elif variant == "C":
                print("Aggressive (only hold on favorable continuation)")
            print(f"  {'─'*70}")
            _flush()

            # Quick threshold sweep on train/test split
            print(f"\n  Threshold sweep (80/20 split)...")
            print(f"  {'Thresh':>8} {'WinRate':>8} {'AvgWin':>8} {'AvgLoss':>8} {'R:R':>6} {'EV/trade':>10} {'PnL':>12}")
            print(f"  {'─'*66}")
            _flush()

            split_idx = int(len(valid_indices) * 0.80)
            train_idx = valid_indices[:split_idx]
            test_idx = valid_indices[split_idx:]

            X_tr = np.array([[all_feature_dicts[i].get(f, 0.0) for f in top_features] for i in train_idx])
            y_tr = np.array([continuation_targets[i] for i in train_idx])
            X_te = np.array([[all_feature_dicts[i].get(f, 0.0) for f in top_features] for i in test_idx])

            mdl = HistGradientBoostingClassifier(
                max_iter=300, max_depth=4, learning_rate=0.05,
                min_samples_leaf=80, l2_regularization=1.0,
                max_bins=128, random_state=42,
            )
            mdl.fit(X_tr, y_tr)

            train_rets = np.array([btc_returns_at_n[i] for i in train_idx])
            train_ups = np.array([1 if window_data[i].resolution == "Up" else 0 for i in train_idx])
            pricing = PolymarketPricingModel()
            pricing.fit(train_rets, train_ups)

            test_probas = mdl.predict_proba(X_te)[:, 1]

            best_sweep: dict[str, Any] = {"ev": -999, "threshold": 0.50}

            for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
                trades: list[HybridTradeResult] = []
                for j_idx, idx in enumerate(test_idx):
                    wd = window_data[idx]
                    trade = simulate_hybrid_trade(
                        wd, windows, entry_minute, float(test_probas[j_idx]),
                        thresh, btc_returns_at_n[idx], pricing, variant,
                    )
                    trades.append(trade)

                if not trades:
                    continue

                pnls = [t.pnl_net for t in trades]
                wins_l = [p for p in pnls if p > 0]
                losses_l = [p for p in pnls if p <= 0]
                wr = len(wins_l) / len(trades)
                aw = sum(wins_l) / len(wins_l) if wins_l else 0
                al = abs(sum(losses_l) / len(losses_l)) if losses_l else 0.001
                rr = aw / al if al > 0 else 999
                ev = sum(pnls) / len(trades)
                total = sum(pnls)

                marker = " ***" if ev > best_sweep["ev"] else ""
                print(f"  {thresh:>8.2f} {wr*100:>7.1f}% ${aw:>7.2f} ${al:>7.2f} {rr:>5.2f} ${ev:>9.2f} ${total:>11,.2f}{marker}")
                _flush()

                if ev > best_sweep["ev"]:
                    best_sweep = {"ev": ev, "threshold": thresh, "win_rate": wr, "rr": rr}

            best_thresh = best_sweep["threshold"]
            print(f"\n  Best threshold: {best_thresh:.2f} (EV ${best_sweep['ev']:.2f}/trade)")

            # ------------------------------------------------------------------
            # Walk-forward with best threshold
            # ------------------------------------------------------------------
            print(f"\n  --- Walk-Forward (threshold={best_thresh:.2f}) ---")
            print(f"  {'Period':>10} {'WinRate':>8} {'AvgWin':>8} {'AvgLoss':>8} {'R:R':>6} "
                  f"{'PnL':>10} {'Baseline':>10} {'Edge':>10} {'Exits':>6} {'Prof':>5}")
            print(f"  {'─'*92}")
            _flush()

            wf_results, wf_metrics = walk_forward_hybrid(
                window_data, windows, all_feature_dicts, top_features,
                continuation_targets, valid_mask, btc_returns_at_n,
                entry_minute, best_thresh, variant,
            )

            for pr in wf_results:
                prof = "Y" if pr["profitable"] else "N"
                print(f"  {pr['period']:>10} {pr['win_rate']:>7.1f}% ${pr['avg_win']:>7.2f} "
                      f"${pr['avg_loss']:>7.2f} {pr['rr']:>5.2f} ${pr['pnl']:>9,.0f} "
                      f"${pr['baseline_pnl']:>9,.0f} ${pr['edge']:>9,.0f} {pr['early_exits']:>5} {prof:>5}")

            if wf_results:
                n_prof = sum(1 for pr in wf_results if pr["profitable"])
                print(f"\n  Walk-forward: {n_prof}/{len(wf_results)} months profitable")
            _flush()

            # Print comprehensive metrics
            print_metrics(wf_metrics)

            # Store in report
            entry_report["variants"][variant] = {
                "best_threshold": best_thresh,
                "sweep": best_sweep,
                "walk_forward_periods": wf_results,
                "metrics": {
                    "total_trades": wf_metrics.total_trades,
                    "win_rate": round(wf_metrics.win_rate, 4),
                    "avg_win": round(wf_metrics.avg_win, 2),
                    "avg_loss": round(wf_metrics.avg_loss, 2),
                    "risk_reward_ratio": round(wf_metrics.risk_reward_ratio, 2),
                    "total_pnl": round(wf_metrics.total_pnl, 2),
                    "ev_per_trade": round(wf_metrics.ev_per_trade, 2),
                    "sharpe": round(wf_metrics.sharpe, 2),
                    "max_drawdown": round(wf_metrics.max_drawdown, 2),
                    "kelly_fraction": round(wf_metrics.kelly_fraction, 4),
                    "baseline_pnl": round(wf_metrics.baseline_pnl, 2),
                    "edge_vs_baseline": round(wf_metrics.edge_vs_baseline, 2),
                    "n_hold_win": wf_metrics.n_hold_win,
                    "n_hold_lose": wf_metrics.n_hold_lose,
                    "n_sell_cut_loss": wf_metrics.n_sell_cut_loss,
                    "n_sell_take_profit": wf_metrics.n_sell_take_profit,
                },
            }

        report["results"][str(entry_minute)] = entry_report

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*100}")
    print(f"  OVERALL SUMMARY")
    print(f"{'='*100}")
    print(f"\n  {'Variant':<10} {'Min':<5} {'WinRate':>8} {'AvgWin':>8} {'AvgLoss':>8} {'R:R':>6} "
          f"{'EV/trade':>9} {'Sharpe':>7} {'TotalPnL':>12} {'Edge':>12} {'Kelly':>6}")
    print(f"  {'─'*100}")

    for em in ENTRY_MINUTES:
        key = str(em)
        if key not in report["results"]:
            continue
        for var in VARIANTS:
            if var not in report["results"][key]["variants"]:
                continue
            m = report["results"][key]["variants"][var]["metrics"]
            print(f"  {var:<10} {em:<5} {m['win_rate']*100:>7.1f}% ${m['avg_win']:>7.2f} "
                  f"${m['avg_loss']:>7.2f} {m['risk_reward_ratio']:>5.2f} "
                  f"${m['ev_per_trade']:>8.2f} {m['sharpe']:>6.2f} "
                  f"${m['total_pnl']:>11,.2f} ${m['edge_vs_baseline']:>11,.2f} "
                  f"{m['kelly_fraction']*100:>5.1f}%")

    # Save report
    with open(REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved: {REPORT_JSON}")
    print(f"{'='*100}")
    _flush()


if __name__ == "__main__":
    main()
