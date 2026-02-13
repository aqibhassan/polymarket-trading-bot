"""Before Window Open (BWO) Backtest — pre-window BTC signals for Polymarket 15m markets.

Enters BEFORE minute 0 at ~$0.50 entry price. At $0.50 on a binary $0/$1 payout,
break-even is ~52% after fees. Tests whether pre-window BTC signals can predict
the next 15m candle direction above that threshold.

Uses 7 pre-window feature functions (no lookahead) and 6 strategy variants.
Includes walk-forward validation, bootstrap CI, and permutation tests.

Performance: processes 70K+ windows in <60s via index-based lookups (no list copies).
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fast_loader import FastCandle, group_into_15m_windows, load_csv_fast

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BTC_CSV = PROJECT_ROOT / "data" / "btc_1m_2y.csv"
REPORT_JSON = PROJECT_ROOT / "data" / "bwo_backtest_report.json"
TRADES_CSV = PROJECT_ROOT / "data" / "bwo_trades.csv"

POSITION_SIZE = 100.0
ENTRY_PRICE = 0.50  # pre-window fair value
FEE_CONSTANT = 0.25
SLIPPAGE_BPS = 5

# Walk-forward: 3-month train / 1-month test rolling
WF_TRAIN_MONTHS = 3
WF_TEST_MONTHS = 1
BOOTSTRAP_ITERATIONS = 2000
PERMUTATION_ITERATIONS = 2000
MIN_TRADES = 500

# Trading session UTC hour ranges
SESSIONS = {
    "asia": range(0, 8),
    "europe": range(8, 14),
    "us": range(14, 22),
    "late": range(22, 24),
}


def _flush() -> None:
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Fee model (reused from validate_polymarket_real.py)
# ---------------------------------------------------------------------------

def polymarket_fee(position_size: float, entry_price: float) -> float:
    if entry_price <= 0 or entry_price >= 1:
        return 0.0
    return position_size * FEE_CONSTANT * (entry_price ** 2) * ((1.0 - entry_price) ** 2)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class WindowData:
    """A single 15m window with pre-window context (index-based, no copies)."""
    window_idx: int  # index into windows list
    candle_idx: int  # index of window[0] in all_candles
    resolution: str  # "Up" or "Down"
    timestamp: datetime
    entry_minute: int = 0  # which minute within window to enter (0=pre-window)


@dataclass
class TradeResult:
    timestamp: datetime
    strategy: str
    direction: str  # "Up" or "Down"
    resolution: str
    correct: bool
    entry_price: float
    settlement: float
    pnl_gross: float
    pnl_net: float
    fee: float
    features: dict[str, float] = field(default_factory=dict)


@dataclass
class StrategyMetrics:
    name: str
    total_trades: int = 0
    correct: int = 0
    accuracy: float = 0.0
    ev_per_trade: float = 0.0
    net_pnl: float = 0.0
    gross_pnl: float = 0.0
    total_fees: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    monthly_pnls: dict[str, float] = field(default_factory=dict)
    profitable_months_pct: float = 0.0


# ---------------------------------------------------------------------------
# Feature functions — index-based, ONLY use data BEFORE window start
# ---------------------------------------------------------------------------

def compute_all_features(
    wd: WindowData,
    windows: list[list[FastCandle]],
    all_candles: list[FastCandle],
    entry_minute: int = 0,
) -> dict[str, float]:
    """Compute all features for a window using index-based access.

    Pre-window features (7 original) + early-window features (6) + TA indicators (6).
    entry_minute: which minute within the window to enter (0=pre-window, 2-5=early-window).
    Early-window features use only candles 0 through entry_minute (no lookahead).
    """
    features: dict[str, float] = {}
    ci = wd.candle_idx  # index of first candle in this window
    wi = wd.window_idx

    # --- Feature 1: Prior window momentum ---
    if wi > 0:
        pw = windows[wi - 1]
        op = pw[0].open
        cp = pw[-1].close
        if op != 0:
            ret = (cp - op) / op
            features["prior_dir"] = 1.0 if ret > 0 else (-1.0 if ret < 0 else 0.0)
            features["prior_mag"] = abs(ret)
        else:
            features["prior_dir"] = 0.0
            features["prior_mag"] = 0.0
    else:
        features["prior_dir"] = 0.0
        features["prior_mag"] = 0.0

    # Determine history range (up to 1440 candles before window start)
    hist_start = max(0, ci - 1440)
    hist_end = ci  # exclusive — no lookahead

    # --- Feature 2: Multi-timeframe trend (15m / 1h / 4h) ---
    hist_len = hist_end - hist_start
    if hist_len >= 60:
        def _dir(start: int, end: int) -> float:
            o = all_candles[start].open
            c = all_candles[end - 1].close
            if o == 0:
                return 0.0
            r = (c - o) / o
            return 1.0 if r > 0 else (-1.0 if r < 0 else 0.0)

        d15 = _dir(hist_end - 15, hist_end)
        d60 = _dir(hist_end - 60, hist_end)
        d240 = _dir(hist_end - min(240, hist_len), hist_end)
        features["mtf_score"] = d15 + d60 + d240
        features["mtf_15m"] = d15
        features["mtf_1h"] = d60
        features["mtf_4h"] = d240
    else:
        features["mtf_score"] = 0.0
        features["mtf_15m"] = 0.0
        features["mtf_1h"] = 0.0
        features["mtf_4h"] = 0.0

    # --- Feature 3: Short-term momentum (last 5 candles) ---
    if hist_len >= 5:
        s5 = hist_end - 5
        o5 = all_candles[s5].open
        c5 = all_candles[hist_end - 1].close
        if o5 != 0:
            ret5 = (c5 - o5) / o5
            features["stm_dir"] = 1.0 if ret5 > 0 else (-1.0 if ret5 < 0 else 0.0)
        else:
            features["stm_dir"] = 0.0
        green = sum(1 for j in range(s5, hist_end) if all_candles[j].close > all_candles[j].open)
        features["stm_strength"] = green / 5.0
    else:
        features["stm_dir"] = 0.0
        features["stm_strength"] = 0.0

    # --- Feature 4: Volatility regime (simplified: recent 15 vs full history) ---
    if hist_len >= 60:
        # Compute returns
        recent_sq = 0.0
        for j in range(hist_end - 14, hist_end):
            prev_c = all_candles[j - 1].close
            if prev_c > 0:
                r = (all_candles[j].close - prev_c) / prev_c
                recent_sq += r * r
        recent_vol = math.sqrt(recent_sq / 14) if recent_sq > 0 else 0.0

        full_sq = 0.0
        full_n = 0
        for j in range(hist_start + 1, hist_end):
            prev_c = all_candles[j - 1].close
            if prev_c > 0:
                r = (all_candles[j].close - prev_c) / prev_c
                full_sq += r * r
                full_n += 1
        full_vol = math.sqrt(full_sq / full_n) if full_n > 0 else 0.001

        vol_ratio = recent_vol / full_vol if full_vol > 0 else 1.0
        # High vol if recent > 1.5x average, low vol if < 0.5x
        if vol_ratio > 1.5:
            features["vol_regime"] = 1.0
        elif vol_ratio < 0.5:
            features["vol_regime"] = -1.0
        else:
            features["vol_regime"] = 0.0
        features["vol_percentile"] = min(vol_ratio / 2.0, 1.0)
    else:
        features["vol_regime"] = 0.0
        features["vol_percentile"] = 0.5

    # --- Feature 5: Time of day ---
    hour = wd.timestamp.hour
    features["tod_hour"] = float(hour)
    features["tod_asia"] = 1.0 if hour in SESSIONS["asia"] else 0.0
    features["tod_europe"] = 1.0 if hour in SESSIONS["europe"] else 0.0
    features["tod_us"] = 1.0 if hour in SESSIONS["us"] else 0.0
    features["tod_late"] = 1.0 if hour in SESSIONS["late"] else 0.0

    # --- Feature 6: Candle pattern (consecutive streak) ---
    if hist_len >= 2:
        last_c = all_candles[hist_end - 1]
        last_dir = 1.0 if last_c.close > last_c.open else -1.0
        streak = 1
        for j in range(hist_end - 2, max(hist_end - 11, hist_start - 1), -1):
            c = all_candles[j]
            d = 1.0 if c.close > c.open else -1.0
            if d == last_dir:
                streak += 1
            else:
                break
        features["streak_len"] = float(streak)
        features["streak_dir"] = last_dir
    else:
        features["streak_len"] = 0.0
        features["streak_dir"] = 0.0

    # --- Feature 7: Volume profile ---
    if hist_len >= 60:
        total_vol = 0.0
        for j in range(hist_start, hist_end):
            total_vol += all_candles[j].volume
        avg_vol = total_vol / hist_len if hist_len > 0 else 1.0
        if avg_vol == 0:
            avg_vol = 1.0

        recent_vol_sum = 0.0
        up_vol = 0.0
        down_vol = 0.0
        for j in range(hist_end - 15, hist_end):
            c = all_candles[j]
            recent_vol_sum += c.volume
            if c.close > c.open:
                up_vol += c.volume
            elif c.close < c.open:
                down_vol += c.volume
        recent_avg = recent_vol_sum / 15.0
        features["vol_ratio"] = recent_avg / avg_vol

        total_dir_vol = up_vol + down_vol
        if total_dir_vol > 0:
            features["vol_dir_align"] = (up_vol - down_vol) / total_dir_vol
        else:
            features["vol_dir_align"] = 0.0
    else:
        features["vol_ratio"] = 1.0
        features["vol_dir_align"] = 0.0

    # --- Feature 8: Early-window features (minutes 0 through entry_minute) ---
    window_candles = windows[wd.window_idx]
    if entry_minute > 0 and entry_minute <= len(window_candles):
        early = window_candles[:entry_minute]
        open_price = early[0].open
        close_price = early[-1].close
        if open_price != 0:
            early_cum_return = (close_price - open_price) / open_price
        else:
            early_cum_return = 0.0
        features["early_cum_return"] = early_cum_return
        features["early_direction"] = 1.0 if early_cum_return > 0 else (-1.0 if early_cum_return < 0 else 0.0)
        features["early_magnitude"] = abs(early_cum_return)
        green_count = sum(1 for c in early if c.close > c.open)
        features["early_green_ratio"] = green_count / len(early)
        if len(early) >= 2:
            sq_sum = 0.0
            for j in range(1, len(early)):
                prev_c = early[j - 1].close
                if prev_c > 0:
                    r = (early[j].close - prev_c) / prev_c
                    sq_sum += r * r
            features["early_vol"] = math.sqrt(sq_sum / (len(early) - 1)) if sq_sum > 0 else 0.0
        else:
            features["early_vol"] = 0.0
        max_move = 0.0
        for c in early:
            if c.open != 0:
                move = abs((c.close - c.open) / c.open)
                if move > max_move:
                    max_move = move
        features["early_max_move"] = max_move
    else:
        features["early_cum_return"] = 0.0
        features["early_direction"] = 0.0
        features["early_magnitude"] = 0.0
        features["early_green_ratio"] = 0.0
        features["early_vol"] = 0.0
        features["early_max_move"] = 0.0

    # --- Feature 9: RSI(14) ---
    if hist_len >= 15:
        gains = 0.0
        losses = 0.0
        for j in range(hist_end - 14, hist_end):
            delta = all_candles[j].close - all_candles[j - 1].close
            if delta > 0:
                gains += delta
            else:
                losses += abs(delta)
        avg_gain = gains / 14.0
        avg_loss = losses / 14.0
        if avg_loss == 0:
            features["rsi_14"] = 100.0
        else:
            rs = avg_gain / avg_loss
            features["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
    else:
        features["rsi_14"] = 50.0

    # --- Feature 10: MACD(12,26,9) histogram sign ---
    if hist_len >= 34:
        closes = [all_candles[j].close for j in range(hist_start, hist_end)]

        def _ema(data: list[float], period: int) -> float:
            if len(data) < period:
                return data[-1] if data else 0.0
            k = 2.0 / (period + 1)
            ema_val = sum(data[:period]) / period
            for val in data[period:]:
                ema_val = val * k + ema_val * (1 - k)
            return ema_val

        ema12 = _ema(closes, 12)
        ema26 = _ema(closes, 26)
        macd_line = ema12 - ema26
        features["macd_histogram_sign"] = 1.0 if macd_line > 0 else (-1.0 if macd_line < 0 else 0.0)
    else:
        features["macd_histogram_sign"] = 0.0

    # --- Feature 11: Bollinger %B(20,2) ---
    if hist_len >= 20:
        bb_closes = [all_candles[j].close for j in range(hist_end - 20, hist_end)]
        bb_mean = sum(bb_closes) / 20.0
        bb_var = sum((c - bb_mean) ** 2 for c in bb_closes) / 20.0
        bb_std = math.sqrt(bb_var) if bb_var > 0 else 0.001
        upper_bb = bb_mean + 2 * bb_std
        lower_bb = bb_mean - 2 * bb_std
        band_width = upper_bb - lower_bb
        if band_width > 0:
            features["bb_pct_b"] = (bb_closes[-1] - lower_bb) / band_width
        else:
            features["bb_pct_b"] = 0.5
    else:
        features["bb_pct_b"] = 0.5

    # --- Feature 12: ATR(14) ---
    if hist_len >= 15:
        atr_sum = 0.0
        for j in range(hist_end - 14, hist_end):
            c = all_candles[j]
            prev_close = all_candles[j - 1].close
            tr = max(c.high - c.low, abs(c.high - prev_close), abs(c.low - prev_close))
            atr_sum += tr
        features["atr_14"] = atr_sum / 14.0
    else:
        features["atr_14"] = 0.0

    # --- Feature 13: Mean reversion z-score (vs 60-candle SMA) ---
    if hist_len >= 60:
        mr_closes = [all_candles[j].close for j in range(hist_end - 60, hist_end)]
        mr_mean = sum(mr_closes) / 60.0
        mr_var = sum((c - mr_mean) ** 2 for c in mr_closes) / 60.0
        mr_std = math.sqrt(mr_var) if mr_var > 0 else 0.001
        features["mean_reversion_z"] = (mr_closes[-1] - mr_mean) / mr_std
    else:
        features["mean_reversion_z"] = 0.0

    # --- Feature 14: Price vs VWAP (60-candle) ---
    if hist_len >= 60:
        vwap_pv = 0.0
        vwap_vol = 0.0
        for j in range(hist_end - 60, hist_end):
            c = all_candles[j]
            typical = (c.high + c.low + c.close) / 3.0
            vwap_pv += typical * c.volume
            vwap_vol += c.volume
        if vwap_vol > 0:
            vwap = vwap_pv / vwap_vol
            current_close = all_candles[hist_end - 1].close
            features["price_vs_vwap"] = (current_close - vwap) / vwap if vwap != 0 else 0.0
        else:
            features["price_vs_vwap"] = 0.0
    else:
        features["price_vs_vwap"] = 0.0

    return features


# ---------------------------------------------------------------------------
# Strategy variants
# ---------------------------------------------------------------------------

def strategy_a_prior_momentum(features: dict[str, float]) -> str | None:
    """A: Follow previous 15m candle direction."""
    d = features.get("prior_dir", 0.0)
    if d > 0:
        return "Up"
    if d < 0:
        return "Down"
    return None


def strategy_b_multi_tf_align(features: dict[str, float]) -> str | None:
    """B: Trade only when 15m/1h/4h agree."""
    score = features.get("mtf_score", 0.0)
    if score == 3.0:
        return "Up"
    if score == -3.0:
        return "Down"
    return None


def strategy_c_short_carry(features: dict[str, float]) -> str | None:
    """C: Follow last 5-min momentum into new window."""
    d = features.get("stm_dir", 0.0)
    strength = features.get("stm_strength", 0.0)
    if d > 0 and strength >= 0.6:
        return "Up"
    if d < 0 and (1.0 - strength) >= 0.6:
        return "Down"
    return None


def strategy_d_vol_conditional(features: dict[str, float]) -> str | None:
    """D: Prior momentum, only in high-vol regime."""
    if features.get("vol_regime", 0.0) != 1.0:
        return None
    return strategy_a_prior_momentum(features)


def strategy_e_time_filtered(features: dict[str, float]) -> str | None:
    """E: Prior momentum, only during US/Europe sessions."""
    if features.get("tod_us", 0.0) != 1.0 and features.get("tod_europe", 0.0) != 1.0:
        return None
    return strategy_a_prior_momentum(features)


def strategy_f_ensemble(features: dict[str, float]) -> str | None:
    """F: Weighted vote of all features."""
    votes: dict[str, float] = {}

    # Prior momentum (0.25)
    d = features.get("prior_dir", 0.0)
    if d != 0:
        direction = "Up" if d > 0 else "Down"
        votes[direction] = votes.get(direction, 0.0) + 0.25

    # Multi-TF (0.20) — require 2+ alignment
    score = features.get("mtf_score", 0.0)
    if abs(score) >= 2.0:
        direction = "Up" if score > 0 else "Down"
        votes[direction] = votes.get(direction, 0.0) + 0.20

    # Short-term momentum (0.20)
    stm_d = features.get("stm_dir", 0.0)
    if stm_d != 0:
        direction = "Up" if stm_d > 0 else "Down"
        votes[direction] = votes.get(direction, 0.0) + 0.20

    # Candle pattern (0.15) — require 3+ streak
    streak_dir = features.get("streak_dir", 0.0)
    if streak_dir != 0 and features.get("streak_len", 0.0) >= 3:
        direction = "Up" if streak_dir > 0 else "Down"
        votes[direction] = votes.get(direction, 0.0) + 0.15

    # Volume alignment (0.10)
    vol_align = features.get("vol_dir_align", 0.0)
    if abs(vol_align) > 0.3:
        direction = "Up" if vol_align > 0 else "Down"
        votes[direction] = votes.get(direction, 0.0) + 0.10

    # Vol regime boost (0.10)
    if features.get("vol_regime", 0.0) == 1.0 and votes:
        leading = max(votes, key=votes.get)  # type: ignore[arg-type]
        votes[leading] = votes.get(leading, 0.0) + 0.10

    if not votes:
        return None
    leading_dir = max(votes, key=votes.get)  # type: ignore[arg-type]
    if votes[leading_dir] < 0.40:
        return None
    return leading_dir


STRATEGIES: dict[str, Any] = {
    "A_prior_momentum": strategy_a_prior_momentum,
    "B_multi_tf_align": strategy_b_multi_tf_align,
    "C_short_carry": strategy_c_short_carry,
    "D_vol_conditional": strategy_d_vol_conditional,
    "E_time_filtered": strategy_e_time_filtered,
    "F_ensemble": strategy_f_ensemble,
}


# ---------------------------------------------------------------------------
# Trade simulation
# ---------------------------------------------------------------------------

def simulate_trade(
    direction: str,
    resolution: str,
    entry_price: float = ENTRY_PRICE,
) -> tuple[float, float, float, float]:
    """Returns (settlement, pnl_gross, pnl_net, fee)."""
    correct = direction == resolution
    settlement = 1.0 if correct else 0.0
    pnl_gross = ((settlement - entry_price) / entry_price) * POSITION_SIZE
    fee = polymarket_fee(POSITION_SIZE, entry_price)
    slip = POSITION_SIZE * SLIPPAGE_BPS / 10000
    total_fee = fee + slip
    pnl_net = pnl_gross - total_fee
    return settlement, pnl_gross, pnl_net, total_fee


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(trades: list[TradeResult], name: str) -> StrategyMetrics:
    m = StrategyMetrics(name=name)
    if not trades:
        return m

    m.total_trades = len(trades)
    m.correct = sum(1 for t in trades if t.correct)
    m.accuracy = m.correct / m.total_trades

    net_pnls = [t.pnl_net for t in trades]
    m.net_pnl = sum(net_pnls)
    m.gross_pnl = sum(t.pnl_gross for t in trades)
    m.total_fees = sum(t.fee for t in trades)
    m.ev_per_trade = m.net_pnl / m.total_trades

    # Sharpe (annualized, 35040 periods/year)
    returns = [p / POSITION_SIZE for p in net_pnls]
    mean_r = sum(returns) / len(returns)
    var_r = sum((r - mean_r) ** 2 for r in returns) / len(returns)
    std_r = math.sqrt(var_r) if var_r > 0 else 0.001
    m.sharpe = (mean_r / std_r) * math.sqrt(35040)

    # Max drawdown
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in net_pnls:
        cum += p
        if cum > peak:
            peak = cum
        dd = (peak - cum) / POSITION_SIZE
        if dd > max_dd:
            max_dd = dd
    m.max_drawdown = max_dd

    # Profit factor
    gains = sum(p for p in net_pnls if p > 0)
    losses = abs(sum(p for p in net_pnls if p < 0))
    m.profit_factor = gains / losses if losses > 0 else float("inf")

    # Monthly
    by_month: dict[str, float] = defaultdict(float)
    for t in trades:
        by_month[t.timestamp.strftime("%Y-%m")] += t.pnl_net
    m.monthly_pnls = dict(by_month)
    profitable = sum(1 for v in by_month.values() if v > 0)
    m.profitable_months_pct = (profitable / len(by_month) * 100) if by_month else 0.0

    return m


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def bootstrap_ci(correct_list: list[bool], n: int = BOOTSTRAP_ITERATIONS) -> tuple[float, float, float]:
    sz = len(correct_list)
    if sz == 0:
        return 0.0, 0.0, 0.0
    means = []
    for _ in range(n):
        s = sum(correct_list[random.randint(0, sz - 1)] for _ in range(sz))
        means.append(s / sz)
    means.sort()
    lo = means[int(0.025 * n)]
    hi = means[int(0.975 * n)]
    return sum(correct_list) / sz, lo, hi


def permutation_test(preds: list[str], actuals: list[str], n: int = PERMUTATION_ITERATIONS) -> float:
    sz = len(preds)
    if sz == 0:
        return 1.0
    obs = sum(1 for p, a in zip(preds, actuals) if p == a) / sz
    count = 0
    for _ in range(n):
        shuffled = list(actuals)
        random.shuffle(shuffled)
        perm = sum(1 for p, a in zip(preds, shuffled) if p == a) / sz
        if perm >= obs:
            count += 1
    return count / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 120)
    print("  BEFORE WINDOW OPEN (BWO) STRATEGY — BACKTEST")
    print("  Entry at $0.50 pre-window | Break-even ~52% after fees")
    print("=" * 120)
    _flush()

    if not BTC_CSV.exists():
        print(f"  ERROR: {BTC_CSV} not found.")
        sys.exit(1)

    # Load data
    print("\n  Loading BTC 1m candle data...", end=" ")
    _flush()
    all_candles = load_csv_fast(BTC_CSV)
    all_candles.sort(key=lambda c: c.timestamp)

    # Deduplicate
    seen: set[datetime] = set()
    unique: list[FastCandle] = []
    for c in all_candles:
        if c.timestamp not in seen:
            seen.add(c.timestamp)
            unique.append(c)
    all_candles = unique
    print(f"{len(all_candles):,} unique candles")
    _flush()

    # Group into 15m windows
    print("  Grouping into 15m windows...", end=" ")
    _flush()
    windows = group_into_15m_windows(all_candles)
    print(f"{len(windows):,} complete windows")
    _flush()

    # Build timestamp→candle_index map
    candle_by_ts: dict[datetime, int] = {}
    for i, c in enumerate(all_candles):
        candle_by_ts[c.timestamp] = i

    # Prepare window data (lightweight — indices only)
    print("  Building window metadata...", end=" ")
    _flush()
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
    print(f"{len(window_data):,} windows with context")
    _flush()

    # Base rate
    up_count = sum(1 for wd in window_data if wd.resolution == "Up")
    down_count = len(window_data) - up_count
    print(f"\n  {'='*110}")
    print(f"  BASE RATE: Up {up_count:,} ({up_count/len(window_data)*100:.1f}%) | "
          f"Down {down_count:,} ({down_count/len(window_data)*100:.1f}%)")
    _flush()

    # Train/test split (80/20 chronological)
    split_idx = int(len(window_data) * 0.80)
    train_data = window_data[:split_idx]
    test_data = window_data[split_idx:]
    print(f"  Train: {len(train_data):,} | Test: {len(test_data):,}")
    print(f"  Train: {train_data[0].timestamp} → {train_data[-1].timestamp}")
    print(f"  Test:  {test_data[0].timestamp} → {test_data[-1].timestamp}")
    _flush()

    # Pre-compute ALL features once
    print(f"\n  Pre-computing features for {len(window_data):,} windows...", end=" ")
    _flush()
    all_features: list[dict[str, float]] = []
    for i, wd in enumerate(window_data):
        all_features.append(compute_all_features(wd, windows, all_candles))
        if (i + 1) % 10000 == 0:
            print(f"{i+1:,}...", end=" ")
            _flush()
    print("done")
    _flush()

    # Index mapping: window_data index → features
    train_features = all_features[:split_idx]
    test_features = all_features[split_idx:]

    # Report
    report: dict[str, Any] = {
        "meta": {
            "total_candles": len(all_candles),
            "total_windows": len(window_data),
            "base_rate_up_pct": round(up_count / len(window_data) * 100, 2),
            "train_size": len(train_data),
            "test_size": len(test_data),
            "entry_price": ENTRY_PRICE,
            "position_size": POSITION_SIZE,
            "fee_at_050": round(polymarket_fee(POSITION_SIZE, 0.50), 4),
        },
        "strategies": {},
    }
    all_trade_rows: list[dict[str, Any]] = []

    # Run strategies
    for strat_name, strat_fn in STRATEGIES.items():
        print(f"\n  {'='*110}")
        print(f"  STRATEGY {strat_name}")
        print(f"  {'='*110}")
        _flush()

        # Train
        train_trades: list[TradeResult] = []
        for i, wd in enumerate(train_data):
            direction = strat_fn(train_features[i])
            if direction is None:
                continue
            settlement, pnl_g, pnl_n, fee = simulate_trade(direction, wd.resolution)
            train_trades.append(TradeResult(
                timestamp=wd.timestamp, strategy=strat_name, direction=direction,
                resolution=wd.resolution, correct=direction == wd.resolution,
                entry_price=ENTRY_PRICE, settlement=settlement,
                pnl_gross=pnl_g, pnl_net=pnl_n, fee=fee,
            ))
        train_m = compute_metrics(train_trades, f"{strat_name}_train")

        # Test
        test_trades: list[TradeResult] = []
        for i, wd in enumerate(test_data):
            direction = strat_fn(test_features[i])
            if direction is None:
                continue
            settlement, pnl_g, pnl_n, fee = simulate_trade(direction, wd.resolution)
            test_trades.append(TradeResult(
                timestamp=wd.timestamp, strategy=strat_name, direction=direction,
                resolution=wd.resolution, correct=direction == wd.resolution,
                entry_price=ENTRY_PRICE, settlement=settlement,
                pnl_gross=pnl_g, pnl_net=pnl_n, fee=fee,
                features=test_features[i],
            ))
        test_m = compute_metrics(test_trades, f"{strat_name}_test")

        # Print
        print(f"\n  {'Split':<8} {'Trades':>8} {'Acc%':>7} {'EV/trade':>10} {'Net PnL':>12} "
              f"{'Sharpe':>8} {'MaxDD':>8} {'PF':>8} {'Mo+%':>6}")
        print(f"  {'-'*78}")
        for label, m in [("Train", train_m), ("Test", test_m)]:
            pf_str = f"{m.profit_factor:>7.2f}" if m.profit_factor != float("inf") else "    inf"
            print(f"  {label:<8} {m.total_trades:>8,} {m.accuracy*100:>6.1f}% ${m.ev_per_trade:>+8.2f} "
                  f"${m.net_pnl:>+10,.0f} {m.sharpe:>8.2f} {m.max_drawdown:>7.1%} "
                  f"{pf_str} {m.profitable_months_pct:>5.0f}%")

        # IS/OOS gap
        gap = abs(train_m.accuracy - test_m.accuracy) * 100
        overfit = gap > 5.0
        print(f"  IS/OOS gap: {gap:.1f}%{' — WARNING: possible overfitting' if overfit else ' (OK)'}")
        _flush()

        # Bootstrap + permutation on test
        if test_trades:
            correct_list = [t.correct for t in test_trades]
            _, ci_lo, ci_hi = bootstrap_ci(correct_list)
            print(f"  Bootstrap 95% CI: [{ci_lo*100:.1f}%, {ci_hi*100:.1f}%]")

            preds = [t.direction for t in test_trades]
            actuals = [t.resolution for t in test_trades]
            p_val = permutation_test(preds, actuals)
            sig = p_val < 0.05
            print(f"  Permutation p-value: {p_val:.4f} {'(significant)' if sig else '(not significant)'}")
        else:
            ci_lo, ci_hi, p_val, sig = 0.0, 0.0, 1.0, False
        _flush()

        # Walk-forward
        print(f"  Walk-forward...", end=" ")
        _flush()
        by_month_idx: dict[str, list[int]] = defaultdict(list)
        for idx_wd, wd in enumerate(window_data):
            by_month_idx[wd.timestamp.strftime("%Y-%m")].append(idx_wd)
        months = sorted(by_month_idx.keys())
        wf_results: list[StrategyMetrics] = []
        for mi in range(WF_TRAIN_MONTHS, len(months), WF_TEST_MONTHS):
            if mi >= len(months):
                break
            test_month = months[mi]
            wf_trades: list[TradeResult] = []
            for idx_wd in by_month_idx[test_month]:
                wd = window_data[idx_wd]
                direction = strat_fn(all_features[idx_wd])
                if direction is None:
                    continue
                s, pg, pn, f = simulate_trade(direction, wd.resolution)
                wf_trades.append(TradeResult(
                    timestamp=wd.timestamp, strategy=f"WF_{test_month}",
                    direction=direction, resolution=wd.resolution,
                    correct=direction == wd.resolution, entry_price=ENTRY_PRICE,
                    settlement=s, pnl_gross=pg, pnl_net=pn, fee=f,
                ))
            if wf_trades:
                wf_results.append(compute_metrics(wf_trades, f"WF_{test_month}"))

        if wf_results:
            wf_accs = [m.accuracy for m in wf_results if m.total_trades > 0]
            wf_pnls = [m.net_pnl for m in wf_results]
            wf_mean = sum(wf_accs) / len(wf_accs) if wf_accs else 0
            wf_pnl = sum(wf_pnls)
            wf_prof = sum(1 for p in wf_pnls if p > 0)
            print(f"{len(wf_results)} periods, mean acc {wf_mean*100:.1f}%, "
                  f"PnL ${wf_pnl:+,.0f}, profitable {wf_prof}/{len(wf_results)}")
        else:
            wf_mean, wf_pnl = 0.0, 0.0
        _flush()

        # Store report
        report["strategies"][strat_name] = {
            "train": {
                "trades": train_m.total_trades, "accuracy": round(train_m.accuracy, 4),
                "ev_per_trade": round(train_m.ev_per_trade, 4), "net_pnl": round(train_m.net_pnl, 2),
                "sharpe": round(train_m.sharpe, 2), "max_drawdown": round(train_m.max_drawdown, 4),
                "profit_factor": round(min(train_m.profit_factor, 9999), 2),
                "profitable_months_pct": round(train_m.profitable_months_pct, 1),
            },
            "test": {
                "trades": test_m.total_trades, "accuracy": round(test_m.accuracy, 4),
                "ev_per_trade": round(test_m.ev_per_trade, 4), "net_pnl": round(test_m.net_pnl, 2),
                "sharpe": round(test_m.sharpe, 2), "max_drawdown": round(test_m.max_drawdown, 4),
                "profit_factor": round(min(test_m.profit_factor, 9999), 2),
                "profitable_months_pct": round(test_m.profitable_months_pct, 1),
            },
            "validation": {
                "is_oos_gap_pct": round(gap, 2),
                "overfit_flag": overfit,
                "bootstrap_ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
                "permutation_p_value": round(p_val, 4),
                "statistically_significant": sig,
                "walk_forward_mean_accuracy": round(wf_mean, 4),
                "walk_forward_total_pnl": round(wf_pnl, 2),
            },
            "passes_minimum": (
                test_m.total_trades >= MIN_TRADES
                and test_m.accuracy > 0.52
                and sig
                and not overfit
            ),
        }

        # Trade rows for CSV
        for t in test_trades:
            row: dict[str, Any] = {
                "timestamp": t.timestamp.isoformat(), "strategy": t.strategy,
                "direction": t.direction, "resolution": t.resolution,
                "correct": t.correct, "entry_price": t.entry_price,
                "settlement": t.settlement,
                "pnl_gross": round(t.pnl_gross, 4), "pnl_net": round(t.pnl_net, 4),
                "fee": round(t.fee, 4),
            }
            row.update({k: round(v, 6) for k, v in t.features.items()})
            all_trade_rows.append(row)

    # Summary
    print(f"\n\n  {'='*110}")
    print(f"  SUMMARY — ALL STRATEGIES")
    print(f"  {'='*110}")
    print(f"\n  {'Strategy':<25} {'Test Acc':>9} {'Trades':>8} {'Net PnL':>12} {'p-val':>7} {'Pass':>6}")
    print(f"  {'-'*70}")

    best_strat = None
    best_acc = 0.0
    for name, data in report["strategies"].items():
        t = data["test"]
        v = data["validation"]
        passes = data["passes_minimum"]
        print(f"  {name:<25} {t['accuracy']*100:>8.1f}% {t['trades']:>8,} "
              f"${t['net_pnl']:>+10,.0f} {v['permutation_p_value']:>6.4f} "
              f"{'YES' if passes else 'NO':>6}")
        if passes and t["accuracy"] > best_acc:
            best_acc = t["accuracy"]
            best_strat = name

    if best_strat:
        print(f"\n  BEST: {best_strat} (test acc {best_acc*100:.1f}%)")
        report["best_strategy"] = best_strat
    else:
        print(f"\n  NO STRATEGY passes (>52% acc, p<0.05, 500+ trades, <5% gap)")
        report["best_strategy"] = None

    # Save
    with open(REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report: {REPORT_JSON}")

    if all_trade_rows:
        fieldnames = list(all_trade_rows[0].keys())
        with open(TRADES_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_trade_rows)
        print(f"  Trades: {TRADES_CSV} ({len(all_trade_rows):,} rows)")

    print(f"\n{'='*120}")
    _flush()


if __name__ == "__main__":
    main()
