"""BWO 15m Fixed Strategies Backtest.

Fixes two CRITICAL issues in the v5 continuation backtest:
1. Entry prices used 0.50 + 0.3*cum_ret ≈ $0.50 always (wrong by ~1000x)
   Real min-3 prices are $0.60-$0.80 for the winning direction.
2. Backtest traded symmetrically (buy winning direction) but live Hybrid B
   always bought YES (only profits when BTC goes UP → 50% base rate loss).

Tests TWO fixed strategies with REALISTIC sigmoid pricing:

Strategy A — "Symmetric Min-3 Entry":
  Like the working 5m strategy: wait until min 3, observe BTC direction,
  buy winning direction at estimated market price, only if model is
  confident AND EV-positive. Hold to settlement.

Strategy B — "Improved BuyBoth":
  Buy both YES+NO at $0.50 pre-window. At min 3:
  - If cont_prob >= HIGH_THRESHOLD: keep winner, sell loser at bid
  - If cont_prob < HIGH_THRESHOLD: sell BOTH at bids (small loss only)
  Higher threshold prevents catastrophic losses from wrong-but-confident.

Walk-forward validated with realistic Polymarket pricing.
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
from sklearn.ensemble import HistGradientBoostingClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest_before_window import (
    WindowData,
    compute_all_features,
)
from scripts.fast_loader import FastCandle, group_into_15m_windows, load_csv_fast

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BTC_SPOT_CSV = PROJECT_ROOT / "data" / "btc_1m_2y.csv"
BTC_FUTURES_CSV = PROJECT_ROOT / "data" / "btc_futures_1m_2y.csv"
ETH_FUTURES_CSV = PROJECT_ROOT / "data" / "eth_futures_1m_2y.csv"
DVOL_CSV = PROJECT_ROOT / "data" / "deribit_dvol_1m.csv"
DERIBIT_PERP_CSV = PROJECT_ROOT / "data" / "deribit_btc_perp_1m.csv"
REPORT_JSON = PROJECT_ROOT / "data" / "bwo_15m_fixed_report.json"

POSITION_SIZE = 200.0  # $200 bankroll per trade (matches paper trader)
FEE_CONSTANT = 0.25
SLIPPAGE_BPS = 5

WF_TRAIN_MONTHS = 3
WF_TEST_MONTHS = 1

ENTRY_MINUTE = 3  # Observe first 3 candles, then decide

# Pricing model: sigmoid sensitivity for 15m window at min 3
# At min 3, 20% of 15m window has elapsed
# Calibrated to paper trading observations: $0.60-$0.80 for typical moves
PRICE_SENSITIVITY = 1200

# Strategy A parameters
MIN_EDGES = [0.02, 0.03, 0.05]  # Test multiple min edge values
MAX_ENTRY_PRICE = 0.85

# Strategy B parameters
BUYBOTH_ENTRY = 0.50  # Pre-window entry price per side

# Test thresholds
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def _flush() -> None:
    sys.stdout.flush()


def polymarket_fee(size: float, price: float) -> float:
    if price <= 0 or price >= 1:
        return 0.0
    return size * FEE_CONSTANT * (price ** 2) * ((1.0 - price) ** 2)


def _parse_ts(raw: str) -> datetime | None:
    raw = raw.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(raw)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Pricing model (realistic min-3 Polymarket prices)
# ---------------------------------------------------------------------------

def estimate_min3_prices(cum_ret: float, early_dir: float) -> dict[str, float]:
    """Estimate Polymarket YES/NO bid/ask prices at minute 3.

    Uses sigmoid model calibrated to paper trading observations.
    Returns conservative (slightly high) ask prices to ensure
    backtest profitability translates to live.
    """
    logit = cum_ret * PRICE_SENSITIVITY
    if abs(logit) > 500:
        yes_prob = 1.0 if logit > 0 else 0.0
    else:
        yes_prob = 1.0 / (1.0 + math.exp(-logit))

    spread = 0.02  # 2 cent spread (ask above, bid below fair)
    yes_ask = min(yes_prob + spread, 0.95)
    yes_bid = max(yes_prob - spread, 0.05)
    no_ask = min((1.0 - yes_prob) + spread, 0.95)
    no_bid = max((1.0 - yes_prob) - spread, 0.05)

    # Determine winning direction prices
    if early_dir > 0:
        # BTC went UP → winning side is YES
        winner_ask = yes_ask
        winner_bid = yes_bid
        loser_ask = no_ask
        loser_bid = no_bid
    else:
        # BTC went DOWN → winning side is NO
        winner_ask = no_ask
        winner_bid = no_bid
        loser_ask = yes_ask
        loser_bid = yes_bid

    # Clamp to realistic range
    winner_ask = max(0.52, min(winner_ask, 0.85))
    loser_bid = max(0.05, min(loser_bid, 0.45))

    return {
        "winner_ask": winner_ask,
        "winner_bid": winner_bid,
        "loser_ask": loser_ask,
        "loser_bid": loser_bid,
        "yes_prob": yes_prob,
    }


# ---------------------------------------------------------------------------
# Signal Quality Features (from first N candles)
# ---------------------------------------------------------------------------

def compute_signal_quality_features(
    window_candles: list[FastCandle],
    entry_minute: int,
    all_candles: list[FastCandle],
    candle_idx: int,
) -> dict[str, float]:
    features: dict[str, float] = {}

    if entry_minute < 1 or entry_minute > len(window_candles):
        for name in [
            "signal_size", "signal_volume_ratio", "signal_consistency",
            "signal_body_ratio", "signal_vs_atr", "signal_volume_surge",
            "signal_candle_range", "signal_close_position",
        ]:
            features[name] = 0.0
        return features

    early = window_candles[:entry_minute]
    open_price = early[0].open
    close_price = early[-1].close
    if open_price > 0:
        cum_return = (close_price - open_price) / open_price
    else:
        cum_return = 0.0

    features["signal_size"] = abs(cum_return)

    # Volume ratio vs recent average
    lookback = 60
    start_idx = max(0, candle_idx - lookback)
    recent_vols = [all_candles[j].volume for j in range(start_idx, candle_idx)]
    avg_vol = sum(recent_vols) / len(recent_vols) if recent_vols else 1.0
    signal_vol = sum(c.volume for c in early)
    features["signal_volume_ratio"] = signal_vol / (avg_vol * entry_minute) if avg_vol > 0 else 1.0

    # Consistency: fraction of candles in same direction
    if cum_return > 0:
        consistent = sum(1 for c in early if c.close > c.open)
    elif cum_return < 0:
        consistent = sum(1 for c in early if c.close < c.open)
    else:
        consistent = 0
    features["signal_consistency"] = consistent / len(early) if early else 0.0

    # Body ratio (body / wick) for signal candle
    total_body = 0.0
    total_range = 0.0
    for c in early:
        body = abs(c.close - c.open)
        rng = c.high - c.low
        total_body += body
        total_range += rng
    features["signal_body_ratio"] = total_body / total_range if total_range > 0 else 0.5

    # Signal vs ATR
    atr_lookback = 14
    start_atr = max(0, candle_idx - atr_lookback)
    trs = []
    for j in range(start_atr, candle_idx):
        c = all_candles[j]
        trs.append(c.high - c.low)
    atr = sum(trs) / len(trs) if trs else 0.001
    features["signal_vs_atr"] = abs(close_price - open_price) / atr if atr > 0 else 0.0

    # Volume surge
    if entry_minute >= 2 and len(early) >= 2:
        last_vol = early[-1].volume
        prev_avg = sum(c.volume for c in early[:-1]) / (len(early) - 1) if len(early) > 1 else 1.0
        features["signal_volume_surge"] = last_vol / prev_avg if prev_avg > 0 else 1.0
    else:
        features["signal_volume_surge"] = 1.0

    # Candle range
    features["signal_candle_range"] = total_range / open_price if open_price > 0 else 0.0

    # Close position within range
    window_high = max(c.high for c in early)
    window_low = min(c.low for c in early)
    window_range = window_high - window_low
    features["signal_close_position"] = (close_price - window_low) / window_range if window_range > 0 else 0.5

    return features


def compute_taker_features_simple(
    ci: int,
    futures_data: list[dict],
    futures_lookup: dict[datetime, int],
    all_candles: list[FastCandle],
) -> dict[str, float]:
    features: dict[str, float] = {}
    target_ts = all_candles[ci].timestamp
    fi = futures_lookup.get(target_ts)

    names = ["taker_imbalance_5", "taker_imbalance_15", "taker_imbalance_60",
             "taker_accel", "taker_intensity"]

    if fi is None or fi < 60:
        return {n: 0.0 for n in names}

    def _imbalance(lookback: int) -> float:
        start = max(0, fi - lookback)
        total_vol = 0.0
        buy_vol = 0.0
        for j in range(start, fi):
            d = futures_data[j]
            v = d.get("volume", 0)
            bv = d.get("taker_buy_volume", 0)
            total_vol += v
            buy_vol += bv
        if total_vol == 0:
            return 0.0
        sell_vol = total_vol - buy_vol
        return (buy_vol - sell_vol) / total_vol

    imb_5 = _imbalance(5)
    imb_15 = _imbalance(15)
    imb_60 = _imbalance(60)
    features["taker_imbalance_5"] = imb_5
    features["taker_imbalance_15"] = imb_15
    features["taker_imbalance_60"] = imb_60
    features["taker_accel"] = imb_5 - imb_15

    start_ti = max(0, fi - 15)
    total_buy = sum(futures_data[j].get("taker_buy_volume", 0) for j in range(start_ti, fi))
    total_trades = sum(futures_data[j].get("num_trades", 0) for j in range(start_ti, fi))
    features["taker_intensity"] = total_buy / total_trades if total_trades > 0 else 0.0

    return features


# ---------------------------------------------------------------------------
# Walk-forward with realistic strategy simulation
# ---------------------------------------------------------------------------

@dataclass
class TradeResult:
    window_ts: datetime
    strategy: str
    entered: bool = False
    direction: str = ""  # "Up" or "Down"
    cont_prob: float = 0.0
    entry_price: float = 0.0
    exit_price: float = 0.0
    shares: float = 0.0
    pnl_gross: float = 0.0
    fee: float = 0.0
    pnl_net: float = 0.0
    correct: bool = False
    skip_reason: str = ""


def walk_forward_fixed(
    window_data: list[WindowData],
    all_feature_dicts: list[dict[str, float]],
    feature_names: list[str],
    continuation_targets: list[int],
    valid_mask: list[bool],
    entry_prices: list[float],  # Estimated min-3 winner ask prices
    loser_bids: list[float],    # Estimated min-3 loser bid prices
    threshold: float,
    min_edge: float,
    strategy: str,  # "symmetric" or "buyboth"
) -> list[dict[str, Any]]:
    """Walk-forward with realistic pricing for both strategies."""
    by_month: dict[str, list[int]] = defaultdict(list)
    for i, wd in enumerate(window_data):
        if valid_mask[i]:
            by_month[wd.timestamp.strftime("%Y-%m")].append(i)
    months = sorted(by_month.keys())

    if len(months) < WF_TRAIN_MONTHS + WF_TEST_MONTHS:
        return []

    results: list[dict[str, Any]] = []

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

        X_train = np.array([[all_feature_dicts[i].get(f, 0.0) for f in feature_names] for i in train_indices])
        y_train = np.array([continuation_targets[i] for i in train_indices])
        X_test = np.array([[all_feature_dicts[i].get(f, 0.0) for f in feature_names] for i in test_indices])
        y_test = np.array([continuation_targets[i] for i in test_indices])

        model = HistGradientBoostingClassifier(
            max_iter=300, max_depth=4, learning_rate=0.05,
            min_samples_leaf=80, l2_regularization=1.0,
            max_bins=128, random_state=42,
        )
        model.fit(X_train, y_train)
        probas = model.predict_proba(X_test)[:, 1]

        # --- Simulate trades for this month ---
        trades: list[TradeResult] = []
        net_pnl = 0.0

        for j, idx in enumerate(test_indices):
            wd = window_data[idx]
            cp = probas[j]
            continued = continuation_targets[idx] == 1
            ep = entry_prices[idx]
            lb = loser_bids[idx]

            if strategy == "symmetric":
                # Strategy A: Buy winning direction at min-3 price
                # Only if confident + EV-positive
                if cp >= threshold and cp > (ep + min_edge) and ep <= MAX_ENTRY_PRICE:
                    # Enter: buy winning direction at estimated ask
                    shares = POSITION_SIZE / ep
                    fee = polymarket_fee(POSITION_SIZE, ep)
                    slip = POSITION_SIZE * SLIPPAGE_BPS / 10000

                    if continued:
                        # Won: winning side settles at $1.00
                        pnl_gross = shares * (1.0 - ep)
                    else:
                        # Lost: winning side settles at $0.00
                        pnl_gross = -shares * ep  # = -POSITION_SIZE

                    pnl_net = pnl_gross - fee - slip
                    net_pnl += pnl_net
                    trades.append(TradeResult(
                        window_ts=wd.timestamp, strategy="symmetric",
                        entered=True, cont_prob=cp,
                        entry_price=ep, exit_price=1.0 if continued else 0.0,
                        shares=shares, pnl_gross=pnl_gross,
                        fee=fee + slip, pnl_net=pnl_net, correct=continued,
                    ))
                else:
                    trades.append(TradeResult(
                        window_ts=wd.timestamp, strategy="symmetric",
                        entered=False, cont_prob=cp,
                    ))

            elif strategy == "buyboth":
                # Strategy B: Buy both at $0.50 pre-window
                # At min 3: decide based on model
                yes_entry = BUYBOTH_ENTRY
                no_entry = BUYBOTH_ENTRY
                total_entry = yes_entry + no_entry  # ~$1.00

                # Position sizing: use half of POSITION_SIZE per side
                half_pos = POSITION_SIZE / 2.0
                yes_shares = half_pos / yes_entry
                no_shares = half_pos / no_entry
                entry_fee = polymarket_fee(half_pos, yes_entry) + polymarket_fee(half_pos, no_entry)

                if cp >= threshold:
                    # Confident: keep winning side, sell losing side at bid
                    sell_price = lb  # Loser's bid price at min-3
                    sell_revenue = (half_pos / (1.0 - sell_price + 0.02)) * sell_price  # Approximate
                    # Simpler: losing shares sell at loser_bid
                    loser_shares = half_pos / (1.0 - ep + 0.02)  # Approx loser entry was $0.50
                    sell_revenue = (half_pos / BUYBOTH_ENTRY) * sell_price
                    sell_fee = polymarket_fee(sell_revenue, sell_price)

                    if continued:
                        # Winner settles at $1.00
                        winner_revenue = (half_pos / BUYBOTH_ENTRY) * 1.0
                        pnl_gross = winner_revenue + sell_revenue - POSITION_SIZE
                    else:
                        # Winner settles at $0.00 (direction reversed)
                        winner_revenue = 0.0
                        pnl_gross = winner_revenue + sell_revenue - POSITION_SIZE
                    pnl_net = pnl_gross - entry_fee - sell_fee
                else:
                    # Not confident: sell BOTH at min-3 bids (small loss)
                    # Winner bid ≈ ep - 0.04 (winner bid is slightly below ask)
                    winner_bid = max(ep - 0.04, 0.50)
                    # Sell winner at winner_bid, loser at loser_bid
                    winner_shares = half_pos / BUYBOTH_ENTRY
                    loser_shares = half_pos / BUYBOTH_ENTRY
                    sell_rev_winner = winner_shares * winner_bid
                    sell_rev_loser = loser_shares * lb
                    sell_fee = (polymarket_fee(sell_rev_winner, winner_bid) +
                                polymarket_fee(sell_rev_loser, lb))
                    pnl_gross = sell_rev_winner + sell_rev_loser - POSITION_SIZE
                    pnl_net = pnl_gross - entry_fee - sell_fee

                net_pnl += pnl_net
                trades.append(TradeResult(
                    window_ts=wd.timestamp, strategy="buyboth",
                    entered=True, cont_prob=cp,
                    entry_price=BUYBOTH_ENTRY,
                    pnl_gross=pnl_gross, fee=entry_fee,
                    pnl_net=pnl_net, correct=continued,
                ))

        # --- Month stats ---
        entered_trades = [t for t in trades if t.entered]
        n_entered = len(entered_trades)
        n_correct = sum(1 for t in entered_trades if t.correct)

        results.append({
            "period": test_month,
            "total": len(test_indices),
            "traded": n_entered,
            "skip_rate": 1.0 - n_entered / len(test_indices) if test_indices else 1.0,
            "correct": n_correct,
            "accuracy": n_correct / n_entered if n_entered > 0 else 0.0,
            "net_pnl": float(net_pnl),
            "profitable": net_pnl > 0,
            "avg_pnl": float(net_pnl / n_entered) if n_entered > 0 else 0.0,
            "wins": sum(1 for t in entered_trades if t.pnl_net > 0),
            "losses": sum(1 for t in entered_trades if t.pnl_net <= 0),
        })

    return results


# ---------------------------------------------------------------------------
# Data loading (reused from v5 backtest)
# ---------------------------------------------------------------------------

def load_csv_to_dict(path: Path, fields: list[str]) -> tuple[list[dict], dict[datetime, int]]:
    import csv
    rows: list[dict] = []
    lookup: dict[datetime, int] = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = _parse_ts(row.get("timestamp", ""))
            if ts is None:
                continue
            d = {"timestamp": ts}
            for fld in fields:
                if fld in row:
                    try:
                        d[fld] = float(row[fld])
                    except (ValueError, TypeError):
                        d[fld] = 0.0
            lookup[ts] = len(rows)
            rows.append(d)
    return rows, lookup


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 100)
    print("  BWO 15m FIXED STRATEGIES BACKTEST")
    print("  Strategy A: Symmetric min-3 entry (like working 5m strategy)")
    print("  Strategy B: Improved BuyBoth (higher threshold, proper exits)")
    print("  Pricing: Sigmoid model (sensitivity=1200, 2c spread)")
    print("=" * 100)
    _flush()

    # ------------------------------------------------------------------
    # Load data
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
    # Build 15m windows
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
    _flush()

    # ------------------------------------------------------------------
    # Compute features + realistic prices
    # ------------------------------------------------------------------
    print(f"\n  Computing features (entry_minute={ENTRY_MINUTE})...")
    _flush()
    t0 = time.time()

    all_feature_dicts: list[dict[str, float]] = []
    continuation_targets: list[int] = []
    valid_mask: list[bool] = []
    entry_prices: list[float] = []  # Winner ask at min 3
    loser_bids: list[float] = []    # Loser bid at min 3

    for i, wd in enumerate(window_data):
        ci = wd.candle_idx
        w = windows[wd.window_idx]

        # Original features
        original = compute_all_features(wd, windows, all_candles, entry_minute=ENTRY_MINUTE)

        # Signal quality features
        signal = compute_signal_quality_features(w, ENTRY_MINUTE, all_candles, ci)

        # Taker features
        taker = compute_taker_features_simple(ci, futures_data, futures_lookup, all_candles)

        # DVOL features
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
                    m = sum(recent) / len(recent)
                    v = sum((x - m) ** 2 for x in recent) / len(recent)
                    s = math.sqrt(v) if v > 0 else 0.001
                    dvol_feats["dvol_z"] = (dvol_now - m) / s
                else:
                    dvol_feats["dvol_z"] = 0.0
            else:
                dvol_feats = {"dvol_level": 0.0, "dvol_change_5": 0.0, "dvol_z": 0.0}
        else:
            dvol_feats = {"dvol_level": 0.0, "dvol_change_5": 0.0, "dvol_z": 0.0}

        # ETH cross-asset
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

        # Combine all features
        combined = {**original, **signal, **taker, **dvol_feats, **eth_feats, **cross_feats}
        all_feature_dicts.append(combined)

        # Continuation target
        early_dir = original.get("early_direction", 0.0)
        cum_ret = original.get("early_cum_return", 0.0)

        if early_dir == 0.0:
            continuation_targets.append(0)
            valid_mask.append(False)
            entry_prices.append(0.52)
            loser_bids.append(0.48)
        else:
            continued = (
                (wd.resolution == "Up" and early_dir > 0) or
                (wd.resolution == "Down" and early_dir < 0)
            )
            continuation_targets.append(1 if continued else 0)
            valid_mask.append(True)

            # REALISTIC pricing using sigmoid model
            prices = estimate_min3_prices(cum_ret, early_dir)
            entry_prices.append(prices["winner_ask"])
            loser_bids.append(prices["loser_bid"])

        if (i + 1) % 10000 == 0:
            elapsed = time.time() - t0
            print(f"    {i+1:>8,} / {len(window_data):,} ({(i+1)/len(window_data)*100:.1f}%) - {elapsed:.0f}s")
            _flush()

    elapsed = time.time() - t0
    print(f"  Done: {elapsed:.1f}s")

    # Stats
    n_valid = sum(valid_mask)
    n_continued = sum(1 for i in range(len(window_data)) if valid_mask[i] and continuation_targets[i] == 1)
    cont_rate = n_continued / n_valid if n_valid > 0 else 0
    print(f"\n  Valid windows: {n_valid:,} / {len(window_data):,}")
    print(f"  Continuation base rate: {cont_rate*100:.1f}%")

    # Price distribution
    valid_prices = [entry_prices[i] for i in range(len(window_data)) if valid_mask[i]]
    print(f"  Entry price stats: min=${min(valid_prices):.3f}, "
          f"mean=${sum(valid_prices)/len(valid_prices):.3f}, "
          f"max=${max(valid_prices):.3f}")
    _flush()

    # ------------------------------------------------------------------
    # Feature selection (same as v5)
    # ------------------------------------------------------------------
    all_feature_names = sorted(set().union(*(d.keys() for d in all_feature_dicts[:100])))
    valid_indices = [i for i in range(len(window_data)) if valid_mask[i]]
    target_arr = np.array([continuation_targets[i] for i in valid_indices])

    from sklearn.metrics import mutual_info_score

    def compute_mi_bits(feature_values, target, n_bins=20):
        try:
            bins = np.quantile(feature_values, np.linspace(0, 1, n_bins + 1))
            bins = np.unique(bins)
            if len(bins) < 2:
                return 0.0
            digitized = np.digitize(feature_values, bins[1:-1])
        except (ValueError, IndexError):
            return 0.0
        return mutual_info_score(digitized, target) / math.log(2)

    mi_results = {}
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

    print(f"\n  Using {len(top_features)} features (MI >= 0.0005)")
    for fname, mi in sorted_mi[:10]:
        print(f"    {fname:<35} {mi:.6f} bits")
    _flush()

    # ------------------------------------------------------------------
    # Walk-forward: test all strategy × threshold × min_edge combinations
    # ------------------------------------------------------------------
    report: dict[str, Any] = {
        "meta": {
            "total_windows": len(window_data),
            "valid_windows": n_valid,
            "continuation_base_rate": round(cont_rate, 4),
            "entry_minute": ENTRY_MINUTE,
            "pricing_sensitivity": PRICE_SENSITIVITY,
            "position_size": POSITION_SIZE,
        },
        "strategies": {},
    }

    print(f"\n{'='*100}")
    print(f"  WALK-FORWARD STRATEGY SIMULATION")
    print(f"{'='*100}")

    # --- Strategy A: Symmetric Entry ---
    print(f"\n  --- STRATEGY A: Symmetric Min-3 Entry ---")
    print(f"  (Buy winning direction at estimated market price)")
    _flush()

    best_a = {"ev": -999, "config": ""}
    strat_a_results = {}

    for min_edge in MIN_EDGES:
        for threshold in THRESHOLDS:
            t0 = time.time()
            wf = walk_forward_fixed(
                window_data, all_feature_dicts, top_features,
                continuation_targets, valid_mask,
                entry_prices, loser_bids,
                threshold=threshold, min_edge=min_edge,
                strategy="symmetric",
            )
            elapsed = time.time() - t0

            if not wf:
                continue

            total_traded = sum(w["traded"] for w in wf)
            total_pnl = sum(w["net_pnl"] for w in wf)
            total_wins = sum(w["wins"] for w in wf)
            total_losses = sum(w["losses"] for w in wf)
            months_profitable = sum(1 for w in wf if w["profitable"])
            avg_pnl = total_pnl / total_traded if total_traded > 0 else 0
            win_rate = total_wins / total_traded if total_traded > 0 else 0
            avg_skip = sum(w["skip_rate"] for w in wf) / len(wf)

            config_key = f"thresh={threshold:.2f}_edge={min_edge:.2f}"
            strat_a_results[config_key] = {
                "threshold": threshold,
                "min_edge": min_edge,
                "periods": len(wf),
                "traded": total_traded,
                "wins": total_wins,
                "losses": total_losses,
                "win_rate": round(win_rate, 4),
                "total_pnl": round(total_pnl, 2),
                "avg_pnl": round(avg_pnl, 2),
                "months_profitable": months_profitable,
                "avg_skip_rate": round(avg_skip, 4),
                "details": wf,
            }

            marker = ""
            if avg_pnl > best_a["ev"]:
                best_a = {"ev": avg_pnl, "config": config_key}
                marker = " *** BEST"
            if total_traded > 0:
                print(f"  thresh={threshold:.2f} edge={min_edge:.2f}: "
                      f"{total_traded:>5} trades, {win_rate*100:>5.1f}% WR, "
                      f"EV ${avg_pnl:>+7.2f}/trade, "
                      f"PnL ${total_pnl:>+9,.0f}, "
                      f"months {months_profitable}/{len(wf)} "
                      f"({elapsed:.1f}s){marker}")
            _flush()

    report["strategies"]["symmetric"] = strat_a_results

    # --- Strategy B: Improved BuyBoth ---
    print(f"\n  --- STRATEGY B: Improved BuyBoth ---")
    print(f"  (Buy both at $0.50 pre-window, manage at min 3)")
    _flush()

    best_b = {"ev": -999, "config": ""}
    strat_b_results = {}

    for threshold in THRESHOLDS:
        t0 = time.time()
        wf = walk_forward_fixed(
            window_data, all_feature_dicts, top_features,
            continuation_targets, valid_mask,
            entry_prices, loser_bids,
            threshold=threshold, min_edge=0.0,
            strategy="buyboth",
        )
        elapsed = time.time() - t0

        if not wf:
            continue

        total_traded = sum(w["traded"] for w in wf)
        total_pnl = sum(w["net_pnl"] for w in wf)
        total_wins = sum(w["wins"] for w in wf)
        total_losses = sum(w["losses"] for w in wf)
        months_profitable = sum(1 for w in wf if w["profitable"])
        avg_pnl = total_pnl / total_traded if total_traded > 0 else 0
        win_rate = total_wins / total_traded if total_traded > 0 else 0
        avg_skip = sum(w["skip_rate"] for w in wf) / len(wf)

        config_key = f"thresh={threshold:.2f}"
        strat_b_results[config_key] = {
            "threshold": threshold,
            "periods": len(wf),
            "traded": total_traded,
            "wins": total_wins,
            "losses": total_losses,
            "win_rate": round(win_rate, 4),
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(avg_pnl, 2),
            "months_profitable": months_profitable,
            "avg_skip_rate": round(avg_skip, 4),
            "details": wf,
        }

        marker = ""
        if avg_pnl > best_b["ev"]:
            best_b = {"ev": avg_pnl, "config": config_key}
            marker = " *** BEST"
        if total_traded > 0:
            print(f"  thresh={threshold:.2f}: "
                  f"{total_traded:>5} trades, {win_rate*100:>5.1f}% WR, "
                  f"EV ${avg_pnl:>+7.2f}/trade, "
                  f"PnL ${total_pnl:>+9,.0f}, "
                  f"months {months_profitable}/{len(wf)} "
                  f"({elapsed:.1f}s){marker}")
        _flush()

    report["strategies"]["buyboth"] = strat_b_results

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n\n{'='*100}")
    print(f"  SUMMARY")
    print(f"{'='*100}")

    if best_a["config"] and best_a["config"] in strat_a_results:
        ba = strat_a_results[best_a["config"]]
        print(f"\n  BEST SYMMETRIC: {best_a['config']}")
        print(f"    Trades: {ba['traded']}, Win Rate: {ba['win_rate']*100:.1f}%")
        print(f"    EV/Trade: ${ba['avg_pnl']:+.2f}")
        print(f"    Total PnL: ${ba['total_pnl']:+,.0f}")
        print(f"    Months Profitable: {ba['months_profitable']}/{ba['periods']}")
        print(f"    Skip Rate: {ba['avg_skip_rate']*100:.1f}%")

        # Print monthly detail
        print(f"\n    {'Period':>10} {'Traded':>8} {'W':>4} {'L':>4} {'Acc':>8} {'PnL':>10} {'Prof':>6}")
        print(f"    {'-'*54}")
        for wf in ba["details"]:
            print(f"    {wf['period']:>10} {wf['traded']:>8} {wf['wins']:>4} {wf['losses']:>4} "
                  f"{wf['accuracy']*100:>7.1f}% ${wf['net_pnl']:>+9,.0f} "
                  f"{'Y' if wf['profitable'] else 'N':>6}")

    if best_b["config"] and best_b["config"] in strat_b_results:
        bb = strat_b_results[best_b["config"]]
        print(f"\n  BEST BUYBOTH: {best_b['config']}")
        print(f"    Trades: {bb['traded']}, Win Rate: {bb['win_rate']*100:.1f}%")
        print(f"    EV/Trade: ${bb['avg_pnl']:+.2f}")
        print(f"    Total PnL: ${bb['total_pnl']:+,.0f}")
        print(f"    Months Profitable: {bb['months_profitable']}/{bb['periods']}")
        print(f"    Skip Rate: {bb['avg_skip_rate']*100:.1f}%")

        print(f"\n    {'Period':>10} {'Traded':>8} {'W':>4} {'L':>4} {'Acc':>8} {'PnL':>10} {'Prof':>6}")
        print(f"    {'-'*54}")
        for wf in bb["details"]:
            print(f"    {wf['period']:>10} {wf['traded']:>8} {wf['wins']:>4} {wf['losses']:>4} "
                  f"{wf['accuracy']*100:>7.1f}% ${wf['net_pnl']:>+9,.0f} "
                  f"{'Y' if wf['profitable'] else 'N':>6}")

    _flush()

    # Save report
    # Strip details for compact report
    compact_report = {
        "meta": report["meta"],
        "best_symmetric": best_a,
        "best_buyboth": best_b,
        "symmetric_summary": {k: {kk: vv for kk, vv in v.items() if kk != "details"}
                              for k, v in strat_a_results.items()},
        "buyboth_summary": {k: {kk: vv for kk, vv in v.items() if kk != "details"}
                            for k, v in strat_b_results.items()},
    }

    with open(REPORT_JSON, "w") as f:
        json.dump(compact_report, f, indent=2, default=str)
    print(f"\n  Report saved to {REPORT_JSON}")
    print(f"\n  DONE.")


if __name__ == "__main__":
    main()
