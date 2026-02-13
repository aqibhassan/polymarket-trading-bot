"""BWO v5 — Momentum Continuation Filter.

KEY INSIGHT: We can't predict direction (50% base rate, MI=0 on 174 features).
But we CAN predict WHETHER early momentum continues (62% base rate at min 1).

Instead of: "Will BTC go Up or Down?" (impossible)
We ask:   "Given minute-1 moved Up, will the window settle Up?" (filterable)

Strategy:
  1. Observe first 1m candle direction at window open
  2. Only trade when model predicts HIGH continuation probability
  3. Enter in direction of minute-1 move
  4. Hold to settlement

Target: 80% continuation accuracy, <50% skip rate, walk-forward validated.

Uses ALL available data: spot OHLCV, futures taker flow, ETH cross-asset,
DVOL options, cross-exchange, PLUS new "signal quality" features from the
first candle itself.
"""

from __future__ import annotations

import json
import math
import sys
import time
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest_before_window import (
    ENTRY_PRICE,
    POSITION_SIZE,
    WindowData,
    bootstrap_ci,
    compute_all_features,
    compute_metrics,
    permutation_test,
    simulate_trade,
)
from scripts.fast_loader import FastCandle, group_into_15m_windows, load_csv_fast

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import mutual_info_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BTC_SPOT_CSV = PROJECT_ROOT / "data" / "btc_1m_2y.csv"
BTC_FUTURES_CSV = PROJECT_ROOT / "data" / "btc_futures_1m_2y.csv"
ETH_FUTURES_CSV = PROJECT_ROOT / "data" / "eth_futures_1m_2y.csv"
DVOL_CSV = PROJECT_ROOT / "data" / "deribit_dvol_1m.csv"
DERIBIT_PERP_CSV = PROJECT_ROOT / "data" / "deribit_btc_perp_1m.csv"
BINANCE_FUNDING_CSV = PROJECT_ROOT / "data" / "btc_funding_rate.csv"
DERIBIT_FUNDING_CSV = PROJECT_ROOT / "data" / "deribit_funding.csv"
REPORT_JSON = PROJECT_ROOT / "data" / "bwo_continuation_report.json"

MI_THRESHOLD = 0.005  # Lower bar — continuation target may have weak but usable signal
WF_TRAIN_MONTHS = 3
WF_TEST_MONTHS = 1

ENTRY_MINUTES = [1, 2, 3]  # Test early entries

# Threshold sweep for confidence filtering
CONF_THRESHOLDS = [round(0.50 + i * 0.01, 2) for i in range(46)]


def _flush() -> None:
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Data loaders (lightweight, reuse patterns)
# ---------------------------------------------------------------------------

def _parse_ts(raw: str) -> datetime | None:
    raw = raw.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def load_csv_to_dict(path: Path, fields: list[str]) -> tuple[list[dict], dict[datetime, int]]:
    """Generic CSV loader returning list of row dicts and timestamp index."""
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
            for field in fields:
                if field in row:
                    try:
                        d[field] = float(row[field])
                    except (ValueError, TypeError):
                        d[field] = 0.0
            lookup[ts] = len(rows)
            rows.append(d)
    return rows, lookup


# ---------------------------------------------------------------------------
# Signal Quality Features (from first N candles — NOT lookahead)
# ---------------------------------------------------------------------------

def compute_signal_quality_features(
    window_candles: list[FastCandle],
    entry_minute: int,
    all_candles: list[FastCandle],
    candle_idx: int,
) -> dict[str, float]:
    """Features that measure the QUALITY of the early momentum signal.

    These use the first entry_minute candles to assess HOW the move happened,
    not just WHAT direction it went. Strong, high-volume moves on a breakout
    may continue more reliably than weak, low-volume drifts.
    """
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

    # 1. Signal size: absolute cumulative return in first N minutes
    open_price = early[0].open
    close_price = early[-1].close
    if open_price > 0:
        cum_return = (close_price - open_price) / open_price
    else:
        cum_return = 0.0
    features["signal_size"] = abs(cum_return)

    # 2. Signal volume ratio: early volume vs recent 15-candle average
    early_vol = sum(c.volume for c in early)
    hist_start = max(0, candle_idx - 15)
    if candle_idx > hist_start:
        hist_vol = sum(all_candles[j].volume for j in range(hist_start, candle_idx))
        avg_hist_vol = hist_vol / (candle_idx - hist_start)
        features["signal_volume_ratio"] = (early_vol / entry_minute) / avg_hist_vol if avg_hist_vol > 0 else 1.0
    else:
        features["signal_volume_ratio"] = 1.0

    # 3. Signal consistency: fraction of early candles agreeing with overall direction
    if cum_return > 0:
        agreeing = sum(1 for c in early if c.close > c.open)
    elif cum_return < 0:
        agreeing = sum(1 for c in early if c.close < c.open)
    else:
        agreeing = 0
    features["signal_consistency"] = agreeing / entry_minute

    # 4. Body ratio: average (body / total range) of early candles
    body_ratios: list[float] = []
    for c in early:
        total_range = c.high - c.low
        body = abs(c.close - c.open)
        if total_range > 0:
            body_ratios.append(body / total_range)
    features["signal_body_ratio"] = sum(body_ratios) / len(body_ratios) if body_ratios else 0.0

    # 5. Signal vs ATR: move size relative to recent ATR(14)
    if candle_idx >= 15:
        atr_sum = 0.0
        for j in range(candle_idx - 14, candle_idx):
            c = all_candles[j]
            prev_close = all_candles[j - 1].close
            tr = max(c.high - c.low, abs(c.high - prev_close), abs(c.low - prev_close))
            atr_sum += tr
        atr = atr_sum / 14.0
        if atr > 0:
            features["signal_vs_atr"] = abs(close_price - open_price) / atr
        else:
            features["signal_vs_atr"] = 0.0
    else:
        features["signal_vs_atr"] = 0.0

    # 6. Volume surge: is early volume unusually high?
    if candle_idx >= 60:
        hist_vols = [all_candles[j].volume for j in range(candle_idx - 60, candle_idx)]
        hist_vols.sort()
        p90 = hist_vols[int(0.9 * len(hist_vols))]
        features["signal_volume_surge"] = 1.0 if (early_vol / entry_minute) > p90 else 0.0
    else:
        features["signal_volume_surge"] = 0.0

    # 7. Candle range: total high-low range of first candle relative to recent
    if entry_minute >= 1:
        first_range = early[0].high - early[0].low
        if candle_idx >= 15:
            recent_ranges = [all_candles[j].high - all_candles[j].low for j in range(candle_idx - 15, candle_idx)]
            avg_range = sum(recent_ranges) / len(recent_ranges) if recent_ranges else 0.001
            features["signal_candle_range"] = first_range / avg_range if avg_range > 0 else 1.0
        else:
            features["signal_candle_range"] = 1.0
    else:
        features["signal_candle_range"] = 1.0

    # 8. Close position: where did the first candle close within its range?
    # Close near high (for up candle) = strong, close near low = weak
    if entry_minute >= 1:
        c = early[0]
        total_range = c.high - c.low
        if total_range > 0:
            features["signal_close_position"] = (c.close - c.low) / total_range
        else:
            features["signal_close_position"] = 0.5
    else:
        features["signal_close_position"] = 0.5

    return features


# ---------------------------------------------------------------------------
# Compute taker features (simplified — reuse from v3 if available)
# ---------------------------------------------------------------------------

def compute_taker_features_simple(
    ci: int,
    futures_data: list[dict],
    futures_lookup: dict[datetime, int],
    all_candles: list[FastCandle],
) -> dict[str, float]:
    """Simplified taker features using futures data."""
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
# MI calculation
# ---------------------------------------------------------------------------

def compute_mi_bits(feature_values: np.ndarray, target: np.ndarray, n_bins: int = 20) -> float:
    try:
        bins = np.quantile(feature_values, np.linspace(0, 1, n_bins + 1))
        bins = np.unique(bins)
        if len(bins) < 2:
            return 0.0
        digitized = np.digitize(feature_values, bins[1:-1])
    except (ValueError, IndexError):
        return 0.0
    mi = mutual_info_score(digitized, target)
    return mi / math.log(2)


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------

def walk_forward_continuation(
    window_data: list[WindowData],
    all_feature_dicts: list[dict[str, float]],
    feature_names: list[str],
    continuation_targets: list[int],
    valid_mask: list[bool],
    threshold: float,
) -> list[dict[str, Any]]:
    """Walk-forward for continuation model with confidence threshold."""
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

        # Apply threshold
        probas = model.predict_proba(X_test)[:, 1]
        mask = probas >= threshold
        n_traded = mask.sum()

        if n_traded < 5:
            results.append({
                "period": test_month, "accuracy": 0.0, "traded": 0,
                "total": len(test_indices), "skip_rate": 1.0,
                "net_pnl": 0.0, "profitable": False,
            })
            continue

        filtered_correct = (y_test[mask] == 1).sum()
        accuracy = filtered_correct / n_traded
        skip_rate = 1.0 - n_traded / len(test_indices)

        # PnL: trade in early direction with continuation confidence
        net_pnl = 0.0
        test_idx_arr = [test_indices[j] for j in range(len(test_indices)) if mask[j]]
        for idx in test_idx_arr:
            wd = window_data[idx]
            feats = all_feature_dicts[idx]
            early_dir = feats.get("early_direction", 0.0)
            direction = "Up" if early_dir > 0 else "Down"
            # Entry price at minute 1-2 is slightly above $0.50
            entry_p = 0.50 + abs(feats.get("early_cum_return", 0)) * 0.3
            entry_p = min(entry_p, 0.60)
            _, _, pn, _ = simulate_trade(direction, wd.resolution, entry_price=entry_p)
            net_pnl += pn

        results.append({
            "period": test_month,
            "accuracy": float(accuracy),
            "traded": int(n_traded),
            "total": len(test_indices),
            "skip_rate": float(skip_rate),
            "net_pnl": float(net_pnl),
            "profitable": net_pnl > 0,
        })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 100)
    print("  BWO v5 — MOMENTUM CONTINUATION FILTER")
    print("  Target: predict WHEN early momentum continues (not direction)")
    print("  Base rate: ~62% continuation at min 1 → filter to ≥80%")
    print("=" * 100)
    _flush()

    # ------------------------------------------------------------------
    # Load all data
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

    # Futures (for taker features)
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
    else:
        print(f"  BTC futures: NOT FOUND (skipping taker features)")

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
    # Per entry-minute analysis
    # ------------------------------------------------------------------
    report: dict[str, Any] = {
        "meta": {
            "total_windows": len(window_data),
            "base_rate_up_pct": round(up_count / len(window_data) * 100, 2),
        },
        "entry_minutes": {},
    }

    for entry_minute in ENTRY_MINUTES:
        print(f"\n{'='*100}")
        print(f"  ENTRY MINUTE {entry_minute}")
        print(f"{'='*100}")
        _flush()

        # Compute features for all windows at this entry minute
        print(f"  Computing features (entry_minute={entry_minute})...")
        _flush()
        t0 = time.time()

        all_feature_dicts: list[dict[str, float]] = []
        continuation_targets: list[int] = []
        valid_mask: list[bool] = []

        for i, wd in enumerate(window_data):
            ci = wd.candle_idx
            w = windows[wd.window_idx]

            # Original 31 features (pre-window)
            original = compute_all_features(wd, windows, all_candles, entry_minute=entry_minute)

            # Signal quality features (from first N candles)
            signal = compute_signal_quality_features(w, entry_minute, all_candles, ci)

            # Taker features
            taker = compute_taker_features_simple(ci, futures_data, futures_lookup, all_candles)

            # DVOL features (simplified — just level and change)
            dvol_feats: dict[str, float] = {}
            if dvol_timestamps:
                di = bisect_right(dvol_timestamps, wd.timestamp) - 1
                if di >= 5:
                    dvol_now = dvol_data[di]["close"]
                    dvol_feats["dvol_level"] = dvol_now
                    dvol_feats["dvol_change_5"] = (dvol_now - dvol_data[di - 5]["close"]) / dvol_data[di - 5]["close"] if dvol_data[di - 5]["close"] > 0 else 0.0
                    # Z-score
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

            # ETH cross-asset (simplified)
            eth_feats: dict[str, float] = {}
            ei = eth_lookup.get(wd.timestamp)
            bi_f = futures_lookup.get(wd.timestamp)
            if ei is not None and ei >= 15 and bi_f is not None and bi_f >= 15:
                eth_close = eth_data[ei - 1]["close"]
                eth_open = eth_data[ei - 15]["close"]
                eth_ret = (eth_close - eth_open) / eth_open if eth_open > 0 else 0.0
                btc_close_f = futures_data[bi_f - 1].get("volume", 0)

                eth_feats["eth_momentum_15"] = eth_ret
                # ETH taker alignment
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
            # Round to nearest hour for deribit lookup
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

            # Combine all
            combined = {**original, **signal, **taker, **dvol_feats, **eth_feats, **cross_feats}
            all_feature_dicts.append(combined)

            # Continuation target
            early_dir = original.get("early_direction", 0.0)
            if early_dir == 0.0:
                continuation_targets.append(0)
                valid_mask.append(False)
            else:
                # Did early momentum continue?
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

        # Continuation stats
        n_valid = sum(valid_mask)
        n_continued = sum(1 for i in range(len(window_data)) if valid_mask[i] and continuation_targets[i] == 1)
        cont_rate = n_continued / n_valid if n_valid > 0 else 0
        print(f"\n  Valid windows (early_dir != 0): {n_valid:,} / {len(window_data):,}")
        print(f"  Continuation base rate: {cont_rate*100:.1f}%")
        _flush()

        # ------------------------------------------------------------------
        # Feature names
        # ------------------------------------------------------------------
        # Get all unique feature names from a sample
        all_feature_names = sorted(set().union(*(d.keys() for d in all_feature_dicts[:100])))

        # ------------------------------------------------------------------
        # PHASE 0: MI CHECK on continuation target
        # ------------------------------------------------------------------
        print(f"\n  --- MI Check (Continuation Target) ---")
        print(f"  {'Feature':<40} {'MI (bits)':>12}")
        print(f"  {'-'*55}")
        _flush()

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

        # Sort by MI descending
        sorted_mi = sorted(mi_results.items(), key=lambda x: x[1], reverse=True)
        for fname, mi in sorted_mi[:25]:
            marker = " ***" if mi >= MI_THRESHOLD else ""
            print(f"  {fname:<40} {mi:>12.6f}{marker}")

        max_mi = sorted_mi[0][1] if sorted_mi else 0.0
        above_threshold = sum(1 for _, mi in sorted_mi if mi >= MI_THRESHOLD)
        print(f"\n  Max MI: {max_mi:.6f} bits")
        print(f"  Features above {MI_THRESHOLD}: {above_threshold}")
        _flush()

        # ------------------------------------------------------------------
        # ML: HistGradientBoosting on continuation target
        # ------------------------------------------------------------------
        print(f"\n  --- ML Model (Continuation Prediction) ---")
        _flush()

        # Use top features by MI (avoid noisy features)
        top_features = [f for f, mi in sorted_mi if mi >= 0.0005][:40]
        if len(top_features) < 5:
            top_features = [f for f, _ in sorted_mi[:20]]

        print(f"  Using {len(top_features)} features")

        # Build matrices
        split_idx_valid = int(len(valid_indices) * 0.80)
        train_idx = valid_indices[:split_idx_valid]
        test_idx = valid_indices[split_idx_valid:]

        X_train = np.array([[all_feature_dicts[i].get(f, 0.0) for f in top_features] for i in train_idx])
        y_train = np.array([continuation_targets[i] for i in train_idx])
        X_test = np.array([[all_feature_dicts[i].get(f, 0.0) for f in top_features] for i in test_idx])
        y_test = np.array([continuation_targets[i] for i in test_idx])

        print(f"  Train: {X_train.shape} | Test: {X_test.shape}")
        print(f"  Train cont rate: {y_train.mean()*100:.1f}% | Test cont rate: {y_test.mean()*100:.1f}%")
        _flush()

        # Train model
        t0 = time.time()
        model = HistGradientBoostingClassifier(
            max_iter=500, max_depth=5, learning_rate=0.05,
            min_samples_leaf=80, l2_regularization=1.0,
            max_bins=128, random_state=42,
        )
        model.fit(X_train, y_train)
        elapsed = time.time() - t0

        # Raw accuracy
        raw_preds = model.predict(X_test)
        raw_accuracy = float((raw_preds == y_test).mean())
        print(f"  Raw accuracy: {raw_accuracy*100:.1f}% ({elapsed:.1f}s)")

        # Feature importance (permutation-based fallback)
        try:
            importances = model.feature_importances_
        except AttributeError:
            from sklearn.inspection import permutation_importance
            perm = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
            importances = perm.importances_mean
        feat_imp = sorted(zip(top_features, importances.tolist()), key=lambda x: x[1], reverse=True)
        print(f"\n  Top 10 Feature Importances:")
        for fname, imp in feat_imp[:10]:
            print(f"    {fname:<35} {imp:.4f}")
        _flush()

        # ------------------------------------------------------------------
        # Threshold sweep
        # ------------------------------------------------------------------
        print(f"\n  --- Confidence Threshold Sweep ---")
        print(f"  {'Thresh':>8} {'Accuracy':>10} {'SkipRate':>10} {'Traded':>8} {'ContRate':>10}")
        print(f"  {'-'*50}")
        _flush()

        probas = model.predict_proba(X_test)[:, 1]
        sweep_results: list[dict[str, Any]] = []
        best_result: dict[str, Any] = {"accuracy": 0.0, "skip_rate": 1.0, "trades": 0, "threshold": 0.5}

        for thresh in CONF_THRESHOLDS:
            mask = probas >= thresh
            n_traded = int(mask.sum())
            if n_traded < 20:
                continue

            acc = float(y_test[mask].mean())
            skip = 1.0 - n_traded / len(y_test)

            sweep_results.append({
                "threshold": thresh, "accuracy": acc,
                "skip_rate": skip, "trades": n_traded,
            })

            # Print every 5th + any ≥80%
            if int(round(thresh * 100)) % 5 == 0 or acc >= 0.80:
                marker = " <-- TARGET" if acc >= 0.80 and skip < 0.50 else (" *" if acc >= 0.80 else "")
                print(f"  {thresh:>8.2f} {acc*100:>9.1f}% {skip*100:>9.1f}% {n_traded:>8,} {cont_rate*100:>9.1f}%{marker}")

            # Track best meeting targets
            if acc >= 0.80 and skip < 0.50 and n_traded > best_result.get("trades", 0):
                best_result = {
                    "accuracy": acc, "skip_rate": skip,
                    "trades": n_traded, "threshold": thresh,
                }

        _flush()

        if best_result["trades"] > 0:
            print(f"\n  BEST: thresh={best_result['threshold']:.2f} → "
                  f"acc {best_result['accuracy']*100:.1f}%, "
                  f"skip {best_result['skip_rate']*100:.1f}%, "
                  f"trades {best_result['trades']:,}")
        else:
            # Find best accuracy regardless of skip rate
            if sweep_results:
                best_any = max(sweep_results, key=lambda r: r["accuracy"])
                print(f"\n  BEST (any skip): thresh={best_any['threshold']:.2f} → "
                      f"acc {best_any['accuracy']*100:.1f}%, "
                      f"skip {best_any['skip_rate']*100:.1f}%, "
                      f"trades {best_any['trades']:,}")
            print(f"  NO threshold meets 80% acc + <50% skip")
        _flush()

        # ------------------------------------------------------------------
        # Walk-forward validation on best threshold
        # ------------------------------------------------------------------
        wf_threshold = best_result["threshold"] if best_result["trades"] > 0 else 0.65

        print(f"\n  --- Walk-Forward Validation (threshold={wf_threshold:.2f}) ---")
        _flush()
        t0 = time.time()
        wf_results = walk_forward_continuation(
            window_data, all_feature_dicts, top_features,
            continuation_targets, valid_mask, wf_threshold,
        )
        elapsed = time.time() - t0

        if wf_results:
            print(f"  {'Period':>10} {'Acc':>8} {'Traded':>8} {'Total':>8} {'Skip':>8} {'PnL':>10} {'Prof':>6}")
            print(f"  {'-'*62}")
            for wf in wf_results:
                print(f"  {wf['period']:>10} {wf['accuracy']*100:>7.1f}% {wf['traded']:>8} "
                      f"{wf['total']:>8} {wf['skip_rate']*100:>7.1f}% "
                      f"${wf['net_pnl']:>+8,.0f} {'Y' if wf['profitable'] else 'N':>6}")

            wf_accs = [w["accuracy"] for w in wf_results if w["traded"] > 0]
            wf_mean = sum(wf_accs) / len(wf_accs) if wf_accs else 0.0
            wf_prof = sum(1 for w in wf_results if w["profitable"])
            wf_pnl = sum(w["net_pnl"] for w in wf_results)
            print(f"\n  WF: mean acc {wf_mean*100:.1f}%, "
                  f"profitable {wf_prof}/{len(wf_results)}, "
                  f"total PnL ${wf_pnl:+,.0f} ({elapsed:.1f}s)")
        else:
            wf_mean = 0.0
            wf_pnl = 0.0
            print(f"  No walk-forward periods ({elapsed:.1f}s)")
        _flush()

        # ------------------------------------------------------------------
        # Bootstrap CI and permutation test
        # ------------------------------------------------------------------
        if best_result["trades"] > 0:
            mask = probas >= best_result["threshold"]
            filtered_y = y_test[mask]
            correct_list = [bool(v) for v in filtered_y]
            ci_mean, ci_lo, ci_hi = bootstrap_ci(correct_list)
            print(f"\n  Bootstrap 95% CI: [{ci_lo*100:.1f}%, {ci_hi*100:.1f}%]")

            preds_str = ["Up" if v else "Down" for v in filtered_y]
            actuals_str = ["Up" if continuation_targets[test_idx[j]] else "Down"
                          for j in range(len(mask)) if mask[j]]
            p_val = permutation_test(preds_str, actuals_str)
            print(f"  Permutation p-value: {p_val:.4f}")
        _flush()

        # Store results
        minute_report: dict[str, Any] = {
            "entry_minute": entry_minute,
            "n_valid": n_valid,
            "continuation_base_rate": round(cont_rate, 4),
            "top_mi_features": [(f, round(mi, 6)) for f, mi in sorted_mi[:15]],
            "max_mi": round(max_mi, 6),
            "features_above_threshold": above_threshold,
            "raw_ml_accuracy": round(raw_accuracy, 4),
            "feature_importances": [(f, round(imp, 4)) for f, imp in feat_imp[:15]],
            "sweep_results": sweep_results,
            "best_result": best_result,
            "walk_forward": {
                "threshold": wf_threshold,
                "mean_accuracy": round(wf_mean, 4) if wf_results else 0.0,
                "total_pnl": round(wf_pnl, 2) if wf_results else 0.0,
                "details": wf_results,
            },
        }
        report["entry_minutes"][str(entry_minute)] = minute_report

    # ------------------------------------------------------------------
    # OVERALL SUMMARY
    # ------------------------------------------------------------------
    print(f"\n\n{'='*100}")
    print(f"  OVERALL SUMMARY")
    print(f"{'='*100}")

    print(f"\n  {'Entry':>8} {'ContRate':>10} {'MaxMI':>10} {'RawAcc':>10} {'BestAcc':>10} {'Skip':>10} {'WF Mean':>10}")
    print(f"  {'-'*72}")

    best_overall = None
    for em in ENTRY_MINUTES:
        key = str(em)
        if key not in report["entry_minutes"]:
            continue
        mr = report["entry_minutes"][key]
        br = mr.get("best_result", {})
        wf = mr.get("walk_forward", {})

        print(f"  min {em:>4} {mr['continuation_base_rate']*100:>9.1f}% "
              f"{mr['max_mi']:>10.6f} {mr['raw_ml_accuracy']*100:>9.1f}% "
              f"{br.get('accuracy', 0)*100:>9.1f}% {br.get('skip_rate', 1)*100:>9.1f}% "
              f"{wf.get('mean_accuracy', 0)*100:>9.1f}%")

        if br.get("accuracy", 0) >= 0.80 and br.get("skip_rate", 1) < 0.50:
            if best_overall is None or br["trades"] > best_overall.get("trades", 0):
                best_overall = {**br, "entry_minute": em}

    if best_overall:
        print(f"\n  TARGET MET: entry min {best_overall['entry_minute']}, "
              f"acc {best_overall['accuracy']*100:.1f}%, "
              f"skip {best_overall['skip_rate']*100:.1f}%")
        report["conclusion"] = "TARGET MET"
    else:
        print(f"\n  TARGET NOT MET (80% acc + <50% skip)")
        # Show best achievable
        best_any_acc = 0.0
        for em in ENTRY_MINUTES:
            key = str(em)
            if key in report["entry_minutes"]:
                for sr in report["entry_minutes"][key].get("sweep_results", []):
                    if sr["accuracy"] > best_any_acc:
                        best_any_acc = sr["accuracy"]
        print(f"  Best achievable accuracy: {best_any_acc*100:.1f}%")
        report["conclusion"] = f"Best accuracy: {best_any_acc*100:.1f}%"

    with open(REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved: {REPORT_JSON}")
    print(f"\n{'='*100}")
    _flush()


if __name__ == "__main__":
    main()
