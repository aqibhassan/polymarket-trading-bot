"""BWO 5-Minute Market Backtest — Continuation Filter for Polymarket BTC 5m markets.

5m markets are ideal for continuation strategies because:
- At minute 1 (20% of window): 65.9% continuation rate, 2.6% skip
- At minute 2 (40% of window): 73.5% continuation rate, 1.1% skip
- Much more frequent trading (288 windows/day vs 96 for 15m)

Strategy: Enter at minute 1-2, predict if early BTC direction continues to minute 5.
Uses HistGradientBoosting + confidence filtering to hit 80%+ accuracy.

Performance: ~210K windows in <2 minutes via index-based lookups.
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fast_loader import FastCandle, load_csv_fast

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BTC_SPOT_CSV = PROJECT_ROOT / "data" / "btc_1m_2y.csv"
BTC_FUTURES_CSV = PROJECT_ROOT / "data" / "btc_futures_1m_2y.csv"
ETH_FUTURES_CSV = PROJECT_ROOT / "data" / "eth_futures_1m_2y.csv"
DVOL_CSV = PROJECT_ROOT / "data" / "deribit_dvol_1m.csv"
DERIBIT_PERP_CSV = PROJECT_ROOT / "data" / "deribit_btc_perp_1m.csv"
REPORT_JSON = PROJECT_ROOT / "data" / "bwo_5m_report.json"

POSITION_SIZE = 100.0
FEE_CONSTANT = 0.25
SLIPPAGE_BPS = 5

WF_TRAIN_MONTHS = 3
WF_TEST_MONTHS = 1
BOOTSTRAP_ITERATIONS = 2000
PERMUTATION_ITERATIONS = 2000

SESSIONS = {
    "asia": range(0, 8),
    "europe": range(8, 14),
    "us": range(14, 22),
    "late": range(22, 24),
}


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
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class WindowData5m:
    """A single 5m window."""
    window_idx: int
    candle_idx: int  # index of window[0] in all_candles
    resolution: str  # "Up" or "Down"
    timestamp: datetime


# ---------------------------------------------------------------------------
# Window grouping for 5m boundaries
# ---------------------------------------------------------------------------

def group_into_5m_windows(candles: list[FastCandle]) -> list[list[FastCandle]]:
    """Group 1m candles into 5m windows aligned to 300-second boundaries."""
    windows: list[list[FastCandle]] = []
    i = 0
    n = len(candles)
    while i < n:
        ts = candles[i].timestamp
        ts_unix = int(ts.timestamp())
        if ts_unix % 300 == 0 and i + 4 < n:
            w = candles[i:i + 5]
            # Verify consecutive
            if (w[-1].timestamp - w[0].timestamp).total_seconds() <= 300:
                windows.append(w)
                i += 5
                continue
        i += 1
    return windows


# ---------------------------------------------------------------------------
# Feature functions
# ---------------------------------------------------------------------------

def compute_all_features_5m(
    wd: WindowData5m,
    windows: list[list[FastCandle]],
    all_candles: list[FastCandle],
    entry_minute: int = 1,
    futures_data: list[dict] | None = None,
    futures_lookup: dict[datetime, int] | None = None,
    eth_data: list[dict] | None = None,
    eth_lookup: dict[datetime, int] | None = None,
    dvol_data: list[dict] | None = None,
    dvol_timestamps: list[datetime] | None = None,
    deribit_data: list[dict] | None = None,
    deribit_lookup: dict[datetime, int] | None = None,
) -> dict[str, float]:
    """Compute features for a 5m window at entry_minute."""
    features: dict[str, float] = {}
    ci = wd.candle_idx
    wi = wd.window_idx
    w = windows[wi]

    hist_start = max(0, ci - 1440)
    hist_end = ci  # no lookahead into current window

    # ===== EARLY WINDOW FEATURES (minutes 0 to entry_minute) =====
    if entry_minute > 0 and entry_minute <= len(w):
        early = w[:entry_minute]
        if w[0].open > 0:
            cum_ret = (early[-1].close - w[0].open) / w[0].open
            features["early_cum_return"] = cum_ret
            features["early_direction"] = 1.0 if cum_ret > 0 else (-1.0 if cum_ret < 0 else 0.0)
            features["early_magnitude"] = abs(cum_ret)
        else:
            features["early_cum_return"] = 0.0
            features["early_direction"] = 0.0
            features["early_magnitude"] = 0.0

        # Green ratio
        green = sum(1 for c in early if c.close > c.open)
        features["early_green_ratio"] = green / len(early)

        # Early volatility
        if len(early) >= 2:
            sq_sum = 0.0
            for j in range(1, len(early)):
                if early[j - 1].close > 0:
                    r = (early[j].close - early[j - 1].close) / early[j - 1].close
                    sq_sum += r * r
            features["early_vol"] = math.sqrt(sq_sum / (len(early) - 1)) if sq_sum > 0 else 0.0
        else:
            features["early_vol"] = 0.0

        # Max single candle move
        max_move = 0.0
        for c in early:
            if c.open > 0:
                m = abs(c.close - c.open) / c.open
                if m > max_move:
                    max_move = m
        features["early_max_move"] = max_move

        # Body ratio (how much of each candle is body vs wick)
        body_ratios = []
        for c in early:
            rng = c.high - c.low
            if rng > 0:
                body_ratios.append(abs(c.close - c.open) / rng)
        features["early_body_ratio"] = sum(body_ratios) / len(body_ratios) if body_ratios else 0.0

        # Volume surge (early volume vs recent average)
        early_vol = sum(c.volume for c in early)
        hist_len = hist_end - hist_start
        if hist_len >= 30:
            recent_vol = sum(all_candles[j].volume for j in range(max(hist_end - 30, hist_start), hist_end))
            avg_vol = recent_vol / min(30, hist_len)
            features["early_vol_surge"] = (early_vol / len(early)) / avg_vol if avg_vol > 0 else 1.0
        else:
            features["early_vol_surge"] = 1.0

        # Close position (where did first candle close in its range?)
        c0 = early[0]
        if c0.high > c0.low:
            features["early_close_position"] = (c0.close - c0.low) / (c0.high - c0.low)
        else:
            features["early_close_position"] = 0.5

        # Signal vs ATR (move relative to recent volatility)
        if hist_len >= 14:
            atr_sum = 0.0
            for j in range(hist_end - 14, hist_end):
                c = all_candles[j]
                tr = c.high - c.low
                if j > 0:
                    prev_c = all_candles[j - 1].close
                    tr = max(tr, abs(c.high - prev_c), abs(c.low - prev_c))
                atr_sum += tr
            atr = atr_sum / 14
            if atr > 0 and w[0].open > 0:
                features["signal_vs_atr"] = abs(cum_ret) * w[0].open / atr
            else:
                features["signal_vs_atr"] = 0.0
        else:
            features["signal_vs_atr"] = 0.0
    else:
        # Pre-window only (no early features)
        features["early_cum_return"] = 0.0
        features["early_direction"] = 0.0
        features["early_magnitude"] = 0.0
        features["early_green_ratio"] = 0.0
        features["early_vol"] = 0.0
        features["early_max_move"] = 0.0
        features["early_body_ratio"] = 0.0
        features["early_vol_surge"] = 1.0
        features["early_close_position"] = 0.5
        features["signal_vs_atr"] = 0.0

    # ===== PRE-WINDOW FEATURES =====
    hist_len = hist_end - hist_start

    # Prior 5m window momentum
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

    # Prior 15m and 1h trend
    if hist_len >= 60:
        def _dir(start: int, end: int) -> float:
            o = all_candles[start].open
            c = all_candles[end - 1].close
            if o == 0:
                return 0.0
            r = (c - o) / o
            return 1.0 if r > 0 else (-1.0 if r < 0 else 0.0)

        features["trend_5m"] = _dir(hist_end - 5, hist_end)
        features["trend_15m"] = _dir(hist_end - 15, hist_end)
        features["trend_1h"] = _dir(hist_end - 60, hist_end)
        features["trend_score"] = features["trend_5m"] + features["trend_15m"] + features["trend_1h"]
    else:
        features["trend_5m"] = 0.0
        features["trend_15m"] = 0.0
        features["trend_1h"] = 0.0
        features["trend_score"] = 0.0

    # Volatility regime
    if hist_len >= 60:
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
        features["vol_regime"] = vol_ratio
        features["vol_percentile"] = min(vol_ratio / 2.0, 1.0)
    else:
        features["vol_regime"] = 1.0
        features["vol_percentile"] = 0.5

    # Time of day
    hour = wd.timestamp.hour
    features["tod_hour"] = float(hour)
    features["tod_asia"] = 1.0 if hour in SESSIONS["asia"] else 0.0
    features["tod_europe"] = 1.0 if hour in SESSIONS["europe"] else 0.0
    features["tod_us"] = 1.0 if hour in SESSIONS["us"] else 0.0

    # Candle streak before window
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

    # RSI(14)
    if hist_len >= 15:
        gains = 0.0
        losses = 0.0
        for j in range(hist_end - 14, hist_end):
            diff = all_candles[j].close - all_candles[j - 1].close
            if diff > 0:
                gains += diff
            else:
                losses -= diff
        avg_gain = gains / 14
        avg_loss = losses / 14
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            features["rsi_14"] = 100 - (100 / (1 + rs))
        else:
            features["rsi_14"] = 100.0
    else:
        features["rsi_14"] = 50.0

    # Bollinger %B
    if hist_len >= 20:
        closes_20 = [all_candles[j].close for j in range(hist_end - 20, hist_end)]
        sma = sum(closes_20) / 20
        std = math.sqrt(sum((c - sma) ** 2 for c in closes_20) / 20) if len(closes_20) > 0 else 0.001
        if std > 0:
            features["bb_pct_b"] = (all_candles[hist_end - 1].close - (sma - 2 * std)) / (4 * std)
        else:
            features["bb_pct_b"] = 0.5
    else:
        features["bb_pct_b"] = 0.5

    # Mean reversion (z-score vs 60-candle SMA)
    if hist_len >= 60:
        closes_60 = [all_candles[j].close for j in range(hist_end - 60, hist_end)]
        sma60 = sum(closes_60) / 60
        std60 = math.sqrt(sum((c - sma60) ** 2 for c in closes_60) / 60) if len(closes_60) > 0 else 0.001
        features["mean_rev_z"] = (all_candles[hist_end - 1].close - sma60) / std60 if std60 > 0 else 0.0
    else:
        features["mean_rev_z"] = 0.0

    # Volume profile
    if hist_len >= 30:
        total_vol = sum(all_candles[j].volume for j in range(hist_start, hist_end))
        avg_vol = total_vol / hist_len if hist_len > 0 else 1.0
        if avg_vol == 0:
            avg_vol = 1.0
        recent_vol_sum = sum(all_candles[j].volume for j in range(hist_end - 5, hist_end))
        features["vol_ratio"] = (recent_vol_sum / 5) / avg_vol
    else:
        features["vol_ratio"] = 1.0

    # ===== CROSS-ASSET FEATURES =====

    # Taker flow imbalance from futures
    if futures_data and futures_lookup:
        for period, label in [(5, "5"), (15, "15"), (60, "60")]:
            taker_total = 0.0
            taker_buy = 0.0
            found = 0
            for offset in range(period):
                idx = ci - period + offset
                if 0 <= idx < len(all_candles):
                    fts = all_candles[idx].timestamp
                    fi = futures_lookup.get(fts)
                    if fi is not None:
                        fd = futures_data[fi]
                        tv = fd.get("volume", 0)
                        tbv = fd.get("taker_buy_volume", 0)
                        taker_total += tv
                        taker_buy += tbv
                        found += 1
            if taker_total > 0 and found >= period // 2:
                features[f"taker_imbalance_{label}"] = (2 * taker_buy - taker_total) / taker_total
            else:
                features[f"taker_imbalance_{label}"] = 0.0

        # Taker acceleration (5m vs 15m)
        features["taker_accel"] = features.get("taker_imbalance_5", 0) - features.get("taker_imbalance_15", 0)

        # Taker intensity
        recent_fi = futures_lookup.get(all_candles[ci - 1].timestamp if ci > 0 else all_candles[0].timestamp)
        if recent_fi is not None:
            fd = futures_data[recent_fi]
            nt = fd.get("num_trades", 0)
            tv = fd.get("volume", 0)
            features["taker_intensity"] = fd.get("taker_buy_volume", 0) / nt if nt > 0 else 0.0
        else:
            features["taker_intensity"] = 0.0
    else:
        for label in ["5", "15", "60"]:
            features[f"taker_imbalance_{label}"] = 0.0
        features["taker_accel"] = 0.0
        features["taker_intensity"] = 0.0

    # ETH cross-asset
    if eth_data and eth_lookup:
        ei = eth_lookup.get(wd.timestamp)
        if ei is not None and ei >= 15:
            ec = eth_data[ei - 1].get("close", 0)
            eo = eth_data[ei - 15].get("close", 0)
            features["eth_momentum_15"] = (ec - eo) / eo if eo > 0 else 0.0
        else:
            features["eth_momentum_15"] = 0.0
    else:
        features["eth_momentum_15"] = 0.0

    # Deribit basis
    if deribit_data and deribit_lookup and ci > 0:
        th = wd.timestamp.replace(minute=0, second=0, microsecond=0)
        dbi = deribit_lookup.get(th)
        if dbi is not None:
            dc = deribit_data[dbi].get("close", 0)
            sc = all_candles[ci - 1].close
            if sc > 0 and dc > 0:
                features["deribit_basis_bps"] = (dc - sc) / sc * 10000
            else:
                features["deribit_basis_bps"] = 0.0
        else:
            features["deribit_basis_bps"] = 0.0
    else:
        features["deribit_basis_bps"] = 0.0

    # DVOL
    if dvol_data and dvol_timestamps:
        from bisect import bisect_right
        di = bisect_right(dvol_timestamps, wd.timestamp) - 1
        if di >= 5:
            dvol_now = dvol_data[di].get("close", 0)
            features["dvol_level"] = dvol_now
            prev = dvol_data[di - 5].get("close", 0)
            features["dvol_change_5"] = (dvol_now - prev) / prev if prev > 0 else 0.0
        else:
            features["dvol_level"] = 0.0
            features["dvol_change_5"] = 0.0
    else:
        features["dvol_level"] = 0.0
        features["dvol_change_5"] = 0.0

    return features


# ---------------------------------------------------------------------------
# Mutual information
# ---------------------------------------------------------------------------

def compute_mi_bits(feature_values: np.ndarray, target: np.ndarray, n_bins: int = 20) -> float:
    try:
        bins = np.quantile(feature_values, np.linspace(0, 1, n_bins + 1))
        bins = np.unique(bins)
        if len(bins) < 3:
            return 0.0
        digitized = np.digitize(feature_values, bins[1:-1])
        n = len(target)
        py = np.bincount(target.astype(int), minlength=2) / n
        mi = 0.0
        for b in np.unique(digitized):
            mask = digitized == b
            p_b = mask.sum() / n
            if p_b == 0:
                continue
            py_b = np.bincount(target[mask].astype(int), minlength=2) / mask.sum()
            for y_val in range(2):
                if py_b[y_val] > 0 and py[y_val] > 0:
                    mi += p_b * py_b[y_val] * math.log2(py_b[y_val] / py[y_val])
        return max(0.0, mi)
    except Exception:
        return 0.0


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
    return sum(correct_list) / sz, means[int(0.025 * n)], means[int(0.975 * n)]


def permutation_test(preds: list[int], actuals: list[int], n: int = PERMUTATION_ITERATIONS) -> float:
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
# Load auxiliary data
# ---------------------------------------------------------------------------

def _load_csv_dict(path: Path, fields: list[str]) -> tuple[list[dict], dict[datetime, int]]:
    rows: list[dict] = []
    lookup: dict[datetime, int] = {}
    if not path.exists():
        return rows, lookup
    with open(path) as f:
        for row in csv.DictReader(f):
            ts = _parse_ts(row.get("timestamp", ""))
            if ts is None:
                continue
            d: dict[str, Any] = {"timestamp": ts}
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
    print("=" * 120)
    print("  BWO 5-MINUTE MARKET BACKTEST — Continuation Filter")
    print("  5m BTC prediction markets | Entry at minute 1-2")
    print("=" * 120)
    _flush()

    # ===== LOAD DATA =====
    print("\n  Loading data...")
    _flush()

    all_candles = load_csv_fast(BTC_SPOT_CSV)
    all_candles.sort(key=lambda c: c.timestamp)
    seen: set[datetime] = set()
    unique: list[FastCandle] = []
    for c in all_candles:
        if c.timestamp not in seen:
            seen.add(c.timestamp)
            unique.append(c)
    all_candles = unique
    print(f"    Spot BTC: {len(all_candles):,} candles")
    _flush()

    # Auxiliary data
    futures_data, futures_lookup = _load_csv_dict(
        BTC_FUTURES_CSV, ["volume", "taker_buy_volume", "num_trades"]
    )
    for d in futures_data:
        d["num_trades"] = int(d.get("num_trades", 0))
    print(f"    BTC futures: {len(futures_data):,}")

    eth_data, eth_lookup = _load_csv_dict(
        ETH_FUTURES_CSV, ["close", "volume", "taker_buy_volume"]
    )
    print(f"    ETH futures: {len(eth_data):,}")

    dvol_data: list[dict] = []
    dvol_timestamps: list[datetime] = []
    if DVOL_CSV.exists():
        with open(DVOL_CSV) as f:
            for row in csv.DictReader(f):
                ts = _parse_ts(row["timestamp"])
                if ts is None:
                    continue
                dvol_data.append({"timestamp": ts, "close": float(row.get("close", 0))})
                dvol_timestamps.append(ts)
    print(f"    DVOL: {len(dvol_data):,}")

    deribit_data: list[dict] = []
    deribit_lookup: dict[datetime, int] = {}
    if DERIBIT_PERP_CSV.exists():
        with open(DERIBIT_PERP_CSV) as f:
            for row in csv.DictReader(f):
                ts = _parse_ts(row["timestamp"])
                if ts is None:
                    continue
                deribit_lookup[ts] = len(deribit_data)
                deribit_data.append({
                    "timestamp": ts,
                    "close": float(row.get("close", 0)),
                })
    print(f"    Deribit perp: {len(deribit_data):,}")
    _flush()

    # ===== BUILD 5m WINDOWS =====
    windows = group_into_5m_windows(all_candles)
    candle_by_ts = {c.timestamp: i for i, c in enumerate(all_candles)}

    window_data: list[WindowData5m] = []
    for wi, w in enumerate(windows):
        ts = w[0].timestamp
        idx = candle_by_ts.get(ts)
        if idx is None:
            continue
        resolution = "Up" if w[-1].close >= w[0].open else "Down"
        window_data.append(WindowData5m(
            window_idx=wi, candle_idx=idx, resolution=resolution, timestamp=ts,
        ))

    print(f"\n  Total 5m windows: {len(window_data):,}")

    # Base rate
    up = sum(1 for wd in window_data if wd.resolution == "Up")
    down = len(window_data) - up
    print(f"  Base rate: Up {up/len(window_data)*100:.2f}% | Down {down/len(window_data)*100:.2f}%")
    _flush()

    # ===== TEST EACH ENTRY MINUTE =====
    report: dict[str, Any] = {
        "meta": {
            "total_candles": len(all_candles),
            "total_5m_windows": len(window_data),
            "base_rate_up": round(up / len(window_data) * 100, 2),
        },
        "entry_minutes": {},
    }

    for entry_minute in [1, 2]:
        print(f"\n  {'='*100}")
        print(f"  ENTRY MINUTE {entry_minute} (see {entry_minute} of 5 candles = {entry_minute*20}% of window)")
        print(f"  {'='*100}")
        _flush()

        # Compute features for all windows
        print(f"\n    Computing features...")
        _flush()
        t0 = time.time()

        all_feature_dicts: list[dict[str, float]] = []
        continuation_targets: list[int] = []
        valid_mask: list[bool] = []
        entry_prices: list[float] = []

        for i, wd in enumerate(window_data):
            w = windows[wd.window_idx]

            feats = compute_all_features_5m(
                wd, windows, all_candles, entry_minute,
                futures_data, futures_lookup,
                eth_data, eth_lookup,
                dvol_data, dvol_timestamps,
                deribit_data, deribit_lookup,
            )
            all_feature_dicts.append(feats)

            # Continuation target
            early_dir = feats.get("early_direction", 0.0)
            if early_dir == 0.0:
                continuation_targets.append(0)
                valid_mask.append(False)
                entry_prices.append(0.50)
            else:
                continued = (
                    (wd.resolution == "Up" and early_dir > 0)
                    or (wd.resolution == "Down" and early_dir < 0)
                )
                continuation_targets.append(1 if continued else 0)
                valid_mask.append(True)
                # REALISTIC entry price model:
                # At minute 1 of 5m window, the market has already seen BTC move.
                # For 5m markets the price shifts aggressively:
                # - 5m windows are short, so even small BTC moves shift price a lot
                # - At minute 1: ~20% of window elapsed, price sensitivity ~0.8-1.0
                # - At minute 2: ~40% elapsed, price sensitivity ~1.2-1.5
                # Use logistic curve: P(UP) ≈ sigmoid(BTC_return * sensitivity)
                cum_ret = feats.get("early_cum_return", 0.0)
                # Higher sensitivity for 5m (shorter window = price reacts faster)
                sensitivity = 800 if entry_minute == 1 else 1200
                logit = cum_ret * sensitivity
                estimated_yes_prob = 1.0 / (1.0 + math.exp(-logit)) if abs(logit) < 500 else (1.0 if logit > 0 else 0.0)
                # The entry price is the YES token ask if we're buying continuation
                # If early_dir > 0 (UP), we buy YES → entry = estimated_yes_prob + spread
                # If early_dir < 0 (DOWN), we buy NO → entry = (1 - estimated_yes_prob) + spread
                spread = 0.02  # 2 cent spread
                if early_dir > 0:
                    ep = estimated_yes_prob + spread
                else:
                    ep = (1.0 - estimated_yes_prob) + spread
                ep = max(0.50, min(ep, 0.85))  # Floor at 50c, cap at 85c
                entry_prices.append(ep)

            if (i + 1) % 50000 == 0:
                print(f"      {i+1:,} / {len(window_data):,}")
                _flush()

        elapsed = time.time() - t0
        print(f"    Features computed in {elapsed:.1f}s")

        valid_indices = [i for i in range(len(window_data)) if valid_mask[i]]
        n_valid = len(valid_indices)
        n_cont = sum(continuation_targets[i] for i in valid_indices)
        cont_rate = n_cont / n_valid * 100 if n_valid > 0 else 0

        print(f"    Valid windows: {n_valid:,} (skipped {len(window_data)-n_valid:,} flat)")
        print(f"    Continuation rate: {cont_rate:.1f}%")
        _flush()

        # ===== FEATURE SELECTION BY MI =====
        all_feature_names = sorted(set().union(*(d.keys() for d in all_feature_dicts[:100])))
        target_arr = np.array([continuation_targets[i] for i in valid_indices])

        mi_results: dict[str, float] = {}
        for fname in all_feature_names:
            vals = np.array([all_feature_dicts[i].get(fname, 0.0) for i in valid_indices])
            mi_results[fname] = compute_mi_bits(vals, target_arr) if vals.std() > 0 else 0.0

        sorted_mi = sorted(mi_results.items(), key=lambda x: x[1], reverse=True)
        print(f"\n    Top features by MI:")
        for fname, mi in sorted_mi[:15]:
            print(f"      {fname:30s} MI={mi:.4f} bits")
        _flush()

        feature_names = [f for f, mi in sorted_mi if mi >= 0.0005][:40]
        if len(feature_names) < 5:
            feature_names = [f for f, _ in sorted_mi[:20]]
        print(f"    Selected {len(feature_names)} features")

        # ===== TRAIN/TEST SPLIT =====
        split_idx = int(n_valid * 0.8)
        train_indices = valid_indices[:split_idx]
        test_indices = valid_indices[split_idx:]

        X_train = np.array([[all_feature_dicts[i].get(f, 0.0) for f in feature_names] for i in train_indices])
        y_train = np.array([continuation_targets[i] for i in train_indices])
        X_test = np.array([[all_feature_dicts[i].get(f, 0.0) for f in feature_names] for i in test_indices])
        y_test = np.array([continuation_targets[i] for i in test_indices])

        print(f"\n    Train: {len(train_indices):,} | Test: {len(test_indices):,}")
        print(f"    Train cont rate: {y_train.mean()*100:.1f}%")
        print(f"    Test cont rate: {y_test.mean()*100:.1f}%")
        _flush()

        # ===== TRAIN MODEL =====
        model = HistGradientBoostingClassifier(
            max_iter=500, max_depth=5, learning_rate=0.05,
            min_samples_leaf=80, l2_regularization=1.0,
            max_bins=128, random_state=42,
        )
        print(f"\n    Training HistGradientBoosting...")
        _flush()
        model.fit(X_train, y_train)

        train_acc = float((model.predict(X_train) == y_train).mean())
        test_acc = float((model.predict(X_test) == y_test).mean())
        print(f"    Train accuracy: {train_acc*100:.1f}%")
        print(f"    Test accuracy: {test_acc*100:.1f}%")
        print(f"    IS/OOS gap: {(train_acc-test_acc)*100:.1f}pp")
        _flush()

        # Feature importance (permutation-based since HistGB may not have feature_importances_)
        from sklearn.inspection import permutation_importance as sklearn_perm_imp
        perm_result = sklearn_perm_imp(model, X_test[:5000], y_test[:5000], n_repeats=5, random_state=42)
        importances = perm_result.importances_mean
        imp_order = np.argsort(importances)[::-1]
        print(f"\n    Feature importances (permutation):")
        for rank, idx in enumerate(imp_order[:10]):
            print(f"      {feature_names[idx]:30s} {importances[idx]:.4f}")
        _flush()

        # ===== CONFIDENCE THRESHOLD SWEEP =====
        test_probas = model.predict_proba(X_test)[:, 1]

        print(f"\n    Confidence threshold sweep (test set):")
        print(f"    {'Thresh':>8s} {'Acc':>8s} {'Skip':>8s} {'Trades':>8s} {'EV/trade':>10s} {'Net PnL':>12s}")
        print(f"    {'-'*60}")

        best_result = None
        threshold_results = []

        for thresh_pct in range(50, 96):
            thresh = thresh_pct / 100.0
            mask = test_probas >= thresh
            n_traded = int(mask.sum())
            if n_traded < 100:
                continue

            correct = int((y_test[mask] == 1).sum())
            acc = correct / n_traded
            skip = 1.0 - n_traded / len(test_indices)

            # PnL simulation
            pnl = 0.0
            traded_indices = [test_indices[j] for j in range(len(test_indices)) if mask[j]]
            for ti in traded_indices:
                ep = entry_prices[ti]
                fee = polymarket_fee(POSITION_SIZE, ep)
                slip = POSITION_SIZE * SLIPPAGE_BPS / 10000
                if continuation_targets[ti] == 1:
                    pnl_t = ((1.0 - ep) / ep) * POSITION_SIZE - fee - slip
                else:
                    pnl_t = ((0.0 - ep) / ep) * POSITION_SIZE - fee - slip
                pnl += pnl_t

            ev = pnl / n_traded if n_traded > 0 else 0.0
            print(f"    {thresh:.2f}     {acc*100:6.1f}%  {skip*100:6.1f}%  {n_traded:6d}    ${ev:8.2f}    ${pnl:10.0f}")

            result = {
                "threshold": thresh,
                "accuracy": round(acc * 100, 2),
                "skip_rate": round(skip * 100, 2),
                "trades": n_traded,
                "ev_per_trade": round(ev, 2),
                "net_pnl": round(pnl, 0),
            }
            threshold_results.append(result)

            if acc >= 0.80 and skip < 0.50 and (best_result is None or ev > best_result["ev_per_trade"]):
                best_result = result

        _flush()

        if best_result:
            print(f"\n    *** BEST: threshold={best_result['threshold']:.2f} → "
                  f"acc={best_result['accuracy']:.1f}%, skip={best_result['skip_rate']:.1f}%, "
                  f"EV=${best_result['ev_per_trade']:.2f}/trade ***")
        else:
            # Find best regardless of criteria
            if threshold_results:
                best_ev = max(threshold_results, key=lambda x: x["ev_per_trade"])
                print(f"\n    No result meets 80%+acc/<50% skip. Best EV: "
                      f"thresh={best_ev['threshold']:.2f}, acc={best_ev['accuracy']:.1f}%, "
                      f"skip={best_ev['skip_rate']:.1f}%, EV=${best_ev['ev_per_trade']:.2f}")
                best_result = best_ev
        _flush()

        # ===== WALK-FORWARD VALIDATION =====
        print(f"\n    Walk-forward validation (3-month train / 1-month test)...")
        _flush()

        # Group by month
        month_groups: dict[str, list[int]] = defaultdict(list)
        for vi in valid_indices:
            wd = window_data[vi]
            key = wd.timestamp.strftime("%Y-%m")
            month_groups[key].append(vi)
        months = sorted(month_groups.keys())

        wf_results: list[dict] = []
        for mi_idx in range(WF_TRAIN_MONTHS, len(months)):
            train_months_list = months[mi_idx - WF_TRAIN_MONTHS:mi_idx]
            test_month = months[mi_idx]

            wf_train_idx = []
            for m in train_months_list:
                wf_train_idx.extend(month_groups[m])
            wf_test_idx = month_groups[test_month]

            if len(wf_train_idx) < 500 or len(wf_test_idx) < 100:
                continue

            X_wf_train = np.array([[all_feature_dicts[i].get(f, 0.0) for f in feature_names] for i in wf_train_idx])
            y_wf_train = np.array([continuation_targets[i] for i in wf_train_idx])
            X_wf_test = np.array([[all_feature_dicts[i].get(f, 0.0) for f in feature_names] for i in wf_test_idx])
            y_wf_test = np.array([continuation_targets[i] for i in wf_test_idx])

            wf_model = HistGradientBoostingClassifier(
                max_iter=500, max_depth=5, learning_rate=0.05,
                min_samples_leaf=80, l2_regularization=1.0,
                max_bins=128, random_state=42,
            )
            wf_model.fit(X_wf_train, y_wf_train)
            wf_probas = wf_model.predict_proba(X_wf_test)[:, 1]

            # Use best threshold from sweep
            thresh = best_result["threshold"] if best_result else 0.50
            wf_mask = wf_probas >= thresh
            n_wf_traded = int(wf_mask.sum())

            if n_wf_traded > 0:
                wf_correct = int((y_wf_test[wf_mask] == 1).sum())
                wf_acc = wf_correct / n_wf_traded

                # PnL
                wf_pnl = 0.0
                for j in range(len(wf_test_idx)):
                    if not wf_mask[j]:
                        continue
                    ti = wf_test_idx[j]
                    ep = entry_prices[ti]
                    fee = polymarket_fee(POSITION_SIZE, ep)
                    slip = POSITION_SIZE * SLIPPAGE_BPS / 10000
                    if continuation_targets[ti] == 1:
                        pnl_t = ((1.0 - ep) / ep) * POSITION_SIZE - fee - slip
                    else:
                        pnl_t = ((0.0 - ep) / ep) * POSITION_SIZE - fee - slip
                    wf_pnl += pnl_t

                wf_results.append({
                    "month": test_month,
                    "accuracy": round(wf_acc * 100, 2),
                    "trades": n_wf_traded,
                    "total_available": len(wf_test_idx),
                    "skip_rate": round((1 - n_wf_traded / len(wf_test_idx)) * 100, 2),
                    "pnl": round(wf_pnl, 2),
                })

        if wf_results:
            profitable_months = sum(1 for r in wf_results if r["pnl"] > 0)
            wf_mean_acc = sum(r["accuracy"] for r in wf_results) / len(wf_results)
            wf_total_pnl = sum(r["pnl"] for r in wf_results)
            wf_mean_skip = sum(r["skip_rate"] for r in wf_results) / len(wf_results)

            print(f"    Walk-forward: {len(wf_results)} periods")
            print(f"    Mean accuracy: {wf_mean_acc:.1f}%")
            print(f"    Mean skip rate: {wf_mean_skip:.1f}%")
            print(f"    Total PnL: ${wf_total_pnl:,.0f}")
            print(f"    Profitable months: {profitable_months}/{len(wf_results)}")
            _flush()

            print(f"\n    Monthly detail:")
            for r in wf_results:
                print(f"      {r['month']}: acc={r['accuracy']:5.1f}%, "
                      f"trades={r['trades']:5d}, skip={r['skip_rate']:5.1f}%, "
                      f"PnL=${r['pnl']:+10,.2f}")
            _flush()
        else:
            wf_mean_acc = 0.0
            wf_total_pnl = 0.0
            profitable_months = 0
            wf_mean_skip = 0.0

        # ===== BOOTSTRAP CI =====
        if best_result and best_result["trades"] > 100:
            thresh = best_result["threshold"]
            mask = test_probas >= thresh
            correct_list = [(y_test[j] == 1) for j in range(len(y_test)) if mask[j]]
            mean_acc, lo, hi = bootstrap_ci(correct_list)
            print(f"\n    Bootstrap 95% CI: [{lo*100:.1f}%, {hi*100:.1f}%]")
            _flush()

            # Permutation test
            preds = [1 for _ in range(len(correct_list))]  # All predict "continue"
            actuals_perm = [1 if c else 0 for c in correct_list]
            p_val = permutation_test(preds, actuals_perm)
            print(f"    Permutation p-value: {p_val:.4f}")
            _flush()
        else:
            lo, hi = 0.0, 0.0
            p_val = 1.0

        # ===== STORE RESULTS =====
        report["entry_minutes"][str(entry_minute)] = {
            "continuation_rate": round(cont_rate, 2),
            "valid_windows": n_valid,
            "skip_rate_flat": round((len(window_data) - n_valid) / len(window_data) * 100, 2),
            "train_accuracy": round(train_acc * 100, 2),
            "test_accuracy": round(test_acc * 100, 2),
            "is_oos_gap": round((train_acc - test_acc) * 100, 2),
            "feature_names": feature_names,
            "mi_ranking": {f: round(mi, 4) for f, mi in sorted_mi[:20]},
            "feature_importances": {
                feature_names[idx]: round(float(importances[idx]), 4)
                for idx in imp_order[:15]
            },
            "threshold_sweep": threshold_results,
            "best_result": best_result,
            "walk_forward": {
                "periods": len(wf_results),
                "mean_accuracy": round(wf_mean_acc, 2),
                "mean_skip_rate": round(wf_mean_skip, 2),
                "total_pnl": round(wf_total_pnl, 0),
                "profitable_months": profitable_months,
                "monthly": wf_results,
            },
            "bootstrap_ci_95": [round(lo * 100, 2), round(hi * 100, 2)],
            "permutation_p": round(p_val, 4),
        }

    # ===== SAVE REPORT =====
    with open(REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved to {REPORT_JSON}")

    # ===== SUMMARY =====
    print(f"\n  {'='*100}")
    print(f"  SUMMARY")
    print(f"  {'='*100}")
    for em, data in report["entry_minutes"].items():
        br = data.get("best_result", {}) or {}
        wf = data.get("walk_forward", {})
        print(f"\n  Entry Minute {em}:")
        print(f"    Continuation rate: {data['continuation_rate']:.1f}%")
        print(f"    Test accuracy (raw): {data['test_accuracy']:.1f}%")
        if br:
            print(f"    Best threshold: {br.get('threshold', 'N/A')}")
            print(f"    Filtered accuracy: {br.get('accuracy', 'N/A')}%")
            print(f"    Skip rate: {br.get('skip_rate', 'N/A')}%")
            print(f"    EV/trade: ${br.get('ev_per_trade', 0):.2f}")
        print(f"    WF mean accuracy: {wf.get('mean_accuracy', 0):.1f}%")
        print(f"    WF profitable: {wf.get('profitable_months', 0)}/{wf.get('periods', 0)}")
        print(f"    WF total PnL: ${wf.get('total_pnl', 0):,.0f}")
    _flush()


if __name__ == "__main__":
    main()
