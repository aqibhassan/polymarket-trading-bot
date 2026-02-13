"""BWO v4 Options & Cross-Exchange Backtest.

Tests whether forward-looking options data (DVOL implied volatility) and
cross-exchange signals (Deribit vs Binance) can predict 15m BTC direction.

Phase 0: MI fail-fast on all options features (STOP if max MI < 0.01)
Phase 1: DVOL features only (10)
Phase 2: + Cross-exchange (15)
Phase 3: + Combined with original 31 (46 total)

4 approaches: Regime Mining, Contrarian Stacking, Meta-Model, Full ML.
Walk-forward: 3-month train / 1-month test.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from bisect import bisect_right
from collections import defaultdict, namedtuple
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

# Reuse approaches from v3
from scripts.bwo_orderflow_backtest import (
    ORIGINAL_FEATURE_NAMES,
    compute_mi_bits,
    contrarian_stacking,
    full_ml_approach,
    regime_mining,
    run_phase,
    two_stage_meta_model,
    walk_forward_ml,
)

from sklearn.ensemble import HistGradientBoostingClassifier

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BTC_SPOT_CSV = PROJECT_ROOT / "data" / "btc_1m_2y.csv"
DVOL_CSV = PROJECT_ROOT / "data" / "deribit_dvol_1m.csv"
DERIBIT_PERP_CSV = PROJECT_ROOT / "data" / "deribit_btc_perp_1m.csv"
DERIBIT_FUNDING_CSV = PROJECT_ROOT / "data" / "deribit_funding.csv"
BINANCE_FUNDING_CSV = PROJECT_ROOT / "data" / "btc_funding_rate.csv"
BTC_FUTURES_CSV = PROJECT_ROOT / "data" / "btc_futures_1m_2y.csv"
REPORT_JSON = PROJECT_ROOT / "data" / "bwo_options_report.json"

MI_FAIL_FAST_THRESHOLD = 0.01  # bits

DvolCandle = namedtuple("DvolCandle", ["timestamp", "open", "high", "low", "close"])
DeribitCandle = namedtuple("DeribitCandle", ["timestamp", "open", "high", "low", "close", "volume"])
FundingEntry = namedtuple("FundingEntry", ["timestamp", "interest_8h", "index_price"])


def _flush() -> None:
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Data Loaders
# ---------------------------------------------------------------------------

def _parse_ts(raw: str) -> datetime | None:
    raw = raw.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def load_dvol(path: Path) -> list[DvolCandle]:
    candles: list[DvolCandle] = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = _parse_ts(row["timestamp"])
            if ts is None:
                continue
            candles.append(DvolCandle(
                timestamp=ts,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
            ))
    candles.sort(key=lambda c: c.timestamp)
    return candles


def load_deribit_perp(path: Path) -> list[DeribitCandle]:
    candles: list[DeribitCandle] = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = _parse_ts(row["timestamp"])
            if ts is None:
                continue
            candles.append(DeribitCandle(
                timestamp=ts,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volume", 0)),
            ))
    candles.sort(key=lambda c: c.timestamp)
    return candles


def load_deribit_funding(path: Path) -> list[FundingEntry]:
    entries: list[FundingEntry] = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = _parse_ts(row["timestamp"])
            if ts is None:
                continue
            entries.append(FundingEntry(
                timestamp=ts,
                interest_8h=float(row.get("interest_8h", 0)),
                index_price=float(row.get("index_price", 0)),
            ))
    entries.sort(key=lambda e: e.timestamp)
    return entries


def load_binance_funding(path: Path) -> list[tuple[datetime, float]]:
    entries: list[tuple[datetime, float]] = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = _parse_ts(row["timestamp"])
            if ts is None:
                continue
            entries.append((ts, float(row["funding_rate"])))
    entries.sort(key=lambda e: e[0])
    return entries


# ---------------------------------------------------------------------------
# DVOL Features (10 features)
# ---------------------------------------------------------------------------

DVOL_FEATURE_NAMES = [
    "dvol_level", "dvol_change_5", "dvol_change_15", "dvol_change_60",
    "dvol_accel", "dvol_percentile_60", "dvol_z_score",
    "vol_risk_premium", "dvol_regime", "dvol_speed",
]


def compute_dvol_features(
    ci: int,
    all_candles: list[FastCandle],
    dvol_candles: list[DvolCandle],
    dvol_timestamps: list[datetime],
) -> dict[str, float]:
    """Compute 10 DVOL-based features.

    DVOL data is hourly — use binary search to find the most recent DVOL
    candle before the window start timestamp.
    """
    features: dict[str, float] = {}
    target_ts = all_candles[ci].timestamp

    # Find the most recent DVOL candle at or before target_ts
    di = bisect_right(dvol_timestamps, target_ts) - 1

    defaults = {f: 0.0 for f in DVOL_FEATURE_NAMES}

    if di < 60:
        features.update(defaults)
        return features

    # Current DVOL level
    dvol_now = dvol_candles[di - 1].close
    features["dvol_level"] = dvol_now

    # DVOL changes at different lookbacks
    def _dvol_change(lookback: int) -> float:
        if di >= lookback + 1:
            prev = dvol_candles[di - lookback - 1].close
            if prev > 0:
                return (dvol_now - prev) / prev
        return 0.0

    features["dvol_change_5"] = _dvol_change(5)
    features["dvol_change_15"] = _dvol_change(15)
    features["dvol_change_60"] = _dvol_change(60)

    # Acceleration: short-term change minus medium-term change
    features["dvol_accel"] = features["dvol_change_5"] - features["dvol_change_15"]

    # Percentile rank in last 60 candles
    start_p = max(0, di - 60)
    recent_dvols = [dvol_candles[j].close for j in range(start_p, di)]
    if recent_dvols:
        below = sum(1 for d in recent_dvols if d <= dvol_now)
        features["dvol_percentile_60"] = below / len(recent_dvols)
    else:
        features["dvol_percentile_60"] = 0.5

    # Z-score vs 60-candle distribution
    if len(recent_dvols) >= 10:
        mean_d = sum(recent_dvols) / len(recent_dvols)
        var_d = sum((d - mean_d) ** 2 for d in recent_dvols) / len(recent_dvols)
        std_d = math.sqrt(var_d) if var_d > 0 else 0.001
        features["dvol_z_score"] = (dvol_now - mean_d) / std_d
    else:
        features["dvol_z_score"] = 0.0

    # Volatility risk premium: DVOL (implied) vs realized vol
    if ci >= 60:
        sq_sum = 0.0
        n_rets = 0
        for j in range(ci - 59, ci):
            prev_c = all_candles[j - 1].close
            if prev_c > 0:
                r = (all_candles[j].close - prev_c) / prev_c
                sq_sum += r * r
                n_rets += 1
        if n_rets > 0:
            realized_vol = math.sqrt(sq_sum / n_rets) * math.sqrt(525600) * 100  # Annualized %
            features["vol_risk_premium"] = dvol_now - realized_vol
        else:
            features["vol_risk_premium"] = 0.0
    else:
        features["vol_risk_premium"] = 0.0

    # DVOL regime: high/normal/low
    if len(recent_dvols) >= 20:
        sorted_dvols = sorted(recent_dvols)
        p25 = sorted_dvols[int(0.25 * len(sorted_dvols))]
        p75 = sorted_dvols[int(0.75 * len(sorted_dvols))]
        if dvol_now > p75:
            features["dvol_regime"] = 1.0  # High vol
        elif dvol_now < p25:
            features["dvol_regime"] = -1.0  # Low vol
        else:
            features["dvol_regime"] = 0.0  # Normal
    else:
        features["dvol_regime"] = 0.0

    # DVOL speed: rate of change (slope of DVOL over last 15 candles)
    if di >= 15:
        dvol_vals = [dvol_candles[j].close for j in range(di - 15, di)]
        n = len(dvol_vals)
        xm = (n - 1) / 2.0
        ym = sum(dvol_vals) / n
        num = sum((i - xm) * (dvol_vals[i] - ym) for i in range(n))
        den = sum((i - xm) ** 2 for i in range(n))
        features["dvol_speed"] = num / den if den > 0 else 0.0
    else:
        features["dvol_speed"] = 0.0

    return features


# ---------------------------------------------------------------------------
# Cross-Exchange Features (5 features)
# ---------------------------------------------------------------------------

CROSS_EXCHANGE_FEATURE_NAMES = [
    "deribit_binance_basis", "deribit_binance_basis_momentum",
    "deribit_volume_ratio", "deribit_funding_spread",
    "cross_exchange_lead",
]


def compute_cross_exchange_features(
    ci: int,
    all_candles: list[FastCandle],
    deribit_perp: list[DeribitCandle],
    deribit_lookup: dict[datetime, int],
    btc_futures_candles: list | None,
    btc_futures_lookup: dict[datetime, int] | None,
    deribit_funding: list[FundingEntry],
    deribit_funding_ts: list[datetime],
    binance_funding: list[tuple[datetime, float]],
    binance_funding_ts: list[datetime],
) -> dict[str, float]:
    """Compute 5 cross-exchange features."""
    features: dict[str, float] = {}
    target_ts = all_candles[ci].timestamp
    di = deribit_lookup.get(target_ts)

    defaults = {f: 0.0 for f in CROSS_EXCHANGE_FEATURE_NAMES}

    if di is None or di < 15:
        features.update(defaults)
        return features

    binance_close = all_candles[ci - 1].close if ci > 0 else all_candles[ci].close
    deribit_close = deribit_perp[di - 1].close

    # Deribit-Binance basis (price difference in bps)
    if binance_close > 0:
        basis = (deribit_close - binance_close) / binance_close * 10000  # bps
        features["deribit_binance_basis"] = basis
    else:
        features["deribit_binance_basis"] = 0.0

    # Basis momentum (change in basis over 15 candles)
    if di >= 15 and ci >= 15:
        prev_binance = all_candles[ci - 15].close
        prev_deribit = deribit_perp[di - 15].close
        if prev_binance > 0:
            prev_basis = (prev_deribit - prev_binance) / prev_binance * 10000
            features["deribit_binance_basis_momentum"] = features["deribit_binance_basis"] - prev_basis
        else:
            features["deribit_binance_basis_momentum"] = 0.0
    else:
        features["deribit_binance_basis_momentum"] = 0.0

    # Volume ratio: Deribit vs Binance (15-candle)
    if di >= 15 and ci >= 15:
        deribit_vol = sum(deribit_perp[j].volume for j in range(di - 15, di))
        binance_vol = sum(all_candles[j].volume for j in range(ci - 15, ci))
        features["deribit_volume_ratio"] = deribit_vol / binance_vol if binance_vol > 0 else 0.0
    else:
        features["deribit_volume_ratio"] = 0.0

    # Funding rate spread: Deribit vs Binance
    if deribit_funding and binance_funding:
        # Find latest Deribit funding before target
        d_idx = bisect_right(deribit_funding_ts, target_ts) - 1
        b_idx = bisect_right(binance_funding_ts, target_ts) - 1
        if d_idx >= 0 and b_idx >= 0:
            d_rate = deribit_funding[d_idx].interest_8h
            b_rate = binance_funding[b_idx][1]
            features["deribit_funding_spread"] = d_rate - b_rate
        else:
            features["deribit_funding_spread"] = 0.0
    else:
        features["deribit_funding_spread"] = 0.0

    # Cross-exchange lead: Deribit 5m return vs Binance 5m return
    # If Deribit moved first (bigger return), it might lead
    if di >= 5 and ci >= 5:
        d_ret = (deribit_perp[di - 1].close - deribit_perp[di - 5].open) / deribit_perp[di - 5].open if deribit_perp[di - 5].open > 0 else 0.0
        b_ret = (all_candles[ci - 1].close - all_candles[ci - 5].open) / all_candles[ci - 5].open if all_candles[ci - 5].open > 0 else 0.0
        features["cross_exchange_lead"] = d_ret - b_ret  # Positive if Deribit leading
    else:
        features["cross_exchange_lead"] = 0.0

    return features


# ---------------------------------------------------------------------------
# Interaction Features (3 features)
# ---------------------------------------------------------------------------

INTERACTION_FEATURE_NAMES = [
    "dvol_momentum_interaction", "dvol_basis_interaction", "vol_premium_direction",
]


def compute_interaction_features(
    dvol_feats: dict[str, float],
    cross_feats: dict[str, float],
    original_feats: dict[str, float],
) -> dict[str, float]:
    """Compute 3 interaction features."""
    features: dict[str, float] = {}

    # DVOL change × price momentum direction
    mtf_sign = 1.0 if original_feats.get("mtf_score", 0) > 0 else (-1.0 if original_feats.get("mtf_score", 0) < 0 else 0.0)
    features["dvol_momentum_interaction"] = dvol_feats.get("dvol_change_5", 0) * mtf_sign

    # DVOL × basis (high vol + positive basis = speculative frenzy)
    features["dvol_basis_interaction"] = dvol_feats.get("dvol_level", 0) * cross_feats.get("deribit_binance_basis", 0) / 10000

    # Vol risk premium × recent price direction
    # Positive premium + down move = fear → continuation likely
    # Negative premium + up move = complacency → reversal?
    stm_dir = original_feats.get("stm_dir", 0.0)
    features["vol_premium_direction"] = dvol_feats.get("vol_risk_premium", 0) * stm_dir

    return features


# ---------------------------------------------------------------------------
# All feature names
# ---------------------------------------------------------------------------

ALL_OPTIONS_FEATURES = DVOL_FEATURE_NAMES + CROSS_EXCHANGE_FEATURE_NAMES + INTERACTION_FEATURE_NAMES


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 100)
    print("  BWO v4 — OPTIONS & CROSS-EXCHANGE BACKTEST")
    print("  DVOL implied volatility + Deribit/Binance cross-exchange signals")
    print("  Fail-fast MI gate | 4 approaches | Walk-forward validation")
    print("=" * 100)
    _flush()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("\n  Loading spot BTC data...", end=" ")
    _flush()
    if not BTC_SPOT_CSV.exists():
        print(f"\n  ERROR: {BTC_SPOT_CSV} not found.")
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
    print(f"{len(all_candles):,} candles")
    _flush()

    # DVOL
    print("  Loading DVOL data...", end=" ")
    _flush()
    if not DVOL_CSV.exists():
        print(f"\n  ERROR: {DVOL_CSV} not found. Run download_deribit_data.py first.")
        sys.exit(1)
    dvol_candles = load_dvol(DVOL_CSV)
    dvol_timestamps = [c.timestamp for c in dvol_candles]
    print(f"{len(dvol_candles):,} candles")
    _flush()

    # Deribit perpetual
    print("  Loading Deribit BTC-PERPETUAL...", end=" ")
    _flush()
    if DERIBIT_PERP_CSV.exists():
        deribit_perp = load_deribit_perp(DERIBIT_PERP_CSV)
        deribit_lookup: dict[datetime, int] = {c.timestamp: i for i, c in enumerate(deribit_perp)}
        print(f"{len(deribit_perp):,} candles")
    else:
        print("NOT FOUND (will skip cross-exchange features)")
        deribit_perp = []
        deribit_lookup = {}
    _flush()

    # BTC Futures (Binance) — for cross-exchange comparison
    btc_futures_candles = None
    btc_futures_lookup: dict[datetime, int] | None = None
    if BTC_FUTURES_CSV.exists():
        from scripts.fast_loader_v3 import load_futures_csv, build_futures_lookup
        print("  Loading Binance BTC futures...", end=" ")
        _flush()
        _btc_fut = load_futures_csv(BTC_FUTURES_CSV)
        _btc_fut.sort(key=lambda c: c.timestamp)
        btc_futures_lookup = build_futures_lookup(_btc_fut)
        btc_futures_candles = _btc_fut
        print(f"{len(_btc_fut):,} candles")
        _flush()

    # Deribit funding
    print("  Loading Deribit funding rates...", end=" ")
    _flush()
    if DERIBIT_FUNDING_CSV.exists():
        deribit_funding = load_deribit_funding(DERIBIT_FUNDING_CSV)
        deribit_funding_ts = [e.timestamp for e in deribit_funding]
        print(f"{len(deribit_funding):,} entries")
    else:
        print("NOT FOUND (will use zeros)")
        deribit_funding = []
        deribit_funding_ts = []
    _flush()

    # Binance funding
    print("  Loading Binance funding rates...", end=" ")
    _flush()
    if BINANCE_FUNDING_CSV.exists():
        binance_funding = load_binance_funding(BINANCE_FUNDING_CSV)
        binance_funding_ts = [e[0] for e in binance_funding]
        print(f"{len(binance_funding):,} entries")
    else:
        print("NOT FOUND (will use zeros)")
        binance_funding = []
        binance_funding_ts = []
    _flush()

    # ------------------------------------------------------------------
    # Build 15m windows
    # ------------------------------------------------------------------
    print("\n  Grouping into 15m windows...", end=" ")
    _flush()
    windows = group_into_15m_windows(all_candles)
    print(f"{len(windows):,} complete windows")

    candle_by_ts: dict[datetime, int] = {c.timestamp: i for i, c in enumerate(all_candles)}

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

    # Check DVOL coverage (hourly — count windows that have DVOL data before them)
    dvol_match = sum(1 for wd in window_data if bisect_right(dvol_timestamps, wd.timestamp) > 60)
    print(f"  {len(window_data):,} windows, DVOL coverage: {dvol_match:,} ({dvol_match/len(window_data)*100:.1f}%)")

    up_count = sum(1 for wd in window_data if wd.resolution == "Up")
    print(f"  Base rate: Up {up_count/len(window_data)*100:.1f}%")
    _flush()

    split_idx = int(len(window_data) * 0.80)

    # ------------------------------------------------------------------
    # Compute features
    # ------------------------------------------------------------------
    print(f"\n  Computing features for {len(window_data):,} windows...")
    _flush()
    t0 = time.time()

    all_feature_dicts: list[dict[str, float]] = []
    for i, wd in enumerate(window_data):
        ci = wd.candle_idx

        # Original 31 features
        original = compute_all_features(wd, windows, all_candles)

        # DVOL features (10)
        dvol = compute_dvol_features(ci, all_candles, dvol_candles, dvol_timestamps)

        # Cross-exchange features (5)
        cross = compute_cross_exchange_features(
            ci, all_candles, deribit_perp, deribit_lookup,
            btc_futures_candles, btc_futures_lookup,
            deribit_funding, deribit_funding_ts,
            binance_funding, binance_funding_ts,
        )

        # Interaction features (3)
        interaction = compute_interaction_features(dvol, cross, original)

        combined = {**original, **dvol, **cross, **interaction}
        all_feature_dicts.append(combined)

        if (i + 1) % 10000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(window_data) - i - 1) / rate
            print(f"    {i+1:>8,} / {len(window_data):,} ({(i+1)/len(window_data)*100:.1f}%) "
                  f"- {elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining")
            _flush()

    total_elapsed = time.time() - t0
    print(f"  Feature computation complete: {total_elapsed:.1f}s")
    _flush()

    # ------------------------------------------------------------------
    # PHASE 0: FAIL-FAST MI CHECK
    # ------------------------------------------------------------------
    print(f"\n{'='*100}")
    print(f"  PHASE 0: FAIL-FAST — Mutual Information Check (Options Features)")
    print(f"{'='*100}")
    _flush()

    target = np.array([1 if wd.resolution == "Up" else 0 for wd in window_data])

    print(f"\n  {'Feature':<40} {'MI (bits)':>12} {'Status':>10}")
    print(f"  {'-'*65}")

    mi_results: dict[str, float] = {}
    max_mi = 0.0
    max_mi_feature = ""

    for fname in ALL_OPTIONS_FEATURES:
        vals = np.array([fd.get(fname, 0.0) for fd in all_feature_dicts])
        # Check if feature has any variance
        if vals.std() == 0:
            mi = 0.0
        else:
            mi = compute_mi_bits(vals, target)
        mi_results[fname] = mi
        status = "OK" if mi >= MI_FAIL_FAST_THRESHOLD else "LOW"
        print(f"  {fname:<40} {mi:>12.6f} {status:>10}")
        if mi > max_mi:
            max_mi = mi
            max_mi_feature = fname

    print(f"\n  Max MI: {max_mi:.6f} bits ({max_mi_feature})")
    print(f"  Threshold: {MI_FAIL_FAST_THRESHOLD} bits")
    _flush()

    report: dict[str, Any] = {
        "meta": {
            "total_candles": len(all_candles),
            "dvol_candles": len(dvol_candles),
            "deribit_perp_candles": len(deribit_perp),
            "deribit_funding_entries": len(deribit_funding),
            "binance_funding_entries": len(binance_funding),
            "total_windows": len(window_data),
            "dvol_candle_count": len(dvol_candles),
            "dvol_coverage_pct": round(dvol_match / len(window_data) * 100, 1),
            "train_size": split_idx,
            "test_size": len(window_data) - split_idx,
            "base_rate_up_pct": round(up_count / len(window_data) * 100, 2),
        },
        "phase_0_mi": {
            "features": {k: round(v, 6) for k, v in mi_results.items()},
            "max_mi": round(max_mi, 6),
            "max_mi_feature": max_mi_feature,
            "threshold": MI_FAIL_FAST_THRESHOLD,
        },
    }

    if max_mi < MI_FAIL_FAST_THRESHOLD:
        print(f"\n  *** FAIL-FAST TRIGGERED ***")
        print(f"  Max MI ({max_mi:.6f}) < threshold ({MI_FAIL_FAST_THRESHOLD})")
        print(f"  Options/cross-exchange features have NO predictive power.")
        report["phase_0_mi"]["fail_fast"] = True
        report["conclusion"] = "FAIL-FAST: No options signal detected."

        # Still save report
        with open(REPORT_JSON, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n  Report saved: {REPORT_JSON}")
        print(f"\n{'='*100}")
        _flush()
        return
    else:
        print(f"\n  PASS — Options features show MI above threshold. Proceeding.")
        report["phase_0_mi"]["fail_fast"] = False
    _flush()

    # ------------------------------------------------------------------
    # PHASE 1: DVOL features only (10)
    # ------------------------------------------------------------------
    phase1_result = run_phase(
        "PHASE 1: DVOL Features Only",
        DVOL_FEATURE_NAMES,
        window_data, all_feature_dicts, split_idx,
    )
    report["phase_1_dvol"] = phase1_result

    # ------------------------------------------------------------------
    # PHASE 2: + Cross-exchange (15)
    # ------------------------------------------------------------------
    phase2_features = DVOL_FEATURE_NAMES + CROSS_EXCHANGE_FEATURE_NAMES
    phase2_result = run_phase(
        "PHASE 2: DVOL + Cross-Exchange",
        phase2_features,
        window_data, all_feature_dicts, split_idx,
    )
    report["phase_2_dvol_cross"] = phase2_result

    # ------------------------------------------------------------------
    # PHASE 3: All options + interactions (18)
    # ------------------------------------------------------------------
    phase3_result = run_phase(
        "PHASE 3: All Options Features (18)",
        ALL_OPTIONS_FEATURES,
        window_data, all_feature_dicts, split_idx,
    )
    report["phase_3_all_options"] = phase3_result

    # ------------------------------------------------------------------
    # PHASE 4: Combined with original 31 (49 total)
    # ------------------------------------------------------------------
    phase4_features = ORIGINAL_FEATURE_NAMES + ALL_OPTIONS_FEATURES
    phase4_result = run_phase(
        "PHASE 4: Combined (31 original + 18 options = 49)",
        phase4_features,
        window_data, all_feature_dicts, split_idx,
    )
    report["phase_4_combined"] = phase4_result

    # ------------------------------------------------------------------
    # OVERALL SUMMARY
    # ------------------------------------------------------------------
    print(f"\n\n{'='*100}")
    print(f"  OVERALL SUMMARY")
    print(f"{'='*100}")

    phases = [
        ("Phase 1 (DVOL 10)", phase1_result),
        ("Phase 2 (DVOL+Cross 15)", phase2_result),
        ("Phase 3 (All Options 18)", phase3_result),
        ("Phase 4 (Combined 49)", phase4_result),
    ]

    print(f"\n  {'Phase':<30} {'BestAcc':>9} {'Regime':>9} {'Stack':>9} {'Meta':>9} {'ML':>9} {'WF Mean':>9}")
    print(f"  {'-'*88}")

    for name, result in phases:
        regime_acc = result.get("regime_mining", {}).get("best_test_accuracy", 0)
        stack_acc = result.get("contrarian_stacking", {}).get("test_accuracy", 0)
        meta_acc = result.get("meta_model", {}).get("accuracy", 0)
        ml_acc = result.get("full_ml", {}).get("raw_accuracy", 0)
        wf_mean = result.get("walk_forward", {}).get("mean_accuracy", 0)

        print(f"  {name:<30} {result['best_accuracy']*100:>8.1f}% "
              f"{regime_acc*100:>8.1f}% {stack_acc*100:>8.1f}% "
              f"{meta_acc*100:>8.1f}% {ml_acc*100:>8.1f}% {wf_mean*100:>8.1f}%")

    best_overall = max(r["best_accuracy"] for _, r in phases)
    wf_best = max(r.get("walk_forward", {}).get("mean_accuracy", 0) for _, r in phases)

    if best_overall >= 0.80:
        tier = "Tier 1"
        action = "Ship to live strategy"
    elif best_overall >= 0.65:
        tier = "Tier 2"
        action = "Use as filter; collect CLOB snapshots for v5"
    else:
        tier = "Tier 3"
        action = "Pivot to entry-price optimization"

    print(f"\n  RESULT: {tier}")
    print(f"  Best accuracy: {best_overall*100:.1f}% | Best WF mean: {wf_best*100:.1f}%")
    print(f"  Action: {action}")

    report["summary"] = {
        "best_accuracy": round(best_overall, 4),
        "best_wf_accuracy": round(wf_best, 4),
        "tier": tier,
        "action": action,
    }

    with open(REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved: {REPORT_JSON}")
    print(f"\n{'='*100}")
    _flush()


if __name__ == "__main__":
    main()
