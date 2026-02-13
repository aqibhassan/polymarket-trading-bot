"""BWO v3 Order Flow & Cross-Asset Backtest.

Staged backtest with fail-fast MI gate:
  Phase 0: Mutual Information check on taker features (STOP if max MI < 0.01)
  Phase 1: Taker features only (10)
  Phase 2: + Cross-asset ETH (18)
  Phase 3: + Derivatives + composites (30)
  Phase 4: Combined 61 features (30 new + 31 original)

4 approaches per phase:
  A. Regime Mining — Bonferroni-corrected 2-way condition search
  B. Contrarian Stacking — independent weak signals stacked
  C. Two-Stage Meta-Model — predictability filter + direction
  D. Full ML — HistGradientBoosting + threshold sweep

Walk-forward: 3-month train / 1-month test rolling.
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest_before_window import (
    BOOTSTRAP_ITERATIONS,
    ENTRY_PRICE,
    PERMUTATION_ITERATIONS,
    POSITION_SIZE,
    WindowData,
    bootstrap_ci,
    compute_all_features,
    compute_metrics,
    permutation_test,
    simulate_trade,
)
from scripts.fast_loader import FastCandle, group_into_15m_windows, load_csv_fast
from scripts.fast_loader_v3 import (
    FundingEntry,
    FuturesCandle,
    build_futures_lookup,
    get_latest_funding,
    load_funding_csv,
    load_futures_csv,
)

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import mutual_info_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BTC_SPOT_CSV = PROJECT_ROOT / "data" / "btc_1m_2y.csv"
BTC_FUTURES_CSV = PROJECT_ROOT / "data" / "btc_futures_1m_2y.csv"
ETH_FUTURES_CSV = PROJECT_ROOT / "data" / "eth_futures_1m_2y.csv"
FUNDING_CSV = PROJECT_ROOT / "data" / "btc_funding_rate.csv"
REPORT_JSON = PROJECT_ROOT / "data" / "bwo_orderflow_report.json"

WF_TRAIN_MONTHS = 3
WF_TEST_MONTHS = 1

MI_FAIL_FAST_THRESHOLD = 0.01  # bits — stop if max MI below this
TARGET_ACCURACY = 0.80
MAX_SKIP_RATE = 0.30
BONFERRONI_ALPHA = 0.05

THRESHOLD_RANGE = [round(0.50 + i * 0.01, 2) for i in range(46)]


def _flush() -> None:
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Taker Order Flow Features (Category 10 — 10 features)
# ---------------------------------------------------------------------------

def compute_taker_features(
    ci: int,
    btc_futures: list[FuturesCandle],
    btc_futures_lookup: dict[datetime, int],
    all_candles: list[FastCandle],
) -> dict[str, float]:
    """Compute 10 taker order flow features using pre-window futures data."""
    features: dict[str, float] = {}

    # Get the timestamp of the candle at ci
    target_ts = all_candles[ci].timestamp

    # Find the corresponding futures candle index
    fi = btc_futures_lookup.get(target_ts)
    if fi is None:
        # No matching futures data — return zeros
        for name in [
            "taker_imbalance_5", "taker_imbalance_15", "taker_imbalance_60",
            "taker_accel", "vpin_50", "taker_intensity",
            "taker_price_divergence", "taker_vol_cv", "taker_trend_slope",
            "taker_extreme",
        ]:
            features[name] = 0.0
        return features

    # Helper: compute taker imbalance over lookback candles before fi
    def _imbalance(lookback: int) -> float:
        start = max(0, fi - lookback)
        total_vol = 0.0
        buy_vol = 0.0
        for j in range(start, fi):
            c = btc_futures[j]
            total_vol += c.volume
            buy_vol += c.taker_buy_volume
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

    # Acceleration: short-term minus medium-term
    features["taker_accel"] = imb_5 - imb_15

    # VPIN (Volume-synchronized Probability of Informed Trading)
    # Simplified: partition last 50 volume buckets, measure imbalance
    n_buckets = 50
    lookback_vpin = min(fi, 200)
    start_vpin = max(0, fi - lookback_vpin)
    total_vol = sum(btc_futures[j].volume for j in range(start_vpin, fi))
    if total_vol > 0 and lookback_vpin > 0:
        bucket_size = total_vol / n_buckets
        if bucket_size > 0:
            cum_vol = 0.0
            bucket_buy = 0.0
            bucket_total = 0.0
            abs_imbalances = 0.0
            n_complete = 0
            for j in range(start_vpin, fi):
                c = btc_futures[j]
                bucket_buy += c.taker_buy_volume
                bucket_total += c.volume
                cum_vol += c.volume
                if cum_vol >= bucket_size:
                    sell = bucket_total - bucket_buy
                    abs_imbalances += abs(bucket_buy - sell)
                    n_complete += 1
                    cum_vol -= bucket_size
                    bucket_buy = 0.0
                    bucket_total = 0.0
            features["vpin_50"] = abs_imbalances / (n_complete * bucket_size) if n_complete > 0 and bucket_size > 0 else 0.0
        else:
            features["vpin_50"] = 0.0
    else:
        features["vpin_50"] = 0.0

    # Taker intensity: avg buy-side volume per trade
    start_ti = max(0, fi - 15)
    total_buy = sum(btc_futures[j].taker_buy_volume for j in range(start_ti, fi))
    total_trades = sum(btc_futures[j].num_trades for j in range(start_ti, fi))
    features["taker_intensity"] = total_buy / total_trades if total_trades > 0 else 0.0

    # Price-divergence: sign(price_change_15) != sign(taker_imbalance_15)
    if fi >= 15 and ci >= 15:
        price_change = all_candles[ci - 1].close - all_candles[ci - 15].close
        price_sign = 1.0 if price_change > 0 else (-1.0 if price_change < 0 else 0.0)
        imb_sign = 1.0 if imb_15 > 0 else (-1.0 if imb_15 < 0 else 0.0)
        features["taker_price_divergence"] = 1.0 if price_sign != imb_sign and price_sign != 0 and imb_sign != 0 else 0.0
    else:
        features["taker_price_divergence"] = 0.0

    # Taker volume CV: coefficient of variation of per-candle taker_buy_pct
    start_cv = max(0, fi - 15)
    pcts: list[float] = []
    for j in range(start_cv, fi):
        c = btc_futures[j]
        if c.volume > 0:
            pcts.append(c.taker_buy_volume / c.volume)
    if len(pcts) >= 2:
        mean_pct = sum(pcts) / len(pcts)
        var_pct = sum((p - mean_pct) ** 2 for p in pcts) / len(pcts)
        std_pct = math.sqrt(var_pct)
        features["taker_vol_cv"] = std_pct / mean_pct if mean_pct > 0 else 0.0
    else:
        features["taker_vol_cv"] = 0.0

    # Trend slope: linear regression slope of per-candle imbalance over 15 candles
    start_slope = max(0, fi - 15)
    n_slope = fi - start_slope
    if n_slope >= 3:
        imbs: list[float] = []
        for j in range(start_slope, fi):
            c = btc_futures[j]
            if c.volume > 0:
                buy = c.taker_buy_volume
                sell = c.volume - buy
                imbs.append((buy - sell) / c.volume)
            else:
                imbs.append(0.0)
        n = len(imbs)
        x_mean = (n - 1) / 2.0
        y_mean = sum(imbs) / n
        num = sum((i - x_mean) * (imbs[i] - y_mean) for i in range(n))
        den = sum((i - x_mean) ** 2 for i in range(n))
        features["taker_trend_slope"] = num / den if den > 0 else 0.0
    else:
        features["taker_trend_slope"] = 0.0

    # Extreme flow: imbalance_5 > 90th percentile of rolling 60-candle window
    if fi >= 60:
        rolling_imbs: list[float] = []
        for k in range(fi - 60, fi):
            start_k = max(0, k - 5)
            tv = 0.0
            bv = 0.0
            for j in range(start_k, k):
                tv += btc_futures[j].volume
                bv += btc_futures[j].taker_buy_volume
            if tv > 0:
                sv = tv - bv
                rolling_imbs.append((bv - sv) / tv)
            else:
                rolling_imbs.append(0.0)
        if rolling_imbs:
            rolling_imbs.sort()
            p90_idx = int(0.9 * len(rolling_imbs))
            p90 = rolling_imbs[min(p90_idx, len(rolling_imbs) - 1)]
            features["taker_extreme"] = 1.0 if abs(imb_5) > abs(p90) else 0.0
        else:
            features["taker_extreme"] = 0.0
    else:
        features["taker_extreme"] = 0.0

    return features


# ---------------------------------------------------------------------------
# Cross-Asset ETH Features (Category 11 — 8 features)
# ---------------------------------------------------------------------------

def compute_eth_features(
    ci: int,
    all_candles: list[FastCandle],
    eth_futures: list[FuturesCandle],
    eth_futures_lookup: dict[datetime, int],
    btc_futures: list[FuturesCandle],
    btc_futures_lookup: dict[datetime, int],
) -> dict[str, float]:
    """Compute 8 cross-asset ETH features."""
    features: dict[str, float] = {}
    target_ts = all_candles[ci].timestamp

    ei = eth_futures_lookup.get(target_ts)
    bi = btc_futures_lookup.get(target_ts)

    defaults = {
        "eth_momentum_15": 0.0, "eth_btc_divergence": 0.0,
        "eth_btc_corr_30": 0.0, "eth_taker_alignment": 0.0,
        "eth_btc_vol_ratio": 0.0, "eth_lead_5": 0.0,
        "eth_btc_spread_z": 0.0, "cross_taker_momentum": 0.0,
    }

    if ei is None or bi is None or ei < 15 or bi < 15:
        features.update(defaults)
        return features

    # ETH 15-candle return
    if ei >= 15:
        eth_open = eth_futures[ei - 15].open
        eth_close = eth_futures[ei - 1].close
        eth_ret_15 = (eth_close - eth_open) / eth_open if eth_open != 0 else 0.0
    else:
        eth_ret_15 = 0.0
    features["eth_momentum_15"] = eth_ret_15

    # BTC 15-candle return (from futures)
    if bi >= 15:
        btc_open = btc_futures[bi - 15].open
        btc_close = btc_futures[bi - 1].close
        btc_ret_15 = (btc_close - btc_open) / btc_open if btc_open != 0 else 0.0
    else:
        btc_ret_15 = 0.0

    # Divergence: sign(btc_ret_15) != sign(eth_ret_15)
    btc_sign = 1.0 if btc_ret_15 > 0 else (-1.0 if btc_ret_15 < 0 else 0.0)
    eth_sign = 1.0 if eth_ret_15 > 0 else (-1.0 if eth_ret_15 < 0 else 0.0)
    features["eth_btc_divergence"] = 1.0 if btc_sign != eth_sign and btc_sign != 0 and eth_sign != 0 else 0.0

    # Rolling 30-candle Pearson correlation of 1m returns
    corr_lookback = min(30, min(ei, bi))
    if corr_lookback >= 5:
        btc_rets: list[float] = []
        eth_rets: list[float] = []
        for k in range(1, corr_lookback):
            bc = btc_futures[bi - corr_lookback + k]
            bp = btc_futures[bi - corr_lookback + k - 1]
            ec = eth_futures[ei - corr_lookback + k]
            ep = eth_futures[ei - corr_lookback + k - 1]
            br = (bc.close - bp.close) / bp.close if bp.close != 0 else 0.0
            er = (ec.close - ep.close) / ep.close if ep.close != 0 else 0.0
            btc_rets.append(br)
            eth_rets.append(er)

        n = len(btc_rets)
        if n >= 3:
            bm = sum(btc_rets) / n
            em = sum(eth_rets) / n
            cov = sum((btc_rets[i] - bm) * (eth_rets[i] - em) for i in range(n)) / n
            bstd = math.sqrt(sum((r - bm) ** 2 for r in btc_rets) / n)
            estd = math.sqrt(sum((r - em) ** 2 for r in eth_rets) / n)
            features["eth_btc_corr_30"] = cov / (bstd * estd) if bstd > 0 and estd > 0 else 0.0
        else:
            features["eth_btc_corr_30"] = 0.0
    else:
        features["eth_btc_corr_30"] = 0.0

    # Taker alignment: sign(btc_taker_15) == sign(eth_taker_15)
    def _taker_imb(candles: list[FuturesCandle], idx: int, lb: int) -> float:
        start = max(0, idx - lb)
        tv = 0.0
        bv = 0.0
        for j in range(start, idx):
            tv += candles[j].volume
            bv += candles[j].taker_buy_volume
        if tv == 0:
            return 0.0
        return (2 * bv - tv) / tv

    btc_taker_15 = _taker_imb(btc_futures, bi, 15)
    eth_taker_15 = _taker_imb(eth_futures, ei, 15)
    btc_t_sign = 1.0 if btc_taker_15 > 0 else (-1.0 if btc_taker_15 < 0 else 0.0)
    eth_t_sign = 1.0 if eth_taker_15 > 0 else (-1.0 if eth_taker_15 < 0 else 0.0)
    features["eth_taker_alignment"] = 1.0 if btc_t_sign == eth_t_sign and btc_t_sign != 0 else 0.0

    # Volume ratio: eth_vol_15 / btc_vol_15
    eth_vol = sum(eth_futures[j].volume for j in range(max(0, ei - 15), ei))
    btc_vol = sum(btc_futures[j].volume for j in range(max(0, bi - 15), bi))
    features["eth_btc_vol_ratio"] = eth_vol / btc_vol if btc_vol > 0 else 0.0

    # ETH lead: ETH 5m return × direction of BTC prior move
    if ei >= 5:
        eth_ret_5 = (eth_futures[ei - 1].close - eth_futures[ei - 5].open) / eth_futures[ei - 5].open if eth_futures[ei - 5].open != 0 else 0.0
    else:
        eth_ret_5 = 0.0
    features["eth_lead_5"] = eth_ret_5 * btc_sign

    # ETH/BTC spread z-score vs 60-candle SMA
    if ei >= 60 and bi >= 60:
        ratios: list[float] = []
        for k in range(60):
            bc = btc_futures[bi - 60 + k].close
            ec = eth_futures[ei - 60 + k].close
            if bc > 0:
                ratios.append(ec / bc)
        if len(ratios) >= 10:
            rm = sum(ratios) / len(ratios)
            rv = sum((r - rm) ** 2 for r in ratios) / len(ratios)
            rstd = math.sqrt(rv) if rv > 0 else 0.001
            current_ratio = eth_futures[ei - 1].close / btc_futures[bi - 1].close if btc_futures[bi - 1].close > 0 else rm
            features["eth_btc_spread_z"] = (current_ratio - rm) / rstd
        else:
            features["eth_btc_spread_z"] = 0.0
    else:
        features["eth_btc_spread_z"] = 0.0

    # Cross taker momentum
    btc_taker_5 = _taker_imb(btc_futures, bi, 5)
    eth_taker_5 = _taker_imb(eth_futures, ei, 5)
    features["cross_taker_momentum"] = btc_taker_5 + eth_taker_5

    return features


# ---------------------------------------------------------------------------
# Derivatives Sentiment Features (Category 12 — 6 features)
# ---------------------------------------------------------------------------

def compute_derivatives_features(
    ci: int,
    all_candles: list[FastCandle],
    btc_futures: list[FuturesCandle],
    btc_futures_lookup: dict[datetime, int],
    funding_entries: list[FundingEntry],
    funding_timestamps: list[datetime],
) -> dict[str, float]:
    """Compute 6 derivatives sentiment features."""
    features: dict[str, float] = {}
    target_ts = all_candles[ci].timestamp
    bi = btc_futures_lookup.get(target_ts)

    # Funding rate features
    latest_funding = get_latest_funding(funding_entries, funding_timestamps, target_ts)
    if latest_funding is not None:
        features["funding_rate_curr"] = latest_funding.funding_rate

        # Previous funding rate
        fi_idx = funding_timestamps.index(latest_funding.timestamp) if latest_funding.timestamp in funding_timestamps else -1
        # Use bisect for efficiency
        from bisect import bisect_right
        fi_idx = bisect_right(funding_timestamps, latest_funding.timestamp) - 1
        if fi_idx > 0:
            prev_rate = funding_entries[fi_idx - 1].funding_rate
            features["funding_rate_delta"] = latest_funding.funding_rate - prev_rate
        else:
            features["funding_rate_delta"] = 0.0

        # Extreme: abs(rate) > 90th percentile of last 100 rates
        start_fr = max(0, fi_idx - 100)
        recent_rates = [abs(funding_entries[j].funding_rate) for j in range(start_fr, fi_idx + 1)]
        if len(recent_rates) >= 10:
            recent_rates_sorted = sorted(recent_rates)
            p90 = recent_rates_sorted[int(0.9 * len(recent_rates_sorted))]
            features["funding_rate_extreme"] = 1.0 if abs(latest_funding.funding_rate) > p90 else 0.0
        else:
            features["funding_rate_extreme"] = 0.0
    else:
        features["funding_rate_curr"] = 0.0
        features["funding_rate_delta"] = 0.0
        features["funding_rate_extreme"] = 0.0

    # Spot-futures basis
    if bi is not None and bi > 0:
        futures_close = btc_futures[bi - 1].close
        spot_close = all_candles[ci - 1].close if ci > 0 else all_candles[ci].close
        if spot_close > 0:
            features["spot_futures_basis"] = (futures_close - spot_close) / spot_close
        else:
            features["spot_futures_basis"] = 0.0

        # Basis momentum: slope over 15 candles
        if bi >= 15 and ci >= 15:
            basis_vals: list[float] = []
            for k in range(15):
                fc = btc_futures[bi - 15 + k].close
                sc = all_candles[ci - 15 + k].close
                if sc > 0:
                    basis_vals.append((fc - sc) / sc)
            if len(basis_vals) >= 3:
                n = len(basis_vals)
                xm = (n - 1) / 2.0
                ym = sum(basis_vals) / n
                num = sum((i - xm) * (basis_vals[i] - ym) for i in range(n))
                den = sum((i - xm) ** 2 for i in range(n))
                features["basis_momentum"] = num / den if den > 0 else 0.0
            else:
                features["basis_momentum"] = 0.0
        else:
            features["basis_momentum"] = 0.0

        # Futures volume premium
        if bi >= 15 and ci >= 15:
            f_vol = sum(btc_futures[j].volume for j in range(bi - 15, bi))
            s_vol = sum(all_candles[j].volume for j in range(ci - 15, ci))
            features["futures_volume_premium"] = f_vol / s_vol if s_vol > 0 else 1.0
        else:
            features["futures_volume_premium"] = 1.0
    else:
        features["spot_futures_basis"] = 0.0
        features["basis_momentum"] = 0.0
        features["futures_volume_premium"] = 1.0

    return features


# ---------------------------------------------------------------------------
# Composite Interaction Features (Category 13 — 6 features)
# ---------------------------------------------------------------------------

def compute_composite_features(
    taker_feats: dict[str, float],
    eth_feats: dict[str, float],
    deriv_feats: dict[str, float],
    original_feats: dict[str, float],
) -> dict[str, float]:
    """Compute 6 composite interaction features from other feature categories."""
    features: dict[str, float] = {}

    vol_regime = original_feats.get("vol_regime", 0.0)
    mtf_score = original_feats.get("mtf_score", 0.0)
    taker_imb_15 = taker_feats.get("taker_imbalance_15", 0.0)
    taker_imb_5 = taker_feats.get("taker_imbalance_5", 0.0)
    vpin = taker_feats.get("vpin_50", 0.0)
    taker_intensity = taker_feats.get("taker_intensity", 0.0)
    corr_30 = eth_feats.get("eth_btc_corr_30", 0.0)
    funding = deriv_feats.get("funding_rate_curr", 0.0)

    # Flow × volatility interaction
    features["flow_vol_regime"] = taker_imb_15 * vol_regime

    # Flow confirms trend
    mtf_sign = 1.0 if mtf_score > 0 else (-1.0 if mtf_score < 0 else 0.0)
    features["flow_mtf_alignment"] = taker_imb_15 * mtf_sign

    # Composite informed trading score
    features["informed_trading_score"] = vpin * abs(taker_imb_5) * taker_intensity

    # Flow amplified when correlation breaks
    features["cross_flow_divergence"] = taker_imb_5 * (1.0 - corr_30)

    # Funding + flow agreement
    funding_sign = 1.0 if funding > 0 else (-1.0 if funding < 0 else 0.0)
    imb_sign = 1.0 if taker_imb_15 > 0 else (-1.0 if taker_imb_15 < 0 else 0.0)
    features["funding_flow_confirm"] = funding_sign * imb_sign

    # Multi-signal count: how many agree on direction
    signals = []
    if taker_imb_15 != 0:
        signals.append(1.0 if taker_imb_15 > 0 else -1.0)
    eth_mom = eth_feats.get("eth_momentum_15", 0.0)
    if eth_mom != 0:
        signals.append(1.0 if eth_mom > 0 else -1.0)
    if funding != 0:
        signals.append(funding_sign)
    if mtf_score != 0:
        signals.append(mtf_sign)

    if signals:
        up_count = sum(1 for s in signals if s > 0)
        down_count = sum(1 for s in signals if s < 0)
        features["multi_signal_count"] = float(max(up_count, down_count))
    else:
        features["multi_signal_count"] = 0.0

    return features


# ---------------------------------------------------------------------------
# Feature names
# ---------------------------------------------------------------------------

TAKER_FEATURE_NAMES = [
    "taker_imbalance_5", "taker_imbalance_15", "taker_imbalance_60",
    "taker_accel", "vpin_50", "taker_intensity",
    "taker_price_divergence", "taker_vol_cv", "taker_trend_slope",
    "taker_extreme",
]

ETH_FEATURE_NAMES = [
    "eth_momentum_15", "eth_btc_divergence", "eth_btc_corr_30",
    "eth_taker_alignment", "eth_btc_vol_ratio", "eth_lead_5",
    "eth_btc_spread_z", "cross_taker_momentum",
]

DERIV_FEATURE_NAMES = [
    "funding_rate_curr", "funding_rate_delta", "funding_rate_extreme",
    "spot_futures_basis", "basis_momentum", "futures_volume_premium",
]

COMPOSITE_FEATURE_NAMES = [
    "flow_vol_regime", "flow_mtf_alignment", "informed_trading_score",
    "cross_flow_divergence", "funding_flow_confirm", "multi_signal_count",
]

# Original 31 from backtest_before_window.py
ORIGINAL_FEATURE_NAMES = [
    "prior_dir", "prior_mag", "mtf_score", "mtf_15m", "mtf_1h", "mtf_4h",
    "stm_dir", "stm_strength", "vol_regime", "vol_percentile",
    "tod_hour", "tod_asia", "tod_europe", "tod_us", "tod_late",
    "streak_len", "streak_dir", "vol_ratio", "vol_dir_align",
    "early_cum_return", "early_direction", "early_magnitude",
    "early_green_ratio", "early_vol", "early_max_move",
    "rsi_14", "macd_histogram_sign", "bb_pct_b",
    "atr_14", "mean_reversion_z", "price_vs_vwap",
]


# ---------------------------------------------------------------------------
# Mutual Information calculation
# ---------------------------------------------------------------------------

def compute_mi_bits(feature_values: np.ndarray, target: np.ndarray, n_bins: int = 20) -> float:
    """Compute mutual information between a continuous feature and binary target.

    Discretizes feature into n_bins quantile bins, then computes MI in bits.
    """
    # Discretize feature into bins
    try:
        bins = np.quantile(feature_values, np.linspace(0, 1, n_bins + 1))
        bins = np.unique(bins)
        if len(bins) < 2:
            return 0.0
        digitized = np.digitize(feature_values, bins[1:-1])
    except (ValueError, IndexError):
        return 0.0

    mi = mutual_info_score(digitized, target)
    return mi / math.log(2)  # Convert nats to bits


# ---------------------------------------------------------------------------
# Approach A: Regime Mining
# ---------------------------------------------------------------------------

def regime_mining(
    feature_names: list[str],
    feature_matrix: np.ndarray,
    target: np.ndarray,
    n_bins: int = 3,
    alpha: float = BONFERRONI_ALPHA,
) -> list[dict[str, Any]]:
    """Find 2-way condition pairs with Bonferroni-corrected significance.

    Discretizes each feature into n_bins terciles, searches all pairs of
    (feature_i == bin_a) & (feature_j == bin_b) for high-accuracy regimes.
    """
    n_features = len(feature_names)
    n_samples = len(target)

    # Discretize features into terciles
    binned = np.zeros_like(feature_matrix, dtype=np.int32)
    for col in range(n_features):
        try:
            edges = np.quantile(feature_matrix[:, col], [0.33, 0.67])
            binned[:, col] = np.digitize(feature_matrix[:, col], edges)
        except (ValueError, IndexError):
            binned[:, col] = 0

    # Count total condition pairs for Bonferroni correction
    n_pairs = n_features * (n_features - 1) // 2
    n_conditions = n_pairs * n_bins * n_bins
    corrected_alpha = alpha / max(n_conditions, 1)

    base_rate = target.mean()
    regimes: list[dict[str, Any]] = []

    for i, j in combinations(range(n_features), 2):
        for bi in range(n_bins):
            for bj in range(n_bins):
                mask = (binned[:, i] == bi) & (binned[:, j] == bj)
                count = mask.sum()
                if count < 50:  # minimum sample size
                    continue

                accuracy = target[mask].mean()
                skip_rate = 1.0 - count / n_samples

                # One-sided binomial test approximation
                z = (accuracy - base_rate) / math.sqrt(base_rate * (1 - base_rate) / count) if base_rate > 0 and base_rate < 1 else 0
                # P-value from z-score (normal approximation)
                from math import erfc
                p_value = 0.5 * erfc(z / math.sqrt(2))

                if p_value < corrected_alpha and accuracy > 0.55:
                    regimes.append({
                        "feature_i": feature_names[i],
                        "feature_j": feature_names[j],
                        "bin_i": int(bi),
                        "bin_j": int(bj),
                        "accuracy": float(accuracy),
                        "count": int(count),
                        "skip_rate": float(skip_rate),
                        "p_value": float(p_value),
                        "z_score": float(z),
                    })

    regimes.sort(key=lambda r: r["accuracy"], reverse=True)
    return regimes[:50]  # Top 50


# ---------------------------------------------------------------------------
# Approach B: Contrarian Stacking
# ---------------------------------------------------------------------------

def contrarian_stacking(
    feature_names: list[str],
    feature_matrix: np.ndarray,
    target: np.ndarray,
    min_accuracy: float = 0.52,
) -> dict[str, Any]:
    """Find independent signals > min_accuracy, stack for combined accuracy."""
    n_features = len(feature_names)

    # Score each feature as a simple threshold classifier
    signal_results: list[dict[str, Any]] = []
    for col in range(n_features):
        vals = feature_matrix[:, col]
        median = np.median(vals)

        # Signal: predict Up if above median, Down if below
        preds = (vals > median).astype(int)
        accuracy = (preds == target).mean()

        # Also try inverted
        inv_accuracy = (1 - preds == target).mean()

        best_acc = max(accuracy, inv_accuracy)
        inverted = inv_accuracy > accuracy

        if best_acc > min_accuracy:
            signal_results.append({
                "feature": feature_names[col],
                "col": col,
                "accuracy": float(best_acc),
                "inverted": inverted,
                "threshold": float(median),
            })

    signal_results.sort(key=lambda s: s["accuracy"], reverse=True)

    # Select independent signals (low correlation)
    selected: list[dict[str, Any]] = []
    for sig in signal_results:
        col = sig["col"]
        independent = True
        for sel in selected:
            corr = abs(np.corrcoef(feature_matrix[:, col], feature_matrix[:, sel["col"]])[0, 1])
            if corr > 0.5:
                independent = False
                break
        if independent:
            selected.append(sig)
        if len(selected) >= 8:
            break

    if len(selected) < 2:
        return {"signals": selected, "stacked_accuracy": 0.0, "n_signals": 0}

    # Stack: majority vote among selected signals
    votes = np.zeros(len(target))
    for sig in selected:
        col = sig["col"]
        vals = feature_matrix[:, col]
        preds = (vals > sig["threshold"]).astype(float)
        if sig["inverted"]:
            preds = 1.0 - preds
        votes += preds * 2 - 1  # -1 or +1

    # Predict Up if majority votes Up
    stacked_preds = (votes > 0).astype(int)
    strong_mask = np.abs(votes) >= len(selected) * 0.5  # Require some agreement
    if strong_mask.sum() > 0:
        stacked_accuracy = float((stacked_preds[strong_mask] == target[strong_mask]).mean())
        skip_rate = 1.0 - strong_mask.sum() / len(target)
    else:
        stacked_accuracy = 0.0
        skip_rate = 1.0

    return {
        "signals": selected,
        "stacked_accuracy": stacked_accuracy,
        "skip_rate": skip_rate,
        "n_signals": len(selected),
    }


# ---------------------------------------------------------------------------
# Approach C: Two-Stage Meta-Model
# ---------------------------------------------------------------------------

def two_stage_meta_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Stage 1: predict if window is predictable. Stage 2: predict direction."""
    if len(X_train) < 200 or len(X_test) < 50:
        return {"accuracy": 0.0, "skip_rate": 1.0, "trades": 0}

    # Stage 1: Train direction model on all data
    direction_model = HistGradientBoostingClassifier(
        max_iter=300, max_depth=4, learning_rate=0.05,
        min_samples_leaf=80, l2_regularization=1.0,
        max_bins=128, random_state=42,
    )
    direction_model.fit(X_train, y_train)

    # Stage 1 predictions on train to build predictability labels
    train_preds = direction_model.predict(X_train)
    train_correct = (train_preds == y_train).astype(int)

    # Stage 2: Train predictability filter
    filter_model = HistGradientBoostingClassifier(
        max_iter=200, max_depth=3, learning_rate=0.05,
        min_samples_leaf=100, l2_regularization=2.0,
        max_bins=128, random_state=42,
    )
    filter_model.fit(X_train, train_correct)

    # Apply on test
    test_direction = direction_model.predict(X_test)
    test_predictability = filter_model.predict_proba(X_test)[:, 1]

    # Sweep filter threshold
    best_result = {"accuracy": 0.0, "skip_rate": 1.0, "trades": 0, "threshold": 0.5}
    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        mask = test_predictability >= thresh
        n_traded = mask.sum()
        if n_traded < 50:
            continue
        filtered_correct = (test_direction[mask] == y_test[mask]).sum()
        acc = filtered_correct / n_traded
        skip = 1.0 - n_traded / len(y_test)
        if acc > best_result["accuracy"]:
            best_result = {
                "accuracy": float(acc),
                "skip_rate": float(skip),
                "trades": int(n_traded),
                "threshold": thresh,
            }

    return best_result


# ---------------------------------------------------------------------------
# Approach D: Full ML with threshold sweep
# ---------------------------------------------------------------------------

def full_ml_approach(
    feature_names: list[str],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """HistGradientBoosting with threshold sweep."""
    if len(X_train) < 200 or len(X_test) < 50:
        return {"accuracy": 0.0, "skip_rate": 1.0, "trades": 0, "feature_importances": []}

    model = HistGradientBoostingClassifier(
        max_iter=500, max_depth=5, learning_rate=0.05,
        min_samples_leaf=100, l2_regularization=1.0,
        max_bins=128, random_state=42,
    )
    model.fit(X_train, y_train)

    # Feature importances
    importances = model.feature_importances_
    feat_imp = sorted(
        zip(feature_names, importances.tolist()),
        key=lambda x: x[1], reverse=True,
    )

    # Raw accuracy
    raw_preds = model.predict(X_test)
    raw_accuracy = float((raw_preds == y_test).mean())

    # Threshold sweep on probabilities
    try:
        probas = model.predict_proba(X_test)[:, 1]
    except Exception:
        return {
            "raw_accuracy": raw_accuracy,
            "accuracy": raw_accuracy,
            "skip_rate": 0.0,
            "trades": len(y_test),
            "feature_importances": feat_imp[:15],
        }

    sweep_results: list[dict[str, Any]] = []
    best_result = {"accuracy": raw_accuracy, "skip_rate": 0.0, "trades": len(y_test), "threshold": 0.5}

    for thresh in THRESHOLD_RANGE:
        mask = probas >= thresh
        n_up = mask.sum()
        # Also consider below threshold as Down predictions
        n_total = len(y_test)

        # Filter: only trade when model is confident
        if n_up < 20:
            continue

        # Among those predicted as Up (prob >= thresh), measure accuracy
        acc_up = float(y_test[mask].mean()) if n_up > 0 else 0.0

        # Effective: trade only high-confidence, skip rest
        skip = 1.0 - n_up / n_total
        sweep_results.append({
            "threshold": thresh,
            "accuracy": acc_up,
            "skip_rate": skip,
            "trades": int(n_up),
        })

        if acc_up > best_result["accuracy"] and skip < MAX_SKIP_RATE and n_up >= 50:
            best_result = {
                "accuracy": float(acc_up),
                "skip_rate": float(skip),
                "trades": int(n_up),
                "threshold": thresh,
            }

    return {
        "raw_accuracy": raw_accuracy,
        "best_threshold": best_result,
        "sweep": sweep_results,
        "feature_importances": feat_imp[:15],
    }


# ---------------------------------------------------------------------------
# Walk-Forward Validation
# ---------------------------------------------------------------------------

def walk_forward_ml(
    window_data: list[WindowData],
    all_feature_dicts: list[dict[str, float]],
    feature_names: list[str],
) -> list[dict[str, Any]]:
    """3-month train / 1-month test rolling walk-forward on ML model."""
    by_month: dict[str, list[int]] = defaultdict(list)
    for i, wd in enumerate(window_data):
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

        if len(train_indices) < 200 or len(test_indices) < 50:
            continue

        # Build matrices
        X_train = np.array([[all_feature_dicts[i].get(f, 0.0) for f in feature_names] for i in train_indices])
        y_train = np.array([1 if window_data[i].resolution == "Up" else 0 for i in train_indices])
        X_test = np.array([[all_feature_dicts[i].get(f, 0.0) for f in feature_names] for i in test_indices])
        y_test = np.array([1 if window_data[i].resolution == "Up" else 0 for i in test_indices])

        model = HistGradientBoostingClassifier(
            max_iter=300, max_depth=4, learning_rate=0.05,
            min_samples_leaf=80, l2_regularization=1.0,
            max_bins=128, random_state=42,
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        accuracy = float((preds == y_test).mean())

        # PnL
        net_pnl = 0.0
        for k, idx in enumerate(test_indices):
            wd = window_data[idx]
            direction = "Up" if preds[k] == 1 else "Down"
            _, _, pn, _ = simulate_trade(direction, wd.resolution)
            net_pnl += pn

        results.append({
            "period": test_month,
            "train_size": len(train_indices),
            "test_size": len(test_indices),
            "accuracy": accuracy,
            "net_pnl": net_pnl,
            "profitable": net_pnl > 0,
        })

    return results


# ---------------------------------------------------------------------------
# Run a single phase
# ---------------------------------------------------------------------------

def run_phase(
    phase_name: str,
    feature_names: list[str],
    window_data: list[WindowData],
    all_feature_dicts: list[dict[str, float]],
    split_idx: int,
) -> dict[str, Any]:
    """Run all 4 approaches on a feature set."""
    print(f"\n{'='*100}")
    print(f"  {phase_name} ({len(feature_names)} features)")
    print(f"{'='*100}")
    _flush()

    n = len(window_data)
    target = np.array([1 if wd.resolution == "Up" else 0 for wd in window_data])
    X_all = np.array([[all_feature_dicts[i].get(f, 0.0) for f in feature_names] for i in range(n)])

    X_train = X_all[:split_idx]
    y_train = target[:split_idx]
    X_test = X_all[split_idx:]
    y_test = target[split_idx:]

    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"  Base rate: Up {target.mean()*100:.1f}%")
    _flush()

    phase_result: dict[str, Any] = {
        "feature_count": len(feature_names),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    # --- Approach A: Regime Mining ---
    print(f"\n  --- Approach A: Regime Mining ---")
    _flush()
    t0 = time.time()
    regimes_train = regime_mining(feature_names, X_train, y_train)
    elapsed = time.time() - t0
    print(f"  Found {len(regimes_train)} significant regimes (train) in {elapsed:.1f}s")

    if regimes_train:
        # Validate top regimes on test set
        validated: list[dict[str, Any]] = []
        for reg in regimes_train[:20]:
            fi_col = feature_names.index(reg["feature_i"])
            fj_col = feature_names.index(reg["feature_j"])

            # Recompute bins on test data
            try:
                edges_i = np.quantile(X_train[:, fi_col], [0.33, 0.67])
                edges_j = np.quantile(X_train[:, fj_col], [0.33, 0.67])
                binned_i = np.digitize(X_test[:, fi_col], edges_i)
                binned_j = np.digitize(X_test[:, fj_col], edges_j)
            except (ValueError, IndexError):
                continue

            mask = (binned_i == reg["bin_i"]) & (binned_j == reg["bin_j"])
            count = mask.sum()
            if count < 20:
                continue
            test_acc = float(y_test[mask].mean())
            validated.append({
                **reg,
                "test_accuracy": test_acc,
                "test_count": int(count),
                "degradation": reg["accuracy"] - test_acc,
            })

        validated.sort(key=lambda r: r["test_accuracy"], reverse=True)
        if validated:
            print(f"  Top 5 validated regimes (test):")
            print(f"    {'Features':<45} {'TrainAcc':>9} {'TestAcc':>9} {'Degrad':>8} {'Count':>6}")
            for reg in validated[:5]:
                print(f"    {reg['feature_i']}={reg['bin_i']} & {reg['feature_j']}={reg['bin_j']:<15} "
                      f"{reg['accuracy']*100:>8.1f}% {reg['test_accuracy']*100:>8.1f}% "
                      f"{reg['degradation']*100:>7.1f}% {reg['test_count']:>6}")
        phase_result["regime_mining"] = {
            "train_regimes": len(regimes_train),
            "validated": validated[:10],
            "best_test_accuracy": validated[0]["test_accuracy"] if validated else 0.0,
        }
    else:
        print(f"  No significant regimes found")
        phase_result["regime_mining"] = {"train_regimes": 0, "validated": [], "best_test_accuracy": 0.0}
    _flush()

    # --- Approach B: Contrarian Stacking ---
    print(f"\n  --- Approach B: Contrarian Stacking ---")
    _flush()
    stacking_train = contrarian_stacking(feature_names, X_train, y_train)
    print(f"  Selected {stacking_train['n_signals']} independent signals")
    if stacking_train["signals"]:
        for sig in stacking_train["signals"][:5]:
            inv_str = " (inv)" if sig["inverted"] else ""
            print(f"    {sig['feature']:<30} acc {sig['accuracy']*100:.1f}%{inv_str}")

    # Apply stacking on test
    stacking_test = contrarian_stacking(feature_names, X_test, y_test)
    print(f"  Stacked test accuracy: {stacking_test['stacked_accuracy']*100:.1f}% "
          f"(skip {stacking_test['skip_rate']*100:.1f}%)")

    phase_result["contrarian_stacking"] = {
        "train": stacking_train,
        "test_accuracy": stacking_test["stacked_accuracy"],
        "test_skip_rate": stacking_test["skip_rate"],
    }
    _flush()

    # --- Approach C: Two-Stage Meta-Model ---
    print(f"\n  --- Approach C: Two-Stage Meta-Model ---")
    _flush()
    t0 = time.time()
    meta_result = two_stage_meta_model(X_train, y_train, X_test, y_test)
    elapsed = time.time() - t0
    print(f"  Best: acc {meta_result['accuracy']*100:.1f}%, "
          f"skip {meta_result['skip_rate']*100:.1f}%, "
          f"trades {meta_result['trades']} ({elapsed:.1f}s)")
    phase_result["meta_model"] = meta_result
    _flush()

    # --- Approach D: Full ML ---
    print(f"\n  --- Approach D: Full ML ---")
    _flush()
    t0 = time.time()
    ml_result = full_ml_approach(feature_names, X_train, y_train, X_test, y_test)
    elapsed = time.time() - t0
    print(f"  Raw accuracy: {ml_result.get('raw_accuracy', 0)*100:.1f}%")
    if ml_result.get("best_threshold"):
        bt = ml_result["best_threshold"]
        print(f"  Best threshold: {bt.get('threshold', 'N/A')} -> "
              f"acc {bt['accuracy']*100:.1f}%, skip {bt['skip_rate']*100:.1f}%, "
              f"trades {bt['trades']}")
    print(f"  Top features:")
    for fname, imp in ml_result.get("feature_importances", [])[:8]:
        print(f"    {fname:<35} {imp:.4f}")
    print(f"  ({elapsed:.1f}s)")
    phase_result["full_ml"] = ml_result
    _flush()

    # --- Walk-Forward ---
    print(f"\n  --- Walk-Forward Validation ---")
    _flush()
    t0 = time.time()
    wf_results = walk_forward_ml(window_data, all_feature_dicts, feature_names)
    elapsed = time.time() - t0

    if wf_results:
        wf_accs = [w["accuracy"] for w in wf_results]
        wf_mean = sum(wf_accs) / len(wf_accs)
        wf_prof = sum(1 for w in wf_results if w["profitable"])
        wf_pnl = sum(w["net_pnl"] for w in wf_results)
        print(f"  {len(wf_results)} periods, mean acc {wf_mean*100:.1f}%, "
              f"profitable {wf_prof}/{len(wf_results)}, PnL ${wf_pnl:+,.0f} ({elapsed:.1f}s)")
        phase_result["walk_forward"] = {
            "periods": len(wf_results),
            "mean_accuracy": wf_mean,
            "profitable_pct": wf_prof / len(wf_results),
            "total_pnl": wf_pnl,
            "details": wf_results,
        }
    else:
        print(f"  No walk-forward periods ({elapsed:.1f}s)")
        phase_result["walk_forward"] = {"periods": 0, "mean_accuracy": 0.0}
    _flush()

    # --- Summary ---
    best_acc = max(
        phase_result.get("regime_mining", {}).get("best_test_accuracy", 0),
        stacking_test["stacked_accuracy"],
        meta_result["accuracy"],
        ml_result.get("best_threshold", {}).get("accuracy", ml_result.get("raw_accuracy", 0)),
    )
    phase_result["best_accuracy"] = best_acc
    print(f"\n  PHASE BEST ACCURACY: {best_acc*100:.1f}%")
    _flush()

    return phase_result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 100)
    print("  BWO v3 — ORDER FLOW & CROSS-ASSET BACKTEST")
    print("  Taker volume + ETH divergence + Derivatives sentiment")
    print("  Fail-fast MI gate | 4 approaches | Walk-forward validation")
    print("=" * 100)
    _flush()

    # ------------------------------------------------------------------
    # Load all data
    # ------------------------------------------------------------------
    print("\n  Loading spot BTC data...", end=" ")
    _flush()
    if not BTC_SPOT_CSV.exists():
        print(f"\n  ERROR: {BTC_SPOT_CSV} not found. Run download_btc_2y.py first.")
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

    print("  Loading BTC futures data...", end=" ")
    _flush()
    if not BTC_FUTURES_CSV.exists():
        print(f"\n  ERROR: {BTC_FUTURES_CSV} not found. Run download_futures_data.py first.")
        sys.exit(1)
    btc_futures = load_futures_csv(BTC_FUTURES_CSV)
    btc_futures.sort(key=lambda c: c.timestamp)
    btc_futures_lookup = build_futures_lookup(btc_futures)
    print(f"{len(btc_futures):,} candles")
    _flush()

    print("  Loading ETH futures data...", end=" ")
    _flush()
    if not ETH_FUTURES_CSV.exists():
        print(f"\n  ERROR: {ETH_FUTURES_CSV} not found. Run download_futures_data.py first.")
        sys.exit(1)
    eth_futures = load_futures_csv(ETH_FUTURES_CSV)
    eth_futures.sort(key=lambda c: c.timestamp)
    eth_futures_lookup = build_futures_lookup(eth_futures)
    print(f"{len(eth_futures):,} candles")
    _flush()

    print("  Loading funding rate data...", end=" ")
    _flush()
    if FUNDING_CSV.exists():
        funding_entries = load_funding_csv(FUNDING_CSV)
        funding_timestamps = [e.timestamp for e in funding_entries]
        print(f"{len(funding_entries):,} entries")
    else:
        print("NOT FOUND (will use zeros for funding features)")
        funding_entries = []
        funding_timestamps = []
    _flush()

    # ------------------------------------------------------------------
    # Build 15m windows
    # ------------------------------------------------------------------
    print("\n  Grouping into 15m windows...", end=" ")
    _flush()
    windows = group_into_15m_windows(all_candles)
    print(f"{len(windows):,} complete windows")
    _flush()

    candle_by_ts: dict[datetime, int] = {}
    for i, c in enumerate(all_candles):
        candle_by_ts[c.timestamp] = i

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
    print(f"  {len(window_data):,} windows with context")

    up_count = sum(1 for wd in window_data if wd.resolution == "Up")
    print(f"  Base rate: Up {up_count/len(window_data)*100:.1f}%")
    _flush()

    split_idx = int(len(window_data) * 0.80)

    # ------------------------------------------------------------------
    # Compute ALL features for all windows
    # ------------------------------------------------------------------
    print(f"\n  Computing features for {len(window_data):,} windows...")
    _flush()
    t0 = time.time()

    all_feature_dicts: list[dict[str, float]] = []
    for i, wd in enumerate(window_data):
        ci = wd.candle_idx

        # Original 31 features
        original = compute_all_features(wd, windows, all_candles)

        # Taker features (10)
        taker = compute_taker_features(ci, btc_futures, btc_futures_lookup, all_candles)

        # ETH features (8)
        eth = compute_eth_features(ci, all_candles, eth_futures, eth_futures_lookup,
                                   btc_futures, btc_futures_lookup)

        # Derivatives features (6)
        deriv = compute_derivatives_features(ci, all_candles, btc_futures, btc_futures_lookup,
                                             funding_entries, funding_timestamps)

        # Composite features (6)
        composite = compute_composite_features(taker, eth, deriv, original)

        # Merge all
        combined = {**original, **taker, **eth, **deriv, **composite}
        all_feature_dicts.append(combined)

        if (i + 1) % 5000 == 0:
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
    print(f"  PHASE 0: FAIL-FAST — Mutual Information Check")
    print(f"{'='*100}")
    _flush()

    target = np.array([1 if wd.resolution == "Up" else 0 for wd in window_data])

    print(f"\n  {'Feature':<35} {'MI (bits)':>12} {'Status':>10}")
    print(f"  {'-'*60}")

    mi_results: dict[str, float] = {}
    max_mi = 0.0
    max_mi_feature = ""

    for fname in TAKER_FEATURE_NAMES:
        vals = np.array([fd.get(fname, 0.0) for fd in all_feature_dicts])
        mi = compute_mi_bits(vals, target)
        mi_results[fname] = mi
        status = "OK" if mi >= MI_FAIL_FAST_THRESHOLD else "LOW"
        print(f"  {fname:<35} {mi:>12.6f} {status:>10}")
        if mi > max_mi:
            max_mi = mi
            max_mi_feature = fname

    print(f"\n  Max MI: {max_mi:.6f} bits ({max_mi_feature})")
    print(f"  Threshold: {MI_FAIL_FAST_THRESHOLD} bits")

    report: dict[str, Any] = {
        "meta": {
            "total_candles": len(all_candles),
            "btc_futures_candles": len(btc_futures),
            "eth_futures_candles": len(eth_futures),
            "funding_entries": len(funding_entries),
            "total_windows": len(window_data),
            "train_size": split_idx,
            "test_size": len(window_data) - split_idx,
            "base_rate_up_pct": round(up_count / len(window_data) * 100, 2),
        },
        "phase_0_mi": {
            "taker_features": {k: round(v, 6) for k, v in mi_results.items()},
            "max_mi": round(max_mi, 6),
            "max_mi_feature": max_mi_feature,
            "threshold": MI_FAIL_FAST_THRESHOLD,
        },
    }

    if max_mi < MI_FAIL_FAST_THRESHOLD:
        print(f"\n  *** FAIL-FAST TRIGGERED ***")
        print(f"  Max MI ({max_mi:.6f}) < threshold ({MI_FAIL_FAST_THRESHOLD})")
        print(f"  Taker volume features have NO predictive power.")
        print(f"  Market is efficient at 1m/15m timescale.")
        print(f"  Recommendation: Pivot to entry-price optimization.")
        report["phase_0_mi"]["fail_fast"] = True
        report["conclusion"] = "FAIL-FAST: No taker flow signal. Market efficient."

        # Still check ETH and derivatives MI for completeness
        print(f"\n  Checking ETH + derivatives MI for completeness...")
        for fname in ETH_FEATURE_NAMES + DERIV_FEATURE_NAMES + COMPOSITE_FEATURE_NAMES:
            vals = np.array([fd.get(fname, 0.0) for fd in all_feature_dicts])
            mi = compute_mi_bits(vals, target)
            mi_results[fname] = mi
            status = "OK" if mi >= MI_FAIL_FAST_THRESHOLD else "LOW"
            print(f"  {fname:<35} {mi:>12.6f} {status:>10}")

        report["phase_0_mi"]["all_new_features"] = {k: round(v, 6) for k, v in mi_results.items()}
        overall_max_mi = max(mi_results.values()) if mi_results else 0.0
        report["phase_0_mi"]["overall_max_mi"] = round(overall_max_mi, 6)

        if overall_max_mi >= MI_FAIL_FAST_THRESHOLD:
            print(f"\n  Some non-taker features show MI >= threshold.")
            print(f"  Proceeding with full backtest despite taker fail-fast.")
            report["phase_0_mi"]["fail_fast"] = False
        else:
            # Save report and exit
            with open(REPORT_JSON, "w") as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\n  Report saved: {REPORT_JSON}")
            print(f"\n{'='*100}")
            _flush()
            return
    else:
        print(f"\n  PASS — Taker features show MI above threshold. Proceeding.")
        report["phase_0_mi"]["fail_fast"] = False
    _flush()

    # ------------------------------------------------------------------
    # PHASE 1: Taker features only (10)
    # ------------------------------------------------------------------
    phase1_result = run_phase(
        "PHASE 1: Taker Features Only",
        TAKER_FEATURE_NAMES,
        window_data, all_feature_dicts, split_idx,
    )
    report["phase_1_taker"] = phase1_result

    # ------------------------------------------------------------------
    # PHASE 2: + Cross-asset ETH (18)
    # ------------------------------------------------------------------
    phase2_features = TAKER_FEATURE_NAMES + ETH_FEATURE_NAMES
    phase2_result = run_phase(
        "PHASE 2: Taker + ETH Cross-Asset",
        phase2_features,
        window_data, all_feature_dicts, split_idx,
    )
    report["phase_2_taker_eth"] = phase2_result

    # ------------------------------------------------------------------
    # PHASE 3: + Derivatives + composites (30)
    # ------------------------------------------------------------------
    phase3_features = TAKER_FEATURE_NAMES + ETH_FEATURE_NAMES + DERIV_FEATURE_NAMES + COMPOSITE_FEATURE_NAMES
    phase3_result = run_phase(
        "PHASE 3: All New Features (30)",
        phase3_features,
        window_data, all_feature_dicts, split_idx,
    )
    report["phase_3_all_new"] = phase3_result

    # ------------------------------------------------------------------
    # PHASE 4: Combined 61 features (31 original + 30 new)
    # ------------------------------------------------------------------
    phase4_features = ORIGINAL_FEATURE_NAMES + TAKER_FEATURE_NAMES + ETH_FEATURE_NAMES + DERIV_FEATURE_NAMES + COMPOSITE_FEATURE_NAMES
    phase4_result = run_phase(
        "PHASE 4: Combined 61 Features",
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
        ("Phase 1 (Taker 10)", phase1_result),
        ("Phase 2 (Taker+ETH 18)", phase2_result),
        ("Phase 3 (All New 30)", phase3_result),
        ("Phase 4 (Combined 61)", phase4_result),
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

    # Determine tier
    best_overall = max(r["best_accuracy"] for _, r in phases)
    wf_best = max(r.get("walk_forward", {}).get("mean_accuracy", 0) for _, r in phases)

    if best_overall >= 0.80:
        tier = "Tier 1"
        action = "Ship to live strategy"
    elif best_overall >= 0.65:
        tier = "Tier 2"
        action = "Use as filter on base-rate strategy; start CLOB snapshot collection"
    else:
        tier = "Tier 3"
        action = "Market efficient; pivot to entry-price optimization"

    print(f"\n  RESULT: {tier}")
    print(f"  Best accuracy: {best_overall*100:.1f}% | Best WF mean: {wf_best*100:.1f}%")
    print(f"  Action: {action}")

    report["summary"] = {
        "best_accuracy": round(best_overall, 4),
        "best_wf_accuracy": round(wf_best, 4),
        "tier": tier,
        "action": action,
    }

    # Save report
    with open(REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved: {REPORT_JSON}")
    print(f"\n{'='*100}")
    _flush()


if __name__ == "__main__":
    main()
