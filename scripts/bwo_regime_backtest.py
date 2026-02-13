"""BWO Strategy v2 — Novel Regime Mining Backtest.

Implements 64 novel features (microstructure, entropy, order flow, volatility
structure, cross-scale patterns) on top of the original 31 features.

Four approaches:
  A. Conditional Regime Mining (Bonferroni-corrected)
  B. Contrarian Signal Stacking (Bayesian independence)
  C. Two-Stage Meta-Model (predictability filter + direction)
  D. ML on all 95 features (HistGradientBoosting)

All results via walk-forward validation (3-month train / 1-month test).
Statistical validation: Bootstrap CI, permutation test, Bonferroni correction.
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import KFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fast_loader import FastCandle, group_into_15m_windows, load_csv_fast
from scripts.backtest_before_window import (
    BTC_CSV,
    ENTRY_PRICE,
    POSITION_SIZE,
    TradeResult,
    WindowData,
    bootstrap_ci,
    compute_all_features,
    compute_metrics,
    permutation_test,
    polymarket_fee,
    simulate_trade,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPORT_JSON = PROJECT_ROOT / "data" / "bwo_regime_report.json"
WF_TRAIN_MONTHS = 3
WF_TEST_MONTHS = 1
MIN_REGIME_WINDOWS = 500
BONFERRONI_ALPHA = 0.05

ORIGINAL_FEATURE_NAMES = [
    "prior_dir", "prior_mag", "mtf_score", "mtf_15m", "mtf_1h", "mtf_4h",
    "stm_dir", "stm_strength", "vol_regime", "vol_percentile",
    "tod_hour", "tod_asia", "tod_europe", "tod_us", "tod_late",
    "streak_len", "streak_dir", "vol_ratio", "vol_dir_align",
    "early_cum_return", "early_direction", "early_magnitude",
    "early_green_ratio", "early_vol", "early_max_move",
    "rsi_14", "macd_histogram_sign", "bb_pct_b", "atr_14",
    "mean_reversion_z", "price_vs_vwap",
]

NOVEL_FEATURE_NAMES = [
    # Cat 1: Order Flow Proxy (8)
    "flow_imbalance_5", "flow_imbalance_10", "flow_imbalance_15", "flow_accel",
    "cum_delta_30", "delta_price_divergence", "aggressive_buy_ratio_15", "obv_slope_15",
    # Cat 2: Candle Microstructure (10)
    "hl_position", "body_range_ratio", "upper_wick_ratio", "lower_wick_ratio",
    "is_doji", "is_hammer", "is_shooting_star", "avg_body_range_5", "engulfing", "inside_bar",
    # Cat 3: Volatility Structure (8)
    "range_compression", "parkinson_vol", "garman_klass_vol", "yang_zhang_ratio",
    "vol_of_vol", "variance_ratio", "bollinger_bandwidth", "range_contraction_5",
    # Cat 4: Entropy & Complexity (6)
    "approx_entropy", "sample_entropy", "permutation_entropy",
    "lempel_ziv_complexity", "direction_entropy", "return_kurtosis",
    # Cat 5: Autocorrelation (7)
    "acf_1", "acf_2", "acf_3", "acf_5", "pacf_1", "ljung_box_stat", "hurst_exponent",
    # Cat 6: Volume Dynamics (7)
    "volume_accel", "volume_spike", "volume_spike_ratio", "vol_price_corr",
    "vol_weighted_return", "volume_trend", "volume_cv",
    # Cat 7: Calendar & Cyclical (8)
    "hour_sin", "hour_cos", "minute_sin", "minute_cos",
    "near_funding", "day_of_week", "is_weekend", "us_open_overlap",
    # Cat 8: Cross-Scale Patterns (4)
    "fractal_dim", "dfa_exponent", "high_freq_energy_ratio", "trend_strength_index",
    # Cat 9: Prior Window Deep (6)
    "prior_range_pct", "prior_close_position", "prior_vol_profile",
    "prior_early_vs_late", "prior_reversal_depth", "prior_was_volatile",
]

ALL_FEATURE_NAMES = ORIGINAL_FEATURE_NAMES + NOVEL_FEATURE_NAMES


def _flush():
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Helper: Linear regression
# ---------------------------------------------------------------------------

def _linreg_slope(y_vals):
    """Slope for equally-spaced x values."""
    n = len(y_vals)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(y_vals) / n
    num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(y_vals))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den != 0 else 0.0


def _linreg_slope_xy(x, y):
    """Slope of y = a + b*x."""
    n = len(x)
    if n < 2:
        return 0.0
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xm, ym = np.mean(x), np.mean(y)
    den = np.sum((x - xm) ** 2)
    return float(np.sum((x - xm) * (y - ym)) / den) if den > 0 else 0.0


# ---------------------------------------------------------------------------
# Helper: Entropy & complexity functions
# ---------------------------------------------------------------------------

def _approx_entropy(data, m=2, r_factor=0.2):
    N = len(data)
    if N < m + 2:
        return 0.0
    std = np.std(data)
    if std == 0:
        return 0.0
    r = r_factor * std

    def phi(mv):
        nt = N - mv + 1
        if nt <= 0:
            return 0.0
        templates = np.array([data[i:i + mv] for i in range(nt)])
        C = np.zeros(nt)
        for i in range(nt):
            dists = np.max(np.abs(templates - templates[i]), axis=1)
            C[i] = np.sum(dists <= r) / nt
        valid = C[C > 0]
        return float(np.mean(np.log(valid))) if len(valid) > 0 else 0.0

    return phi(m) - phi(m + 1)


def _sample_entropy(data, m=2, r_factor=0.2):
    N = len(data)
    if N < m + 2:
        return 0.0
    std = np.std(data)
    if std == 0:
        return 0.0
    r = r_factor * std

    def count_matches(mv):
        nt = N - mv
        if nt <= 1:
            return 0
        templates = np.array([data[i:i + mv] for i in range(nt)])
        total = 0
        for i in range(nt):
            dists = np.max(np.abs(templates - templates[i]), axis=1)
            total += int(np.sum(dists <= r)) - 1
        return total

    A = count_matches(m + 1)
    B = count_matches(m)
    if B == 0 or A == 0:
        return 0.0
    return -math.log(A / B)


def _permutation_entropy(data, order=3):
    N = len(data)
    if N < order + 1:
        return 0.0
    perms = []
    for i in range(N - order + 1):
        w = data[i:i + order]
        perms.append(tuple(sorted(range(order), key=lambda k: w[k])))
    counts = Counter(perms)
    total = len(perms)
    probs = [c / total for c in counts.values()]
    max_ent = math.log(math.factorial(order))
    if max_ent == 0:
        return 0.0
    return -sum(p * math.log(p) for p in probs if p > 0) / max_ent


def _lempel_ziv_complexity(binary_seq):
    s = binary_seq
    n = len(s)
    if n <= 1:
        return 0.0
    c, k, l = 1, 1, 1
    while k + l <= n:
        if s[k:k + l] in s[:k + l - 1]:
            l += 1
        else:
            c += 1
            k += l
            l = 1
    return c * math.log2(n) / n if n > 0 else 0.0


def _hurst_exponent(data):
    N = len(data)
    if N < 20:
        return 0.5
    ns, rs_vals = [], []
    for size in range(10, min(N // 2, 50) + 1, 5):
        n_seg = N // size
        if n_seg < 1:
            continue
        rs_list = []
        for seg in range(n_seg):
            segment = data[seg * size:(seg + 1) * size]
            mean_v = np.mean(segment)
            cumdev = np.cumsum(segment - mean_v)
            R = float(np.max(cumdev) - np.min(cumdev))
            S = float(np.std(segment))
            if S > 0:
                rs_list.append(R / S)
        if rs_list:
            ns.append(size)
            rs_vals.append(np.mean(rs_list))
    if len(ns) < 3:
        return 0.5
    slope = _linreg_slope_xy(np.log(ns), np.log(np.array(rs_vals) + 1e-10))
    return max(0.0, min(1.0, slope))


def _higuchi_fd(data, kmax=8):
    N = len(data)
    if N < kmax + 1:
        return 1.5
    L, ks = [], list(range(1, kmax + 1))
    for k in ks:
        Lk, cnt = 0.0, 0
        for m in range(1, k + 1):
            indices = list(range(m - 1, N, k))
            if len(indices) < 2:
                continue
            Lm = sum(abs(data[indices[j + 1]] - data[indices[j]]) for j in range(len(indices) - 1))
            norm_f = (N - 1) / (k * (len(indices) - 1)) if len(indices) > 1 else 1
            Lk += Lm * norm_f / k
            cnt += 1
        L.append(Lk / cnt if cnt > 0 else 0)
    valid = [(k, l) for k, l in zip(ks, L) if l > 0]
    if len(valid) < 3:
        return 1.5
    return _linreg_slope_xy([math.log(1.0 / k) for k, _ in valid],
                            [math.log(l) for _, l in valid])


def _dfa_exponent(data):
    N = len(data)
    if N < 16:
        return 0.5
    y = np.cumsum(data - np.mean(data))
    scales = [s for s in [4, 8, 16, 32, 64] if s <= N // 2]
    if len(scales) < 2:
        return 0.5
    fluct = []
    for scale in scales:
        n_seg = N // scale
        if n_seg < 1:
            continue
        f_list = []
        for seg in range(n_seg):
            segment = y[seg * scale:(seg + 1) * scale]
            x_fit = np.arange(scale)
            try:
                coeffs = np.polyfit(x_fit, segment, 1)
                trend = np.polyval(coeffs, x_fit)
                f_list.append(np.sqrt(np.mean((segment - trend) ** 2)))
            except (np.linalg.LinAlgError, ValueError):
                continue
        if f_list:
            fluct.append((scale, np.mean(f_list)))
    if len(fluct) < 2:
        return 0.5
    return max(0.0, min(2.0, _linreg_slope_xy(
        [math.log(s) for s, _ in fluct], [math.log(f) for _, f in fluct])))


# ---------------------------------------------------------------------------
# Novel Feature Computation (64 features)
# ---------------------------------------------------------------------------

def compute_novel_features(
    wd: WindowData,
    windows: list[list[FastCandle]],
    all_candles: list[FastCandle],
) -> dict[str, float]:
    """Compute 64 novel features using ONLY pre-window data."""
    f: dict[str, float] = {}
    ci = wd.candle_idx
    wi = wd.window_idx
    hist_end = ci  # exclusive — no lookahead
    hist_start = max(0, ci - 1440)
    hist_len = hist_end - hist_start

    # Pre-compute returns for entropy/autocorrelation
    if hist_len >= 30:
        closes_30 = [all_candles[j].close for j in range(hist_end - 30, hist_end)]
        returns_30 = [(closes_30[i] - closes_30[i - 1]) / closes_30[i - 1]
                      if closes_30[i - 1] != 0 else 0.0 for i in range(1, 30)]
    else:
        closes_30, returns_30 = [], []

    # ===== Category 1: Order Flow Proxy (8) =====
    for lookback, name in [(5, "5"), (10, "10"), (15, "15")]:
        if hist_len >= lookback:
            buy_v, sell_v = 0.0, 0.0
            for j in range(hist_end - lookback, hist_end):
                c = all_candles[j]
                rng = c.high - c.low
                if rng > 0:
                    buy_v += c.volume * (c.close - c.low) / rng
                    sell_v += c.volume * (c.high - c.close) / rng
            tot = buy_v + sell_v
            f[f"flow_imbalance_{name}"] = (buy_v - sell_v) / tot if tot > 0 else 0.0
        else:
            f[f"flow_imbalance_{name}"] = 0.0

    f["flow_accel"] = f["flow_imbalance_5"] - f["flow_imbalance_15"]

    if hist_len >= 30:
        cd = 0.0
        for j in range(hist_end - 30, hist_end):
            c = all_candles[j]
            rng = c.high - c.low
            if rng > 0:
                cd += c.volume * (c.close - c.low) / rng - c.volume * (c.high - c.close) / rng
        f["cum_delta_30"] = cd
    else:
        f["cum_delta_30"] = 0.0

    if hist_len >= 15:
        price_d = 1.0 if all_candles[hist_end - 1].close > all_candles[hist_end - 15].close else -1.0
        delta_d = 1.0 if f["flow_imbalance_15"] > 0 else -1.0
        f["delta_price_divergence"] = 1.0 if price_d != delta_d else 0.0
    else:
        f["delta_price_divergence"] = 0.0

    if hist_len >= 15:
        agg = sum(1 for j in range(hist_end - 15, hist_end)
                  if (all_candles[j].high - all_candles[j].low) > 0
                  and (all_candles[j].close - all_candles[j].low) / (all_candles[j].high - all_candles[j].low) > 0.75)
        f["aggressive_buy_ratio_15"] = agg / 15.0
    else:
        f["aggressive_buy_ratio_15"] = 0.0

    if hist_len >= 15:
        obv = [0.0]
        for j in range(hist_end - 14, hist_end):
            c = all_candles[j]
            pc = all_candles[j - 1].close
            if c.close > pc:
                obv.append(obv[-1] + c.volume)
            elif c.close < pc:
                obv.append(obv[-1] - c.volume)
            else:
                obv.append(obv[-1])
        f["obv_slope_15"] = _linreg_slope(obv)
    else:
        f["obv_slope_15"] = 0.0

    # ===== Category 2: Candle Microstructure (10) =====
    if hist_len >= 1:
        last = all_candles[hist_end - 1]
        rng = last.high - last.low
        body = abs(last.close - last.open)
        if rng > 0:
            f["hl_position"] = (last.close - last.low) / rng
            f["body_range_ratio"] = body / rng
            f["upper_wick_ratio"] = (last.high - max(last.open, last.close)) / rng
            f["lower_wick_ratio"] = (min(last.open, last.close) - last.low) / rng
            f["is_doji"] = 1.0 if body / rng < 0.1 else 0.0
            f["is_hammer"] = 1.0 if f["lower_wick_ratio"] > 0.6 and f["body_range_ratio"] < 0.3 else 0.0
            f["is_shooting_star"] = 1.0 if f["upper_wick_ratio"] > 0.6 and f["body_range_ratio"] < 0.3 else 0.0
        else:
            for k in ["hl_position", "body_range_ratio", "upper_wick_ratio",
                       "lower_wick_ratio", "is_doji", "is_hammer", "is_shooting_star"]:
                f[k] = 0.0
    else:
        for k in ["hl_position", "body_range_ratio", "upper_wick_ratio",
                   "lower_wick_ratio", "is_doji", "is_hammer", "is_shooting_star"]:
            f[k] = 0.0

    if hist_len >= 5:
        br_s, br_c = 0.0, 0
        for j in range(hist_end - 5, hist_end):
            c = all_candles[j]
            rng = c.high - c.low
            if rng > 0:
                br_s += abs(c.close - c.open) / rng
                br_c += 1
        f["avg_body_range_5"] = br_s / br_c if br_c > 0 else 0.0
    else:
        f["avg_body_range_5"] = 0.0

    if hist_len >= 2:
        curr, prev = all_candles[hist_end - 1], all_candles[hist_end - 2]
        ct, cb = max(curr.open, curr.close), min(curr.open, curr.close)
        pt, pb = max(prev.open, prev.close), min(prev.open, prev.close)
        if curr.close > curr.open and prev.close < prev.open and ct > pt and cb < pb:
            f["engulfing"] = 1.0
        elif curr.close < curr.open and prev.close > prev.open and ct > pt and cb < pb:
            f["engulfing"] = -1.0
        else:
            f["engulfing"] = 0.0
        f["inside_bar"] = 1.0 if curr.high <= prev.high and curr.low >= prev.low else 0.0
    else:
        f["engulfing"] = 0.0
        f["inside_bar"] = 0.0

    # ===== Category 3: Volatility Structure (8) =====
    if hist_len >= 31:
        atr5 = sum(all_candles[j].high - all_candles[j].low for j in range(hist_end - 5, hist_end)) / 5.0
        atr30 = sum(all_candles[j].high - all_candles[j].low for j in range(hist_end - 30, hist_end)) / 30.0
        f["range_compression"] = atr5 / atr30 if atr30 > 0 else 1.0
    else:
        f["range_compression"] = 1.0

    if hist_len >= 15:
        pk = sum((math.log(all_candles[j].high / all_candles[j].low)) ** 2
                 for j in range(hist_end - 15, hist_end)
                 if all_candles[j].low > 0)
        f["parkinson_vol"] = math.sqrt(pk / (4 * 15 * math.log(2))) if pk > 0 else 0.0
    else:
        f["parkinson_vol"] = 0.0

    if hist_len >= 15:
        gk = 0.0
        for j in range(hist_end - 15, hist_end):
            c = all_candles[j]
            if c.low > 0 and c.open > 0:
                hl = math.log(c.high / c.low)
                co = math.log(c.close / c.open)
                gk += 0.5 * hl ** 2 - (2 * math.log(2) - 1) * co ** 2
        f["garman_klass_vol"] = math.sqrt(max(0, gk / 15))
    else:
        f["garman_klass_vol"] = 0.0

    if hist_len >= 15:
        cc_v, oc_v = 0.0, 0.0
        for j in range(hist_end - 14, hist_end):
            c, pc = all_candles[j], all_candles[j - 1]
            if pc.close > 0 and c.open > 0:
                cc_v += math.log(c.close / pc.close) ** 2
                oc_v += math.log(c.close / c.open) ** 2
        f["yang_zhang_ratio"] = cc_v / oc_v if oc_v > 0 else 1.0
    else:
        f["yang_zhang_ratio"] = 1.0

    if hist_len >= 30:
        atrs = []
        for k in range(6):
            s = hist_end - 30 + k * 5
            e = s + 5
            if e <= hist_end:
                atrs.append(sum(all_candles[j].high - all_candles[j].low for j in range(s, e)) / 5.0)
        if len(atrs) >= 3:
            ma = sum(atrs) / len(atrs)
            va = sum((a - ma) ** 2 for a in atrs) / len(atrs)
            f["vol_of_vol"] = math.sqrt(va) / ma if ma > 0 else 0.0
        else:
            f["vol_of_vol"] = 0.0
    else:
        f["vol_of_vol"] = 0.0

    if hist_len >= 30 and returns_30:
        sr = returns_30[-5:]
        vs = sum(r ** 2 for r in sr) / len(sr) if sr else 0.001
        vl = sum(r ** 2 for r in returns_30) / len(returns_30) if returns_30 else 0.001
        f["variance_ratio"] = vs / vl if vl > 0 else 1.0
    else:
        f["variance_ratio"] = 1.0

    if hist_len >= 20:
        bbc = [all_candles[j].close for j in range(hist_end - 20, hist_end)]
        bbm = sum(bbc) / 20.0
        bbs = math.sqrt(sum((c - bbm) ** 2 for c in bbc) / 20.0)
        f["bollinger_bandwidth"] = 4 * bbs / bbm if bbm > 0 else 0.0
    else:
        f["bollinger_bandwidth"] = 0.0

    if hist_len >= 5:
        f["range_contraction_5"] = float(sum(
            1 for j in range(hist_end - 4, hist_end)
            if (all_candles[j].high - all_candles[j].low) < (all_candles[j - 1].high - all_candles[j - 1].low)))
    else:
        f["range_contraction_5"] = 0.0

    # ===== Category 4: Entropy & Complexity (6) =====
    if len(returns_30) >= 10:
        r_arr = np.array(returns_30, dtype=np.float64)
        f["approx_entropy"] = _approx_entropy(r_arr)
        f["sample_entropy"] = _sample_entropy(r_arr)
        f["permutation_entropy"] = _permutation_entropy(r_arr)
        binary = "".join("1" if r > 0 else "0" for r in returns_30)
        f["lempel_ziv_complexity"] = _lempel_ziv_complexity(binary)
        ups = sum(1 for r in returns_30 if r > 0)
        downs = sum(1 for r in returns_30 if r < 0)
        flats = len(returns_30) - ups - downs
        total = len(returns_30)
        probs = [x / total for x in [ups, downs, flats] if x > 0]
        f["direction_entropy"] = -sum(p * math.log(p) for p in probs) / math.log(3) if probs else 0.0
        mr = sum(returns_30) / len(returns_30)
        vr = sum((r - mr) ** 2 for r in returns_30) / len(returns_30)
        f["return_kurtosis"] = (sum((r - mr) ** 4 for r in returns_30) / (len(returns_30) * vr ** 2) - 3.0) if vr > 0 else 0.0
    else:
        for k in ["approx_entropy", "sample_entropy", "permutation_entropy",
                   "lempel_ziv_complexity", "direction_entropy", "return_kurtosis"]:
            f[k] = 0.0

    # ===== Category 5: Autocorrelation (7) =====
    if len(returns_30) >= 10:
        r_arr = np.array(returns_30, dtype=np.float64)
        rm = np.mean(r_arr)
        rc = r_arr - rm
        var = float(np.sum(rc ** 2))
        if var > 0:
            for lag in [1, 2, 3, 5]:
                if lag < len(rc):
                    f[f"acf_{lag}"] = float(np.sum(rc[lag:] * rc[:-lag]) / var)
                else:
                    f[f"acf_{lag}"] = 0.0
            f["pacf_1"] = f["acf_1"]
            n = len(r_arr)
            lb = 0.0
            for lag in range(1, min(6, n)):
                acf_v = float(np.sum(rc[lag:] * rc[:-lag]) / var)
                lb += acf_v ** 2 / (n - lag)
            f["ljung_box_stat"] = n * (n + 2) * lb
        else:
            for k in ["acf_1", "acf_2", "acf_3", "acf_5", "pacf_1", "ljung_box_stat"]:
                f[k] = 0.0
        f["hurst_exponent"] = _hurst_exponent(r_arr)
    else:
        for k in ["acf_1", "acf_2", "acf_3", "acf_5", "pacf_1", "ljung_box_stat", "hurst_exponent"]:
            f[k] = 0.0

    # ===== Category 6: Volume Dynamics (7) =====
    if hist_len >= 15:
        v5 = sum(all_candles[j].volume for j in range(hist_end - 5, hist_end)) / 5.0
        v15 = sum(all_candles[j].volume for j in range(hist_end - 15, hist_end)) / 15.0
        f["volume_accel"] = v5 / v15 if v15 > 0 else 1.0
        avg10 = sum(all_candles[j].volume for j in range(hist_end - 10, hist_end)) / 10.0
        lv = all_candles[hist_end - 1].volume
        f["volume_spike"] = 1.0 if lv > 3 * avg10 else 0.0
        f["volume_spike_ratio"] = lv / avg10 if avg10 > 0 else 1.0
        vols = [all_candles[j].volume for j in range(hist_end - 15, hist_end)]
        prices = [all_candles[j].close for j in range(hist_end - 15, hist_end)]
        va, pa = np.array(vols), np.array(prices)
        if np.std(va) > 0 and np.std(pa) > 0:
            f["vol_price_corr"] = float(np.corrcoef(va, pa)[0, 1])
        else:
            f["vol_price_corr"] = 0.0
        vw_s, tv = 0.0, 0.0
        for j in range(hist_end - 15, hist_end):
            c = all_candles[j]
            pc = all_candles[j - 1] if j > 0 else c
            if pc.close > 0:
                vw_s += ((c.close - pc.close) / pc.close) * c.volume
                tv += c.volume
        f["vol_weighted_return"] = vw_s / tv if tv > 0 else 0.0
        fh = sum(all_candles[j].volume for j in range(hist_end - 15, hist_end - 7))
        sh = sum(all_candles[j].volume for j in range(hist_end - 7, hist_end))
        f["volume_trend"] = sh / fh if fh > 0 else 1.0
        mv = sum(vols) / len(vols)
        vv = sum((v - mv) ** 2 for v in vols) / len(vols)
        f["volume_cv"] = math.sqrt(vv) / mv if mv > 0 else 0.0
    else:
        for k in ["volume_accel", "volume_spike", "volume_spike_ratio", "vol_price_corr",
                   "vol_weighted_return", "volume_trend", "volume_cv"]:
            f[k] = 0.0

    # ===== Category 7: Calendar & Cyclical (8) =====
    ts = wd.timestamp
    f["hour_sin"] = math.sin(2 * math.pi * ts.hour / 24)
    f["hour_cos"] = math.cos(2 * math.pi * ts.hour / 24)
    f["minute_sin"] = math.sin(2 * math.pi * ts.minute / 60)
    f["minute_cos"] = math.cos(2 * math.pi * ts.minute / 60)
    f["near_funding"] = 1.0 if min(
        abs(ts.hour * 60 + ts.minute - h * 60) for h in [0, 8, 16, 24]) <= 30 else 0.0
    f["day_of_week"] = float(ts.weekday())
    f["is_weekend"] = 1.0 if ts.weekday() >= 5 else 0.0
    f["us_open_overlap"] = 1.0 if 13 <= ts.hour < 16 else 0.0

    # ===== Category 8: Cross-Scale Patterns (4) =====
    if len(returns_30) >= 10:
        r_arr = np.array(returns_30, dtype=np.float64)
        f["fractal_dim"] = _higuchi_fd(r_arr)
        f["dfa_exponent"] = _dfa_exponent(r_arr)
        fft_v = np.fft.rfft(r_arr)
        power = np.abs(fft_v) ** 2
        tp = float(np.sum(power))
        if tp > 0:
            mid = len(power) // 2
            f["high_freq_energy_ratio"] = float(np.sum(power[mid:])) / tp
        else:
            f["high_freq_energy_ratio"] = 0.5
        net = abs(sum(returns_30))
        tot = sum(abs(r) for r in returns_30)
        f["trend_strength_index"] = net / tot if tot > 0 else 0.0
    else:
        f["fractal_dim"] = 1.5
        f["dfa_exponent"] = 0.5
        f["high_freq_energy_ratio"] = 0.5
        f["trend_strength_index"] = 0.0

    # ===== Category 9: Prior Window Deep Analysis (6) =====
    if wi > 0:
        pw = windows[wi - 1]
        po, pc_ = pw[0].open, pw[-1].close
        ph = max(c.high for c in pw)
        pl = min(c.low for c in pw)
        pr = ph - pl
        f["prior_range_pct"] = pr / po * 100 if po > 0 else 0.0
        f["prior_close_position"] = (pc_ - pl) / pr if pr > 0 else 0.5
        fhv = sum(c.volume for c in pw[:7])
        shv = sum(c.volume for c in pw[7:])
        f["prior_vol_profile"] = shv / fhv if fhv > 0 else 1.0
        er = (pw[7].close - pw[0].open) / pw[0].open if pw[0].open > 0 else 0.0
        lr = (pw[-1].close - pw[7].close) / pw[7].close if pw[7].close > 0 else 0.0
        f["prior_early_vs_late"] = er - lr
        if po > 0:
            d = 1.0 if pc_ > po else -1.0
            mae = max((po - c.low) / po if d > 0 else (c.high - po) / po for c in pw)
            f["prior_reversal_depth"] = mae
        else:
            f["prior_reversal_depth"] = 0.0
        if wi >= 5:
            prev_r = [max(c.high for c in windows[k]) - min(c.low for c in windows[k])
                      for k in range(wi - 5, wi - 1)]
            med_r = sorted(prev_r)[len(prev_r) // 2]
            f["prior_was_volatile"] = 1.0 if pr > 2 * med_r else 0.0
        else:
            f["prior_was_volatile"] = 0.0
    else:
        for k in ["prior_range_pct", "prior_close_position", "prior_vol_profile",
                   "prior_early_vs_late", "prior_reversal_depth", "prior_was_volatile"]:
            f[k] = 0.0

    # Sanitize NaN/inf
    for k, v in f.items():
        if not math.isfinite(v):
            f[k] = 0.0

    return f


# ---------------------------------------------------------------------------
# Combined feature computation (95 features)
# ---------------------------------------------------------------------------

def compute_combined_features(
    wd: WindowData,
    windows: list[list[FastCandle]],
    all_candles: list[FastCandle],
) -> dict[str, float]:
    orig = compute_all_features(wd, windows, all_candles, entry_minute=0)
    orig.update(compute_novel_features(wd, windows, all_candles))
    return orig


def features_to_array(feat_dict: dict[str, float]) -> np.ndarray:
    """Convert feature dict to numpy array in canonical order."""
    return np.array([feat_dict.get(n, 0.0) for n in ALL_FEATURE_NAMES], dtype=np.float64)


# ---------------------------------------------------------------------------
# Feature Quality Analysis
# ---------------------------------------------------------------------------

def analyze_feature_quality(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> dict:
    """Phase 2: MI ranking, correlation, per-feature accuracy."""
    print("  Computing Mutual Information...", end=" ")
    _flush()
    mi = mutual_info_classif(X, y, discrete_features=False, random_state=42, n_neighbors=5)
    mi_ranking = sorted(zip(feature_names, mi.tolist()), key=lambda x: -x[1])
    print("done")

    print("  Computing per-feature directional accuracy...", end=" ")
    _flush()
    feat_acc = {}
    for i, name in enumerate(feature_names):
        med = np.median(X[:, i])
        above = y[X[:, i] > med]
        if len(above) > 100:
            up_rate = float(above.mean())
            acc = max(up_rate, 1 - up_rate)
            feat_acc[name] = {"accuracy": round(acc, 4),
                              "direction": "Up" if up_rate > 0.5 else "Down",
                              "contrarian": up_rate < 0.5}
        else:
            feat_acc[name] = {"accuracy": 0.5, "direction": "Up", "contrarian": False}
    print("done")

    print("  Computing pairwise correlations...", end=" ")
    _flush()
    corr_matrix = np.corrcoef(X.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    high_corr_pairs = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            r = abs(corr_matrix[i, j])
            if r > 0.7:
                high_corr_pairs.append((feature_names[i], feature_names[j], round(float(r), 3)))
    high_corr_pairs.sort(key=lambda x: -x[2])
    print(f"{len(high_corr_pairs)} highly correlated pairs")

    return {
        "mi_ranking": [(n, round(float(v), 6)) for n, v in mi_ranking[:30]],
        "top_mi_features": [n for n, v in mi_ranking[:20] if v > 0],
        "per_feature_accuracy": feat_acc,
        "high_correlation_pairs": high_corr_pairs[:20],
        "any_novel_feature_has_mi": any(v > 0.001 for n, v in mi_ranking if n in NOVEL_FEATURE_NAMES),
    }


# ---------------------------------------------------------------------------
# Approach A: Conditional Regime Mining (Bonferroni-corrected)
# ---------------------------------------------------------------------------

class RegimeMiner:
    """Find Bonferroni-significant 2-way feature condition pairs."""

    def __init__(self):
        self.best_regimes: list[dict] = []

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: list[str]):
        n_feat = X_train.shape[1]
        percentiles_vals = [10, 25, 50, 75, 90]
        thresholds = np.percentile(X_train, percentiles_vals, axis=0)  # (5, n_feat)

        # Build condition masks: each condition = (feature, percentile_idx, direction)
        # direction 0 = ">", direction 1 = "<"
        conditions = []
        cond_meta = []
        for fi in range(n_feat):
            for pi, pv in enumerate(percentiles_vals):
                mask_gt = X_train[:, fi] > thresholds[pi, fi]
                mask_lt = X_train[:, fi] < thresholds[pi, fi]
                conditions.append(mask_gt)
                cond_meta.append((feature_names[fi], f">{pv}th"))
                conditions.append(mask_lt)
                cond_meta.append((feature_names[fi], f"<{pv}th"))

        n_cond = len(conditions)
        # Vectorized: convert to float32 matrix
        C = np.array(conditions, dtype=np.float32)  # (n_cond, N_train)
        y_f = y_train.astype(np.float32)

        # Pairwise counts via matrix multiply
        N_mat = C @ C.T  # (n_cond, n_cond) — pairwise AND counts
        UP_mat = (C * y_f[np.newaxis, :]) @ C.T  # pairwise AND with y=1

        # Search for significant pairs
        n_tests = n_cond * (n_cond - 1) // 2
        bonf_threshold = BONFERRONI_ALPHA / max(n_tests, 1)
        significant = []

        # Use upper triangle
        for i in range(n_cond):
            for j in range(i + 1, n_cond):
                n = N_mat[i, j]
                if n < MIN_REGIME_WINDOWS:
                    continue
                up = UP_mat[i, j]
                up_rate = up / n
                acc = max(up_rate, 1 - up_rate)
                if acc < 0.53:
                    continue
                # Normal approximation for binomial p-value
                z = (float(acc) - 0.5) / math.sqrt(0.25 / float(n))
                p_val = 1 - norm.cdf(z)
                if p_val < bonf_threshold:
                    direction = "Up" if up_rate > 0.5 else "Down"
                    significant.append({
                        "cond1": cond_meta[i],
                        "cond2": cond_meta[j],
                        "accuracy": round(float(acc), 4),
                        "n_windows": int(n),
                        "direction": direction,
                        "p_value": float(p_val),
                        "z_score": round(z, 2),
                    })

        significant.sort(key=lambda x: -x["accuracy"])
        self.best_regimes = significant[:20]
        return self

    def predict(self, X_test: np.ndarray, X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply best regime to test data. Returns (predictions, trade_mask)."""
        N = X_test.shape[0]
        preds = np.zeros(N, dtype=int)
        mask = np.zeros(N, dtype=bool)

        if not self.best_regimes:
            return preds, mask

        best = self.best_regimes[0]
        c1_feat, c1_op = best["cond1"]
        c2_feat, c2_op = best["cond2"]
        direction = 1 if best["direction"] == "Up" else 0

        fi1 = ALL_FEATURE_NAMES.index(c1_feat)
        fi2 = ALL_FEATURE_NAMES.index(c2_feat)

        # Compute thresholds from training data
        p1 = int(c1_op[1:-2])
        p2 = int(c2_op[1:-2])
        t1 = np.percentile(X_train[:, fi1], p1)
        t2 = np.percentile(X_train[:, fi2], p2)

        m1 = X_test[:, fi1] > t1 if c1_op[0] == ">" else X_test[:, fi1] < t1
        m2 = X_test[:, fi2] > t2 if c2_op[0] == ">" else X_test[:, fi2] < t2

        combined = m1 & m2
        mask = combined
        preds[mask] = direction

        return preds, mask


# ---------------------------------------------------------------------------
# Approach B: Contrarian Signal Stacking
# ---------------------------------------------------------------------------

class ContrarianStacker:
    """Select independent signals and stack for higher accuracy."""

    def __init__(self):
        self.selected_signals: list[dict] = []
        self.best_n: int = 3
        self.best_acc: float = 0.0
        self.best_skip: float = 1.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: list[str]):
        n_feat = X_train.shape[1]

        # Step 1: Per-feature directional accuracy
        signals = []
        for i in range(n_feat):
            med = np.median(X_train[:, i])
            above = X_train[:, i] > med
            if above.sum() < 100 or (~above).sum() < 100:
                continue
            up_rate = y_train[above].mean()
            if up_rate > 0.5:
                acc = up_rate
                direction = "above_up"
            else:
                acc = 1 - up_rate
                direction = "above_down"
            if acc > 0.51:
                signals.append({
                    "feature_idx": i,
                    "feature_name": feature_names[i],
                    "threshold": float(med),
                    "accuracy": float(acc),
                    "direction": direction,
                })

        signals.sort(key=lambda x: -x["accuracy"])

        # Step 2: Greedy independence selection (|corr| < 0.3)
        selected = []
        for sig in signals:
            independent = True
            for sel in selected:
                corr = abs(float(np.corrcoef(X_train[:, sig["feature_idx"]],
                                              X_train[:, sel["feature_idx"]])[0, 1]))
                if corr > 0.3:
                    independent = False
                    break
            if independent:
                selected.append(sig)
            if len(selected) >= 15:
                break

        self.selected_signals = selected

        # Step 3: Sweep N (number of agreeing signals)
        best_n, best_acc, best_skip = 3, 0.0, 1.0
        for n_agree in range(3, min(len(selected) + 1, 12)):
            # For each window, compute signal votes
            votes_up = np.zeros(len(y_train))
            votes_down = np.zeros(len(y_train))
            for sig in selected[:n_agree]:
                fi = sig["feature_idx"]
                above = X_train[:, fi] > sig["threshold"]
                if sig["direction"] == "above_up":
                    votes_up += above.astype(float)
                    votes_down += (~above).astype(float)
                else:
                    votes_down += above.astype(float)
                    votes_up += (~above).astype(float)

            # All agree on Up or all agree on Down
            all_up = votes_up == n_agree
            all_down = votes_down == n_agree
            traded = all_up | all_down

            if traded.sum() < 100:
                continue

            correct = (all_up & (y_train == 1)) | (all_down & (y_train == 0))
            acc = float(correct.sum()) / float(traded.sum())
            skip = 1 - float(traded.sum()) / len(y_train)

            if acc > best_acc and skip < 0.95:
                best_n = n_agree
                best_acc = acc
                best_skip = skip

        self.best_n = best_n
        self.best_acc = best_acc
        self.best_skip = best_skip
        return self

    def predict(self, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        N = X_test.shape[0]
        preds = np.zeros(N, dtype=int)
        mask = np.zeros(N, dtype=bool)

        if len(self.selected_signals) < self.best_n:
            return preds, mask

        votes_up = np.zeros(N)
        votes_down = np.zeros(N)
        for sig in self.selected_signals[:self.best_n]:
            fi = sig["feature_idx"]
            above = X_test[:, fi] > sig["threshold"]
            if sig["direction"] == "above_up":
                votes_up += above.astype(float)
                votes_down += (~above).astype(float)
            else:
                votes_down += above.astype(float)
                votes_up += (~above).astype(float)

        all_up = votes_up == self.best_n
        all_down = votes_down == self.best_n
        mask = all_up | all_down
        preds[all_up] = 1
        preds[all_down] = 0

        return preds, mask


# ---------------------------------------------------------------------------
# Approach C: Two-Stage Meta-Model
# ---------------------------------------------------------------------------

class MetaModel:
    """Stage 1: predictability filter. Stage 2: direction model."""

    def __init__(self):
        self.stage1 = None
        self.stage2 = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: list[str]):
        # Stage 2: direction model
        self.stage2 = HistGradientBoostingClassifier(
            max_depth=3, min_samples_leaf=100, l2_regularization=1.0,
            max_iter=200, learning_rate=0.05, random_state=42)
        self.stage2.fit(X_train, y_train)

        # Generate meta-labels via 5-fold cross-val
        kf = KFold(n_splits=5, shuffle=False)
        meta_labels = np.zeros(len(y_train), dtype=int)
        for tr_idx, val_idx in kf.split(X_train):
            fold_model = HistGradientBoostingClassifier(
                max_depth=3, min_samples_leaf=100, l2_regularization=1.0,
                max_iter=200, learning_rate=0.05, random_state=42)
            fold_model.fit(X_train[tr_idx], y_train[tr_idx])
            fold_preds = fold_model.predict(X_train[val_idx])
            meta_labels[val_idx] = (fold_preds == y_train[val_idx]).astype(int)

        # Stage 1: predictability model
        self.stage1 = HistGradientBoostingClassifier(
            max_depth=3, min_samples_leaf=150, l2_regularization=2.0,
            max_iter=200, learning_rate=0.05, random_state=42)
        self.stage1.fit(X_train, meta_labels)
        return self

    def predict(self, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        predictable = self.stage1.predict(X_test)
        directions = self.stage2.predict(X_test)
        mask = predictable == 1
        preds = directions.copy()
        return preds, mask


# ---------------------------------------------------------------------------
# Approach D: Full ML on 95 Features
# ---------------------------------------------------------------------------

class FullML:
    """HistGradientBoosting on all 95 features with threshold sweep."""

    def __init__(self):
        self.model = None
        self.threshold = 0.5

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: list[str]):
        self.model = HistGradientBoostingClassifier(
            max_depth=3, min_samples_leaf=100, l2_regularization=1.0,
            max_iter=200, learning_rate=0.05, random_state=42)
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        proba = self.model.predict_proba(X_test)[:, 1]
        preds = (proba > 0.5).astype(int)
        mask = np.ones(len(X_test), dtype=bool)  # trade all
        return preds, mask

    def predict_with_threshold(self, X_test: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
        proba = self.model.predict_proba(X_test)[:, 1]
        preds = (proba > 0.5).astype(int)
        mask = (proba > threshold) | (proba < (1 - threshold))
        return preds, mask


# ---------------------------------------------------------------------------
# Walk-Forward Validation
# ---------------------------------------------------------------------------

def run_walk_forward(
    approach_name: str,
    approach_class: type,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    months: list[str],
    by_month_idx: dict[str, list[int]],
) -> dict:
    """Run walk-forward with 3-month train / 1-month test."""
    all_preds, all_actuals, all_traded_idx = [], [], []
    fold_results = []

    for mi in range(WF_TRAIN_MONTHS, len(months)):
        train_idx = []
        for m_off in range(WF_TRAIN_MONTHS):
            month = months[mi - WF_TRAIN_MONTHS + m_off]
            train_idx.extend(by_month_idx[month])
        test_idx = by_month_idx[months[mi]]

        if not train_idx or not test_idx:
            continue

        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]

        approach = approach_class()
        approach.fit(X_train, y_train, feature_names)

        if approach_name == "A_regime_mining":
            preds, mask = approach.predict(X_test, X_train)
        else:
            preds, mask = approach.predict(X_test)

        n_traded = int(mask.sum())
        if n_traded > 0:
            correct = int(((preds[mask] == y_test[mask]).sum()))
            acc = correct / n_traded
        else:
            correct, acc = 0, 0.0
        skip_rate = 1 - n_traded / len(y_test) if len(y_test) > 0 else 1.0

        fold_results.append({
            "month": months[mi],
            "n_test": len(y_test),
            "n_traded": n_traded,
            "correct": correct,
            "accuracy": round(acc, 4),
            "skip_rate": round(skip_rate, 4),
        })

        for i in range(len(y_test)):
            if mask[i]:
                all_preds.append(int(preds[i]))
                all_actuals.append(int(y_test[i]))
                all_traded_idx.append(test_idx[i])

    total_traded = len(all_preds)
    total_correct = sum(1 for p, a in zip(all_preds, all_actuals) if p == a)
    total_test = sum(fr["n_test"] for fr in fold_results)
    overall_acc = total_correct / total_traded if total_traded > 0 else 0.0
    overall_skip = 1 - total_traded / total_test if total_test > 0 else 1.0

    # PnL simulation
    pnl = 0.0
    for p, a in zip(all_preds, all_actuals):
        direction = "Up" if p == 1 else "Down"
        resolution = "Up" if a == 1 else "Down"
        _, _, pnl_net, _ = simulate_trade(direction, resolution)
        pnl += pnl_net

    # Profitable folds
    profitable_folds = sum(1 for fr in fold_results if fr["n_traded"] > 0 and fr["accuracy"] > 0.52)
    total_active_folds = sum(1 for fr in fold_results if fr["n_traded"] > 0)

    return {
        "total_traded": total_traded,
        "total_correct": total_correct,
        "accuracy": round(overall_acc, 4),
        "skip_rate": round(overall_skip, 4),
        "net_pnl": round(pnl, 2),
        "profitable_folds_pct": round(profitable_folds / total_active_folds * 100, 1) if total_active_folds > 0 else 0.0,
        "fold_results": fold_results,
        "predictions": all_preds,
        "actuals": all_actuals,
    }


# ---------------------------------------------------------------------------
# Statistical Validation
# ---------------------------------------------------------------------------

def validate_results(results: dict) -> dict:
    """Bootstrap CI, permutation test on walk-forward predictions."""
    preds = results["predictions"]
    actuals = results["actuals"]

    if len(preds) < 50:
        return {"bootstrap_ci_95": [0.0, 0.0], "permutation_p_value": 1.0,
                "statistically_significant": False}

    correct_list = [p == a for p, a in zip(preds, actuals)]
    _, ci_lo, ci_hi = bootstrap_ci(correct_list, n=5000)

    pred_strs = ["Up" if p == 1 else "Down" for p in preds]
    act_strs = ["Up" if a == 1 else "Down" for a in actuals]
    p_val = permutation_test(pred_strs, act_strs, n=5000)

    return {
        "bootstrap_ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
        "permutation_p_value": round(p_val, 4),
        "statistically_significant": p_val < 0.05,
    }


# ---------------------------------------------------------------------------
# Threshold Sweep for Full ML
# ---------------------------------------------------------------------------

def run_ml_threshold_sweep(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    months: list[str],
    by_month_idx: dict[str, list[int]],
) -> list[dict]:
    """Run walk-forward for Full ML at multiple thresholds."""
    # Collect all fold probabilities
    all_probas, all_actuals = [], []

    for mi in range(WF_TRAIN_MONTHS, len(months)):
        train_idx = []
        for m_off in range(WF_TRAIN_MONTHS):
            train_idx.extend(by_month_idx[months[mi - WF_TRAIN_MONTHS + m_off]])
        test_idx = by_month_idx[months[mi]]
        if not train_idx or not test_idx:
            continue

        model = HistGradientBoostingClassifier(
            max_depth=3, min_samples_leaf=100, l2_regularization=1.0,
            max_iter=200, learning_rate=0.05, random_state=42)
        model.fit(X[train_idx], y[train_idx])
        probas = model.predict_proba(X[test_idx])[:, 1]
        all_probas.extend(probas.tolist())
        all_actuals.extend(y[test_idx].tolist())

    probas = np.array(all_probas)
    actuals = np.array(all_actuals)
    sweep = []
    for thr in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        mask = (probas > thr) | (probas < (1 - thr))
        n_traded = int(mask.sum())
        if n_traded < 10:
            continue
        preds = (probas[mask] > 0.5).astype(int)
        correct = int((preds == actuals[mask]).sum())
        acc = correct / n_traded
        skip = 1 - n_traded / len(actuals)
        pnl = 0.0
        for p, a in zip(preds, actuals[mask]):
            d = "Up" if p == 1 else "Down"
            r = "Up" if a == 1 else "Down"
            _, _, pn, _ = simulate_trade(d, r)
            pnl += pn
        sweep.append({
            "threshold": thr,
            "traded": n_traded,
            "accuracy": round(acc, 4),
            "skip_rate": round(skip, 4),
            "net_pnl": round(pnl, 2),
        })
    return sweep


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 120)
    print("  BWO STRATEGY v2 — NOVEL REGIME MINING BACKTEST")
    print("  64 novel features + 31 original = 95 total")
    print("  4 approaches | Walk-forward validation | Bonferroni correction")
    print("=" * 120)
    _flush()

    if not BTC_CSV.exists():
        print(f"  ERROR: {BTC_CSV} not found.")
        sys.exit(1)

    # --- Load data ---
    print("\n  Loading BTC 1m candle data...", end=" ")
    _flush()
    all_candles = load_csv_fast(BTC_CSV)
    all_candles.sort(key=lambda c: c.timestamp)
    seen: set[datetime] = set()
    unique: list[FastCandle] = []
    for c in all_candles:
        if c.timestamp not in seen:
            seen.add(c.timestamp)
            unique.append(c)
    all_candles = unique
    print(f"{len(all_candles):,} unique candles")
    _flush()

    print("  Grouping into 15m windows...", end=" ")
    _flush()
    windows = group_into_15m_windows(all_candles)
    print(f"{len(windows):,} complete windows")
    _flush()

    # Build index map
    candle_by_ts: dict[datetime, int] = {c.timestamp: i for i, c in enumerate(all_candles)}

    print("  Building window metadata...", end=" ")
    _flush()
    window_data: list[WindowData] = []
    for wi, w in enumerate(windows):
        ts = w[0].timestamp
        idx = candle_by_ts.get(ts)
        if idx is None:
            continue
        resolution = "Up" if w[-1].close > w[0].open else "Down"
        window_data.append(WindowData(window_idx=wi, candle_idx=idx,
                                       resolution=resolution, timestamp=ts))
    print(f"{len(window_data):,} windows")
    _flush()

    up_count = sum(1 for wd in window_data if wd.resolution == "Up")
    print(f"\n  Base rate: Up {up_count:,} ({up_count/len(window_data)*100:.1f}%) | "
          f"Down {len(window_data)-up_count:,} ({(len(window_data)-up_count)/len(window_data)*100:.1f}%)")

    # --- Compute ALL 95 features ---
    print(f"\n  Computing 95 features for {len(window_data):,} windows...")
    _flush()
    all_feat_dicts: list[dict[str, float]] = []
    for i, wd in enumerate(window_data):
        all_feat_dicts.append(compute_combined_features(wd, windows, all_candles))
        if (i + 1) % 5000 == 0:
            print(f"    {i+1:,}/{len(window_data):,}...")
            _flush()
    print(f"    done — {len(all_feat_dicts):,} feature vectors")
    _flush()

    # Convert to numpy
    X = np.array([features_to_array(fd) for fd in all_feat_dicts], dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array([1 if wd.resolution == "Up" else 0 for wd in window_data], dtype=int)

    # --- Month index for walk-forward ---
    by_month_idx: dict[str, list[int]] = defaultdict(list)
    for idx, wd in enumerate(window_data):
        by_month_idx[wd.timestamp.strftime("%Y-%m")].append(idx)
    months = sorted(by_month_idx.keys())
    print(f"  {len(months)} months: {months[0]} → {months[-1]}")

    # --- Phase 2: Feature Quality Analysis ---
    print(f"\n{'='*120}")
    print("  PHASE 2: FEATURE QUALITY ANALYSIS")
    print(f"{'='*120}")
    _flush()
    # Use first 80% for quality analysis
    split_idx = int(len(window_data) * 0.80)
    quality = analyze_feature_quality(X[:split_idx], y[:split_idx], ALL_FEATURE_NAMES)

    print(f"\n  Top 15 features by MI:")
    for name, mi_val in quality["mi_ranking"][:15]:
        cat = "NOVEL" if name in NOVEL_FEATURE_NAMES else "ORIG"
        acc_info = quality["per_feature_accuracy"].get(name, {})
        print(f"    {name:<35} MI={mi_val:.6f}  acc={acc_info.get('accuracy',0):.4f} "
              f"dir={acc_info.get('direction','?'):<5} [{cat}]")

    print(f"\n  Any novel feature has MI > 0.001: {quality['any_novel_feature_has_mi']}")
    if quality["high_correlation_pairs"]:
        print(f"  Top correlated pairs: {len(quality['high_correlation_pairs'])}")
        for f1, f2, r in quality["high_correlation_pairs"][:5]:
            print(f"    {f1} <-> {f2}: r={r}")
    _flush()

    # --- Phase 3+4: Run All Approaches with Walk-Forward ---
    report: dict[str, Any] = {
        "meta": {
            "total_candles": len(all_candles),
            "total_windows": len(window_data),
            "base_rate_up_pct": round(up_count / len(window_data) * 100, 2),
            "n_features": len(ALL_FEATURE_NAMES),
            "n_original": len(ORIGINAL_FEATURE_NAMES),
            "n_novel": len(NOVEL_FEATURE_NAMES),
        },
        "feature_quality": quality,
    }

    approaches = [
        ("A_regime_mining", RegimeMiner),
        ("B_contrarian_stacking", ContrarianStacker),
        ("C_meta_model", MetaModel),
        ("D_full_ml", FullML),
    ]

    for approach_name, approach_class in approaches:
        print(f"\n{'='*120}")
        print(f"  APPROACH {approach_name}")
        print(f"{'='*120}")
        _flush()

        results = run_walk_forward(
            approach_name, approach_class, X, y, ALL_FEATURE_NAMES,
            months, by_month_idx)

        print(f"  Walk-forward results:")
        print(f"    Traded: {results['total_traded']:,} / "
              f"{sum(fr['n_test'] for fr in results['fold_results']):,} "
              f"(skip {results['skip_rate']*100:.1f}%)")
        print(f"    Accuracy: {results['accuracy']*100:.1f}% "
              f"({results['total_correct']:,}/{results['total_traded']:,})")
        print(f"    Net PnL: ${results['net_pnl']:+,.0f}")
        print(f"    Profitable folds: {results['profitable_folds_pct']:.0f}%")
        _flush()

        # Statistical validation
        validation = validate_results(results)
        print(f"    Bootstrap 95% CI: [{validation['bootstrap_ci_95'][0]*100:.1f}%, "
              f"{validation['bootstrap_ci_95'][1]*100:.1f}%]")
        print(f"    Permutation p-value: {validation['permutation_p_value']:.4f} "
              f"{'(significant)' if validation['statistically_significant'] else '(not significant)'}")
        _flush()

        # Approach-specific details
        if approach_name == "A_regime_mining":
            # Run once on full training set to report regimes found
            miner = RegimeMiner()
            miner.fit(X[:split_idx], y[:split_idx], ALL_FEATURE_NAMES)
            print(f"    Bonferroni-significant regimes found: {len(miner.best_regimes)}")
            for i, reg in enumerate(miner.best_regimes[:5]):
                print(f"      #{i+1}: {reg['cond1'][0]} {reg['cond1'][1]} AND "
                      f"{reg['cond2'][0]} {reg['cond2'][1]} → {reg['direction']} "
                      f"acc={reg['accuracy']*100:.1f}% n={reg['n_windows']:,} "
                      f"z={reg['z_score']:.1f} p={reg['p_value']:.2e}")
            results["regimes_found"] = len(miner.best_regimes)
            results["top_regimes"] = miner.best_regimes[:10]

        elif approach_name == "B_contrarian_stacking":
            stacker = ContrarianStacker()
            stacker.fit(X[:split_idx], y[:split_idx], ALL_FEATURE_NAMES)
            print(f"    Selected signals: {len(stacker.selected_signals)}")
            for sig in stacker.selected_signals[:5]:
                print(f"      {sig['feature_name']}: acc={sig['accuracy']*100:.1f}% "
                      f"dir={sig['direction']}")
            print(f"    Best N: {stacker.best_n} (train acc={stacker.best_acc*100:.1f}%, "
                  f"skip={stacker.best_skip*100:.1f}%)")
            results["n_signals"] = len(stacker.selected_signals)
            results["best_n"] = stacker.best_n

        # Store in report
        results_clean = {k: v for k, v in results.items()
                         if k not in ("predictions", "actuals")}
        results_clean["validation"] = validation
        results_clean["meets_target"] = (
            results["accuracy"] >= 0.80
            and results["skip_rate"] < 0.30
            and validation["statistically_significant"]
            and results["total_traded"] >= 500
            and results["net_pnl"] > 0
        )
        report[approach_name] = results_clean
        _flush()

    # --- ML Threshold Sweep ---
    print(f"\n{'='*120}")
    print("  ML THRESHOLD SWEEP (accuracy vs skip rate)")
    print(f"{'='*120}")
    _flush()
    sweep = run_ml_threshold_sweep(X, y, ALL_FEATURE_NAMES, months, by_month_idx)
    print(f"  {'Threshold':>10} {'Traded':>8} {'Accuracy':>10} {'Skip':>8} {'PnL':>10}")
    print(f"  {'-'*50}")
    for s in sweep:
        print(f"  {s['threshold']:>10.2f} {s['traded']:>8,} {s['accuracy']*100:>9.1f}% "
              f"{s['skip_rate']*100:>7.1f}% ${s['net_pnl']:>+8,.0f}")
    report["ml_threshold_sweep"] = sweep

    # --- Success Criteria ---
    print(f"\n{'='*120}")
    print("  SUCCESS CRITERIA CHECK")
    print(f"{'='*120}")

    any_met = False
    best_approach = None
    best_acc = 0.0
    for name in ["A_regime_mining", "B_contrarian_stacking", "C_meta_model", "D_full_ml"]:
        data = report[name]
        meets = data.get("meets_target", False)
        print(f"  {name}: acc={data['accuracy']*100:.1f}% skip={data['skip_rate']*100:.1f}% "
              f"pnl=${data['net_pnl']:+,.0f} sig={data['validation']['statistically_significant']} "
              f"→ {'PASS' if meets else 'FAIL'}")
        if meets:
            any_met = True
            if data["accuracy"] > best_acc:
                best_acc = data["accuracy"]
                best_approach = name

    report["success_criteria"] = {
        "any_approach_met_target": any_met,
        "best_approach": best_approach,
        "target_accuracy": 0.80,
        "target_skip_rate": 0.30,
        "target_min_trades": 500,
    }

    if any_met:
        print(f"\n  TARGET MET: {best_approach} with {best_acc*100:.1f}% accuracy")
    else:
        print(f"\n  NO APPROACH meets 80% accuracy + <30% skip + p<0.05 + positive PnL")
        print(f"  Conclusion: Pre-window BTC 1m OHLCV has no predictive edge for 15m direction.")
        print(f"  Alternative data sources needed: orderbook depth, cross-asset, funding rates, sentiment.")

    # --- Save report ---
    with open(REPORT_JSON, "w") as fout:
        json.dump(report, fout, indent=2, default=str)
    print(f"\n  Report saved: {REPORT_JSON}")
    print(f"{'='*120}")
    _flush()


if __name__ == "__main__":
    main()
