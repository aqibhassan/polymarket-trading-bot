"""BWO ML Backtest — HistGradientBoosting with early-window features.

Trains a calibrated HistGradientBoostingClassifier per entry minute (2-5) on
pre-window + early-window + TA features. Sweeps confidence thresholds to find
the operating point where accuracy >= 90% with skip rate < 50%.

Target: does early momentum continue through settlement?
  target = 1 if early_direction aligns with final resolution, else 0
  (skip windows where early_direction == 0)

Walk-forward validation: 3-month train / 1-month test rolling.
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest_before_window import (
    BOOTSTRAP_ITERATIONS,
    FEE_CONSTANT,
    PERMUTATION_ITERATIONS,
    POSITION_SIZE,
    SLIPPAGE_BPS,
    StrategyMetrics,
    TradeResult,
    WindowData,
    bootstrap_ci,
    compute_all_features,
    compute_metrics,
    permutation_test,
    polymarket_fee,
    simulate_trade,
)
from scripts.fast_loader import FastCandle, group_into_15m_windows, load_csv_fast

# Late imports — sklearn / joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BTC_CSV = PROJECT_ROOT / "data" / "btc_1m_2y.csv"
REPORT_JSON = PROJECT_ROOT / "data" / "bwo_ml_report.json"
MODEL_DIR = PROJECT_ROOT / "models"

ENTRY_MINUTES = [2, 3, 4, 5]
THRESHOLD_RANGE = [round(0.50 + i * 0.01, 2) for i in range(46)]  # 0.50..0.95

WF_TRAIN_MONTHS = 3
WF_TEST_MONTHS = 1

# Success criteria
MIN_ACCURACY = 0.90
MAX_SKIP_RATE = 0.50
MIN_PERMUTATION_SIG = 0.05
MAX_IS_OOS_GAP = 0.05
MIN_WF_PROFITABLE_PCT = 0.75
MIN_TEST_TRADES = 500


def _flush() -> None:
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Feature matrix construction
# ---------------------------------------------------------------------------

# Canonical feature order (matches compute_all_features output)
FEATURE_NAMES: list[str] = [
    # Pre-window (7 groups → 19 raw features)
    "prior_dir", "prior_mag",
    "mtf_score", "mtf_15m", "mtf_1h", "mtf_4h",
    "stm_dir", "stm_strength",
    "vol_regime", "vol_percentile",
    "tod_hour", "tod_asia", "tod_europe", "tod_us", "tod_late",
    "streak_len", "streak_dir",
    "vol_ratio", "vol_dir_align",
    # Early-window (6)
    "early_cum_return", "early_direction", "early_magnitude",
    "early_green_ratio", "early_vol", "early_max_move",
    # TA (6)
    "rsi_14", "macd_histogram_sign", "bb_pct_b",
    "atr_14", "mean_reversion_z", "price_vs_vwap",
]

NUM_FEATURES = len(FEATURE_NAMES)


def build_feature_matrix(
    window_data: list[WindowData],
    all_features_list: list[dict[str, float]],
    entry_minute: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Build X matrix and y target from features. Skip neutral early_direction.

    Target: 1 if early momentum continues (early_direction aligns with resolution),
            0 otherwise.

    Returns (X, y, valid_indices) where valid_indices maps back to window_data.
    """
    n = len(window_data)
    X_rows: list[list[float]] = []
    y_vals: list[int] = []
    valid_indices: list[int] = []

    for i in range(n):
        feats = all_features_list[i]
        early_dir = feats.get("early_direction", 0.0)

        # Skip windows where early_direction is neutral
        if early_dir == 0.0:
            continue

        wd = window_data[i]
        resolution = wd.resolution

        # Target: does early momentum continue?
        if (resolution == "Up" and early_dir > 0) or (resolution == "Down" and early_dir < 0):
            target = 1
        else:
            target = 0

        row = [feats.get(fname, 0.0) for fname in FEATURE_NAMES]
        X_rows.append(row)
        y_vals.append(target)
        valid_indices.append(i)

    X = np.array(X_rows, dtype=np.float64)
    y = np.array(y_vals, dtype=np.int32)
    return X, y, valid_indices


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> CalibratedClassifierCV:
    """Train calibrated HistGradientBoosting."""
    base = HistGradientBoostingClassifier(
        max_iter=500,
        max_depth=5,
        learning_rate=0.05,
        min_samples_leaf=100,
        l2_regularization=1.0,
        max_bins=128,
        random_state=42,
    )
    calibrated = CalibratedClassifierCV(base, cv=5, method="isotonic")
    calibrated.fit(X_train, y_train)
    return calibrated


# ---------------------------------------------------------------------------
# Entry price adjustment
# ---------------------------------------------------------------------------

def adjusted_entry_price(early_cum_return: float) -> float:
    """At minute N, entry price = 0.50 + abs(early_cum_return) * 0.5, capped at 0.70."""
    price = 0.50 + abs(early_cum_return) * 0.5
    return min(price, 0.70)


def adjusted_pnl(
    direction: str,
    resolution: str,
    entry_price: float,
) -> tuple[float, float, float, float]:
    """Simulate trade at adjusted entry price. Returns (settlement, pnl_gross, pnl_net, fee)."""
    correct = direction == resolution
    settlement = 1.0 if correct else 0.0
    pnl_gross = ((settlement - entry_price) / entry_price) * POSITION_SIZE
    fee = polymarket_fee(POSITION_SIZE, entry_price)
    slip = POSITION_SIZE * SLIPPAGE_BPS / 10000
    total_fee = fee + slip
    pnl_net = pnl_gross - total_fee
    return settlement, pnl_gross, pnl_net, total_fee


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------

@dataclass
class ThresholdResult:
    threshold: float
    accuracy: float
    skip_rate: float
    trades: int
    net_pnl: float
    gross_pnl: float
    total_fees: float
    correct: int
    total_windows: int


def threshold_sweep(
    model: CalibratedClassifierCV,
    X_test: np.ndarray,
    y_test: np.ndarray,
    window_data_test: list[WindowData],
    all_features_test: list[dict[str, float]],
    valid_indices_test: list[int],
    entry_minute: int,
    total_test_windows: int,
) -> list[ThresholdResult]:
    """Sweep thresholds and return calibration table."""
    probas = model.predict_proba(X_test)[:, 1]  # P(momentum continues)
    results: list[ThresholdResult] = []

    for threshold in THRESHOLD_RANGE:
        mask = probas >= threshold
        n_traded = int(mask.sum())

        if n_traded == 0:
            results.append(ThresholdResult(
                threshold=threshold, accuracy=0.0,
                skip_rate=1.0, trades=0, net_pnl=0.0,
                gross_pnl=0.0, total_fees=0.0, correct=0,
                total_windows=total_test_windows,
            ))
            continue

        y_traded = y_test[mask]
        correct = int(y_traded.sum())
        accuracy = correct / n_traded

        # Compute PnL for traded windows
        traded_global_indices = [valid_indices_test[j] for j in range(len(mask)) if mask[j]]
        net_pnl = 0.0
        gross_pnl = 0.0
        total_fees = 0.0

        for gi in traded_global_indices:
            feats = all_features_test[gi]
            wd = window_data_test[gi]
            early_dir = feats.get("early_direction", 0.0)
            direction = "Up" if early_dir > 0 else "Down"
            ep = adjusted_entry_price(feats.get("early_cum_return", 0.0))
            _, pg, pn, fee = adjusted_pnl(direction, wd.resolution, ep)
            net_pnl += pn
            gross_pnl += pg
            total_fees += fee

        skip_rate = 1.0 - (n_traded / total_test_windows)

        results.append(ThresholdResult(
            threshold=threshold,
            accuracy=accuracy,
            skip_rate=skip_rate,
            trades=n_traded,
            net_pnl=net_pnl,
            gross_pnl=gross_pnl,
            total_fees=total_fees,
            correct=correct,
            total_windows=total_test_windows,
        ))

    return results


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

def walk_forward_validation(
    window_data: list[WindowData],
    windows: list[list[FastCandle]],
    all_candles: list[FastCandle],
    entry_minute: int,
) -> list[dict[str, Any]]:
    """3-month train / 1-month test rolling validation.

    For each fold, train model on prior 3 months, test on next month.
    Returns list of per-period results.
    """
    # Group window indices by month
    by_month: dict[str, list[int]] = defaultdict(list)
    for i, wd in enumerate(window_data):
        by_month[wd.timestamp.strftime("%Y-%m")].append(i)
    months = sorted(by_month.keys())

    if len(months) < WF_TRAIN_MONTHS + WF_TEST_MONTHS:
        return []

    fold_results: list[dict[str, Any]] = []

    for mi in range(WF_TRAIN_MONTHS, len(months), WF_TEST_MONTHS):
        if mi >= len(months):
            break

        test_month = months[mi]
        train_months = months[max(0, mi - WF_TRAIN_MONTHS):mi]

        # Gather train indices
        train_indices: list[int] = []
        for m in train_months:
            train_indices.extend(by_month[m])

        test_indices = by_month[test_month]

        if not train_indices or not test_indices:
            continue

        # Compute features for this fold
        train_feats: list[dict[str, float]] = []
        train_wds: list[WindowData] = []
        for idx in train_indices:
            wd = window_data[idx]
            feats = compute_all_features(wd, windows, all_candles, entry_minute=entry_minute)
            train_feats.append(feats)
            train_wds.append(wd)

        test_feats: list[dict[str, float]] = []
        test_wds: list[WindowData] = []
        for idx in test_indices:
            wd = window_data[idx]
            feats = compute_all_features(wd, windows, all_candles, entry_minute=entry_minute)
            test_feats.append(feats)
            test_wds.append(wd)

        # Build matrices
        X_tr, y_tr, _ = build_feature_matrix(train_wds, train_feats, entry_minute)
        X_te, y_te, valid_te = build_feature_matrix(test_wds, test_feats, entry_minute)

        if len(X_tr) < 100 or len(X_te) < 10:
            continue

        # Train model for this fold (use base model, no calibration for speed)
        base = HistGradientBoostingClassifier(
            max_iter=500,
            max_depth=5,
            learning_rate=0.05,
            min_samples_leaf=100,
            l2_regularization=1.0,
            max_bins=128,
            random_state=42,
        )
        base.fit(X_tr, y_tr)

        # Predict and compute accuracy
        preds = base.predict(X_te)
        correct = int((preds == y_te).sum())
        accuracy = correct / len(y_te) if len(y_te) > 0 else 0.0

        # Compute PnL for test period
        net_pnl = 0.0
        for j in valid_te:
            feats = test_feats[j]
            wd = test_wds[j]
            early_dir = feats.get("early_direction", 0.0)
            direction = "Up" if early_dir > 0 else "Down"
            ep = adjusted_entry_price(feats.get("early_cum_return", 0.0))
            _, _, pn, _ = adjusted_pnl(direction, wd.resolution, ep)
            net_pnl += pn

        fold_results.append({
            "period": test_month,
            "train_months": train_months,
            "train_size": len(X_tr),
            "test_size": len(X_te),
            "accuracy": accuracy,
            "correct": correct,
            "net_pnl": net_pnl,
            "profitable": net_pnl > 0,
        })

    return fold_results


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def print_feature_importance(model: CalibratedClassifierCV, top_n: int = 10) -> list[tuple[str, float]]:
    """Extract and print top features by importance from the calibrated model."""
    # CalibratedClassifierCV wraps estimators; get the base model's feature importances
    importances = np.zeros(NUM_FEATURES)
    count = 0
    for cal_est in model.calibrated_classifiers_:
        base_est = cal_est.estimator
        if hasattr(base_est, "feature_importances_"):
            importances += base_est.feature_importances_
            count += 1
    if count > 0:
        importances /= count

    # Sort by importance
    indices = np.argsort(importances)[::-1]
    top_features: list[tuple[str, float]] = []

    print(f"\n  {'Rank':<6} {'Feature':<25} {'Importance':>12}")
    print(f"  {'-'*45}")
    for rank, idx in enumerate(indices[:top_n], 1):
        name = FEATURE_NAMES[idx]
        imp = importances[idx]
        print(f"  {rank:<6} {name:<25} {imp:>12.4f}")
        top_features.append((name, float(imp)))

    return top_features


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 120)
    print("  BWO ML BACKTEST — HistGradientBoosting with Early-Window Features")
    print("  Calibrated classifier | Threshold sweep | Walk-forward validation")
    print("=" * 120)
    _flush()

    # Ensure model directory exists
    MODEL_DIR.mkdir(exist_ok=True)

    if not BTC_CSV.exists():
        print(f"  ERROR: {BTC_CSV} not found.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load data (same as backtest_before_window.py)
    # ------------------------------------------------------------------
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

    # Build timestamp -> candle_index map
    candle_by_ts: dict[datetime, int] = {}
    for i, c in enumerate(all_candles):
        candle_by_ts[c.timestamp] = i

    # Prepare window data
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

    # Train/test split (80/20 chronological)
    split_idx = int(len(window_data) * 0.80)
    train_data = window_data[:split_idx]
    test_data = window_data[split_idx:]
    print(f"  Train: {len(train_data):,} | Test: {len(test_data):,}")
    print(f"  Train: {train_data[0].timestamp} -> {train_data[-1].timestamp}")
    print(f"  Test:  {test_data[0].timestamp} -> {test_data[-1].timestamp}")
    _flush()

    # ------------------------------------------------------------------
    # Per entry-minute loop
    # ------------------------------------------------------------------
    report: dict[str, Any] = {
        "meta": {
            "total_candles": len(all_candles),
            "total_windows": len(window_data),
            "train_size": len(train_data),
            "test_size": len(test_data),
            "position_size": POSITION_SIZE,
            "entry_minutes": ENTRY_MINUTES,
        },
        "entry_minutes": {},
    }

    best_overall_entry_minute: int | None = None
    best_overall_net_pnl = -float("inf")
    best_overall_threshold: float | None = None

    for entry_minute in ENTRY_MINUTES:
        print(f"\n{'='*120}")
        print(f"  ENTRY MINUTE {entry_minute}")
        print(f"{'='*120}")
        _flush()

        # Compute features for all windows at this entry minute
        print(f"  Computing features (entry_minute={entry_minute})...", end=" ")
        _flush()
        t0 = time.time()
        all_features_list: list[dict[str, float]] = []
        for i, wd in enumerate(window_data):
            all_features_list.append(
                compute_all_features(wd, windows, all_candles, entry_minute=entry_minute)
            )
            if (i + 1) % 10000 == 0:
                print(f"{i+1:,}...", end=" ")
                _flush()
        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s)")
        _flush()

        # Split features
        train_features = all_features_list[:split_idx]
        test_features = all_features_list[split_idx:]

        # Build matrices
        print("  Building feature matrices...", end=" ")
        _flush()
        X_train, y_train, valid_train = build_feature_matrix(train_data, train_features, entry_minute)
        X_test, y_test, valid_test = build_feature_matrix(test_data, test_features, entry_minute)
        print(f"train {X_train.shape}, test {X_test.shape}")
        print(f"  Train target: {y_train.sum()} continue / {len(y_train) - y_train.sum()} reverse "
              f"({y_train.mean()*100:.1f}% continue)")
        print(f"  Test target:  {y_test.sum()} continue / {len(y_test) - y_test.sum()} reverse "
              f"({y_test.mean()*100:.1f}% continue)")
        _flush()

        if len(X_train) < 200 or len(X_test) < 50:
            print("  SKIP: insufficient data")
            continue

        # Train model
        print("  Training calibrated HistGradientBoosting...", end=" ")
        _flush()
        t0 = time.time()
        model = train_model(X_train, y_train)
        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s)")
        _flush()

        # In-sample accuracy
        is_preds = model.predict(X_train)
        is_accuracy = float((is_preds == y_train).sum()) / len(y_train)
        print(f"  In-sample accuracy: {is_accuracy*100:.1f}%")

        # Out-of-sample accuracy (raw, before threshold)
        oos_preds = model.predict(X_test)
        oos_accuracy = float((oos_preds == y_test).sum()) / len(y_test)
        print(f"  OOS accuracy (raw): {oos_accuracy*100:.1f}%")
        print(f"  IS/OOS gap: {abs(is_accuracy - oos_accuracy)*100:.1f}%")
        _flush()

        # Feature importance
        print("\n  Feature Importance (Top 10):")
        top_feats = print_feature_importance(model)
        _flush()

        # Threshold sweep
        print(f"\n  Threshold sweep (0.50 -> 0.95):")
        print(f"  {'Thresh':>8} {'Accuracy':>10} {'SkipRate':>10} {'Trades':>8} {'NetPnL':>12} {'Pass':>6}")
        print(f"  {'-'*58}")
        _flush()

        sweep_results = threshold_sweep(
            model, X_test, y_test, test_data, test_features,
            valid_test, entry_minute, len(test_data),
        )

        best_threshold: float | None = None
        best_threshold_result: ThresholdResult | None = None

        for tr in sweep_results:
            passes = (
                tr.accuracy >= MIN_ACCURACY
                and tr.skip_rate < MAX_SKIP_RATE
                and tr.trades >= MIN_TEST_TRADES
            )
            marker = " <--" if passes else ""

            # Print every 5th threshold + any passing threshold
            if int(round(tr.threshold * 100)) % 5 == 0 or passes:
                print(f"  {tr.threshold:>8.2f} {tr.accuracy*100:>9.1f}% {tr.skip_rate*100:>9.1f}% "
                      f"{tr.trades:>8,} ${tr.net_pnl:>+10,.0f}{marker}")

            # Track best passing threshold (highest accuracy with constraints)
            if passes:
                if best_threshold_result is None or tr.accuracy > best_threshold_result.accuracy:
                    best_threshold = tr.threshold
                    best_threshold_result = tr

        _flush()

        if best_threshold_result is not None:
            print(f"\n  BEST threshold: {best_threshold:.2f} -> "
                  f"acc {best_threshold_result.accuracy*100:.1f}%, "
                  f"skip {best_threshold_result.skip_rate*100:.1f}%, "
                  f"trades {best_threshold_result.trades:,}, "
                  f"PnL ${best_threshold_result.net_pnl:+,.0f}")
        else:
            print(f"\n  NO threshold meets criteria (acc>={MIN_ACCURACY*100}%, "
                  f"skip<{MAX_SKIP_RATE*100}%, trades>={MIN_TEST_TRADES})")

        # Walk-forward validation
        print(f"\n  Walk-forward validation (3mo train / 1mo test)...", end=" ")
        _flush()
        t0 = time.time()
        wf_results = walk_forward_validation(window_data, windows, all_candles, entry_minute)
        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s, {len(wf_results)} periods)")
        _flush()

        if wf_results:
            print(f"\n  {'Period':>10} {'TrainN':>8} {'TestN':>8} {'Accuracy':>10} {'NetPnL':>12} {'Prof':>6}")
            print(f"  {'-'*58}")
            for wf in wf_results:
                print(f"  {wf['period']:>10} {wf['train_size']:>8} {wf['test_size']:>8} "
                      f"{wf['accuracy']*100:>9.1f}% ${wf['net_pnl']:>+10,.0f} "
                      f"{'Y' if wf['profitable'] else 'N':>6}")

            wf_accs = [w["accuracy"] for w in wf_results]
            wf_profitable = sum(1 for w in wf_results if w["profitable"])
            wf_profitable_pct = wf_profitable / len(wf_results) if wf_results else 0.0
            wf_total_pnl = sum(w["net_pnl"] for w in wf_results)
            wf_mean_acc = sum(wf_accs) / len(wf_accs) if wf_accs else 0.0
            print(f"\n  WF summary: mean acc {wf_mean_acc*100:.1f}%, "
                  f"profitable {wf_profitable}/{len(wf_results)} ({wf_profitable_pct*100:.0f}%), "
                  f"total PnL ${wf_total_pnl:+,.0f}")
        else:
            wf_profitable_pct = 0.0
            wf_total_pnl = 0.0
            wf_mean_acc = 0.0
        _flush()

        # Bootstrap CI and permutation test on best threshold's filtered trades
        perm_p_value = 1.0
        ci_mean = 0.0
        ci_lo = 0.0
        ci_hi = 0.0

        if best_threshold_result is not None and best_threshold_result.trades > 0:
            probas = model.predict_proba(X_test)[:, 1]
            mask = probas >= best_threshold

            # Build correct_list for bootstrap
            y_filtered = y_test[mask]
            correct_list = [bool(v) for v in y_filtered]
            ci_mean, ci_lo, ci_hi = bootstrap_ci(correct_list)
            print(f"\n  Bootstrap 95% CI: [{ci_lo*100:.1f}%, {ci_hi*100:.1f}%] (mean {ci_mean*100:.1f}%)")

            # Permutation test: map predictions to direction strings
            filtered_indices = [valid_test[j] for j in range(len(mask)) if mask[j]]
            preds_str: list[str] = []
            actuals_str: list[str] = []
            for gi in filtered_indices:
                feats = test_features[gi]
                wd = test_data[gi]
                early_dir = feats.get("early_direction", 0.0)
                direction = "Up" if early_dir > 0 else "Down"
                preds_str.append(direction)
                actuals_str.append(wd.resolution)

            perm_p_value = permutation_test(preds_str, actuals_str)
            sig = perm_p_value < MIN_PERMUTATION_SIG
            print(f"  Permutation p-value: {perm_p_value:.4f} "
                  f"{'(significant)' if sig else '(not significant)'}")
        _flush()

        # Save model
        model_path = MODEL_DIR / f"bwo_model_min{entry_minute}.joblib"
        joblib.dump(model, model_path)
        print(f"\n  Model saved: {model_path}")
        _flush()

        # Determine IS/OOS gap
        is_oos_gap = abs(is_accuracy - oos_accuracy)

        # Store per-minute report
        minute_report: dict[str, Any] = {
            "in_sample_accuracy": round(is_accuracy, 4),
            "oos_accuracy_raw": round(oos_accuracy, 4),
            "is_oos_gap": round(is_oos_gap, 4),
            "train_samples": int(X_train.shape[0]),
            "test_samples": int(X_test.shape[0]),
            "target_continue_rate_train": round(float(y_train.mean()), 4),
            "target_continue_rate_test": round(float(y_test.mean()), 4),
            "top_features": [(name, round(imp, 4)) for name, imp in top_feats],
            "walk_forward": {
                "periods": len(wf_results),
                "mean_accuracy": round(wf_mean_acc, 4),
                "profitable_pct": round(wf_profitable_pct, 4),
                "total_pnl": round(wf_total_pnl, 2),
                "details": wf_results,
            },
            "bootstrap_ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
            "permutation_p_value": round(perm_p_value, 4),
        }

        if best_threshold_result is not None:
            minute_report["best_threshold"] = {
                "threshold": best_threshold,
                "accuracy": round(best_threshold_result.accuracy, 4),
                "skip_rate": round(best_threshold_result.skip_rate, 4),
                "trades": best_threshold_result.trades,
                "net_pnl": round(best_threshold_result.net_pnl, 2),
                "gross_pnl": round(best_threshold_result.gross_pnl, 2),
                "total_fees": round(best_threshold_result.total_fees, 2),
            }

            # Track overall best
            if best_threshold_result.net_pnl > best_overall_net_pnl:
                best_overall_net_pnl = best_threshold_result.net_pnl
                best_overall_entry_minute = entry_minute
                best_overall_threshold = best_threshold
        else:
            minute_report["best_threshold"] = None

        # Calibration table (full)
        minute_report["calibration_table"] = [
            {
                "threshold": tr.threshold,
                "accuracy": round(tr.accuracy, 4),
                "skip_rate": round(tr.skip_rate, 4),
                "trades": tr.trades,
                "net_pnl": round(tr.net_pnl, 2),
            }
            for tr in sweep_results
        ]

        # Success criteria check
        criteria: dict[str, bool] = {}
        if best_threshold_result is not None:
            criteria["accuracy_gte_90"] = best_threshold_result.accuracy >= MIN_ACCURACY
            criteria["skip_rate_lt_50"] = best_threshold_result.skip_rate < MAX_SKIP_RATE
            criteria["permutation_sig"] = perm_p_value < MIN_PERMUTATION_SIG
            criteria["is_oos_gap_lt_5"] = is_oos_gap < MAX_IS_OOS_GAP
            criteria["wf_75pct_profitable"] = wf_profitable_pct >= MIN_WF_PROFITABLE_PCT
            criteria["net_pnl_positive"] = best_threshold_result.net_pnl > 0
            criteria["min_500_trades"] = best_threshold_result.trades >= MIN_TEST_TRADES
            criteria["all_pass"] = all(criteria.values())
        else:
            criteria["accuracy_gte_90"] = False
            criteria["skip_rate_lt_50"] = False
            criteria["permutation_sig"] = False
            criteria["is_oos_gap_lt_5"] = is_oos_gap < MAX_IS_OOS_GAP
            criteria["wf_75pct_profitable"] = wf_profitable_pct >= MIN_WF_PROFITABLE_PCT
            criteria["net_pnl_positive"] = False
            criteria["min_500_trades"] = False
            criteria["all_pass"] = False

        minute_report["success_criteria"] = criteria

        report["entry_minutes"][str(entry_minute)] = minute_report

        # Print success criteria
        print(f"\n  Success Criteria (entry minute {entry_minute}):")
        print(f"  {'-'*55}")
        for cname, passed in criteria.items():
            if cname == "all_pass":
                continue
            status = "PASS" if passed else "FAIL"
            print(f"  [{status:>4}] {cname}")
        all_pass = criteria.get("all_pass", False)
        print(f"\n  {'ALL CRITERIA MET' if all_pass else 'NOT ALL CRITERIA MET'}")
        _flush()

    # ------------------------------------------------------------------
    # Overall summary
    # ------------------------------------------------------------------
    print(f"\n\n{'='*120}")
    print(f"  OVERALL SUMMARY")
    print(f"{'='*120}")

    print(f"\n  {'EntryMin':>10} {'ISAcc':>8} {'OOSAcc':>8} {'Gap':>7} "
          f"{'BestThresh':>12} {'ThreshAcc':>11} {'Trades':>8} {'NetPnL':>12} {'AllPass':>9}")
    print(f"  {'-'*92}")

    for em in ENTRY_MINUTES:
        key = str(em)
        if key not in report["entry_minutes"]:
            continue
        mr = report["entry_minutes"][key]
        bt = mr.get("best_threshold")
        crit = mr.get("success_criteria", {})

        if bt:
            print(f"  {em:>10} {mr['in_sample_accuracy']*100:>7.1f}% {mr['oos_accuracy_raw']*100:>7.1f}% "
                  f"{mr['is_oos_gap']*100:>6.1f}% "
                  f"{bt['threshold']:>12.2f} {bt['accuracy']*100:>10.1f}% "
                  f"{bt['trades']:>8,} ${bt['net_pnl']:>+10,.0f} "
                  f"{'YES' if crit.get('all_pass') else 'NO':>9}")
        else:
            print(f"  {em:>10} {mr['in_sample_accuracy']*100:>7.1f}% {mr['oos_accuracy_raw']*100:>7.1f}% "
                  f"{mr['is_oos_gap']*100:>6.1f}% "
                  f"{'N/A':>12} {'N/A':>11} {'N/A':>8} {'N/A':>12} {'NO':>9}")

    if best_overall_entry_minute is not None:
        print(f"\n  BEST: entry minute {best_overall_entry_minute}, "
              f"threshold {best_overall_threshold:.2f}, "
              f"net PnL ${best_overall_net_pnl:+,.0f}")
        report["best_entry_minute"] = best_overall_entry_minute
        report["best_threshold"] = best_overall_threshold
    else:
        print(f"\n  NO entry minute meets all success criteria.")
        report["best_entry_minute"] = None
        report["best_threshold"] = None

    # Save report
    with open(REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved: {REPORT_JSON}")
    print(f"  Models saved: {MODEL_DIR}/bwo_model_min*.joblib")
    print(f"\n{'='*120}")
    _flush()


if __name__ == "__main__":
    main()
