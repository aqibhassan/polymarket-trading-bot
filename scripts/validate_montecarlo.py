"""Monte Carlo & statistical validation over 2 years.

Bootstrap CI, permutation tests, rolling stability, ruin probability,
serial correlation, minimum sample size.

Uses numpy for vectorized bootstrap/permutation (10-100x faster than pure Python).
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fast_loader import load_csv_fast, group_into_15m_windows

CSV_2Y = PROJECT_ROOT / "data" / "btc_1m_2y.csv"
MINUTES_PER_WINDOW = 15
POSITION_SIZE = 100.0
SENSITIVITY = 0.07
TIERS = [(8, 0.0010), (9, 0.0008), (10, 0.0005)]
N_BOOTSTRAP = 10000
N_PERMUTATIONS = 10000
SEED = 42


def _sigmoid(cum_ret: float) -> float:
    x = SENSITIVITY * cum_ret * 10000
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class Trade:
    pnl: float
    pnl_dollar: float
    correct: bool
    entry_price: float


def run_strategy(windows: list[list[Any]]) -> list[Trade]:
    trades: list[Trade] = []
    for window in windows:
        window_open = float(window[0].open)
        if window_open == 0:
            continue
        final_close = float(window[-1].close)
        actual_green = final_close > window_open

        for entry_min, threshold in TIERS:
            if entry_min >= MINUTES_PER_WINDOW:
                continue
            current_close = float(window[entry_min].close)
            cum_ret = (current_close - window_open) / window_open
            if abs(cum_ret) < threshold:
                continue
            predict_green = cum_ret > 0
            entry_yes = _sigmoid(cum_ret)
            entry_price = entry_yes if predict_green else 1.0 - entry_yes
            if entry_price <= 0.01:
                continue
            settlement = (1.0 if actual_green else 0.0) if predict_green else (1.0 if not actual_green else 0.0)
            correct = predict_green == actual_green
            pnl = (settlement - entry_price) / entry_price
            trades.append(Trade(
                pnl=round(pnl, 6),
                pnl_dollar=round(pnl * POSITION_SIZE, 2),
                correct=correct,
                entry_price=round(entry_price, 6),
            ))
            break
    return trades


def bootstrap_ci(values: np.ndarray, n_boot: int, ci: float = 0.95) -> tuple[float, float, float]:
    """Bootstrap confidence interval using numpy vectorization."""
    rng = np.random.RandomState(SEED)
    n = len(values)
    # Generate all bootstrap indices at once: (n_boot, n) matrix
    indices = rng.randint(0, n, size=(n_boot, n))
    # Compute all means in one vectorized operation
    means = values[indices].mean(axis=1)
    means.sort()
    alpha = (1 - ci) / 2
    lower_idx = int(alpha * n_boot)
    upper_idx = int((1 - alpha) * n_boot)
    return float(values.mean()), float(means[lower_idx]), float(means[upper_idx])


def main() -> None:
    print("=" * 100)
    print("  MONTE CARLO & STATISTICAL VALIDATION (2 Years)")
    print("=" * 100)

    if not CSV_2Y.exists():
        print(f"  ERROR: {CSV_2Y} not found.")
        sys.exit(1)

    all_candles = load_csv_fast(CSV_2Y)
    all_candles.sort(key=lambda c: c.timestamp)
    seen: set[datetime] = set()
    unique = [c for c in all_candles if c.timestamp not in seen and not seen.add(c.timestamp)]  # type: ignore
    all_candles = unique
    print(f"\n  Loaded {len(all_candles):,} candles")

    windows = group_into_15m_windows(all_candles)
    trades = run_strategy(windows)
    n = len(trades)
    print(f"  {len(windows):,} windows, {n:,} trades")

    pnls = np.array([t.pnl for t in trades])
    correct_flags = np.array([1.0 if t.correct else 0.0 for t in trades])
    pnl_dollars = np.array([t.pnl_dollar for t in trades])

    # ========== 1. Bootstrap CI ==========
    print(f"\n  {'='*90}")
    print(f"  1. BOOTSTRAP CONFIDENCE INTERVALS ({N_BOOTSTRAP:,} resamples)")
    print(f"  {'='*90}")

    acc_mean, acc_lo, acc_hi = bootstrap_ci(correct_flags, N_BOOTSTRAP)
    print(f"  Accuracy: {acc_mean*100:.2f}%  95% CI: [{acc_lo*100:.2f}%, {acc_hi*100:.2f}%]")

    ev_mean, ev_lo, ev_hi = bootstrap_ci(pnls, N_BOOTSTRAP)
    print(f"  EV/trade: ${ev_mean*100:+.2f}  95% CI: [${ev_lo*100:+.2f}, ${ev_hi*100:+.2f}]")

    pnl_mean, pnl_lo, pnl_hi = bootstrap_ci(pnl_dollars, N_BOOTSTRAP)
    total_n = n
    print(f"  Total P&L: ${pnl_mean*total_n:+,.0f}  95% CI: [${pnl_lo*total_n:+,.0f}, ${pnl_hi*total_n:+,.0f}]")

    # Bootstrap Sharpe (vectorized)
    rng_np = np.random.RandomState(SEED + 1)
    indices = rng_np.randint(0, n, size=(N_BOOTSTRAP, n))
    samples = pnls[indices]  # (N_BOOTSTRAP, n)
    means = samples.mean(axis=1)
    stds = samples.std(axis=1)
    stds[stds == 0] = 0.001
    sharpes = (means / stds) * math.sqrt(35040)
    sharpes_sorted = np.sort(sharpes)

    real_sharpe = float((pnls.mean() / (pnls.std() or 0.001)) * math.sqrt(35040))
    print(f"  Sharpe: {real_sharpe:.1f}  "
          f"95% CI: [{sharpes_sorted[int(0.025*N_BOOTSTRAP)]:.1f}, {sharpes_sorted[int(0.975*N_BOOTSTRAP)]:.1f}]")

    # ========== 2. Permutation test ==========
    print(f"\n  {'='*90}")
    print(f"  2. PERMUTATION TEST ({N_PERMUTATIONS:,} shuffles)")
    print(f"  {'='*90}")

    real_total = float(pnl_dollars.sum())
    rng_perm = np.random.RandomState(SEED + 2)
    # Vectorized: generate random sign flips for all permutations at once
    signs = rng_perm.choice([-1, 1], size=(N_PERMUTATIONS, n))
    random_totals = (signs * pnl_dollars).sum(axis=1)
    beat_count = int((random_totals >= real_total).sum())

    p_value = beat_count / N_PERMUTATIONS
    print(f"  Real total P&L: ${real_total:+,.0f}")
    print(f"  Random strategies beating real: {beat_count}/{N_PERMUTATIONS}")
    print(f"  p-value: {p_value:.4f}")
    if p_value < 0.01:
        print(f"  RESULT: Highly significant (p < 0.01)")
    elif p_value < 0.05:
        print(f"  RESULT: Significant (p < 0.05)")
    else:
        print(f"  WARNING: Not statistically significant (p >= 0.05)")

    # ========== 3. Rolling stability ==========
    print(f"\n  {'='*90}")
    print(f"  3. ROLLING STABILITY (500-trade windows)")
    print(f"  {'='*90}")

    window_size = 500
    roll_pfs: list[float] = []
    roll_sharpes: list[float] = []
    roll_accs: list[float] = []
    min_pf = float("inf")
    min_sharpe = float("inf")
    min_acc = 1.0

    for i in range(0, n - window_size + 1, 100):
        chunk = trades[i : i + window_size]
        wins = [t for t in chunk if t.correct]
        losses = [t for t in chunk if not t.correct]
        gross_win = sum(t.pnl_dollar for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl_dollar for t in losses)) if losses else 0.001
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
        roll_pfs.append(pf)

        chunk_pnls = [t.pnl for t in chunk]
        m = sum(chunk_pnls) / window_size
        v = sum((p - m) ** 2 for p in chunk_pnls) / window_size
        s = math.sqrt(v) if v > 0 else 0.001
        sharpe = (m / s) * math.sqrt(35040)
        roll_sharpes.append(sharpe)

        acc = sum(1 for t in chunk if t.correct) / window_size
        roll_accs.append(acc)

        min_pf = min(min_pf, pf)
        min_sharpe = min(min_sharpe, sharpe)
        min_acc = min(min_acc, acc)

    print(f"  Profit factor — min: {min_pf:.2f}, avg: {sum(roll_pfs)/len(roll_pfs):.2f}, "
          f"always >1.5: {'YES' if min_pf > 1.5 else 'NO'}")
    print(f"  Sharpe ratio  — min: {min_sharpe:.1f}, avg: {sum(roll_sharpes)/len(roll_sharpes):.1f}, "
          f"always >1.0: {'YES' if min_sharpe > 1.0 else 'NO'}")
    print(f"  Accuracy      — min: {min_acc*100:.1f}%, avg: {sum(roll_accs)/len(roll_accs)*100:.1f}%, "
          f"always >80%: {'YES' if min_acc > 0.80 else 'NO'}")

    below_80 = sum(1 for a in roll_accs if a < 0.80)
    print(f"  Windows with <80% accuracy: {below_80}/{len(roll_accs)}")

    # ========== 4. Monte Carlo drawdown simulation ==========
    print(f"\n  {'='*90}")
    print(f"  4. MONTE CARLO DRAWDOWN SIMULATION ({N_BOOTSTRAP:,} paths)")
    print(f"  {'='*90}")

    win_rate = sum(1 for t in trades if t.correct) / n
    avg_win_pnl = sum(t.pnl_dollar for t in trades if t.correct) / max(1, sum(1 for t in trades if t.correct))
    avg_loss_pnl = sum(t.pnl_dollar for t in trades if not t.correct) / max(1, sum(1 for t in trades if not t.correct))

    print(f"  Using: win_rate={win_rate:.4f}, avg_win=${avg_win_pnl:.2f}, avg_loss=${avg_loss_pnl:.2f}")

    rng_mc = np.random.RandomState(SEED + 3)
    # Generate all random outcomes at once: (N_BOOTSTRAP, n)
    wins = rng_mc.random((N_BOOTSTRAP, n)) < win_rate
    # Build PnL matrix
    pnl_matrix = np.where(wins, avg_win_pnl, avg_loss_pnl)
    # Cumulative equity curves
    equity_curves = POSITION_SIZE + np.cumsum(pnl_matrix, axis=1)
    # Running peak
    peaks = np.maximum.accumulate(equity_curves, axis=1)
    # Drawdowns
    drawdowns = np.where(peaks > 0, (peaks - equity_curves) / peaks, 0)
    max_dds_arr = drawdowns.max(axis=1)

    # Ruin: equity ever <= 50, before ever reaching 200
    ever_ruined = (equity_curves <= 0.5 * POSITION_SIZE).any(axis=1)
    ever_doubled = (equity_curves >= 2 * POSITION_SIZE).any(axis=1)
    # Ruin before doubling: find first occurrence
    ruin_count = 0
    for i in range(N_BOOTSTRAP):
        if ever_ruined[i]:
            ruin_idx = np.argmax(equity_curves[i] <= 0.5 * POSITION_SIZE)
            if ever_doubled[i]:
                double_idx = np.argmax(equity_curves[i] >= 2 * POSITION_SIZE)
                if ruin_idx < double_idx:
                    ruin_count += 1
            else:
                ruin_count += 1

    max_dds = np.sort(max_dds_arr)
    print(f"  Max drawdown percentiles:")
    for pct in [50, 75, 90, 95, 99]:
        idx = int(pct / 100 * N_BOOTSTRAP)
        print(f"    {pct}th: {max_dds[idx]*100:.1f}%")

    ruin_prob = int(ruin_count) / N_BOOTSTRAP
    print(f"\n  Ruin probability (50% loss before doubling): {ruin_prob*100:.2f}%")
    if ruin_prob < 0.01:
        print(f"  RESULT: Very low ruin risk")
    elif ruin_prob < 0.05:
        print(f"  RESULT: Acceptable ruin risk")
    else:
        print(f"  WARNING: High ruin risk ({ruin_prob*100:.1f}%)")

    # ========== 5. Serial correlation ==========
    print(f"\n  {'='*90}")
    print(f"  5. SERIAL CORRELATION (autocorrelation of wins/losses)")
    print(f"  {'='*90}")

    # Lag-1 autocorrelation (numpy)
    outcomes = np.array([1.0 if t.correct else 0.0 for t in trades])
    mean_o = outcomes.mean()
    var_o = float(((outcomes - mean_o) ** 2).mean())

    if var_o > 0:
        cov_1 = float(((outcomes[:-1] - mean_o) * (outcomes[1:] - mean_o)).mean())
        autocorr_1 = cov_1 / var_o
    else:
        autocorr_1 = 0.0

    print(f"  Lag-1 autocorrelation: {autocorr_1:.4f}")
    se = 1.0 / math.sqrt(n)
    z_score = autocorr_1 / se
    print(f"  Z-score: {z_score:.2f} (significant if |z| > 1.96)")
    if abs(z_score) > 1.96:
        print(f"  RESULT: Significant serial correlation detected")
        if autocorr_1 > 0:
            print(f"    Wins tend to follow wins (momentum in strategy performance)")
        else:
            print(f"    Wins tend to follow losses (mean-reversion in strategy performance)")
    else:
        print(f"  RESULT: No significant serial correlation (wins/losses are independent)")

    for lag in [2, 3, 5]:
        if var_o > 0:
            ac = float(((outcomes[:-lag] - mean_o) * (outcomes[lag:] - mean_o)).mean()) / var_o
        else:
            ac = 0.0
        z = ac / se
        sig = "***" if abs(z) > 1.96 else ""
        print(f"  Lag-{lag}: {ac:.4f} (z={z:.2f}) {sig}")

    # ========== 6. Minimum sample size ==========
    print(f"\n  {'='*90}")
    print(f"  6. MINIMUM SAMPLE SIZE ANALYSIS")
    print(f"  {'='*90}")

    # For a binomial test: n >= z^2 * p * (1-p) / E^2
    # where E is the margin of error we want
    z_95 = 1.96
    p = win_rate
    for margin in [0.01, 0.02, 0.05]:
        min_n = math.ceil(z_95**2 * p * (1 - p) / margin**2)
        print(f"  For {margin*100:.0f}% margin of error: need {min_n:,} trades (have {n:,}) -> {'OK' if n >= min_n else 'NEED MORE'}")

    # How many trades to be 95% confident edge > 0?
    # Z = (p - 0.5) / sqrt(p*(1-p)/n) >= 1.645 (one-sided)
    if p > 0.5:
        min_n_edge = math.ceil(1.645**2 * p * (1 - p) / (p - 0.5)**2)
        print(f"  For 95% confidence edge > 50%: need {min_n_edge:,} trades (have {n:,}) -> {'OK' if n >= min_n_edge else 'NEED MORE'}")
    else:
        print(f"  WARNING: Win rate {p*100:.1f}% is not above 50%")

    # ========== Summary ==========
    print(f"\n  {'='*90}")
    print(f"  STATISTICAL VALIDATION SUMMARY")
    print(f"  {'='*90}")
    print(f"  Trades: {n:,}")
    print(f"  Accuracy: {win_rate*100:.2f}% [{acc_lo*100:.2f}%, {acc_hi*100:.2f}%] 95% CI")
    print(f"  EV/trade: ${ev_mean*100:+.2f} [${ev_lo*100:+.2f}, ${ev_hi*100:+.2f}] 95% CI")
    print(f"  p-value: {p_value:.4f} ({'significant' if p_value < 0.05 else 'NOT significant'})")
    print(f"  Min rolling PF: {min_pf:.2f}, Min rolling Sharpe: {min_sharpe:.1f}")
    print(f"  Ruin probability: {ruin_prob*100:.2f}%")
    print(f"  Serial correlation: {autocorr_1:.4f} ({'significant' if abs(z_score) > 1.96 else 'not significant'})")

    # Overall assessment
    issues = []
    if p_value >= 0.05:
        issues.append("Not statistically significant (p >= 0.05)")
    if min_pf < 1.0:
        issues.append(f"Profit factor drops below 1.0 ({min_pf:.2f})")
    if ruin_prob > 0.05:
        issues.append(f"High ruin probability ({ruin_prob*100:.1f}%)")
    if acc_lo < 0.80:
        issues.append(f"Lower CI bound below 80% ({acc_lo*100:.1f}%)")

    if issues:
        print(f"\n  ISSUES:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print(f"\n  ALL CHECKS PASSED - Strategy is statistically robust.")

    print(f"\n{'='*100}")


if __name__ == "__main__":
    main()
