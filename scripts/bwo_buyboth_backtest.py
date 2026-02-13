"""BWO v7 — Buy-Both Strategy Backtest.

STRATEGY:
  1. Buy BOTH YES and NO at $0.50 each BEFORE window ($1.00 total per trade)
  2. At minute N, run continuation filter on early BTC candles
  3. Decision:
     - Continuation confirmed → KEEP winner side, SELL loser side
     - No continuation       → SELL BOTH sides (small loss, avoid risk)
     - Flat / no signal      → SELL BOTH

ADVANTAGE: You always own the winning position — filter just decides whether to keep it.
RISK: Double fees/capital vs single-side. Filter errors mean keeping the loser.

COMPARISON: vs Hybrid v6 (single-side) and vs naive baselines.
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
    FEE_CONSTANT,
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
REPORT_JSON = PROJECT_ROOT / "data" / "bwo_buyboth_report.json"

# Each leg = $100 → total outlay = $200 per trade
LEG_SIZE = 100.0
ENTRY_PRICE = 0.50
ENTRY_SLIPPAGE = 0.005
EXIT_SPREAD = 0.02
EXIT_SLIPPAGE = 0.005

WF_TRAIN_MONTHS = 3
WF_TEST_MONTHS = 1

ENTRY_MINUTES = [1, 2, 3]


def _flush() -> None:
    sys.stdout.flush()


def polymarket_fee(position_size: float, price: float) -> float:
    if price <= 0 or price >= 1:
        return 0.0
    return position_size * FEE_CONSTANT * (price ** 2) * ((1.0 - price) ** 2)


# ---------------------------------------------------------------------------
# Pricing model (same as v6)
# ---------------------------------------------------------------------------

class PricingModel:
    def __init__(self) -> None:
        self.model: LogisticRegression | None = None

    def fit(self, returns: np.ndarray, final_up: np.ndarray) -> None:
        self.model = LogisticRegression(C=1.0, max_iter=1000)
        self.model.fit(returns.reshape(-1, 1), final_up)

    def yes_fair(self, ret: float) -> float:
        if self.model is None:
            return 0.50
        return float(np.clip(self.model.predict_proba(np.array([[ret]]))[0, 1], 0.01, 0.99))

    def no_fair(self, ret: float) -> float:
        return 1.0 - self.yes_fair(ret)

    def yes_sell(self, ret: float) -> float:
        return max(0.01, self.yes_fair(ret) - EXIT_SPREAD - EXIT_SLIPPAGE)

    def no_sell(self, ret: float) -> float:
        return max(0.01, self.no_fair(ret) - EXIT_SPREAD - EXIT_SLIPPAGE)


# ---------------------------------------------------------------------------
# Trade result
# ---------------------------------------------------------------------------

@dataclass
class BothTradeResult:
    action: str           # "keep_yes", "keep_no", "sell_both"
    # YES leg
    yes_entry: float
    yes_exit: float
    yes_pnl: float
    yes_fee: float
    # NO leg
    no_entry: float
    no_exit: float
    no_pnl: float
    no_fee: float
    # Combined
    total_pnl: float
    total_fee: float
    correct: bool         # Was the overall trade profitable?
    resolution: str       # "Up" or "Down"
    btc_return_at_n: float
    cont_prob: float


def simulate_buyboth(
    wd: WindowData,
    btc_ret_n: float,
    cont_prob: float,
    cont_threshold: float,
    pricing: PricingModel,
    variant: str,
) -> BothTradeResult:
    """Simulate a buy-both trade.

    Always buy YES at 0.50 and NO at 0.50 before window.
    At minute N, decide based on continuation filter.
    """
    eff_entry = ENTRY_PRICE + ENTRY_SLIPPAGE  # 0.505
    n_yes = LEG_SIZE / eff_entry
    n_no = LEG_SIZE / eff_entry
    fee_yes_entry = polymarket_fee(LEG_SIZE, eff_entry)
    fee_no_entry = polymarket_fee(LEG_SIZE, eff_entry)

    btc_up = btc_ret_n > 0
    btc_down = btc_ret_n < 0
    filter_continue = cont_prob >= cont_threshold

    # Settlement values
    yes_settles = 1.0 if wd.resolution == "Up" else 0.0
    no_settles = 1.0 - yes_settles

    if variant == "D":
        # Buy-both + continuation filter
        # Continuation → keep winner, sell loser
        # No continuation → sell both
        if filter_continue and (btc_up or btc_down):
            if btc_up:
                # Keep YES (winner), sell NO (loser)
                yes_exit = yes_settles  # hold to settlement
                no_exit_price = pricing.no_sell(btc_ret_n)
                no_exit = no_exit_price

                fee_yes_exit = polymarket_fee(LEG_SIZE, yes_settles) if yes_settles > 0 else 0.0
                fee_no_exit = polymarket_fee(LEG_SIZE, no_exit_price)

                yes_pnl = n_yes * (yes_exit - eff_entry) - fee_yes_entry - fee_yes_exit
                no_pnl = n_no * (no_exit - eff_entry) - fee_no_entry - fee_no_exit
                action = "keep_yes"
            else:
                # Keep NO (winner), sell YES (loser)
                no_exit = no_settles
                yes_exit_price = pricing.yes_sell(btc_ret_n)
                yes_exit = yes_exit_price

                fee_no_exit = polymarket_fee(LEG_SIZE, no_settles) if no_settles > 0 else 0.0
                fee_yes_exit = polymarket_fee(LEG_SIZE, yes_exit_price)

                yes_pnl = n_yes * (yes_exit - eff_entry) - fee_yes_entry - fee_yes_exit
                no_pnl = n_no * (no_exit - eff_entry) - fee_no_entry - fee_no_exit
                action = "keep_no"
        else:
            # Sell both
            yes_exit_price = pricing.yes_sell(btc_ret_n)
            no_exit_price = pricing.no_sell(btc_ret_n)
            yes_exit = yes_exit_price
            no_exit = no_exit_price

            fee_yes_exit = polymarket_fee(LEG_SIZE, yes_exit_price)
            fee_no_exit = polymarket_fee(LEG_SIZE, no_exit_price)

            yes_pnl = n_yes * (yes_exit - eff_entry) - fee_yes_entry - fee_yes_exit
            no_pnl = n_no * (no_exit - eff_entry) - fee_no_entry - fee_no_exit
            action = "sell_both"

    elif variant == "E":
        # Buy-both + always keep winner (no filter) — baseline comparison
        if btc_up:
            yes_exit = yes_settles
            no_exit = pricing.no_sell(btc_ret_n)
            fee_yes_exit = polymarket_fee(LEG_SIZE, yes_settles) if yes_settles > 0 else 0.0
            fee_no_exit = polymarket_fee(LEG_SIZE, no_exit)
            yes_pnl = n_yes * (yes_exit - eff_entry) - fee_yes_entry - fee_yes_exit
            no_pnl = n_no * (no_exit - eff_entry) - fee_no_entry - fee_no_exit
            action = "keep_yes"
        elif btc_down:
            no_exit = no_settles
            yes_exit = pricing.yes_sell(btc_ret_n)
            fee_no_exit = polymarket_fee(LEG_SIZE, no_settles) if no_settles > 0 else 0.0
            fee_yes_exit = polymarket_fee(LEG_SIZE, yes_exit)
            yes_pnl = n_yes * (yes_exit - eff_entry) - fee_yes_entry - fee_yes_exit
            no_pnl = n_no * (no_exit - eff_entry) - fee_no_entry - fee_no_exit
            action = "keep_no"
        else:
            yes_exit = pricing.yes_sell(btc_ret_n)
            no_exit = pricing.no_sell(btc_ret_n)
            fee_yes_exit = polymarket_fee(LEG_SIZE, yes_exit)
            fee_no_exit = polymarket_fee(LEG_SIZE, no_exit)
            yes_pnl = n_yes * (yes_exit - eff_entry) - fee_yes_entry - fee_yes_exit
            no_pnl = n_no * (no_exit - eff_entry) - fee_no_entry - fee_no_exit
            action = "sell_both"

    elif variant == "F":
        # Buy-both + SELECTIVE: only keep winner when filter says continue
        # AND sell both when filter says don't continue
        # PLUS: if filter says continue but direction is WRONG → still holding loser
        # This is same as D but we also check: if continuation says continue
        # in OTHER direction (BTC up but continuation of UP is low → sell both)
        # Actually this is the same as D. Let me make F different:
        # F = keep winner ONLY if filter confidence is HIGH (>0.70), else sell both
        high_conf = cont_prob >= 0.70
        if high_conf and (btc_up or btc_down):
            if btc_up:
                yes_exit = yes_settles
                no_exit = pricing.no_sell(btc_ret_n)
                fee_yes_exit = polymarket_fee(LEG_SIZE, yes_settles) if yes_settles > 0 else 0.0
                fee_no_exit = polymarket_fee(LEG_SIZE, no_exit)
                yes_pnl = n_yes * (yes_exit - eff_entry) - fee_yes_entry - fee_yes_exit
                no_pnl = n_no * (no_exit - eff_entry) - fee_no_entry - fee_no_exit
                action = "keep_yes"
            else:
                no_exit = no_settles
                yes_exit = pricing.yes_sell(btc_ret_n)
                fee_no_exit = polymarket_fee(LEG_SIZE, no_settles) if no_settles > 0 else 0.0
                fee_yes_exit = polymarket_fee(LEG_SIZE, yes_exit)
                yes_pnl = n_yes * (yes_exit - eff_entry) - fee_yes_entry - fee_yes_exit
                no_pnl = n_no * (no_exit - eff_entry) - fee_no_entry - fee_no_exit
                action = "keep_no"
        else:
            yes_exit = pricing.yes_sell(btc_ret_n)
            no_exit = pricing.no_sell(btc_ret_n)
            fee_yes_exit = polymarket_fee(LEG_SIZE, yes_exit)
            fee_no_exit = polymarket_fee(LEG_SIZE, no_exit)
            yes_pnl = n_yes * (yes_exit - eff_entry) - fee_yes_entry - fee_yes_exit
            no_pnl = n_no * (no_exit - eff_entry) - fee_no_entry - fee_no_exit
            action = "sell_both"
    else:
        raise ValueError(f"Unknown variant: {variant}")

    total_pnl = yes_pnl + no_pnl
    total_fee = fee_yes_entry + fee_no_entry + fee_yes_exit + fee_no_exit

    return BothTradeResult(
        action=action,
        yes_entry=eff_entry, yes_exit=yes_exit, yes_pnl=yes_pnl, yes_fee=fee_yes_entry + fee_yes_exit,
        no_entry=eff_entry, no_exit=no_exit, no_pnl=no_pnl, no_fee=fee_no_entry + fee_no_exit,
        total_pnl=total_pnl, total_fee=total_fee,
        correct=total_pnl > 0,
        resolution=wd.resolution,
        btc_return_at_n=btc_ret_n,
        cont_prob=cont_prob,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(trades: list[BothTradeResult], variant: str) -> dict[str, Any]:
    if not trades:
        return {}

    pnls = [t.total_pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    n = len(trades)
    win_rate = len(wins) / n
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 0.001
    rr = avg_win / avg_loss if avg_loss > 0 else 999
    ev = total_pnl / n

    # Sharpe
    rets = [p / (LEG_SIZE * 2) for p in pnls]  # per $200 deployed
    mean_r = sum(rets) / len(rets)
    var_r = sum((r - mean_r) ** 2 for r in rets) / len(rets)
    std_r = math.sqrt(var_r) if var_r > 0 else 0.001
    sharpe = (mean_r / std_r) * math.sqrt(35040)

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

    # Kelly
    if avg_loss > 0 and rr > 0:
        p = win_rate
        q = 1 - p
        kelly = max(0, (p * rr - q) / rr)
    else:
        kelly = 0

    # Action breakdown
    n_keep_yes = sum(1 for t in trades if t.action == "keep_yes")
    n_keep_no = sum(1 for t in trades if t.action == "keep_no")
    n_sell_both = sum(1 for t in trades if t.action == "sell_both")

    # PnL by action
    keep_pnls = [t.total_pnl for t in trades if t.action in ("keep_yes", "keep_no")]
    sell_pnls = [t.total_pnl for t in trades if t.action == "sell_both"]
    keep_avg = sum(keep_pnls) / len(keep_pnls) if keep_pnls else 0
    sell_avg = sum(sell_pnls) / len(sell_pnls) if sell_pnls else 0

    # When keeping: win rate (correct prediction)
    keep_wins = sum(1 for t in trades if t.action in ("keep_yes", "keep_no") and t.total_pnl > 0)
    keep_total = n_keep_yes + n_keep_no
    keep_win_rate = keep_wins / keep_total if keep_total > 0 else 0

    return {
        "variant": variant,
        "total_trades": n,
        "win_rate": round(win_rate, 4),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "rr": round(rr, 2),
        "ev_per_trade": round(ev, 2),
        "total_pnl": round(total_pnl, 2),
        "total_fees": round(sum(t.total_fee for t in trades), 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(max_dd, 2),
        "kelly": round(kelly, 4),
        "n_keep_yes": n_keep_yes,
        "n_keep_no": n_keep_no,
        "n_sell_both": n_sell_both,
        "keep_avg_pnl": round(keep_avg, 2),
        "sell_avg_pnl": round(sell_avg, 2),
        "keep_win_rate": round(keep_win_rate, 4),
        "capital_per_trade": LEG_SIZE * 2,
        "ev_per_dollar": round(ev / (LEG_SIZE * 2), 4),
    }


def print_metrics(m: dict[str, Any]) -> None:
    print(f"\n  {'='*70}")
    print(f"  VARIANT {m['variant']}")
    print(f"  {'='*70}")
    n = m["total_trades"]
    print(f"  Total trades:        {n:,}  (capital: ${m['capital_per_trade']:.0f}/trade)")
    print(f"")
    print(f"  --- Actions ---")
    print(f"  Keep YES (hold):     {m['n_keep_yes']:>6,}  ({m['n_keep_yes']/n*100:>5.1f}%)  avg PnL: see below")
    print(f"  Keep NO  (hold):     {m['n_keep_no']:>6,}  ({m['n_keep_no']/n*100:>5.1f}%)")
    print(f"  Sell BOTH (exit):    {m['n_sell_both']:>6,}  ({m['n_sell_both']/n*100:>5.1f}%)")
    print(f"  Keep trades avg PnL: ${m['keep_avg_pnl']:>8.2f}  (win rate: {m['keep_win_rate']*100:.1f}%)")
    print(f"  Sell trades avg PnL: ${m['sell_avg_pnl']:>8.2f}")
    print(f"")
    print(f"  --- Win / Loss ---")
    print(f"  Win rate:            {m['win_rate']*100:.1f}%")
    print(f"  Avg win:             ${m['avg_win']:>8.2f}")
    print(f"  Avg loss:            ${m['avg_loss']:>8.2f}")
    print(f"  Risk/Reward:         {m['rr']:.2f}x")
    print(f"")
    print(f"  --- P&L ---")
    print(f"  Total PnL:           ${m['total_pnl']:>12,.2f}")
    print(f"  Total fees:          ${m['total_fees']:>12,.2f}")
    print(f"  EV per trade:        ${m['ev_per_trade']:>8.2f}")
    print(f"  EV per $ deployed:   ${m['ev_per_dollar']:.4f}  ({m['ev_per_dollar']*100:.2f}%)")
    print(f"")
    print(f"  --- Risk ---")
    print(f"  Sharpe (ann):        {m['sharpe']:.2f}")
    print(f"  Max drawdown:        ${m['max_drawdown']:>10,.2f}")
    print(f"  Kelly fraction:      {m['kelly']*100:.1f}%")
    _flush()


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------

def walk_forward(
    window_data: list[WindowData],
    windows: list[list[FastCandle]],
    all_feature_dicts: list[dict[str, float]],
    feature_names: list[str],
    continuation_targets: list[int],
    valid_mask: list[bool],
    btc_returns: list[float],
    cont_threshold: float,
    variant: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:

    by_month: dict[str, list[int]] = defaultdict(list)
    for i, wd in enumerate(window_data):
        if valid_mask[i]:
            by_month[wd.timestamp.strftime("%Y-%m")].append(i)
    months = sorted(by_month.keys())

    if len(months) < WF_TRAIN_MONTHS + WF_TEST_MONTHS:
        return [], {}

    all_trades: list[BothTradeResult] = []
    period_results: list[dict[str, Any]] = []

    for mi in range(WF_TRAIN_MONTHS, len(months), WF_TEST_MONTHS):
        if mi >= len(months):
            break

        test_month = months[mi]
        train_months = months[max(0, mi - WF_TRAIN_MONTHS):mi]

        train_idx: list[int] = []
        for m in train_months:
            train_idx.extend(by_month[m])
        test_idx = by_month[test_month]

        if len(train_idx) < 100 or len(test_idx) < 20:
            continue

        # Train continuation model
        X_tr = np.array([[all_feature_dicts[i].get(f, 0.0) for f in feature_names] for i in train_idx])
        y_tr = np.array([continuation_targets[i] for i in train_idx])

        model = HistGradientBoostingClassifier(
            max_iter=300, max_depth=4, learning_rate=0.05,
            min_samples_leaf=80, l2_regularization=1.0,
            max_bins=128, random_state=42,
        )
        model.fit(X_tr, y_tr)

        # Train pricing model
        tr_rets = np.array([btc_returns[i] for i in train_idx])
        tr_ups = np.array([1 if window_data[i].resolution == "Up" else 0 for i in train_idx])
        pricing = PricingModel()
        pricing.fit(tr_rets, tr_ups)

        # Test
        X_te = np.array([[all_feature_dicts[i].get(f, 0.0) for f in feature_names] for i in test_idx])
        probas = model.predict_proba(X_te)[:, 1]

        period_trades: list[BothTradeResult] = []
        for j, idx in enumerate(test_idx):
            trade = simulate_buyboth(
                window_data[idx], btc_returns[idx],
                float(probas[j]), cont_threshold,
                pricing, variant,
            )
            period_trades.append(trade)

        all_trades.extend(period_trades)

        # Period stats
        pnls = [t.total_pnl for t in period_trades]
        p_wins = [p for p in pnls if p > 0]
        p_losses = [p for p in pnls if p <= 0]
        wr = len(p_wins) / len(pnls) if pnls else 0
        aw = sum(p_wins) / len(p_wins) if p_wins else 0
        al = abs(sum(p_losses) / len(p_losses)) if p_losses else 0.001
        rr = aw / al if al > 0 else 999
        total = sum(pnls)
        n_keeps = sum(1 for t in period_trades if t.action in ("keep_yes", "keep_no"))
        n_sells = sum(1 for t in period_trades if t.action == "sell_both")

        period_results.append({
            "period": test_month,
            "trades": len(period_trades),
            "win_rate": round(wr * 100, 1),
            "avg_win": round(aw, 2),
            "avg_loss": round(al, 2),
            "rr": round(rr, 2),
            "pnl": round(total, 2),
            "keeps": n_keeps,
            "sells": n_sells,
            "profitable": total > 0,
        })

    overall = compute_metrics(all_trades, variant)
    return period_results, overall


# ---------------------------------------------------------------------------
# Feature computation (reuse from v5/v6)
# ---------------------------------------------------------------------------

def compute_all_window_features(
    window_data: list[WindowData],
    windows: list[list[FastCandle]],
    all_candles: list[FastCandle],
    entry_minute: int,
    futures_data: list[dict],
    futures_lookup: dict[datetime, int],
    eth_data: list[dict],
    eth_lookup: dict[datetime, int],
    dvol_data: list[dict],
    dvol_timestamps: list[datetime],
    deribit_data: list[dict],
    deribit_lookup: dict[datetime, int],
) -> tuple[list[dict[str, float]], list[int], list[bool], list[float]]:
    """Compute features, targets, masks, and BTC returns for all windows."""

    all_feature_dicts: list[dict[str, float]] = []
    continuation_targets: list[int] = []
    valid_mask: list[bool] = []
    btc_returns: list[float] = []

    t0 = time.time()
    for i, wd in enumerate(window_data):
        ci = wd.candle_idx
        w = windows[wd.window_idx]

        # BTC return at minute N
        if entry_minute <= len(w) and w[0].open > 0:
            ret_n = (w[entry_minute - 1].close - w[0].open) / w[0].open
        else:
            ret_n = 0.0
        btc_returns.append(ret_n)

        original = compute_all_features(wd, windows, all_candles, entry_minute=entry_minute)
        signal = compute_signal_quality_features(w, entry_minute, all_candles, ci)
        taker = compute_taker_features_simple(ci, futures_data, futures_lookup, all_candles)

        # DVOL
        dvol_feats: dict[str, float] = {"dvol_level": 0.0, "dvol_change_5": 0.0, "dvol_z": 0.0}
        if dvol_timestamps:
            di = bisect_right(dvol_timestamps, wd.timestamp) - 1
            if di >= 5:
                dvol_now = dvol_data[di]["close"]
                dvol_feats["dvol_level"] = dvol_now
                prev = dvol_data[di - 5]["close"]
                dvol_feats["dvol_change_5"] = (dvol_now - prev) / prev if prev > 0 else 0.0
                start_z = max(0, di - 24)
                recent = [dvol_data[j]["close"] for j in range(start_z, di + 1)]
                if len(recent) >= 5:
                    m_val = sum(recent) / len(recent)
                    v_val = sum((x - m_val) ** 2 for x in recent) / len(recent)
                    s_val = math.sqrt(v_val) if v_val > 0 else 0.001
                    dvol_feats["dvol_z"] = (dvol_now - m_val) / s_val

        # ETH
        eth_feats: dict[str, float] = {"eth_momentum_15": 0.0, "eth_taker_alignment": 0.0}
        ei = eth_lookup.get(wd.timestamp)
        bi_f = futures_lookup.get(wd.timestamp)
        if ei is not None and ei >= 15 and bi_f is not None and bi_f >= 15:
            ec = eth_data[ei - 1]["close"]
            eo = eth_data[ei - 15]["close"]
            eth_feats["eth_momentum_15"] = (ec - eo) / eo if eo > 0 else 0.0
            etv = sum(eth_data[j].get("volume", 0) for j in range(ei - 15, ei))
            ebv = sum(eth_data[j].get("taker_buy_volume", 0) for j in range(ei - 15, ei))
            eimb = (2 * ebv - etv) / etv if etv > 0 else 0.0
            bimb = taker.get("taker_imbalance_15", 0.0)
            eth_feats["eth_taker_alignment"] = 1.0 if (bimb > 0 and eimb > 0) or (bimb < 0 and eimb < 0) else 0.0

        # Deribit basis
        cross_feats: dict[str, float] = {"deribit_basis_bps": 0.0}
        th = wd.timestamp.replace(minute=0, second=0, microsecond=0)
        dbi = deribit_lookup.get(th)
        if dbi is not None and ci > 0:
            dc = deribit_data[dbi]["close"]
            sc = all_candles[ci - 1].close
            if sc > 0:
                cross_feats["deribit_basis_bps"] = (dc - sc) / sc * 10000

        combined = {**original, **signal, **taker, **dvol_feats, **eth_feats, **cross_feats}
        all_feature_dicts.append(combined)

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
            print(f"    {i+1:>8,} / {len(window_data):,} - {time.time()-t0:.0f}s")
            _flush()

    print(f"  Done: {time.time()-t0:.1f}s")
    return all_feature_dicts, continuation_targets, valid_mask, btc_returns


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 100)
    print("  BWO v7 — BUY-BOTH STRATEGY BACKTEST")
    print("  Buy YES + NO at $0.50 each BEFORE window ($200 total)")
    print("  Continuation filter → keep winner OR sell both")
    print("=" * 100)
    _flush()

    # Load data (same as v5/v6)
    print("\n  Loading data...")
    all_candles = load_csv_fast(BTC_SPOT_CSV)
    all_candles.sort(key=lambda c: c.timestamp)
    seen: set[datetime] = set()
    unique: list[FastCandle] = []
    for c in all_candles:
        if c.timestamp not in seen:
            seen.add(c.timestamp)
            unique.append(c)
    all_candles = unique
    print(f"  Spot BTC: {len(all_candles):,}")

    futures_data, futures_lookup = _load_futures(BTC_FUTURES_CSV, "BTC futures")
    eth_data, eth_lookup = _load_futures(ETH_FUTURES_CSV, "ETH futures")
    dvol_data, dvol_timestamps = _load_dvol()
    deribit_data, deribit_lookup = _load_deribit()
    _flush()

    # Windows
    print("\n  Building 15m windows...", end=" ")
    windows = group_into_15m_windows(all_candles)
    candle_by_ts = {c.timestamp: i for i, c in enumerate(all_candles)}
    window_data: list[WindowData] = []
    for wi, w in enumerate(windows):
        ts = w[0].timestamp
        idx = candle_by_ts.get(ts)
        if idx is None:
            continue
        resolution = "Up" if w[-1].close > w[0].open else "Down"
        window_data.append(WindowData(window_idx=wi, candle_idx=idx, resolution=resolution, timestamp=ts))
    print(f"{len(window_data):,} windows")
    _flush()

    report: dict[str, Any] = {"meta": {"total_windows": len(window_data), "leg_size": LEG_SIZE}, "results": {}}

    VARIANTS = ["D", "E", "F"]
    VAR_NAMES = {
        "D": "Buy-Both + Continuation Filter (threshold=dynamic)",
        "E": "Buy-Both + Always Keep Winner (no filter, baseline)",
        "F": "Buy-Both + High-Confidence Only (threshold=0.70)",
    }

    for entry_minute in ENTRY_MINUTES:
        print(f"\n{'='*100}")
        print(f"  ENTRY MINUTE {entry_minute}")
        print(f"{'='*100}")

        print(f"  Computing features...")
        _flush()

        feats, targets, mask, rets = compute_all_window_features(
            window_data, windows, all_candles, entry_minute,
            futures_data, futures_lookup, eth_data, eth_lookup,
            dvol_data, dvol_timestamps, deribit_data, deribit_lookup,
        )

        n_valid = sum(mask)
        n_cont = sum(1 for i in range(len(window_data)) if mask[i] and targets[i] == 1)
        cont_rate = n_cont / n_valid if n_valid > 0 else 0
        print(f"  Valid: {n_valid:,} | Continuation rate: {cont_rate*100:.1f}%")

        # Feature selection
        all_fnames = sorted(set().union(*(d.keys() for d in feats[:100])))
        valid_idx = [i for i in range(len(window_data)) if mask[i]]
        tgt = np.array([targets[i] for i in valid_idx])
        mi_results = {}
        for fn in all_fnames:
            vals = np.array([feats[i].get(fn, 0.0) for i in valid_idx])
            mi_results[fn] = compute_mi_bits(vals, tgt) if vals.std() > 0 else 0.0
        sorted_mi = sorted(mi_results.items(), key=lambda x: x[1], reverse=True)
        top_features = [f for f, mi in sorted_mi if mi >= 0.0005][:40]
        if len(top_features) < 5:
            top_features = [f for f, _ in sorted_mi[:20]]
        print(f"  Using {len(top_features)} features")
        _flush()

        entry_report: dict[str, Any] = {"entry_minute": entry_minute, "variants": {}}

        for variant in VARIANTS:
            print(f"\n  {'─'*70}")
            print(f"  VARIANT {variant}: {VAR_NAMES[variant]}")
            print(f"  {'─'*70}")
            _flush()

            # Determine threshold
            if variant == "D":
                # Sweep to find best
                print(f"\n  Threshold sweep...")
                print(f"  {'Thresh':>8} {'WinRate':>8} {'AvgWin':>8} {'AvgLoss':>8} {'R:R':>6} {'EV':>9} {'Keeps':>6} {'Sells':>6}")
                print(f"  {'─'*66}")

                split = int(len(valid_idx) * 0.8)
                tr_i = valid_idx[:split]
                te_i = valid_idx[split:]

                X_tr = np.array([[feats[i].get(f, 0.0) for f in top_features] for i in tr_i])
                y_tr = np.array([targets[i] for i in tr_i])
                mdl = HistGradientBoostingClassifier(
                    max_iter=300, max_depth=4, learning_rate=0.05,
                    min_samples_leaf=80, l2_regularization=1.0, max_bins=128, random_state=42,
                )
                mdl.fit(X_tr, y_tr)

                tr_rets = np.array([rets[i] for i in tr_i])
                tr_ups = np.array([1 if window_data[i].resolution == "Up" else 0 for i in tr_i])
                pm = PricingModel()
                pm.fit(tr_rets, tr_ups)

                X_te = np.array([[feats[i].get(f, 0.0) for f in top_features] for i in te_i])
                te_probs = mdl.predict_proba(X_te)[:, 1]

                best = {"ev": -999, "thresh": 0.50}
                for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
                    trades = [simulate_buyboth(window_data[te_i[j]], rets[te_i[j]], float(te_probs[j]), thresh, pm, "D")
                              for j in range(len(te_i))]
                    pnls = [t.total_pnl for t in trades]
                    ws = [p for p in pnls if p > 0]
                    ls = [p for p in pnls if p <= 0]
                    wr = len(ws) / len(pnls)
                    aw = sum(ws) / len(ws) if ws else 0
                    al = abs(sum(ls) / len(ls)) if ls else 0.001
                    ev = sum(pnls) / len(pnls)
                    nk = sum(1 for t in trades if t.action != "sell_both")
                    ns = sum(1 for t in trades if t.action == "sell_both")
                    marker = " ***" if ev > best["ev"] else ""
                    print(f"  {thresh:>8.2f} {wr*100:>7.1f}% ${aw:>7.2f} ${al:>7.2f} {aw/al if al > 0 else 999:>5.2f} ${ev:>8.2f} {nk:>5} {ns:>5}{marker}")
                    if ev > best["ev"]:
                        best = {"ev": ev, "thresh": thresh}
                ct = best["thresh"]
                print(f"\n  Best threshold: {ct}")
            elif variant == "E":
                ct = 0.0   # Always keep winner
            elif variant == "F":
                ct = 0.70  # High confidence only
            else:
                ct = 0.50
            _flush()

            # Walk-forward
            print(f"\n  --- Walk-Forward ---")
            print(f"  {'Period':>10} {'WinRate':>8} {'AvgWin':>8} {'AvgLoss':>8} {'R:R':>6} {'PnL':>10} {'Keeps':>6} {'Sells':>6} {'Prof':>5}")
            print(f"  {'─'*78}")
            _flush()

            wf_periods, wf_metrics = walk_forward(
                window_data, windows, feats, top_features,
                targets, mask, rets, ct, variant,
            )

            for pr in wf_periods:
                prof = "Y" if pr["profitable"] else "N"
                print(f"  {pr['period']:>10} {pr['win_rate']:>7.1f}% ${pr['avg_win']:>7.2f} "
                      f"${pr['avg_loss']:>7.2f} {pr['rr']:>5.2f} ${pr['pnl']:>9,.0f} "
                      f"{pr['keeps']:>5} {pr['sells']:>5} {prof:>5}")

            if wf_periods:
                n_prof = sum(1 for p in wf_periods if p["profitable"])
                print(f"\n  Walk-forward: {n_prof}/{len(wf_periods)} months profitable")

            print_metrics(wf_metrics)
            entry_report["variants"][variant] = {"threshold": ct, "walk_forward": wf_periods, "metrics": wf_metrics}

        report["results"][str(entry_minute)] = entry_report

    # ------------------------------------------------------------------
    # Comparison with v6 Hybrid B
    # ------------------------------------------------------------------
    print(f"\n{'='*100}")
    print(f"  COMPARISON: Buy-Both vs Single-Side (Hybrid v6-B)")
    print(f"{'='*100}")
    print(f"\n  {'Strategy':<40} {'Capital':>8} {'EV/trade':>10} {'EV/$':>8} {'R:R':>6} {'Sharpe':>7} {'TotalPnL':>12}")
    print(f"  {'─'*95}")

    for em in ENTRY_MINUTES:
        key = str(em)
        if key not in report["results"]:
            continue
        for var in ["D", "E", "F"]:
            if var not in report["results"][key]["variants"]:
                continue
            m = report["results"][key]["variants"][var].get("metrics", {})
            if not m:
                continue
            cap = m.get("capital_per_trade", 200)
            name = f"Min{em} {VAR_NAMES.get(var, var)[:30]}"
            print(f"  {name:<40} ${cap:>6.0f} ${m.get('ev_per_trade', 0):>9.2f} "
                  f"{m.get('ev_per_dollar', 0)*100:>6.2f}% {m.get('rr', 0):>5.2f} "
                  f"{m.get('sharpe', 0):>6.2f} ${m.get('total_pnl', 0):>11,.2f}")

    # v6 reference (from previous run)
    print(f"  {'─'*95}")
    print(f"  {'Min3 Hybrid v6-B (single YES)':<40} ${'100':>6} ${'18.79':>9} {'18.79':>6}% {'3.54':>5} {'52.83':>6} ${'1,162,964':>11}")

    with open(REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved: {REPORT_JSON}")
    print(f"{'='*100}")
    _flush()


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_futures(path: Path, name: str) -> tuple[list[dict], dict[datetime, int]]:
    data: list[dict] = []
    lookup: dict[datetime, int] = {}
    if path.exists():
        import csv
        with open(path) as f:
            for row in csv.DictReader(f):
                ts = _parse_ts(row["timestamp"])
                if ts is None:
                    continue
                lookup[ts] = len(data)
                data.append({
                    "timestamp": ts,
                    "close": float(row.get("close", 0)),
                    "volume": float(row.get("volume", 0)),
                    "taker_buy_volume": float(row.get("taker_buy_volume", 0)),
                    "num_trades": int(float(row.get("num_trades", 0))),
                })
        print(f"  {name}: {len(data):,}")
    return data, lookup


def _load_dvol() -> tuple[list[dict], list[datetime]]:
    data: list[dict] = []
    timestamps: list[datetime] = []
    if DVOL_CSV.exists():
        import csv
        with open(DVOL_CSV) as f:
            for row in csv.DictReader(f):
                ts = _parse_ts(row["timestamp"])
                if ts is None:
                    continue
                data.append({"timestamp": ts, "close": float(row.get("close", 0))})
                timestamps.append(ts)
        print(f"  DVOL: {len(data):,}")
    return data, timestamps


def _load_deribit() -> tuple[list[dict], dict[datetime, int]]:
    data: list[dict] = []
    lookup: dict[datetime, int] = {}
    if DERIBIT_PERP_CSV.exists():
        import csv
        with open(DERIBIT_PERP_CSV) as f:
            for row in csv.DictReader(f):
                ts = _parse_ts(row["timestamp"])
                if ts is None:
                    continue
                lookup[ts] = len(data)
                data.append({"timestamp": ts, "close": float(row.get("close", 0)), "volume": float(row.get("volume", 0))})
        print(f"  Deribit perp: {len(data):,}")
    return data, lookup


if __name__ == "__main__":
    main()
