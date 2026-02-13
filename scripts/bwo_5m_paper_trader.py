#!/usr/bin/env python3
"""BWO 5-Minute Paper Trader — Continuation Filter on Real Polymarket 5m Markets.

5m markets at minute 2 entry: 86.9% accuracy, 49.8% skip rate, $20.71 EV/trade.

Strategy:
  1. At minute 0: Discover 5m Polymarket market, fetch CLOB prices
  2. At minute 2: Observe BTC return, run continuation model
  3. If model says continue + confidence >= threshold: BUY continuation side
  4. Hold to settlement at minute 5

Usage:  python scripts/bwo_5m_paper_trader.py
"""

from __future__ import annotations

import csv
import json
import logging
import math
import sys
import time
import traceback
from bisect import bisect_right
from collections import namedtuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fast_loader import FastCandle, load_csv_fast

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("bwo_5m_paper")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "logs"
MODEL_DIR = PROJECT_ROOT / "models"
for _d in [LOG_DIR, MODEL_DIR]:
    _d.mkdir(exist_ok=True)

TRADES_JSONL = LOG_DIR / "bwo_5m_paper_trades.jsonl"
SUMMARY_JSON = LOG_DIR / "bwo_5m_paper_summary.json"
BTC_SPOT_CSV = DATA_DIR / "btc_1m_2y.csv"
BTC_FUTURES_CSV = DATA_DIR / "btc_futures_1m_2y.csv"
ETH_FUTURES_CSV = DATA_DIR / "eth_futures_1m_2y.csv"
DVOL_CSV = DATA_DIR / "deribit_dvol_1m.csv"
DERIBIT_PERP_CSV = DATA_DIR / "deribit_btc_perp_1m.csv"

CONT_MODEL_PATH = MODEL_DIR / "paper_5m_continuation_min2.joblib"
FEATURE_NAMES_PATH = MODEL_DIR / "paper_5m_feature_names.json"

# Use data-api.binance.vision — works from US/EU droplets (no geo-restriction)
BINANCE_SPOT_URL = "https://data-api.binance.vision/api/v3/klines"
DERIBIT_TICKER_URL = "https://www.deribit.com/api/v2/public/ticker"
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

INITIAL_BANKROLL = 200.0
POSITION_PCT = 0.30  # Reduced from 0.50 — limits max single loss
MAX_ENTRY_PRICE = 0.90  # Hard cap — EV check below is the real gatekeeper
MIN_ENTRY_PRICE = 0.65  # Floor — $0.60-0.65 entries had 50% WR (death zone)
MIN_EDGE = 0.03  # Minimum required edge: cont_prob must exceed entry_price + MIN_EDGE
MIN_BTC_RETURN = 0.0003  # 0.03% — skip noise moves
ENTRY_MINUTE = 2
CONT_THRESHOLD = 0.80  # Raised from 0.75 — 0.75-0.80 bucket had -$517 PnL
FEE_CONSTANT = 0.25
POLL_INTERVAL = 5  # 5s for 5m windows (faster than 15m)
WINDOW_SECONDS = 300  # 5 minutes

OUTCOME_MAP = {"UP": "YES", "DOWN": "NO", "YES": "YES", "NO": "NO"}

_client: httpx.Client | None = None


def get_client() -> httpx.Client:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.Client(timeout=httpx.Timeout(15.0))
    return _client


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


# =========================================================================
# SECTION 1: Data Loading & Model Training
# =========================================================================

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


def group_into_5m_windows(candles: list[FastCandle]) -> list[list[FastCandle]]:
    windows: list[list[FastCandle]] = []
    i = 0
    n = len(candles)
    while i < n:
        ts = candles[i].timestamp
        ts_unix = int(ts.timestamp())
        if ts_unix % 300 == 0 and i + 4 < n:
            w = candles[i:i + 5]
            if (w[-1].timestamp - w[0].timestamp).total_seconds() <= 300:
                windows.append(w)
                i += 5
                continue
        i += 1
    return windows


def compute_features_5m(
    candles: list[FastCandle],
    window_candles: list[FastCandle],
    entry_minute: int,
    ci: int,
    futures_data: list[dict] | None = None,
    futures_lookup: dict[datetime, int] | None = None,
    deribit_price: float = 0.0,
) -> dict[str, float]:
    """Compute features for a live 5m window at entry_minute."""
    features: dict[str, float] = {}

    hist_start = max(0, ci - 1440)
    hist_end = ci
    hist_len = hist_end - hist_start

    # Early window features
    early = window_candles[:entry_minute]
    if early and window_candles[0].open > 0:
        cum_ret = (early[-1].close - window_candles[0].open) / window_candles[0].open
        features["early_cum_return"] = cum_ret
        features["early_direction"] = 1.0 if cum_ret > 0 else (-1.0 if cum_ret < 0 else 0.0)
        features["early_magnitude"] = abs(cum_ret)
    else:
        features["early_cum_return"] = 0.0
        features["early_direction"] = 0.0
        features["early_magnitude"] = 0.0

    green = sum(1 for c in early if c.close > c.open) if early else 0
    features["early_green_ratio"] = green / len(early) if early else 0.0

    if len(early) >= 2:
        sq_sum = 0.0
        for j in range(1, len(early)):
            if early[j - 1].close > 0:
                r = (early[j].close - early[j - 1].close) / early[j - 1].close
                sq_sum += r * r
        features["early_vol"] = math.sqrt(sq_sum / (len(early) - 1)) if sq_sum > 0 else 0.0
    else:
        features["early_vol"] = 0.0

    max_move = 0.0
    for c in early:
        if c.open > 0:
            m = abs(c.close - c.open) / c.open
            if m > max_move:
                max_move = m
    features["early_max_move"] = max_move

    body_ratios = []
    for c in early:
        rng = c.high - c.low
        if rng > 0:
            body_ratios.append(abs(c.close - c.open) / rng)
    features["early_body_ratio"] = sum(body_ratios) / len(body_ratios) if body_ratios else 0.0

    early_vol_sum = sum(c.volume for c in early)
    if hist_len >= 30:
        recent_vol = sum(candles[j].volume for j in range(max(hist_end - 30, hist_start), hist_end))
        avg_vol = recent_vol / min(30, hist_len)
        features["early_vol_surge"] = (early_vol_sum / len(early)) / avg_vol if avg_vol > 0 and early else 1.0
    else:
        features["early_vol_surge"] = 1.0

    if early:
        c0 = early[0]
        if c0.high > c0.low:
            features["early_close_position"] = (c0.close - c0.low) / (c0.high - c0.low)
        else:
            features["early_close_position"] = 0.5
    else:
        features["early_close_position"] = 0.5

    # Signal vs ATR
    if hist_len >= 14 and window_candles and window_candles[0].open > 0:
        atr_sum = 0.0
        for j in range(hist_end - 14, hist_end):
            c = candles[j]
            tr = c.high - c.low
            if j > 0:
                prev_c = candles[j - 1].close
                tr = max(tr, abs(c.high - prev_c), abs(c.low - prev_c))
            atr_sum += tr
        atr = atr_sum / 14
        features["signal_vs_atr"] = abs(features["early_cum_return"]) * window_candles[0].open / atr if atr > 0 else 0.0
    else:
        features["signal_vs_atr"] = 0.0

    # Deribit basis
    if deribit_price > 0 and ci > 0:
        sc = candles[ci - 1].close
        if sc > 0:
            features["deribit_basis_bps"] = (deribit_price - sc) / sc * 10000
        else:
            features["deribit_basis_bps"] = 0.0
    else:
        features["deribit_basis_bps"] = 0.0

    return features


def train_5m_model() -> tuple[Any, list[str]]:
    """Train continuation model on last 3 months of 5m windows."""
    log.info("Training 5m continuation model...")

    all_candles = load_csv_fast(BTC_SPOT_CSV)
    all_candles.sort(key=lambda c: c.timestamp)
    seen: set[datetime] = set()
    unique: list[FastCandle] = []
    for c in all_candles:
        if c.timestamp not in seen:
            seen.add(c.timestamp)
            unique.append(c)
    all_candles = unique
    log.info(f"  Spot BTC: {len(all_candles):,} candles")

    # Load auxiliary data for features
    futures_data, futures_lookup = _load_csv_dict(
        BTC_FUTURES_CSV, ["volume", "taker_buy_volume", "num_trades"]
    )
    for d in futures_data:
        d["num_trades"] = int(d.get("num_trades", 0))

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

    deribit_data: list[dict] = []
    deribit_lookup: dict[datetime, int] = {}
    if DERIBIT_PERP_CSV.exists():
        with open(DERIBIT_PERP_CSV) as f:
            for row in csv.DictReader(f):
                ts = _parse_ts(row["timestamp"])
                if ts is None:
                    continue
                deribit_lookup[ts] = len(deribit_data)
                deribit_data.append({"timestamp": ts, "close": float(row.get("close", 0))})

    # Build 5m windows
    windows = group_into_5m_windows(all_candles)
    candle_by_ts = {c.timestamp: i for i, c in enumerate(all_candles)}

    # Use last 3 months only
    cutoff = all_candles[-1].timestamp - timedelta(days=90)

    # Import compute function from backtest
    from scripts.bwo_5m_backtest import WindowData5m, compute_all_features_5m, compute_mi_bits

    window_data = []
    for wi, w in enumerate(windows):
        ts = w[0].timestamp
        if ts < cutoff:
            continue
        idx = candle_by_ts.get(ts)
        if idx is None:
            continue
        resolution = "Up" if w[-1].close >= w[0].open else "Down"
        window_data.append(WindowData5m(
            window_idx=wi, candle_idx=idx, resolution=resolution, timestamp=ts,
        ))

    log.info(f"  Training windows: {len(window_data):,}")

    # Compute features
    all_feature_dicts: list[dict[str, float]] = []
    targets: list[int] = []
    valid_mask: list[bool] = []

    for i, wd in enumerate(window_data):
        feats = compute_all_features_5m(
            wd, windows, all_candles, ENTRY_MINUTE,
            futures_data, futures_lookup,
            None, None,
            dvol_data, dvol_timestamps,
            deribit_data, deribit_lookup,
        )
        all_feature_dicts.append(feats)

        early_dir = feats.get("early_direction", 0.0)
        if early_dir == 0.0:
            targets.append(0)
            valid_mask.append(False)
        else:
            continued = (
                (wd.resolution == "Up" and early_dir > 0)
                or (wd.resolution == "Down" and early_dir < 0)
            )
            targets.append(1 if continued else 0)
            valid_mask.append(True)

        if (i + 1) % 5000 == 0:
            log.info(f"    {i+1:,} / {len(window_data):,}")

    # Feature selection by MI
    valid_indices = [i for i in range(len(window_data)) if valid_mask[i]]
    target_arr = np.array([targets[i] for i in valid_indices])

    all_feature_names = sorted(set().union(*(d.keys() for d in all_feature_dicts[:100])))
    mi_results: dict[str, float] = {}
    for fname in all_feature_names:
        vals = np.array([all_feature_dicts[i].get(fname, 0.0) for i in valid_indices])
        mi_results[fname] = compute_mi_bits(vals, target_arr) if vals.std() > 0 else 0.0

    sorted_mi = sorted(mi_results.items(), key=lambda x: x[1], reverse=True)
    feature_names = [f for f, mi in sorted_mi if mi >= 0.0005][:40]
    if len(feature_names) < 5:
        feature_names = [f for f, _ in sorted_mi[:20]]
    log.info(f"  Selected {len(feature_names)} features")

    X = np.array([[all_feature_dicts[i].get(f, 0.0) for f in feature_names] for i in valid_indices])
    y = target_arr

    model = HistGradientBoostingClassifier(
        max_iter=500, max_depth=5, learning_rate=0.05,
        min_samples_leaf=80, l2_regularization=1.0,
        max_bins=128, random_state=42,
    )
    model.fit(X, y)
    train_acc = float((model.predict(X) == y).mean())
    log.info(f"  Train accuracy: {train_acc*100:.1f}%")

    joblib.dump(model, CONT_MODEL_PATH)
    with open(FEATURE_NAMES_PATH, "w") as f:
        json.dump(feature_names, f)
    log.info(f"  Model saved to {CONT_MODEL_PATH}")

    return model, feature_names


def load_or_train_model() -> tuple[Any, list[str]]:
    if CONT_MODEL_PATH.exists() and FEATURE_NAMES_PATH.exists():
        log.info("Loading saved 5m model...")
        model = joblib.load(CONT_MODEL_PATH)
        with open(FEATURE_NAMES_PATH) as f:
            feature_names = json.load(f)
        log.info(f"  Loaded: {len(feature_names)} features")
        return model, feature_names
    return train_5m_model()


# =========================================================================
# SECTION 2: Binance API
# =========================================================================

def fetch_binance_spot(symbol: str = "BTCUSDT", limit: int = 500) -> list[FastCandle]:
    try:
        resp = get_client().get(
            BINANCE_SPOT_URL,
            params={"symbol": symbol, "interval": "1m", "limit": limit},
        )
        resp.raise_for_status()
        candles = []
        for k in resp.json():
            ts = datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc)
            candles.append(FastCandle(
                timestamp=ts, open=float(k[1]), high=float(k[2]),
                low=float(k[3]), close=float(k[4]), volume=float(k[5]),
            ))
        return candles
    except Exception as e:
        log.warning(f"Binance spot error: {e}")
        return []


def fetch_deribit_price() -> float:
    try:
        resp = get_client().get(
            DERIBIT_TICKER_URL,
            params={"instrument_name": "BTC-PERPETUAL"},
        )
        resp.raise_for_status()
        return float(resp.json().get("result", {}).get("last_price", 0))
    except Exception as e:
        log.warning(f"Deribit error: {e}")
        return 0.0


# =========================================================================
# SECTION 3: Polymarket API (5m markets)
# =========================================================================

def discover_5m_market(window_ts: int) -> dict[str, Any] | None:
    """Find Polymarket 5m market for a given window timestamp."""
    slug = f"btc-updown-5m-{window_ts}"
    try:
        resp = get_client().get(f"{GAMMA_API}/events", params={"slug": slug})
        resp.raise_for_status()
        events = resp.json()
    except Exception as e:
        log.warning(f"Gamma API error for {slug}: {e}")
        return None

    for event in events:
        for mkt in event.get("markets", []):
            question = (mkt.get("question") or "").lower()
            if "btc" not in question and "bitcoin" not in question:
                continue
            if mkt.get("closed", False):
                continue

            outcomes_raw = mkt.get("outcomes", "")
            if isinstance(outcomes_raw, str) and outcomes_raw:
                try:
                    outcomes = json.loads(outcomes_raw)
                except (json.JSONDecodeError, ValueError):
                    outcomes = []
            elif isinstance(outcomes_raw, list):
                outcomes = outcomes_raw
            else:
                outcomes = []

            clob_ids_raw = mkt.get("clobTokenIds", "")
            if isinstance(clob_ids_raw, str) and clob_ids_raw:
                try:
                    clob_ids = json.loads(clob_ids_raw)
                except (json.JSONDecodeError, ValueError):
                    clob_ids = []
            elif isinstance(clob_ids_raw, list):
                clob_ids = clob_ids_raw
            else:
                clob_ids = []

            yes_token = ""
            no_token = ""
            if clob_ids and outcomes and len(clob_ids) >= 2 and len(outcomes) >= 2:
                for idx_o, outcome in enumerate(outcomes):
                    mapped = OUTCOME_MAP.get(outcome.upper(), "")
                    if mapped == "YES" and idx_o < len(clob_ids):
                        yes_token = str(clob_ids[idx_o])
                    elif mapped == "NO" and idx_o < len(clob_ids):
                        no_token = str(clob_ids[idx_o])

            if not yes_token or not no_token:
                continue

            market_id = mkt.get("conditionId", "") or str(mkt.get("id", ""))
            return {
                "market_id": market_id,
                "question": mkt.get("question", ""),
                "yes_token_id": yes_token,
                "no_token_id": no_token,
                "slug": slug,
            }
    return None


def fetch_clob_orderbook(token_id: str) -> dict[str, Any]:
    result = {
        "best_bid": 0.0, "best_ask": 1.0,
        "bid_size": 0.0, "ask_size": 0.0,
        "total_bid_depth": 0.0, "total_ask_depth": 0.0,
    }
    try:
        resp = get_client().get(f"{CLOB_API}/book", params={"token_id": token_id})
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.warning(f"CLOB error: {e}")
        return result

    bids_raw = data.get("bids", [])
    asks_raw = data.get("asks", [])

    # CLOB API returns bids ascending, asks descending — sort to get best prices
    bids = sorted(bids_raw, key=lambda x: float(x.get("price", 0)), reverse=True)
    asks = sorted(asks_raw, key=lambda x: float(x.get("price", 0)))

    if bids:
        result["best_bid"] = float(bids[0].get("price", 0))
        result["bid_size"] = float(bids[0].get("size", 0))
        result["total_bid_depth"] = sum(float(b.get("size", 0)) for b in bids)
    if asks:
        result["best_ask"] = float(asks[0].get("price", 0))
        result["ask_size"] = float(asks[0].get("size", 0))
        result["total_ask_depth"] = sum(float(a.get("size", 0)) for a in asks)

    return result


# =========================================================================
# SECTION 4: Logging
# =========================================================================

def log_trade(record: dict[str, Any]) -> None:
    with open(TRADES_JSONL, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def update_summary(bankroll: float, trades: list[dict]) -> None:
    if not trades:
        pnls = []
    else:
        pnls = [t.get("pnl_net", 0) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    summary = {
        "updated_at": datetime.now(tz=timezone.utc).isoformat(),
        "strategy": "5m_continuation_min2",
        "bankroll": round(bankroll, 2),
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(trades), 4) if trades else 0,
        "avg_win": round(sum(wins) / len(wins), 2) if wins else 0,
        "avg_loss": round(abs(sum(losses) / len(losses)), 2) if losses else 0,
        "total_pnl": round(sum(pnls), 2),
        "ev_per_trade": round(sum(pnls) / len(trades), 2) if trades else 0,
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2, default=str)


# =========================================================================
# SECTION 5: Window Tracker
# =========================================================================

@dataclass
class Window5mTracker:
    ts: int  # Unix timestamp of 5m window start
    market: dict[str, Any] | None = None
    yes_book: dict[str, Any] | None = None
    no_book: dict[str, Any] | None = None
    # Phases
    market_discovered: bool = False
    decision_done: bool = False
    settlement_done: bool = False
    # Trade state
    entered: bool = False
    side: str = ""  # "YES" or "NO"
    entry_price: float = 0.0
    shares: float = 0.0
    fee: float = 0.0
    exit_price: float = 0.0
    pnl_net: float = 0.0
    # Model output
    cont_prob: float = 0.0
    btc_return: float = 0.0
    btc_direction: str = ""
    early_direction: float = 0.0
    skip_reason: str = ""


# =========================================================================
# SECTION 6: Main Loop
# =========================================================================

def main() -> None:
    log.info("=" * 70)
    log.info("BWO 5-MINUTE PAPER TRADER — Continuation at Minute 2")
    log.info(f"Threshold={CONT_THRESHOLD}, Price=[${MIN_ENTRY_PRICE}-${MAX_ENTRY_PRICE}], Position={POSITION_PCT*100:.0f}%, MinRet={MIN_BTC_RETURN*100:.2f}%")
    log.info("=" * 70)

    # Train or load model
    cont_model, feature_names = load_or_train_model()

    # Initialize state
    bankroll = INITIAL_BANKROLL
    trades: list[dict] = []

    # Load existing trades
    if TRADES_JSONL.exists():
        with open(TRADES_JSONL) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("entered"):
                        trades.append(rec)
                        bankroll = rec.get("bankroll_after", bankroll)
                except json.JSONDecodeError:
                    pass
        log.info(f"Resumed: {len(trades)} trades, bankroll=${bankroll:.2f}")

    # Data buffers
    spot_candles: list[FastCandle] = []
    deribit_price: float = 0.0
    active_windows: dict[int, Window5mTracker] = {}
    last_data_poll = 0.0

    log.info(f"Starting main loop (poll every {POLL_INTERVAL}s)...")

    while True:
        try:
            now = int(time.time())
            window_start = (now // WINDOW_SECONDS) * WINDOW_SECONDS
            elapsed = now - window_start
            next_window = window_start + WINDOW_SECONDS

            # Poll data
            if time.time() - last_data_poll >= POLL_INTERVAL - 1:
                spot_candles = fetch_binance_spot("BTCUSDT", 500)
                deribit_price = fetch_deribit_price()
                last_data_poll = time.time()

            # --- Phase 1: Discover market for current window (seconds 0-30) ---
            if elapsed < 30:
                if window_start not in active_windows:
                    active_windows[window_start] = Window5mTracker(ts=window_start)

                w = active_windows[window_start]
                if not w.market_discovered:
                    mkt = discover_5m_market(window_start)
                    if mkt:
                        w.market = mkt
                        w.market_discovered = True
                        log.info(f"[DISCOVER] {mkt['question'][:60]}")
                    else:
                        log.debug(f"[DISCOVER] No market for {window_start}")

            # Also try to pre-discover next window (last 30s of current)
            if elapsed >= 270:
                if next_window not in active_windows:
                    active_windows[next_window] = Window5mTracker(ts=next_window)
                nw = active_windows[next_window]
                if not nw.market_discovered:
                    mkt = discover_5m_market(next_window)
                    if mkt:
                        nw.market = mkt
                        nw.market_discovered = True
                        log.info(f"[PRE-DISCOVER] Next: {mkt['question'][:60]}")

            # --- Phase 2: Decision at minute 2 (seconds 120-180) ---
            if 115 <= elapsed < 180:
                w = active_windows.get(window_start)
                if w and w.market_discovered and not w.decision_done:
                    # Get enough spot candles
                    window_dt = datetime.fromtimestamp(window_start, tz=timezone.utc)
                    window_candles = [
                        c for c in spot_candles
                        if window_dt <= c.timestamp < window_dt + timedelta(minutes=ENTRY_MINUTE)
                    ]

                    if len(window_candles) < ENTRY_MINUTE:
                        if elapsed > 140:
                            log.debug(f"[WAIT] Only {len(window_candles)} candles at {elapsed}s")
                    else:
                        # Compute BTC return
                        if window_candles[0].open > 0:
                            btc_return = (window_candles[-1].close - window_candles[0].open) / window_candles[0].open
                        else:
                            btc_return = 0.0

                        w.btc_return = btc_return
                        w.btc_direction = "UP" if btc_return > 0 else ("DOWN" if btc_return < 0 else "FLAT")
                        w.early_direction = 1.0 if btc_return > 0 else (-1.0 if btc_return < 0 else 0.0)

                        if abs(btc_return) < MIN_BTC_RETURN:
                            w.skip_reason = f"noise: |ret|={abs(btc_return)*100:.4f}%<{MIN_BTC_RETURN*100:.2f}%"
                            w.decision_done = True
                            log.info(f"[SKIP] {w.skip_reason}")
                        else:
                            # Compute features
                            ci = None
                            for idx, c in enumerate(spot_candles):
                                if c.timestamp == window_candles[0].timestamp:
                                    ci = idx
                                    break

                            if ci is not None:
                                feats = compute_features_5m(
                                    spot_candles, window_candles, ENTRY_MINUTE, ci,
                                    deribit_price=deribit_price,
                                )

                                # Build feature vector
                                X = np.array([[feats.get(f, 0.0) for f in feature_names]])
                                cont_prob = float(cont_model.predict_proba(X)[0, 1])
                                w.cont_prob = cont_prob

                                log.info(
                                    f"[MIN2] BTC {w.btc_direction} ({btc_return*100:+.3f}%), "
                                    f"cont_prob={cont_prob:.3f}"
                                )

                                if cont_prob >= CONT_THRESHOLD:
                                    # Determine which side to buy
                                    if w.early_direction > 0:
                                        # BTC UP → buy YES (continuation = UP)
                                        side = "YES"
                                        token_id = w.market["yes_token_id"]
                                    else:
                                        # BTC DOWN → buy NO (continuation = DOWN)
                                        side = "NO"
                                        token_id = w.market["no_token_id"]

                                    # Fetch real CLOB price
                                    book = fetch_clob_orderbook(token_id)
                                    ask_price = book["best_ask"]
                                    ask_size = book["ask_size"]

                                    log.info(
                                        f"[ENTRY] {side} ask=${ask_price:.3f} "
                                        f"(size={ask_size:.0f})"
                                    )

                                    # EV-based entry: cont_prob must exceed entry_price + MIN_EDGE
                                    ev_ok = cont_prob > (ask_price + MIN_EDGE)
                                    price_ok = ask_price >= MIN_ENTRY_PRICE and ask_price <= MAX_ENTRY_PRICE
                                    depth_ok = ask_size >= 5  # Paper trading: relaxed depth

                                    if ev_ok and price_ok and depth_ok:
                                        # Paper buy
                                        position_value = bankroll * POSITION_PCT
                                        shares = position_value / ask_price
                                        fee = polymarket_fee(position_value, ask_price)

                                        w.entered = True
                                        w.side = side
                                        w.entry_price = ask_price
                                        w.shares = shares
                                        w.fee = fee

                                        expected_ev = cont_prob * (1.0 - ask_price) - (1.0 - cont_prob) * ask_price
                                        log.info(
                                            f"[BUY] {side} {shares:.1f} shares @ ${ask_price:.3f} "
                                            f"(${position_value:.2f}, fee=${fee:.2f}, "
                                            f"EV=${expected_ev:.3f}/unit)"
                                        )
                                    else:
                                        reasons = []
                                        if not ev_ok:
                                            reasons.append(f"EV: need cont>{ask_price+MIN_EDGE:.2f}, got {cont_prob:.3f}")
                                        if not price_ok:
                                            if ask_price < MIN_ENTRY_PRICE:
                                                reasons.append(f"price ${ask_price:.3f}<${MIN_ENTRY_PRICE}")
                                            else:
                                                reasons.append(f"price ${ask_price:.3f}>${MAX_ENTRY_PRICE}")
                                        if not depth_ok:
                                            reasons.append(f"depth={ask_size:.0f}<10")
                                        w.skip_reason = f"{side} ask=${ask_price:.3f}: {', '.join(reasons)}"
                                        log.info(f"[SKIP] {w.skip_reason}")
                                else:
                                    w.skip_reason = f"cont_prob={cont_prob:.3f} < {CONT_THRESHOLD}"
                                    log.info(f"[SKIP] {w.skip_reason}")
                            else:
                                w.skip_reason = "could not find window start in candle buffer"
                                log.warning(f"[SKIP] {w.skip_reason}")

                        w.decision_done = True

            # --- Phase 3: Settlement (seconds 280-300+) ---
            if elapsed >= 280:
                w = active_windows.get(window_start)
                if w and w.decision_done and not w.settlement_done:
                    window_dt = datetime.fromtimestamp(window_start, tz=timezone.utc)
                    window_candles = [
                        c for c in spot_candles
                        if window_dt <= c.timestamp < window_dt + timedelta(minutes=5)
                    ]

                    if len(window_candles) >= 4:
                        btc_up = window_candles[-1].close >= window_candles[0].open
                        resolution = "Up" if btc_up else "Down"

                        record: dict[str, Any] = {
                            "window_ts": window_dt.isoformat(),
                            "window_ts_unix": window_start,
                            "market_id": w.market["market_id"] if w.market else "",
                            "resolution": resolution,
                            "btc_return_pct": round(w.btc_return * 100, 4),
                            "cont_prob": round(w.cont_prob, 4),
                            "early_direction": w.early_direction,
                            "btc_direction": w.btc_direction,
                        }

                        if w.entered:
                            # Did continuation happen?
                            if w.side == "YES":
                                settles = 1.0 if btc_up else 0.0
                            else:  # NO
                                settles = 1.0 if not btc_up else 0.0

                            # Settlement fee
                            if settles > 0:
                                exit_fee = polymarket_fee(w.shares * settles, settles)
                            else:
                                exit_fee = 0.0
                            total_fee = w.fee + exit_fee

                            pnl_gross = w.shares * (settles - w.entry_price)
                            pnl_net = pnl_gross - total_fee
                            bankroll += pnl_net

                            correct = settles > 0

                            trade_rec = {
                                "entered": True,
                                "side": w.side,
                                "entry_price": round(w.entry_price, 4),
                                "settlement": round(settles, 2),
                                "shares": round(w.shares, 2),
                                "pnl_gross": round(pnl_gross, 2),
                                "fee": round(total_fee, 2),
                                "pnl_net": round(pnl_net, 2),
                                "bankroll_after": round(bankroll, 2),
                                "correct": correct,
                            }
                            record.update(trade_rec)
                            trades.append(trade_rec)

                            log.info(
                                f"[SETTLE] {resolution} | {w.side} → "
                                f"{'WIN' if correct else 'LOSS'} | "
                                f"PnL=${pnl_net:+.2f} | bankroll=${bankroll:.2f}"
                            )
                        else:
                            record["entered"] = False
                            record["skip_reason"] = w.skip_reason
                            log.info(f"[SETTLE] {resolution} (skipped: {w.skip_reason})")

                        log_trade(record)
                        update_summary(bankroll, trades)
                        w.settlement_done = True
                    elif elapsed > 290:
                        log.debug(f"[SETTLE] Only {len(window_candles)} candles — waiting")

            # Cleanup
            stale = [ts for ts in active_windows if ts < window_start - 600]
            for ts in stale:
                del active_windows[ts]

        except KeyboardInterrupt:
            log.info("Shutting down...")
            break
        except Exception:
            log.error(f"Loop error:\n{traceback.format_exc()}")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
