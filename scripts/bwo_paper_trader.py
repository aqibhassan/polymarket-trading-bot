#!/usr/bin/env python3
"""BWO Paper Trader — Hybrid B + BuyBoth D on Real Polymarket Prices.

Runs BOTH strategies simultaneously on live Polymarket BTC 15m markets.
Uses real CLOB prices for entry/exit, Binance for BTC data, Deribit for basis.

Strategies:
  Hybrid B: Buy YES pre-window, continuation filter at min 3 manages position
  BuyBoth D: Buy YES+NO pre-window, continuation filter decides keep/sell

Usage:  python scripts/bwo_paper_trader.py
Deploy: bash scripts/deploy_paper.sh
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
from sklearn.linear_model import LogisticRegression
import joblib

# ---------------------------------------------------------------------------
# Path setup (works both locally and on droplet)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest_before_window import (
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

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("bwo_paper")

# Suppress verbose httpx request logging
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

TRADES_JSONL = LOG_DIR / "bwo_paper_trades.jsonl"
SUMMARY_JSON = LOG_DIR / "bwo_paper_summary.json"

BTC_SPOT_CSV = DATA_DIR / "btc_1m_2y.csv"
BTC_FUTURES_CSV = DATA_DIR / "btc_futures_1m_2y.csv"
ETH_FUTURES_CSV = DATA_DIR / "eth_futures_1m_2y.csv"
DVOL_CSV = DATA_DIR / "deribit_dvol_1m.csv"
DERIBIT_PERP_CSV = DATA_DIR / "deribit_btc_perp_1m.csv"

CONT_MODEL_PATH = MODEL_DIR / "paper_continuation_min3.joblib"
PRICING_MODEL_PATH = MODEL_DIR / "paper_pricing_min3.joblib"
FEATURE_NAMES_PATH = MODEL_DIR / "paper_feature_names.json"

# Use data-api.binance.vision — works from US/EU droplets (no geo-restriction)
BINANCE_SPOT_URL = "https://data-api.binance.vision/api/v3/klines"
BINANCE_FUTURES_URL = "https://fapi.binance.com/fapi/v1/klines"
BINANCE_FUTURES_URL_ALT = "https://data-api.binance.vision/api/v3/klines"  # Fallback
DERIBIT_TICKER_URL = "https://www.deribit.com/api/v2/public/ticker"
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

INITIAL_BANKROLL = 200.0
POSITION_PCT = 0.50
MAX_ENTRY_PRICE = 0.52
ENTRY_MINUTE = 3
CONT_THRESHOLD = 0.50
FEE_CONSTANT = 0.25
POLL_INTERVAL = 10
WF_TRAIN_MONTHS = 3

OUTCOME_MAP = {"UP": "YES", "DOWN": "NO", "YES": "YES", "NO": "NO"}

# Global HTTP client (reused for connection pooling)
_client: httpx.Client | None = None


def get_client() -> httpx.Client:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.Client(timeout=httpx.Timeout(15.0))
    return _client


# ---------------------------------------------------------------------------
# Fee model
# ---------------------------------------------------------------------------
def polymarket_fee(size: float, price: float) -> float:
    if price <= 0 or price >= 1:
        return 0.0
    return size * FEE_CONSTANT * (price ** 2) * ((1.0 - price) ** 2)


# =========================================================================
# SECTION 1: Model Training (startup)
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


def train_models() -> tuple[Any, Any, list[str]]:
    """Train continuation + pricing models on last 3 months of CSV data.

    Returns (continuation_model, pricing_model, feature_names).
    """
    log.info("Loading historical data for training...")

    # Load spot candles
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

    # Load futures
    futures_data, futures_lookup = _load_csv_dict(
        BTC_FUTURES_CSV, ["volume", "taker_buy_volume", "num_trades"]
    )
    for d in futures_data:
        d["num_trades"] = int(d.get("num_trades", 0))
    log.info(f"  BTC futures: {len(futures_data):,}")

    # ETH futures
    eth_data, eth_lookup = _load_csv_dict(
        ETH_FUTURES_CSV, ["close", "volume", "taker_buy_volume"]
    )
    log.info(f"  ETH futures: {len(eth_data):,}")

    # DVOL
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
    log.info(f"  DVOL: {len(dvol_data):,}")

    # Deribit perpetual
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
                    "volume": float(row.get("volume", 0)),
                })
    log.info(f"  Deribit perp: {len(deribit_data):,}")

    # Build 15m windows
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
    log.info(f"  Total windows: {len(window_data):,}")

    # Filter to last 3 months for training
    if window_data:
        cutoff = window_data[-1].timestamp - timedelta(days=90)
        train_window_data = [wd for wd in window_data if wd.timestamp >= cutoff]
    else:
        train_window_data = window_data
    log.info(f"  Training windows (last 3 months): {len(train_window_data):,}")

    # Compute features for training windows
    log.info(f"  Computing features (entry_minute={ENTRY_MINUTE})...")
    t0 = time.time()

    all_feature_dicts: list[dict[str, float]] = []
    continuation_targets: list[int] = []
    valid_mask: list[bool] = []
    btc_returns: list[float] = []

    for i, wd in enumerate(train_window_data):
        ci = wd.candle_idx
        w = windows[wd.window_idx]

        # BTC return at minute N
        if ENTRY_MINUTE <= len(w) and w[0].open > 0:
            ret_n = (w[ENTRY_MINUTE - 1].close - w[0].open) / w[0].open
        else:
            ret_n = 0.0
        btc_returns.append(ret_n)

        # Pre-window + early-window features
        original = compute_all_features(wd, windows, all_candles, entry_minute=ENTRY_MINUTE)
        signal = compute_signal_quality_features(w, ENTRY_MINUTE, all_candles, ci)
        taker = compute_taker_features_simple(ci, futures_data, futures_lookup, all_candles)

        # DVOL features
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
                    m = sum(recent) / len(recent)
                    v = sum((x - m) ** 2 for x in recent) / len(recent)
                    s = math.sqrt(v) if v > 0 else 0.001
                    dvol_feats["dvol_z"] = (dvol_now - m) / s

        # ETH features
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
            eth_feats["eth_taker_alignment"] = (
                1.0 if (bimb > 0 and eimb > 0) or (bimb < 0 and eimb < 0) else 0.0
            )

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

        # Continuation target
        early_dir = original.get("early_direction", 0.0)
        if early_dir == 0.0:
            continuation_targets.append(0)
            valid_mask.append(False)
        else:
            continued = (
                (wd.resolution == "Up" and early_dir > 0)
                or (wd.resolution == "Down" and early_dir < 0)
            )
            continuation_targets.append(1 if continued else 0)
            valid_mask.append(True)

        if (i + 1) % 2000 == 0:
            log.info(f"    {i+1:,} / {len(train_window_data):,}")

    log.info(f"  Features computed in {time.time()-t0:.1f}s")

    # Feature selection by MI
    all_feature_names = sorted(set().union(*(d.keys() for d in all_feature_dicts[:100])))
    valid_indices = [i for i in range(len(train_window_data)) if valid_mask[i]]
    target_arr = np.array([continuation_targets[i] for i in valid_indices])

    mi_results: dict[str, float] = {}
    for fname in all_feature_names:
        vals = np.array([all_feature_dicts[i].get(fname, 0.0) for i in valid_indices])
        mi_results[fname] = compute_mi_bits(vals, target_arr) if vals.std() > 0 else 0.0

    sorted_mi = sorted(mi_results.items(), key=lambda x: x[1], reverse=True)
    feature_names = [f for f, mi in sorted_mi if mi >= 0.0005][:40]
    if len(feature_names) < 5:
        feature_names = [f for f, _ in sorted_mi[:20]]
    log.info(f"  Selected {len(feature_names)} features by MI")

    # Build training matrices
    X = np.array([[all_feature_dicts[i].get(f, 0.0) for f in feature_names] for i in valid_indices])
    y = target_arr

    n_cont = int(y.sum())
    log.info(f"  Valid samples: {len(valid_indices):,}, continuation rate: {n_cont/len(y)*100:.1f}%")

    # Train continuation model
    cont_model = HistGradientBoostingClassifier(
        max_iter=300, max_depth=4, learning_rate=0.05,
        min_samples_leaf=80, l2_regularization=1.0,
        max_bins=128, random_state=42,
    )
    cont_model.fit(X, y)
    train_acc = float((cont_model.predict(X) == y).mean())
    log.info(f"  Continuation model train accuracy: {train_acc*100:.1f}%")

    # Train pricing model (P(UP | BTC return at min N))
    valid_returns = np.array([btc_returns[i] for i in valid_indices])
    valid_final_up = np.array([
        1 if train_window_data[i].resolution == "Up" else 0
        for i in valid_indices
    ])
    pricing_model = LogisticRegression(C=1.0, max_iter=1000)
    pricing_model.fit(valid_returns.reshape(-1, 1), valid_final_up)
    log.info(f"  Pricing model coef: {float(pricing_model.coef_[0][0]):.1f}")

    # Save models
    joblib.dump(cont_model, CONT_MODEL_PATH)
    joblib.dump(pricing_model, PRICING_MODEL_PATH)
    with open(FEATURE_NAMES_PATH, "w") as f:
        json.dump(feature_names, f)
    log.info(f"  Models saved to {MODEL_DIR}")

    return cont_model, pricing_model, feature_names


def load_or_train_models() -> tuple[Any, Any, list[str]]:
    """Load saved models or train from scratch."""
    if CONT_MODEL_PATH.exists() and PRICING_MODEL_PATH.exists() and FEATURE_NAMES_PATH.exists():
        log.info("Loading saved models...")
        cont_model = joblib.load(CONT_MODEL_PATH)
        pricing_model = joblib.load(PRICING_MODEL_PATH)
        with open(FEATURE_NAMES_PATH) as f:
            feature_names = json.load(f)
        log.info(f"  Loaded: {len(feature_names)} features")
        return cont_model, pricing_model, feature_names
    return train_models()


# =========================================================================
# SECTION 2: Binance API
# =========================================================================

def parse_binance_klines_spot(data: list) -> list[FastCandle]:
    candles = []
    for k in data:
        ts = datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc)
        candles.append(FastCandle(
            timestamp=ts,
            open=float(k[1]), high=float(k[2]),
            low=float(k[3]), close=float(k[4]),
            volume=float(k[5]),
        ))
    return candles


def parse_binance_klines_futures(data: list) -> list[dict]:
    rows = []
    for k in data:
        ts = datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc)
        rows.append({
            "timestamp": ts,
            "open": float(k[1]), "high": float(k[2]),
            "low": float(k[3]), "close": float(k[4]),
            "volume": float(k[5]),
            "taker_buy_volume": float(k[9]),
            "num_trades": int(k[8]),
        })
    return rows


def fetch_binance_spot(symbol: str = "BTCUSDT", limit: int = 500) -> list[FastCandle]:
    try:
        resp = get_client().get(
            BINANCE_SPOT_URL,
            params={"symbol": symbol, "interval": "1m", "limit": limit},
        )
        resp.raise_for_status()
        return parse_binance_klines_spot(resp.json())
    except Exception as e:
        log.warning(f"Binance spot error: {e}")
        return []


def fetch_binance_futures(symbol: str = "BTCUSDT", limit: int = 500) -> list[dict]:
    for url in [BINANCE_FUTURES_URL, BINANCE_FUTURES_URL_ALT]:
        try:
            resp = get_client().get(
                url,
                params={"symbol": symbol, "interval": "1m", "limit": limit},
            )
            resp.raise_for_status()
            return parse_binance_klines_futures(resp.json())
        except Exception:
            continue
    log.warning(f"Binance futures: all endpoints failed for {symbol}")
    return []


def fetch_binance_eth(limit: int = 100) -> list[dict]:
    return fetch_binance_futures("ETHUSDT", limit)


# =========================================================================
# SECTION 3: Deribit API
# =========================================================================

def fetch_deribit_price() -> float:
    """Fetch BTC-PERPETUAL last price from Deribit."""
    try:
        resp = get_client().get(
            DERIBIT_TICKER_URL,
            params={"instrument_name": "BTC-PERPETUAL"},
        )
        resp.raise_for_status()
        data = resp.json()
        return float(data.get("result", {}).get("last_price", 0))
    except Exception as e:
        log.warning(f"Deribit error: {e}")
        return 0.0


# =========================================================================
# SECTION 4: Polymarket API
# =========================================================================

def discover_market(window_ts: int) -> dict[str, Any] | None:
    """Find Polymarket market for a given 15m window timestamp."""
    slug = f"btc-updown-15m-{window_ts}"
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
            has_btc = "btc" in question or "bitcoin" in question
            if not has_btc:
                continue
            if mkt.get("closed", False):
                continue

            # Parse outcomes and token IDs
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
    """Fetch CLOB orderbook for a token. Returns best bid/ask + depth."""
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
        log.warning(f"CLOB error for {token_id[:20]}: {e}")
        return result

    bids = data.get("bids", [])
    asks = data.get("asks", [])

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
# SECTION 5: Runtime Feature Computation
# =========================================================================

def compute_runtime_features(
    spot_candles: list[FastCandle],
    futures_buffer: list[dict],
    eth_buffer: list[dict],
    deribit_price: float,
    window_start_ts: int,
    feature_names: list[str],
) -> dict[str, float] | None:
    """Compute features for the current window at minute 3 using live data."""
    if len(spot_candles) < 100:
        log.warning("Not enough spot candles for feature computation")
        return None

    # Build lookup
    candle_by_ts = {c.timestamp: i for i, c in enumerate(spot_candles)}

    # Group complete 15m windows from buffer
    windows = group_into_15m_windows(spot_candles)

    # Extract current window candles (first 3 minutes)
    window_dt = datetime.fromtimestamp(window_start_ts, tz=timezone.utc)
    current_candles = [
        c for c in spot_candles
        if window_dt <= c.timestamp < window_dt + timedelta(minutes=ENTRY_MINUTE)
    ]

    if len(current_candles) < ENTRY_MINUTE:
        log.warning(f"Only {len(current_candles)} candles for current window (need {ENTRY_MINUTE})")
        return None

    # Append current incomplete window
    windows.append(current_candles)
    current_wi = len(windows) - 1

    ci = candle_by_ts.get(current_candles[0].timestamp)
    if ci is None:
        log.warning("Cannot find current window start candle in buffer")
        return None

    wd = WindowData(
        window_idx=current_wi, candle_idx=ci,
        resolution="Unknown", timestamp=current_candles[0].timestamp,
    )

    # Compute features using same functions as training
    original = compute_all_features(wd, windows, spot_candles, entry_minute=ENTRY_MINUTE)
    signal = compute_signal_quality_features(current_candles, ENTRY_MINUTE, spot_candles, ci)

    # Taker features from futures buffer
    futures_lookup = {d["timestamp"]: i for i, d in enumerate(futures_buffer)}
    taker = compute_taker_features_simple(ci, futures_buffer, futures_lookup, spot_candles)

    # Deribit basis (live)
    cross_feats: dict[str, float] = {"deribit_basis_bps": 0.0}
    if deribit_price > 0 and ci > 0:
        spot_close = spot_candles[ci - 1].close
        if spot_close > 0:
            cross_feats["deribit_basis_bps"] = (deribit_price - spot_close) / spot_close * 10000

    # DVOL not available at runtime
    dvol_feats = {"dvol_level": 0.0, "dvol_change_5": 0.0, "dvol_z": 0.0}

    # ETH features from buffer
    eth_feats: dict[str, float] = {"eth_momentum_15": 0.0, "eth_taker_alignment": 0.0}
    if len(eth_buffer) >= 15:
        ec = eth_buffer[-1].get("close", 0)
        eo = eth_buffer[-15].get("close", 0)
        if eo > 0:
            eth_feats["eth_momentum_15"] = (ec - eo) / eo
        etv = sum(d.get("volume", 0) for d in eth_buffer[-15:])
        ebv = sum(d.get("taker_buy_volume", 0) for d in eth_buffer[-15:])
        eimb = (2 * ebv - etv) / etv if etv > 0 else 0.0
        bimb = taker.get("taker_imbalance_15", 0.0)
        eth_feats["eth_taker_alignment"] = (
            1.0 if (bimb > 0 and eimb > 0) or (bimb < 0 and eimb < 0) else 0.0
        )

    combined = {**original, **signal, **taker, **dvol_feats, **eth_feats, **cross_feats}

    # Build feature vector in model order
    return {f: combined.get(f, 0.0) for f in feature_names}


# =========================================================================
# SECTION 6: Trade Decisions
# =========================================================================

def decide_hybrid_b(
    btc_return: float, cont_prob: float, cont_threshold: float,
) -> str:
    """Hybrid B decision at minute 3.

    Returns: "HOLD", "SELL_CUT_LOSS", "SELL_TAKE_PROFIT"
    """
    btc_up = btc_return > 0
    btc_down = btc_return < 0
    filter_continue = cont_prob >= cont_threshold

    # We bought YES, so BTC UP = favorable
    if btc_down and filter_continue:
        return "SELL_CUT_LOSS"
    if btc_up and not filter_continue:
        return "SELL_TAKE_PROFIT"
    return "HOLD"


def decide_buyboth_d(
    btc_return: float, cont_prob: float, cont_threshold: float,
) -> str:
    """BuyBoth D decision at minute 3.

    Returns: "KEEP_YES", "KEEP_NO", "SELL_BOTH"
    """
    btc_up = btc_return > 0
    btc_down = btc_return < 0
    filter_continue = cont_prob >= cont_threshold

    if filter_continue and (btc_up or btc_down):
        return "KEEP_YES" if btc_up else "KEEP_NO"
    return "SELL_BOTH"


# =========================================================================
# SECTION 7: Logging
# =========================================================================

def log_trade(record: dict[str, Any]) -> None:
    """Append a trade record to the JSONL log."""
    with open(TRADES_JSONL, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def update_summary(
    hybrid_bankroll: float, buyboth_bankroll: float,
    hybrid_trades: list[dict], buyboth_trades: list[dict],
) -> None:
    """Write running summary to JSON."""
    def _stats(trades: list[dict]) -> dict[str, Any]:
        if not trades:
            return {"total": 0}
        pnls = [t.get("pnl_net", 0) for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        return {
            "total": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(trades), 4) if trades else 0,
            "avg_win": round(sum(wins) / len(wins), 2) if wins else 0,
            "avg_loss": round(abs(sum(losses) / len(losses)), 2) if losses else 0,
            "total_pnl": round(sum(pnls), 2),
            "ev_per_trade": round(sum(pnls) / len(trades), 2) if trades else 0,
        }

    summary = {
        "updated_at": datetime.now(tz=timezone.utc).isoformat(),
        "hybrid_b": {
            "bankroll": round(hybrid_bankroll, 2),
            "stats": _stats(hybrid_trades),
        },
        "buyboth_d": {
            "bankroll": round(buyboth_bankroll, 2),
            "stats": _stats(buyboth_trades),
        },
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2, default=str)


# =========================================================================
# SECTION 8: Window State Tracking
# =========================================================================

@dataclass
class WindowTracker:
    ts: int  # Unix timestamp of window start
    # Market info
    market: dict[str, Any] | None = None
    yes_book: dict[str, Any] | None = None
    no_book: dict[str, Any] | None = None
    fillable: bool = False
    # Phase flags
    entry_done: bool = False
    decision_done: bool = False
    settlement_done: bool = False
    # Hybrid B state
    hb_entered: bool = False
    hb_entry_price: float = 0.0
    hb_shares: float = 0.0
    hb_action: str = ""
    hb_exit_price: float = 0.0
    hb_pnl_net: float = 0.0
    hb_fee: float = 0.0
    # BuyBoth D state
    bb_entered: bool = False
    bb_yes_entry: float = 0.0
    bb_no_entry: float = 0.0
    bb_yes_shares: float = 0.0
    bb_no_shares: float = 0.0
    bb_action: str = ""
    bb_yes_exit: float = 0.0
    bb_no_exit: float = 0.0
    bb_pnl_net: float = 0.0
    bb_fee: float = 0.0
    # Continuation filter
    cont_prob: float = 0.0
    btc_return: float = 0.0
    btc_direction: str = ""
    # Raw CLOB data at minute 3
    min3_yes_bid: float = 0.0
    min3_yes_ask: float = 0.0
    min3_no_bid: float = 0.0
    min3_no_ask: float = 0.0


# =========================================================================
# SECTION 9: Main Loop
# =========================================================================

def main() -> None:
    log.info("=" * 70)
    log.info("BWO PAPER TRADER — Hybrid B + BuyBoth D")
    log.info("=" * 70)

    # Train or load models
    cont_model, pricing_model, feature_names = load_or_train_models()

    # Initialize bankrolls
    hybrid_bankroll = INITIAL_BANKROLL
    buyboth_bankroll = INITIAL_BANKROLL
    hybrid_trades: list[dict] = []
    buyboth_trades: list[dict] = []

    # Load existing trades if any
    if TRADES_JSONL.exists():
        with open(TRADES_JSONL) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if "hybrid_b" in rec and rec["hybrid_b"].get("entered"):
                        hybrid_trades.append(rec["hybrid_b"])
                        hybrid_bankroll = rec["hybrid_b"].get("bankroll_after", hybrid_bankroll)
                    if "buyboth_d" in rec and rec["buyboth_d"].get("entered"):
                        buyboth_trades.append(rec["buyboth_d"])
                        buyboth_bankroll = rec["buyboth_d"].get("bankroll_after", buyboth_bankroll)
                except json.JSONDecodeError:
                    pass
        log.info(f"Resumed: {len(hybrid_trades)} hybrid trades, {len(buyboth_trades)} buyboth trades")
        log.info(f"Bankrolls: Hybrid=${hybrid_bankroll:.2f}, BuyBoth=${buyboth_bankroll:.2f}")

    # Rolling data buffers
    spot_candles: list[FastCandle] = []
    futures_buffer: list[dict] = []
    eth_buffer: list[dict] = []
    deribit_price: float = 0.0

    # Window tracking
    active_windows: dict[int, WindowTracker] = {}
    last_data_poll = 0.0

    log.info(f"Starting main loop (poll every {POLL_INTERVAL}s)...")

    while True:
        try:
            now = int(time.time())
            window_start = (now // 900) * 900
            elapsed = now - window_start
            next_window = window_start + 900

            # --- Poll data every cycle ---
            if time.time() - last_data_poll >= POLL_INTERVAL - 1:
                spot_candles = fetch_binance_spot("BTCUSDT", 500)
                futures_buffer = fetch_binance_futures("BTCUSDT", 500)
                eth_buffer = fetch_binance_eth(100)
                deribit_price = fetch_deribit_price()
                last_data_poll = time.time()

            # --- Pre-window for NEXT window (last 120s of current window) ---
            if elapsed >= 780:
                if next_window not in active_windows:
                    active_windows[next_window] = WindowTracker(ts=next_window)

                nw = active_windows[next_window]
                if not nw.entry_done:
                    # Discover market (only once)
                    if nw.market is None:
                        mkt = discover_market(next_window)
                        if mkt:
                            nw.market = mkt
                            log.info(f"[PRE] Found market for {next_window}: {mkt['question'][:60]}")
                        else:
                            log.debug(f"[PRE] No market found for {next_window}")

                    # Re-poll CLOB every cycle until fillable or window opens
                    if nw.market and not nw.fillable:
                        nw.yes_book = fetch_clob_orderbook(nw.market["yes_token_id"])
                        nw.no_book = fetch_clob_orderbook(nw.market["no_token_id"])

                        yes_ask = nw.yes_book["best_ask"]
                        no_ask = nw.no_book["best_ask"] if nw.no_book else 1.0
                        yes_size = nw.yes_book["ask_size"]

                        log.info(
                            f"[PRE] YES ask=${yes_ask:.3f} (size={yes_size:.0f}), "
                            f"NO ask=${no_ask:.3f}, elapsed={elapsed}s"
                        )

                        # Check fillability
                        nw.fillable = yes_ask <= MAX_ENTRY_PRICE and yes_size >= 10

                        if nw.fillable:
                            # Hybrid B: buy YES
                            hb_size = hybrid_bankroll * POSITION_PCT
                            hb_shares = hb_size / yes_ask
                            hb_fee = polymarket_fee(hb_size, yes_ask)
                            nw.hb_entered = True
                            nw.hb_entry_price = yes_ask
                            nw.hb_shares = hb_shares
                            nw.hb_fee = hb_fee
                            log.info(
                                f"[HYBRID B] Paper BUY YES: {hb_shares:.1f} shares "
                                f"@ ${yes_ask:.3f} (${hb_size:.2f})"
                            )

                            # BuyBoth D: buy YES + NO
                            bb_half = buyboth_bankroll * POSITION_PCT / 2
                            bb_yes_shares = bb_half / yes_ask
                            bb_no_shares = bb_half / no_ask
                            bb_fee = polymarket_fee(bb_half, yes_ask) + polymarket_fee(bb_half, no_ask)
                            nw.bb_entered = True
                            nw.bb_yes_entry = yes_ask
                            nw.bb_no_entry = no_ask
                            nw.bb_yes_shares = bb_yes_shares
                            nw.bb_no_shares = bb_no_shares
                            nw.bb_fee = bb_fee
                            log.info(
                                f"[BUYBOTH D] Paper BUY YES+NO: "
                                f"{bb_yes_shares:.1f}@${yes_ask:.3f} + "
                                f"{bb_no_shares:.1f}@${no_ask:.3f}"
                            )
                            nw.entry_done = True

            # --- Early-window entry (first 120s of CURRENT window) ---
            if elapsed < 120:
                w = active_windows.get(window_start)
                if w and not w.entry_done and w.market and not w.fillable:
                    w.yes_book = fetch_clob_orderbook(w.market["yes_token_id"])
                    w.no_book = fetch_clob_orderbook(w.market["no_token_id"])
                    yes_ask = w.yes_book["best_ask"]
                    no_ask = w.no_book["best_ask"] if w.no_book else 1.0
                    yes_size = w.yes_book["ask_size"]
                    log.info(
                        f"[EARLY] YES ask=${yes_ask:.3f} (size={yes_size:.0f}), "
                        f"NO ask=${no_ask:.3f}, elapsed={elapsed}s"
                    )
                    w.fillable = yes_ask <= MAX_ENTRY_PRICE and yes_size >= 10
                    if w.fillable:
                        hb_size = hybrid_bankroll * POSITION_PCT
                        hb_shares = hb_size / yes_ask
                        hb_fee = polymarket_fee(hb_size, yes_ask)
                        w.hb_entered = True
                        w.hb_entry_price = yes_ask
                        w.hb_shares = hb_shares
                        w.hb_fee = hb_fee
                        log.info(f"[HYBRID B] Paper BUY YES: {hb_shares:.1f}@${yes_ask:.3f}")
                        bb_half = buyboth_bankroll * POSITION_PCT / 2
                        bb_yes_shares = bb_half / yes_ask
                        bb_no_shares = bb_half / no_ask
                        bb_fee = polymarket_fee(bb_half, yes_ask) + polymarket_fee(bb_half, no_ask)
                        w.bb_entered = True
                        w.bb_yes_entry = yes_ask
                        w.bb_no_entry = no_ask
                        w.bb_yes_shares = bb_yes_shares
                        w.bb_no_shares = bb_no_shares
                        w.bb_fee = bb_fee
                        log.info(f"[BUYBOTH D] Paper BUY YES+NO: {bb_yes_shares:.1f}@${yes_ask:.3f} + {bb_no_shares:.1f}@${no_ask:.3f}")
                        w.entry_done = True

            # Mark unfilled windows as done at minute 2 and log skip
            if 120 <= elapsed < 140:
                w = active_windows.get(window_start)
                if w and not w.entry_done:
                    last_ask = w.yes_book['best_ask'] if w.yes_book else 0.0
                    log.info(f"[SKIP] Window {window_start} — no fill by minute 2 (last YES ask=${last_ask:.3f})" if w.market else f"[SKIP] Window {window_start} — no market found")
                    w.entry_done = True
                    w.decision_done = True
                    w.settlement_done = True
                    # Log skip record
                    log_trade({
                        "window_ts": datetime.fromtimestamp(window_start, tz=timezone.utc).isoformat(),
                        "window_ts_unix": window_start,
                        "market_id": w.market["market_id"] if w.market else "",
                        "resolution": "SKIPPED",
                        "fillable": False,
                        "skip_reason": f"YES ask=${last_ask:.3f}" if w.market else "no market",
                        "pre_window": {
                            "yes_ask": last_ask,
                            "no_ask": w.no_book["best_ask"] if w.no_book else 0.0,
                        },
                        "hybrid_b": {"entered": False},
                        "buyboth_d": {"entered": False},
                    })

            # --- Decision at minute 3 (seconds 185-300) ---
            if 185 <= elapsed < 300:
                w = active_windows.get(window_start)
                if w and w.entry_done and not w.decision_done and (w.hb_entered or w.bb_entered):
                    # Compute features
                    feats = compute_runtime_features(
                        spot_candles, futures_buffer, eth_buffer,
                        deribit_price, window_start, feature_names,
                    )

                    if feats:
                        # BTC return at minute 3
                        window_dt = datetime.fromtimestamp(window_start, tz=timezone.utc)
                        window_candles = [
                            c for c in spot_candles
                            if window_dt <= c.timestamp < window_dt + timedelta(minutes=ENTRY_MINUTE)
                        ]
                        if window_candles and window_candles[0].open > 0:
                            btc_return = (
                                (window_candles[-1].close - window_candles[0].open)
                                / window_candles[0].open
                            )
                        else:
                            btc_return = 0.0

                        w.btc_return = btc_return
                        w.btc_direction = "UP" if btc_return > 0 else ("DOWN" if btc_return < 0 else "FLAT")

                        # Run continuation model
                        X = np.array([[feats[f] for f in feature_names]])
                        cont_prob = float(cont_model.predict_proba(X)[0, 1])
                        w.cont_prob = cont_prob

                        # Fetch minute-3 CLOB prices
                        if w.market:
                            yes_book_3 = fetch_clob_orderbook(w.market["yes_token_id"])
                            no_book_3 = fetch_clob_orderbook(w.market["no_token_id"])
                            w.min3_yes_bid = yes_book_3["best_bid"]
                            w.min3_yes_ask = yes_book_3["best_ask"]
                            w.min3_no_bid = no_book_3["best_bid"]
                            w.min3_no_ask = no_book_3["best_ask"]

                        log.info(
                            f"[MIN3] BTC {w.btc_direction} ({btc_return*100:+.3f}%), "
                            f"cont_prob={cont_prob:.3f}, "
                            f"YES bid/ask=${w.min3_yes_bid:.3f}/${w.min3_yes_ask:.3f}"
                        )

                        # Hybrid B decision
                        if w.hb_entered:
                            w.hb_action = decide_hybrid_b(btc_return, cont_prob, CONT_THRESHOLD)
                            if w.hb_action.startswith("SELL"):
                                w.hb_exit_price = w.min3_yes_bid
                                exit_fee = polymarket_fee(w.hb_shares * w.hb_exit_price, w.hb_exit_price)
                                w.hb_fee += exit_fee
                            log.info(f"[HYBRID B] Decision: {w.hb_action}")

                        # BuyBoth D decision
                        if w.bb_entered:
                            w.bb_action = decide_buyboth_d(btc_return, cont_prob, CONT_THRESHOLD)
                            if w.bb_action == "KEEP_YES":
                                w.bb_no_exit = w.min3_no_bid
                                exit_fee = polymarket_fee(w.bb_no_shares * w.bb_no_exit, w.bb_no_exit)
                                w.bb_fee += exit_fee
                            elif w.bb_action == "KEEP_NO":
                                w.bb_yes_exit = w.min3_yes_bid
                                exit_fee = polymarket_fee(w.bb_yes_shares * w.bb_yes_exit, w.bb_yes_exit)
                                w.bb_fee += exit_fee
                            elif w.bb_action == "SELL_BOTH":
                                w.bb_yes_exit = w.min3_yes_bid
                                w.bb_no_exit = w.min3_no_bid
                                fee_y = polymarket_fee(w.bb_yes_shares * w.bb_yes_exit, w.bb_yes_exit)
                                fee_n = polymarket_fee(w.bb_no_shares * w.bb_no_exit, w.bb_no_exit)
                                w.bb_fee += fee_y + fee_n
                            log.info(f"[BUYBOTH D] Decision: {w.bb_action}")

                    else:
                        log.warning(f"[MIN3] Feature computation failed — skipping decisions")
                        if w.hb_entered:
                            w.hb_action = "HOLD"  # Default to hold on feature failure
                        if w.bb_entered:
                            w.bb_action = "SELL_BOTH"  # Default to safe exit

                    w.decision_done = True

            # --- Settlement at minute 14+ (seconds 840+) ---
            if elapsed >= 840:
                w = active_windows.get(window_start)
                if w and w.decision_done and not w.settlement_done and (w.hb_entered or w.bb_entered):
                    # Determine outcome from BTC price
                    window_dt = datetime.fromtimestamp(window_start, tz=timezone.utc)
                    window_end_dt = window_dt + timedelta(minutes=15)
                    window_candles = [
                        c for c in spot_candles
                        if window_dt <= c.timestamp < window_end_dt
                    ]

                    if len(window_candles) >= 14:
                        btc_up = window_candles[-1].close > window_candles[0].open
                        resolution = "Up" if btc_up else "Down"
                        yes_settles = 1.0 if btc_up else 0.0
                        no_settles = 1.0 - yes_settles

                        log.info(f"[SETTLE] Window {window_start} → {resolution}")

                        record: dict[str, Any] = {
                            "window_ts": datetime.fromtimestamp(window_start, tz=timezone.utc).isoformat(),
                            "window_ts_unix": window_start,
                            "market_id": w.market["market_id"] if w.market else "",
                            "resolution": resolution,
                            "btc_return_pct": round(w.btc_return * 100, 4),
                            "cont_prob": round(w.cont_prob, 4),
                            "fillable": w.fillable,
                            "pre_window": {
                                "yes_ask": w.hb_entry_price if w.hb_entered else 0,
                                "no_ask": w.bb_no_entry if w.bb_entered else 0,
                            },
                            "minute_3": {
                                "yes_bid": w.min3_yes_bid,
                                "yes_ask": w.min3_yes_ask,
                                "no_bid": w.min3_no_bid,
                                "no_ask": w.min3_no_ask,
                            },
                        }

                        # Hybrid B settlement
                        if w.hb_entered:
                            if w.hb_action == "HOLD":
                                exit_price = yes_settles
                                if yes_settles > 0:
                                    w.hb_fee += polymarket_fee(
                                        w.hb_shares * yes_settles, yes_settles
                                    )
                            else:
                                exit_price = w.hb_exit_price  # Already sold at min 3

                            hb_pnl_gross = w.hb_shares * (exit_price - w.hb_entry_price)
                            hb_pnl_net = hb_pnl_gross - w.hb_fee
                            hybrid_bankroll += hb_pnl_net

                            hb_rec = {
                                "entered": True,
                                "action": w.hb_action,
                                "entry_price": round(w.hb_entry_price, 4),
                                "exit_price": round(exit_price, 4),
                                "shares": round(w.hb_shares, 2),
                                "pnl_gross": round(hb_pnl_gross, 2),
                                "fee": round(w.hb_fee, 2),
                                "pnl_net": round(hb_pnl_net, 2),
                                "bankroll_after": round(hybrid_bankroll, 2),
                            }
                            record["hybrid_b"] = hb_rec
                            hybrid_trades.append(hb_rec)
                            log.info(
                                f"[HYBRID B] {w.hb_action} → exit=${exit_price:.3f}, "
                                f"PnL=${hb_pnl_net:+.2f}, bankroll=${hybrid_bankroll:.2f}"
                            )

                        # BuyBoth D settlement
                        if w.bb_entered:
                            if w.bb_action == "KEEP_YES":
                                yes_exit = yes_settles
                                no_exit = w.bb_no_exit  # Sold at min 3
                                if yes_settles > 0:
                                    w.bb_fee += polymarket_fee(
                                        w.bb_yes_shares * yes_settles, yes_settles
                                    )
                            elif w.bb_action == "KEEP_NO":
                                yes_exit = w.bb_yes_exit  # Sold at min 3
                                no_exit = no_settles
                                if no_settles > 0:
                                    w.bb_fee += polymarket_fee(
                                        w.bb_no_shares * no_settles, no_settles
                                    )
                            else:  # SELL_BOTH
                                yes_exit = w.bb_yes_exit
                                no_exit = w.bb_no_exit

                            bb_pnl_yes = w.bb_yes_shares * (yes_exit - w.bb_yes_entry)
                            bb_pnl_no = w.bb_no_shares * (no_exit - w.bb_no_entry)
                            bb_pnl_gross = bb_pnl_yes + bb_pnl_no
                            bb_pnl_net = bb_pnl_gross - w.bb_fee
                            buyboth_bankroll += bb_pnl_net

                            bb_rec = {
                                "entered": True,
                                "action": w.bb_action,
                                "yes_entry": round(w.bb_yes_entry, 4),
                                "no_entry": round(w.bb_no_entry, 4),
                                "yes_exit": round(yes_exit, 4),
                                "no_exit": round(no_exit, 4),
                                "pnl_gross": round(bb_pnl_gross, 2),
                                "fee": round(w.bb_fee, 2),
                                "pnl_net": round(bb_pnl_net, 2),
                                "bankroll_after": round(buyboth_bankroll, 2),
                            }
                            record["buyboth_d"] = bb_rec
                            buyboth_trades.append(bb_rec)
                            log.info(
                                f"[BUYBOTH D] {w.bb_action} → "
                                f"PnL=${bb_pnl_net:+.2f}, bankroll=${buyboth_bankroll:.2f}"
                            )

                        # Log trade
                        log_trade(record)
                        update_summary(hybrid_bankroll, buyboth_bankroll,
                                       hybrid_trades, buyboth_trades)
                        w.settlement_done = True
                    else:
                        log.debug(f"[SETTLE] Only {len(window_candles)} candles — waiting")

            # --- Cleanup old windows ---
            stale = [ts for ts in active_windows if ts < window_start - 1800]
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
