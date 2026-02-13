#!/usr/bin/env python3
"""BWO Dashboard Data Server — FastAPI app serving paper-trader trade data.

Reads JSONL trade files, summary.json, and console logs from the paper trader.
Serves REST API on port 8100 with 5-second in-memory cache.

Usage:
    uvicorn scripts.bwo_data_server:app --host 0.0.0.0 --port 8100
    # or
    python scripts/bwo_data_server.py
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LOG_DIR = Path(os.environ.get("BWO_LOG_DIR", "./logs"))
TRADES_JSONL = LOG_DIR / "bwo_5m_paper_trades.jsonl"
SUMMARY_JSON = LOG_DIR / "bwo_5m_paper_summary.json"
CONSOLE_LOG = LOG_DIR / "5m_console.log"

INITIAL_BANKROLL = 200.0
WINDOWS_PER_DAY = 288  # 24*60/5 = 288 five-minute windows

# ---------------------------------------------------------------------------
# In-memory cache (5-second TTL)
# ---------------------------------------------------------------------------
_cache: dict[str, Any] = {
    "records": [],
    "mtime": 0.0,
    "loaded_at": 0.0,
}

CACHE_TTL = 5.0


def _load_records() -> list[dict[str, Any]]:
    """Load all JSONL records with 5-second cache."""
    now = time.time()
    try:
        mtime = TRADES_JSONL.stat().st_mtime if TRADES_JSONL.exists() else 0.0
    except OSError:
        mtime = 0.0

    if (now - _cache["loaded_at"] < CACHE_TTL) and (mtime == _cache["mtime"]):
        return _cache["records"]

    records: list[dict[str, Any]] = []
    if TRADES_JSONL.exists():
        with open(TRADES_JSONL) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    _cache["records"] = records
    _cache["mtime"] = mtime
    _cache["loaded_at"] = now
    return records


def _entered_trades(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter to only entered trades."""
    return [r for r in records if r.get("entered", False)]


def _load_summary() -> dict[str, Any]:
    """Load summary.json."""
    if SUMMARY_JSON.exists():
        try:
            with open(SUMMARY_JSON) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="BWO Dashboard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/summary")
def get_summary() -> dict[str, Any]:
    summary = _load_summary()
    records = _load_records()
    entered = _entered_trades(records)
    all_count = len(records)
    entered_count = len(entered)
    skipped_count = all_count - entered_count

    result: dict[str, Any] = {
        "bankroll": summary.get("bankroll", INITIAL_BANKROLL),
        "initial_bankroll": INITIAL_BANKROLL,
        "total_pnl": summary.get("total_pnl", 0.0),
        "total_pnl_pct": 0.0,
        "win_rate": summary.get("win_rate", 0.0),
        "wins": summary.get("wins", 0),
        "losses": summary.get("losses", 0),
        "total_trades": summary.get("total_trades", 0),
        "total_windows": all_count,
        "total_entries": entered_count,
        "total_skips": skipped_count,
        "skip_rate": round(skipped_count / all_count, 4) if all_count > 0 else 0.0,
        "avg_win": summary.get("avg_win", 0.0),
        "avg_loss": summary.get("avg_loss", 0.0),
        "ev_per_trade": summary.get("ev_per_trade", 0.0),
        "strategy": summary.get("strategy", "5m_continuation_min2"),
        "updated_at": summary.get("updated_at", ""),
    }
    bankroll = result["bankroll"]
    if INITIAL_BANKROLL > 0:
        result["total_pnl_pct"] = round((bankroll - INITIAL_BANKROLL) / INITIAL_BANKROLL * 100, 2)
    return result


@app.get("/api/trades")
def get_trades(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    records = _load_records()
    entered = _entered_trades(records)
    # Newest first
    entered.reverse()
    total = len(entered)
    page = entered[offset : offset + limit]
    return {
        "trades": page,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@app.get("/api/equity-curve")
def get_equity_curve() -> list[dict[str, Any]]:
    records = _load_records()
    entered = _entered_trades(records)
    curve: list[dict[str, Any]] = []
    # Add initial point
    if entered:
        first_ts = entered[0].get("window_ts", "")
        curve.append({
            "time": first_ts,
            "bankroll": INITIAL_BANKROLL,
            "cumulative_pnl": 0.0,
        })
    cum_pnl = 0.0
    for t in entered:
        cum_pnl += t.get("pnl_net", 0.0)
        curve.append({
            "time": t.get("window_ts", ""),
            "bankroll": t.get("bankroll_after", INITIAL_BANKROLL),
            "cumulative_pnl": round(cum_pnl, 2),
        })
    return curve


@app.get("/api/daily-pnl")
def get_daily_pnl() -> list[dict[str, Any]]:
    records = _load_records()
    entered = _entered_trades(records)
    days: dict[str, dict[str, Any]] = {}
    for t in entered:
        ts_str = t.get("window_ts", "")
        if not ts_str:
            continue
        try:
            dt = datetime.fromisoformat(ts_str)
            day = dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
        if day not in days:
            days[day] = {"date": day, "pnl": 0.0, "trades": 0, "wins": 0, "losses": 0}
        d = days[day]
        pnl = t.get("pnl_net", 0.0)
        d["pnl"] = round(d["pnl"] + pnl, 2)
        d["trades"] += 1
        if t.get("correct", False):
            d["wins"] += 1
        else:
            d["losses"] += 1
    result = sorted(days.values(), key=lambda x: x["date"])
    return result


@app.get("/api/skip-reasons")
def get_skip_reasons() -> dict[str, Any]:
    records = _load_records()
    all_count = len(records)
    entered = [r for r in records if r.get("entered", False)]
    skipped = [r for r in records if not r.get("entered", False)]
    entered_count = len(entered)
    skipped_count = len(skipped)

    # Categorize reasons
    reason_counts: dict[str, int] = defaultdict(int)
    for r in skipped:
        raw = r.get("skip_reason", "unknown")
        if not raw:
            raw = "unknown"
        # Normalize: extract category prefix
        reason_lower = raw.lower()
        if "noise" in reason_lower or "|ret|" in reason_lower:
            category = "noise (small BTC move)"
        elif "cont_prob" in reason_lower:
            category = "low continuation probability"
        elif "ev" in reason_lower or "edge" in reason_lower:
            category = "insufficient EV/edge"
        elif "price" in reason_lower and "<" in raw:
            category = "entry price too low"
        elif "price" in reason_lower and ">" in raw:
            category = "entry price too high"
        elif "depth" in reason_lower:
            category = "insufficient depth"
        elif "no market" in reason_lower or "could not find" in reason_lower:
            category = "no market found"
        else:
            category = raw[:60]
        reason_counts[category] += 1

    reasons = []
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        reasons.append({
            "reason": reason,
            "count": count,
            "pct": round(count / skipped_count * 100, 1) if skipped_count > 0 else 0.0,
        })

    return {
        "total_windows": all_count,
        "total_skips": skipped_count,
        "total_entries": entered_count,
        "skip_rate": round(skipped_count / all_count, 4) if all_count > 0 else 0.0,
        "reasons": reasons,
    }


@app.get("/api/metrics")
def get_metrics() -> dict[str, Any]:
    records = _load_records()
    entered = _entered_trades(records)

    if not entered:
        return {
            "total_trades": 0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "max_consec_losses": 0,
            "max_consec_wins": 0,
            "total_fees": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
            "expectancy": 0.0,
            "avg_trades_per_day": 0.0,
        }

    pnls = [t.get("pnl_net", 0.0) for t in entered]
    total_trades = len(pnls)

    # Sharpe: mean(pnl) / std(pnl) * sqrt(288)
    mean_pnl = sum(pnls) / total_trades
    variance = sum((p - mean_pnl) ** 2 for p in pnls) / total_trades
    std_pnl = math.sqrt(variance) if variance > 0 else 0.0
    sharpe = (mean_pnl / std_pnl * math.sqrt(WINDOWS_PER_DAY)) if std_pnl > 0 else 0.0

    # Sortino: mean(pnl) / downside_std * sqrt(288)
    down_sq = [((p - mean_pnl) ** 2) for p in pnls if p < mean_pnl]
    downside_var = sum(down_sq) / total_trades if down_sq else 0.0
    downside_std = math.sqrt(downside_var) if downside_var > 0 else 0.0
    sortino = (mean_pnl / downside_std * math.sqrt(WINDOWS_PER_DAY)) if downside_std > 0 else 0.0

    # Profit factor
    gross_win = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0

    # Max drawdown (from equity curve)
    equity = [INITIAL_BANKROLL]
    for t in entered:
        equity.append(t.get("bankroll_after", equity[-1]))
    peak = equity[0]
    max_dd = 0.0
    max_dd_pct = 0.0
    for e in equity:
        if e > peak:
            peak = e
        dd = peak - e
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = (dd / peak * 100) if peak > 0 else 0.0

    # Max consecutive wins/losses
    max_consec_losses = 0
    max_consec_wins = 0
    cur_losses = 0
    cur_wins = 0
    for t in entered:
        if t.get("correct", False):
            cur_wins += 1
            cur_losses = 0
        else:
            cur_losses += 1
            cur_wins = 0
        max_consec_losses = max(max_consec_losses, cur_losses)
        max_consec_wins = max(max_consec_wins, cur_wins)

    # Total fees
    total_fees = sum(t.get("fee", 0.0) for t in entered)

    # Best/worst trade
    best_trade = max(pnls)
    worst_trade = min(pnls)

    # Expectancy = mean PnL
    expectancy = mean_pnl

    # Avg trades per day
    dates: set[str] = set()
    for t in entered:
        ts_str = t.get("window_ts", "")
        if ts_str:
            try:
                dt = datetime.fromisoformat(ts_str)
                dates.add(dt.strftime("%Y-%m-%d"))
            except (ValueError, TypeError):
                pass
    num_days = len(dates) if dates else 1
    avg_trades_per_day = total_trades / num_days

    return {
        "total_trades": total_trades,
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "profit_factor": round(profit_factor, 3) if profit_factor != float("inf") else None,
        "max_drawdown": round(max_dd, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "max_consec_losses": max_consec_losses,
        "max_consec_wins": max_consec_wins,
        "total_fees": round(total_fees, 2),
        "best_trade": round(best_trade, 2),
        "worst_trade": round(worst_trade, 2),
        "expectancy": round(expectancy, 2),
        "avg_trades_per_day": round(avg_trades_per_day, 1),
    }


@app.get("/api/confidence-distribution")
def get_confidence_distribution() -> list[dict[str, Any]]:
    records = _load_records()
    entered = _entered_trades(records)

    # Buckets: 0.75-0.77, 0.77-0.79, ... up to 0.99-1.00 (2% wide)
    buckets: dict[str, dict[str, Any]] = {}
    for t in entered:
        cp = t.get("cont_prob", 0.0)
        # Bucket lower bound: floor to nearest 0.02
        lower = math.floor(cp / 0.02) * 0.02
        upper = lower + 0.02
        label = f"{lower:.2f}-{upper:.2f}"
        if label not in buckets:
            buckets[label] = {"bucket": label, "lower": lower, "total": 0, "wins": 0, "pnl_sum": 0.0}
        b = buckets[label]
        b["total"] += 1
        if t.get("correct", False):
            b["wins"] += 1
        b["pnl_sum"] += t.get("pnl_net", 0.0)

    result = []
    for b in sorted(buckets.values(), key=lambda x: x["lower"]):
        total = b["total"]
        result.append({
            "bucket": b["bucket"],
            "total": total,
            "wins": b["wins"],
            "win_rate": round(b["wins"] / total, 4) if total > 0 else 0.0,
            "avg_pnl": round(b["pnl_sum"] / total, 2) if total > 0 else 0.0,
        })
    return result


@app.get("/api/entry-price-analysis")
def get_entry_price_analysis() -> list[dict[str, Any]]:
    records = _load_records()
    entered = _entered_trades(records)

    # Buckets: $0.05 wide
    buckets: dict[str, dict[str, Any]] = {}
    for t in entered:
        price = t.get("entry_price", 0.0)
        lower = math.floor(price / 0.05) * 0.05
        upper = lower + 0.05
        label = f"${lower:.2f}-${upper:.2f}"
        if label not in buckets:
            buckets[label] = {"bucket": label, "lower": lower, "total": 0, "wins": 0, "pnl_sum": 0.0}
        b = buckets[label]
        b["total"] += 1
        if t.get("correct", False):
            b["wins"] += 1
        b["pnl_sum"] += t.get("pnl_net", 0.0)

    result = []
    for b in sorted(buckets.values(), key=lambda x: x["lower"]):
        total = b["total"]
        result.append({
            "bucket": b["bucket"],
            "total": total,
            "wins": b["wins"],
            "win_rate": round(b["wins"] / total, 4) if total > 0 else 0.0,
            "avg_pnl": round(b["pnl_sum"] / total, 2) if total > 0 else 0.0,
        })
    return result


@app.get("/api/rolling-win-rate")
def get_rolling_win_rate(
    window: int = Query(20, ge=2, le=200),
) -> list[dict[str, Any]]:
    records = _load_records()
    entered = _entered_trades(records)

    if len(entered) < window:
        return []

    result = []
    for i in range(window, len(entered) + 1):
        chunk = entered[i - window : i]
        wins = sum(1 for t in chunk if t.get("correct", False))
        result.append({
            "index": i,
            "win_rate": round(wins / window, 4),
            "time": chunk[-1].get("window_ts", ""),
        })
    return result


@app.get("/api/health")
def get_health() -> dict[str, Any]:
    records = _load_records()

    # JSONL age
    try:
        mtime = TRADES_JSONL.stat().st_mtime if TRADES_JSONL.exists() else 0.0
    except OSError:
        mtime = 0.0
    jsonl_age = time.time() - mtime if mtime > 0 else -1

    # Paper trader "active" heuristic: JSONL modified within last 10 minutes
    paper_trader_active = 0 < jsonl_age < 600

    last_modified = ""
    if mtime > 0:
        last_modified = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()

    return {
        "server_ok": True,
        "paper_trader_active": paper_trader_active,
        "jsonl_age_s": round(jsonl_age, 1),
        "total_records": len(records),
        "last_modified": last_modified,
    }


@app.get("/api/console-log")
def get_console_log(
    lines: int = Query(100, ge=1, le=5000),
) -> dict[str, Any]:
    if not CONSOLE_LOG.exists():
        return {"lines": [], "total_lines": 0}

    try:
        with open(CONSOLE_LOG, "rb") as f:
            # Read from end for efficiency on large files
            f.seek(0, 2)
            file_size = f.tell()
            # Read at most 1MB from the end
            read_size = min(file_size, 1_000_000)
            f.seek(max(0, file_size - read_size))
            content = f.read().decode("utf-8", errors="replace")

        all_lines = content.splitlines()
        tail = all_lines[-lines:] if len(all_lines) > lines else all_lines
        return {"lines": tail, "total_lines": len(all_lines)}
    except OSError:
        return {"lines": [], "total_lines": 0}


@app.get("/api/export/trades")
def export_trades(
    format: str = Query("csv", pattern="^(csv|json)$"),
) -> Any:
    records = _load_records()
    entered = _entered_trades(records)

    if format == "json":
        return StreamingResponse(
            io.BytesIO(json.dumps(entered, indent=2, default=str).encode()),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=bwo_trades.json"},
        )

    # CSV export
    if not entered:
        output = io.StringIO("No trades\n")
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=bwo_trades.csv"},
        )

    fieldnames = [
        "window_ts", "window_ts_unix", "market_id", "resolution",
        "btc_return_pct", "cont_prob", "early_direction", "btc_direction",
        "side", "entry_price", "settlement", "shares",
        "pnl_gross", "fee", "pnl_net", "bankroll_after", "correct",
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for t in entered:
        writer.writerow(t)

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=bwo_trades.csv"},
    )


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "scripts.bwo_data_server:app",
        host="0.0.0.0",
        port=8100,
        reload=True,
    )
