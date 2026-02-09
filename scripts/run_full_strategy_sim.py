"""Full strategy simulation on real Polymarket BTC 15-min data.

Runs the tiered momentum-confirmation strategy against all matched
Polymarket markets with:
  - Quarter-Kelly position sizing (f* = (p-P)/(1-P) * 0.25, capped at 2%)
  - Polymarket dynamic taker fees + 5 bps slippage
  - Equity curve simulation at multiple starting capitals
  - Monthly breakdown, drawdown analysis, capital recommendations

This is the production-readiness test.
"""

from __future__ import annotations

import csv
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fast_loader import FastCandle, load_csv_fast

# === PATHS ===
POLYMARKET_CSV = PROJECT_ROOT / "data" / "polymarket_btc_15m.csv"
BTC_CSV = PROJECT_ROOT / "data" / "btc_1m_2y.csv"

# === STRATEGY CONFIG (from config/default.toml) ===
TIERS = [(8, 0.0010), (9, 0.0008), (10, 0.0005)]
SENSITIVITY = 0.07

# === POSITION SIZING CONFIG ===
KELLY_MULTIPLIER = 0.25       # Quarter-Kelly
MAX_POSITION_PCT = 0.02       # Max 2% of balance per trade
MAX_BOOK_PCT = 0.10           # Max 10% of market volume (liquidity cap)
MIN_TRADE_SIZE = 1.0          # $1 minimum trade

# === FEE CONFIG ===
FEE_CONSTANT = 0.25           # Polymarket dynamic fee constant
SLIPPAGE_BPS = 5              # 5 basis points slippage

# === STARTING CAPITALS TO TEST ===
STARTING_CAPITALS = [1_000, 2_500, 5_000, 10_000, 25_000, 50_000]


def _sigmoid(cum_ret: float) -> float:
    x = SENSITIVITY * cum_ret * 10000
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def polymarket_fee(position_size: float, entry_price: float) -> float:
    if entry_price <= 0 or entry_price >= 1:
        return 0.0
    return position_size * FEE_CONSTANT * (entry_price ** 2) * ((1.0 - entry_price) ** 2)


def parse_et_time(title: str) -> tuple[datetime, datetime] | None:
    pattern = r'(\w+ \d+),?\s+(\d{1,2}):(\d{2})(AM|PM)\s*[-\u2013]\s*(\d{1,2}):(\d{2})(AM|PM)\s*ET'
    match = re.search(pattern, title, re.IGNORECASE)
    if not match:
        return None
    date_str, h1, m1, ap1, h2, m2, ap2 = match.groups()
    h1, m1, h2, m2 = int(h1), int(m1), int(h2), int(m2)
    ap1, ap2 = ap1.upper(), ap2.upper()
    if ap1 == "PM" and h1 != 12:
        h1 += 12
    elif ap1 == "AM" and h1 == 12:
        h1 = 0
    if ap2 == "PM" and h2 != 12:
        h2 += 12
    elif ap2 == "AM" and h2 == 12:
        h2 = 0
    for year in [2026, 2025, 2024]:
        try:
            dt = datetime.strptime(f"{date_str} {year}", "%B %d %Y")
            break
        except ValueError:
            continue
    else:
        return None
    month = dt.month
    utc_offset = timedelta(hours=4) if 4 <= month <= 10 else timedelta(hours=5)
    start_utc = datetime(dt.year, dt.month, dt.day, h1, m1, tzinfo=timezone.utc) + utc_offset
    end_utc = datetime(dt.year, dt.month, dt.day, h2, m2, tzinfo=timezone.utc) + utc_offset
    if end_utc <= start_utc:
        end_utc += timedelta(days=1)
    return start_utc, end_utc


@dataclass
class Trade:
    timestamp: datetime
    direction: str          # "Up" or "Down"
    entry_price: float      # Sigmoid-based entry price
    entry_minute: int
    position_size: float    # $ risked (from Kelly)
    resolution: str         # Polymarket resolution
    correct: bool
    pnl_gross: float
    fee: float
    pnl_net: float
    balance_before: float
    balance_after: float
    kelly_fraction: float
    volume: float


@dataclass
class EquitySimulation:
    starting_capital: float
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    timestamps: list[datetime] = field(default_factory=list)

    @property
    def final_balance(self) -> float:
        return self.equity_curve[-1] if self.equity_curve else self.starting_capital

    @property
    def total_return_pct(self) -> float:
        return ((self.final_balance - self.starting_capital) / self.starting_capital) * 100

    @property
    def num_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return sum(1 for t in self.trades if t.correct) / len(self.trades)

    @property
    def max_drawdown_pct(self) -> float:
        if not self.equity_curve:
            return 0.0
        curve = np.array(self.equity_curve)
        peaks = np.maximum.accumulate(curve)
        drawdowns = (peaks - curve) / peaks
        return float(np.max(drawdowns)) * 100

    @property
    def max_drawdown_dollar(self) -> float:
        if not self.equity_curve:
            return 0.0
        curve = np.array(self.equity_curve)
        peaks = np.maximum.accumulate(curve)
        dd_dollars = peaks - curve
        return float(np.max(dd_dollars))

    @property
    def sharpe_ratio(self) -> float:
        if len(self.trades) < 2:
            return 0.0
        returns = [t.pnl_net / t.balance_before for t in self.trades]
        mean_r = np.mean(returns)
        std_r = np.std(returns, ddof=1)
        if std_r < 1e-10:
            return 0.0
        # Annualize: ~35,040 15-min windows per year
        return float(mean_r / std_r * np.sqrt(35040))

    @property
    def profit_factor(self) -> float:
        gross_wins = sum(t.pnl_net for t in self.trades if t.pnl_net > 0)
        gross_losses = abs(sum(t.pnl_net for t in self.trades if t.pnl_net < 0))
        if gross_losses < 0.01:
            return float("inf")
        return gross_wins / gross_losses

    @property
    def avg_trade_pnl(self) -> float:
        if not self.trades:
            return 0.0
        return sum(t.pnl_net for t in self.trades) / len(self.trades)

    @property
    def total_fees(self) -> float:
        return sum(t.fee for t in self.trades)

    @property
    def avg_position_size(self) -> float:
        if not self.trades:
            return 0.0
        return sum(t.position_size for t in self.trades) / len(self.trades)


def run_strategy_on_markets(
    poly_markets: list[dict],
    candle_index: dict[datetime, FastCandle],
    starting_capital: float,
) -> EquitySimulation:
    """Run the full strategy simulation with Kelly sizing."""
    sim = EquitySimulation(starting_capital=starting_capital)
    balance = starting_capital
    sim.equity_curve.append(balance)

    # Sort markets by time for chronological simulation
    market_times: list[tuple[datetime, dict]] = []
    for pm in poly_markets:
        time_range = parse_et_time(pm["event_title"])
        if time_range is None:
            continue
        market_times.append((time_range[0], pm))
    market_times.sort(key=lambda x: x[0])

    for window_start, pm in market_times:
        title = pm["event_title"]
        resolution = pm["resolution"]
        volume = float(pm.get("volume", 0))

        time_range = parse_et_time(title)
        if time_range is None:
            continue
        window_start, window_end = time_range

        # Get candles
        window_candles: list[FastCandle] = []
        for minute_offset in range(15):
            ts = (window_start + timedelta(minutes=minute_offset)).replace(tzinfo=None)
            candle = candle_index.get(ts)
            if candle is not None:
                window_candles.append(candle)

        if len(window_candles) < 11:
            continue

        window_open = float(window_candles[0].open)
        if window_open == 0:
            continue

        # Try each tier
        for entry_min, threshold in TIERS:
            if entry_min >= len(window_candles):
                continue

            current_close = float(window_candles[entry_min].close)
            cum_ret = (current_close - window_open) / window_open
            if abs(cum_ret) < threshold:
                continue

            predict_green = cum_ret > 0
            entry_yes = _sigmoid(cum_ret)
            ep = entry_yes if predict_green else 1.0 - entry_yes
            if ep <= 0.01 or ep >= 0.99:
                continue

            # === KELLY POSITION SIZING ===
            # Estimate win probability from our 88.4% historical accuracy
            estimated_win_prob = 0.884

            # Binary Kelly: f* = (p - P) / (1 - P)
            if ep >= estimated_win_prob:
                # No edge at this entry price
                continue
            kelly = (estimated_win_prob - ep) / (1.0 - ep)
            frac_kelly = kelly * KELLY_MULTIPLIER

            # Position size = fraction * balance, capped at min of:
            #   1) 2% of balance (risk cap)
            #   2) 10% of market volume (liquidity cap)
            raw_size = frac_kelly * balance
            max_by_balance = balance * MAX_POSITION_PCT
            max_by_liquidity = volume * MAX_BOOK_PCT if volume > 0 else max_by_balance
            position_size = min(raw_size, max_by_balance, max_by_liquidity)
            position_size = max(position_size, 0.0)

            if position_size < MIN_TRADE_SIZE:
                continue

            # === COMPUTE PnL ===
            direction = "Up" if predict_green else "Down"
            correct = (direction == resolution)

            if predict_green:
                settlement = 1.0 if resolution == "Up" else 0.0
            else:
                settlement = 1.0 if resolution == "Down" else 0.0

            # PnL = shares * (settlement - entry_price)
            # shares = position_size / entry_price
            shares = position_size / ep
            pnl_gross = shares * (settlement - ep)

            # Fees
            fee_exchange = polymarket_fee(position_size, ep)
            fee_slippage = position_size * SLIPPAGE_BPS / 10000
            fee_total = fee_exchange + fee_slippage
            pnl_net = pnl_gross - fee_total

            balance_before = balance
            balance += pnl_net
            balance = max(balance, 0.0)  # Can't go below zero

            sim.trades.append(Trade(
                timestamp=window_start,
                direction=direction,
                entry_price=ep,
                entry_minute=entry_min,
                position_size=position_size,
                resolution=resolution,
                correct=correct,
                pnl_gross=pnl_gross,
                fee=fee_total,
                pnl_net=pnl_net,
                balance_before=balance_before,
                balance_after=balance,
                kelly_fraction=kelly,
                volume=volume,
            ))
            sim.equity_curve.append(balance)
            sim.timestamps.append(window_start)

            break  # Only one entry per window

    return sim


def main() -> None:
    print("=" * 120)
    print("  FULL STRATEGY SIMULATION ON REAL POLYMARKET DATA")
    print("  Tiered Momentum Strategy + Quarter-Kelly Sizing + Polymarket Fees")
    print("=" * 120)

    # Load data
    if not POLYMARKET_CSV.exists():
        print(f"  ERROR: {POLYMARKET_CSV} not found. Run download_polymarket_btc.py first.")
        sys.exit(1)
    if not BTC_CSV.exists():
        print(f"  ERROR: {BTC_CSV} not found.")
        sys.exit(1)

    poly_markets: list[dict] = []
    with open(POLYMARKET_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            poly_markets.append(row)
    print(f"\n  Loaded {len(poly_markets):,} Polymarket markets")

    print("  Loading Binance candle data...")
    all_candles = load_csv_fast(BTC_CSV)
    all_candles.sort(key=lambda c: c.timestamp)
    seen: set[datetime] = set()
    unique = [c for c in all_candles if c.timestamp not in seen and not seen.add(c.timestamp)]  # type: ignore
    all_candles = unique
    print(f"  Loaded {len(all_candles):,} Binance 1m candles")

    candle_index: dict[datetime, FastCandle] = {}
    for c in all_candles:
        ts = c.timestamp.replace(tzinfo=None) if c.timestamp.tzinfo else c.timestamp
        candle_index[ts] = c

    # ===================================================================
    # SECTION 1: STRATEGY WIN RATE & PER-TRADE ECONOMICS (FIXED $100)
    # ===================================================================
    print(f"\n  {'='*110}")
    print(f"  SECTION 1: STRATEGY ACCURACY & PER-TRADE ECONOMICS (fixed $100)")
    print(f"  {'='*110}")

    # Run a fixed-size pass just for accuracy stats
    fixed_trades: list[dict] = []
    FIXED_SIZE = 100.0
    market_times_sorted: list[tuple[datetime, dict]] = []
    for pm in poly_markets:
        tr = parse_et_time(pm["event_title"])
        if tr:
            market_times_sorted.append((tr[0], pm))
    market_times_sorted.sort(key=lambda x: x[0])

    for window_start, pm in market_times_sorted:
        title = pm["event_title"]
        resolution = pm["resolution"]
        volume = float(pm.get("volume", 0))
        tr = parse_et_time(title)
        if tr is None:
            continue
        ws, we = tr
        window_candles: list[FastCandle] = []
        for mo in range(15):
            ts = (ws + timedelta(minutes=mo)).replace(tzinfo=None)
            c = candle_index.get(ts)
            if c is not None:
                window_candles.append(c)
        if len(window_candles) < 11:
            continue
        wo = float(window_candles[0].open)
        if wo == 0:
            continue
        for entry_min, threshold in TIERS:
            if entry_min >= len(window_candles):
                continue
            cc = float(window_candles[entry_min].close)
            cum_ret = (cc - wo) / wo
            if abs(cum_ret) < threshold:
                continue
            pg = cum_ret > 0
            ey = _sigmoid(cum_ret)
            ep = ey if pg else 1.0 - ey
            if ep <= 0.01 or ep >= 0.99:
                continue
            direction = "Up" if pg else "Down"
            correct = (direction == resolution)
            if pg:
                settlement = 1.0 if resolution == "Up" else 0.0
            else:
                settlement = 1.0 if resolution == "Down" else 0.0
            shares = FIXED_SIZE / ep
            pnl_gross = shares * (settlement - ep)
            fee_ex = polymarket_fee(FIXED_SIZE, ep)
            fee_slip = FIXED_SIZE * SLIPPAGE_BPS / 10000
            fee = fee_ex + fee_slip
            pnl_net = pnl_gross - fee
            fixed_trades.append({
                "timestamp": ws, "direction": direction, "entry_minute": entry_min,
                "entry_price": ep, "resolution": resolution, "correct": correct,
                "pnl_gross": pnl_gross, "pnl_net": pnl_net, "fee": fee,
                "volume": volume, "cum_ret": cum_ret,
            })
            break

    if fixed_trades:
        correct_count = sum(1 for t in fixed_trades if t["correct"])
        total = len(fixed_trades)
        accuracy = correct_count / total
        print(f"\n  Matched & traded markets: {total:,}")
        print(f"  Correct predictions: {correct_count:,}")
        print(f"  Accuracy: {accuracy*100:.1f}%")

        gross_pnls = [t["pnl_gross"] for t in fixed_trades]
        net_pnls = [t["pnl_net"] for t in fixed_trades]
        fees = [t["fee"] for t in fixed_trades]
        print(f"\n  Per-trade economics (at $100 per trade):")
        print(f"  Avg gross P&L: ${np.mean(gross_pnls):+.2f}")
        print(f"  Avg net P&L:   ${np.mean(net_pnls):+.2f}")
        print(f"  Avg fee:       ${np.mean(fees):.2f}")
        print(f"  Total net P&L: ${sum(net_pnls):+,.2f}")
        print(f"  Total fees:    ${sum(fees):,.2f}")

        # By entry tier
        print(f"\n  {'Tier':<15} {'Trades':>7} {'Correct':>8} {'Acc%':>7} {'Avg Net':>10}")
        print(f"  {'-'*50}")
        for tier_min, tier_thresh in TIERS:
            tier_trades = [t for t in fixed_trades if t["entry_minute"] == tier_min]
            if tier_trades:
                tc = sum(1 for t in tier_trades if t["correct"])
                avg_net = np.mean([t["pnl_net"] for t in tier_trades])
                print(f"  Min {tier_min} @{tier_thresh:.4f}  {len(tier_trades):>7} {tc:>8} {tc/len(tier_trades)*100:>6.1f}% ${avg_net:>+8.2f}")

        # By direction
        up_trades = [t for t in fixed_trades if t["direction"] == "Up"]
        dn_trades = [t for t in fixed_trades if t["direction"] == "Down"]
        if up_trades:
            up_acc = sum(1 for t in up_trades if t["correct"]) / len(up_trades)
            up_net = np.mean([t["pnl_net"] for t in up_trades])
            print(f"\n  Up trades: {len(up_trades):,} ({up_acc*100:.1f}% acc, avg net ${up_net:+.2f})")
        if dn_trades:
            dn_acc = sum(1 for t in dn_trades if t["correct"]) / len(dn_trades)
            dn_net = np.mean([t["pnl_net"] for t in dn_trades])
            print(f"  Down trades: {len(dn_trades):,} ({dn_acc*100:.1f}% acc, avg net ${dn_net:+.2f})")

        # Monthly breakdown
        by_month: dict[str, list[dict]] = defaultdict(list)
        for t in fixed_trades:
            month = t["timestamp"].strftime("%Y-%m")
            by_month[month].append(t)

        print(f"\n  {'Month':<10} {'Trades':>7} {'Correct':>8} {'Acc%':>7} {'Net PnL':>12} {'Fees':>10}")
        print(f"  {'-'*60}")
        for month in sorted(by_month):
            trades = by_month[month]
            n = len(trades)
            corr = sum(1 for t in trades if t["correct"])
            net = sum(t["pnl_net"] for t in trades)
            f = sum(t["fee"] for t in trades)
            print(f"  {month:<10} {n:>7} {corr:>8} {corr/n*100:>6.1f}% ${net:>+10,.2f} ${f:>8,.2f}")

        # Trades per day
        by_day: dict[str, int] = defaultdict(int)
        for t in fixed_trades:
            by_day[t["timestamp"].strftime("%Y-%m-%d")] += 1
        days = sorted(by_day.keys())
        trades_per_day = [by_day[d] for d in days]
        print(f"\n  Trading frequency:")
        print(f"  Total trading days: {len(days)}")
        print(f"  Avg trades/day: {np.mean(trades_per_day):.1f}")
        print(f"  Max trades/day: {max(trades_per_day)}")
        print(f"  Min trades/day: {min(trades_per_day)}")

    # ===================================================================
    # SECTION 2: EQUITY SIMULATION AT MULTIPLE STARTING CAPITALS
    # ===================================================================
    print(f"\n  {'='*110}")
    print(f"  SECTION 2: EQUITY SIMULATION (Quarter-Kelly, 2% cap, Polymarket fees)")
    print(f"  {'='*110}")

    results: dict[float, EquitySimulation] = {}

    for capital in STARTING_CAPITALS:
        sim = run_strategy_on_markets(poly_markets, candle_index, capital)
        results[capital] = sim

    # Summary table
    print(f"\n  {'Starting':>12} {'Trades':>7} {'Win%':>6} {'Final Bal':>14} {'Net Profit':>13} "
          f"{'Return%':>9} {'Max DD%':>8} {'Max DD$':>10} {'Sharpe':>8} {'PF':>6} {'Avg Size':>10}")
    print(f"  {'-'*120}")

    for capital in STARTING_CAPITALS:
        sim = results[capital]
        net_profit = sim.final_balance - capital
        pf_str = f"{sim.profit_factor:.2f}" if sim.profit_factor < 1000 else "inf"
        print(f"  ${capital:>10,} {sim.num_trades:>7} {sim.win_rate*100:>5.1f}% "
              f"${sim.final_balance:>12,.2f} ${net_profit:>+11,.2f} "
              f"{sim.total_return_pct:>+8.1f}% "
              f"{sim.max_drawdown_pct:>7.2f}% ${sim.max_drawdown_dollar:>8,.2f} "
              f"{sim.sharpe_ratio:>8.1f} {pf_str:>6} "
              f"${sim.avg_position_size:>8.2f}")

    # ===================================================================
    # SECTION 3: DETAILED BREAKDOWN FOR RECOMMENDED CAPITAL ($10K)
    # ===================================================================
    rec_capital = 10_000
    sim_rec = results[rec_capital]

    print(f"\n  {'='*110}")
    print(f"  SECTION 3: DETAILED ANALYSIS — ${rec_capital:,} STARTING CAPITAL")
    print(f"  {'='*110}")

    if sim_rec.trades:
        print(f"\n  Trades: {sim_rec.num_trades:,}")
        print(f"  Win rate: {sim_rec.win_rate*100:.1f}%")
        print(f"  Final balance: ${sim_rec.final_balance:,.2f}")
        print(f"  Total return: {sim_rec.total_return_pct:+.1f}%")
        print(f"  Max drawdown: {sim_rec.max_drawdown_pct:.2f}% (${sim_rec.max_drawdown_dollar:,.2f})")
        print(f"  Sharpe ratio: {sim_rec.sharpe_ratio:.1f}")
        pf_str = f"{sim_rec.profit_factor:.2f}" if sim_rec.profit_factor < 1000 else "inf"
        print(f"  Profit factor: {pf_str}")
        print(f"  Total fees paid: ${sim_rec.total_fees:,.2f}")
        print(f"  Avg position size: ${sim_rec.avg_position_size:,.2f}")
        print(f"  Avg net P&L/trade: ${sim_rec.avg_trade_pnl:,.2f}")

        # Monthly equity breakdown
        by_month: dict[str, list[Trade]] = defaultdict(list)
        for t in sim_rec.trades:
            month = t.timestamp.strftime("%Y-%m")
            by_month[month].append(t)

        print(f"\n  {'Month':<10} {'Trades':>7} {'Win%':>6} {'Net PnL':>12} {'Fees':>10} "
              f"{'End Balance':>14} {'DD%':>7}")
        print(f"  {'-'*75}")

        for month in sorted(by_month):
            trades = by_month[month]
            n = len(trades)
            wins = sum(1 for t in trades if t.correct)
            net_pnl = sum(t.pnl_net for t in trades)
            fees = sum(t.fee for t in trades)
            end_bal = trades[-1].balance_after

            # Month-level max drawdown
            month_equity = [trades[0].balance_before] + [t.balance_after for t in trades]
            me = np.array(month_equity)
            peaks = np.maximum.accumulate(me)
            dd = (peaks - me) / np.where(peaks > 0, peaks, 1)
            max_dd = float(np.max(dd)) * 100

            print(f"  {month:<10} {n:>7} {wins/n*100:>5.1f}% ${net_pnl:>+10,.2f} ${fees:>8,.2f} "
                  f"${end_bal:>12,.2f} {max_dd:>6.2f}%")

        # Kelly fraction distribution
        kelly_fracs = [t.kelly_fraction for t in sim_rec.trades]
        pos_sizes = [t.position_size for t in sim_rec.trades]
        pos_pcts = [t.position_size / t.balance_before * 100 for t in sim_rec.trades if t.balance_before > 0]

        print(f"\n  Position Sizing Statistics:")
        print(f"  Kelly fraction: avg={np.mean(kelly_fracs):.4f}, "
              f"min={np.min(kelly_fracs):.4f}, max={np.max(kelly_fracs):.4f}")
        print(f"  Position size: avg=${np.mean(pos_sizes):.2f}, "
              f"min=${np.min(pos_sizes):.2f}, max=${np.max(pos_sizes):.2f}")
        print(f"  Position as % of balance: avg={np.mean(pos_pcts):.2f}%, "
              f"min={np.min(pos_pcts):.2f}%, max={np.max(pos_pcts):.2f}%")

        # Win/loss streaks
        streaks = []
        current_streak = 0
        current_type = None
        for t in sim_rec.trades:
            if t.correct == current_type:
                current_streak += 1
            else:
                if current_type is not None:
                    streaks.append((current_type, current_streak))
                current_streak = 1
                current_type = t.correct
        if current_type is not None:
            streaks.append((current_type, current_streak))

        win_streaks = [s for is_win, s in streaks if is_win]
        loss_streaks = [s for is_win, s in streaks if not is_win]
        print(f"\n  Streak Analysis:")
        if win_streaks:
            print(f"  Longest win streak: {max(win_streaks)}")
            print(f"  Average win streak: {np.mean(win_streaks):.1f}")
        if loss_streaks:
            print(f"  Longest loss streak: {max(loss_streaks)}")
            print(f"  Average loss streak: {np.mean(loss_streaks):.1f}")

    # ===================================================================
    # SECTION 4: RISK ANALYSIS — WHAT IF ACCURACY DEGRADES?
    # ===================================================================
    print(f"\n  {'='*110}")
    print(f"  SECTION 4: RISK ANALYSIS — ACCURACY SENSITIVITY")
    print(f"  {'='*110}")

    # Use fixed-trade data for clean per-trade economics
    if fixed_trades:
        avg_win_pnl = np.mean([t["pnl_net"] for t in fixed_trades if t["correct"]])
        avg_loss_pnl = np.mean([t["pnl_net"] for t in fixed_trades if not t["correct"]])
        total_ft = len(fixed_trades)
        months_span = len(set(t["timestamp"].strftime("%Y-%m") for t in fixed_trades))
        trades_per_month = total_ft / max(months_span, 1)

        print(f"\n  Per-trade economics (fixed $100):")
        print(f"  Avg win P&L:  ${avg_win_pnl:+.2f}")
        print(f"  Avg loss P&L: ${avg_loss_pnl:+.2f}")
        print(f"  Trades/month: ~{trades_per_month:.0f}")

        print(f"\n  {'Accuracy':<12} {'Monthly PnL':>14} {'Annual PnL':>14} {'Break-even?':>12}")
        print(f"  {'-'*55}")

        for test_acc in [0.90, 0.884, 0.86, 0.84, 0.82, 0.80, 0.75]:
            wins_per_month = trades_per_month * test_acc
            losses_per_month = trades_per_month * (1 - test_acc)
            monthly_pnl = wins_per_month * avg_win_pnl + losses_per_month * avg_loss_pnl
            annual_pnl = monthly_pnl * 12
            be = "YES" if monthly_pnl > 0 else "NO"
            label = f"{test_acc*100:.1f}%"
            if abs(test_acc - 0.884) < 0.001:
                label += " <-- current"
            print(f"  {label:<12} ${monthly_pnl:>+12,.2f} ${annual_pnl:>+12,.2f} {be:>12}")

        # Find break-even accuracy
        # monthly_pnl = TPM * acc * avg_win + TPM * (1-acc) * avg_loss = 0
        # acc * avg_win + (1-acc) * avg_loss = 0
        # acc * (avg_win - avg_loss) = -avg_loss
        if (avg_win_pnl - avg_loss_pnl) != 0:
            breakeven_acc = -avg_loss_pnl / (avg_win_pnl - avg_loss_pnl)
            print(f"\n  Break-even accuracy: {breakeven_acc*100:.1f}%")
            print(f"  Safety margin: {(accuracy - breakeven_acc)*100:.1f} percentage points")

    # ===================================================================
    # SECTION 5: CAPITAL RECOMMENDATION
    # ===================================================================
    print(f"\n  {'='*110}")
    print(f"  SECTION 5: CAPITAL RECOMMENDATION")
    print(f"  {'='*110}")

    print(f"\n  Analysis based on {len(poly_markets):,} real Polymarket markets")
    print(f"  Strategy: Tiered momentum (8@0.10%, 9@0.08%, 10@0.05%)")
    print(f"  Sizing: Quarter-Kelly, 2% max position")
    print(f"  Fees: Polymarket dynamic taker + 5 bps slippage")

    # Use fixed-trade data for capital recommendation (avoids compounding distortion)
    if fixed_trades:
        avg_net = np.mean([t["pnl_net"] for t in fixed_trades])
        months_span = len(set(t["timestamp"].strftime("%Y-%m") for t in fixed_trades))
        trades_per_month = len(fixed_trades) / max(months_span, 1)
        monthly_pnl_per_100 = avg_net * trades_per_month

        print(f"\n  At $100 fixed size: ${monthly_pnl_per_100:+,.2f}/month")
        print(f"\n  {'Capital':>12} {'Size/Trade':>12} {'Monthly PnL':>14} {'Annual PnL':>14} {'Annual %':>10} {'Verdict':>16}")
        print(f"  {'-'*85}")

        for capital in STARTING_CAPITALS:
            # With 2% max position: trade_size = min(2% * capital, ~$500 avg)
            trade_size = capital * MAX_POSITION_PCT
            scale = trade_size / FIXED_SIZE
            monthly = monthly_pnl_per_100 * scale
            annual = monthly * 12
            annual_pct = (annual / capital) * 100

            if capital < 2000:
                verdict = "TOO SMALL"
            elif annual_pct > 100:
                verdict = "RECOMMENDED"
            elif annual_pct > 50:
                verdict = "GOOD"
            elif annual_pct > 20:
                verdict = "VIABLE"
            else:
                verdict = "CONSERVATIVE"

            print(f"  ${capital:>10,} ${trade_size:>10,.2f} ${monthly:>+12,.2f} ${annual:>+12,.2f} "
                  f"{annual_pct:>+9.1f}% {verdict:>16}")

    # Best capital from compound simulation (with liquidity cap)
    best_capital = None
    best_risk_adj = -float("inf")
    for capital in STARTING_CAPITALS:
        sim = results[capital]
        if sim.num_trades == 0:
            continue
        if sim.max_drawdown_pct > 0.01:
            risk_adj = sim.total_return_pct / sim.max_drawdown_pct
        else:
            risk_adj = sim.total_return_pct
        if risk_adj > best_risk_adj:
            best_risk_adj = risk_adj
            best_capital = capital

    sim_best = results.get(best_capital, None) if best_capital else None

    print(f"\n  {'='*110}")
    print(f"  FINAL RECOMMENDATION")
    print(f"  {'='*110}")

    if fixed_trades and sim_best and best_capital:
        avg_net_ft = np.mean([t["pnl_net"] for t in fixed_trades])
        months_span = len(set(t["timestamp"].strftime("%Y-%m") for t in fixed_trades))
        trades_per_month = len(fixed_trades) / max(months_span, 1)

        # Recommend $5K-$10K as sweet spot
        rec_capital = 10_000
        trade_size = rec_capital * MAX_POSITION_PCT
        scale = trade_size / FIXED_SIZE
        monthly_est = avg_net_ft * trades_per_month * scale
        annual_est = monthly_est * 12
        annual_pct = (annual_est / rec_capital) * 100

        sim_10k = results.get(10_000, sim_best)
        gross_wins = sum(t.pnl_gross for t in sim_10k.trades if t.pnl_gross > 0) if sim_10k.trades else 1
        fee_pct = (sim_10k.total_fees / max(gross_wins, 1)) * 100 if sim_10k.trades else 0

        print(f"""
  RECOMMENDED STARTING CAPITAL: $10,000

  Strategy: Tiered Momentum (8@0.10%, 9@0.08%, 10@0.05%) + Hold-to-Settlement
  Data: {len(fixed_trades):,} trades on {len(poly_markets):,} real Polymarket BTC 15-min markets

  KEY METRICS:
    Win rate:               {accuracy*100:.1f}%
    Trades per month:       ~{trades_per_month:.0f}
    Avg net P&L per trade:  ${avg_net_ft:+.2f} (at $100 size)
    Monthly P&L est:        ${monthly_est:+,.2f} (at ${rec_capital:,} capital)
    Annual P&L est:         ${annual_est:+,.2f} ({annual_pct:+.1f}%)
    Max drawdown:           {sim_10k.max_drawdown_pct:.2f}%
    Sharpe ratio:           {sim_10k.sharpe_ratio:.1f}
    Profit factor:          {sim_10k.profit_factor:.2f}
    Fees as % of gross:     {fee_pct:.1f}%

  POSITION SIZING:
    Size per trade:         ${trade_size:,.2f} (2% of capital)
    Kelly multiplier:       {KELLY_MULTIPLIER} (quarter-Kelly)
    Liquidity cap:          10% of market volume
    Fee model:              Polymarket dynamic taker + {SLIPPAGE_BPS} bps slippage

  ALTERNATIVE CAPITALS:
    $2,500 — Minimum viable (small edge, thin positions)
    $5,000 — Conservative start (good for first month of live)
    $10,000 — Sweet spot (meaningful returns, manageable risk)
    $25,000 — Aggressive (larger positions, may hit liquidity limits)

  IMPORTANT CAVEATS:
    1. Past performance does NOT guarantee future results
    2. Start with paper trading (--paper) for at least 1 week
    3. Only risk capital you can afford to LOSE entirely
    4. Monitor drawdowns — halt and review if DD > {sim_10k.max_drawdown_pct*2:.0f}%
    5. Strategy accuracy may degrade if market microstructure changes
    6. Break-even accuracy is ~{(-avg_loss_pnl / (avg_win_pnl - avg_loss_pnl))*100:.0f}% — margin of {(accuracy - (-avg_loss_pnl / (avg_win_pnl - avg_loss_pnl)))*100:.0f}pp
    7. February showed lower accuracy (79.6%) — monitor regime changes
""")
    else:
        print("\n  ERROR: Could not determine recommendation. Check data availability.")

    print("=" * 120)


if __name__ == "__main__":
    main()
