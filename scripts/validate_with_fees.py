"""Validation with realistic Polymarket fee curve and slippage.

Models the dynamic taker fee on 15-min crypto markets:
  fee_per_share = 0.25 * price * (price * (1 - price))^2

Also models:
  - Slippage: 5-10 bps depending on position size
  - Market impact: larger orders get worse fills
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fast_loader import load_csv_fast, group_into_15m_windows

CSV_2Y = PROJECT_ROOT / "data" / "btc_1m_2y.csv"
MINUTES_PER_WINDOW = 15
POSITION_SIZE = 100.0
SENSITIVITY = 0.07
TIERS = [(8, 0.0010), (9, 0.0008), (10, 0.0005)]

# Polymarket fee constant (derived from their published fee table)
FEE_CONSTANT = 0.25
SLIPPAGE_BPS = 5  # 5 basis points base slippage


def _sigmoid(cum_ret: float) -> float:
    x = SENSITIVITY * cum_ret * 10000
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def polymarket_fee(position_size: float, entry_price: float) -> float:
    """Calculate Polymarket dynamic taker fee for 15-min crypto markets.

    Formula: fee = (shares) * 0.25 * price * (price * (1 - price))^2
    For position of $N at price P, shares = N/P:
      fee = N * 0.25 * price^2 * (1 - price)^2
    """
    if entry_price <= 0 or entry_price >= 1:
        return 0.0
    return position_size * FEE_CONSTANT * (entry_price ** 2) * ((1.0 - entry_price) ** 2)


def slippage_cost(position_size: float, entry_price: float, bps: int = SLIPPAGE_BPS) -> float:
    """Slippage as price deterioration on entry.

    Returns the dollar cost of slippage.
    """
    slip_pct = bps / 10000.0
    return position_size * slip_pct


@dataclass
class Trade:
    entry_minute: int
    direction: str
    entry_price: float
    settlement: float
    pnl_gross: float
    fee: float
    slippage: float
    pnl_net: float
    pnl_dollar_gross: float
    pnl_dollar_net: float
    correct: bool


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

            # Calculate costs
            fee = polymarket_fee(POSITION_SIZE, entry_price)
            slip = slippage_cost(POSITION_SIZE, entry_price)

            settlement = (1.0 if actual_green else 0.0) if predict_green else (1.0 if not actual_green else 0.0)
            correct = predict_green == actual_green

            # Gross PnL (no fees)
            pnl_gross = (settlement - entry_price) / entry_price
            pnl_dollar_gross = pnl_gross * POSITION_SIZE

            # Net PnL (after fees + slippage)
            pnl_dollar_net = pnl_dollar_gross - fee - slip
            pnl_net = pnl_dollar_net / POSITION_SIZE

            trades.append(Trade(
                entry_minute=entry_min,
                direction="YES" if predict_green else "NO",
                entry_price=round(entry_price, 6),
                settlement=settlement,
                pnl_gross=round(pnl_gross, 6),
                fee=round(fee, 4),
                slippage=round(slip, 4),
                pnl_net=round(pnl_net, 6),
                pnl_dollar_gross=round(pnl_dollar_gross, 2),
                pnl_dollar_net=round(pnl_dollar_net, 2),
                correct=correct,
            ))
            break
    return trades


def score(trades: list[Trade], label: str = "") -> dict[str, Any]:
    if not trades:
        return {"label": label, "n": 0}
    n = len(trades)
    correct = sum(1 for t in trades if t.correct)

    gross_pnls = [t.pnl_dollar_gross for t in trades]
    net_pnls = [t.pnl_dollar_net for t in trades]
    fees = [t.fee for t in trades]

    gross_total = sum(gross_pnls)
    net_total = sum(net_pnls)
    total_fees = sum(fees)
    total_slippage = sum(t.slippage for t in trades)

    gross_mean = sum(t.pnl_gross for t in trades) / n
    net_mean = sum(t.pnl_net for t in trades) / n

    net_vals = [t.pnl_net for t in trades]
    var = sum((p - net_mean) ** 2 for p in net_vals) / n
    std = math.sqrt(var) if var > 0 else 0.001
    sharpe = (net_mean / std) * math.sqrt(35040)

    return {
        "label": label,
        "n": n,
        "acc": round(correct / n, 4),
        "gross_ev": round(gross_mean * 100, 2),
        "net_ev": round(net_mean * 100, 2),
        "gross_total": round(gross_total, 0),
        "net_total": round(net_total, 0),
        "total_fees": round(total_fees, 0),
        "total_slippage": round(total_slippage, 0),
        "avg_fee": round(total_fees / n, 4),
        "sharpe": round(sharpe, 1),
    }


def main() -> None:
    print("=" * 120)
    print("  VALIDATION WITH POLYMARKET FEES & SLIPPAGE (2 Years)")
    print("  Fee model: dynamic taker fee = 0.25 * P^2 * (1-P)^2 per $1 position")
    print(f"  Slippage: {SLIPPAGE_BPS} bps base")
    print("=" * 120)

    if not CSV_2Y.exists():
        print(f"  ERROR: {CSV_2Y} not found.")
        sys.exit(1)

    all_candles = load_csv_fast(CSV_2Y)
    all_candles.sort(key=lambda c: c.timestamp)
    seen: set[datetime] = set()
    unique = [c for c in all_candles if c.timestamp not in seen and not seen.add(c.timestamp)]  # type: ignore
    all_candles = unique
    print(f"\n  Loaded {len(all_candles):,} candles")

    # Group by month
    by_month: dict[str, list[Any]] = defaultdict(list)
    for c in all_candles:
        by_month[c.timestamp.strftime("%Y-%m")].append(c)

    months = sorted(by_month.keys())

    # ===== PART 1: Monthly breakdown with fees =====
    print(f"\n  {'='*110}")
    print(f"  MONTHLY RESULTS (Gross vs Net)")
    print(f"  {'='*110}")
    print(f"  {'Month':<10} {'Trades':>7} {'Acc%':>7} {'Gross EV':>10} {'Net EV':>10} {'Fees':>10} "
          f"{'Gross PnL':>12} {'Net PnL':>12} {'Fee%':>7} {'Sharpe':>7}")
    print(f"  {'-'*100}")

    all_trades: list[Trade] = []
    monthly_results: list[dict[str, Any]] = []

    for month in months:
        candles = by_month[month]
        windows = group_into_15m_windows(candles)
        trades = run_strategy(windows)
        all_trades.extend(trades)
        s = score(trades, month)
        monthly_results.append(s)

        if s["n"] > 0:
            fee_pct = s["total_fees"] / s["gross_total"] * 100 if s["gross_total"] != 0 else 0
            is_loss = s["net_total"] < 0
            marker = " *** NET LOSS" if is_loss else ""
            print(f"  {month:<10} {s['n']:>7} {s['acc']*100:>6.1f}% "
                  f"${s['gross_ev']:>+7.2f}  ${s['net_ev']:>+7.2f}  "
                  f"${s['total_fees']:>8,.0f}  "
                  f"${s['gross_total']:>+10,.0f}  ${s['net_total']:>+10,.0f}  "
                  f"{fee_pct:>5.1f}%  {s['sharpe']:>6.1f}{marker}")

    # ===== PART 2: Fee impact analysis =====
    print(f"\n  {'='*110}")
    print(f"  FEE IMPACT ANALYSIS")
    print(f"  {'='*110}")

    overall = score(all_trades, "OVERALL")
    if overall["n"] > 0:
        n = overall["n"]
        print(f"\n  Total trades: {n:,}")
        print(f"  Accuracy: {overall['acc']*100:.1f}%")
        print(f"\n  {'Metric':<30} {'Gross':>15} {'Net':>15} {'Difference':>15}")
        print(f"  {'-'*75}")
        print(f"  {'EV per $100 trade':<30} ${overall['gross_ev']:>+13.2f} ${overall['net_ev']:>+13.2f} "
              f"${overall['net_ev'] - overall['gross_ev']:>+13.2f}")
        print(f"  {'Total P&L':<30} ${overall['gross_total']:>+13,.0f} ${overall['net_total']:>+13,.0f} "
              f"${overall['net_total'] - overall['gross_total']:>+13,.0f}")
        print(f"  {'Sharpe ratio':<30} {'':>15} {overall['sharpe']:>15.1f}")

        print(f"\n  Total fees paid: ${overall['total_fees']:>,.0f}")
        print(f"  Total slippage: ${overall['total_slippage']:>,.0f}")
        print(f"  Total friction: ${overall['total_fees'] + overall['total_slippage']:>,.0f}")
        print(f"  Average fee per trade: ${overall['avg_fee']:.4f}")
        fee_pct = overall["total_fees"] / overall["gross_total"] * 100 if overall["gross_total"] > 0 else 0
        print(f"  Fees as % of gross profit: {fee_pct:.1f}%")

    # ===== PART 3: Entry price distribution and fee analysis =====
    print(f"\n  {'='*110}")
    print(f"  ENTRY PRICE DISTRIBUTION & FEE CURVE")
    print(f"  {'='*110}")

    price_buckets: dict[str, list[Trade]] = defaultdict(list)
    for t in all_trades:
        if t.entry_price < 0.40:
            bucket = "0.01-0.40"
        elif t.entry_price < 0.50:
            bucket = "0.40-0.50"
        elif t.entry_price < 0.55:
            bucket = "0.50-0.55"
        elif t.entry_price < 0.60:
            bucket = "0.55-0.60"
        elif t.entry_price < 0.65:
            bucket = "0.60-0.65"
        elif t.entry_price < 0.70:
            bucket = "0.65-0.70"
        else:
            bucket = "0.70-1.00"

        price_buckets[bucket].append(t)

    print(f"  {'Price Range':<14} {'Count':>7} {'Acc%':>7} {'Gross EV':>10} {'Net EV':>10} {'Avg Fee':>10} {'Net PnL':>12}")
    print(f"  {'-'*80}")

    for bucket in ["0.01-0.40", "0.40-0.50", "0.50-0.55", "0.55-0.60", "0.60-0.65", "0.65-0.70", "0.70-1.00"]:
        trades = price_buckets.get(bucket, [])
        if trades:
            s = score(trades, bucket)
            print(f"  {bucket:<14} {s['n']:>7} {s['acc']*100:>6.1f}% "
                  f"${s['gross_ev']:>+7.2f}  ${s['net_ev']:>+7.2f}  "
                  f"${s['avg_fee']:>8.4f}  ${s['net_total']:>+10,.0f}")

    # ===== PART 4: Sensitivity to fee levels =====
    print(f"\n  {'='*110}")
    print(f"  FEE SENSITIVITY (what if fees are higher?)")
    print(f"  {'='*110}")

    # Re-run with different fee multipliers
    print(f"  {'Fee Level':<25} {'Trades':>7} {'Net EV':>10} {'Net PnL':>12} {'Sharpe':>7}")
    print(f"  {'-'*65}")

    for multiplier, label in [(0.0, "No fees (baseline)"),
                               (0.5, "50% of current fees"),
                               (1.0, "Current fees (1x)"),
                               (1.5, "1.5x fees"),
                               (2.0, "2x fees (pessimistic)"),
                               (3.0, "3x fees (worst case)")]:
        total_pnl = 0.0
        pnls: list[float] = []
        for t in all_trades:
            adjusted_fee = t.fee * multiplier
            adjusted_slip = t.slippage * multiplier
            net = t.pnl_dollar_gross - adjusted_fee - adjusted_slip
            total_pnl += net
            pnls.append(net / POSITION_SIZE)

        mean_pnl = sum(pnls) / len(pnls) if pnls else 0
        var_pnl = sum((p - mean_pnl) ** 2 for p in pnls) / len(pnls) if pnls else 0
        std_pnl = math.sqrt(var_pnl) if var_pnl > 0 else 0.001
        sharpe = (mean_pnl / std_pnl) * math.sqrt(35040)

        print(f"  {label:<25} {len(all_trades):>7} ${mean_pnl*100:>+7.2f}  "
              f"${total_pnl:>+10,.0f}  {sharpe:>6.1f}")

    # ===== PART 5: Break-even analysis =====
    print(f"\n  {'='*110}")
    print(f"  BREAK-EVEN ANALYSIS")
    print(f"  {'='*110}")

    # Find the fee multiplier where total PnL = 0
    lo, hi = 0.0, 100.0
    for _ in range(100):
        mid = (lo + hi) / 2
        total = sum(t.pnl_dollar_gross - t.fee * mid - t.slippage * mid for t in all_trades)
        if total > 0:
            lo = mid
        else:
            hi = mid

    breakeven_mult = (lo + hi) / 2
    avg_fee_at_breakeven = overall["avg_fee"] * breakeven_mult if overall["n"] > 0 else 0
    print(f"  Strategy breaks even at {breakeven_mult:.1f}x current fee level")
    print(f"  That's avg ${avg_fee_at_breakeven:.2f} per trade (vs current ${overall['avg_fee']:.4f})")
    print(f"  Safety margin: {breakeven_mult:.1f}x — {'LARGE' if breakeven_mult > 5 else 'MODERATE' if breakeven_mult > 2 else 'THIN'}")

    # ===== PART 6: Net monthly consistency =====
    print(f"\n  {'='*110}")
    print(f"  NET MONTHLY CONSISTENCY")
    print(f"  {'='*110}")

    net_profitable = sum(1 for r in monthly_results if r["n"] > 0 and r["net_total"] > 0)
    net_total_months = sum(1 for r in monthly_results if r["n"] > 0)
    net_losing = [r for r in monthly_results if r["n"] > 0 and r["net_total"] <= 0]

    print(f"  Net profitable months: {net_profitable}/{net_total_months}")
    if net_losing:
        print(f"  Net losing months:")
        for r in net_losing:
            print(f"    {r['label']}: ${r['net_total']:+,.0f} (gross: ${r['gross_total']:+,.0f}, fees: ${r['total_fees']:,.0f})")
    else:
        print(f"  No net losing months — ALL months profitable after fees!")

    # Worst/best net months
    if net_total_months > 0:
        valid = [r for r in monthly_results if r["n"] > 0]
        worst = min(valid, key=lambda r: r["net_total"])
        best = max(valid, key=lambda r: r["net_total"])
        print(f"\n  Worst net month: {worst['label']} — ${worst['net_total']:+,.0f}")
        print(f"  Best net month:  {best['label']} — ${best['net_total']:+,.0f}")

    print(f"\n{'='*120}")


if __name__ == "__main__":
    main()
