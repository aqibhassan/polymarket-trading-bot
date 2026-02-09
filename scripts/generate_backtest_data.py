"""Generate realistic BTC 15-minute OHLCV data for backtesting.

Uses a random walk with mean reversion and injected trending periods
to produce ~500 candles (~5 days of 15m data) starting around $60,000.
"""

from __future__ import annotations

import csv
import math
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
NUM_CANDLES = 500
START_PRICE = 60_000.0
START_TIME = datetime(2024, 6, 1, 0, 0, 0, tzinfo=UTC)
INTERVAL = timedelta(minutes=15)
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "btc_15m_backtest.csv"

# Random walk parameters
VOLATILITY = 0.0018  # per-candle return std (~0.18 %)
MEAN_REVERSION_STRENGTH = 0.02  # pull towards anchor
DRIFT = 0.0  # neutral drift

# Volume parameters
BASE_VOLUME_MIN = 50.0
BASE_VOLUME_MAX = 500.0

# Trend injection: list of (start_index, length, direction)
# direction > 0 => bullish run, < 0 => bearish run
TREND_INJECTIONS: list[tuple[int, int, float]] = [
    (30, 5, 0.004),    # 5 consecutive green candles early on
    (80, 4, -0.003),   # 4 consecutive red candles
    (150, 5, 0.005),   # strong bullish run
    (220, 3, -0.004),  # short bearish streak
    (300, 5, 0.003),   # another bullish streak
    (370, 4, -0.003),  # bearish before recovery
    (430, 5, 0.004),   # late bullish run
]


def _build_trend_map(injections: list[tuple[int, int, float]]) -> dict[int, float]:
    """Expand injection specs into a per-bar forced-return map."""
    forced: dict[int, float] = {}
    for start, length, per_bar_ret in injections:
        for offset in range(length):
            forced[start + offset] = per_bar_ret
    return forced


def generate_candles(
    num: int = NUM_CANDLES,
    seed: int = SEED,
) -> list[dict[str, str]]:
    """Generate *num* realistic BTC 15m OHLCV rows.

    Returns a list of dicts ready for csv.DictWriter.
    """
    rng = random.Random(seed)
    anchor = START_PRICE
    price = START_PRICE
    ts = START_TIME
    trend_map = _build_trend_map(TREND_INJECTIONS)
    rows: list[dict[str, str]] = []

    for i in range(num):
        # Determine return for this bar
        if i in trend_map:
            ret = trend_map[i] + rng.gauss(0, VOLATILITY * 0.3)
        else:
            ret = (
                DRIFT
                + MEAN_REVERSION_STRENGTH * (math.log(anchor) - math.log(price))
                + rng.gauss(0, VOLATILITY)
            )

        close = price * math.exp(ret)

        # Intra-bar high/low simulation
        intra_vol = abs(ret) + rng.uniform(0.0005, 0.002)
        raw_high = max(price, close) * (1 + rng.uniform(0, intra_vol))
        raw_low = min(price, close) * (1 - rng.uniform(0, intra_vol))

        open_price = round(price, 2)
        close_price = round(close, 2)
        high_price = round(raw_high, 2)
        low_price = round(raw_low, 2)

        # Enforce OHLCV invariants
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)

        # Volume — higher during trending periods, time-of-day modulation
        hour = ts.hour
        # Peak volume in US hours (14–22 UTC)
        tod_factor = 1.0 + 0.5 * math.exp(-((hour - 18) ** 2) / 20)
        vol_base = rng.uniform(BASE_VOLUME_MIN, BASE_VOLUME_MAX) * tod_factor
        if i in trend_map:
            vol_base *= rng.uniform(1.5, 2.5)  # trending = more volume
        volume = round(vol_base, 4)

        rows.append(
            {
                "timestamp": ts.isoformat(),
                "open": f"{open_price:.2f}",
                "high": f"{high_price:.2f}",
                "low": f"{low_price:.2f}",
                "close": f"{close_price:.2f}",
                "volume": f"{volume:.4f}",
            }
        )

        # Advance
        price = close
        ts += INTERVAL

    return rows


def main() -> None:
    rows = generate_candles()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["timestamp", "open", "high", "low", "close", "volume"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} candles -> {OUTPUT_PATH}")
    print(f"  Start : {rows[0]['timestamp']}  open={rows[0]['open']}")
    print(f"  End   : {rows[-1]['timestamp']}  close={rows[-1]['close']}")
    print(f"  High  : {max(float(r['high']) for r in rows):.2f}")
    print(f"  Low   : {min(float(r['low']) for r in rows):.2f}")


if __name__ == "__main__":
    main()
