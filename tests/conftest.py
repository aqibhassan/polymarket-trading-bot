"""Shared test fixtures."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path  # noqa: TCH003

import pytest

from src.config.loader import ConfigLoader
from src.models.market import (
    Candle,
    MarketState,
    OrderBookLevel,
    OrderBookSnapshot,
    Position,
    Side,
)
from src.models.order import Fill, Order, OrderSide, OrderType


@pytest.fixture()
def config_dir(tmp_path: Path) -> Path:
    """Create a temp config directory with default.toml."""
    config = tmp_path / "config"
    config.mkdir()

    default_toml = config / "default.toml"
    default_toml.write_text(
        """\
[strategy.false_sentiment]
entry_threshold_base = 0.59
threshold_time_scaling = 0.15
lookback_candles = 5
min_confidence = 0.6
max_hold_minutes = 7
force_exit_minute = 11
no_entry_after_minute = 8

[exit]
profit_target_pct = 0.05
trailing_stop_pct = 0.03
hard_stop_loss_pct = 0.04
max_hold_seconds = 420

[risk]
max_position_pct = 0.02
max_daily_drawdown_pct = 0.05
max_order_book_pct = 0.10
min_spread_threshold = 0.02
min_profitable_move = 0.03

[orderbook]
snapshot_window_seconds = 30
spoofing_detection_threshold = 0.5
heavy_book_multiplier = 3.0

[liquidity]
min_hourly_volume = 100
max_spread_cents = 0.05

[binance]
symbol = "btcusdt"
ws_url = "wss://stream.binance.com:9443/ws/btcusdt@kline_15m"

[polymarket]
clob_url = "https://clob.polymarket.com"
ws_url = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
chain_id = 137
"""
    )
    return config


@pytest.fixture()
def config_loader(config_dir: Path) -> ConfigLoader:
    """ConfigLoader with test config."""
    loader = ConfigLoader(config_dir=config_dir, env="test")
    loader.load()
    return loader


@pytest.fixture()
def sample_candle() -> Candle:
    return Candle(
        exchange="binance",
        symbol="BTCUSDT",
        open=Decimal("50000.00"),
        high=Decimal("50500.00"),
        low=Decimal("49800.00"),
        close=Decimal("50300.00"),
        volume=Decimal("123.45"),
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
    )


@pytest.fixture()
def green_candles() -> list[Candle]:
    """5 consecutive green candles (uptrend)."""
    candles = []
    base = Decimal("50000")
    start = datetime(2024, 1, 1, 12, 0, 0)
    for i in range(5):
        o = base + Decimal(str(i * 100))
        c = o + Decimal("80")
        candles.append(
            Candle(
                exchange="binance",
                symbol="BTCUSDT",
                open=o,
                high=c + Decimal("20"),
                low=o - Decimal("10"),
                close=c,
                volume=Decimal("100"),
                timestamp=start + timedelta(minutes=i * 15),
            )
        )
    return candles


@pytest.fixture()
def red_candles() -> list[Candle]:
    """5 consecutive red candles (downtrend)."""
    candles = []
    base = Decimal("50000")
    start = datetime(2024, 1, 1, 12, 0, 0)
    for i in range(5):
        o = base - Decimal(str(i * 100))
        c = o - Decimal("80")
        candles.append(
            Candle(
                exchange="binance",
                symbol="BTCUSDT",
                open=o,
                high=o + Decimal("10"),
                low=c - Decimal("20"),
                close=c,
                volume=Decimal("100"),
                timestamp=start + timedelta(minutes=i * 15),
            )
        )
    return candles


@pytest.fixture()
def sample_orderbook() -> OrderBookSnapshot:
    return OrderBookSnapshot(
        bids=[
            OrderBookLevel(price=Decimal("0.55"), size=Decimal("100")),
            OrderBookLevel(price=Decimal("0.54"), size=Decimal("200")),
            OrderBookLevel(price=Decimal("0.53"), size=Decimal("150")),
        ],
        asks=[
            OrderBookLevel(price=Decimal("0.57"), size=Decimal("100")),
            OrderBookLevel(price=Decimal("0.58"), size=Decimal("200")),
            OrderBookLevel(price=Decimal("0.59"), size=Decimal("150")),
        ],
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
        market_id="test-market",
    )


@pytest.fixture()
def sample_market_state() -> MarketState:
    return MarketState(
        market_id="test-market-123",
        yes_price=Decimal("0.62"),
        no_price=Decimal("0.38"),
        yes_bid=Decimal("0.61"),
        yes_ask=Decimal("0.63"),
        no_bid=Decimal("0.37"),
        no_ask=Decimal("0.39"),
        time_remaining_seconds=600,
        question="Will BTC 15m candle be green?",
    )


@pytest.fixture()
def sample_position() -> Position:
    return Position(
        market_id="test-market-123",
        side=Side.YES,
        token_id="token-yes-123",
        entry_price=Decimal("0.40"),
        quantity=Decimal("100"),
        entry_time=datetime(2024, 1, 1, 12, 0, 0),
        stop_loss=Decimal("0.36"),
        take_profit=Decimal("0.50"),
    )


@pytest.fixture()
def sample_order() -> Order:
    return Order(
        market_id="test-market-123",
        token_id="token-yes-123",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=Decimal("0.40"),
        size=Decimal("100"),
    )


@pytest.fixture()
def sample_fill(sample_order: Order) -> Fill:
    return Fill(
        order_id=sample_order.id,
        price=Decimal("0.40"),
        size=Decimal("50"),
        fee=Decimal("0.10"),
    )
