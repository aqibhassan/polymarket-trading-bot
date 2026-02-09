"""Market data models — Candle, OrderBook, MarketState, Position."""

from __future__ import annotations

from datetime import datetime  # noqa: TCH003 — Pydantic needs runtime access
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field, model_validator


class CandleDirection(str, Enum):
    GREEN = "green"
    RED = "red"
    NEUTRAL = "neutral"


class Side(str, Enum):
    YES = "YES"
    NO = "NO"


class Candle(BaseModel):
    """OHLCV candle from an exchange."""

    exchange: str
    symbol: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    timestamp: datetime
    interval: str = "15m"

    @property
    def direction(self) -> CandleDirection:
        if self.close > self.open:
            return CandleDirection.GREEN
        if self.close < self.open:
            return CandleDirection.RED
        return CandleDirection.NEUTRAL

    @property
    def body_size(self) -> Decimal:
        return abs(self.close - self.open)

    @property
    def range_size(self) -> Decimal:
        return self.high - self.low

    @model_validator(mode="after")
    def high_gte_low(self) -> Candle:
        if self.high < self.low:
            msg = "high must be >= low"
            raise ValueError(msg)
        return self

    model_config = {"frozen": True}


class OrderBookLevel(BaseModel):
    """Single price level in an order book."""

    price: Decimal
    size: Decimal

    model_config = {"frozen": True}


class OrderBookSnapshot(BaseModel):
    """Point-in-time snapshot of an order book."""

    bids: list[OrderBookLevel] = Field(default_factory=list)
    asks: list[OrderBookLevel] = Field(default_factory=list)
    timestamp: datetime
    market_id: str = ""

    @property
    def total_bid_depth(self) -> Decimal:
        return sum((lvl.size for lvl in self.bids), Decimal("0"))

    @property
    def total_ask_depth(self) -> Decimal:
        return sum((lvl.size for lvl in self.asks), Decimal("0"))

    @property
    def best_bid(self) -> Decimal | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Decimal | None:
        return self.asks[0].price if self.asks else None

    @property
    def spread(self) -> Decimal | None:
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None

    model_config = {"frozen": True}


class MarketState(BaseModel):
    """Current state of a Polymarket prediction market."""

    market_id: str
    condition_id: str = ""
    yes_token_id: str = ""
    no_token_id: str = ""
    yes_price: Decimal
    no_price: Decimal
    yes_bid: Decimal = Decimal("0")
    yes_ask: Decimal = Decimal("1")
    no_bid: Decimal = Decimal("0")
    no_ask: Decimal = Decimal("1")
    time_remaining_seconds: int
    candle_start_time: datetime | None = None
    question: str = ""

    @property
    def dominant_side(self) -> Side:
        return Side.YES if self.yes_price > self.no_price else Side.NO

    @property
    def dominant_price(self) -> Decimal:
        return max(self.yes_price, self.no_price)

    @property
    def minutes_elapsed(self) -> float:
        total = 15 * 60  # 15-minute candle
        elapsed = total - self.time_remaining_seconds
        return elapsed / 60.0

    model_config = {"frozen": True}


class Position(BaseModel):
    """An open trading position."""

    market_id: str
    side: Side
    token_id: str
    entry_price: Decimal
    quantity: Decimal
    entry_time: datetime
    stop_loss: Decimal
    take_profit: Decimal
    peak_price: Decimal | None = None
    exit_price: Decimal | None = None
    exit_time: datetime | None = None
    exit_reason: str | None = None

    @property
    def is_open(self) -> bool:
        return self.exit_price is None

    def unrealized_pnl(self, current_price: Decimal) -> Decimal:
        """Compute unrealized P&L given current market price.

        Args:
            current_price: Current market price of the token.

        Returns:
            Unrealized profit/loss as Decimal.
        """
        return (current_price - self.entry_price) * self.quantity

    def realized_pnl(self) -> Decimal | None:
        if self.exit_price is None:
            return None
        return (self.exit_price - self.entry_price) * self.quantity

    def pnl_pct(self) -> Decimal | None:
        if self.exit_price is None or self.entry_price == 0:
            return None
        return (self.exit_price - self.entry_price) / self.entry_price
