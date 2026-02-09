"""Binance Futures WebSocket feed for BTC perpetual aggregate trades."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections import deque
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import websockets
from pydantic import BaseModel

from src.config.loader import ConfigLoader
from src.core.logging import get_logger

log = get_logger(__name__)


class BinanceFuturesWSFeed:
    """WebSocket feed for Binance USDT-M Futures aggregate trades.

    Implements DataFeed protocol from src.interfaces.
    Connects to the perpetual futures aggTrade stream and tracks
    latest price, recent trades, and tick velocity.
    """

    def __init__(
        self,
        ws_url: str = "wss://fstream.binance.com/ws/btcusdt@aggTrade",
        symbol: str = "BTCUSDT",
        max_trades: int = 1000,
        max_velocity_ticks: int = 5000,
    ) -> None:
        self._ws_url = ws_url
        self._symbol = symbol
        self._max_trades = max_trades
        self._trades: deque[dict[str, Any]] = deque(maxlen=max_trades)
        self._velocity_ticks: deque[tuple[float, Decimal]] = deque(
            maxlen=max_velocity_ticks,
        )
        self._latest_price: Decimal | None = None
        self._ws: Any = None
        self._connected = False
        self._running = False
        self._recv_task: asyncio.Task[None] | None = None
        self._backoff = 1.0
        self._max_backoff = 60.0
        self._heartbeat_interval = 30.0
        self._app_heartbeat_timeout = 15.0
        self._last_msg_time: float = 0.0

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_latest_price(self) -> Decimal | None:
        """Return the most recent futures trade price."""
        return self._latest_price

    def get_velocity(self, window_seconds: float = 60.0) -> float:
        """Return percentage price change over the given window.

        Scans the velocity tick buffer for the oldest tick within
        the window and computes (newest - oldest) / oldest * 100.
        Returns 0.0 if insufficient data.
        """
        if len(self._velocity_ticks) < 2:
            return 0.0

        now = time.monotonic()
        cutoff = now - window_seconds

        # Find the oldest tick within the window
        oldest_price: Decimal | None = None
        for ts, price in self._velocity_ticks:
            if ts >= cutoff:
                oldest_price = price
                break

        if oldest_price is None or oldest_price == 0:
            return 0.0

        newest_price = self._velocity_ticks[-1][1]
        return float((newest_price - oldest_price) / oldest_price * 100)

    async def connect(self) -> None:
        """Connect to Binance Futures WebSocket and start receiving trades."""
        self._running = True
        self._recv_task = asyncio.create_task(self._connection_loop())
        log.info(
            "binance_futures_ws.starting",
            url=self._ws_url,
            symbol=self._symbol,
        )

    async def disconnect(self) -> None:
        """Disconnect and clean up."""
        self._running = False
        if self._recv_task is not None:
            self._recv_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._recv_task
            self._recv_task = None
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
        self._connected = False
        log.info("binance_futures_ws.disconnected")

    async def subscribe(self, symbol: str) -> None:
        """Subscribe to a symbol's aggTrade stream (reconnects with new URL)."""
        self._symbol = symbol.upper()
        self._ws_url = (
            f"wss://fstream.binance.com/ws/{symbol.lower()}@aggTrade"
        )
        if self._running:
            await self.disconnect()
            await self.connect()

    async def _connection_loop(self) -> None:
        """Main loop: connect, receive, reconnect on failure."""
        while self._running:
            try:
                async with websockets.connect(
                    self._ws_url,
                    ping_interval=self._heartbeat_interval,
                    ping_timeout=10,
                ) as ws:
                    self._ws = ws
                    self._connected = True
                    self._backoff = 1.0
                    log.info("binance_futures_ws.connected", url=self._ws_url)

                    while self._running:
                        try:
                            raw_msg = await asyncio.wait_for(
                                ws.recv(),
                                timeout=self._app_heartbeat_timeout,
                            )
                        except TimeoutError:
                            log.warning(
                                "binance_futures_ws.heartbeat_timeout",
                                timeout_s=self._app_heartbeat_timeout,
                            )
                            break  # force reconnect

                        self._last_msg_time = asyncio.get_event_loop().time()
                        self._handle_message(raw_msg)

            except asyncio.CancelledError:
                break
            except Exception:
                self._connected = False
                if not self._running:
                    break
                log.warning(
                    "binance_futures_ws.reconnecting",
                    backoff_s=self._backoff,
                    exc_info=True,
                )
                await asyncio.sleep(self._backoff)
                self._backoff = min(self._backoff * 2, self._max_backoff)

        self._connected = False

    def _handle_message(self, raw_msg: str | bytes) -> None:
        """Parse an aggTrade WebSocket message and update state."""
        try:
            data: dict[str, Any] = json.loads(raw_msg)
        except (json.JSONDecodeError, TypeError):
            log.warning(
                "binance_futures_ws.invalid_json",
                raw=str(raw_msg)[:200],
            )
            return

        if data.get("e") != "aggTrade":
            return

        price = Decimal(str(data["p"]))
        quantity = Decimal(str(data["q"]))
        trade_time_ms: int = data["T"]
        is_buyer_maker: bool = data["m"]

        trade = {
            "price": price,
            "quantity": quantity,
            "timestamp": datetime.fromtimestamp(
                trade_time_ms / 1000,
                tz=UTC,
            ),
            "is_buyer_maker": is_buyer_maker,
            "agg_trade_id": data["a"],
        }

        self._trades.append(trade)
        self._latest_price = price
        self._velocity_ticks.append((time.monotonic(), price))

        log.debug(
            "binance_futures_ws.agg_trade",
            symbol=self._symbol,
            price=str(price),
            qty=str(quantity),
            side="sell" if is_buyer_maker else "buy",
        )


class LeadLagResult(BaseModel):
    """Result of futures lead-lag analysis."""

    basis_pct: float
    is_fast_move: bool
    futures_velocity_pct: float
    mispricing_pct: float
    signal_strength: float
    direction: str
    model_config = {"frozen": True}


class FuturesLeadLagDetector:
    """Detects when Binance futures lead Polymarket price discovery.

    Compares futures price, spot price, and Polymarket YES token price
    to identify mispricing opportunities when futures move first.
    """

    def __init__(self, config: ConfigLoader) -> None:
        self._lead_threshold_pct: float = config.get(
            "strategy.singularity.futures_lead_threshold_pct",
            default=0.15,
        )
        self._velocity_threshold_pct: float = config.get(
            "strategy.singularity.futures_velocity_threshold_pct",
            default=1.0,
        )

    def detect(
        self,
        futures_price: Decimal,
        spot_price: Decimal,
        polymarket_yes_price: Decimal,
        futures_velocity_pct_per_min: float,
    ) -> LeadLagResult:
        """Detect lead-lag mispricing opportunity.

        Args:
            futures_price: Current Binance USDT-M perpetual price.
            spot_price: Current Binance spot price.
            polymarket_yes_price: Current Polymarket YES token price (0-1).
            futures_velocity_pct_per_min: Futures price % change per minute.

        Returns:
            LeadLagResult with basis, velocity, mispricing, and signal info.
        """
        if spot_price == 0:
            return LeadLagResult(
                basis_pct=0.0,
                is_fast_move=False,
                futures_velocity_pct=futures_velocity_pct_per_min,
                mispricing_pct=0.0,
                signal_strength=0.0,
                direction="neutral",
            )

        # Basis: how far futures deviate from spot
        basis_pct = float(
            (futures_price - spot_price) / spot_price * 100,
        )

        # Is futures moving fast enough to signal?
        is_fast_move = (
            abs(futures_velocity_pct_per_min) >= self._velocity_threshold_pct
        )

        # Direction: determined by futures velocity
        if futures_velocity_pct_per_min > self._velocity_threshold_pct:
            direction = "long"
        elif futures_velocity_pct_per_min < -self._velocity_threshold_pct:
            direction = "short"
        else:
            direction = "neutral"

        # Mispricing: how much Polymarket lags the futures-implied fair value
        # If futures are moving up fast and basis is positive,
        # Polymarket YES price should be higher than it currently is.
        # We estimate the lag as abs(basis) when a fast move is detected.
        mispricing_pct = abs(basis_pct) if is_fast_move else 0.0

        # Signal strength: 0-1 based on how far above threshold we are
        if not is_fast_move or abs(basis_pct) < self._lead_threshold_pct:
            signal_strength = 0.0
        else:
            # Scale linearly: at threshold = 0, at 3x threshold = 1
            raw_strength = (
                abs(basis_pct) - self._lead_threshold_pct
            ) / (self._lead_threshold_pct * 2)
            signal_strength = min(1.0, max(0.0, raw_strength))

        return LeadLagResult(
            basis_pct=basis_pct,
            is_fast_move=is_fast_move,
            futures_velocity_pct=futures_velocity_pct_per_min,
            mispricing_pct=mispricing_pct,
            signal_strength=signal_strength,
            direction=direction,
        )
