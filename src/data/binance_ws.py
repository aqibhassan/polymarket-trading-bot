"""Binance WebSocket feed for BTC kline data."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections import deque
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import websockets

from src.core.logging import get_logger
from src.models.market import Candle

log = get_logger(__name__)


class BinanceWSFeed:
    """WebSocket feed for Binance kline (candlestick) data.

    Implements DataFeed and CandleFeed protocols from src.interfaces.
    """

    def __init__(
        self,
        ws_url: str = "wss://stream.binance.com:9443/ws/btcusdt@kline_15m",
        symbol: str = "BTCUSDT",
        interval: str = "15m",
        max_candles: int = 50,
    ) -> None:
        self._ws_url = ws_url
        self._symbol = symbol
        self._interval = interval
        self._max_candles = max_candles
        self._candles: deque[Candle] = deque(maxlen=max_candles)
        self._ws: Any = None
        self._connected = False
        self._running = False
        self._recv_task: asyncio.Task[None] | None = None
        self._backoff = 1.0
        self._max_backoff = 60.0
        self._heartbeat_interval = 30.0
        self._app_heartbeat_timeout = 15.0  # force reconnect if no msg in 15s
        self._last_msg_time: float = 0.0

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        """Connect to Binance WebSocket and start receiving klines."""
        self._running = True
        self._recv_task = asyncio.create_task(self._connection_loop())
        log.info("binance_ws.starting", url=self._ws_url, symbol=self._symbol)

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
        log.info("binance_ws.disconnected")

    async def subscribe(self, symbol: str) -> None:
        """Subscribe to a symbol's kline stream (reconnects with new URL)."""
        self._symbol = symbol.upper()
        self._ws_url = (
            f"wss://stream.binance.com:9443/ws/{symbol.lower()}@kline_{self._interval}"
        )
        if self._running:
            await self.disconnect()
            await self.connect()

    async def get_candles(self, symbol: str, limit: int = 5) -> list[Candle]:
        """Return the most recent candles from the buffer."""
        matching = [c for c in self._candles if c.symbol == symbol.upper()]
        return matching[-limit:]

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
                    log.info("binance_ws.connected", url=self._ws_url)

                    while self._running:
                        try:
                            raw_msg = await asyncio.wait_for(
                                ws.recv(), timeout=self._app_heartbeat_timeout,
                            )
                        except TimeoutError:
                            log.warning(
                                "binance_ws.heartbeat_timeout",
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
                    "binance_ws.reconnecting",
                    backoff_s=self._backoff,
                    exc_info=True,
                )
                await asyncio.sleep(self._backoff)
                self._backoff = min(self._backoff * 2, self._max_backoff)

        self._connected = False

    def _handle_message(self, raw_msg: str | bytes) -> None:
        """Parse a kline WebSocket message and store the candle."""
        try:
            data: dict[str, Any] = json.loads(raw_msg)
        except (json.JSONDecodeError, TypeError):
            log.warning("binance_ws.invalid_json", raw=str(raw_msg)[:200])
            return

        if "k" not in data:
            return

        kline = data["k"]
        candle = Candle(
            exchange="binance",
            symbol=self._symbol,
            open=Decimal(str(kline["o"])),
            high=Decimal(str(kline["h"])),
            low=Decimal(str(kline["l"])),
            close=Decimal(str(kline["c"])),
            volume=Decimal(str(kline["v"])),
            timestamp=datetime.fromtimestamp(kline["t"] / 1000, tz=timezone.utc),
            interval=self._interval,
        )

        is_closed = kline.get("x", False)
        if is_closed:
            self._candles.append(candle)
            log.debug(
                "binance_ws.candle_closed",
                symbol=self._symbol,
                close=str(candle.close),
                ts=candle.timestamp.isoformat(),
            )
        else:
            log.debug(
                "binance_ws.candle_update",
                symbol=self._symbol,
                close=str(candle.close),
            )
