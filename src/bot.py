"""MVHE Bot orchestrator — wires all components together.

This module is the main event loop that subscribes to data feeds,
generates signals, checks risk, and executes trades.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import signal
from decimal import Decimal
from pathlib import Path
from typing import Any

from src.config.loader import ConfigLoader
from src.core.logging import get_logger
from src.data.redis_cache import RedisCache

# K8s health probe sentinel files
_SENTINEL_DIR = Path("/tmp")
_READY_FILE = _SENTINEL_DIR / "mvhe_ready"
_HEARTBEAT_FILE = _SENTINEL_DIR / "mvhe_heartbeat"
_WS_CONNECTED_FILE = _SENTINEL_DIR / "mvhe_ws_connected"

# Optional engine components — other teams deliver in parallel
try:
    from src.engine.exit_manager import ExitManager
except ImportError:  # pragma: no cover
    ExitManager = None  # type: ignore[assignment,misc]

logger = get_logger(__name__)

# Sigmoid sensitivity for YES/NO price modelling (matches swing backtest)
_SIGMOID_SENSITIVITY = 0.07


class BotOrchestrator:
    """Main bot event loop: data -> signals -> risk -> execution."""

    def __init__(
        self,
        mode: str,
        strategy_name: str,
        config: ConfigLoader,
        fixed_bet_size: float = 0.0,
    ) -> None:
        self._mode = mode
        self._strategy_name = strategy_name
        self._config = config
        self._fixed_bet_size = Decimal(str(fixed_bet_size)) if fixed_bet_size > 0 else Decimal("0")
        self._running = False
        self._tick_interval = float(config.get("bot.tick_interval_seconds", 5.0))
        self._strategy: Any = None
        self._shutdown_event = asyncio.Event()

    async def start(self) -> int:
        """Initialize components and run the main loop."""
        from src.strategies import registry

        # Ensure strategy modules are imported for registration
        with contextlib.suppress(ImportError):
            import src.strategies.false_sentiment  # noqa: F401
        with contextlib.suppress(ImportError):
            import src.strategies.reversal_catcher  # noqa: F401
        with contextlib.suppress(ImportError):
            import src.strategies.singularity  # noqa: F401

        self._strategy = registry.create(self._strategy_name, self._config)
        logger.info(
            "bot_start",
            mode=self._mode,
            strategy=self._strategy_name,
        )

        self._running = True

        # Install signal handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._request_shutdown, sig)

        logger.info("bot_ready", mode=self._mode, strategy=self._strategy_name)
        self._touch_sentinel(_READY_FILE)

        try:
            if self._strategy_name in ("reversal_catcher", "momentum_confirmation", "singularity"):
                await self._run_swing_loop()
            else:
                await self._main_loop()
        except asyncio.CancelledError:
            logger.info("bot_cancelled")
        finally:
            await self._cleanup()

        return 0

    def _request_shutdown(self, sig: signal.Signals) -> None:
        """Handle OS signal for graceful shutdown."""
        logger.info("shutdown_requested", signal=sig.name)
        self._running = False
        self._shutdown_event.set()

    async def _main_loop(self) -> None:
        """Core tick loop: fetch data, generate signals, check risk, execute."""
        while self._running:
            self._touch_sentinel(_HEARTBEAT_FILE)
            try:
                await self._tick()
            except Exception:
                logger.exception("tick_error")
                # Don't crash on single tick failure — continue running

            # Wait for next tick or shutdown signal
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self._tick_interval,
                )
                break  # Shutdown requested
            except TimeoutError:
                continue

    async def _tick(self) -> None:
        """Single tick: fetch market data, run strategy, process signals.

        Generic fallback for non-momentum strategies — not yet wired
        to real data feeds, risk manager, or execution bridge.
        """
        # Placeholder — will be wired to real feeds
        logger.debug("tick", mode=self._mode, strategy=self._strategy_name)

    @staticmethod
    def _touch_sentinel(path: Path) -> None:
        """Create or update a sentinel file for K8s probes."""
        try:
            path.touch(exist_ok=True)
        except OSError:
            pass  # Non-fatal — probes degrade gracefully

    @staticmethod
    def _remove_sentinel(path: Path) -> None:
        """Remove a sentinel file."""
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    def _cleanup_sentinels(self) -> None:
        """Remove all sentinel files on shutdown."""
        for path in (_READY_FILE, _HEARTBEAT_FILE, _WS_CONNECTED_FILE):
            self._remove_sentinel(path)

    async def _cleanup(self) -> None:
        """Graceful cleanup on shutdown."""
        logger.info("bot_shutdown", mode=self._mode)
        self._running = False
        self._cleanup_sentinels()

    # ------------------------------------------------------------------
    # Swing trading mode (1m candles -> 15m windows)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_yes_price(cum_return: float) -> Decimal:
        """Model YES token price using sigmoid of cumulative BTC return.

        A positive cumulative return (BTC up) drives YES price higher (green
        candle more likely).  Sensitivity matches the swing backtest default.
        """
        prob = 1.0 / (1.0 + math.exp(-cum_return / _SIGMOID_SENSITIVITY))
        return Decimal(str(round(prob, 6)))

    @staticmethod
    def _window_start_ts(epoch_s: int) -> int:
        """Align an epoch timestamp to the enclosing 15-minute boundary."""
        return epoch_s - (epoch_s % 900)

    async def _publish_state(
        self,
        cache: RedisCache,
        *,
        start_time: Any,
        paper_trader: Any,
        initial_balance: Any,
        daily_pnl: Any,
        trade_count: int,
        win_count: int,
        loss_count: int,
        has_open_position: bool,
        active_market: Any,
        entry_price: Any | None = None,
        position_size: Any | None = None,
        entry_side: str | None = None,
        entry_time: Any | None = None,
        current_window_start: int | None = None,
        minute_in_window: int = 0,
        cum_return: float = 0.0,
        yes_price: Any | None = None,
        btc_close: Any | None = None,
        ws_connected: bool = False,
    ) -> None:
        """Publish bot state to Redis for dashboard consumption."""
        from datetime import datetime, timezone

        now = datetime.now(tz=timezone.utc)
        try:
            # Heartbeat
            uptime_s = (now - start_time).total_seconds()
            await cache.set("bot:heartbeat", {
                "timestamp": now.isoformat(),
                "uptime_s": round(uptime_s, 1),
                "mode": self._mode,
                "strategy": self._strategy_name,
            }, ttl=120)

            # Balance
            balance = paper_trader.balance
            pnl = balance - initial_balance
            pnl_pct = float(pnl / initial_balance * 100) if initial_balance else 0.0
            await cache.set("bot:balance", {
                "balance": str(balance),
                "initial_balance": str(initial_balance),
                "pnl": str(pnl),
                "pnl_pct": round(pnl_pct, 4),
            }, ttl=300)

            # Position
            if has_open_position and active_market is not None:
                await cache.set("bot:position", {
                    "market_id": active_market.market_id,
                    "side": entry_side or "YES",
                    "entry_price": str(entry_price or "0"),
                    "size": str(position_size or "0"),
                    "entry_time": str(entry_time or now.isoformat()),
                }, ttl=300)
            else:
                await cache.set("bot:position", None, ttl=300)

            # Window
            await cache.set("bot:window", {
                "start_ts": current_window_start or 0,
                "minute": minute_in_window,
                "cum_return_pct": round(cum_return * 100, 4),
                "yes_price": str(yes_price or "0.5"),
                "btc_close": str(btc_close or "0"),
            }, ttl=120)

            # Daily stats
            await cache.set("bot:daily", {
                "date": now.strftime("%Y-%m-%d"),
                "trade_count": trade_count,
                "daily_pnl": str(daily_pnl),
                "win_count": win_count,
                "loss_count": loss_count,
            }, ttl=300)

            # WebSocket status
            await cache.set("bot:ws_status", {
                "binance": ws_connected,
                "polymarket": active_market is not None,
            }, ttl=120)

        except Exception:
            logger.debug("state_publish_failed", exc_info=True)

    async def _record_completed_trade(
        self,
        ch_store: Any,
        *,
        market_id: str,
        direction: str,
        entry_price: Any,
        exit_price: Any,
        position_size: Any,
        entry_time: Any,
        exit_reason: str,
        window_minute: int,
        cum_return_pct: float,
        confidence: float,
        fee_cost: Any = Decimal("0"),
        pnl: Any = Decimal("0"),
    ) -> None:
        """Record a completed trade to ClickHouse for dashboard display."""
        import uuid
        from datetime import datetime, timezone

        exit_time = datetime.now(tz=timezone.utc)
        trade_data = {
            "trade_id": f"paper-{uuid.uuid4().hex[:8]}",
            "market_id": market_id,
            "strategy": self._strategy_name,
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "position_size": position_size,
            "pnl": pnl,
            "fee_cost": fee_cost,
            "entry_time": entry_time or exit_time,
            "exit_time": exit_time,
            "exit_reason": exit_reason,
            "window_minute": window_minute,
            "cum_return_pct": cum_return_pct,
            "confidence": confidence,
        }
        try:
            await ch_store.insert_trade(trade_data)
            await ch_store.insert_audit_event({
                "order_id": f"paper-ord-{uuid.uuid4().hex[:8]}",
                "event_type": "PAPER_TRADE_CLOSE",
                "market_id": market_id,
                "strategy": self._strategy_name,
                "details": f"Closed {direction} at {exit_price}, PnL: {pnl}, reason: {exit_reason}",
                "timestamp": exit_time,
            })
            logger.info(
                "trade_recorded_to_clickhouse",
                direction=direction,
                pnl=str(pnl),
                exit_reason=exit_reason,
            )
        except Exception:
            logger.warning("clickhouse_trade_record_failed", exc_info=True)

    async def _run_swing_loop(self) -> None:
        """Core loop for 1m swing trading mode.

        Subscribes to Binance 1m BTCUSDT klines, collects candles into
        15-minute windows, runs the momentum_confirmation strategy each minute,
        and force-closes positions at the window boundary.

        Uses Kelly position sizing and real Portfolio drawdown tracking.
        """
        from datetime import datetime, timezone

        from src.data.binance_futures_ws import BinanceFuturesWSFeed
        from src.data.binance_ws import BinanceWSFeed
        from src.data.polymarket_scanner import PolymarketScanner
        from src.execution.bridge import ExecutionBridge
        from src.execution.paper_trader import PaperTrader
        from src.models.market import Candle, MarketState, OrderBookSnapshot, Side
        from src.models.order import OrderSide, OrderType
        from src.models.signal import SignalType
        from src.risk.cost_calculator import CostCalculator
        from src.risk.kill_switch import KillSwitch
        from src.risk.portfolio import Portfolio
        from src.risk.position_sizer import PositionSize, PositionSizer
        from src.data.clickhouse_store import ClickHouseStore
        from src.risk.risk_manager import RiskManager

        # --- Config ---
        max_position_pct = Decimal(
            str(self._config.get("risk.max_position_pct", 0.02))
        )
        max_daily_drawdown_pct = Decimal(
            str(self._config.get("risk.max_daily_drawdown_pct", 0.05))
        )
        kelly_multiplier = Decimal(
            str(self._config.get("risk.kelly_multiplier", 0.25))
        )
        initial_balance = Decimal(
            str(self._config.get("paper.initial_balance", 10000))
        )
        # Estimated win probability — strategy-specific
        if self._strategy_name == "singularity":
            estimated_win_prob = Decimal(
                str(self._config.get("strategy.singularity.estimated_win_prob", 0.90))
            )
        else:
            estimated_win_prob = Decimal(
                str(self._config.get("strategy.momentum_confirmation.estimated_win_prob", 0.884))
            )

        # --- Component setup ---
        ws_feed = BinanceWSFeed(
            ws_url="wss://stream.binance.com:9443/ws/btcusdt@kline_1m",
            symbol="BTCUSDT",
            interval="1m",
        )
        futures_feed = BinanceFuturesWSFeed(
            ws_url="wss://fstream.binance.com/ws/btcusdt@aggTrade",
            symbol="BTCUSDT",
        )
        scanner = PolymarketScanner()
        cache = RedisCache()
        redis_conn = await cache._get_redis()
        kill_switch = KillSwitch(
            max_daily_drawdown_pct, redis_client=redis_conn, async_redis=True,
        )
        await kill_switch.async_load_state()
        risk_mgr = RiskManager(
            max_position_pct=max_position_pct,
            max_daily_drawdown_pct=max_daily_drawdown_pct,
            kill_switch=kill_switch,
        )
        position_sizer = PositionSizer(
            max_position_pct=max_position_pct,
            kelly_multiplier=kelly_multiplier,
        )
        cost_calculator = CostCalculator()
        portfolio = Portfolio()
        portfolio.update_equity(initial_balance)
        paper_trader = PaperTrader(initial_balance=initial_balance)
        bridge = ExecutionBridge(mode=self._mode, paper_trader=paper_trader)
        ch_store: ClickHouseStore | None = None
        try:
            ch_store = ClickHouseStore()
            await ch_store.connect()
        except Exception:
            logger.warning("clickhouse_connect_failed", exc_info=True)
            ch_store = None

        await ws_feed.connect()
        await futures_feed.connect()
        ws_connected = True
        self._touch_sentinel(_WS_CONNECTED_FILE)

        # --- Window state ---
        window_candles: list[Candle] = []
        window_open_price: Decimal | None = None
        current_window_start: int | None = None
        has_open_position = False
        active_market: MarketState | None = None
        daily_pnl = Decimal("0")
        trade_count = 0
        win_count = 0
        loss_count = 0
        last_entry_price: Decimal | None = None
        last_entry_side: str | None = None
        last_entry_time: datetime | None = None
        last_position_size: Decimal | None = None
        start_time = datetime.now(tz=timezone.utc)
        last_confidence: float = 0.0
        last_entry_minute: int = 0
        last_cum_return_entry: float = 0.0
        last_fee_cost = Decimal("0")
        last_token_id: str | None = None
        last_tick_yes_price = Decimal("0.5")
        last_market_id: str | None = None
        last_reset_date = datetime.now(tz=timezone.utc).date()

        logger.info(
            "swing_loop_started",
            mode=self._mode,
            initial_balance=str(initial_balance),
            kelly_multiplier=str(kelly_multiplier),
            max_position_pct=str(max_position_pct),
        )

        try:
            while self._running:
                self._touch_sentinel(_HEARTBEAT_FILE)

                # Daily stats reset at UTC midnight
                now_utc = datetime.now(tz=timezone.utc)
                if now_utc.date() != last_reset_date:
                    daily_pnl = Decimal("0")
                    trade_count = 0
                    win_count = 0
                    loss_count = 0
                    # Reset portfolio drawdown tracking for new day
                    portfolio = Portfolio()
                    portfolio.update_equity(paper_trader.balance)
                    last_reset_date = now_utc.date()
                    logger.info("daily_stats_reset", date=str(last_reset_date))

                # Wait for a 1m candle to close
                candles = await ws_feed.get_candles("BTCUSDT", limit=1)

                if not candles:
                    # No candle yet — wait and retry
                    try:
                        await asyncio.wait_for(
                            self._shutdown_event.wait(), timeout=5.0,
                        )
                        break
                    except TimeoutError:
                        continue

                latest = candles[-1]
                candle_epoch = int(latest.timestamp.timestamp())
                w_start = self._window_start_ts(candle_epoch)

                # --- Window boundary transition ---
                if current_window_start is not None and w_start != current_window_start:
                    # Force-close any open position at window end
                    if has_open_position and last_entry_price is not None:
                        close_market_id = last_market_id or (
                            active_market.market_id if active_market else "unknown"
                        )
                        # Binary settlement: $1 if correct, $0 if wrong
                        if window_candles and window_open_price is not None and float(window_open_price) > 0:
                            final_close = float(window_candles[-1].close)
                            final_cum_return = (
                                (final_close - float(window_open_price))
                                / float(window_open_price)
                            )
                        else:
                            final_cum_return = cum_return  # fallback
                            logger.warning(
                                "swing_window_close_fallback",
                                fallback_cum_return=cum_return,
                                window_open_price=str(window_open_price),
                                has_candles=bool(window_candles),
                                msg="using stale cum_return — window_candles or open_price unavailable",
                            )

                        winning_side = "YES" if final_cum_return >= 0 else "NO"
                        if last_entry_side == winning_side:
                            exit_price_wb = Decimal("1")  # correct prediction
                        else:
                            exit_price_wb = Decimal("0")  # wrong prediction
                        # Submit sell to update PaperTrader balance
                        try:
                            await bridge.submit_order(
                                market_id=close_market_id,
                                token_id=last_token_id or "",
                                side=OrderSide.SELL,
                                order_type=OrderType.MARKET,
                                price=exit_price_wb,
                                size=last_position_size or Decimal("0"),
                                strategy_id=self._strategy_name,
                            )
                        except Exception:
                            logger.warning("swing_window_sell_failed", exc_info=True)
                        pnl_wb = (exit_price_wb - last_entry_price) * (
                            last_position_size or Decimal("0")
                        )
                        daily_pnl += pnl_wb
                        portfolio.update_equity(paper_trader.balance)
                        if pnl_wb > 0:
                            win_count += 1
                        else:
                            loss_count += 1
                        if ch_store is not None:
                            await self._record_completed_trade(
                                ch_store,
                                market_id=close_market_id,
                                direction=last_entry_side or "YES",
                                entry_price=last_entry_price,
                                exit_price=exit_price_wb,
                                position_size=last_entry_price * (
                                    last_position_size or Decimal("0")
                                ),
                                entry_time=last_entry_time,
                                exit_reason="settlement",
                                window_minute=last_entry_minute,
                                cum_return_pct=last_cum_return_entry,
                                confidence=last_confidence,
                                fee_cost=last_fee_cost,
                                pnl=pnl_wb,
                            )
                        logger.info(
                            "swing_window_force_close",
                            window_start=current_window_start,
                            market_id=close_market_id,
                            pnl=str(pnl_wb),
                        )
                        has_open_position = False
                        last_entry_price = None
                        last_entry_side = None
                        last_entry_time = None
                        last_position_size = None

                    # Always clear active_market and stale position metadata
                    # at window boundary to prevent state leak between windows
                    active_market = None
                    last_token_id = None
                    last_market_id = None

                    window_candles = []
                    window_open_price = None
                    current_window_start = None

                # --- First candle in a new window ---
                if current_window_start is None:
                    current_window_start = w_start
                    window_open_price = latest.open

                    # Discover current Polymarket BTC 15m market
                    # Pick the latest market (last in list) since scanner returns
                    # previous/current/next windows — the latest one is still
                    # actively trading and will have a live orderbook.
                    try:
                        active_markets = await scanner.scan_active_markets()
                        if active_markets:
                            active_market = active_markets[-1]
                            logger.info(
                                "swing_market_found",
                                market_id=active_market.market_id,
                                question=active_market.question[:80] if active_market.question else "",
                                yes_price=str(active_market.yes_price),
                                candidates=len(active_markets),
                            )
                        else:
                            active_market = None
                            logger.info("swing_no_active_market")
                    except Exception:
                        active_market = None
                        logger.warning("swing_market_scan_failed", exc_info=True)

                    logger.info(
                        "swing_window_new",
                        window_start=datetime.fromtimestamp(w_start, tz=timezone.utc).isoformat(),
                        open_price=str(window_open_price),
                    )

                # Deduplicate: skip if we already have this candle
                if window_candles and window_candles[-1].timestamp == latest.timestamp:
                    try:
                        await asyncio.wait_for(
                            self._shutdown_event.wait(), timeout=10.0,
                        )
                        break
                    except TimeoutError:
                        continue

                window_candles.append(latest)
                minute_in_window = (candle_epoch - current_window_start) // 60
                assert window_open_price is not None

                # Cumulative return from window open
                cum_return = float(
                    (latest.close - window_open_price) / window_open_price
                )
                yes_price = self._compute_yes_price(cum_return)
                last_tick_yes_price = yes_price

                logger.info(
                    "swing_tick",
                    minute=minute_in_window,
                    close=str(latest.close),
                    cum_return_pct=round(cum_return * 100, 4),
                    yes_price=str(yes_price),
                    balance=str(paper_trader.balance),
                    daily_pnl=str(daily_pnl),
                    trades_today=trade_count,
                )

                # Publish state to Redis for dashboard
                await self._publish_state(
                    cache,
                    start_time=start_time,
                    paper_trader=paper_trader,
                    initial_balance=initial_balance,
                    daily_pnl=daily_pnl,
                    trade_count=trade_count,
                    win_count=win_count,
                    loss_count=loss_count,
                    has_open_position=has_open_position,
                    active_market=active_market,
                    entry_price=last_entry_price,
                    position_size=last_position_size,
                    entry_side=last_entry_side,
                    entry_time=last_entry_time,
                    current_window_start=current_window_start,
                    minute_in_window=minute_in_window,
                    cum_return=cum_return,
                    yes_price=yes_price,
                    btc_close=latest.close,
                    ws_connected=ws_connected,
                )

                # --- Check kill switch ---
                if await kill_switch.async_check(daily_pnl, paper_trader.balance):
                    logger.critical(
                        "swing_kill_switch_active",
                        daily_pnl=str(daily_pnl),
                        balance=str(paper_trader.balance),
                    )
                    try:
                        await asyncio.wait_for(
                            self._shutdown_event.wait(), timeout=60.0,
                        )
                        break
                    except TimeoutError:
                        continue

                # --- Build context for strategy ---
                # Futures lead-lag data from Binance perpetuals
                futures_price = futures_feed.get_latest_price()
                spot_price = latest.close  # Binance spot 1m candle close

                context: dict[str, Any] = {
                    "candles_1m": window_candles,
                    "window_open_price": window_open_price,
                    "minute_in_window": minute_in_window,
                    "yes_price": yes_price,
                    # Futures lead-lag data
                    "futures_price": futures_price,
                    "spot_price": spot_price,
                    "futures_velocity_pct_per_min": futures_feed.get_velocity(60.0),
                    # Tick buffer for vol regime
                    "recent_ticks": [c.close for c in window_candles],
                }

                # Build MarketState from scanner result or synthetic
                market_id = active_market.market_id if active_market else "btc_15m_swing"
                yes_token_id = active_market.yes_token_id if active_market else ""
                no_token_id = active_market.no_token_id if active_market else ""

                market_state = MarketState(
                    market_id=market_id,
                    yes_token_id=yes_token_id,
                    no_token_id=no_token_id,
                    yes_price=yes_price,
                    no_price=Decimal("1") - yes_price,
                    time_remaining_seconds=max(0, (14 - minute_in_window) * 60),
                )

                # Fetch real orderbook from Polymarket CLOB API
                try:
                    if yes_token_id:
                        orderbook = await scanner.get_market_orderbook(yes_token_id)
                    else:
                        orderbook = OrderBookSnapshot(
                            timestamp=datetime.now(tz=timezone.utc),
                            market_id=market_id,
                        )
                except Exception:
                    logger.debug("orderbook_fetch_failed", market_id=market_id, exc_info=True)
                    orderbook = OrderBookSnapshot(
                        timestamp=datetime.now(tz=timezone.utc),
                        market_id=market_id,
                    )

                # --- Generate signals ---
                signals = self._strategy.generate_signals(
                    market_state, orderbook, context,
                )

                for sig in signals:
                    if sig.signal_type == SignalType.ENTRY and not has_open_position:
                        entry_price = sig.entry_price or yes_price

                        # --- Position sizing: fixed bet or Kelly ---
                        if self._fixed_bet_size > 0:
                            # Flat bet mode — fixed USD per trade
                            position_size = min(self._fixed_bet_size, paper_trader.balance)
                            sizing = PositionSize(
                                recommended_size=position_size,
                                max_allowed=position_size,
                                kelly_fraction=Decimal("0"),
                                capped_reason="fixed_bet",
                            )
                        else:
                            sizing = position_sizer.calculate_binary(
                                balance=paper_trader.balance,
                                entry_price=entry_price,
                                estimated_win_prob=estimated_win_prob,
                            )
                            position_size = sizing.recommended_size

                        if position_size <= 0:
                            logger.info(
                                "swing_entry_skip_no_edge",
                                entry_price=str(entry_price),
                                kelly=str(sizing.kelly_fraction),
                                reason=sizing.capped_reason or "zero size",
                            )
                            continue

                        # Cost check — reduce position size to leave room for fees
                        costs = cost_calculator.calculate_binary(
                            entry_price=entry_price,
                            position_size=position_size,
                        )
                        # Ensure position + fees fits within max_position_pct cap
                        max_allowed = paper_trader.balance * max_position_pct
                        if position_size + costs.fee_cost > max_allowed and costs.fee_cost > 0:
                            position_size = max_allowed - costs.fee_cost
                            if position_size <= 0:
                                continue
                            # Recalculate fees at reduced size
                            costs = cost_calculator.calculate_binary(
                                entry_price=entry_price,
                                position_size=position_size,
                            )

                        # Risk gate (include fees in size check)
                        current_drawdown = portfolio.max_drawdown
                        decision = risk_mgr.check_order(
                            signal=sig,
                            position_size=position_size,
                            current_drawdown=current_drawdown,
                            balance=paper_trader.balance,
                            estimated_fee=costs.fee_cost,
                        )
                        if decision.approved:
                            token_id = (
                                yes_token_id
                                if sig.direction == Side.YES
                                else no_token_id
                            )
                            entry_order = await bridge.submit_order(
                                market_id=sig.market_id,
                                token_id=token_id,
                                side=OrderSide.BUY,
                                order_type=OrderType.LIMIT,
                                price=entry_price,
                                size=position_size,
                                strategy_id=sig.strategy_id,
                            )
                            # Use actual filled size (handles partial fills)
                            actual_fill_size = entry_order.filled_size or position_size
                            actual_fill_price = entry_order.avg_fill_price or entry_price
                            has_open_position = True
                            trade_count += 1
                            last_entry_price = actual_fill_price
                            last_entry_side = sig.direction.value
                            last_entry_time = datetime.now(tz=timezone.utc)
                            last_position_size = actual_fill_size
                            last_confidence = sig.confidence.overall if sig.confidence else 0.0
                            last_entry_minute = minute_in_window
                            last_cum_return_entry = cum_return * 100
                            last_fee_cost = costs.fee_cost
                            last_token_id = token_id
                            last_market_id = market_id
                            # Record entry audit event
                            if ch_store is not None:
                                try:
                                    await ch_store.insert_audit_event({
                                        "event_type": "PAPER_ENTRY",
                                        "market_id": market_id,
                                        "strategy": self._strategy_name,
                                        "details": (
                                            f"Entered {sig.direction.value} at "
                                            f"{entry_price}, size={position_size}, "
                                            f"kelly={sizing.kelly_fraction}"
                                        ),
                                        "timestamp": last_entry_time,
                                    })
                                except Exception:
                                    logger.debug("clickhouse_audit_failed", exc_info=True)
                            # Track equity after entry (balance decreased by cost)
                            portfolio.update_equity(paper_trader.balance)
                            logger.info(
                                "swing_entry_executed",
                                direction=sig.direction.value,
                                price=str(entry_price),
                                size=str(position_size),
                                kelly=str(sizing.kelly_fraction),
                                capped=sizing.capped_reason or "none",
                                est_fee=str(costs.fee_cost),
                                balance=str(paper_trader.balance),
                            )
                        else:
                            logger.info(
                                "swing_entry_rejected",
                                reason=decision.reason,
                                requested_size=str(position_size),
                            )

                    elif sig.signal_type == SignalType.EXIT and has_open_position:
                        # Binary settlement: $1 if correct, $0 if wrong
                        # Same logic as window boundary close
                        if window_candles and window_open_price is not None and float(window_open_price) > 0:
                            final_close_ex = float(window_candles[-1].close)
                            final_cum_return_ex = (
                                (final_close_ex - float(window_open_price))
                                / float(window_open_price)
                            )
                        else:
                            final_cum_return_ex = cum_return
                        winning_side_ex = "YES" if final_cum_return_ex >= 0 else "NO"
                        if last_entry_side == winning_side_ex:
                            exit_price_ex = Decimal("1")
                        else:
                            exit_price_ex = Decimal("0")
                        # Submit sell to credit proceeds
                        try:
                            await bridge.submit_order(
                                market_id=market_id,
                                token_id=last_token_id or "",
                                side=OrderSide.SELL,
                                order_type=OrderType.MARKET,
                                price=exit_price_ex,
                                size=last_position_size or Decimal("0"),
                                strategy_id=self._strategy_name,
                            )
                        except Exception:
                            logger.warning("swing_exit_sell_failed", exc_info=True)
                        pnl_ex = (exit_price_ex - (last_entry_price or Decimal("0"))) * (
                            last_position_size or Decimal("0")
                        )
                        daily_pnl += pnl_ex
                        portfolio.update_equity(paper_trader.balance)
                        if pnl_ex > 0:
                            win_count += 1
                        else:
                            loss_count += 1
                        if ch_store is not None:
                            await self._record_completed_trade(
                                ch_store,
                                market_id=market_id,
                                direction=last_entry_side or "YES",
                                entry_price=last_entry_price or Decimal("0"),
                                exit_price=exit_price_ex,
                                position_size=(last_entry_price or Decimal("0")) * (
                                    last_position_size or Decimal("0")
                                ),
                                entry_time=last_entry_time,
                                exit_reason=str(sig.exit_reason) if sig.exit_reason else "signal_exit",
                                window_minute=last_entry_minute,
                                cum_return_pct=cum_return * 100,
                                confidence=last_confidence,
                                fee_cost=last_fee_cost,
                                pnl=pnl_ex,
                            )
                        logger.info(
                            "swing_exit_executed",
                            reason=str(sig.exit_reason),
                            pnl=str(pnl_ex),
                            balance=str(paper_trader.balance),
                        )
                        has_open_position = False
                        last_entry_price = None
                        last_entry_side = None
                        last_entry_time = None
                        last_position_size = None

                # Wait for next candle (or shutdown)
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(), timeout=60.0,
                    )
                    break
                except TimeoutError:
                    continue

        finally:
            ws_connected = False
            await ws_feed.disconnect()
            await futures_feed.disconnect()
            self._remove_sentinel(_WS_CONNECTED_FILE)
            await cache.close()
            if ch_store is not None:
                await ch_store.disconnect()
            logger.info(
                "swing_loop_stopped",
                final_balance=str(paper_trader.balance),
                total_trades=trade_count,
                daily_pnl=str(daily_pnl),
            )


def run_bot(
    mode: str,
    strategy: str,
    config_dir: str = "config",
    env: str | None = None,
    fixed_bet_size: float = 0.0,
) -> int:
    """Run the trading bot.

    Args:
        mode: "paper" or "live".
        strategy: Strategy name to use.
        config_dir: Path to config directory.
        env: Environment name.
        fixed_bet_size: Fixed bet in USD (0 = use Kelly sizing).

    Returns:
        Exit code (0 = success).
    """
    config = ConfigLoader(config_dir=config_dir, env=env)
    config.load()
    config.load_strategy(strategy)
    config.validate_ranges()

    orchestrator = BotOrchestrator(
        mode=mode, strategy_name=strategy, config=config,
        fixed_bet_size=fixed_bet_size,
    )
    return asyncio.run(orchestrator.start())


def run_backtest(
    strategy: str,
    config_dir: str = "config",
    env: str | None = None,
    data_dir: str = "data",
) -> int:
    """Run backtesting.

    Loads historical candle data from CSV, synthesises per-bar market
    states and orderbook snapshots, then runs the BacktestEngine with
    the selected strategy and risk manager.

    Args:
        strategy: Strategy name.
        config_dir: Path to config directory.
        env: Environment name.
        data_dir: Path to directory containing CSV data files.

    Returns:
        Exit code (0 = success, 1 = failure).
    """
    import math
    from datetime import datetime, timezone
    from pathlib import Path

    from src.backtesting.data_loader import DataLoader
    from src.backtesting.engine import BacktestEngine
    from src.backtesting.reporter import BacktestReporter
    from src.models.market import MarketState, OrderBookSnapshot
    from src.risk.kill_switch import KillSwitch
    from src.risk.risk_manager import RiskManager
    from src.strategies import registry

    # --- Config ---
    config = ConfigLoader(config_dir=config_dir, env=env)
    config.load()
    config.load_strategy(strategy)
    config.validate_ranges()

    logger.info("backtest_start", strategy=strategy)

    # --- Load data ---
    loader = DataLoader()
    data_path = Path(data_dir)
    csv_files = sorted(data_path.glob("*.csv"))
    if not csv_files:
        logger.error("backtest_no_data", data_dir=data_dir)
        return 1

    # Use the first available CSV file (typically btc_15m_backtest.csv)
    csv_file = csv_files[0]
    try:
        candles = loader.load_csv(csv_file)
    except Exception:
        logger.exception("backtest_data_load_failed", path=str(csv_file))
        return 1

    if not candles:
        logger.error("backtest_empty_data", path=str(csv_file))
        return 1

    logger.info("backtest_data_loaded", path=str(csv_file), candles=len(candles))

    # --- Synthesise market states and orderbooks from candle data ---
    market_states: list[MarketState] = []
    orderbooks: list[OrderBookSnapshot] = []
    sigmoid_sensitivity = 0.07

    for i, candle in enumerate(candles):
        # Compute cumulative return from the first candle's open
        cum_return = float((candle.close - candles[0].open) / candles[0].open)
        yes_prob = 1.0 / (1.0 + math.exp(-cum_return / sigmoid_sensitivity))
        yes_price = Decimal(str(round(yes_prob, 6)))
        no_price = Decimal("1") - yes_price

        market_states.append(
            MarketState(
                market_id="backtest",
                yes_price=yes_price,
                no_price=no_price,
                time_remaining_seconds=max(0, (14 - (i % 15)) * 60),
            )
        )
        orderbooks.append(
            OrderBookSnapshot(
                timestamp=candle.timestamp,
                market_id="backtest",
            )
        )

    # --- Create strategy and risk manager ---
    # Ensure strategy modules are imported for registration
    import contextlib

    with contextlib.suppress(ImportError):
        import src.strategies.false_sentiment  # noqa: F401
    with contextlib.suppress(ImportError):
        import src.strategies.reversal_catcher  # noqa: F401
    with contextlib.suppress(ImportError):
        import src.strategies.singularity  # noqa: F401

    try:
        strat = registry.create(strategy, config)
    except KeyError:
        logger.error("backtest_unknown_strategy", strategy=strategy)
        return 1

    max_position_pct = Decimal(str(config.get("risk.max_position_pct", 0.02)))
    max_daily_drawdown_pct = Decimal(str(config.get("risk.max_daily_drawdown_pct", 0.05)))
    kill_switch = KillSwitch(max_daily_drawdown_pct)
    risk_mgr = RiskManager(
        max_position_pct=max_position_pct,
        max_daily_drawdown_pct=max_daily_drawdown_pct,
        kill_switch=kill_switch,
    )

    initial_balance = Decimal(str(config.get("paper.initial_balance", 10000)))
    engine = BacktestEngine(initial_balance=initial_balance)

    # --- Run backtest ---
    try:
        result = engine.run(
            candles=candles,
            market_states=market_states,
            orderbooks=orderbooks,
            strategy=strat,
            risk_manager=risk_mgr,
        )
    except Exception:
        logger.exception("backtest_engine_failed")
        return 1

    # --- Report results ---
    reporter = BacktestReporter()
    result_dict = result.model_dump()
    reporter.print_summary(result_dict)

    logger.info("backtest_complete", strategy=strategy, trades=len(result.trades))
    return 0
