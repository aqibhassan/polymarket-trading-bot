"""MVHE Bot orchestrator — wires all components together.

This module is the main event loop that subscribes to data feeds,
generates signals, checks risk, and executes trades.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import os
import signal
from datetime import UTC
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

# ExitManager — available but exit logic is handled by strategy/orchestrator

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
            import src.strategies.false_sentiment
        with contextlib.suppress(ImportError):
            import src.strategies.reversal_catcher
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
    # Position state persistence (survives Docker restart)
    # ------------------------------------------------------------------

    _POS_KEY = "mvhe:open_position"

    @staticmethod
    async def _save_position(
        redis_conn: Any,
        *,
        market_id: str,
        token_id: str,
        condition_id: str,
        entry_price: Decimal,
        entry_side: str,
        position_size: Decimal,
        fee_cost: Decimal,
    ) -> None:
        """Persist open position state to Redis."""
        import json as _json

        data = _json.dumps({
            "market_id": market_id,
            "token_id": token_id,
            "condition_id": condition_id,
            "entry_price": str(entry_price),
            "entry_side": entry_side,
            "position_size": str(position_size),
            "fee_cost": str(fee_cost),
        })
        await redis_conn.set(BotOrchestrator._POS_KEY, data)

    @staticmethod
    async def _clear_position(redis_conn: Any) -> None:
        """Remove persisted position state from Redis."""
        await redis_conn.delete(BotOrchestrator._POS_KEY)

    @staticmethod
    async def _load_position(redis_conn: Any) -> dict[str, Any] | None:
        """Load persisted position state from Redis, if any."""
        import json as _json

        raw = await redis_conn.get(BotOrchestrator._POS_KEY)
        if raw is None:
            return None
        try:
            return _json.loads(raw)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Swing trading mode (1m candles -> 15m windows)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_yes_price(cum_return: float) -> Decimal:
        """Model YES token price using sigmoid of cumulative BTC return.

        A positive cumulative return (BTC up) drives YES price higher (green
        candle more likely).  Sensitivity matches the swing backtest default.
        """
        try:
            prob = 1.0 / (1.0 + math.exp(-cum_return / _SIGMOID_SENSITIVITY))
        except OverflowError:
            # Extreme negative return → exp overflow → prob approaches 0
            prob = 0.0 if cum_return < 0 else 1.0
        # Clamp to avoid degenerate 0.0 / 1.0 prices
        prob = max(0.01, min(0.99, prob))
        return Decimal(str(round(prob, 6)))

    @staticmethod
    def _compute_confidence_discount(
        signal_conf: float,
        min_discount_pct: float = 0.005,
        max_discount_pct: float = 0.05,
    ) -> float:
        """Compute entry price discount percentage from signal confidence.

        Maps confidence linearly:
          - conf=0.50 -> max_discount_pct (e.g. 5%)
          - conf=1.00 -> min_discount_pct (e.g. 0.5%)

        Higher confidence means we are more willing to pay near market price,
        so the discount is smaller.  Lower confidence demands a better entry
        (bigger discount).  If the GTC order doesn't fill, we skip the window.

        Returns:
            Discount as a fraction (0.005 to 0.05 by default).
        """
        conf_range = 0.5  # from 0.5 to 1.0
        discount = max_discount_pct - (
            (signal_conf - 0.5)
            * (max_discount_pct - min_discount_pct)
            / conf_range
        )
        return max(min_discount_pct, min(max_discount_pct, discount))

    @staticmethod
    def _apply_confidence_discount(
        entry_price: Decimal,
        discount_pct: float,
    ) -> Decimal:
        """Apply a discount to an entry price and round to Polymarket tick.

        Polymarket tick size is 0.01 (1 cent).  We round *down* (floor) to
        ensure our limit price is always at or below the intended discount.

        Returns:
            Discounted price, floored to 0.01 tick, minimum 0.01.
        """
        import decimal as _decimal

        discount_factor = Decimal("1") - Decimal(str(round(discount_pct, 4)))
        discounted = entry_price * discount_factor
        # Floor to Polymarket tick (0.01) — always round DOWN
        discounted = (
            (discounted * 100).to_integral_value(rounding=_decimal.ROUND_FLOOR)
            / 100
        )
        return max(discounted, Decimal("0.01"))

    @staticmethod
    def _select_order_type(
        clob_last_trade: Decimal | None,
        clob_spread: Decimal,
        minute_in_window: int,
        *,
        fok_spread_threshold: Decimal = Decimal("0.30"),
        fok_late_minute: int = 11,
    ) -> "OrderType":
        """Choose GTC (maker) or FOK (taker) based on CLOB state.

        FOK when:
          - Real trades exist AND spread < threshold (liquid market)
          - OR minute >= late threshold (last chance, cross the spread)
        GTC otherwise (rest as maker when market is illiquid).
        """
        from src.models.order import OrderType

        has_real_trades = clob_last_trade is not None
        spread_tight = clob_spread < fok_spread_threshold
        late_window = minute_in_window >= fok_late_minute

        if (has_real_trades and spread_tight) or late_window:
            return OrderType.FOK
        return OrderType.GTC

    @staticmethod
    def _window_start_ts(epoch_s: int) -> int:
        """Align an epoch timestamp to the enclosing 15-minute boundary."""
        return epoch_s - (epoch_s % 900)

    async def _publish_state(
        self,
        cache: RedisCache,
        *,
        start_time: Any,
        current_balance: Any,
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
        pending_gtc: bool = False,
    ) -> None:
        """Publish bot state to Redis for dashboard consumption."""
        from datetime import datetime

        now = datetime.now(tz=UTC)
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
            balance = current_balance
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
                    "status": "filled",
                }, ttl=300)
            elif pending_gtc and active_market is not None:
                await cache.set("bot:position", {
                    "market_id": active_market.market_id,
                    "side": entry_side or "YES",
                    "entry_price": str(entry_price or "0"),
                    "size": str(position_size or "0"),
                    "entry_time": str(entry_time or now.isoformat()),
                    "status": "gtc_pending",
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
        signal_details: str = "",
        clob_entry_price: Any = None,
        sigmoid_entry_price: Any = None,
    ) -> str:
        """Record a completed trade to ClickHouse for dashboard display.

        Returns:
            The trade_id of the recorded trade.
        """
        import uuid
        from datetime import datetime

        exit_time = datetime.now(tz=UTC)
        trade_id = f"trade-{uuid.uuid4().hex[:8]}"
        # Compute bet-to-win ratio: risking entry_price to win (1 - entry_price)
        ep = float(entry_price) if entry_price else 0
        bet_to_win = ep / (1 - ep) if 0 < ep < 1 else 0.0
        trade_data = {
            "trade_id": trade_id,
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
            "signal_details": signal_details,
            "clob_entry_price": clob_entry_price or Decimal("0"),
            "sigmoid_entry_price": sigmoid_entry_price or Decimal("0"),
            "bet_to_win_ratio": bet_to_win,
        }
        try:
            await ch_store.insert_trade(trade_data)
            await ch_store.insert_audit_event({
                "order_id": f"ord-{uuid.uuid4().hex[:8]}",
                "event_type": "TRADE_CLOSE",
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
        return trade_id

    async def _run_swing_loop(self) -> None:
        """Core loop for 1m swing trading mode.

        Subscribes to Binance 1m BTCUSDT klines, collects candles into
        15-minute windows, runs the momentum_confirmation strategy each minute,
        and force-closes positions at the window boundary.

        Uses Kelly position sizing and real Portfolio drawdown tracking.
        """
        from datetime import datetime

        from src.data.binance_futures_ws import BinanceFuturesWSFeed
        from src.data.binance_ws import BinanceWSFeed
        from src.data.clickhouse_store import ClickHouseStore
        from src.data.polymarket_scanner import PolymarketScanner
        from src.data.polymarket_ws import PolymarketWSFeed
        from src.execution.bridge import ExecutionBridge
        from src.execution.paper_trader import PaperTrader
        from src.models.market import Candle, MarketState, OrderBookSnapshot, Side
        from src.models.order import OrderSide, OrderStatus, OrderType
        from src.models.signal import SignalType
        from src.risk.cost_calculator import CostCalculator
        from src.risk.kill_switch import KillSwitch
        from src.risk.portfolio import Portfolio
        from src.risk.position_sizer import PositionSize, PositionSizer
        from src.risk.risk_manager import RiskManager

        # --- Config ---
        max_position_pct = Decimal(
            str(self._config.get("risk.max_position_pct", 0.05))
        )
        min_position_pct = Decimal(
            str(self._config.get("risk.min_position_pct", 0.01))
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

        # --- Live safety config ---
        balance_floor_pct = Decimal(
            str(self._config.get("risk.live_safety.balance_floor_pct", 0))
        )
        max_consecutive_losses = int(
            self._config.get("risk.live_safety.max_consecutive_losses", 0)
        )
        ramp_enabled = bool(
            self._config.get("risk.live_safety.ramp_enabled", False)
        )
        ramp_day_1_max_pct = Decimal(
            str(self._config.get("risk.live_safety.ramp_day_1_max_pct", 0.01))
        )
        ramp_day_7_max_pct = Decimal(
            str(self._config.get("risk.live_safety.ramp_day_7_max_pct", 0.02))
        )
        ramp_day_30_max_pct = Decimal(
            str(self._config.get("risk.live_safety.ramp_day_30_max_pct", 0.05))
        )

        # Execution config
        order_timeout = float(
            self._config.get("execution.order_timeout_seconds", 0)
        )
        order_max_retries = int(
            self._config.get("execution.order_max_retries", 0)
        )
        cb_max_failures = int(
            self._config.get("execution.circuit_breaker_max_failures", 5)
        )
        cb_cooldown = float(
            self._config.get("execution.circuit_breaker_cooldown_seconds", 60)
        )

        # Confidence-based limit pricing config
        use_confidence_discount = bool(
            self._config.get("execution.pricing.use_confidence_discount", True)
        )
        pricing_min_discount_pct = float(
            self._config.get("execution.pricing.min_discount_pct", 0.005)
        )
        pricing_max_discount_pct = float(
            self._config.get("execution.pricing.max_discount_pct", 0.05)
        )

        # CLOB pricing config
        use_clob_pricing = bool(
            self._config.get("execution.pricing.use_clob_pricing", True)
        )
        max_clob_entry_price = Decimal(
            str(self._config.get("execution.pricing.max_clob_entry_price", 0.80))
        )
        max_clob_spread = Decimal(
            str(self._config.get("execution.pricing.max_clob_spread", 0.50))
        )

        # FOK (Fill-Or-Kill) taker execution config
        fok_spread_threshold = Decimal(
            str(self._config.get("execution.pricing.fok_spread_threshold", 0.30))
        )
        fok_late_minute = int(
            self._config.get("execution.pricing.fok_late_minute_threshold", 11)
        )
        fok_max_entry_price = Decimal(
            str(self._config.get("execution.pricing.fok_max_entry_price", 0.85))
        )

        # Confidence-adaptive max entry price config
        conf_adaptive = bool(
            self._config.get("execution.pricing.confidence_adaptive_pricing", False)
        )
        conf_adaptive_base = Decimal(
            str(self._config.get("execution.pricing.confidence_adaptive_base", 0.85))
        )
        conf_adaptive_scale = Decimal(
            str(self._config.get("execution.pricing.confidence_adaptive_scale", 0.50))
        )
        conf_adaptive_cap = Decimal(
            str(self._config.get("execution.pricing.confidence_adaptive_cap", 0.92))
        )

        # Per-signal confidence Kelly sizing config
        use_signal_kelly = bool(
            self._config.get("execution.pricing.use_signal_confidence_kelly", False)
        )
        kelly_conf_floor = Decimal(
            str(self._config.get("execution.pricing.kelly_confidence_floor", 0.80))
        )
        kelly_conf_cap = Decimal(
            str(self._config.get("execution.pricing.kelly_confidence_cap", 0.95))
        )

        # --- Component setup ---
        kline_ws_url = os.environ.get(
            "BINANCE_WS_URL",
            "wss://fstream.binance.com/ws/btcusdt@kline_1m",
        )
        ws_feed = BinanceWSFeed(
            ws_url=kline_ws_url,
            symbol="BTCUSDT",
            interval="1m",
        )
        futures_feed = BinanceFuturesWSFeed(
            ws_url="wss://fstream.binance.com/ws/btcusdt@aggTrade",
            symbol="BTCUSDT",
        )
        scanner = PolymarketScanner()
        poly_ws_url = str(self._config.get(
            "polymarket.ws_url",
            "wss://ws-subscriptions-clob.polymarket.com/ws/market",
        ))
        poly_ws_feed = PolymarketWSFeed(ws_url=poly_ws_url)
        cache = RedisCache()
        redis_conn = await cache._get_redis()
        # Clear stale bot state keys from prior run so dashboard
        # never shows outdated data before fresh publishes arrive.
        for key in await redis_conn.keys("mvhe:bot:*"):
            await redis_conn.delete(key)
        kill_switch = KillSwitch(
            max_daily_drawdown_pct,
            redis_client=redis_conn,
            async_redis=True,
            max_consecutive_losses=max_consecutive_losses,
        )
        await kill_switch.async_load_state()
        await kill_switch.async_load_consecutive_losses()
        risk_mgr = RiskManager(
            max_position_pct=max_position_pct,
            max_daily_drawdown_pct=max_daily_drawdown_pct,
            kill_switch=kill_switch,
            balance_floor_pct=balance_floor_pct,
            initial_balance=initial_balance,
        )
        position_sizer = PositionSizer(
            max_position_pct=max_position_pct,
            min_position_pct=min_position_pct,
            kelly_multiplier=kelly_multiplier,
        )
        cost_calculator = CostCalculator()
        # Portfolio is created after live balance sync (below) to avoid
        # peak_equity being set to the paper default ($10k) before the
        # real live balance ($99) is known — which causes a false 99% drawdown.

        from src.execution.circuit_breaker import CircuitBreaker

        circuit_breaker = CircuitBreaker(
            max_failures=cb_max_failures,
            cooldown_seconds=cb_cooldown,
        )

        # --- Paper vs Live trader setup ---
        live_trader: Any = None
        paper_trader: Any = None
        if self._mode == "live":
            from src.execution.polymarket_signer import PolymarketLiveTrader

            live_trader = PolymarketLiveTrader(
                clob_url=str(self._config.get(
                    "polymarket.clob_url", "https://clob.polymarket.com",
                )),
                chain_id=int(self._config.get("polymarket.chain_id", 137)),
            )
            live_balance = await live_trader.get_balance()
            if live_balance is not None and live_balance > 0:
                initial_balance = Decimal(str(live_balance))
                risk_mgr._initial_balance = initial_balance
                logger.info("live_balance_synced", balance=str(initial_balance))
            elif live_balance is not None and live_balance <= 0:
                logger.critical("live_zero_balance", balance=str(live_balance))
                return
            # Enforce minimum starting balance for live trading
            min_starting = Decimal(
                str(self._config.get("risk.min_starting_balance", 10))
            )
            if initial_balance < min_starting:
                logger.critical(
                    "live_below_min_starting_balance",
                    balance=str(initial_balance),
                    min_required=str(min_starting),
                )
                return
            # Auto-confirm: CLI already gates on --auto-confirm +
            # MVHE_LIVE_AUTO_CONFIRM=true; bridge needs skip_confirmation
            # so it doesn't call input() in non-TTY Docker containers.
            auto_confirm = (
                os.environ.get("MVHE_LIVE_AUTO_CONFIRM", "").lower() == "true"
            )
            bridge = ExecutionBridge(
                mode=self._mode,
                live_trader=live_trader,
                circuit_breaker=circuit_breaker,
                order_timeout_seconds=order_timeout,
                order_max_retries=order_max_retries,
                skip_confirmation=auto_confirm,
            )
        else:
            paper_trader = PaperTrader(initial_balance=initial_balance)
            bridge = ExecutionBridge(
                mode=self._mode,
                paper_trader=paper_trader,
                circuit_breaker=circuit_breaker,
                order_timeout_seconds=order_timeout,
                order_max_retries=order_max_retries,
            )

        # Publish initial balance to Redis so dashboard shows correct
        # data during the startup sweep (which can block for minutes).
        try:
            await cache.set("bot:balance", {
                "balance": str(initial_balance),
                "initial_balance": str(initial_balance),
                "pnl": "0",
                "pnl_pct": 0.0,
            }, ttl=600)
        except Exception:
            logger.debug("early_balance_publish_failed", exc_info=True)

        # Auto-claim redeemer (live mode only)
        redeemer = None
        if self._mode == "live":
            try:
                from src.execution.polymarket_redeemer import PolymarketRedeemer
                redeemer = PolymarketRedeemer()
                matic = await redeemer.check_matic_balance()
                logger.info("redeemer.matic_balance", matic=f"{matic:.4f}")
                if matic < 0.01:
                    logger.warning("redeemer.low_matic", matic=f"{matic:.4f}")

                # Sweep all past unclaimed settled positions at startup
                if matic >= 0.01:
                    try:
                        sweep_hashes = await redeemer.sweep_all_settled(
                            lookback_hours=24,
                        )
                        if sweep_hashes:
                            logger.info(
                                "startup_sweep.redeemed",
                                count=len(sweep_hashes),
                                tx_hashes=sweep_hashes[:5],
                            )
                            # Refresh balance after sweep
                            if live_trader is not None:
                                await asyncio.sleep(10)
                                synced = await live_trader.get_balance()
                                if synced is not None and synced > 0:
                                    initial_balance = synced
                                    logger.info(
                                        "startup_sweep.balance_refreshed",
                                        balance=str(initial_balance),
                                    )
                    except Exception:
                        logger.warning("startup_sweep.failed", exc_info=True)
            except Exception:
                logger.warning("redeemer.init_failed", exc_info=True)

        # Portfolio is created AFTER initial_balance is finalized (live or paper)
        # so peak_equity starts at the correct value.
        portfolio = Portfolio()
        portfolio.update_equity(initial_balance)

        # Cached balance for both modes
        cached_balance = initial_balance
        last_balance_sync = datetime.now(tz=UTC)
        ch_store: ClickHouseStore | None = None
        try:
            ch_store = ClickHouseStore()
            await ch_store.connect()
        except Exception:
            logger.warning("clickhouse_connect_failed", exc_info=True)
            ch_store = None

        await ws_feed.connect()
        await futures_feed.connect()
        try:
            await poly_ws_feed.connect()
        except Exception:
            logger.warning("polymarket_ws_connect_failed", exc_info=True)
            # Bot continues — CLOB data will fallback to REST orderbook
        ws_connected = True
        self._touch_sentinel(_WS_CONNECTED_FILE)

        # --- Position ramp schedule for live ---
        live_start_date = datetime.now(tz=UTC).date()
        if self._mode == "live" and ramp_enabled:
            # Try to load live_start_date from Redis; persist if first run
            try:
                stored = await redis_conn.get("mvhe:live_start_date")
                if stored is not None:
                    raw = stored if isinstance(stored, str) else stored.decode()
                    from datetime import date as _date_cls
                    live_start_date = _date_cls.fromisoformat(raw)
                else:
                    await redis_conn.set(
                        "mvhe:live_start_date",
                        live_start_date.isoformat(),
                    )
            except Exception:
                logger.warning("ramp_live_start_date_load_failed", exc_info=True)

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
        start_time = datetime.now(tz=UTC)
        last_confidence: float = 0.0
        last_entry_minute: int = 0
        last_cum_return_entry: float = 0.0
        last_fee_cost = Decimal("0")
        last_token_id: str | None = None
        last_condition_id: str | None = None
        pending_gtc_oid: str | None = None  # Exchange OID of resting GTC
        pending_gtc_internal_id: str | None = None
        pending_gtc_position: bool = False  # True while GTC resting, not yet MATCHED
        gtc_poll_counter: int = 0  # Throttle mid-tick GTC status polling
        last_tick_yes_price = Decimal("0.5")
        last_market_id: str | None = None
        last_reset_date = datetime.now(tz=UTC).date()
        last_signal_details_str: str = ""
        last_clob_entry_price: Decimal | None = None  # CLOB price at signal time
        last_sigmoid_entry_price: Decimal | None = None  # Sigmoid price at signal time
        current_subscribed_yes: str | None = None  # YES token subscribed on poly WS
        current_subscribed_no: str | None = None  # NO token subscribed on poly WS
        last_clob_snapshot_minute: int = -1  # Track per-minute CLOB snapshots
        window_activity_published = False  # one event per 15-min window
        window_last_skip_eval: dict[str, Any] = {}  # accumulate last skip reason

        # Recover orphaned position from Redis (survives Docker restart)
        recovered_pos = await self._load_position(redis_conn)
        if recovered_pos is not None:
            has_open_position = True
            last_entry_price = Decimal(recovered_pos["entry_price"])
            last_entry_side = recovered_pos["entry_side"]
            last_position_size = Decimal(recovered_pos["position_size"])
            last_token_id = recovered_pos.get("token_id")
            last_market_id = recovered_pos.get("market_id")
            last_condition_id = recovered_pos.get("condition_id")
            last_fee_cost = Decimal(recovered_pos.get("fee_cost", "0"))
            last_entry_time = datetime.now(tz=UTC)
            logger.warning(
                "position_recovered_from_redis",
                market_id=last_market_id,
                entry_price=str(last_entry_price),
                side=last_entry_side,
                size=str(last_position_size),
            )

        logger.info(
            "swing_loop_started",
            mode=self._mode,
            initial_balance=str(initial_balance),
            kelly_multiplier=str(kelly_multiplier),
            max_position_pct=str(max_position_pct),
            min_position_pct=str(min_position_pct),
        )

        try:
            while self._running:
                self._touch_sentinel(_HEARTBEAT_FILE)

                # Daily stats reset at UTC midnight
                now_utc = datetime.now(tz=UTC)
                if now_utc.date() != last_reset_date:
                    daily_pnl = Decimal("0")
                    trade_count = 0
                    win_count = 0
                    loss_count = 0
                    # Reset portfolio drawdown tracking for new day
                    portfolio = Portfolio()
                    portfolio.update_equity(cached_balance)
                    last_reset_date = now_utc.date()
                    logger.info("daily_stats_reset", date=str(last_reset_date))

                # Check WS connectivity — skip trading on stale data
                ws_connected = ws_feed.is_connected
                if not ws_connected:
                    self._remove_sentinel(_WS_CONNECTED_FILE)
                    logger.warning("swing_ws_disconnected")
                    await asyncio.sleep(self._tick_interval)
                    continue
                self._touch_sentinel(_WS_CONNECTED_FILE)

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
                    # Check if pending GTC was filled before recording PnL
                    if pending_gtc_oid and bridge.mode == "live" and bridge._live_trader:
                        try:
                            resp = bridge._live_trader._clob_client.get_order(pending_gtc_oid)
                            gtc_status = resp.get("status", "") if isinstance(resp, dict) else ""
                            if gtc_status == "MATCHED":
                                # GTC filled — NOW we have a real position.
                                has_open_position = True
                                trade_count += 1
                                # Extract actual fill size from CLOB response
                                raw_matched = resp.get("size_matched", resp.get("sizeMatched", "0"))
                                actual_matched = Decimal(str(raw_matched)) if raw_matched else Decimal("0")
                                raw_original = resp.get("original_size", resp.get("originalSize", "0"))
                                original_size = Decimal(str(raw_original)) if raw_original else last_position_size
                                if actual_matched > 0:
                                    last_position_size = actual_matched
                                    # Recalculate fee for actual fill
                                    actual_notional = actual_matched * last_entry_price
                                    last_fee_cost = CostCalculator.polymarket_fee(
                                        actual_notional, last_entry_price,
                                    )
                                logger.info(
                                    "gtc_filled_at_settlement",
                                    exchange_oid=pending_gtc_oid,
                                    requested_size=str(original_size),
                                    actual_fill=str(actual_matched),
                                    fill_pct=f"{float(actual_matched / original_size * 100):.1f}%" if original_size > 0 else "N/A",
                                )
                            else:
                                logger.info(
                                    "gtc_not_filled_at_settlement",
                                    exchange_oid=pending_gtc_oid,
                                    status=gtc_status,
                                )
                                # Position never opened — clear tracking
                                has_open_position = False
                        except Exception:
                            logger.debug("gtc_settlement_check_failed", exc_info=True)
                            has_open_position = False
                        pending_gtc_oid = None
                        pending_gtc_internal_id = None
                        pending_gtc_position = False

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
                            final_cum_return = 0.0  # fallback: no data
                            logger.warning(
                                "swing_window_close_fallback",
                                fallback_cum_return=0.0,
                                window_open_price=str(window_open_price),
                                has_candles=bool(window_candles),
                                msg="using stale cum_return — window_candles or open_price unavailable",
                            )

                        winning_side = "YES" if final_cum_return >= 0 else "NO"
                        if last_entry_side == winning_side:
                            exit_price_wb = Decimal("1")  # correct prediction
                        else:
                            exit_price_wb = Decimal("0")  # wrong prediction
                        # Submit sell to update PaperTrader balance.
                        # In live mode, settlement happens on-chain automatically
                        # (tokens redeem at $1/$0) — skip CLOB sell.
                        if bridge.mode != "live":
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
                        else:
                            logger.info(
                                "live_settlement_skip_clob_sell",
                                market_id=close_market_id,
                                exit_price=str(exit_price_wb),
                                reason="redeeming on-chain via CTF",
                            )
                            # Auto-claim: redeem settled tokens on-chain (only winners)
                            if exit_price_wb == Decimal("1"):
                                if redeemer is not None and last_condition_id and last_token_id:
                                    # UMA oracle takes 5-15 min to resolve after window close.
                                    # Schedule background redemption with delay + retries instead
                                    # of blocking the trading loop with an immediate (likely failing) call.
                                    _cid = last_condition_id
                                    _tid = last_token_id

                                    async def _background_redeem(
                                        r: object, cid: str, tid: str,
                                    ) -> None:
                                        delays = [60, 120, 300]  # 1 min, 2 min, 5 min
                                        for attempt, delay in enumerate(delays):
                                            await asyncio.sleep(delay)
                                            try:
                                                tx = await r.redeem_positions(  # type: ignore[union-attr]
                                                    condition_id=cid,
                                                    token_id=tid,
                                                )
                                                if tx:
                                                    logger.info(
                                                        "live_redeem_success_bg",
                                                        tx_hash=tx,
                                                        condition_id=cid[:16],
                                                        attempt=attempt,
                                                    )
                                                    return
                                            except Exception:
                                                logger.debug(
                                                    "live_redeem_retry_failed",
                                                    attempt=attempt,
                                                    condition_id=cid[:16],
                                                    exc_info=True,
                                                )
                                        logger.info(
                                            "live_redeem_deferred_to_sweep",
                                            condition_id=cid[:16],
                                            reason="all retries failed, startup sweep will handle",
                                        )

                                    asyncio.create_task(_background_redeem(redeemer, _cid, _tid))
                                    logger.info(
                                        "live_redeem_scheduled",
                                        condition_id=last_condition_id[:16],
                                        token_id=last_token_id[:16],
                                        delays="60s,120s,300s",
                                    )
                            else:
                                logger.info(
                                    "live_loss_no_redeem",
                                    exit_price=str(exit_price_wb),
                                    reason="losing position, tokens worthless",
                                )
                        # P&L with fee deduction (entry + exit)
                        gross_pnl_wb = (exit_price_wb - last_entry_price) * (
                            last_position_size or Decimal("0")
                        )
                        exit_notional_wb = exit_price_wb * (last_position_size or Decimal("0"))
                        exit_fee_wb = CostCalculator.polymarket_fee(exit_notional_wb, exit_price_wb)
                        pnl_wb = gross_pnl_wb - last_fee_cost - exit_fee_wb
                        daily_pnl += pnl_wb

                        # Sync balance after trade
                        if self._mode == "live" and live_trader is not None:
                            try:
                                synced = await live_trader.get_balance()
                                if synced is not None and synced > 0:
                                    cached_balance = Decimal(str(synced))
                                    last_balance_sync = datetime.now(tz=UTC)
                            except Exception:
                                logger.debug("post_trade_balance_sync_failed", exc_info=True)
                        elif paper_trader is not None:
                            cached_balance = paper_trader.balance

                        portfolio.update_equity(cached_balance)

                        is_win_wb = pnl_wb >= 0
                        if is_win_wb:
                            win_count += 1
                        else:
                            loss_count += 1
                        await kill_switch.async_record_trade_result(is_win=is_win_wb)
                        trade_id_wb = ""
                        if ch_store is not None:
                            trade_id_wb = await self._record_completed_trade(
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
                                signal_details=last_signal_details_str,
                                clob_entry_price=last_clob_entry_price,
                                sigmoid_entry_price=last_sigmoid_entry_price,
                            )
                        # Publish last trade to Redis for dashboard
                        try:
                            await cache.set("bot:last_trade", {
                                "trade_id": trade_id_wb,
                                "direction": last_entry_side or "YES",
                                "entry_price": str(last_entry_price),
                                "exit_price": str(exit_price_wb),
                                "pnl": str(pnl_wb),
                                "exit_reason": "settlement",
                                "window_minute": last_entry_minute,
                                "confidence": last_confidence,
                                "timestamp": datetime.now(tz=UTC).isoformat(),
                            }, ttl=300)
                        except Exception:
                            logger.debug("last_trade_publish_failed", exc_info=True)
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
                        last_clob_entry_price = None
                        last_sigmoid_entry_price = None
                        # Clear persisted position from Redis
                        try:
                            await self._clear_position(redis_conn)
                        except Exception:
                            logger.debug("position_clear_failed", exc_info=True)

                    # Always clear active_market and stale position metadata
                    # at window boundary to prevent state leak between windows
                    active_market = None
                    last_token_id = None
                    last_market_id = None
                    last_condition_id = None

                    # Publish window summary if bot didn't trade this window
                    if not window_activity_published and window_last_skip_eval:
                        try:
                            import uuid as _uuid_mod
                            summary_event = {
                                "id": str(_uuid_mod.uuid4())[:8],
                                "timestamp": datetime.now(tz=UTC).isoformat(),
                                "minute": window_last_skip_eval.get("minute", 0),
                                "market_id": window_last_skip_eval.get("market_id", ""),
                                "outcome": "skip",
                                "reason": window_last_skip_eval.get("reason", ""),
                                "direction": window_last_skip_eval.get("direction", ""),
                                "confidence": window_last_skip_eval.get("confidence", 0),
                                "votes": window_last_skip_eval.get("votes", {}),
                                "detail": window_last_skip_eval.get("detail", ""),
                            }
                            await cache.push_to_list(
                                "bot:signal_activity", summary_event,
                                max_length=20, ttl=86400,
                            )
                            # Persist to ClickHouse for historical analysis
                            if ch_store is not None:
                                try:
                                    from datetime import datetime as _dt_parse
                                    await ch_store.insert_signal_evaluation({
                                        "eval_id": summary_event["id"],
                                        "timestamp": _dt_parse.fromisoformat(summary_event["timestamp"]),
                                        "strategy": self._strategy_name,
                                        "market_id": summary_event["market_id"],
                                        "minute": summary_event["minute"],
                                        "outcome": summary_event["outcome"],
                                        "reason": summary_event["reason"],
                                        "direction": summary_event.get("direction", ""),
                                        "confidence": summary_event.get("confidence", 0),
                                        "votes_yes": summary_event.get("votes", {}).get("yes", 0),
                                        "votes_no": summary_event.get("votes", {}).get("no", 0),
                                        "votes_neutral": summary_event.get("votes", {}).get("neutral", 0),
                                        "detail": summary_event.get("detail", ""),
                                    })
                                except Exception:
                                    logger.debug("clickhouse_signal_eval_skip_failed", exc_info=True)
                        except Exception:
                            logger.debug("signal_activity_window_summary_failed", exc_info=True)

                    window_candles = []
                    window_open_price = None
                    current_window_start = None
                    window_activity_published = False
                    window_last_skip_eval = {}
                    last_clob_snapshot_minute = -1
                    # Unsubscribe from old tokens at window boundary
                    for _old_tok in (current_subscribed_yes, current_subscribed_no):
                        if _old_tok:
                            try:
                                await poly_ws_feed.unsubscribe(_old_tok)
                            except Exception:
                                logger.debug("poly_ws_boundary_unsubscribe_failed", exc_info=True)
                    current_subscribed_yes = None
                    current_subscribed_no = None

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
                                condition_id=active_market.condition_id[:16] if active_market.condition_id else "empty",
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
                        window_start=datetime.fromtimestamp(w_start, tz=UTC).isoformat(),
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
                if window_open_price == 0:
                    cum_return = 0.0
                else:
                    cum_return = float(
                        (latest.close - window_open_price) / window_open_price
                    )
                yes_price = self._compute_yes_price(cum_return)
                last_tick_yes_price = yes_price

                # --- Position ramp override for live ---
                effective_max_position_pct = max_position_pct
                if self._mode == "live" and ramp_enabled:
                    days_live = (datetime.now(tz=UTC).date() - live_start_date).days
                    if days_live < 7:
                        effective_max_position_pct = min(
                            max_position_pct, ramp_day_1_max_pct,
                        )
                    elif days_live < 30:
                        effective_max_position_pct = min(
                            max_position_pct, ramp_day_7_max_pct,
                        )
                    else:
                        effective_max_position_pct = min(
                            max_position_pct, ramp_day_30_max_pct,
                        )

                # --- Periodic balance sync (every 5 min) for live ---
                now_sync = datetime.now(tz=UTC)
                if (
                    self._mode == "live"
                    and live_trader is not None
                    and (now_sync - last_balance_sync).total_seconds() >= 300
                ):
                    try:
                        synced = await live_trader.get_balance()
                        if synced is not None and synced > 0:
                            cached_balance = Decimal(str(synced))
                            last_balance_sync = now_sync
                    except Exception:
                        logger.debug("periodic_balance_sync_failed", exc_info=True)

                logger.info(
                    "swing_tick",
                    minute=minute_in_window,
                    close=str(latest.close),
                    cum_return_pct=round(cum_return * 100, 4),
                    yes_price=str(yes_price),
                    balance=str(cached_balance),
                    daily_pnl=str(daily_pnl),
                    trades_today=trade_count,
                )

                # Publish state to Redis for dashboard
                await self._publish_state(
                    cache,
                    start_time=start_time,
                    current_balance=cached_balance,
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
                    pending_gtc=pending_gtc_position,
                )

                # --- Check kill switch ---
                if await kill_switch.async_check(daily_pnl, cached_balance):
                    logger.critical(
                        "swing_kill_switch_active",
                        daily_pnl=str(daily_pnl),
                        balance=str(cached_balance),
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
                # Skip signal processing if no real market found (empty
                # token_id would cause live order failures)
                if active_market is None:
                    logger.info("swing_skip_no_market", minute=minute_in_window)
                    await asyncio.sleep(self._tick_interval)
                    continue
                market_id = active_market.market_id
                yes_token_id = active_market.yes_token_id
                no_token_id = active_market.no_token_id

                # Subscribe to Polymarket WS for real-time CLOB prices (both YES + NO)
                _needs_rest_prime = False
                if yes_token_id and yes_token_id != current_subscribed_yes:
                    if current_subscribed_yes:
                        try:
                            await poly_ws_feed.unsubscribe(current_subscribed_yes)
                        except Exception:
                            logger.debug("poly_ws_unsubscribe_yes_failed", exc_info=True)
                    try:
                        await poly_ws_feed.subscribe(yes_token_id)
                        current_subscribed_yes = yes_token_id
                        _needs_rest_prime = True
                    except Exception:
                        logger.debug("poly_ws_subscribe_yes_failed", exc_info=True)
                if no_token_id and no_token_id != current_subscribed_no:
                    if current_subscribed_no:
                        try:
                            await poly_ws_feed.unsubscribe(current_subscribed_no)
                        except Exception:
                            logger.debug("poly_ws_unsubscribe_no_failed", exc_info=True)
                    try:
                        await poly_ws_feed.subscribe(no_token_id)
                        current_subscribed_no = no_token_id
                        _needs_rest_prime = True
                    except Exception:
                        logger.debug("poly_ws_subscribe_no_failed", exc_info=True)

                # REST prime: fetch CLOB data once on new subscription
                # to fill timing gap before WS stream kicks in.
                if _needs_rest_prime and bridge.mode == "live" and bridge._live_trader:
                    _clob_client = bridge._live_trader._clob_client
                    for _tid, _label in [(yes_token_id, "yes"), (no_token_id, "no")]:
                        if not _tid:
                            continue
                        try:
                            _rest_ltp = await asyncio.to_thread(
                                _clob_client.get_last_trade_price, _tid,
                            )
                            _rest_mid = await asyncio.to_thread(
                                _clob_client.get_midpoint, _tid,
                            )
                            _rest_spread = await asyncio.to_thread(
                                _clob_client.get_spread, _tid,
                            )
                            logger.info(
                                "rest_clob_prime",
                                token=_label,
                                last_trade=str(_rest_ltp),
                                midpoint=str(_rest_mid),
                                spread=str(_rest_spread),
                            )
                            poly_ws_feed.seed_rest_data(
                                _tid, _rest_ltp, _rest_mid, _rest_spread,
                            )
                        except Exception:
                            logger.info("rest_prime_failed", token=_label, exc_info=True)

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
                            timestamp=datetime.now(tz=UTC),
                            market_id=market_id,
                        )
                except Exception:
                    logger.debug("orderbook_fetch_failed", market_id=market_id, exc_info=True)
                    orderbook = OrderBookSnapshot(
                        timestamp=datetime.now(tz=UTC),
                        market_id=market_id,
                    )

                # --- Read CLOB state from Polymarket WS ---
                clob_best_ask: Decimal | None = None
                clob_best_bid: Decimal | None = None
                clob_midpoint: Decimal | None = None
                clob_last_trade: Decimal | None = None
                if use_clob_pricing and yes_token_id:
                    clob_state = poly_ws_feed.get_clob_state(yes_token_id)
                    # Only use CLOB data if fresh (< 960s old).
                    # Must cover the full 15-min window (900s) + startup buffer.
                    # On desert markets the WS sends one book snapshot on subscribe
                    # then goes silent; data from REST prime must persist throughout.
                    # State is cleared on unsubscribe between windows, so no risk
                    # of cross-window contamination.
                    clob_fresh = (
                        clob_state is not None
                        and clob_state.best_ask is not None
                        and clob_state.last_updated is not None
                        and (datetime.now(tz=UTC) - clob_state.last_updated).total_seconds() < 960
                    )
                    if clob_fresh:
                        assert clob_state is not None  # for type narrowing
                        clob_best_ask = clob_state.best_ask
                        clob_best_bid = clob_state.best_bid
                        clob_midpoint = clob_state.midpoint
                        clob_last_trade = clob_state.last_trade_price
                    elif orderbook.asks:
                        # Fallback: use REST orderbook best ask
                        clob_best_ask = orderbook.asks[0].price
                        clob_best_bid = orderbook.bids[0].price if orderbook.bids else None

                # --- Read NO token CLOB state for NO-direction trades ---
                no_clob_best_ask: Decimal | None = None
                no_clob_best_bid: Decimal | None = None
                no_clob_midpoint: Decimal | None = None
                no_clob_last_trade: Decimal | None = None
                if use_clob_pricing and no_token_id:
                    no_clob_state = poly_ws_feed.get_clob_state(no_token_id)
                    no_clob_fresh = (
                        no_clob_state is not None
                        and no_clob_state.best_ask is not None
                        and no_clob_state.last_updated is not None
                        and (datetime.now(tz=UTC) - no_clob_state.last_updated).total_seconds() < 960
                    )
                    if no_clob_fresh:
                        assert no_clob_state is not None
                        no_clob_best_ask = no_clob_state.best_ask
                        no_clob_best_bid = no_clob_state.best_bid
                        no_clob_midpoint = no_clob_state.midpoint
                        no_clob_last_trade = no_clob_state.last_trade_price

                # Log CLOB vs sigmoid comparison every tick
                if clob_best_ask is not None:
                    logger.info(
                        "clob_vs_sigmoid",
                        minute=minute_in_window,
                        sigmoid_price=str(yes_price),
                        clob_best_ask=str(clob_best_ask),
                        clob_best_bid=str(clob_best_bid),
                        clob_midpoint=str(clob_midpoint),
                        clob_last_trade=str(clob_last_trade),
                        delta=str(clob_best_ask - yes_price),
                    )

                # Log NO token CLOB state for visibility
                if no_clob_best_ask is not None or no_clob_last_trade is not None:
                    logger.info(
                        "no_clob_state",
                        minute=minute_in_window,
                        no_best_ask=str(no_clob_best_ask),
                        no_best_bid=str(no_clob_best_bid),
                        no_midpoint=str(no_clob_midpoint),
                        no_last_trade=str(no_clob_last_trade),
                    )

                # Store CLOB snapshot to ClickHouse (once per minute)
                if (
                    ch_store is not None
                    and clob_best_ask is not None
                    and minute_in_window != last_clob_snapshot_minute
                ):
                    last_clob_snapshot_minute = minute_in_window
                    try:
                        spread = (clob_best_ask - clob_best_bid) if clob_best_bid else Decimal("0")
                        await ch_store.insert_clob_snapshot({
                            "timestamp": datetime.now(tz=UTC),
                            "market_id": market_id,
                            "token_id": yes_token_id,
                            "best_bid": clob_best_bid or Decimal("0"),
                            "best_ask": clob_best_ask,
                            "midpoint": clob_midpoint or Decimal("0"),
                            "spread": spread,
                            "last_trade_price": clob_last_trade or Decimal("0"),
                            "sigmoid_price": yes_price,
                        })
                    except Exception:
                        logger.debug("clob_snapshot_insert_failed", exc_info=True)

                # --- Mid-tick GTC fill check (every 3rd tick ≈ 15s) ---
                # Detect fills, cancellations, or upgrade GTC→FOK when
                # liquidity appears mid-window.
                gtc_poll_counter += 1
                if (
                    pending_gtc_oid
                    and bridge.mode == "live"
                    and bridge._live_trader
                    and gtc_poll_counter % 3 == 0
                ):
                    try:
                        _gtc_resp = bridge._live_trader._clob_client.get_order(
                            pending_gtc_oid,
                        )
                        _gtc_st = (
                            _gtc_resp.get("status", "")
                            if isinstance(_gtc_resp, dict)
                            else ""
                        )
                        if _gtc_st == "MATCHED":
                            _raw = _gtc_resp.get(
                                "size_matched",
                                _gtc_resp.get("sizeMatched", "0"),
                            )
                            _actual = (
                                Decimal(str(_raw)) if _raw else last_position_size
                            )
                            if _actual and _actual > 0 and last_entry_price is not None:
                                last_position_size = _actual
                                last_fee_cost = CostCalculator.polymarket_fee(
                                    _actual * last_entry_price,
                                    last_entry_price,
                                )
                            has_open_position = True
                            pending_gtc_oid = None
                            pending_gtc_internal_id = None
                            pending_gtc_position = False
                            trade_count += 1
                            logger.info(
                                "gtc_filled_mid_tick",
                                minute=minute_in_window,
                                fill_size=str(_actual),
                            )
                        elif _gtc_st in ("CANCELLED", "EXPIRED"):
                            pending_gtc_oid = None
                            pending_gtc_internal_id = None
                            pending_gtc_position = False
                            logger.info(
                                "gtc_cancelled_mid_tick", status=_gtc_st,
                            )
                        else:
                            # Still resting — check if we should upgrade to FOK
                            # Use direction-aware spread for upgrade decision
                            if last_entry_side == "NO" and no_clob_best_ask is not None and no_clob_best_bid is not None:
                                _clob_sp = no_clob_best_ask - no_clob_best_bid
                            elif clob_best_ask is not None and clob_best_bid is not None:
                                _clob_sp = clob_best_ask - clob_best_bid
                            else:
                                _clob_sp = Decimal("1")
                            _new_type = self._select_order_type(
                                clob_last_trade=clob_last_trade,
                                clob_spread=_clob_sp,
                                minute_in_window=minute_in_window,
                                fok_spread_threshold=fok_spread_threshold,
                                fok_late_minute=fok_late_minute,
                            )
                            if _new_type == OrderType.FOK:
                                _cancel_id = (
                                    pending_gtc_internal_id or pending_gtc_oid
                                )
                                _cancelled = await bridge.cancel_order(_cancel_id)
                                if _cancelled and last_entry_side in ("YES", "NO"):
                                    # Price for FOK: use appropriate token side
                                    _fok_price: Decimal | None = None
                                    if last_entry_side == "YES":
                                        _fok_price = clob_best_ask
                                    elif last_entry_side == "NO":
                                        _fok_price = no_clob_best_ask or (
                                            Decimal("1") - clob_best_bid
                                            if clob_best_bid
                                            else None
                                        )
                                    # Apply confidence-adaptive max to mid-tick FOK upgrade
                                    _upgrade_fok_max = fok_max_entry_price
                                    if conf_adaptive and last_confidence > float(conf_adaptive_base):
                                        _up_boost = Decimal(str(last_confidence - float(conf_adaptive_base))) * conf_adaptive_scale
                                        _upgrade_fok_max = min(fok_max_entry_price + _up_boost, conf_adaptive_cap)
                                    if (
                                        _fok_price is not None
                                        and _fok_price <= _upgrade_fok_max
                                        and last_token_id
                                        and last_market_id
                                    ):
                                        _fok_order = await bridge.submit_order(
                                            market_id=last_market_id,
                                            token_id=last_token_id,
                                            side=OrderSide.BUY,
                                            order_type=OrderType.FOK,
                                            price=_fok_price,
                                            size=last_position_size
                                            or Decimal("0"),
                                            strategy_id=self._strategy.name,
                                        )
                                        if (
                                            _fok_order.status
                                            == OrderStatus.FILLED
                                        ):
                                            has_open_position = True
                                            trade_count += 1
                                            last_entry_price = (
                                                _fok_order.avg_fill_price
                                                or _fok_price
                                            )
                                            logger.info(
                                                "gtc_upgraded_to_fok",
                                                price=str(_fok_price),
                                                minute=minute_in_window,
                                            )
                                        else:
                                            logger.info(
                                                "fok_upgrade_not_filled",
                                                price=str(_fok_price),
                                                status=_fok_order.status.value,
                                            )
                                    pending_gtc_oid = None
                                    pending_gtc_internal_id = None
                                    pending_gtc_position = False
                    except Exception:
                        logger.debug(
                            "mid_tick_gtc_check_failed", exc_info=True,
                        )

                # --- Generate signals ---
                signals = self._strategy.generate_signals(
                    market_state, orderbook, context,
                )

                # Publish signal breakdown to Redis for dashboard
                last_signal_details = ""
                try:
                    entry_signals = [
                        s for s in signals
                        if s.signal_type == SignalType.ENTRY
                    ]
                    # Read vote details from strategy (always available, even during skips)
                    strat_eval = getattr(self._strategy, '_last_evaluation', {})
                    vote_details = strat_eval.get("vote_details", [])
                    parsed_votes = [
                        {"name": vd["name"], "direction": vd["direction"], "strength": round(float(vd["strength"]) * 100, 1)}
                        for vd in vote_details
                        if isinstance(vd, dict) and "name" in vd
                    ]
                    if parsed_votes:
                        import json as _json
                        last_signal_details = _json.dumps(parsed_votes)
                    # Confidence/direction from entry signal if available, else from strategy eval
                    overall_conf = entry_signals[0].confidence.overall if entry_signals and entry_signals[0].confidence else strat_eval.get("confidence", 0.0)
                    overall_dir = entry_signals[0].metadata.get("direction", "").upper() if entry_signals else strat_eval.get("direction", "").upper()
                    await cache.set("bot:signals", {
                        "timestamp": datetime.now(tz=UTC).isoformat(),
                        "minute": minute_in_window,
                        "votes": parsed_votes,
                        "overall_confidence": round(float(overall_conf), 4),
                        "direction": overall_dir,
                        "entry_generated": bool(entry_signals),
                    }, ttl=120)
                except Exception:
                    logger.debug("signal_publish_failed", exc_info=True)

                # Accumulate signal activity (one event per window)
                last_eval: dict[str, Any] = {}
                try:
                    last_eval = getattr(self._strategy, '_last_evaluation', {})
                    eval_outcome = last_eval.get("outcome")
                    if eval_outcome == "skip" and not window_activity_published:
                        # Keep updating the last skip reason — published at window end
                        window_last_skip_eval = dict(last_eval)
                except Exception:
                    logger.debug("signal_activity_accumulate_failed", exc_info=True)

                for sig in signals:
                    if sig.signal_type == SignalType.ENTRY and not has_open_position and not pending_gtc_position:
                        sigmoid_price = sig.entry_price or yes_price

                        # Compute direction-aware CLOB spread for order-type decisions.
                        # Use the TARGET token's spread (NO token for NO signals).
                        if sig.direction == Side.NO and no_clob_best_ask is not None and no_clob_best_bid is not None:
                            clob_spread = no_clob_best_ask - no_clob_best_bid
                        elif clob_best_ask is not None and clob_best_bid is not None:
                            clob_spread = clob_best_ask - clob_best_bid
                        else:
                            clob_spread = Decimal("1")

                        # Select order type: GTC (maker) or FOK (taker)
                        order_type_selected = (
                            self._select_order_type(
                                clob_last_trade=clob_last_trade,
                                clob_spread=clob_spread,
                                minute_in_window=minute_in_window,
                                fok_spread_threshold=fok_spread_threshold,
                                fok_late_minute=fok_late_minute,
                            )
                            if bridge.mode == "live"
                            else OrderType.LIMIT
                        )

                        # GTC spread gate REMOVED: GTC is a resting limit order —
                        # placing at computed_mid on a wide-spread book costs nothing.
                        # max_clob_entry_price gate (line ~1877) prevents overpaying.

                        # --- Entry pricing ---
                        # For NO direction: use real NO token CLOB data if available.
                        # For YES direction (or fallback): use YES token CLOB data.
                        entry_price: Decimal | None = None
                        price_source = "none"

                        if sig.direction == Side.NO and no_clob_last_trade is not None:
                            entry_price = no_clob_last_trade
                            price_source = "no_clob_last_trade"
                        elif sig.direction == Side.NO and no_clob_midpoint is not None:
                            entry_price = no_clob_midpoint
                            price_source = "no_clob_midpoint"
                        elif sig.direction == Side.NO and no_clob_best_ask is not None and no_clob_best_bid is not None:
                            entry_price = (no_clob_best_ask + no_clob_best_bid) / 2
                            price_source = "no_clob_computed_mid"
                        elif sig.direction == Side.YES or sig.direction == Side.NO:
                            # YES direction, or NO fallback to inverted YES
                            if clob_last_trade is not None:
                                entry_price = clob_last_trade
                                price_source = "clob_last_trade"
                            elif clob_midpoint is not None:
                                entry_price = clob_midpoint
                                price_source = "clob_midpoint"
                            elif clob_best_ask is not None and clob_best_bid is not None:
                                entry_price = (clob_best_ask + clob_best_bid) / 2
                                price_source = "clob_computed_mid"
                            elif orderbook.bids and orderbook.asks:
                                entry_price = (orderbook.bids[0].price + orderbook.asks[0].price) / 2
                                price_source = "rest_orderbook_mid"
                            elif clob_best_ask is not None:
                                entry_price = clob_best_ask
                                price_source = "clob_best_ask"

                            # Invert YES price for NO direction (fallback)
                            if sig.direction == Side.NO and entry_price is not None:
                                entry_price = Decimal("1") - entry_price
                                price_source = f"{price_source}_inverted_for_NO"

                        # REST fallback: when WS data yields no price or a desert
                        # computed_mid (~0.50 from 0.01/0.99 book), call REST API.
                        _is_desert_price = (
                            "computed_mid" in price_source
                            and entry_price is not None
                            and abs(entry_price - Decimal("0.50")) < Decimal("0.05")
                        )
                        if (entry_price is None or _is_desert_price) and bridge.mode == "live" and bridge._live_trader:
                            _clob_client = bridge._live_trader._clob_client
                            _target_tid = no_token_id if sig.direction == Side.NO else yes_token_id
                            if _target_tid:
                                # Try REST last_trade_price (non-blocking)
                                try:
                                    _rest_ltp = await asyncio.to_thread(
                                        _clob_client.get_last_trade_price, _target_tid,
                                    )
                                    if _rest_ltp is not None:
                                        _ltp_val = _rest_ltp
                                        if isinstance(_ltp_val, dict):
                                            _ltp_val = _ltp_val.get("price", _ltp_val)
                                        _parsed_ltp = Decimal(str(_ltp_val))
                                        if _parsed_ltp > 0:
                                            entry_price = _parsed_ltp
                                            _dir_label = "no" if sig.direction == Side.NO else "yes"
                                            price_source = f"rest_last_trade_{_dir_label}"
                                            _is_desert_price = False
                                            logger.info(
                                                "rest_fallback_price",
                                                price=str(entry_price),
                                                source=price_source,
                                            )
                                except Exception:
                                    logger.info("rest_price_fallback_failed", exc_info=True)
                                # If still no price, try REST midpoint
                                if entry_price is None or _is_desert_price:
                                    try:
                                        _rest_mid = await asyncio.to_thread(
                                            _clob_client.get_midpoint, _target_tid,
                                        )
                                        if _rest_mid is not None:
                                            _mid_val = _rest_mid
                                            if isinstance(_mid_val, dict):
                                                _mid_val = _mid_val.get("mid", _mid_val)
                                            _parsed_mid = Decimal(str(_mid_val))
                                            if _parsed_mid > 0:
                                                entry_price = _parsed_mid
                                                _dir_label = "no" if sig.direction == Side.NO else "yes"
                                                price_source = f"rest_midpoint_{_dir_label}"
                                                logger.info(
                                                    "rest_fallback_price",
                                                    price=str(entry_price),
                                                    source=price_source,
                                                )
                                    except Exception:
                                        logger.info("rest_midpoint_fallback_failed", exc_info=True)

                        if entry_price is None or entry_price <= 0:
                            logger.warning(
                                "no_real_price_available",
                                minute=minute_in_window,
                                sigmoid=str(sigmoid_price),
                            )
                            continue  # Skip — no real exchange price

                        # FOK price override: use best_ask to cross the spread.
                        # The signer does this too, but we want the correct price
                        # for sizing and max-price validation.
                        # Fallback: invert the opposite token's best_bid when the
                        # target token has no WS data (common on illiquid 15m markets).
                        if order_type_selected == OrderType.FOK:
                            if sig.direction == Side.YES:
                                _fok_ask = clob_best_ask
                                if _fok_ask is None and no_clob_best_bid is not None:
                                    _fok_ask = Decimal("1") - no_clob_best_bid
                                if _fok_ask is not None:
                                    entry_price = _fok_ask
                                    price_source = "clob_best_ask_fok"
                            elif sig.direction == Side.NO:
                                _fok_ask = no_clob_best_ask
                                if _fok_ask is None and clob_best_bid is not None:
                                    _fok_ask = Decimal("1") - clob_best_bid
                                if _fok_ask is not None:
                                    entry_price = _fok_ask
                                    price_source = "no_clob_best_ask_fok"
                            # FOK MUST have a real ask to cross; skip if we
                            # could not determine one from either token.
                            if price_source not in (
                                "clob_best_ask_fok",
                                "no_clob_best_ask_fok",
                            ):
                                logger.info(
                                    "fok_no_real_ask_to_cross",
                                    price=str(entry_price),
                                    source=price_source,
                                    minute=minute_in_window,
                                )
                                continue

                        logger.debug(
                            "entry_price_source",
                            price=str(entry_price),
                            source=price_source,
                            sigmoid=str(sigmoid_price),
                            direction=sig.direction.value,
                            order_type=order_type_selected.value,
                        )

                        # Safety gate: reject entries where CLOB price is too high.
                        # FOK taker has a more permissive limit (pays more for guaranteed fill).
                        # Confidence-adaptive: high-confidence signals can buy at higher prices.
                        _sig_conf = float(sig.confidence.overall) if sig.confidence else 0.5
                        if conf_adaptive and _sig_conf > float(conf_adaptive_base):
                            _conf_boost = Decimal(str(_sig_conf - float(conf_adaptive_base))) * conf_adaptive_scale
                            effective_max_clob = min(max_clob_entry_price + _conf_boost, conf_adaptive_cap)
                            effective_fok_max = min(fok_max_entry_price + _conf_boost, conf_adaptive_cap)
                        else:
                            effective_max_clob = max_clob_entry_price
                            effective_fok_max = fok_max_entry_price

                        _effective_max = (
                            effective_fok_max
                            if order_type_selected == OrderType.FOK
                            else effective_max_clob
                        )
                        if conf_adaptive:
                            logger.info(
                                "confidence_adaptive_pricing",
                                effective_max=str(_effective_max),
                                base_max=str(max_clob_entry_price if order_type_selected != OrderType.FOK else fok_max_entry_price),
                                confidence=round(_sig_conf, 4),
                                order_type=order_type_selected.value,
                            )
                        if entry_price > _effective_max:
                            logger.info(
                                "clob_entry_price_too_high",
                                price=str(entry_price),
                                max_allowed=str(_effective_max),
                                order_type=order_type_selected.value,
                            )
                            continue

                        # --- Position sizing: fixed bet or Kelly ---
                        # Sizer returns USD amount; convert to shares (USD / entry_price)
                        # because downstream (cost calc, paper trader, PnL) all expect shares.
                        if self._fixed_bet_size > 0:
                            # Flat bet mode — fixed USD per trade → shares
                            usd_bet = min(self._fixed_bet_size, cached_balance)
                            position_size = usd_bet / entry_price
                            sizing = PositionSize(
                                recommended_size=usd_bet,
                                max_allowed=usd_bet,
                                kelly_fraction=Decimal("0"),
                                capped_reason="fixed_bet",
                            )
                        else:
                            # Per-signal confidence Kelly: use signal confidence
                            # instead of static estimated_win_prob when it's higher.
                            if use_signal_kelly and _sig_conf > float(estimated_win_prob):
                                effective_win_prob = min(
                                    max(Decimal(str(_sig_conf)), kelly_conf_floor),
                                    kelly_conf_cap,
                                )
                            else:
                                effective_win_prob = estimated_win_prob
                            if effective_win_prob != estimated_win_prob:
                                logger.info(
                                    "signal_confidence_kelly",
                                    static_prob=str(estimated_win_prob),
                                    effective_prob=str(effective_win_prob),
                                    signal_confidence=round(_sig_conf, 4),
                                )
                            sizing = position_sizer.calculate_binary(
                                balance=cached_balance,
                                entry_price=entry_price,
                                estimated_win_prob=effective_win_prob,
                            )
                            # Convert USD recommendation to shares
                            position_size = sizing.recommended_size / entry_price

                        # Scale position by signal confidence to improve R:R.
                        # High-confidence trades (0.95) get full size; low-confidence
                        # trades (0.60) get ~35% size. This means wins (correlated with
                        # high confidence) are bigger than losses (low confidence).
                        signal_conf = float(sig.confidence.overall) if sig.confidence else 0.5
                        conf_factor = max(0.3, min(1.0, (signal_conf - 0.40) / 0.55))
                        position_size = position_size * Decimal(str(round(conf_factor, 4)))

                        # Floor: never drop below Polymarket CLOB minimum (5 tokens)
                        # plus a small buffer.  Without this, confidence scaling on
                        # small balances can produce 2-3 token orders that get rejected.
                        _MIN_TOKENS = Decimal("6")
                        if position_size < _MIN_TOKENS and position_size > 0:
                            position_size = _MIN_TOKENS

                        logger.info(
                            "confidence_position_scaling",
                            signal_confidence=round(signal_conf, 4),
                            confidence_factor=round(conf_factor, 4),
                            scaled_size=str(position_size),
                        )

                        # Confidence-based limit price: discount entry to improve R:R.
                        # Only applies to GTC maker orders — FOK crosses the spread
                        # immediately so discounting makes no sense.
                        original_entry_price = entry_price
                        if (
                            use_confidence_discount
                            and self._mode == "live"
                            and signal_conf > 0
                            and order_type_selected == OrderType.GTC
                        ):
                            discount_pct = self._compute_confidence_discount(
                                signal_conf,
                                min_discount_pct=pricing_min_discount_pct,
                                max_discount_pct=pricing_max_discount_pct,
                            )
                            entry_price = self._apply_confidence_discount(
                                entry_price, discount_pct,
                            )
                            logger.info(
                                "confidence_limit_price",
                                original_price=str(original_entry_price),
                                adjusted_price=str(entry_price),
                                discount_pct=round(discount_pct * 100, 2),
                                direction=sig.direction.value,
                                signal_confidence=round(signal_conf, 4),
                            )

                        # Publish sizing details to Redis for dashboard
                        try:
                            await cache.set("bot:sizing", {
                                "kelly_fraction": str(sizing.kelly_fraction),
                                "recommended_size": str(sizing.recommended_size),
                                "max_allowed": str(sizing.max_allowed),
                                "capped_reason": sizing.capped_reason or "none",
                                "balance": str(cached_balance),
                                "entry_price": str(entry_price),
                                "original_entry_price": str(original_entry_price),
                                "discount_applied": str(entry_price != original_entry_price),
                                "estimated_win_prob": str(estimated_win_prob),
                                "effective_win_prob": str(effective_win_prob) if not self._fixed_bet_size else str(estimated_win_prob),
                                "effective_max_entry": str(_effective_max),
                            }, ttl=120)
                        except Exception:
                            logger.debug("sizing_publish_failed", exc_info=True)

                        if position_size <= 0:
                            logger.info(
                                "swing_entry_skip_no_edge",
                                entry_price=str(entry_price),
                                kelly=str(sizing.kelly_fraction),
                                reason=sizing.capped_reason or "zero size",
                            )
                            continue

                        # Cost check — reduce position size to leave room for fees
                        # position_size is shares; actual USDC cost = shares * entry_price
                        costs = cost_calculator.calculate_binary(
                            entry_price=entry_price,
                            position_size=position_size,
                        )
                        # Ensure USDC cost + fees fits within position cap (ramp-aware)
                        max_allowed = cached_balance * effective_max_position_pct
                        usdc_cost = position_size * entry_price + costs.fee_cost
                        if usdc_cost > max_allowed and costs.fee_cost > 0:
                            position_size = (max_allowed - costs.fee_cost) / entry_price
                            if position_size <= 0:
                                continue
                            # Recalculate fees at reduced size
                            costs = cost_calculator.calculate_binary(
                                entry_price=entry_price,
                                position_size=position_size,
                            )

                        # Risk gate (include fees in size check)
                        # position_size is SHARES; check_order expects USDC notional
                        notional_usdc = position_size * entry_price
                        # Adjust drawdown for pending settlements: winning tokens
                        # on-chain haven't been credited to USDC balance yet, so
                        # raw drawdown over-estimates risk.  Add daily_pnl back.
                        if daily_pnl > 0 and portfolio._peak_equity > 0:
                            adjusted_equity = cached_balance + daily_pnl
                            current_drawdown = max(
                                Decimal("0"),
                                (portfolio._peak_equity - adjusted_equity)
                                / portfolio._peak_equity,
                            )
                        else:
                            current_drawdown = portfolio.max_drawdown
                        decision = risk_mgr.check_order(
                            signal=sig,
                            position_size=notional_usdc,
                            current_drawdown=current_drawdown,
                            balance=cached_balance,
                            estimated_fee=costs.fee_cost,
                        )
                        if decision.approved:
                            token_id = (
                                yes_token_id
                                if sig.direction == Side.YES
                                else no_token_id
                            )
                            live_mode = bridge.mode == "live"

                            # Live: dynamic order type (GTC maker or FOK taker).
                            # GTC rests in the book; FOK fills immediately or rejects.
                            # Paper: instant-fill LIMIT (PaperTrader always fills).
                            entry_order = await bridge.submit_order(
                                market_id=sig.market_id,
                                token_id=token_id,
                                side=OrderSide.BUY,
                                order_type=order_type_selected,
                                price=entry_price,
                                size=position_size,
                                strategy_id=sig.strategy_id,
                            )
                            logger.info(
                                "order_type_selected",
                                order_type=order_type_selected.value,
                                price=str(entry_price),
                                spread=str(clob_spread),
                                minute=minute_in_window,
                                has_last_trade=clob_last_trade is not None,
                            )
                            if entry_order.status == OrderStatus.REJECTED:
                                logger.warning(
                                    "entry_order_rejected",
                                    market_id=sig.market_id,
                                    reason="order returned REJECTED status",
                                )
                                continue

                            # GTC SUBMITTED: order is resting in the book.
                            # Don't block — let the market come to us. Track
                            # the pending order; we'll check fill status on
                            # subsequent ticks and at window boundary.
                            if live_mode and entry_order.status == OrderStatus.SUBMITTED:
                                pending_gtc_oid = entry_order.exchange_order_id
                                pending_gtc_internal_id = str(entry_order.id)
                                logger.info(
                                    "gtc_order_resting",
                                    market_id=sig.market_id,
                                    exchange_oid=pending_gtc_oid,
                                    price=str(entry_price),
                                    size=str(position_size),
                                    direction=sig.direction.value,
                                )
                                # Track pending GTC details — but do NOT mark
                                # as open position until MATCHED.  This prevents
                                # the dashboard from showing "In Trade" for an
                                # unfilled resting order.
                                has_open_position = False
                                pending_gtc_position = True
                                last_entry_price = entry_price
                                last_entry_side = sig.direction.value
                                last_entry_time = datetime.now(tz=UTC)
                                last_position_size = position_size
                                last_confidence = sig.confidence.overall if sig.confidence else 0.0
                                last_entry_minute = minute_in_window
                                last_cum_return_entry = cum_return * 100
                                last_fee_cost = costs.fee_cost
                                last_token_id = token_id
                                last_market_id = market_id
                                last_condition_id = (
                                    (active_market.condition_id or active_market.market_id)
                                    if active_market else market_id
                                )
                                last_signal_details_str = last_signal_details
                                last_clob_entry_price = clob_last_trade or clob_midpoint or clob_best_ask
                                last_sigmoid_entry_price = sigmoid_price

                                # Publish SKIP event for GTC orders — not a real entry until filled
                                if not window_activity_published:
                                    try:
                                        import uuid as _uuid_gtc
                                        gtc_entry_event = {
                                            "id": str(_uuid_gtc.uuid4())[:8],
                                            "timestamp": datetime.now(tz=UTC).isoformat(),
                                            "minute": minute_in_window,
                                            "market_id": sig.market_id,
                                            "outcome": "skip",
                                            "reason": "gtc_pending",
                                            "direction": sig.direction.value,
                                            "confidence": last_confidence,
                                            "votes": last_eval.get("votes", {}) if last_eval.get("outcome") == "entry" else {},
                                            "detail": f"GTC resting, Kelly={sizing.kelly_fraction}, size={position_size}",
                                        }
                                        await cache.push_to_list(
                                            "bot:signal_activity", gtc_entry_event,
                                            max_length=20, ttl=86400,
                                        )
                                        if ch_store is not None:
                                            try:
                                                from datetime import datetime as _dt_gtc
                                                await ch_store.insert_signal_evaluation({
                                                    "eval_id": gtc_entry_event["id"],
                                                    "timestamp": _dt_gtc.fromisoformat(gtc_entry_event["timestamp"]),
                                                    "strategy": self._strategy_name,
                                                    "market_id": gtc_entry_event["market_id"],
                                                    "minute": gtc_entry_event["minute"],
                                                    "outcome": gtc_entry_event["outcome"],
                                                    "reason": gtc_entry_event["reason"],
                                                    "direction": gtc_entry_event.get("direction", ""),
                                                    "confidence": gtc_entry_event.get("confidence", 0),
                                                    "votes_yes": gtc_entry_event.get("votes", {}).get("yes", 0),
                                                    "votes_no": gtc_entry_event.get("votes", {}).get("no", 0),
                                                    "votes_neutral": gtc_entry_event.get("votes", {}).get("neutral", 0),
                                                    "detail": gtc_entry_event.get("detail", ""),
                                                })
                                            except Exception:
                                                logger.debug("clickhouse_signal_eval_gtc_entry_failed", exc_info=True)
                                        window_activity_published = True
                                    except Exception:
                                        logger.debug("signal_activity_gtc_entry_publish_failed", exc_info=True)

                                continue

                            elif entry_order.status == OrderStatus.SUBMITTED:
                                # Paper mode — shouldn't happen
                                continue
                            # Use actual filled size (handles partial fills)
                            actual_fill_size = entry_order.filled_size or position_size
                            actual_fill_price = entry_order.avg_fill_price or entry_price
                            has_open_position = True
                            trade_count += 1
                            last_entry_price = actual_fill_price
                            last_entry_side = sig.direction.value
                            last_entry_time = datetime.now(tz=UTC)
                            last_position_size = actual_fill_size
                            last_confidence = sig.confidence.overall if sig.confidence else 0.0
                            last_entry_minute = minute_in_window
                            last_cum_return_entry = cum_return * 100
                            last_fee_cost = costs.fee_cost
                            last_token_id = token_id
                            last_market_id = market_id
                            last_condition_id = (
                                (active_market.condition_id or active_market.market_id)
                                if active_market else market_id
                            )
                            last_signal_details_str = last_signal_details
                            last_clob_entry_price = clob_last_trade or clob_midpoint or clob_best_ask
                            last_sigmoid_entry_price = sigmoid_price

                            # Persist position to Redis (survives restart)
                            try:
                                await self._save_position(
                                    redis_conn,
                                    market_id=market_id,
                                    token_id=token_id,
                                    condition_id=last_condition_id or "",
                                    entry_price=actual_fill_price,
                                    entry_side=sig.direction.value,
                                    position_size=actual_fill_size,
                                    fee_cost=costs.fee_cost,
                                )
                            except Exception:
                                logger.warning("position_persist_failed", exc_info=True)

                            # Sync balance after entry
                            if self._mode == "live" and live_trader is not None:
                                try:
                                    synced = await live_trader.get_balance()
                                    if synced is not None and synced > 0:
                                        cached_balance = Decimal(str(synced))
                                        last_balance_sync = datetime.now(tz=UTC)
                                except Exception:
                                    logger.debug("post_entry_balance_sync_failed", exc_info=True)
                            elif paper_trader is not None:
                                cached_balance = paper_trader.balance

                            # Record entry audit event
                            entry_event = "LIVE_ENTRY" if self._mode == "live" else "PAPER_ENTRY"
                            if ch_store is not None:
                                try:
                                    await ch_store.insert_audit_event({
                                        "event_type": entry_event,
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
                            portfolio.update_equity(cached_balance)
                            logger.info(
                                "swing_entry_executed",
                                direction=sig.direction.value,
                                price=str(last_entry_price),
                                size=str(last_position_size),
                                model_price=str(entry_price),
                                model_size=str(position_size),
                                kelly=str(sizing.kelly_fraction),
                                capped=sizing.capped_reason or "none",
                                est_fee=str(costs.fee_cost),
                                balance=str(cached_balance),
                            )
                            # Publish ENTRY event immediately (one per window)
                            if not window_activity_published:
                                try:
                                    import uuid as _uuid_entry
                                    entry_event = {
                                        "id": str(_uuid_entry.uuid4())[:8],
                                        "timestamp": datetime.now(tz=UTC).isoformat(),
                                        "minute": minute_in_window,
                                        "market_id": market_id,
                                        "outcome": "entry",
                                        "reason": "entry_signal",
                                        "direction": sig.direction.value,
                                        "confidence": last_confidence,
                                        "votes": last_eval.get("votes", {}) if last_eval.get("outcome") == "entry" else {},
                                        "detail": f"Kelly={sizing.kelly_fraction}, size={position_size}",
                                    }
                                    await cache.push_to_list(
                                        "bot:signal_activity", entry_event,
                                        max_length=20, ttl=86400,
                                    )
                                    # Persist to ClickHouse for historical analysis
                                    if ch_store is not None:
                                        try:
                                            from datetime import datetime as _dt_parse2
                                            await ch_store.insert_signal_evaluation({
                                                "eval_id": entry_event["id"],
                                                "timestamp": _dt_parse2.fromisoformat(entry_event["timestamp"]),
                                                "strategy": self._strategy_name,
                                                "market_id": entry_event["market_id"],
                                                "minute": entry_event["minute"],
                                                "outcome": entry_event["outcome"],
                                                "reason": entry_event["reason"],
                                                "direction": entry_event.get("direction", ""),
                                                "confidence": entry_event.get("confidence", 0),
                                                "votes_yes": entry_event.get("votes", {}).get("yes", 0),
                                                "votes_no": entry_event.get("votes", {}).get("no", 0),
                                                "votes_neutral": entry_event.get("votes", {}).get("neutral", 0),
                                                "detail": entry_event.get("detail", ""),
                                            })
                                        except Exception:
                                            logger.debug("clickhouse_signal_eval_entry_failed", exc_info=True)
                                    window_activity_published = True
                                except Exception:
                                    logger.debug("signal_activity_entry_publish_failed", exc_info=True)
                        else:
                            logger.info(
                                "swing_entry_rejected",
                                reason=decision.reason,
                                requested_size=str(position_size),
                            )
                            # Publish rejection to signal activity feed
                            try:
                                import uuid as _uuid
                                reject_event = {
                                    "id": str(_uuid.uuid4())[:8],
                                    "timestamp": datetime.now(tz=UTC).isoformat(),
                                    "minute": minute_in_window,
                                    "market_id": sig.market_id,
                                    "outcome": "rejected",
                                    "reason": f"risk: {decision.reason}",
                                    "direction": sig.direction.value if hasattr(sig.direction, "value") else str(sig.direction),
                                    "confidence": last_confidence,
                                    "votes": last_eval.get("votes", {}) if last_eval else {},
                                    "detail": f"size={position_size}, balance={cached_balance}",
                                }
                                await cache.push_to_list(
                                    "bot:signal_activity", reject_event,
                                    max_length=20, ttl=86400,
                                )
                                window_activity_published = True
                            except Exception:
                                logger.debug("signal_activity_reject_publish_failed", exc_info=True)

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
                        # Submit sell to credit proceeds (paper only).
                        # In live mode, settlement is on-chain.
                        if bridge.mode != "live":
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
                        else:
                            logger.info(
                                "live_settlement_skip_clob_sell",
                                market_id=market_id,
                                exit_price=str(exit_price_ex),
                                reason="redeeming on-chain via CTF",
                            )
                            # Auto-claim: redeem settled tokens on-chain (only winners)
                            if exit_price_ex == Decimal("1"):
                                if redeemer is not None and last_condition_id and last_token_id:
                                    _cid_ex = last_condition_id
                                    _tid_ex = last_token_id

                                    async def _background_redeem_ex(
                                        r: object, cid: str, tid: str,
                                    ) -> None:
                                        delays = [60, 120, 300]
                                        for attempt, delay in enumerate(delays):
                                            await asyncio.sleep(delay)
                                            try:
                                                tx = await r.redeem_positions(  # type: ignore[union-attr]
                                                    condition_id=cid,
                                                    token_id=tid,
                                                )
                                                if tx:
                                                    logger.info(
                                                        "live_redeem_success_bg",
                                                        tx_hash=tx,
                                                        condition_id=cid[:16],
                                                        attempt=attempt,
                                                    )
                                                    return
                                            except Exception:
                                                logger.debug(
                                                    "live_redeem_retry_failed",
                                                    attempt=attempt,
                                                    condition_id=cid[:16],
                                                    exc_info=True,
                                                )
                                        logger.info(
                                            "live_redeem_deferred_to_sweep",
                                            condition_id=cid[:16],
                                            reason="all retries failed, startup sweep will handle",
                                        )

                                    asyncio.create_task(_background_redeem_ex(redeemer, _cid_ex, _tid_ex))
                                    logger.info(
                                        "live_redeem_scheduled",
                                        condition_id=last_condition_id[:16],
                                        token_id=last_token_id[:16],
                                        delays="60s,120s,300s",
                                    )
                            else:
                                logger.info(
                                    "live_loss_no_redeem",
                                    exit_price=str(exit_price_ex),
                                    reason="losing position, tokens worthless",
                                )
                        # P&L with fee deduction (entry + exit)
                        gross_pnl_ex = (exit_price_ex - (last_entry_price or Decimal("0"))) * (
                            last_position_size or Decimal("0")
                        )
                        exit_notional_ex = exit_price_ex * (last_position_size or Decimal("0"))
                        exit_fee_ex = CostCalculator.polymarket_fee(exit_notional_ex, exit_price_ex)
                        pnl_ex = gross_pnl_ex - last_fee_cost - exit_fee_ex
                        daily_pnl += pnl_ex

                        # Sync balance after trade
                        if self._mode == "live" and live_trader is not None:
                            try:
                                synced = await live_trader.get_balance()
                                if synced is not None and synced > 0:
                                    cached_balance = Decimal(str(synced))
                                    last_balance_sync = datetime.now(tz=UTC)
                            except Exception:
                                logger.debug("post_trade_balance_sync_failed", exc_info=True)
                        elif paper_trader is not None:
                            cached_balance = paper_trader.balance

                        portfolio.update_equity(cached_balance)

                        is_win_ex = pnl_ex > 0
                        if is_win_ex:
                            win_count += 1
                        else:
                            loss_count += 1
                        await kill_switch.async_record_trade_result(is_win=is_win_ex)
                        exit_reason_str = str(sig.exit_reason) if sig.exit_reason else "signal_exit"
                        trade_id_ex = ""
                        if ch_store is not None:
                            trade_id_ex = await self._record_completed_trade(
                                ch_store,
                                market_id=market_id,
                                direction=last_entry_side or "YES",
                                entry_price=last_entry_price or Decimal("0"),
                                exit_price=exit_price_ex,
                                position_size=(last_entry_price or Decimal("0")) * (
                                    last_position_size or Decimal("0")
                                ),
                                entry_time=last_entry_time,
                                exit_reason=exit_reason_str,
                                window_minute=last_entry_minute,
                                cum_return_pct=cum_return * 100,
                                confidence=last_confidence,
                                fee_cost=last_fee_cost,
                                pnl=pnl_ex,
                                signal_details=last_signal_details_str,
                                clob_entry_price=last_clob_entry_price,
                                sigmoid_entry_price=last_sigmoid_entry_price,
                            )
                        # Publish last trade to Redis for dashboard
                        try:
                            await cache.set("bot:last_trade", {
                                "trade_id": trade_id_ex,
                                "direction": last_entry_side or "YES",
                                "entry_price": str(last_entry_price or Decimal("0")),
                                "exit_price": str(exit_price_ex),
                                "pnl": str(pnl_ex),
                                "exit_reason": exit_reason_str,
                                "window_minute": last_entry_minute,
                                "confidence": last_confidence,
                                "timestamp": datetime.now(tz=UTC).isoformat(),
                            }, ttl=300)
                        except Exception:
                            logger.debug("last_trade_publish_failed", exc_info=True)
                        logger.info(
                            "swing_exit_executed",
                            reason=exit_reason_str,
                            pnl=str(pnl_ex),
                            balance=str(cached_balance),
                        )
                        has_open_position = False
                        last_entry_price = None
                        last_entry_side = None
                        last_entry_time = None
                        last_position_size = None
                        last_clob_entry_price = None
                        last_sigmoid_entry_price = None
                        # Clear persisted position from Redis
                        try:
                            await self._clear_position(redis_conn)
                        except Exception:
                            logger.debug("position_clear_failed", exc_info=True)

                # Wait for next candle (or shutdown)
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(), timeout=60.0,
                    )
                    break
                except TimeoutError:
                    continue

        finally:
            # --- Graceful shutdown: close open position if any ---
            if has_open_position and last_position_size and last_entry_price:
                logger.warning(
                    "shutdown_closing_position",
                    market_id=last_market_id,
                    entry_side=last_entry_side,
                    size=str(last_position_size),
                )
                try:
                    shutdown_close = asyncio.wait_for(
                        bridge.submit_order(
                            market_id=last_market_id or "btc_15m_swing",
                            token_id=last_token_id or "",
                            side=OrderSide.SELL,
                            order_type=OrderType.MARKET,
                            price=last_tick_yes_price,
                            size=last_position_size,
                            strategy_id=self._strategy_name,
                        ),
                        timeout=60.0,
                    )
                    await shutdown_close
                    logger.info("shutdown_position_closed")
                except Exception:
                    logger.error("shutdown_position_close_failed", exc_info=True)

            ws_connected = False
            await ws_feed.disconnect()
            await futures_feed.disconnect()
            await poly_ws_feed.disconnect()
            self._remove_sentinel(_WS_CONNECTED_FILE)
            await cache.close()
            if ch_store is not None:
                await ch_store.disconnect()
            logger.info(
                "swing_loop_stopped",
                final_balance=str(cached_balance),
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
        import src.strategies.false_sentiment
    with contextlib.suppress(ImportError):
        import src.strategies.reversal_catcher
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
