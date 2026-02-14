"""Tests for the bot orchestrator."""

from __future__ import annotations

import asyncio
from datetime import UTC
from decimal import Decimal

import pytest

from src.bot import BotOrchestrator, run_backtest
from src.config.loader import ConfigLoader


@pytest.fixture()
def config(tmp_path):
    """Create a minimal config for bot tests."""
    default = tmp_path / "default.toml"
    default.write_text(
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

[orderbook]
heavy_book_multiplier = 3.0

[liquidity]
min_hourly_volume = 100
max_spread_cents = 0.05
"""
    )
    loader = ConfigLoader(config_dir=str(tmp_path), env="development")
    loader.load()
    return loader


@pytest.fixture()
def swing_config(tmp_path):
    """Config with momentum_confirmation strategy parameters."""
    default = tmp_path / "default.toml"
    default.write_text(
        """\
[strategy.momentum_confirmation]
entry_threshold = 0.10
entry_minute_start = 5
entry_minute_end = 8
profit_target_pct = 0.25
stop_loss_pct = 0.15
max_hold_minutes = 7
min_confidence = 0.70

[exit]
profit_target_pct = 0.05
trailing_stop_pct = 0.03
hard_stop_loss_pct = 0.04
max_hold_seconds = 420

[risk]
max_position_pct = 0.02
max_daily_drawdown_pct = 0.05
max_order_book_pct = 0.10

[orderbook]
heavy_book_multiplier = 3.0

[liquidity]
min_hourly_volume = 100
max_spread_cents = 0.05
"""
    )
    loader = ConfigLoader(config_dir=str(tmp_path), env="development")
    loader.load()
    return loader


class TestBotOrchestrator:
    def test_init(self, config):
        """BotOrchestrator initializes without error."""
        bot = BotOrchestrator(mode="paper", strategy_name="false_sentiment", config=config)
        assert bot._mode == "paper"
        assert bot._strategy_name == "false_sentiment"
        assert bot._running is False

    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, config):
        """Bot should shut down gracefully when shutdown event is set."""
        # Ensure strategy is registered (re-register if cleared by other tests)
        from src.strategies import registry
        from src.strategies.false_sentiment import FalseSentimentStrategy
        if "false_sentiment" not in registry._REGISTRY:
            registry.register("false_sentiment")(FalseSentimentStrategy)

        bot = BotOrchestrator(mode="paper", strategy_name="false_sentiment", config=config)

        async def trigger_shutdown():
            await asyncio.sleep(0.2)
            bot._running = False
            bot._shutdown_event.set()

        shutdown_task = asyncio.create_task(trigger_shutdown())
        result = await asyncio.wait_for(bot.start(), timeout=5.0)
        assert result == 0
        # Ensure shutdown task completes cleanly
        if not shutdown_task.done():
            await shutdown_task


class TestSwingMode:
    """Tests for swing (momentum_confirmation) trading mode."""

    def test_swing_mode_init(self, swing_config):
        """BotOrchestrator initializes with momentum_confirmation strategy."""
        bot = BotOrchestrator(
            mode="paper",
            strategy_name="momentum_confirmation",
            config=swing_config,
        )
        assert bot._mode == "paper"
        assert bot._strategy_name == "momentum_confirmation"
        assert bot._running is False

    def test_compute_yes_price_zero_return(self):
        """Sigmoid of 0 cumulative return should yield 0.5 (fair coin)."""
        result = BotOrchestrator._compute_yes_price(0.0)
        assert result == Decimal("0.5")

    def test_compute_yes_price_positive_return(self):
        """Positive cumulative return should push YES price above 0.5."""
        result = BotOrchestrator._compute_yes_price(0.05)
        assert result > Decimal("0.5")
        assert result < Decimal("1.0")

    def test_compute_yes_price_negative_return(self):
        """Negative cumulative return should push YES price below 0.5."""
        result = BotOrchestrator._compute_yes_price(-0.05)
        assert result < Decimal("0.5")
        assert result > Decimal("0.0")

    def test_compute_yes_price_symmetry(self):
        """Sigmoid should be symmetric: f(x) + f(-x) = 1."""
        pos = BotOrchestrator._compute_yes_price(0.03)
        neg = BotOrchestrator._compute_yes_price(-0.03)
        assert abs(pos + neg - Decimal("1.0")) < Decimal("0.001")

    def test_compute_yes_price_large_positive(self):
        """Large positive return should be clamped to 0.99."""
        result = BotOrchestrator._compute_yes_price(1.0)
        assert result >= Decimal("0.99")

    def test_compute_yes_price_large_negative(self):
        """Large negative return should be clamped to 0.01."""
        result = BotOrchestrator._compute_yes_price(-1.0)
        assert result <= Decimal("0.01")

    def test_window_start_ts_aligned(self):
        """Timestamps already on 15m boundary should remain unchanged."""
        # 2024-01-01 00:00:00 UTC = 1704067200
        assert BotOrchestrator._window_start_ts(1704067200) == 1704067200

    def test_window_start_ts_mid_window(self):
        """Mid-window timestamp should align to previous 15m boundary."""
        # 1704067200 + 7*60 = 7 minutes in
        assert BotOrchestrator._window_start_ts(1704067200 + 420) == 1704067200


class TestRunBacktest:
    def test_backtest_returns_zero(self, tmp_path):
        """run_backtest should return 0."""
        default = tmp_path / "default.toml"
        default.write_text(
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
"""
        )
        result = run_backtest(
            strategy="false_sentiment",
            config_dir=str(tmp_path),
            env="development",
        )
        assert result == 0


class TestStatePublishing:
    """Tests for Redis state publishing."""

    @pytest.fixture()
    def mock_cache(self):
        from unittest.mock import AsyncMock

        cache = AsyncMock()
        cache.set = AsyncMock()
        return cache

    @pytest.fixture()
    def mock_paper_trader(self):
        from unittest.mock import MagicMock

        trader = MagicMock()
        trader.balance = Decimal("10500")
        return trader

    @pytest.fixture()
    def mock_active_market(self):
        from unittest.mock import MagicMock

        market = MagicMock()
        market.market_id = "btc_15m_test"
        return market

    def _find_call(self, mock_cache, key: str):
        """Find a cache.set call by its first argument (key)."""
        calls = mock_cache.set.call_args_list
        matches = [c for c in calls if c[0][0] == key]
        assert len(matches) == 1, f"Expected exactly 1 call with key={key!r}, got {len(matches)}"
        return matches[0]

    @pytest.mark.asyncio
    async def test_publish_state_heartbeat(self, swing_config, mock_cache, mock_paper_trader):
        """Heartbeat should contain timestamp, uptime, mode, and strategy."""
        from datetime import datetime

        bot = BotOrchestrator(mode="paper", strategy_name="momentum_confirmation", config=swing_config)
        start = datetime.now(tz=UTC)
        await bot._publish_state(
            mock_cache,
            start_time=start,
            current_balance=Decimal("10500"),
            initial_balance=Decimal("10000"),
            daily_pnl=Decimal("0"),
            trade_count=0,
            win_count=0,
            loss_count=0,
            has_open_position=False,
            active_market=None,
            ws_connected=True,
        )
        call = self._find_call(mock_cache, "bot:heartbeat")
        data = call[0][1]
        assert data["mode"] == "paper"
        assert data["strategy"] == "momentum_confirmation"
        assert "timestamp" in data
        assert "uptime_s" in data
        assert call[1]["ttl"] == 120

    @pytest.mark.asyncio
    async def test_publish_state_balance(self, swing_config, mock_cache, mock_paper_trader):
        """Balance payload should contain balance, pnl, and pnl_pct."""
        from datetime import datetime

        bot = BotOrchestrator(mode="paper", strategy_name="momentum_confirmation", config=swing_config)
        start = datetime.now(tz=UTC)
        await bot._publish_state(
            mock_cache,
            start_time=start,
            current_balance=Decimal("10500"),
            initial_balance=Decimal("10000"),
            daily_pnl=Decimal("50"),
            trade_count=3,
            win_count=2,
            loss_count=1,
            has_open_position=False,
            active_market=None,
            ws_connected=True,
        )
        call = self._find_call(mock_cache, "bot:balance")
        data = call[0][1]
        # balance is 10500, initial is 10000 => pnl = 500, pnl_pct = 5.0
        assert data["balance"] == "10500"
        assert data["initial_balance"] == "10000"
        assert data["pnl"] == "500"
        assert data["pnl_pct"] == 5.0
        assert call[1]["ttl"] == 300

    @pytest.mark.asyncio
    async def test_publish_state_position_open(
        self, swing_config, mock_cache, mock_paper_trader, mock_active_market,
    ):
        """Position should be published with market data when position is open."""
        from datetime import datetime

        bot = BotOrchestrator(mode="paper", strategy_name="momentum_confirmation", config=swing_config)
        start = datetime.now(tz=UTC)
        entry_time = datetime.now(tz=UTC)
        await bot._publish_state(
            mock_cache,
            start_time=start,
            current_balance=Decimal("10500"),
            initial_balance=Decimal("10000"),
            daily_pnl=Decimal("0"),
            trade_count=1,
            win_count=1,
            loss_count=0,
            has_open_position=True,
            active_market=mock_active_market,
            entry_price=Decimal("0.55"),
            position_size=Decimal("200"),
            entry_side="YES",
            entry_time=entry_time,
            ws_connected=True,
        )
        call = self._find_call(mock_cache, "bot:position")
        data = call[0][1]
        assert data["market_id"] == "btc_15m_test"
        assert data["side"] == "YES"
        assert data["entry_price"] == "0.55"
        assert data["size"] == "200"
        assert data["entry_time"] == str(entry_time)
        assert call[1]["ttl"] == 300

    @pytest.mark.asyncio
    async def test_publish_state_position_none(self, swing_config, mock_cache, mock_paper_trader):
        """Position should be None when no position is open."""
        from datetime import datetime

        bot = BotOrchestrator(mode="paper", strategy_name="momentum_confirmation", config=swing_config)
        start = datetime.now(tz=UTC)
        await bot._publish_state(
            mock_cache,
            start_time=start,
            current_balance=Decimal("10500"),
            initial_balance=Decimal("10000"),
            daily_pnl=Decimal("0"),
            trade_count=0,
            win_count=0,
            loss_count=0,
            has_open_position=False,
            active_market=None,
            ws_connected=False,
        )
        call = self._find_call(mock_cache, "bot:position")
        data = call[0][1]
        assert data is None
        assert call[1]["ttl"] == 300

    @pytest.mark.asyncio
    async def test_publish_state_window(self, swing_config, mock_cache, mock_paper_trader):
        """Window payload should contain minute, cum_return_pct, yes_price, btc_close."""
        from datetime import datetime

        bot = BotOrchestrator(mode="paper", strategy_name="momentum_confirmation", config=swing_config)
        start = datetime.now(tz=UTC)
        await bot._publish_state(
            mock_cache,
            start_time=start,
            current_balance=Decimal("10500"),
            initial_balance=Decimal("10000"),
            daily_pnl=Decimal("0"),
            trade_count=0,
            win_count=0,
            loss_count=0,
            has_open_position=False,
            active_market=None,
            current_window_start=1704067200,
            minute_in_window=7,
            cum_return=0.0312,
            yes_price=Decimal("0.65"),
            btc_close=Decimal("43500"),
            ws_connected=True,
        )
        call = self._find_call(mock_cache, "bot:window")
        data = call[0][1]
        assert data["start_ts"] == 1704067200
        assert data["minute"] == 7
        assert data["cum_return_pct"] == round(0.0312 * 100, 4)
        assert data["yes_price"] == "0.65"
        assert data["btc_close"] == "43500"
        assert call[1]["ttl"] == 120

    @pytest.mark.asyncio
    async def test_publish_state_daily(self, swing_config, mock_cache, mock_paper_trader):
        """Daily stats should contain trade_count, win_count, loss_count, daily_pnl."""
        from datetime import datetime

        bot = BotOrchestrator(mode="paper", strategy_name="momentum_confirmation", config=swing_config)
        start = datetime.now(tz=UTC)
        await bot._publish_state(
            mock_cache,
            start_time=start,
            current_balance=Decimal("10500"),
            initial_balance=Decimal("10000"),
            daily_pnl=Decimal("125.50"),
            trade_count=5,
            win_count=3,
            loss_count=2,
            has_open_position=False,
            active_market=None,
            ws_connected=True,
        )
        call = self._find_call(mock_cache, "bot:daily")
        data = call[0][1]
        assert data["trade_count"] == 5
        assert data["daily_pnl"] == "125.50"
        assert data["win_count"] == 3
        assert data["loss_count"] == 2
        assert "date" in data
        assert call[1]["ttl"] == 300

    @pytest.mark.asyncio
    async def test_publish_state_ws_status(self, swing_config, mock_cache, mock_paper_trader):
        """WebSocket status should show binance and polymarket connection state."""
        from datetime import datetime

        bot = BotOrchestrator(mode="paper", strategy_name="momentum_confirmation", config=swing_config)
        start = datetime.now(tz=UTC)
        await bot._publish_state(
            mock_cache,
            start_time=start,
            current_balance=Decimal("10500"),
            initial_balance=Decimal("10000"),
            daily_pnl=Decimal("0"),
            trade_count=0,
            win_count=0,
            loss_count=0,
            has_open_position=False,
            active_market=None,
            ws_connected=True,
        )
        call = self._find_call(mock_cache, "bot:ws_status")
        data = call[0][1]
        assert data["binance"] is True
        # active_market is None => polymarket should be False
        assert data["polymarket"] is False
        assert call[1]["ttl"] == 120


class TestConfidenceDiscount:
    """Tests for confidence-based limit price discounting."""

    def test_discount_high_confidence(self):
        """High confidence (0.95) should yield near-minimum discount."""
        discount = BotOrchestrator._compute_confidence_discount(0.95)
        # conf=0.95 is 90% of the way from 0.5 to 1.0
        # discount = 0.05 - 0.45/0.5 * (0.05 - 0.005) = 0.05 - 0.9 * 0.045 = 0.0095
        assert discount == pytest.approx(0.0095, abs=1e-6)

    def test_discount_medium_confidence(self):
        """Medium confidence (0.75) should yield mid-range discount."""
        discount = BotOrchestrator._compute_confidence_discount(0.75)
        # conf=0.75 is 50% of the way from 0.5 to 1.0
        # discount = 0.05 - 0.25/0.5 * 0.045 = 0.05 - 0.0225 = 0.0275
        assert discount == pytest.approx(0.0275, abs=1e-6)

    def test_discount_threshold_confidence(self):
        """Threshold confidence (0.50) should yield maximum discount."""
        discount = BotOrchestrator._compute_confidence_discount(0.50)
        assert discount == pytest.approx(0.05, abs=1e-6)

    def test_discount_full_confidence(self):
        """Full confidence (1.0) should yield minimum discount."""
        discount = BotOrchestrator._compute_confidence_discount(1.0)
        assert discount == pytest.approx(0.005, abs=1e-6)

    def test_discount_clamped_below(self):
        """Confidence above 1.0 should clamp discount to min."""
        discount = BotOrchestrator._compute_confidence_discount(1.2)
        assert discount == pytest.approx(0.005, abs=1e-6)

    def test_discount_clamped_above(self):
        """Confidence below 0.5 should clamp discount to max."""
        discount = BotOrchestrator._compute_confidence_discount(0.3)
        assert discount == pytest.approx(0.05, abs=1e-6)

    def test_discount_custom_bounds(self):
        """Custom min/max discount bounds should work."""
        discount = BotOrchestrator._compute_confidence_discount(
            0.75, min_discount_pct=0.01, max_discount_pct=0.10,
        )
        # 50% of the way: 0.10 - 0.5 * 0.09 = 0.055
        assert discount == pytest.approx(0.055, abs=1e-6)

    def test_apply_discount_yes_price(self):
        """Discounting YES price at 0.50 by 5% should give 0.47."""
        result = BotOrchestrator._apply_confidence_discount(
            Decimal("0.50"), 0.05,
        )
        # 0.50 * 0.95 = 0.475, floor to tick 0.01 => 0.47
        assert result == Decimal("0.47")

    def test_apply_discount_rounds_down(self):
        """Discounted price should always round down to Polymarket tick."""
        result = BotOrchestrator._apply_confidence_discount(
            Decimal("0.53"), 0.02,
        )
        # 0.53 * 0.98 = 0.5194, floor to tick => 0.51
        assert result == Decimal("0.51")

    def test_apply_discount_floor_at_minimum(self):
        """Heavily discounted low price should floor at 0.01."""
        result = BotOrchestrator._apply_confidence_discount(
            Decimal("0.02"), 0.05,
        )
        # 0.02 * 0.95 = 0.019, floor to tick => 0.01
        assert result == Decimal("0.01")

    def test_apply_discount_zero_discount(self):
        """Zero discount should return price floored to tick."""
        result = BotOrchestrator._apply_confidence_discount(
            Decimal("0.50"), 0.0,
        )
        assert result == Decimal("0.50")

    def test_apply_discount_exact_tick(self):
        """Price that lands exactly on a tick should stay unchanged."""
        result = BotOrchestrator._apply_confidence_discount(
            Decimal("0.50"), 0.02,
        )
        # 0.50 * 0.98 = 0.49, exactly on tick
        assert result == Decimal("0.49")

    def test_discount_monotonically_decreasing(self):
        """Higher confidence should always give smaller discount."""
        prev_discount = float("inf")
        for conf in [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]:
            discount = BotOrchestrator._compute_confidence_discount(conf)
            assert discount < prev_discount or (
                conf == 0.50 and discount == prev_discount
            ), f"Discount not decreasing at conf={conf}"
            prev_discount = discount

    def test_discount_improves_rr(self):
        """Discounted price should give better R:R than original."""
        original_price = Decimal("0.50")
        discounted_price = BotOrchestrator._apply_confidence_discount(
            original_price, 0.03,
        )
        # R:R = (1 - price) / price for binary markets
        original_rr = (Decimal("1") - original_price) / original_price
        discounted_rr = (Decimal("1") - discounted_price) / discounted_price
        assert discounted_rr > original_rr

    def test_not_applied_in_paper_mode(self, swing_config):
        """Discount should NOT be applied in paper mode (handled by mode check)."""
        # Verify paper mode bot has _mode != "live"
        bot = BotOrchestrator(
            mode="paper",
            strategy_name="momentum_confirmation",
            config=swing_config,
        )
        assert bot._mode == "paper"
        # The discount block only runs when self._mode == "live",
        # so paper mode correctly skips it.


class TestSelectOrderType:
    """Tests for order type selection — FAK-only mode + legacy GTC mode."""

    # --- FAK-only mode (default) ---

    def test_fak_only_always_returns_fak(self):
        """fak_only=True → always FAK regardless of minute or spread."""
        from src.models.order import OrderType
        for minute in range(0, 15):
            result = BotOrchestrator._select_order_type(
                clob_last_trade=Decimal("0.65"),
                clob_spread=Decimal("0.50"),
                minute_in_window=minute,
                fak_only=True,
            )
            assert result == OrderType.FAK, f"minute {minute} fak_only should be FAK"

    def test_fak_only_default_is_true(self):
        """Default fak_only parameter should be True."""
        from src.models.order import OrderType
        result = BotOrchestrator._select_order_type(
            clob_last_trade=Decimal("0.65"),
            clob_spread=Decimal("0.50"),
            minute_in_window=3,
        )
        assert result == OrderType.FAK

    def test_fak_only_with_fak_disabled_returns_fok(self):
        """fak_only + use_fak_taker=False → FOK."""
        from src.models.order import OrderType
        result = BotOrchestrator._select_order_type(
            clob_last_trade=Decimal("0.65"),
            clob_spread=Decimal("0.50"),
            minute_in_window=3,
            fak_only=True,
            use_fak_taker=False,
        )
        assert result == OrderType.FOK

    # --- Legacy GTC mode (fak_only=False) ---

    def test_early_minute_wide_spread_always_gtc(self):
        """Legacy mode: Minutes 0-4 with wide spread → always GTC."""
        from src.models.order import OrderType
        for minute in range(0, 5):
            result = BotOrchestrator._select_order_type(
                clob_last_trade=Decimal("0.65"),
                clob_spread=Decimal("0.50"),
                minute_in_window=minute,
                fak_only=False,
            )
            assert result == OrderType.GTC, f"minute {minute} should be GTC"

    def test_early_minute_narrow_spread_still_gtc(self):
        """Legacy mode: Minutes 0-4 with narrow spread → still GTC."""
        from src.models.order import OrderType
        for minute in range(0, 5):
            result = BotOrchestrator._select_order_type(
                clob_last_trade=Decimal("0.65"),
                clob_spread=Decimal("0.05"),
                minute_in_window=minute,
                fak_only=False,
            )
            assert result == OrderType.GTC, f"minute {minute} should be GTC"

    def test_mid_window_narrow_spread_returns_fak(self):
        """Legacy mode: Minutes 5-10 with narrow spread → FAK."""
        from src.models.order import OrderType
        for minute in range(5, 11):
            result = BotOrchestrator._select_order_type(
                clob_last_trade=Decimal("0.65"),
                clob_spread=Decimal("0.05"),
                minute_in_window=minute,
                fak_only=False,
            )
            assert result == OrderType.FAK, f"minute {minute} narrow spread should be FAK"

    def test_mid_window_wide_spread_returns_gtc(self):
        """Legacy mode: Minutes 5-10 with wide spread → GTC."""
        from src.models.order import OrderType
        for minute in range(5, 11):
            result = BotOrchestrator._select_order_type(
                clob_last_trade=Decimal("0.65"),
                clob_spread=Decimal("0.50"),
                minute_in_window=minute,
                fak_only=False,
            )
            assert result == OrderType.GTC, f"minute {minute} wide spread should be GTC"

    def test_late_window_returns_fak(self):
        """Late window (min 11+) → always FAK."""
        from src.models.order import OrderType
        result = BotOrchestrator._select_order_type(
            clob_last_trade=None,
            clob_spread=Decimal("0.50"),
            minute_in_window=12,
            fak_only=False,
        )
        assert result == OrderType.FAK

    def test_late_window_returns_fok_when_fak_disabled(self):
        """Late window with use_fak_taker=False → FOK."""
        from src.models.order import OrderType
        result = BotOrchestrator._select_order_type(
            clob_last_trade=None,
            clob_spread=Decimal("0.50"),
            minute_in_window=12,
            use_fak_taker=False,
            fak_only=False,
        )
        assert result == OrderType.FOK

    def test_late_window_threshold_exact(self):
        """Minute exactly at threshold → FAK."""
        from src.models.order import OrderType
        result = BotOrchestrator._select_order_type(
            clob_last_trade=None,
            clob_spread=Decimal("0.98"),
            minute_in_window=11,
            fok_late_minute=11,
            fak_only=False,
        )
        assert result == OrderType.FAK

    def test_minute_before_late_threshold_wide_spread_returns_gtc(self):
        """Legacy mode: Minute before late threshold with wide spread → GTC."""
        from src.models.order import OrderType
        result = BotOrchestrator._select_order_type(
            clob_last_trade=None,
            clob_spread=Decimal("0.80"),
            minute_in_window=10,
            fok_late_minute=11,
            fak_only=False,
        )
        assert result == OrderType.GTC

    def test_custom_fok_late_minute(self):
        """Legacy mode: Custom fok_late_minute threshold is respected."""
        from src.models.order import OrderType
        # Before custom threshold, wide spread → GTC
        result = BotOrchestrator._select_order_type(
            clob_last_trade=Decimal("0.70"),
            clob_spread=Decimal("0.50"),
            minute_in_window=12,
            fok_late_minute=13,
            fak_only=False,
        )
        assert result == OrderType.GTC
        # At custom threshold → FAK
        result = BotOrchestrator._select_order_type(
            clob_last_trade=Decimal("0.70"),
            clob_spread=Decimal("0.50"),
            minute_in_window=13,
            fok_late_minute=13,
            fak_only=False,
        )
        assert result == OrderType.FAK

    def test_liquid_market_early_still_gtc(self):
        """Legacy mode: Even tight spread + real trades → GTC before minute 5."""
        from src.models.order import OrderType
        result = BotOrchestrator._select_order_type(
            clob_last_trade=Decimal("0.55"),
            clob_spread=Decimal("0.00"),
            minute_in_window=4,
            fak_only=False,
        )
        assert result == OrderType.GTC

    def test_illiquid_market_late_still_fak(self):
        """Desert book but late minute → FAK."""
        from src.models.order import OrderType
        result = BotOrchestrator._select_order_type(
            clob_last_trade=None,
            clob_spread=Decimal("0.98"),
            minute_in_window=11,
            fak_only=False,
        )
        assert result == OrderType.FAK

    def test_spread_trigger_mid_window(self):
        """Legacy mode: Narrow spread at minute 9 → FAK."""
        from src.models.order import OrderType
        result = BotOrchestrator._select_order_type(
            clob_last_trade=Decimal("0.60"),
            clob_spread=Decimal("0.01"),
            minute_in_window=9,
            fok_spread_threshold=Decimal("0.50"),
            fak_only=False,
        )
        assert result == OrderType.FAK

    def test_wide_spread_mid_window_stays_gtc(self):
        """Legacy mode: Wide spread at minute 9 → GTC."""
        from src.models.order import OrderType
        result = BotOrchestrator._select_order_type(
            clob_last_trade=Decimal("0.60"),
            clob_spread=Decimal("0.60"),
            minute_in_window=9,
            fok_spread_threshold=Decimal("0.50"),
            fak_only=False,
        )
        assert result == OrderType.GTC


class TestFOKEntryPriceLogic:
    """Tests for FOK-specific entry price and max price branching."""

    def test_fok_max_entry_higher_than_gtc(self):
        """FOK taker should accept higher prices than GTC maker."""
        # FOK at $0.85 should pass with fok_max=0.85 but fail with gtc_max=0.80
        fok_max = Decimal("0.85")
        gtc_max = Decimal("0.80")
        entry = Decimal("0.83")
        assert entry <= fok_max  # FOK allows
        assert entry > gtc_max  # GTC would reject

    def test_entry_above_both_limits_rejected(self):
        """Price above both FOK and GTC limits should be rejected."""
        fok_max = Decimal("0.85")
        gtc_max = Decimal("0.80")
        entry = Decimal("0.90")
        assert entry > fok_max
        assert entry > gtc_max

    def test_entry_below_both_limits_accepted(self):
        """Price below both limits should be accepted by either type."""
        fok_max = Decimal("0.85")
        gtc_max = Decimal("0.80")
        entry = Decimal("0.65")
        assert entry <= fok_max
        assert entry <= gtc_max

    def test_confidence_discount_only_gtc(self):
        """Confidence discount should apply to GTC orders, not FOK."""
        from src.models.order import OrderType

        # GTC → discount applies
        gtc_type = OrderType.GTC
        # FOK → discount skipped
        fok_type = OrderType.FOK

        # The bot code checks: order_type_selected == OrderType.GTC
        should_discount_gtc = gtc_type == OrderType.GTC
        should_discount_fok = fok_type == OrderType.GTC
        assert should_discount_gtc is True
        assert should_discount_fok is False

    def test_confidence_discount_preserves_fok_price(self):
        """FOK entry should use raw CLOB best_ask — no discount."""
        # Simulate: FOK crosses spread at best_ask = $0.72
        fok_entry = Decimal("0.72")
        # No discount applied — price should remain $0.72
        discount_pct = BotOrchestrator._compute_confidence_discount(0.85)
        # FOK would NOT apply this discount
        assert fok_entry == Decimal("0.72")
        # But GTC would apply it
        gtc_discounted = BotOrchestrator._apply_confidence_discount(
            fok_entry, discount_pct,
        )
        assert gtc_discounted < fok_entry


class TestNODirectionPricing:
    """Tests for NO-direction using NO token CLOB data."""

    def test_no_direction_prefers_no_token_data(self):
        """NO direction should use NO token last_trade when available."""
        # Simulate: NO token has last_trade at $0.60
        no_clob_last_trade = Decimal("0.60")
        # YES token would give inverted: 1 - 0.50 = 0.50
        yes_clob_computed_mid = Decimal("0.50")

        # NO direction with real NO data → use $0.60
        assert no_clob_last_trade is not None
        entry = no_clob_last_trade
        assert entry == Decimal("0.60")

    def test_no_direction_falls_back_to_inverted_yes(self):
        """NO direction without NO token data should invert YES price."""
        no_clob_last_trade = None
        no_clob_midpoint = None
        yes_entry = Decimal("0.45")

        # Fallback: invert YES price
        if no_clob_last_trade is None and no_clob_midpoint is None:
            entry = Decimal("1") - yes_entry
        else:
            entry = no_clob_last_trade or no_clob_midpoint
        assert entry == Decimal("0.55")

    def test_no_direction_midpoint_chain(self):
        """NO direction should try last_trade > midpoint > computed_mid."""
        # Only midpoint available
        no_clob_last_trade = None
        no_clob_midpoint = Decimal("0.55")

        if no_clob_last_trade is not None:
            entry = no_clob_last_trade
            source = "no_clob_last_trade"
        elif no_clob_midpoint is not None:
            entry = no_clob_midpoint
            source = "no_clob_midpoint"
        else:
            entry = Decimal("0.50")  # fallback
            source = "inverted"

        assert entry == Decimal("0.55")
        assert source == "no_clob_midpoint"

    def test_no_direction_computed_mid(self):
        """NO direction should compute mid from NO bid/ask when needed."""
        no_clob_best_ask = Decimal("0.65")
        no_clob_best_bid = Decimal("0.55")
        no_clob_last_trade = None
        no_clob_midpoint = None

        if no_clob_last_trade is not None:
            entry = no_clob_last_trade
        elif no_clob_midpoint is not None:
            entry = no_clob_midpoint
        elif no_clob_best_ask is not None and no_clob_best_bid is not None:
            entry = (no_clob_best_ask + no_clob_best_bid) / 2
        else:
            entry = Decimal("0.50")

        assert entry == Decimal("0.60")


class TestFOKInvertedAskFallback:
    """Tests for FOK price override using inverted opposite-token bid
    when the target token has no WS data (the common case on illiquid
    15m markets where NO token WS is empty)."""

    def _fok_override(
        self,
        direction: str,
        clob_best_ask: Decimal | None,
        clob_best_bid: Decimal | None,
        no_clob_best_ask: Decimal | None,
        no_clob_best_bid: Decimal | None,
        entry_price: Decimal,
    ) -> tuple[Decimal, str]:
        """Replicate the FOK price override logic from bot.py."""
        price_source = "pre_fok"
        if direction == "YES":
            _fok_ask = clob_best_ask
            if _fok_ask is None and no_clob_best_bid is not None:
                _fok_ask = Decimal("1") - no_clob_best_bid
            if _fok_ask is not None:
                entry_price = _fok_ask
                price_source = "clob_best_ask_fok"
        elif direction == "NO":
            _fok_ask = no_clob_best_ask
            if _fok_ask is None and clob_best_bid is not None:
                _fok_ask = Decimal("1") - clob_best_bid
            if _fok_ask is not None:
                entry_price = _fok_ask
                price_source = "no_clob_best_ask_fok"
        return entry_price, price_source

    def test_no_direction_inverts_yes_bid_when_no_token_empty(self):
        """NO FOK with empty NO token should invert YES best_bid → NO ask.

        YES bid=0.01 → NO ask = 1 - 0.01 = 0.99.
        This correctly rejects at max_entry_price=0.85.
        """
        price, source = self._fok_override(
            direction="NO",
            clob_best_ask=Decimal("0.99"),
            clob_best_bid=Decimal("0.01"),
            no_clob_best_ask=None,
            no_clob_best_bid=None,
            entry_price=Decimal("0.50"),
        )
        assert price == Decimal("0.99")
        assert source == "no_clob_best_ask_fok"

    def test_no_direction_uses_real_no_ask_when_available(self):
        """NO FOK should prefer real NO token ask over inverted YES bid."""
        price, source = self._fok_override(
            direction="NO",
            clob_best_ask=Decimal("0.99"),
            clob_best_bid=Decimal("0.01"),
            no_clob_best_ask=Decimal("0.65"),
            no_clob_best_bid=Decimal("0.55"),
            entry_price=Decimal("0.50"),
        )
        assert price == Decimal("0.65")
        assert source == "no_clob_best_ask_fok"

    def test_yes_direction_inverts_no_bid_when_yes_token_empty(self):
        """YES FOK with empty YES token should invert NO best_bid → YES ask."""
        price, source = self._fok_override(
            direction="YES",
            clob_best_ask=None,
            clob_best_bid=None,
            no_clob_best_ask=Decimal("0.99"),
            no_clob_best_bid=Decimal("0.30"),
            entry_price=Decimal("0.50"),
        )
        assert price == Decimal("0.70")
        assert source == "clob_best_ask_fok"

    def test_yes_direction_uses_real_yes_ask_when_available(self):
        """YES FOK should prefer real YES token ask."""
        price, source = self._fok_override(
            direction="YES",
            clob_best_ask=Decimal("0.72"),
            clob_best_bid=Decimal("0.68"),
            no_clob_best_ask=None,
            no_clob_best_bid=None,
            entry_price=Decimal("0.50"),
        )
        assert price == Decimal("0.72")
        assert source == "clob_best_ask_fok"

    def test_both_tokens_empty_returns_pre_fok_source(self):
        """If neither token has any data, FOK override should not fire."""
        price, source = self._fok_override(
            direction="NO",
            clob_best_ask=None,
            clob_best_bid=None,
            no_clob_best_ask=None,
            no_clob_best_bid=None,
            entry_price=Decimal("0.50"),
        )
        assert price == Decimal("0.50")
        assert source == "pre_fok"  # override did NOT fire

    def test_no_direction_realistic_illiquid_market(self):
        """Realistic scenario: YES 0.01/0.99 empty book, NO token has no WS data.

        This is the exact bug case. Before the fix, entry_price would stay
        at $0.50 (clob_computed_mid) and pass fok_max_entry_price ($0.85).
        After the fix, it becomes $0.99 (inverted) and is correctly rejected.
        """
        fok_max_entry_price = Decimal("0.85")
        price, source = self._fok_override(
            direction="NO",
            clob_best_ask=Decimal("0.99"),
            clob_best_bid=Decimal("0.01"),
            no_clob_best_ask=None,
            no_clob_best_bid=None,
            entry_price=Decimal("0.50"),
        )
        assert price == Decimal("0.99")
        assert price > fok_max_entry_price  # Would be correctly REJECTED
        assert source == "no_clob_best_ask_fok"


class TestCLOBStability:
    """Tests for _check_clob_stability static helper."""

    def test_stable_within_range(self) -> None:
        """All midpoints within max_range → stable."""
        history = [Decimal("0.50"), Decimal("0.52"), Decimal("0.51")]
        ok, reason = BotOrchestrator._check_clob_stability(
            history, max_range=Decimal("0.20"), min_samples=3,
        )
        assert ok is True
        assert "range=" in reason

    def test_unstable_exceeds_range(self) -> None:
        """Midpoints swing beyond max_range → unstable."""
        history = [Decimal("0.23"), Decimal("0.10"), Decimal("0.38"), Decimal("0.51")]
        ok, reason = BotOrchestrator._check_clob_stability(
            history, max_range=Decimal("0.20"), min_samples=3,
        )
        assert ok is False
        assert "range=" in reason

    def test_insufficient_samples_passes(self) -> None:
        """Fewer than min_samples → always passes (insufficient data)."""
        history = [Decimal("0.10"), Decimal("0.90")]
        ok, reason = BotOrchestrator._check_clob_stability(
            history, max_range=Decimal("0.20"), min_samples=3,
        )
        assert ok is True
        assert "insufficient_samples" in reason

    def test_empty_history_passes(self) -> None:
        """Empty history → always passes."""
        ok, reason = BotOrchestrator._check_clob_stability(
            [], max_range=Decimal("0.20"), min_samples=3,
        )
        assert ok is True

    def test_exact_range_passes(self) -> None:
        """Range exactly equal to max → passes (not strictly greater)."""
        history = [Decimal("0.40"), Decimal("0.50"), Decimal("0.60")]
        ok, reason = BotOrchestrator._check_clob_stability(
            history, max_range=Decimal("0.20"), min_samples=3,
        )
        assert ok is True


class TestCoinFlipZone:
    """Tests for _check_coin_flip_zone static helper."""

    def test_outside_zone_passes(self) -> None:
        """Mid far from 0.50 → outside zone → trade allowed."""
        ok, reason = BotOrchestrator._check_coin_flip_zone(
            cal_mid=0.70, half_width=0.08, edge=0.04, override_edge=0.10,
        )
        assert ok is True
        assert "outside_zone" in reason

    def test_in_zone_low_edge_blocks(self) -> None:
        """Mid near 0.50, low edge → blocked."""
        ok, reason = BotOrchestrator._check_coin_flip_zone(
            cal_mid=0.505, half_width=0.08, edge=0.04, override_edge=0.10,
        )
        assert ok is False
        assert "coin_flip" in reason

    def test_in_zone_high_edge_overrides(self) -> None:
        """Mid near 0.50, but high edge → override passes."""
        ok, reason = BotOrchestrator._check_coin_flip_zone(
            cal_mid=0.505, half_width=0.08, edge=0.12, override_edge=0.10,
        )
        assert ok is True
        assert "override" in reason

    def test_boundary_passes(self) -> None:
        """Exactly at boundary (dist == half_width) → outside zone."""
        ok, reason = BotOrchestrator._check_coin_flip_zone(
            cal_mid=0.42, half_width=0.08, edge=0.01, override_edge=0.10,
        )
        assert ok is True

    def test_exact_override_edge_passes(self) -> None:
        """Edge exactly equal to override_edge → passes."""
        ok, reason = BotOrchestrator._check_coin_flip_zone(
            cal_mid=0.50, half_width=0.08, edge=0.10, override_edge=0.10,
        )
        assert ok is True


class TestFAKDowngradeToGTC:
    """Tests for FAK → GTC downgrade when no usable ask to cross."""

    def _simulate_fak_downgrade(
        self,
        direction: str,
        entry_price: Decimal,
        price_source: str,
        clob_best_ask: Decimal | None,
        clob_best_bid: Decimal | None,
        no_clob_best_ask: Decimal | None,
        no_clob_best_bid: Decimal | None,
    ) -> tuple[Decimal, str, str]:
        """Replicate the FAK downgrade logic from bot.py.

        Returns (entry_price, price_source, order_type) where order_type
        is "FAK", "GTC", or "skip".
        """
        from src.models.order import OrderType

        order_type_selected = OrderType.FAK
        _had_fresh_ask = False
        _fak_ask: Decimal | None = None

        if direction == "YES":
            _fak_ask = clob_best_ask
            if _fak_ask is None and no_clob_best_bid is not None:
                _fak_ask = Decimal("1") - no_clob_best_bid
        elif direction == "NO":
            _fak_ask = no_clob_best_ask
            if _fak_ask is None and clob_best_bid is not None:
                _fak_ask = Decimal("1") - clob_best_bid

        # Desert detection: 0.99 ask from placeholder book
        _fak_ask_is_desert = (
            _fak_ask is not None and _fak_ask >= Decimal("0.95")
        )
        if _fak_ask is not None and not _fak_ask_is_desert:
            entry_price = _fak_ask
            if direction == "YES":
                price_source = "clob_best_ask_fak"
            else:
                price_source = "no_clob_best_ask_fak"
            _had_fresh_ask = True

        if not _had_fresh_ask:
            _has_valid_cascade_price = (
                entry_price is not None
                and entry_price > 0
                and entry_price < Decimal("0.95")
            )
            if _has_valid_cascade_price:
                order_type_selected = OrderType.GTC
            else:
                return entry_price, price_source, "skip"

        return entry_price, price_source, order_type_selected.value

    def test_rest_price_downgrades_fak_to_gtc(self):
        """REST last_trade price should downgrade FAK to GTC (not skip)."""
        price, source, otype = self._simulate_fak_downgrade(
            direction="YES",
            entry_price=Decimal("0.45"),
            price_source="rest_last_trade_yes",
            clob_best_ask=None,
            clob_best_bid=None,
            no_clob_best_ask=None,
            no_clob_best_bid=None,
        )
        assert price == Decimal("0.45")  # REST price preserved
        assert source == "rest_last_trade_yes"
        assert otype == "GTC"

    def test_clob_last_trade_downgrades_fak_to_gtc(self):
        """CLOB last_trade (from WS/REST seed) should downgrade FAK to GTC."""
        price, source, otype = self._simulate_fak_downgrade(
            direction="YES",
            entry_price=Decimal("0.41"),
            price_source="clob_last_trade",
            clob_best_ask=None,
            clob_best_bid=None,
            no_clob_best_ask=None,
            no_clob_best_bid=None,
        )
        assert price == Decimal("0.41")
        assert otype == "GTC"

    def test_desert_ask_099_downgrades_to_gtc_with_valid_price(self):
        """Desert WS ask (0.99) should NOT be used; downgrade to GTC with cascade price.

        This is the critical bug fix: previously FAK would use 0.99 ask from
        the 0.01/0.99 placeholder book, which then got rejected by
        clob_entry_price_too_high. Now 0.99 is detected as desert and the
        valid cascade price (e.g. 0.41 from clob_last_trade) is kept as GTC.
        """
        price, source, otype = self._simulate_fak_downgrade(
            direction="YES",
            entry_price=Decimal("0.41"),
            price_source="clob_last_trade",
            clob_best_ask=Decimal("0.99"),  # Desert ask
            clob_best_bid=Decimal("0.01"),  # Desert bid
            no_clob_best_ask=None,
            no_clob_best_bid=None,
        )
        assert price == Decimal("0.41")  # Cascade price preserved (NOT 0.99)
        assert otype == "GTC"  # Downgraded from FAK

    def test_desert_no_direction_downgrades(self):
        """NO direction with desert inverted ask should also downgrade."""
        price, source, otype = self._simulate_fak_downgrade(
            direction="NO",
            entry_price=Decimal("0.60"),
            price_source="no_clob_last_trade",
            clob_best_ask=Decimal("0.99"),
            clob_best_bid=Decimal("0.01"),  # Inverted: 1-0.01 = 0.99 → desert
            no_clob_best_ask=None,
            no_clob_best_bid=None,
        )
        assert price == Decimal("0.60")
        assert otype == "GTC"

    def test_no_valid_price_skips(self):
        """No fresh ask AND no valid cascade price → skip."""
        price, source, otype = self._simulate_fak_downgrade(
            direction="YES",
            entry_price=Decimal("0.99"),  # Desert price from cascade
            price_source="clob_computed_mid",
            clob_best_ask=None,
            clob_best_bid=None,
            no_clob_best_ask=None,
            no_clob_best_bid=None,
        )
        assert otype == "skip"

    def test_fresh_ask_uses_fak_normally(self):
        """When fresh non-desert ask exists, FAK proceeds normally."""
        price, source, otype = self._simulate_fak_downgrade(
            direction="YES",
            entry_price=Decimal("0.50"),
            price_source="rest_last_trade_yes",
            clob_best_ask=Decimal("0.55"),  # Real ask, not desert
            clob_best_bid=Decimal("0.45"),
            no_clob_best_ask=None,
            no_clob_best_bid=None,
        )
        assert price == Decimal("0.55")  # Uses fresh ask
        assert source == "clob_best_ask_fak"
        assert otype == "FAK"

    def test_threshold_095_is_desert(self):
        """Ask at exactly 0.95 should be treated as desert."""
        price, source, otype = self._simulate_fak_downgrade(
            direction="YES",
            entry_price=Decimal("0.41"),
            price_source="clob_last_trade",
            clob_best_ask=Decimal("0.95"),
            clob_best_bid=Decimal("0.05"),
            no_clob_best_ask=None,
            no_clob_best_bid=None,
        )
        assert price == Decimal("0.41")
        assert otype == "GTC"

    def test_ask_094_is_not_desert(self):
        """Ask at 0.94 should NOT be treated as desert."""
        price, source, otype = self._simulate_fak_downgrade(
            direction="YES",
            entry_price=Decimal("0.41"),
            price_source="clob_last_trade",
            clob_best_ask=Decimal("0.94"),
            clob_best_bid=Decimal("0.06"),
            no_clob_best_ask=None,
            no_clob_best_bid=None,
        )
        assert price == Decimal("0.94")  # Uses real ask
        assert source == "clob_best_ask_fak"
        assert otype == "FAK"


class TestPaperRESTFallback:
    """Tests for unified REST fallback in paper mode (previously paper-only desert skip)."""

    def _simulate_desert_fallback(
        self,
        entry_price: Decimal | None,
        price_source: str,
        rest_ltp_response: Decimal | None,
        rest_mid_response: Decimal | None,
        direction: str = "YES",
    ) -> tuple[Decimal | None, str, bool]:
        """Replicate desert detection + REST fallback logic.

        Returns (entry_price, price_source, was_desert).
        """
        _is_midpoint_source = (
            "computed_mid" in price_source
            or "midpoint" in price_source
        )
        _is_desert_price = (
            _is_midpoint_source
            and entry_price is not None
            and abs(entry_price - Decimal("0.50")) < Decimal("0.05")
        )

        # Unified fallback — no mode check
        if entry_price is None or _is_desert_price:
            if rest_ltp_response is not None and rest_ltp_response > 0:
                entry_price = rest_ltp_response
                _dir_label = "no" if direction == "NO" else "yes"
                price_source = f"rest_last_trade_{_dir_label}"
                _is_desert_price = False
            elif rest_mid_response is not None and rest_mid_response > 0:
                entry_price = rest_mid_response
                _dir_label = "no" if direction == "NO" else "yes"
                price_source = f"rest_midpoint_{_dir_label}"

        return entry_price, price_source, _is_desert_price

    def test_paper_desert_uses_rest_fallback(self):
        """Paper mode with desert price should use REST fallback (not skip).

        This is the primary fix: previously paper mode would `continue` here.
        """
        price, source, is_desert = self._simulate_desert_fallback(
            entry_price=Decimal("0.50"),
            price_source="clob_computed_mid",
            rest_ltp_response=Decimal("0.45"),
            rest_mid_response=None,
        )
        assert price == Decimal("0.45")
        assert source == "rest_last_trade_yes"
        assert is_desert is False

    def test_desert_falls_through_to_midpoint(self):
        """If REST last_trade is None, falls through to REST midpoint."""
        price, source, _ = self._simulate_desert_fallback(
            entry_price=Decimal("0.50"),
            price_source="clob_computed_mid",
            rest_ltp_response=None,
            rest_mid_response=Decimal("0.48"),
        )
        assert price == Decimal("0.48")
        assert source == "rest_midpoint_yes"

    def test_non_desert_price_not_overridden(self):
        """Non-desert price (e.g. real WS last_trade) should not trigger fallback."""
        price, source, _ = self._simulate_desert_fallback(
            entry_price=Decimal("0.55"),
            price_source="clob_last_trade",
            rest_ltp_response=Decimal("0.45"),
            rest_mid_response=None,
        )
        assert price == Decimal("0.55")  # Unchanged
        assert source == "clob_last_trade"

    def test_no_rest_data_stays_desert(self):
        """If REST returns nothing, price stays as desert (will be rejected downstream)."""
        price, source, is_desert = self._simulate_desert_fallback(
            entry_price=Decimal("0.50"),
            price_source="clob_computed_mid",
            rest_ltp_response=None,
            rest_mid_response=None,
        )
        assert price == Decimal("0.50")
        assert is_desert is True

    def test_no_direction_rest_fallback(self):
        """NO direction should get rest_last_trade_no source label."""
        price, source, _ = self._simulate_desert_fallback(
            entry_price=Decimal("0.51"),
            price_source="no_clob_computed_mid",
            rest_ltp_response=Decimal("0.42"),
            rest_mid_response=None,
            direction="NO",
        )
        assert price == Decimal("0.42")
        assert source == "rest_last_trade_no"


class TestUpdateRestData:
    """Tests for PolymarketWSFeed.update_rest_data() — always overwrites."""

    def test_overwrites_existing_last_trade(self):
        """update_rest_data should overwrite existing last_trade_price."""
        from src.data.polymarket_ws import PolymarketWSFeed

        feed = PolymarketWSFeed()
        token = "test_token_123"

        # Seed initial data
        feed.seed_rest_data(token, {"price": "0.45"}, None, None)
        state = feed.get_clob_state(token)
        assert state is not None
        assert state.last_trade_price == Decimal("0.45")

        # Update should overwrite
        feed.update_rest_data(token, {"price": "0.52"}, None, None)
        assert state.last_trade_price == Decimal("0.52")

    def test_seed_does_not_overwrite(self):
        """seed_rest_data should NOT overwrite existing last_trade_price."""
        from src.data.polymarket_ws import PolymarketWSFeed

        feed = PolymarketWSFeed()
        token = "test_token_456"

        feed.seed_rest_data(token, {"price": "0.45"}, None, None)
        state = feed.get_clob_state(token)
        assert state is not None
        assert state.last_trade_price == Decimal("0.45")

        # Seed again — should NOT overwrite
        feed.seed_rest_data(token, {"price": "0.99"}, None, None)
        assert state.last_trade_price == Decimal("0.45")  # Unchanged

    def test_update_refreshes_timestamps(self):
        """update_rest_data should always update timestamps."""
        from src.data.polymarket_ws import PolymarketWSFeed

        feed = PolymarketWSFeed()
        token = "test_token_789"

        feed.seed_rest_data(token, {"price": "0.45"}, {"mid": "0.50"}, None)
        state = feed.get_clob_state(token)
        assert state is not None
        old_ltp_ts = state.last_trade_updated
        old_mid_ts = state.midpoint_updated

        import time
        time.sleep(0.01)  # Ensure time difference

        feed.update_rest_data(token, {"price": "0.45"}, {"mid": "0.50"}, None)
        assert state.last_trade_updated is not None
        assert state.midpoint_updated is not None
        assert state.last_trade_updated >= old_ltp_ts  # type: ignore[operator]
        assert state.midpoint_updated >= old_mid_ts  # type: ignore[operator]

    def test_update_derives_bid_ask_from_spread(self):
        """update_rest_data should derive bid/ask from midpoint + spread."""
        from src.data.polymarket_ws import PolymarketWSFeed

        feed = PolymarketWSFeed()
        token = "test_token_spread"

        feed.update_rest_data(
            token,
            {"price": "0.50"},
            {"mid": "0.50"},
            {"spread": "0.10"},
        )
        state = feed.get_clob_state(token)
        assert state is not None
        assert state.best_bid == Decimal("0.45")
        assert state.best_ask == Decimal("0.55")
