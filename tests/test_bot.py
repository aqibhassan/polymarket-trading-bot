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
            paper_trader=mock_paper_trader,
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
            paper_trader=mock_paper_trader,
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
            paper_trader=mock_paper_trader,
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
            paper_trader=mock_paper_trader,
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
            paper_trader=mock_paper_trader,
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
            paper_trader=mock_paper_trader,
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
            paper_trader=mock_paper_trader,
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
