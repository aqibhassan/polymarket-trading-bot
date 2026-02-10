"""Tests for KillSwitch — circuit breaker for daily drawdown."""

from __future__ import annotations

import json
from decimal import Decimal
from unittest.mock import MagicMock

from src.risk.kill_switch import KillSwitch


class TestKillSwitchBasic:
    def test_initially_inactive(self) -> None:
        ks = KillSwitch()
        assert ks.is_active is False

    def test_trigger_activates(self) -> None:
        ks = KillSwitch()
        ks.trigger("test reason")
        assert ks.is_active is True

    def test_reset_deactivates(self) -> None:
        ks = KillSwitch()
        ks.trigger("test")
        assert ks.is_active is True
        ks.reset()
        assert ks.is_active is False

    def test_trigger_then_reset_then_trigger(self) -> None:
        ks = KillSwitch()
        ks.trigger("first")
        ks.reset()
        assert ks.is_active is False
        ks.trigger("second")
        assert ks.is_active is True


class TestKillSwitchCheck:
    def test_no_trigger_on_profit(self) -> None:
        ks = KillSwitch(max_daily_drawdown_pct=Decimal("0.05"))
        result = ks.check(daily_pnl=Decimal("100"), balance=Decimal("10000"))
        assert result is False
        assert ks.is_active is False

    def test_no_trigger_below_threshold(self) -> None:
        ks = KillSwitch(max_daily_drawdown_pct=Decimal("0.05"))
        # 4% drawdown, threshold is 5%
        result = ks.check(daily_pnl=Decimal("-400"), balance=Decimal("10000"))
        assert result is False
        assert ks.is_active is False

    def test_trigger_at_threshold(self) -> None:
        ks = KillSwitch(max_daily_drawdown_pct=Decimal("0.05"))
        # Exactly 5% drawdown
        result = ks.check(daily_pnl=Decimal("-500"), balance=Decimal("10000"))
        assert result is True
        assert ks.is_active is True

    def test_trigger_above_threshold(self) -> None:
        ks = KillSwitch(max_daily_drawdown_pct=Decimal("0.05"))
        result = ks.check(daily_pnl=Decimal("-700"), balance=Decimal("10000"))
        assert result is True
        assert ks.is_active is True

    def test_already_active_returns_true(self) -> None:
        ks = KillSwitch()
        ks.trigger("manual")
        result = ks.check(daily_pnl=Decimal("100"), balance=Decimal("10000"))
        assert result is True

    def test_zero_balance_no_trigger(self) -> None:
        ks = KillSwitch()
        result = ks.check(daily_pnl=Decimal("-100"), balance=Decimal("0"))
        assert result is False
        assert ks.is_active is False

    def test_zero_pnl_no_trigger(self) -> None:
        ks = KillSwitch()
        result = ks.check(daily_pnl=Decimal("0"), balance=Decimal("10000"))
        assert result is False


class TestKillSwitchRedis:
    def test_save_state_to_redis(self) -> None:
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        ks = KillSwitch(redis_client=mock_redis)
        ks.trigger("redis test")
        mock_redis.set.assert_called()
        saved_data = json.loads(mock_redis.set.call_args[0][1])
        assert saved_data["active"] is True
        assert saved_data["reason"] == "redis test"

    def test_load_state_from_redis(self) -> None:
        mock_redis = MagicMock()
        state = json.dumps({
            "active": True,
            "reason": "persisted",
            "triggered_at": "2025-01-01T00:00:00+00:00",
        })
        mock_redis.get.return_value = state
        ks = KillSwitch(redis_client=mock_redis)
        assert ks.is_active is True

    def test_load_from_redis_inactive(self) -> None:
        mock_redis = MagicMock()
        state = json.dumps({
            "active": False,
            "reason": "",
            "triggered_at": None,
        })
        mock_redis.get.return_value = state
        ks = KillSwitch(redis_client=mock_redis)
        assert ks.is_active is False

    def test_redis_load_failure_falls_back(self) -> None:
        mock_redis = MagicMock()
        mock_redis.get.side_effect = ConnectionError("no redis")
        ks = KillSwitch(redis_client=mock_redis)
        assert ks.is_active is False

    def test_redis_save_failure_no_crash(self) -> None:
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        mock_redis.set.side_effect = ConnectionError("no redis")
        ks = KillSwitch(redis_client=mock_redis)
        ks.trigger("test")
        assert ks.is_active is True

    def test_reset_saves_to_redis(self) -> None:
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        ks = KillSwitch(redis_client=mock_redis)
        ks.trigger("test")
        mock_redis.set.reset_mock()
        ks.reset()
        mock_redis.set.assert_called_once()
        saved_data = json.loads(mock_redis.set.call_args[0][1])
        assert saved_data["active"] is False

    def test_redis_none_no_persistence(self) -> None:
        ks = KillSwitch(redis_client=None)
        ks.trigger("no redis")
        assert ks.is_active is True
        ks.reset()
        assert ks.is_active is False

    def test_redis_returns_none_on_get(self) -> None:
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        ks = KillSwitch(redis_client=mock_redis)
        assert ks.is_active is False


class TestConsecutiveLosses:
    def test_win_resets_counter(self) -> None:
        ks = KillSwitch(max_consecutive_losses=3)
        ks.record_trade_result(is_win=False)
        ks.record_trade_result(is_win=False)
        assert ks.consecutive_losses == 2
        ks.record_trade_result(is_win=True)
        assert ks.consecutive_losses == 0

    def test_triggers_at_limit(self) -> None:
        ks = KillSwitch(max_consecutive_losses=3)
        ks.record_trade_result(is_win=False)
        ks.record_trade_result(is_win=False)
        assert ks.is_active is False
        triggered = ks.record_trade_result(is_win=False)
        assert triggered is True
        assert ks.is_active is True
        assert ks.consecutive_losses == 3

    def test_no_trigger_below_limit(self) -> None:
        ks = KillSwitch(max_consecutive_losses=5)
        for _ in range(4):
            triggered = ks.record_trade_result(is_win=False)
            assert triggered is False
        assert ks.is_active is False

    def test_disabled_when_zero(self) -> None:
        """max_consecutive_losses=0 disables the feature."""
        ks = KillSwitch(max_consecutive_losses=0)
        for _ in range(100):
            triggered = ks.record_trade_result(is_win=False)
            assert triggered is False
        assert ks.is_active is False

    def test_win_after_losses_prevents_trigger(self) -> None:
        ks = KillSwitch(max_consecutive_losses=3)
        ks.record_trade_result(is_win=False)
        ks.record_trade_result(is_win=False)
        ks.record_trade_result(is_win=True)
        ks.record_trade_result(is_win=False)
        ks.record_trade_result(is_win=False)
        assert ks.is_active is False
        assert ks.consecutive_losses == 2

    def test_persists_to_redis(self) -> None:
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        ks = KillSwitch(redis_client=mock_redis, max_consecutive_losses=5)
        ks.record_trade_result(is_win=False)
        mock_redis.set.assert_called()
        # Check that the consecutive loss key was persisted
        calls = [c for c in mock_redis.set.call_args_list
                 if KillSwitch.CONSECUTIVE_LOSS_KEY in str(c)]
        assert len(calls) > 0
