"""Tests for BaseStrategy."""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from src.config.loader import ConfigError, ConfigLoader
from src.models.market import MarketState, OrderBookSnapshot, Position, Side
from src.models.order import Fill  # noqa: TCH001
from src.models.signal import Signal, SignalType
from src.strategies.base import BaseStrategy


class DummyStrategy(BaseStrategy):
    """Concrete strategy for testing."""

    REQUIRED_PARAMS: ClassVar[list[str]] = []

    def generate_signals(
        self,
        market_state: MarketState,
        orderbook: OrderBookSnapshot,
        context: dict[str, Any],
    ) -> list[Signal]:
        return [
            Signal(
                strategy_id=self.strategy_id,
                market_id=market_state.market_id,
                signal_type=SignalType.SKIP,
                direction=Side.YES,
            )
        ]

    def on_fill(self, fill: Fill, position: Position) -> None:
        self.add_position(position)

    def on_cancel(self, order_id: str, reason: str) -> None:
        pass


class StrictStrategy(BaseStrategy):
    """Strategy that requires specific config keys."""

    REQUIRED_PARAMS: ClassVar[list[str]] = [
        "strategy.false_sentiment.entry_threshold_base",
        "risk.max_position_pct",
    ]

    def generate_signals(
        self,
        market_state: MarketState,
        orderbook: OrderBookSnapshot,
        context: dict[str, Any],
    ) -> list[Signal]:
        return []

    def on_fill(self, fill: Fill, position: Position) -> None:
        pass

    def on_cancel(self, order_id: str, reason: str) -> None:
        pass


class TestBaseStrategy:
    def test_create_strategy(self, config_loader: ConfigLoader) -> None:
        strat = DummyStrategy(config=config_loader)
        assert strat.strategy_id == "DummyStrategy"

    def test_custom_strategy_id(self, config_loader: ConfigLoader) -> None:
        strat = DummyStrategy(config=config_loader, strategy_id="custom_id")
        assert strat.strategy_id == "custom_id"

    def test_generate_signals(
        self,
        config_loader: ConfigLoader,
        sample_market_state: MarketState,
        sample_orderbook: OrderBookSnapshot,
    ) -> None:
        strat = DummyStrategy(config=config_loader)
        signals = strat.generate_signals(sample_market_state, sample_orderbook, {})
        assert len(signals) == 1
        assert signals[0].signal_type == SignalType.SKIP

    def test_positions_tracking(
        self,
        config_loader: ConfigLoader,
        sample_position: Position,
        sample_fill: Fill,
    ) -> None:
        strat = DummyStrategy(config=config_loader)
        assert len(strat.positions) == 0
        strat.on_fill(sample_fill, sample_position)
        assert len(strat.positions) == 1
        assert len(strat.open_positions) == 1

    def test_required_params_present(self, config_loader: ConfigLoader) -> None:
        # Should not raise — required keys exist
        StrictStrategy(config=config_loader)

    def test_required_params_missing(self, config_loader: ConfigLoader) -> None:
        class BadStrategy(BaseStrategy):
            REQUIRED_PARAMS: ClassVar[list[str]] = ["completely.missing.key"]

            def generate_signals(self, ms: Any, ob: Any, ctx: Any) -> list[Signal]:
                return []

            def on_fill(self, fill: Fill, position: Position) -> None:
                pass

            def on_cancel(self, order_id: str, reason: str) -> None:
                pass

        with pytest.raises(ConfigError, match="Missing required config keys"):
            BadStrategy(config=config_loader)

    def test_get_config(self, config_loader: ConfigLoader) -> None:
        strat = DummyStrategy(config=config_loader)
        assert strat.get_config("risk.max_position_pct") == 0.02
        assert strat.get_config("nonexistent", "default") == "default"
