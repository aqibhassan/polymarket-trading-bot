"""Tests for strategy registry."""

from __future__ import annotations

from typing import Any

import pytest

from src.config.loader import ConfigLoader  # noqa: TCH001
from src.models.market import MarketState, OrderBookSnapshot, Position  # noqa: TCH001
from src.models.order import Fill  # noqa: TCH001
from src.models.signal import Signal  # noqa: TCH001
from src.strategies import registry
from src.strategies.base import BaseStrategy


class DummyStrategy(BaseStrategy):
    """Dummy strategy for testing registry."""

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


class TestStrategyRegistry:
    def setup_method(self) -> None:
        """Clear registry before each test."""
        registry._REGISTRY.clear()

    def test_register_and_get(self) -> None:
        registry.register("dummy")(DummyStrategy)
        cls = registry.get("dummy")
        assert cls is DummyStrategy

    def test_get_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown strategy"):
            registry.get("nonexistent")

    def test_list_strategies_empty(self) -> None:
        assert registry.list_strategies() == []

    def test_list_strategies_populated(self) -> None:
        registry.register("alpha")(DummyStrategy)
        registry.register("beta")(DummyStrategy)
        assert registry.list_strategies() == ["alpha", "beta"]

    def test_create_strategy(self, config_loader: ConfigLoader) -> None:
        registry.register("dummy")(DummyStrategy)
        instance = registry.create("dummy", config_loader)
        assert isinstance(instance, DummyStrategy)
        assert instance.strategy_id == "dummy"
