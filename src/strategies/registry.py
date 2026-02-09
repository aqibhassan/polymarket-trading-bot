"""Strategy registry — discover and load strategies by name."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.config.loader import ConfigLoader
    from src.strategies.base import BaseStrategy

_REGISTRY: dict[str, type[BaseStrategy]] = {}


def register(name: str) -> Any:
    """Decorator to register a strategy class.

    Usage:
        @register("false_sentiment")
        class FalseSentimentStrategy(BaseStrategy):
            ...
    """

    def decorator(cls: type[BaseStrategy]) -> type[BaseStrategy]:
        _REGISTRY[name] = cls
        return cls

    return decorator


def get(name: str) -> type[BaseStrategy]:
    """Get a strategy class by registered name.

    Raises:
        KeyError: If strategy name is not registered.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys())) or "(none)"
        msg = f"Unknown strategy '{name}'. Available: {available}"
        raise KeyError(msg)
    return _REGISTRY[name]


def create(name: str, config: ConfigLoader) -> BaseStrategy:
    """Create a strategy instance by name.

    Args:
        name: Registered strategy name.
        config: ConfigLoader instance.

    Returns:
        Instantiated strategy.
    """
    cls = get(name)
    return cls(config=config, strategy_id=name)


def list_strategies() -> list[str]:
    """List all registered strategy names."""
    return sorted(_REGISTRY.keys())
