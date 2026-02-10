"""TOML config loader with environment variable overrides."""

from __future__ import annotations

import os

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]
from pathlib import Path
from typing import Any


class ConfigError(Exception):
    """Raised when config loading or validation fails."""


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base, returning a new dict."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _apply_env_overrides(config: dict[str, Any], prefix: str = "MVHE") -> dict[str, Any]:
    """Apply environment variable overrides.

    Env var naming: MVHE__section__key=value (double underscore separator).
    Nested keys: MVHE__risk__max_position_pct=0.01
    """
    result = dict(config)
    env_prefix = f"{prefix}__"

    for env_key, env_value in os.environ.items():
        if not env_key.startswith(env_prefix):
            continue

        parts = env_key[len(env_prefix) :].lower().split("__")
        target = result
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            if isinstance(target[part], dict):
                target = target[part]
            else:
                break
        else:
            final_key = parts[-1]
            target[final_key] = _coerce_value(env_value)

    return result


def _coerce_value(value: str) -> Any:
    """Coerce string env var value to appropriate Python type.

    Numeric conversion is attempted BEFORE boolean to avoid "0"/"1"
    being interpreted as False/True when they should be integers.
    """
    # Try numeric first — "0" and "1" should stay numeric
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    # Boolean keywords (excluding "0" and "1" which are already handled)
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    return value


class ConfigLoader:
    """Load and merge TOML config files with env var overrides."""

    def __init__(
        self,
        config_dir: str | Path = "config",
        env: str | None = None,
    ) -> None:
        self._config_dir = Path(config_dir)
        self._env = env or os.environ.get("MVHE_ENV", "development")
        self._config: dict[str, Any] = {}

    def load(self) -> dict[str, Any]:
        """Load config: default.toml → {env}.toml → env vars."""
        default_path = self._config_dir / "default.toml"
        if not default_path.exists():
            msg = f"Default config not found: {default_path}"
            raise ConfigError(msg)

        self._config = self._load_toml(default_path)

        env_path = self._config_dir / f"{self._env}.toml"
        if env_path.exists():
            env_config = self._load_toml(env_path)
            self._config = _deep_merge(self._config, env_config)

        self._config = _apply_env_overrides(self._config)
        return self._config

    def load_strategy(self, strategy_name: str) -> dict[str, Any]:
        """Load strategy-specific config on top of base config."""
        if not self._config:
            self.load()

        strategy_path = self._config_dir / "strategies" / f"{strategy_name}.toml"
        if strategy_path.exists():
            strategy_config = self._load_toml(strategy_path)
            self._config = _deep_merge(self._config, strategy_config)

        return self._config

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Get a config value using dotted notation: 'risk.max_position_pct'."""
        if not self._config:
            self.load()

        parts = dotted_key.split(".")
        current: Any = self._config
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current

    def require(self, dotted_key: str) -> Any:
        """Get a config value, raising ConfigError if missing."""
        value = self.get(dotted_key)
        if value is None:
            msg = f"Required config key missing: {dotted_key}"
            raise ConfigError(msg)
        return value

    def validate_keys(self, required_keys: list[str]) -> None:
        """Validate that all required keys exist."""
        missing = [k for k in required_keys if self.get(k) is None]
        if missing:
            msg = f"Missing required config keys: {', '.join(missing)}"
            raise ConfigError(msg)

    def validate_ranges(self) -> None:
        """Validate config value ranges for safety-critical parameters.

        Raises:
            ConfigError: If any risk/exit parameter is out of valid range.
        """
        if not self._config:
            self.load()

        errors: list[str] = []

        # Risk parameters
        max_pos = self.get("risk.max_position_pct")
        if max_pos is not None and not (0 < max_pos <= 1):
            errors.append(f"risk.max_position_pct must be in (0, 1], got {max_pos}")

        max_dd = self.get("risk.max_daily_drawdown_pct")
        if max_dd is not None and not (0 < max_dd <= 1):
            errors.append(f"risk.max_daily_drawdown_pct must be in (0, 1], got {max_dd}")

        max_book = self.get("risk.max_order_book_pct")
        if max_book is not None and not (0 < max_book <= 1):
            errors.append(f"risk.max_order_book_pct must be in (0, 1], got {max_book}")

        # Exit parameters
        hard_stop = self.get("exit.hard_stop_loss_pct")
        if hard_stop is not None and hard_stop <= 0:
            errors.append(f"exit.hard_stop_loss_pct must be > 0, got {hard_stop}")

        profit_target = self.get("exit.profit_target_pct")
        if profit_target is not None and profit_target <= 0:
            errors.append(f"exit.profit_target_pct must be > 0, got {profit_target}")

        trailing_stop = self.get("exit.trailing_stop_pct")
        if trailing_stop is not None and trailing_stop <= 0:
            errors.append(f"exit.trailing_stop_pct must be > 0, got {trailing_stop}")

        max_hold = self.get("exit.max_hold_seconds")
        if max_hold is not None and max_hold <= 0:
            errors.append(f"exit.max_hold_seconds must be > 0, got {max_hold}")

        if errors:
            msg = "Config validation failed:\n  " + "\n  ".join(errors)
            raise ConfigError(msg)

    @property
    def config(self) -> dict[str, Any]:
        if not self._config:
            self.load()
        return self._config

    @staticmethod
    def _load_toml(path: Path) -> dict[str, Any]:
        with open(path, "rb") as f:
            return tomllib.load(f)
