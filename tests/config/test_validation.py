"""Tests for config validation."""

from __future__ import annotations

from pathlib import Path  # noqa: TCH003

import pytest

from src.config.loader import ConfigError, ConfigLoader


class TestConfigValidation:
    def test_valid_config_passes(self, config_dir: Path) -> None:
        loader = ConfigLoader(config_dir=config_dir)
        loader.load()
        loader.validate_ranges()  # Should not raise

    def test_invalid_max_position_pct_zero(self, config_dir: Path) -> None:
        toml = config_dir / "default.toml"
        content = toml.read_text().replace("max_position_pct = 0.02", "max_position_pct = 0.0")
        toml.write_text(content)
        loader = ConfigLoader(config_dir=config_dir)
        loader.load()
        with pytest.raises(ConfigError, match="max_position_pct"):
            loader.validate_ranges()

    def test_invalid_max_position_pct_over_one(self, config_dir: Path) -> None:
        toml = config_dir / "default.toml"
        content = toml.read_text().replace("max_position_pct = 0.02", "max_position_pct = 1.5")
        toml.write_text(content)
        loader = ConfigLoader(config_dir=config_dir)
        loader.load()
        with pytest.raises(ConfigError, match="max_position_pct"):
            loader.validate_ranges()

    def test_invalid_hard_stop_loss_negative(self, config_dir: Path) -> None:
        toml = config_dir / "default.toml"
        content = toml.read_text().replace(
            "hard_stop_loss_pct = 0.04", "hard_stop_loss_pct = -0.01",
        )
        toml.write_text(content)
        loader = ConfigLoader(config_dir=config_dir)
        loader.load()
        with pytest.raises(ConfigError, match="hard_stop_loss_pct"):
            loader.validate_ranges()

    def test_invalid_profit_target_zero(self, config_dir: Path) -> None:
        toml = config_dir / "default.toml"
        content = toml.read_text().replace("profit_target_pct = 0.05", "profit_target_pct = 0")
        toml.write_text(content)
        loader = ConfigLoader(config_dir=config_dir)
        loader.load()
        with pytest.raises(ConfigError, match="profit_target_pct"):
            loader.validate_ranges()

    def test_valid_edge_max_position_pct_one(self, config_dir: Path) -> None:
        toml = config_dir / "default.toml"
        content = toml.read_text().replace("max_position_pct = 0.02", "max_position_pct = 1.0")
        toml.write_text(content)
        loader = ConfigLoader(config_dir=config_dir)
        loader.load()
        loader.validate_ranges()  # 1.0 is valid (inclusive upper bound)

    def test_multiple_validation_errors(self, config_dir: Path) -> None:
        toml = config_dir / "default.toml"
        content = toml.read_text()
        content = content.replace("max_position_pct = 0.02", "max_position_pct = 0.0")
        content = content.replace("hard_stop_loss_pct = 0.04", "hard_stop_loss_pct = -1")
        toml.write_text(content)
        loader = ConfigLoader(config_dir=config_dir)
        loader.load()
        with pytest.raises(ConfigError, match="Config validation failed"):
            loader.validate_ranges()
