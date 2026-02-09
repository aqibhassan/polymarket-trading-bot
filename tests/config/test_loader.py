"""Tests for config loader."""

from __future__ import annotations

from pathlib import Path  # noqa: TCH003

import pytest

from src.config.loader import ConfigError, ConfigLoader


class TestConfigLoader:
    def test_load_default(self, config_dir: Path) -> None:
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()
        assert config["strategy"]["false_sentiment"]["entry_threshold_base"] == 0.59
        assert config["risk"]["max_position_pct"] == 0.02

    def test_get_dotted_key(self, config_loader: ConfigLoader) -> None:
        assert config_loader.get("risk.max_position_pct") == 0.02
        assert config_loader.get("strategy.false_sentiment.lookback_candles") == 5

    def test_get_missing_key_returns_default(self, config_loader: ConfigLoader) -> None:
        assert config_loader.get("nonexistent.key") is None
        assert config_loader.get("nonexistent.key", 42) == 42

    def test_require_existing_key(self, config_loader: ConfigLoader) -> None:
        val = config_loader.require("risk.max_position_pct")
        assert val == 0.02

    def test_require_missing_key_raises(self, config_loader: ConfigLoader) -> None:
        with pytest.raises(ConfigError, match="Required config key missing"):
            config_loader.require("nonexistent.deep.key")

    def test_validate_keys_all_present(self, config_loader: ConfigLoader) -> None:
        config_loader.validate_keys(["risk.max_position_pct", "exit.profit_target_pct"])

    def test_validate_keys_missing(self, config_loader: ConfigLoader) -> None:
        with pytest.raises(ConfigError, match="Missing required config keys"):
            config_loader.validate_keys(["risk.max_position_pct", "missing.key"])

    def test_env_override(self, config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MVHE__risk__max_position_pct", "0.05")
        loader = ConfigLoader(config_dir=config_dir)
        loader.load()
        assert loader.get("risk.max_position_pct") == 0.05

    def test_env_merge(self, config_dir: Path) -> None:
        env_toml = config_dir / "staging.toml"
        env_toml.write_text('[risk]\nmax_position_pct = 0.01\n')
        loader = ConfigLoader(config_dir=config_dir, env="staging")
        loader.load()
        assert loader.get("risk.max_position_pct") == 0.01
        # Other keys still present from default
        assert loader.get("exit.profit_target_pct") == 0.05

    def test_missing_default_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="Default config not found"):
            loader = ConfigLoader(config_dir=tmp_path / "nonexistent")
            loader.load()

    def test_load_strategy(self, config_dir: Path) -> None:
        strat_dir = config_dir / "strategies"
        strat_dir.mkdir()
        strat_toml = strat_dir / "test_strat.toml"
        strat_toml.write_text(
            '[strategy.false_sentiment]\nentry_threshold_base = 0.65\n'
        )
        loader = ConfigLoader(config_dir=config_dir)
        loader.load()
        loader.load_strategy("test_strat")
        assert loader.get("strategy.false_sentiment.entry_threshold_base") == 0.65
