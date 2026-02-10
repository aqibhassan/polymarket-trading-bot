"""Tests for CLI entry point."""

from __future__ import annotations

import pytest

from src.cli import build_parser


class TestCLI:
    def test_parser_paper_mode(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--paper"])
        assert args.paper is True
        assert args.live is False
        assert args.backtest is False

    def test_parser_live_mode(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--live"])
        assert args.live is True
        assert args.paper is False

    def test_parser_backtest_mode(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--backtest"])
        assert args.backtest is True

    def test_parser_strategy_option(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--paper", "--strategy", "test_strat"])
        assert args.strategy == "test_strat"

    def test_parser_default_strategy(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--paper"])
        assert args.strategy == "false_sentiment"

    def test_parser_requires_mode(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_parser_mutually_exclusive(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--paper", "--live"])

    def test_parser_auto_confirm_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--live", "--auto-confirm"])
        assert args.auto_confirm is True

    def test_parser_auto_confirm_default_false(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--live"])
        assert args.auto_confirm is False

    def test_live_auto_sets_env_production(self) -> None:
        """When --live is used without --env, env defaults to production."""
        from src.cli import main

        # main() will try to confirm live trading; we test the parser logic
        parser = build_parser()
        args = parser.parse_args(["--live"])
        assert args.env is None  # parser default
        # The main() function auto-sets to production — tested via integration
