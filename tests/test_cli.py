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
