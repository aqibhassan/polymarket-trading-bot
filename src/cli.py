"""MVHE CLI entry point."""

from __future__ import annotations

import argparse
import sys


def _confirm_live_trading() -> bool:
    """Require explicit confirmation for live trading. SECURITY: mandatory."""
    print("=" * 60)
    print("  WARNING: You are about to start LIVE trading.")
    print("  Real money will be at risk.")
    print("=" * 60)
    response = input('Type "yes" to confirm live trading: ')
    return response.strip().lower() == "yes"


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="mvhe",
        description="Micro-Volatility Harvesting Engine — Polymarket BTC 15m trading",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--paper", action="store_true", help="Run in paper trading mode")
    mode.add_argument("--live", action="store_true", help="Run in live trading mode")
    mode.add_argument("--backtest", action="store_true", help="Run backtesting")

    parser.add_argument(
        "--strategy",
        type=str,
        default="false_sentiment",
        help=(
            "Strategy name (default: false_sentiment). "
            "Use 'momentum_confirmation' (or 'reversal_catcher') for 1m swing trading mode."
        ),
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="Config directory path (default: config)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Environment name (default: from MVHE_ENV)",
    )
    parser.add_argument(
        "--fixed-bet",
        type=float,
        default=0.0,
        help="Fixed bet size in USD (e.g. 200). Overrides Kelly sizing. 0 = use Kelly (default).",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.live and not _confirm_live_trading():
        print("Live trading cancelled.")
        return 1

    if args.paper:
        bet_msg = f"${args.fixed_bet:.0f}/trade" if args.fixed_bet > 0 else "Kelly sizing"
        print(f"Starting paper trading with strategy: {args.strategy} ({bet_msg})")
        from src.bot import run_bot

        return run_bot(
            mode="paper", strategy=args.strategy,
            config_dir=args.config_dir, env=args.env,
            fixed_bet_size=args.fixed_bet,
        )

    if args.live:
        print(f"Starting LIVE trading with strategy: {args.strategy}")
        from src.bot import run_bot

        return run_bot(
            mode="live", strategy=args.strategy,
            config_dir=args.config_dir, env=args.env,
            fixed_bet_size=args.fixed_bet,
        )

    if args.backtest:
        print(f"Running backtest with strategy: {args.strategy}")
        from src.bot import run_backtest

        return run_backtest(strategy=args.strategy, config_dir=args.config_dir, env=args.env)

    return 0


if __name__ == "__main__":
    sys.exit(main())
