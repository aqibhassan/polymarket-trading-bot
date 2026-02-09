"""Backtest report generation — JSON and console output."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from src.core.logging import get_logger

log = get_logger(__name__)


class _DecimalEncoder(json.JSONEncoder):
    """JSON encoder that serialises Decimal as string."""

    def default(self, o: Any) -> Any:
        if isinstance(o, Decimal):
            return str(o)
        return super().default(o)


class BacktestReporter:
    """Generate backtest reports in JSON and console formats."""

    def generate_json(self, result: dict[str, Any]) -> str:
        """Serialise a backtest result to a JSON string.

        Args:
            result: Dict with keys 'metrics', 'trades', 'equity_curve',
                    and optionally 'validation'.

        Returns:
            JSON string.
        """
        return json.dumps(result, cls=_DecimalEncoder, indent=2)

    def print_summary(self, result: dict[str, Any]) -> str:
        """Format a human-readable summary for the console.

        Args:
            result: Dict with 'metrics' sub-dict.

        Returns:
            Formatted summary string (also printed to stdout).
        """
        metrics = result.get("metrics", {})
        lines = [
            "",
            "=" * 50,
            "  BACKTEST SUMMARY",
            "=" * 50,
        ]

        fmt_map = [
            ("Total Return", "total_return_pct", True),
            ("Sharpe Ratio", "sharpe_ratio", False),
            ("Sortino Ratio", "sortino_ratio", False),
            ("Max Drawdown", "max_drawdown_pct", True),
            ("Calmar Ratio", "calmar_ratio", False),
            ("Win Rate", "win_rate", True),
            ("Profit Factor", "profit_factor", False),
            ("Avg Win/Loss", "avg_win_loss_ratio", False),
            ("Total Trades", "total_trades", False),
        ]

        for label, key, as_pct in fmt_map:
            value = metrics.get(key)
            if value is None:
                continue
            if as_pct:
                lines.append(f"  {label:<20} {float(value) * 100:>10.2f}%")
            else:
                lines.append(f"  {label:<20} {float(value):>10.4f}")

        validation = result.get("validation")
        if validation:
            lines.append("")
            lines.append("  --- Validation ---")
            ci = validation.get("sharpe_ci_95")
            if ci:
                lines.append(f"  Sharpe 95% CI       [{ci[0]:.4f}, {ci[1]:.4f}]")
            p_val = validation.get("p_value")
            if p_val is not None:
                lines.append(f"  P-value            {p_val:>10.4f}")
            sig = validation.get("is_significant")
            if sig is not None:
                lines.append(f"  Significant         {'Yes' if sig else 'No'}")
            warn = validation.get("overfitting_warning")
            if warn:
                lines.append(f"  WARNING: {warn}")

        lines.append("=" * 50)
        summary = "\n".join(lines)
        log.info("backtest_summary", summary=summary)
        return summary

    def save_report(self, result: dict[str, Any], path: str | Path) -> None:
        """Write the full JSON report to a file.

        Args:
            result: Backtest result dict.
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        json_str = self.generate_json(result)
        path.write_text(json_str)
        log.info("report_saved", path=str(path))
