"""Half-Kelly position sizing with safety caps.

Supports both general trading and binary prediction market sizing.
For binary markets, uses the correct asymmetric Kelly for binary payoffs:
    f* = p/P - (1-p)/(1-P) = (p - P) / (P * (1 - P))
This accounts for the asymmetric payoff structure where wagering $X at
price P yields $X/P shares, winning $(1-P)/P per dollar risked.
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel

from src.core.logging import get_logger

log = get_logger(__name__)


class PositionSize(BaseModel):
    """Position sizing result."""

    recommended_size: Decimal
    max_allowed: Decimal
    kelly_fraction: Decimal
    capped_reason: str | None = None

    model_config = {"frozen": True}


class PositionSizer:
    """Half-Kelly position sizing with safety caps.

    Calculates optimal position size using the Kelly criterion at half
    leverage, then caps at risk limits.

    For binary prediction markets, use ``calculate_binary`` which applies
    the asymmetric Kelly formula: f* = (p - P) / (P * (1 - P)).
    """

    def __init__(
        self,
        max_position_pct: Decimal = Decimal("0.02"),
        max_order_book_pct: Decimal = Decimal("0.10"),
        kelly_multiplier: Decimal = Decimal("0.5"),
    ) -> None:
        self._max_position_pct = max_position_pct
        self._max_order_book_pct = max_order_book_pct
        self._kelly_multiplier = kelly_multiplier

    def calculate(
        self,
        balance: Decimal,
        signal_confidence: float,
        win_rate: Decimal,
        avg_win: Decimal,
        avg_loss: Decimal,
        book_depth: Decimal = Decimal("0"),
    ) -> PositionSize:
        """Calculate position size using half-Kelly with caps.

        Args:
            balance: Current account balance.
            signal_confidence: Signal confidence score (0-1).
            win_rate: Historical win rate as a decimal (e.g. 0.55).
            avg_win: Average winning trade size.
            avg_loss: Average losing trade size (positive value).
            book_depth: Total order book depth for impact cap.

        Returns:
            PositionSize with recommended size and cap info.
        """
        if balance <= 0 or win_rate <= 0 or avg_loss <= 0:
            log.info("position_size_zero", reason="invalid inputs")
            return PositionSize(
                recommended_size=Decimal("0"),
                max_allowed=Decimal("0"),
                kelly_fraction=Decimal("0"),
                capped_reason="invalid inputs (zero balance, win_rate, or avg_loss)",
            )

        # Kelly criterion: f* = p - (q / b)
        # p = win_rate, q = 1 - win_rate, b = avg_win / avg_loss
        win_loss_ratio = avg_win / avg_loss
        loss_rate = Decimal("1") - win_rate
        kelly = win_rate - (loss_rate / win_loss_ratio)

        # Fractional Kelly for safety
        half_kelly = kelly * self._kelly_multiplier

        # Cannot bet negative
        if half_kelly <= 0:
            log.info("position_size_zero", reason="negative kelly", kelly=str(kelly))
            return PositionSize(
                recommended_size=Decimal("0"),
                max_allowed=Decimal("0"),
                kelly_fraction=kelly,
                capped_reason="negative kelly — edge insufficient",
            )

        # Scale by signal confidence
        adjusted = half_kelly * Decimal(str(signal_confidence))
        raw_size = adjusted * balance

        # Apply caps
        max_by_balance = balance * self._max_position_pct
        max_by_book = book_depth * self._max_order_book_pct if book_depth > 0 else max_by_balance
        max_allowed = min(max_by_balance, max_by_book)

        capped_reason: str | None = None
        recommended = raw_size
        if raw_size > max_by_balance:
            recommended = max_by_balance
            capped_reason = f"capped at max_position_pct ({self._max_position_pct * 100}%)"
        if book_depth > 0 and raw_size > max_by_book:
            recommended = min(recommended, max_by_book)
            capped_reason = f"capped at max_order_book_pct ({self._max_order_book_pct * 100}%)"

        log.info(
            "position_sized",
            kelly=str(kelly),
            half_kelly=str(half_kelly),
            raw_size=str(raw_size),
            recommended=str(recommended),
            capped_reason=capped_reason or "none",
        )

        return PositionSize(
            recommended_size=recommended,
            max_allowed=max_allowed,
            kelly_fraction=kelly,
            capped_reason=capped_reason,
        )

    def calculate_binary(
        self,
        balance: Decimal,
        entry_price: Decimal,
        estimated_win_prob: Decimal,
        book_depth: Decimal = Decimal("0"),
    ) -> PositionSize:
        """Calculate position size for a binary prediction market trade.

        Uses the correct Kelly criterion for asymmetric binary payoffs:
            f* = p/P - (1-p)/(1-P) = (p - P) / (P * (1 - P))
        where p = estimated win probability, P = entry price.

        This accounts for binary market asymmetry: wagering $X at price P
        yields $X/P shares, winning $(1-P)/P per dollar risked.

        Then applies ``kelly_multiplier`` (default 0.5 for half-Kelly)
        and caps at risk limits.

        Args:
            balance: Current account balance (USDC).
            entry_price: Contract entry price (0 < P < 1).
            estimated_win_prob: Estimated probability of winning (0 < p < 1).
            book_depth: Total order book depth for impact cap.

        Returns:
            PositionSize with recommended size and cap info.
        """
        if (
            balance <= 0
            or entry_price <= 0
            or entry_price >= 1
            or estimated_win_prob <= 0
            or estimated_win_prob >= 1
        ):
            log.info("binary_position_zero", reason="invalid inputs")
            return PositionSize(
                recommended_size=Decimal("0"),
                max_allowed=Decimal("0"),
                kelly_fraction=Decimal("0"),
                capped_reason="invalid inputs",
            )

        # Binary Kelly for asymmetric payoffs:
        # f* = p/P - (1-p)/(1-P) = (p - P) / (P * (1 - P))
        kelly = (estimated_win_prob - entry_price) / (entry_price * (Decimal("1") - entry_price))

        # Fractional Kelly for safety
        frac_kelly = kelly * self._kelly_multiplier

        if frac_kelly <= 0:
            log.info(
                "binary_position_zero",
                reason="no edge",
                kelly=str(kelly),
                win_prob=str(estimated_win_prob),
                entry_price=str(entry_price),
            )
            return PositionSize(
                recommended_size=Decimal("0"),
                max_allowed=Decimal("0"),
                kelly_fraction=kelly,
                capped_reason="no edge — estimated win prob <= entry price",
            )

        raw_size = frac_kelly * balance

        # Apply caps
        max_by_balance = balance * self._max_position_pct
        max_by_book = (
            book_depth * self._max_order_book_pct
            if book_depth > 0
            else max_by_balance
        )
        max_allowed = min(max_by_balance, max_by_book)

        capped_reason: str | None = None
        recommended = raw_size
        if raw_size > max_by_balance:
            recommended = max_by_balance
            capped_reason = (
                f"capped at max_position_pct ({self._max_position_pct * 100}%)"
            )
        if book_depth > 0 and raw_size > max_by_book:
            recommended = min(recommended, max_by_book)
            capped_reason = (
                f"capped at max_order_book_pct ({self._max_order_book_pct * 100}%)"
            )

        log.info(
            "binary_position_sized",
            kelly=str(kelly),
            frac_kelly=str(frac_kelly),
            entry_price=str(entry_price),
            win_prob=str(estimated_win_prob),
            raw_size=str(raw_size),
            recommended=str(recommended),
            capped_reason=capped_reason or "none",
        )

        return PositionSize(
            recommended_size=recommended,
            max_allowed=max_allowed,
            kelly_fraction=kelly,
            capped_reason=capped_reason,
        )
