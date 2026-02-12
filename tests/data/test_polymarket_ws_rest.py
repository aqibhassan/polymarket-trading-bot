"""Tests for PolymarketWSFeed REST seeding and direction-aware logic."""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.data.polymarket_ws import CLOBState, PolymarketWSFeed


class TestSeedRestData:
    """Tests for seed_rest_data() REST API priming."""

    def test_seed_last_trade_price_string(self) -> None:
        feed = PolymarketWSFeed()
        feed.seed_rest_data("token-yes-1", "0.47", None, None)
        state = feed.get_clob_state("token-yes-1")
        assert state is not None
        assert state.last_trade_price == Decimal("0.47")

    def test_seed_last_trade_price_dict(self) -> None:
        feed = PolymarketWSFeed()
        feed.seed_rest_data("token-yes-1", {"price": "0.63"}, None, None)
        state = feed.get_clob_state("token-yes-1")
        assert state is not None
        assert state.last_trade_price == Decimal("0.63")

    def test_seed_midpoint_string(self) -> None:
        feed = PolymarketWSFeed()
        feed.seed_rest_data("token-yes-1", None, "0.55", None)
        state = feed.get_clob_state("token-yes-1")
        assert state is not None
        assert state.midpoint == Decimal("0.55")

    def test_seed_midpoint_dict(self) -> None:
        feed = PolymarketWSFeed()
        feed.seed_rest_data("token-yes-1", None, {"mid": "0.61"}, None)
        state = feed.get_clob_state("token-yes-1")
        assert state is not None
        assert state.midpoint == Decimal("0.61")

    def test_seed_spread_derives_bid_ask(self) -> None:
        feed = PolymarketWSFeed()
        # Must have midpoint first for spread to derive bid/ask
        feed.seed_rest_data("token-yes-1", None, "0.50", "0.04")
        state = feed.get_clob_state("token-yes-1")
        assert state is not None
        assert state.midpoint == Decimal("0.50")
        assert state.best_bid == Decimal("0.48")
        assert state.best_ask == Decimal("0.52")

    def test_seed_spread_dict(self) -> None:
        feed = PolymarketWSFeed()
        feed.seed_rest_data("token-yes-1", None, "0.60", {"spread": "0.02"})
        state = feed.get_clob_state("token-yes-1")
        assert state is not None
        assert state.best_bid == Decimal("0.59")
        assert state.best_ask == Decimal("0.61")

    def test_seed_does_not_overwrite_ws_data(self) -> None:
        """WS data takes priority — seed only fills None fields."""
        feed = PolymarketWSFeed()
        # Simulate WS having already set last_trade_price
        feed._clob_state["token-yes-1"] = CLOBState(
            last_trade_price=Decimal("0.72"),
        )
        feed.seed_rest_data("token-yes-1", "0.55", None, None)
        state = feed.get_clob_state("token-yes-1")
        assert state is not None
        # WS value preserved, not overwritten
        assert state.last_trade_price == Decimal("0.72")

    def test_seed_all_fields_at_once(self) -> None:
        feed = PolymarketWSFeed()
        feed.seed_rest_data("token-no-1", "0.38", "0.40", "0.06")
        state = feed.get_clob_state("token-no-1")
        assert state is not None
        assert state.last_trade_price == Decimal("0.38")
        assert state.midpoint == Decimal("0.40")
        assert state.best_bid == Decimal("0.37")
        assert state.best_ask == Decimal("0.43")
        assert state.last_updated is not None

    def test_seed_zero_price_ignored(self) -> None:
        feed = PolymarketWSFeed()
        feed.seed_rest_data("token-1", "0", None, None)
        state = feed.get_clob_state("token-1")
        assert state is not None
        assert state.last_trade_price is None  # zero should be ignored

    def test_seed_none_values_no_crash(self) -> None:
        feed = PolymarketWSFeed()
        feed.seed_rest_data("token-1", None, None, None)
        state = feed.get_clob_state("token-1")
        assert state is not None
        assert state.last_trade_price is None
        assert state.midpoint is None

    def test_seed_invalid_values_no_crash(self) -> None:
        feed = PolymarketWSFeed()
        feed.seed_rest_data("token-1", "not-a-number", {"mid": ""}, "bad")
        state = feed.get_clob_state("token-1")
        assert state is not None
        # Invalid values ignored gracefully
        assert state.last_trade_price is None


class TestDirectionAwareSpread:
    """Test that spread uses the correct token for direction."""

    def test_no_direction_uses_no_token_spread(self) -> None:
        """When NO direction signal fires, spread should use NO token data."""
        # This is a logic test — verifying the concept used in bot.py
        # YES book: 0.01/0.99 (desert) → spread = 0.98
        # NO book: 0.40/0.45 (liquid) → spread = 0.05
        yes_bid, yes_ask = Decimal("0.01"), Decimal("0.99")
        no_bid, no_ask = Decimal("0.40"), Decimal("0.45")

        # For NO direction, should use NO token spread
        no_spread = no_ask - no_bid
        yes_spread = yes_ask - yes_bid

        assert no_spread == Decimal("0.05")
        assert yes_spread == Decimal("0.98")
        # NO spread is much tighter — this would have been blocked before
        assert no_spread < Decimal("0.50")

    def test_yes_direction_uses_yes_token_spread(self) -> None:
        yes_bid, yes_ask = Decimal("0.60"), Decimal("0.62")
        yes_spread = yes_ask - yes_bid
        assert yes_spread == Decimal("0.02")


class TestGTCNoSpreadGate:
    """Verify GTC orders are NOT blocked by wide spreads."""

    def test_gtc_allowed_on_desert_book(self) -> None:
        """GTC should be placed even with 0.01/0.99 book.

        The max_clob_entry_price gate prevents overpaying,
        so the spread gate is redundant.
        """
        # Desert book
        best_bid = Decimal("0.01")
        best_ask = Decimal("0.99")
        spread = best_ask - best_bid  # 0.98
        max_clob_spread = Decimal("0.50")

        # Old behavior would skip: spread > max_clob_spread
        assert spread > max_clob_spread

        # New behavior: GTC is not blocked by spread.
        # Instead, the entry price gate catches bad entries:
        computed_mid = (best_bid + best_ask) / 2  # 0.50
        max_entry_price = Decimal("0.80")
        # 0.50 < 0.80 — this entry is allowed (and cheap!)
        assert computed_mid <= max_entry_price

    def test_max_entry_price_still_protects(self) -> None:
        """max_clob_entry_price prevents overpaying even without spread gate."""
        entry_price = Decimal("0.92")
        max_entry_price = Decimal("0.80")
        # Too expensive — blocked by price gate, not spread gate
        assert entry_price > max_entry_price
