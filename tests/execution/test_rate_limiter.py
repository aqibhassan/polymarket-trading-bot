"""Tests for TokenBucketRateLimiter — bucket, margin, exhaustion, refill."""

from __future__ import annotations

import asyncio
import time

import pytest

from src.execution.rate_limiter import TokenBucketRateLimiter


def test_safety_margin() -> None:
    rl = TokenBucketRateLimiter(max_tokens=100, refill_rate=10.0)
    # Effective max should be 80 (100 * 0.8)
    assert rl.tokens_remaining == 80


@pytest.mark.asyncio
async def test_acquire_success() -> None:
    rl = TokenBucketRateLimiter(max_tokens=10, refill_rate=1.0)
    assert await rl.acquire() is True
    assert rl.tokens_remaining == 7  # 8 - 1


@pytest.mark.asyncio
async def test_acquire_exhaustion() -> None:
    rl = TokenBucketRateLimiter(max_tokens=5, refill_rate=0.0)
    effective = 4  # 5 * 0.8 = 4
    for _ in range(effective):
        assert await rl.acquire() is True
    assert await rl.acquire() is False


@pytest.mark.asyncio
async def test_refill_over_time() -> None:
    rl = TokenBucketRateLimiter(max_tokens=10, refill_rate=100.0)
    # Effective refill_rate = 80/s
    # Drain all tokens
    while await rl.acquire():
        pass
    assert rl.tokens_remaining == 0
    await asyncio.sleep(0.05)  # Should refill ~4 tokens (80 * 0.05)
    remaining = rl.tokens_remaining
    assert remaining >= 2  # At least some tokens refilled


def test_refill_caps_at_max() -> None:
    rl = TokenBucketRateLimiter(max_tokens=10, refill_rate=100.0)
    time.sleep(0.1)
    assert rl.tokens_remaining <= 8  # Capped at effective max (10 * 0.8)


@pytest.mark.asyncio
async def test_wait_acquires_token() -> None:
    rl = TokenBucketRateLimiter(max_tokens=2, refill_rate=100.0)
    # Drain tokens
    while await rl.acquire():
        pass
    # wait() should block briefly then succeed
    await asyncio.wait_for(rl.wait(), timeout=1.0)
    # After wait, we consumed a token


@pytest.mark.asyncio
async def test_multiple_instances_independent() -> None:
    rl1 = TokenBucketRateLimiter(max_tokens=10, refill_rate=1.0)
    rl2 = TokenBucketRateLimiter(max_tokens=20, refill_rate=1.0)
    await rl1.acquire()
    assert rl1.tokens_remaining == 7  # 8 - 1
    assert rl2.tokens_remaining == 16  # 20 * 0.8


@pytest.mark.asyncio
async def test_concurrent_acquire_respects_limit() -> None:
    """Verify that concurrent acquire() calls don't exceed bucket capacity."""
    rl = TokenBucketRateLimiter(max_tokens=5, refill_rate=0.0)
    effective = 4  # 5 * 0.8 = 4

    results = await asyncio.gather(
        *(rl.acquire() for _ in range(10))
    )
    assert sum(results) == effective  # Exactly 4 should succeed
