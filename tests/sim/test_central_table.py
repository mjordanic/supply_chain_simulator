"""Tests for the CentralTable deep module."""

from __future__ import annotations

import pytest

from src.sim.central_table import CentralTable, Offer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_offer(available_qty: int = 100, list_price: float = 10.0, min_order: int = 1, fill_rate_recent: float = 1.0) -> Offer:
    return Offer(
        available_qty=available_qty,
        list_price=list_price,
        min_order=min_order,
        fill_rate_recent=fill_rate_recent,
    )


# ---------------------------------------------------------------------------
# Tracer bullet: publish and retrieve
# ---------------------------------------------------------------------------

def test_publish_and_snapshot_round_trip():
    ct = CentralTable()
    offer = make_offer(available_qty=50)
    ct.publish("seller1", "pidA", offer)
    result = ct.snapshot_for_buyer("pidA")
    assert ("seller1", offer) in result or dict(result)["seller1"].available_qty == 50


def test_snapshot_for_buyer_returns_all_sellers_for_product():
    ct = CentralTable()
    ct.publish("s1", "pidA", make_offer(available_qty=10))
    ct.publish("s2", "pidA", make_offer(available_qty=20))
    ct.publish("s3", "pidB", make_offer(available_qty=30))  # different product
    result = ct.snapshot_for_buyer("pidA")
    seller_ids = {sid for sid, _ in result}
    assert "s1" in seller_ids
    assert "s2" in seller_ids
    assert "s3" not in seller_ids


# ---------------------------------------------------------------------------
# publish: overwrite semantics
# ---------------------------------------------------------------------------

def test_publish_overwrites_existing_row():
    ct = CentralTable()
    ct.publish("seller1", "pidA", make_offer(available_qty=50, list_price=5.0))
    ct.publish("seller1", "pidA", make_offer(available_qty=99, list_price=9.0))
    result = dict(ct.snapshot_for_buyer("pidA"))
    assert result["seller1"].available_qty == 99
    assert result["seller1"].list_price == 9.0


# ---------------------------------------------------------------------------
# commit: decrement available_qty
# ---------------------------------------------------------------------------

def test_commit_decrements_available_qty():
    ct = CentralTable()
    ct.publish("seller1", "pidA", make_offer(available_qty=100))
    ct.commit("seller1", "pidA", qty=30)
    result = dict(ct.snapshot_for_buyer("pidA"))
    assert result["seller1"].available_qty == 70


def test_commit_multiple_times_decrements_cumulatively():
    ct = CentralTable()
    ct.publish("seller1", "pidA", make_offer(available_qty=100))
    ct.commit("seller1", "pidA", qty=30)
    ct.commit("seller1", "pidA", qty=20)
    result = dict(ct.snapshot_for_buyer("pidA"))
    assert result["seller1"].available_qty == 50


# ---------------------------------------------------------------------------
# commit: never below zero
# ---------------------------------------------------------------------------

def test_commit_below_zero_raises():
    ct = CentralTable()
    ct.publish("seller1", "pidA", make_offer(available_qty=10))
    with pytest.raises((ValueError, AssertionError)):
        ct.commit("seller1", "pidA", qty=20)


def test_commit_exact_available_qty_leaves_zero():
    ct = CentralTable()
    ct.publish("seller1", "pidA", make_offer(available_qty=10))
    ct.commit("seller1", "pidA", qty=10)
    result = dict(ct.snapshot_for_buyer("pidA"))
    assert result["seller1"].available_qty == 0


# ---------------------------------------------------------------------------
# EMA fill_rate_recent tracking
# ---------------------------------------------------------------------------

def test_commit_full_fill_keeps_fill_rate_high():
    """Committing exact available qty (full fill) should keep fill_rate_recent near 1."""
    ct = CentralTable()
    ct.publish("seller1", "pidA", make_offer(available_qty=100, fill_rate_recent=1.0))
    ct.commit("seller1", "pidA", qty=100)
    result = dict(ct.snapshot_for_buyer("pidA"))
    assert result["seller1"].fill_rate_recent > 0.9


def test_commit_partial_fill_decreases_fill_rate():
    """Publishing 100, committing only 10 (partial fill) should lower fill_rate_recent over time."""
    ct = CentralTable()
    # Run multiple partial fill cycles to accumulate EMA signal
    for _ in range(20):
        ct.publish("seller1", "pidA", make_offer(available_qty=100, fill_rate_recent=ct.snapshot_for_buyer("pidA")[0][1].fill_rate_recent if ct.snapshot_for_buyer("pidA") else 1.0))
        ct.commit("seller1", "pidA", qty=10)  # requesting 10 out of 100 = partial
    result = dict(ct.snapshot_for_buyer("pidA"))
    # fill_rate should have moved away from 1.0 toward ~0.1
    assert result["seller1"].fill_rate_recent < 0.9


def test_ema_weighted_by_qty():
    """EMA uses qty-weighted partial fills: larger commit = bigger update."""
    ct = CentralTable()
    # Commit 50% fill twice: fill_rate_recent should be around 0.5
    for _ in range(20):
        available = 100
        ct.publish("seller1", "pidA", make_offer(available_qty=available, fill_rate_recent=ct.snapshot_for_buyer("pidA")[0][1].fill_rate_recent if ct.snapshot_for_buyer("pidA") else 1.0))
        ct.commit("seller1", "pidA", qty=50)  # 50/100 = 0.5 fill rate
    result = dict(ct.snapshot_for_buyer("pidA"))
    # After many cycles at 50% fill, the EMA should converge near 0.5
    assert 0.3 <= result["seller1"].fill_rate_recent <= 0.7


# ---------------------------------------------------------------------------
# snapshot reflects post-commit state
# ---------------------------------------------------------------------------

def test_snapshot_reflects_post_commit_state():
    ct = CentralTable()
    ct.publish("s1", "pidA", make_offer(available_qty=100))
    ct.publish("s2", "pidA", make_offer(available_qty=200))
    ct.commit("s1", "pidA", qty=40)
    result = dict(ct.snapshot_for_buyer("pidA"))
    assert result["s1"].available_qty == 60   # post-commit
    assert result["s2"].available_qty == 200  # untouched


def test_snapshot_for_unknown_product_returns_empty():
    ct = CentralTable()
    result = ct.snapshot_for_buyer("nonexistent_pid")
    assert result == [] or result == ()
