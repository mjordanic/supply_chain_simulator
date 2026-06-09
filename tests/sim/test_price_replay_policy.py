"""Tests for PriceReplayPolicy (issue 04).

Acceptance criteria:
- Wrapper delegates to the inner policy: order decisions are bit-identical with
  and without the wrapper (same seeds).
- Offers published by a wrapped node carry the observed price for that tick,
  for every carried product with a price array.
- Products without a price array keep the inner policy's price (partial
  coverage works).
- Price array shorter than n_steps fails fast at construction/build time.
- Composes with the multi-supplier textbook policy family.
"""

from __future__ import annotations

from typing import Any, Mapping

import pytest

from src.sim.policy import IntermediatePolicy, MultiSupplierTextbookPolicy, OrderUpToPolicy, PriceReplayPolicy


# ── minimal inner-policy stub ─────────────────────────────────────────────────


class _FixedPriceInner(IntermediatePolicy):
    """Trivial IntermediatePolicy stub for testing.

    Always returns a fixed price for every pid in obs_intermediate['inventory'],
    with no orders.  Lets us verify that PriceReplayPolicy overwrites only the
    pids it knows about.
    """

    BASE_PRICE = 10.0

    def __init__(self, policy_seed: int | None = None) -> None:
        super().__init__(policy_seed=policy_seed)

    def decide(
        self, obs_intermediate: Mapping[str, Any], central_table: Any
    ) -> dict[str, Any]:
        inventory = dict(obs_intermediate.get("inventory", {}))
        return {
            "order": {},
            "list_price": {pid: self.BASE_PRICE for pid in inventory},
            "min_order_imposed": {},
        }


def _obs(tick: int, inventory: dict) -> dict:
    """Build a minimal obs_intermediate dict."""
    return {
        "tick": tick,
        "inventory": inventory,
        "pending": {},
        "cash": 100_000.0,
        "capacity": 1000,
        "list_prices": {pid: 10.0 for pid in inventory},
        "min_order_imposed": {},
        "observed_sales": {},
        "direct_supplier_ids": {"supplier_0"},
    }


class _FakeCentralTable:
    """Stub central table — never queried in these tests."""

    def snapshot_for_buyer(self, pid: str):
        return []


_TABLE = _FakeCentralTable()


# ── AC4: short array fails at construction ────────────────────────────────────


def test_short_price_array_raises_at_construction():
    """A price array shorter than n_steps must raise ValueError at build time."""
    inner = _FixedPriceInner()
    with pytest.raises(ValueError, match="n_steps"):
        PriceReplayPolicy(inner, prices={"P0000": [1.0, 2.0]}, n_steps=10)


def test_exact_length_array_is_accepted():
    """An array of length exactly n_steps must be accepted."""
    inner = _FixedPriceInner()
    policy = PriceReplayPolicy(inner, prices={"P0000": [5.0] * 10}, n_steps=10)
    assert policy is not None


def test_longer_array_is_accepted():
    """An array longer than n_steps must be accepted (extra values unused)."""
    inner = _FixedPriceInner()
    policy = PriceReplayPolicy(inner, prices={"P0000": [5.0] * 20}, n_steps=10)
    assert policy is not None


# ── AC1: order decisions are bit-identical ────────────────────────────────────


def test_order_decisions_unchanged_by_wrapper():
    """Wrapper passes through order decisions unchanged (bit-identical inner)."""
    seed = 42
    prices = {"P0000": [99.0] * 30, "P0001": [88.0] * 30}
    inventory = {"P0000": 5, "P0001": 0}

    inner_alone = _FixedPriceInner(policy_seed=seed)
    inner_wrapped = _FixedPriceInner(policy_seed=seed)
    wrapper = PriceReplayPolicy(inner_wrapped, prices=prices, n_steps=30)

    for tick in range(5):
        obs = _obs(tick, inventory)
        result_alone = inner_alone.decide(obs, _TABLE)
        result_wrapped = wrapper.decide(obs, _TABLE)
        assert result_alone["order"] == result_wrapped["order"], (
            f"tick {tick}: order mismatch — inner changed by wrapper"
        )


# ── AC2: covered products get observed prices ─────────────────────────────────


def test_covered_products_get_observed_price():
    """Products in the prices dict get the per-tick observed price."""
    prices = {
        "P0000": [10.0, 20.0, 30.0, 40.0, 50.0],
        "P0001": [1.0, 2.0, 3.0, 4.0, 5.0],
    }
    inner = _FixedPriceInner()
    wrapper = PriceReplayPolicy(inner, prices=prices, n_steps=5)
    inventory = {"P0000": 10, "P0001": 10}

    for tick in range(5):
        obs = _obs(tick, inventory)
        result = wrapper.decide(obs, _TABLE)
        assert result["list_price"]["P0000"] == prices["P0000"][tick]
        assert result["list_price"]["P0001"] == prices["P0001"][tick]


# ── AC3: products without a price array keep inner policy's price ─────────────


def test_uncovered_products_keep_inner_price():
    """Products NOT in the prices dict keep the inner policy's list_price."""
    prices = {"P0000": [99.0] * 10}  # only P0000 covered
    inner = _FixedPriceInner()  # inner sets 10.0 for all
    wrapper = PriceReplayPolicy(inner, prices=prices, n_steps=10)
    inventory = {"P0000": 5, "P0001": 5}  # P0001 not in prices

    obs = _obs(0, inventory)
    result = wrapper.decide(obs, _TABLE)
    # P0000 → observed price
    assert result["list_price"]["P0000"] == 99.0
    # P0001 → inner's price (10.0)
    assert result["list_price"]["P0001"] == _FixedPriceInner.BASE_PRICE


# ── AC5: composes with the multi-supplier textbook family ─────────────────────


def test_composes_with_order_up_to_policy():
    """PriceReplayPolicy wraps OrderUpToPolicy without interfering with its logic."""
    inner = OrderUpToPolicy(
        policy_seed=0,
        delivery_lag=2,
        unit_cost=5.0,
        list_price_out=0.0,
    )
    prices = {"P0000": [42.5] * 50}
    wrapper = PriceReplayPolicy(inner, prices=prices, n_steps=50)

    inventory = {"P0000": 100}
    obs = _obs(0, inventory)
    result = wrapper.decide(obs, _TABLE)

    # The wrapper must produce a valid action dict.
    assert "order" in result
    assert "list_price" in result
    # The observed price must be posted for the covered product.
    assert result["list_price"]["P0000"] == 42.5
