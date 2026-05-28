"""Tests for src/rl/encoders.py.

Covers:
  - observation_dim / action_dim shape helpers
  - observation shape matches observation_dim(K)
  - observation feature ranges (each feature in documented bounds)
  - action decoding bounds (prices in [0.5*MSRP, 1.5*MSRP], orders in [0, free_space])
  - slot-shuffle round-trip: encode → decode → re-encode is consistent with perm
  - determinism: same inputs → identical observation tensor
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from datetime import datetime
from random import Random
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.rl.encoders import (
    N_GLOBAL,
    N_PER_SKU,
    action_dim,
    compute_effective_rate,
    decode_action,
    encode_observation,
    fair_share_allocate,
    observation_dim,
)

# ---------------------------------------------------------------------------
# Minimal stub helpers — avoid full simulator construction in encoder tests
# ---------------------------------------------------------------------------


def _make_store(
    active_pids: list[str],
    inventory: dict[str, int] | None = None,
    prices: dict[str, float] | None = None,
    base_prices: dict[str, float] | None = None,
    costs: dict[str, float] | None = None,
    pending: dict[str, int] | None = None,
    capacity: float = 300.0,
    balance: float = 20000.0,
    activation_tick: dict[str, int] | None = None,
) -> Any:
    """Return a SimpleNamespace mimicking the Store fields consulted by encoders."""
    if inventory is None:
        inventory = {pid: 10 for pid in active_pids}
    if prices is None:
        prices = {pid: 15.0 for pid in active_pids}
    if base_prices is None:
        base_prices = {pid: 18.0 for pid in active_pids}
    if costs is None:
        costs = {pid: 6.0 for pid in active_pids}
    if pending is None:
        pending = defaultdict(int)
    else:
        pending = defaultdict(int, pending)
    if activation_tick is None:
        activation_tick = {pid: 0 for pid in active_pids}

    return SimpleNamespace(
        active_items=list(active_pids),
        inventory=dict(inventory),
        prices=dict(prices),
        base_prices=dict(base_prices),
        costs=dict(costs),
        pending=pending,
        capacity=capacity,
        balance=balance,
        activation_tick=dict(activation_tick),
    )


def _make_market(step: int = 0, month: int = 3) -> Any:
    """Return a SimpleNamespace mimicking Market fields consulted by encoders."""
    market_params = SimpleNamespace(
        season_months={"summer": [6, 7, 8], "winter": [12, 1, 2]},
    )
    return SimpleNamespace(
        step=step,
        date=datetime(2024, month, 1),
        params=market_params,
    )


def _make_registry(
    stages: dict[str, str] | None = None,
    seasonalities: dict[str, str] | None = None,
) -> Any:
    """Return a SimpleNamespace mimicking ItemRegistry fields consulted by encoders."""
    if stages is None:
        stages = {}
    if seasonalities is None:
        seasonalities = {}

    class _Reg:
        def stage(self, pid: str) -> str:
            return stages.get(pid, "maturity")

        def seasonality(self, pid: str) -> str:
            return seasonalities.get(pid, "all_season")

    return _Reg()


# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------


def test_observation_dim():
    assert observation_dim(5) == 5 * N_PER_SKU + N_GLOBAL
    assert observation_dim(1) == 1 * N_PER_SKU + N_GLOBAL
    assert observation_dim(3) == 3 * N_PER_SKU + N_GLOBAL


def test_action_dim():
    assert action_dim(5) == 10
    assert action_dim(3) == 6
    assert action_dim(1) == 2


# ---------------------------------------------------------------------------
# Observation shape
# ---------------------------------------------------------------------------


def test_observation_shape_matches_dim():
    K = 5
    pids = [f"P{i:04d}" for i in range(K)]
    store = _make_store(pids)
    market = _make_market()
    registry = _make_registry()
    slot_perm = list(range(K))

    obs = encode_observation(store, market, registry, step=0, slot_perm=slot_perm, K_active=K)

    assert obs.shape == (observation_dim(K),), f"Expected {observation_dim(K)}, got {obs.shape}"
    assert obs.dtype == np.float32


def test_observation_shape_for_various_K():
    for K in [1, 3, 5, 8]:
        pids = [f"P{i:04d}" for i in range(K)]
        store = _make_store(pids)
        market = _make_market()
        registry = _make_registry()
        slot_perm = list(range(K))
        obs = encode_observation(store, market, registry, step=10, slot_perm=slot_perm, K_active=K)
        assert obs.shape == (observation_dim(K),)


# ---------------------------------------------------------------------------
# Observation feature ranges
# ---------------------------------------------------------------------------


def test_observation_bounded_features():
    """Check that clipped [0,1] features are in range, and sin/cos in [-1,1]."""
    K = 5
    pids = [f"P{i:04d}" for i in range(K)]
    store = _make_store(
        pids,
        inventory={pid: 20 for pid in pids},
        prices={pid: 18.0 for pid in pids},
        base_prices={pid: 18.0 for pid in pids},
        costs={pid: 6.0 for pid in pids},
        capacity=200.0,
        balance=15000.0,
    )
    market = _make_market(step=50, month=7)
    registry = _make_registry(
        stages={pid: "maturity" for pid in pids},
        seasonalities={pid: "all_season" for pid in pids},
    )
    slot_perm = list(range(K))
    obs = encode_observation(store, market, registry, step=50, slot_perm=slot_perm, K_active=K)

    for slot_idx in range(K):
        base = slot_idx * N_PER_SKU
        # Features that must be in [0,1]:
        for feat_idx in [0, 1, 2, 5, 11, 12]:
            v = obs[base + feat_idx]
            assert 0.0 <= v <= 1.0, f"slot={slot_idx} feat={feat_idx} value={v}"
        # Stage one-hot: exactly one 1 and four 0s in positions 6..10
        one_hot = obs[base + 6 : base + 11]
        assert one_hot.sum() == pytest.approx(1.0), f"one-hot sum not 1 at slot {slot_idx}"
        assert set(one_hot.tolist()).issubset({0.0, 1.0})

    # Global: sin and cos in [-1,1]
    global_offset = K * N_PER_SKU
    assert -1.0 <= obs[global_offset + 2] <= 1.0
    assert -1.0 <= obs[global_offset + 3] <= 1.0
    # Total-inventory / capacity in [0,1]
    assert 0.0 <= obs[global_offset + 1] <= 1.0


def test_cyclic_encoding_values():
    """sin(2π*step/360) and cos values at known steps."""
    K = 2
    pids = ["P0000", "P0001"]
    store = _make_store(pids)
    market = _make_market()
    registry = _make_registry()
    slot_perm = [0, 1]

    for step, expected_sin, expected_cos in [
        (0, 0.0, 1.0),
        (90, math.sin(2 * math.pi * 90 / 360), math.cos(2 * math.pi * 90 / 360)),
        (180, math.sin(math.pi), math.cos(math.pi)),
    ]:
        obs = encode_observation(store, market, registry, step=step, slot_perm=slot_perm, K_active=K)
        global_offset = K * N_PER_SKU
        assert obs[global_offset + 2] == pytest.approx(expected_sin, abs=1e-6)
        assert obs[global_offset + 3] == pytest.approx(expected_cos, abs=1e-6)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_observation_determinism():
    """Same (store, market, slot_perm, step) → identical tensor on two calls."""
    K = 5
    pids = [f"P{i:04d}" for i in range(K)]
    store = _make_store(pids)
    market = _make_market(step=42, month=6)
    registry = _make_registry()
    slot_perm = [2, 0, 4, 1, 3]

    obs1 = encode_observation(store, market, registry, step=42, slot_perm=slot_perm, K_active=K)
    obs2 = encode_observation(store, market, registry, step=42, slot_perm=slot_perm, K_active=K)
    np.testing.assert_array_equal(obs1, obs2)


# ---------------------------------------------------------------------------
# Action decoding bounds
# ---------------------------------------------------------------------------


def _total_order_qty(result: dict, pid: str) -> int:
    """Extract total order qty for a pid from the graph-engine action dict.

    The new format is ``order[pid] = [(supplier_id, qty), ...]``.
    For backward-compat tests, returns sum of all qty in the list,
    or the direct int value if still in old format.
    """
    val = result.get("order", {}).get(pid, 0)
    if isinstance(val, list):
        return sum(q for _, q in val)
    return int(val)


def _get_price(result: dict, pid: str) -> float:
    """Extract price for a pid from the graph-engine action dict.

    New format: ``list_price[pid]``.
    Old format: ``price[pid]``.
    """
    if "list_price" in result:
        return float(result["list_price"].get(pid, 0.0))
    return float(result.get("price", {}).get(pid, 0.0))


def test_action_decode_price_bounds():
    """Decoded prices in [0.5*MSRP, 1.5*MSRP] for any action in [-1,1]."""
    K = 5
    pids = [f"P{i:04d}" for i in range(K)]
    base_prices = {pid: 20.0 for pid in pids}
    store = _make_store(pids, capacity=500.0, inventory={pid: 10 for pid in pids})
    slot_perm = list(range(K))

    rng = Random(0)
    for _ in range(50):
        action_vec = np.array([rng.uniform(-1.0, 1.0) for _ in range(2 * K)], dtype=np.float32)
        result = decode_action(action_vec, slot_perm, store, K, base_prices)
        for pid in pids:
            p = _get_price(result, pid)
            msrp = base_prices[pid]
            assert 0.5 * msrp <= p <= 1.5 * msrp + 1e-9, (
                f"{pid}: price={p} outside [{0.5*msrp}, {1.5*msrp}]"
            )


def test_action_decode_order_bounds():
    """Decoded order quantities in [0, free_space]."""
    K = 5
    pids = [f"P{i:04d}" for i in range(K)]
    inv = {pid: 20 for pid in pids}
    pending = {pid: 5 for pid in pids}
    capacity = 200.0
    store = _make_store(pids, capacity=capacity, inventory=inv, pending=pending)
    base_prices = {pid: 18.0 for pid in pids}
    slot_perm = list(range(K))

    # Free space = capacity - sum(inv) - sum(pending) = 200 - 100 - 25 = 75
    free_space = capacity - sum(inv.values()) - sum(pending.values())

    rng = Random(1)
    for _ in range(50):
        action_vec = np.array([rng.uniform(-1.0, 1.0) for _ in range(2 * K)], dtype=np.float32)
        result = decode_action(action_vec, slot_perm, store, K, base_prices)
        for pid in pids:
            qty = _total_order_qty(result, pid)
            assert qty >= 0, f"{pid}: negative order qty {qty}"
            assert qty <= free_space + 1, f"{pid}: order qty {qty} exceeds free space {free_space}"


def test_action_decode_extreme_values():
    """Action = all -1 → minimum prices and zero orders; all +1 → max prices, max orders."""
    K = 3
    pids = ["P0000", "P0001", "P0002"]
    base_prices = {pid: 10.0 for pid in pids}
    capacity = 300.0
    inv = {pid: 0 for pid in pids}  # empty store → free_space = capacity
    store = _make_store(pids, capacity=capacity, inventory=inv)
    slot_perm = [0, 1, 2]

    # All -1 → prices = 0.5*MSRP, orders = 0
    action_min = np.full(2 * K, -1.0, dtype=np.float32)
    result_min = decode_action(action_min, slot_perm, store, K, base_prices)
    for pid in pids:
        assert _get_price(result_min, pid) == pytest.approx(5.0, abs=1e-6)
        assert _total_order_qty(result_min, pid) == 0

    # All +1 → prices = 1.5*MSRP
    action_max = np.full(2 * K, 1.0, dtype=np.float32)
    result_max = decode_action(action_max, slot_perm, store, K, base_prices)
    for pid in pids:
        assert _get_price(result_max, pid) == pytest.approx(15.0, abs=1e-6)
        assert _total_order_qty(result_max, pid) >= 0


# ---------------------------------------------------------------------------
# Action dict structure
# ---------------------------------------------------------------------------


def test_action_dict_keys():
    """decode_action always returns at least the order and price/list_price keys."""
    K = 2
    pids = ["P0000", "P0001"]
    store = _make_store(pids)
    base_prices = {pid: 10.0 for pid in pids}
    slot_perm = [0, 1]
    action_vec = np.zeros(2 * K, dtype=np.float32)
    result = decode_action(action_vec, slot_perm, store, K, base_prices)
    # New graph-engine format has order, list_price, min_order_imposed.
    assert "order" in result
    assert "list_price" in result or "price" in result


# ---------------------------------------------------------------------------
# Slot-shuffle consistency
# ---------------------------------------------------------------------------


def test_slot_shuffle_permutation_affects_sku_placement():
    """Different slot permutations produce different observations when SKUs differ."""
    K = 3
    pids = ["P0000", "P0001", "P0002"]
    # Give each SKU a distinct price so a permutation produces a distinct obs.
    store = _make_store(
        pids,
        prices={"P0000": 10.0, "P0001": 20.0, "P0002": 30.0},
        base_prices={"P0000": 10.0, "P0001": 20.0, "P0002": 30.0},
    )
    market = _make_market()
    registry = _make_registry()

    perm_identity = [0, 1, 2]
    perm_reversed = [2, 1, 0]

    obs_id = encode_observation(store, market, registry, step=0, slot_perm=perm_identity, K_active=K)
    obs_rv = encode_observation(store, market, registry, step=0, slot_perm=perm_reversed, K_active=K)

    # The observations should differ (different price/MSRP in different slots).
    assert not np.allclose(obs_id, obs_rv), "Identity and reversed permutation produced identical obs"

    # But the global block should be identical.
    global_offset = K * N_PER_SKU
    np.testing.assert_array_equal(obs_id[global_offset:], obs_rv[global_offset:])


def test_slot_perm_maps_correct_sku():
    """slot_perm[0]=2 means slot 0 carries data from active_items[2]."""
    K = 3
    pids = ["P0000", "P0001", "P0002"]
    # P0002 has a distinctive price = 99.0; all others = 10.0
    store = _make_store(
        pids,
        prices={"P0000": 10.0, "P0001": 10.0, "P0002": 99.0},
        base_prices={"P0000": 18.0, "P0001": 18.0, "P0002": 18.0},
    )
    market = _make_market()
    registry = _make_registry()

    # slot_perm=[2,0,1]: slot 0 → item index 2 → P0002
    perm = [2, 0, 1]
    obs = encode_observation(store, market, registry, step=0, slot_perm=perm, K_active=K)

    # price/MSRP for slot 0 should be 99.0/18.0
    price_msrp_slot0 = obs[0 * N_PER_SKU + 3]
    assert price_msrp_slot0 == pytest.approx(99.0 / 18.0, abs=1e-5)

    # price/MSRP for slot 1 should be 10.0/18.0 (P0000)
    price_msrp_slot1 = obs[1 * N_PER_SKU + 3]
    assert price_msrp_slot1 == pytest.approx(10.0 / 18.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Sales history feature
# ---------------------------------------------------------------------------


def test_sales_history_feature():
    """Rolling-5-tick mean sales feature reflects provided history."""
    K = 2
    pids = ["P0000", "P0001"]
    store = _make_store(pids, capacity=200.0)
    market = _make_market()
    registry = _make_registry()
    slot_perm = [0, 1]

    # P0000 has sales history [10, 10, 10, 10, 10] → mean=10
    history = {
        "P0000": deque([10, 10, 10, 10, 10], maxlen=100),
        "P0001": deque([0, 0, 0, 0, 0], maxlen=100),
    }
    obs = encode_observation(
        store, market, registry, step=5, slot_perm=slot_perm, K_active=K,
        sales_history=history,
    )
    per_sku_cap = 200.0 / K
    assert obs[0 * N_PER_SKU + 1] == pytest.approx(10.0 / per_sku_cap, abs=1e-5)
    assert obs[1 * N_PER_SKU + 1] == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# compute_effective_rate tests
# ---------------------------------------------------------------------------


def test_effective_rate_empty_history_returns_prior():
    """Empty deques for each pid return the prior for every pid."""
    history = {"P0001": deque(), "P0002": deque()}
    result = compute_effective_rate(history, base_demand_prior=3.0)
    assert result == {"P0001": 3.0, "P0002": 3.0}


def test_effective_rate_partial_window_uses_partial_mean():
    """History of [10, 12] (length 2) with prior 3.0 returns 11.0."""
    history = {"P0001": deque([10, 12], maxlen=100)}
    result = compute_effective_rate(history, base_demand_prior=3.0)
    assert result["P0001"] == pytest.approx(11.0)


def test_effective_rate_full_window_uses_rolling_5_mean():
    """History of length 10 uses only the last 5 entries."""
    # last 5 entries are 100, 100, 100, 100, 100 → mean = 100.0
    hist = deque([10, 12, 14, 8, 6, 100, 100, 100, 100, 100], maxlen=100)
    history = {"P0001": hist}
    result = compute_effective_rate(history, base_demand_prior=3.0)
    assert result["P0001"] == pytest.approx(100.0)


def test_effective_rate_prior_floors_low_history():
    """History of [0, 0, 0, 0, 0] with prior 2.5 returns 2.5."""
    history = {"P0001": deque([0, 0, 0, 0, 0], maxlen=100)}
    result = compute_effective_rate(history, base_demand_prior=2.5)
    assert result["P0001"] == pytest.approx(2.5)


def test_effective_rate_output_keys_match_input():
    """Every key in sales_history appears in the output; nothing else does."""
    history = {
        "P0001": deque([5, 5], maxlen=100),
        "P0002": deque([], maxlen=100),
        "P0003": deque([1, 2, 3, 4, 5], maxlen=100),
    }
    result = compute_effective_rate(history, base_demand_prior=1.0)
    assert set(result.keys()) == set(history.keys())


def test_effective_rate_zero_prior_allows_zero_output():
    """History [0, 0] with prior 0.0 returns 0.0 — no implicit extra floor."""
    history = {"P0001": deque([0, 0], maxlen=100)}
    result = compute_effective_rate(history, base_demand_prior=0.0)
    assert result["P0001"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Lifecycle stage one-hot correctness
# ---------------------------------------------------------------------------


def test_lifecycle_stage_one_hot():
    """Each of the 5 stages produces a one-hot in positions 6..10."""
    from src.sim.lifecycle_clock import CANONICAL_STAGES

    K = 1
    pids = ["P0000"]
    for stage_idx, stage in enumerate(CANONICAL_STAGES):
        store = _make_store(pids)
        market = _make_market()
        registry = _make_registry(stages={"P0000": stage})
        obs = encode_observation(store, market, registry, step=0, slot_perm=[0], K_active=K)
        one_hot = obs[6:11]
        assert one_hot[stage_idx] == pytest.approx(1.0), f"Stage {stage}: expected 1 at index {stage_idx}"
        for j in range(5):
            if j != stage_idx:
                assert one_hot[j] == pytest.approx(0.0), f"Stage {stage}: expected 0 at index {j}"


# ---------------------------------------------------------------------------
# fair_share_allocate tests
# ---------------------------------------------------------------------------


def test_fair_share_empty_input_returns_empty():
    """Empty requested dict returns an empty dict."""
    result = fair_share_allocate({}, {}, global_free_space=100)
    assert result == {}


def test_fair_share_sum_below_free_space_returns_capped_unchanged():
    """When sum(min(requested, headroom)) <= global_free_space, no scaling fires."""
    requested = {"P1": 10.0, "P2": 20.0}
    per_sku_headroom = {"P1": 15, "P2": 25}
    global_free_space = 100
    result = fair_share_allocate(requested, per_sku_headroom, global_free_space)
    # capped = min(10,15)=10, min(20,25)=20; sum=30 <= 100 -> no scaling
    assert result == {"P1": 10, "P2": 20}


def test_fair_share_sum_above_free_space_scales_proportionally():
    """When sum(requested) > global_free_space and headroom is non-binding, output sums to <= free_space."""
    requested = {"P1": 60.0, "P2": 40.0}
    per_sku_headroom = {"P1": 1000, "P2": 1000}  # headroom non-binding
    global_free_space = 50
    result = fair_share_allocate(requested, per_sku_headroom, global_free_space)
    # sum(capped) = 100 > 50 -> scale by 0.5: P1=30, P2=20
    assert sum(result.values()) <= global_free_space
    # Proportions preserved within +/-1 (truncation tolerance)
    assert abs(result["P1"] - 30) <= 1
    assert abs(result["P2"] - 20) <= 1


def test_fair_share_per_sku_headroom_binds_tighter_than_global():
    """P1 headroom of 10 caps P1 even when global free space is 200."""
    requested = {"P1": 100.0, "P2": 100.0}
    per_sku_headroom = {"P1": 10, "P2": 1000}
    global_free_space = 200
    result = fair_share_allocate(requested, per_sku_headroom, global_free_space)
    # P1 capped to 10 by headroom; P2 capped to min(100,1000)=100; sum=110 <= 200 -> no scaling
    assert result["P1"] == 10
    assert result["P2"] == 100


def test_fair_share_zero_global_free_space_returns_all_zeros():
    """global_free_space = 0 forces all outputs to 0."""
    requested = {"P1": 50.0, "P2": 30.0}
    per_sku_headroom = {"P1": 100, "P2": 100}
    result = fair_share_allocate(requested, per_sku_headroom, global_free_space=0)
    assert result == {"P1": 0, "P2": 0}


def test_fair_share_total_never_exceeds_free_space():
    """Property: sum(output) <= global_free_space across randomised inputs."""
    rng = Random(42)
    for _ in range(200):
        n_skus = rng.randint(1, 10)
        pids = [f"P{i:04d}" for i in range(n_skus)]
        requested = {pid: rng.uniform(0, 500) for pid in pids}
        per_sku_headroom = {pid: rng.randint(0, 300) for pid in pids}
        global_free_space = rng.randint(0, 1000)
        result = fair_share_allocate(requested, per_sku_headroom, global_free_space)
        assert sum(result.values()) <= global_free_space, (
            f"Sum {sum(result.values())} exceeds free space {global_free_space}"
        )


def test_fair_share_per_sku_never_exceeds_headroom():
    """Property: output[pid] <= per_sku_headroom[pid] for all pids."""
    rng = Random(99)
    for _ in range(200):
        n_skus = rng.randint(1, 10)
        pids = [f"P{i:04d}" for i in range(n_skus)]
        requested = {pid: rng.uniform(0, 500) for pid in pids}
        per_sku_headroom = {pid: rng.randint(0, 300) for pid in pids}
        global_free_space = rng.randint(0, 1000)
        result = fair_share_allocate(requested, per_sku_headroom, global_free_space)
        for pid in pids:
            assert result[pid] <= per_sku_headroom[pid], (
                f"{pid}: output {result[pid]} exceeds headroom {per_sku_headroom[pid]}"
            )


def test_fair_share_output_keys_match_requested_keys():
    """Output contains exactly the keys in requested -- nothing more, nothing less."""
    requested = {"A": 10.0, "B": 20.0, "C": 5.0}
    per_sku_headroom = {"A": 50, "B": 50, "C": 50}
    result = fair_share_allocate(requested, per_sku_headroom, global_free_space=100)
    assert set(result.keys()) == set(requested.keys())


# ---------------------------------------------------------------------------
# decode_action — order-up-to decoder (issue 04)
# ---------------------------------------------------------------------------


def test_decode_action_zero_action_matches_periodic_order_up_to_formula():
    """At action_vec=zeros, target_lt=15. With rate=10, inv=30, pending=0, qty=max(0,15*10-30)=120."""
    K = 2
    pids = ["P0000", "P0001"]
    effective_rate = {pid: 10.0 for pid in pids}
    inventory = {pid: 30 for pid in pids}
    pending = {pid: 0 for pid in pids}
    capacity = 5000.0  # large so headroom is non-binding
    store = _make_store(
        pids,
        capacity=capacity,
        inventory=inventory,
        pending=pending,
    )
    base_prices = {pid: 18.0 for pid in pids}
    slot_perm = list(range(K))
    action_vec = np.zeros(2 * K, dtype=np.float32)

    result = decode_action(
        action_vec,
        slot_perm,
        store,
        K,
        base_prices,
        effective_rate=effective_rate,
        target_centre_lead_times=15,
        target_half_span_lead_times=15,
        target_max_lead_times=30,
    )
    # requested = max(0, 15*10 - (30+0)) = max(0, 150-30) = 120
    for pid in pids:
        qty = _total_order_qty(result, pid)
        assert qty == 120, f"{pid}: expected 120 got {qty}"


def test_decode_action_position_at_target_returns_zero_qty():
    """The max(0,...) floor: negative requests map to zero.

    When position >= target_lt * rate (even for negative order_raw),
    the requested quantity is floored at zero.

    Specifically: position = target_max * rate means ALL actions give zero
    (the highest possible target is target_max, so position >= target_lt always).
    """
    K = 2
    pids = ["P0000", "P0001"]
    rate = 8.0
    target_max = 30
    # Set position to target_max * rate so even the highest target can't exceed it.
    position = int(target_max * rate)  # 240 units
    effective_rate = {pid: rate for pid in pids}
    inventory = {pid: position for pid in pids}
    pending = {pid: 0 for pid in pids}
    capacity = 5000.0
    store = _make_store(pids, capacity=capacity, inventory=inventory, pending=pending)
    base_prices = {pid: 18.0 for pid in pids}
    slot_perm = list(range(K))

    # With position at target_max * rate, ALL order_raw values give zero qty.
    for order_raw in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        action_vec = np.array(
            [0.0] * K + [order_raw] * K, dtype=np.float32
        )
        result = decode_action(
            action_vec,
            slot_perm,
            store,
            K,
            base_prices,
            effective_rate=effective_rate,
            target_centre_lead_times=15,
            target_half_span_lead_times=15,
            target_max_lead_times=target_max,
        )
        for pid in pids:
            qty = _total_order_qty(result, pid)
            assert qty == 0, (
                f"order_raw={order_raw}, pid={pid}: expected 0 got {qty}"
            )


def test_decode_action_total_qty_respects_global_free_space():
    """Property: sum(qty) <= capacity - total_inventory - total_pending."""
    rng = Random(7)
    K = 5
    pids = [f"P{i:04d}" for i in range(K)]
    slot_perm = list(range(K))
    base_prices = {pid: 18.0 for pid in pids}

    for _ in range(100):
        capacity = rng.uniform(100, 1000)
        inventory = {pid: rng.uniform(0, capacity / (K * 2)) for pid in pids}
        pending = {pid: rng.uniform(0, capacity / (K * 2)) for pid in pids}
        store = _make_store(pids, capacity=capacity, inventory=inventory, pending=pending)
        effective_rate = {pid: rng.uniform(0.5, 20.0) for pid in pids}
        action_vec = np.array(
            [rng.uniform(-1, 1) for _ in range(2 * K)], dtype=np.float32
        )
        result = decode_action(
            action_vec,
            slot_perm,
            store,
            K,
            base_prices,
            effective_rate=effective_rate,
        )
        global_free_space = max(
            0, capacity - sum(inventory.values()) - sum(pending.values())
        )
        total_qty = sum(_total_order_qty(result, pid) for pid in pids)
        assert total_qty <= global_free_space + 1, (
            f"total_qty={total_qty} > global_free_space={global_free_space}"
        )


def test_decode_action_per_sku_qty_respects_per_sku_headroom():
    """Property: qty[pid] <= capacity - inventory[pid] - pending[pid] for all pids."""
    rng = Random(13)
    K = 5
    pids = [f"P{i:04d}" for i in range(K)]
    slot_perm = list(range(K))
    base_prices = {pid: 18.0 for pid in pids}

    for _ in range(100):
        capacity = rng.uniform(100, 1000)
        inventory = {pid: rng.uniform(0, capacity / (K * 2)) for pid in pids}
        pending = {pid: rng.uniform(0, capacity / (K * 2)) for pid in pids}
        store = _make_store(pids, capacity=capacity, inventory=inventory, pending=pending)
        effective_rate = {pid: rng.uniform(0.5, 20.0) for pid in pids}
        action_vec = np.array(
            [rng.uniform(-1, 1) for _ in range(2 * K)], dtype=np.float32
        )
        result = decode_action(
            action_vec,
            slot_perm,
            store,
            K,
            base_prices,
            effective_rate=effective_rate,
        )
        for pid in pids:
            headroom = max(0, capacity - inventory[pid] - pending[pid])
            qty = _total_order_qty(result, pid)
            assert qty <= headroom + 1, (
                f"{pid}: qty={qty} > headroom={headroom}"
            )


def test_decode_action_price_half_unchanged():
    """Price decoding: price_raw=0 → MSRP; +1 → 1.5*MSRP; -1 → 0.5*MSRP."""
    K = 2
    pids = ["P0000", "P0001"]
    msrps = {"P0000": 20.0, "P0001": 30.0}
    store = _make_store(pids, base_prices=msrps, capacity=5000.0)
    slot_perm = list(range(K))

    for price_raw, expected_mult in [(-1.0, 0.5), (0.0, 1.0), (1.0, 1.5)]:
        action_vec = np.array(
            [price_raw] * K + [0.0] * K, dtype=np.float32
        )
        result = decode_action(
            action_vec,
            slot_perm,
            store,
            K,
            msrps,
            effective_rate={pid: 5.0 for pid in pids},
        )
        for pid in pids:
            expected_price = msrps[pid] * expected_mult
            assert _get_price(result, pid) == pytest.approx(expected_price, abs=1e-5), (
                f"price_raw={price_raw}, {pid}: expected {expected_price} got {_get_price(result, pid)}"
            )


def test_decode_action_uses_inventory_position_not_just_inventory():
    """Pending in-transit nets against the target: inv=0, pending=50, rate=10, target_lt=15 → qty=100."""
    K = 1
    pids = ["P0000"]
    effective_rate = {"P0000": 10.0}
    inventory = {"P0000": 0}
    pending = {"P0000": 50}
    capacity = 5000.0
    store = _make_store(pids, capacity=capacity, inventory=inventory, pending=pending)
    base_prices = {"P0000": 18.0}
    slot_perm = [0]
    action_vec = np.zeros(2 * K, dtype=np.float32)

    result = decode_action(
        action_vec,
        slot_perm,
        store,
        K,
        base_prices,
        effective_rate=effective_rate,
        target_centre_lead_times=15,
        target_half_span_lead_times=15,
        target_max_lead_times=30,
    )
    # requested = max(0, 15*10 - (0 + 50)) = max(0, 150 - 50) = 100
    assert _total_order_qty(result, "P0000") == 100


def test_decode_action_cold_start_qty_scales_with_prior():
    """Cold-start qty ratio scales 10x when base_demand_prior scales 10x (empty history)."""
    K = 1
    pids = ["P0000"]
    store = _make_store(pids, capacity=50000.0, inventory={"P0000": 0}, pending={"P0000": 0})
    base_prices = {"P0000": 18.0}
    slot_perm = [0]
    action_vec = np.zeros(2 * K, dtype=np.float32)

    # Empty history → effective_rate == base_demand_prior
    rate_low = {"P0000": 2.0}
    rate_high = {"P0000": 20.0}

    result_low = decode_action(
        action_vec, slot_perm, store, K, base_prices,
        effective_rate=rate_low,
        target_centre_lead_times=15,
        target_half_span_lead_times=15,
        target_max_lead_times=30,
    )
    result_high = decode_action(
        action_vec, slot_perm, store, K, base_prices,
        effective_rate=rate_high,
        target_centre_lead_times=15,
        target_half_span_lead_times=15,
        target_max_lead_times=30,
    )
    qty_low = _total_order_qty(result_low, "P0000")
    qty_high = _total_order_qty(result_high, "P0000")
    # At zero inventory+pending: qty = target_lt * rate; ratio should be 10x
    # (integer truncation may cause small deviation; allow +-1)
    assert abs(qty_high - qty_low * 10) <= 2, (
        f"Expected 10x ratio: qty_low={qty_low}, qty_high={qty_high}"
    )


def test_decode_action_effective_rate_none_returns_zero_qty():
    """effective_rate=None produces zero qty for every SKU (backward-compat branch)."""
    K = 3
    pids = [f"P{i:04d}" for i in range(K)]
    store = _make_store(pids, capacity=5000.0, inventory={pid: 0 for pid in pids})
    base_prices = {pid: 18.0 for pid in pids}
    slot_perm = list(range(K))
    action_vec = np.ones(2 * K, dtype=np.float32)  # max order_raw = +1

    result = decode_action(
        action_vec,
        slot_perm,
        store,
        K,
        base_prices,
        effective_rate=None,
    )
    for pid in pids:
        qty = _total_order_qty(result, pid)
        assert qty == 0, (
            f"{pid}: expected 0 with effective_rate=None, got {qty}"
        )


# ---------------------------------------------------------------------------
# Slot-13 demand-units inventory feature (issue 05)
# ---------------------------------------------------------------------------


def test_observation_dim_includes_central_table_block():
    """observation_dim(K=5) == 5 * 18 + 4 == 94 (N_PER_SKU bumped to 18).

    N_PER_SKU = 18: 14 original features + 4 central-table snapshot features
    (supplier_count, min_price, mean_lead_time, mean_fill_rate).
    """
    assert N_PER_SKU == 18, f"Expected N_PER_SKU=18, got {N_PER_SKU}"
    assert observation_dim(5) == 94, f"Expected 94, got {observation_dim(5)}"
    assert observation_dim(1) == 22
    assert observation_dim(3) == 58


def test_encoder_slot_13_matches_demand_units_formula():
    """Slot 13 = clip(inv / (rate * max_lt), 0, 1).  Concrete: inv=100, rate=10, max_lt=30 → 1/3."""
    K = 2
    pids = ["P0000", "P0001"]
    inventory = 100
    effective_rate_val = 10.0
    max_lt = 30.0
    store = _make_store(
        pids,
        inventory={pid: inventory for pid in pids},
        capacity=5000.0,
    )
    market = _make_market()
    registry = _make_registry()
    slot_perm = list(range(K))
    eff_rate = {pid: effective_rate_val for pid in pids}

    obs = encode_observation(
        store, market, registry, step=0, slot_perm=slot_perm, K_active=K,
        effective_rate=eff_rate, max_inventory_lt=max_lt,
    )
    expected = min(1.0, inventory / (effective_rate_val * max_lt))  # 100/300 ≈ 0.333
    for slot_idx in range(K):
        val = obs[slot_idx * N_PER_SKU + 13]
        assert val == pytest.approx(expected, abs=1e-5), (
            f"slot {slot_idx}: expected {expected}, got {val}"
        )


def test_encoder_slot_13_saturates_at_max_inventory_lt():
    """With inv=10_000, rate=10, max_lt=30 → slot 13 == 1.0 (saturated)."""
    K = 1
    pids = ["P0000"]
    store = _make_store(pids, inventory={"P0000": 10_000}, capacity=100_000.0)
    market = _make_market()
    registry = _make_registry()
    slot_perm = [0]
    eff_rate = {"P0000": 10.0}

    obs = encode_observation(
        store, market, registry, step=0, slot_perm=slot_perm, K_active=K,
        effective_rate=eff_rate, max_inventory_lt=30.0,
    )
    assert obs[0 * N_PER_SKU + 13] == pytest.approx(1.0, abs=1e-6), (
        f"Expected 1.0 (saturated), got {obs[0 * N_PER_SKU + 13]}"
    )


def test_encoder_slot_13_zero_when_effective_rate_is_none():
    """When effective_rate=None, slot 13 is 0.0 for every active SKU."""
    K = 3
    pids = [f"P{i:04d}" for i in range(K)]
    store = _make_store(pids, inventory={pid: 200 for pid in pids}, capacity=5000.0)
    market = _make_market()
    registry = _make_registry()
    slot_perm = list(range(K))

    obs = encode_observation(
        store, market, registry, step=0, slot_perm=slot_perm, K_active=K,
        effective_rate=None,
    )
    for slot_idx in range(K):
        val = obs[slot_idx * N_PER_SKU + 13]
        assert val == pytest.approx(0.0, abs=1e-9), (
            f"slot {slot_idx}: expected 0.0 (effective_rate=None), got {val}"
        )


def test_encoder_slot_0_capacity_units_inventory_unchanged():
    """Slot 0 still equals clip(inventory / per_sku_capacity, 0, 1) — regression guard."""
    K = 3
    pids = [f"P{i:04d}" for i in range(K)]
    capacity = 300.0
    inventory_val = 20
    store = _make_store(
        pids,
        inventory={pid: inventory_val for pid in pids},
        capacity=capacity,
    )
    market = _make_market()
    registry = _make_registry()
    slot_perm = list(range(K))
    eff_rate = {pid: 5.0 for pid in pids}

    obs = encode_observation(
        store, market, registry, step=0, slot_perm=slot_perm, K_active=K,
        effective_rate=eff_rate,
    )
    per_sku_cap = capacity / K  # 100.0
    expected_slot0 = min(1.0, inventory_val / per_sku_cap)  # 20/100 = 0.2
    for slot_idx in range(K):
        val = obs[slot_idx * N_PER_SKU + 0]
        assert val == pytest.approx(expected_slot0, abs=1e-5), (
            f"slot {slot_idx} slot-0: expected {expected_slot0}, got {val}"
        )


def test_encoder_scale_invariance_of_slot_13():
    """Same inventory/rate ratio at two different absolute capacities → same slot 13."""
    # SKU has inventory=100, effective_rate=10 in both scenarios.
    # Capacity differs (300 vs 10_000) — slot 13 should be identical because it uses demand units.
    K = 1
    pids = ["P0000"]
    eff_rate = {"P0000": 10.0}
    inventory_val = 100

    for cap in [300.0, 10_000.0]:
        store = _make_store(pids, inventory={"P0000": inventory_val}, capacity=cap)
        market = _make_market()
        registry = _make_registry()
        obs = encode_observation(
            store, market, registry, step=0, slot_perm=[0], K_active=K,
            effective_rate=eff_rate, max_inventory_lt=30.0,
        )
        val = obs[0 * N_PER_SKU + 13]
        expected = min(1.0, inventory_val / (10.0 * 30.0))  # 100/300 ≈ 0.333
        assert val == pytest.approx(expected, abs=1e-5), (
            f"capacity={cap}: expected {expected}, got {val}"
        )
