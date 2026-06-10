"""Tests for src/rl/set_encoder.py.

Covers the new variable-K set encoder/decoder:
  - Row layout matches documented feature list (named constants, no magic offsets)
  - Mask correct for K < K_max; padded rows are exactly zero
  - Decode honours the mask: no orders/prices emitted for padded slots
  - Round-trip through the Arbiter produces per-pid dicts only for active products
  - Layout version constant exists and is the single source of truth
  - Existing flat codec (encoders.py) is untouched
"""

from __future__ import annotations

import math
from collections import deque
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.rl.set_encoder import (
    K_MAX,
    F,
    OBS_LAYOUT_VERSION,
    # Named index constants
    ROW_INVENTORY,
    ROW_SALES,
    ROW_PENDING,
    ROW_PRICE_RATIO,
    ROW_MSRP_RATIO,
    ROW_COST_RATIO,
    ROW_SUPPLIER_COUNT,
    ROW_MIN_PRICE,
    ROW_FILL_RATE,
    ROW_CASH,
    ROW_TOTAL_INV,
    ROW_SIN,
    ROW_COS,
    ROW_CONTENTION_QTY,
    ROW_CONTENTION_COST,
    ROW_MASK,
    # Public API
    encode_set_observation,
    decode_set_action,
    set_obs_shape,
    set_action_shape,
)


# ---------------------------------------------------------------------------
# Minimal stubs
# ---------------------------------------------------------------------------


def _make_node(
    pids: list[str],
    inventory: dict[str, int] | None = None,
    list_prices: dict[str, float] | None = None,
    base_prices: dict[str, float] | None = None,
    costs: dict[str, float] | None = None,
    pending_nested: dict[str, dict[str, int]] | None = None,
    capacity: float = 1000.0,
    cash: float = 50_000.0,
    carried_products: set[str] | None = None,
) -> Any:
    """Stub IntermediateNode."""
    if inventory is None:
        inventory = {pid: 20 for pid in pids}
    if list_prices is None:
        list_prices = {pid: 15.0 for pid in pids}
    if base_prices is None:
        base_prices = {pid: 18.0 for pid in pids}
    if costs is None:
        costs = {}
    if pending_nested is None:
        pending_nested = {}
    if carried_products is None:
        carried_products = set(pids)

    return SimpleNamespace(
        inventory=dict(inventory),
        list_prices=dict(list_prices),
        base_prices=dict(base_prices),
        costs=dict(costs),
        pending=dict(pending_nested),
        capacity=capacity,
        cash=cash,
        carried_products=set(carried_products),
    )


def _make_market(step: int = 0) -> Any:
    return SimpleNamespace(
        step=step,
        date=SimpleNamespace(month=3),
        params=SimpleNamespace(season_months={}),
    )


# ---------------------------------------------------------------------------
# Shape / constants
# ---------------------------------------------------------------------------


def test_layout_constants_are_defined():
    """K_MAX, F, and named row index constants are exported."""
    assert K_MAX == 32
    assert F >= 16  # at least 16 features per row


def test_obs_layout_version_is_int():
    """OBS_LAYOUT_VERSION is an integer >= 1."""
    assert isinstance(OBS_LAYOUT_VERSION, int)
    assert OBS_LAYOUT_VERSION >= 1


def test_set_obs_shape():
    """set_obs_shape() returns (K_MAX, F)."""
    shape = set_obs_shape()
    assert shape == (K_MAX, F)


def test_set_action_shape():
    """set_action_shape() returns (K_MAX, 3)."""
    shape = set_action_shape()
    assert shape == (K_MAX, 3)


def test_row_indices_within_f():
    """All named row index constants are in [0, F)."""
    indices = [
        ROW_INVENTORY, ROW_SALES, ROW_PENDING, ROW_PRICE_RATIO,
        ROW_MSRP_RATIO, ROW_COST_RATIO, ROW_SUPPLIER_COUNT, ROW_MIN_PRICE,
        ROW_FILL_RATE, ROW_CASH, ROW_TOTAL_INV, ROW_SIN, ROW_COS,
        ROW_CONTENTION_QTY, ROW_CONTENTION_COST, ROW_MASK,
    ]
    for idx in indices:
        assert 0 <= idx < F, f"Index {idx} out of range [0, {F})"


def test_row_indices_are_unique():
    """All named row index constants are distinct."""
    indices = [
        ROW_INVENTORY, ROW_SALES, ROW_PENDING, ROW_PRICE_RATIO,
        ROW_MSRP_RATIO, ROW_COST_RATIO, ROW_SUPPLIER_COUNT, ROW_MIN_PRICE,
        ROW_FILL_RATE, ROW_CASH, ROW_TOTAL_INV, ROW_SIN, ROW_COS,
        ROW_CONTENTION_QTY, ROW_CONTENTION_COST, ROW_MASK,
    ]
    assert len(indices) == len(set(indices)), "Duplicate row index constants"


# ---------------------------------------------------------------------------
# Observation shape
# ---------------------------------------------------------------------------


def test_encode_returns_kmax_f_tensor():
    """encode_set_observation returns a float32 array of shape (K_MAX, F)."""
    K = 5
    pids = [f"P{i:04d}" for i in range(K)]
    node = _make_node(pids)
    market = _make_market()

    obs = encode_set_observation(node, market, active_subset=pids, initial_cash=50_000.0)

    assert obs.shape == (K_MAX, F)
    assert obs.dtype == np.float32


def test_encode_dtype_is_float32():
    """Observation tensor dtype is float32."""
    pids = ["P0001", "P0002"]
    node = _make_node(pids)
    market = _make_market()
    obs = encode_set_observation(node, market, active_subset=pids, initial_cash=50_000.0)
    assert obs.dtype == np.float32


# ---------------------------------------------------------------------------
# Mask correctness
# ---------------------------------------------------------------------------


def test_active_rows_have_mask_one():
    """Active product rows have mask channel == 1.0."""
    K = 4
    pids = [f"P{i:04d}" for i in range(K)]
    node = _make_node(pids)
    market = _make_market()

    obs = encode_set_observation(node, market, active_subset=pids, initial_cash=50_000.0)

    for i in range(K):
        assert obs[i, ROW_MASK] == pytest.approx(1.0), f"Row {i} should be active (mask=1)"


def test_padded_rows_have_mask_zero():
    """Padded rows (K to K_MAX - 1) have mask channel == 0.0."""
    K = 3
    pids = [f"P{i:04d}" for i in range(K)]
    node = _make_node(pids)
    market = _make_market()

    obs = encode_set_observation(node, market, active_subset=pids, initial_cash=50_000.0)

    for i in range(K, K_MAX):
        assert obs[i, ROW_MASK] == pytest.approx(0.0), f"Row {i} should be padded (mask=0)"


def test_padded_rows_are_all_zero():
    """Padded rows (beyond K active products) must be exactly zero in all features."""
    K = 5
    pids = [f"P{i:04d}" for i in range(K)]
    node = _make_node(pids)
    market = _make_market()

    obs = encode_set_observation(node, market, active_subset=pids, initial_cash=50_000.0)

    for i in range(K, K_MAX):
        row = obs[i, :]
        assert np.all(row == 0.0), (
            f"Padded row {i} is not all-zero: {row}"
        )


def test_k_equals_kmax_no_padding():
    """When K == K_MAX, all rows are active (no padded rows)."""
    K = K_MAX
    pids = [f"P{i:04d}" for i in range(K)]
    node = _make_node(pids, capacity=K * 100.0)
    market = _make_market()

    obs = encode_set_observation(node, market, active_subset=pids, initial_cash=50_000.0)

    for i in range(K_MAX):
        assert obs[i, ROW_MASK] == pytest.approx(1.0), f"Row {i} should be active"


def test_k_equals_one_only_first_row_active():
    """K = 1 means only the first row is active."""
    pids = ["P0001"]
    node = _make_node(pids)
    market = _make_market()

    obs = encode_set_observation(node, market, active_subset=pids, initial_cash=50_000.0)

    assert obs[0, ROW_MASK] == pytest.approx(1.0)
    for i in range(1, K_MAX):
        assert obs[i, ROW_MASK] == pytest.approx(0.0)
        assert np.all(obs[i, :] == 0.0)


# ---------------------------------------------------------------------------
# Feature values / ranges
# ---------------------------------------------------------------------------


def test_inventory_feature_range():
    """ROW_INVENTORY feature is in [0, 1] for any capacity."""
    K = 4
    pids = [f"P{i:04d}" for i in range(K)]
    node = _make_node(pids, inventory={pid: 50 for pid in pids}, capacity=400.0)
    market = _make_market()

    obs = encode_set_observation(node, market, active_subset=pids, initial_cash=50_000.0)

    for i in range(K):
        assert 0.0 <= obs[i, ROW_INVENTORY] <= 1.0


def test_mask_channel_is_last():
    """ROW_MASK == F - 1: mask is the final channel."""
    assert ROW_MASK == F - 1


def test_global_features_broadcast_to_all_active_rows():
    """Cash, total_inv, sin, cos are the same across all active rows."""
    K = 4
    pids = [f"P{i:04d}" for i in range(K)]
    node = _make_node(pids, capacity=800.0)
    market = _make_market(step=45)

    obs = encode_set_observation(node, market, active_subset=pids, initial_cash=50_000.0)

    for feature_idx in [ROW_CASH, ROW_TOTAL_INV, ROW_SIN, ROW_COS]:
        ref = obs[0, feature_idx]
        for i in range(1, K):
            assert obs[i, feature_idx] == pytest.approx(ref, abs=1e-6), (
                f"Feature {feature_idx} not broadcast: row 0={ref}, row {i}={obs[i, feature_idx]}"
            )


def test_cyclic_encoding_at_known_steps():
    """sin/cos at step=0 → (0.0, 1.0); step=90 → (1.0, 0.0) approx."""
    pids = ["P0001"]
    node = _make_node(pids)

    for step, exp_sin, exp_cos in [
        (0, 0.0, 1.0),
        (90, math.sin(2 * math.pi * 90 / 360), math.cos(2 * math.pi * 90 / 360)),
    ]:
        market = _make_market(step=step)
        obs = encode_set_observation(node, market, active_subset=pids, initial_cash=50_000.0)
        assert obs[0, ROW_SIN] == pytest.approx(exp_sin, abs=1e-6)
        assert obs[0, ROW_COS] == pytest.approx(exp_cos, abs=1e-6)


def test_per_sku_features_differ_across_products():
    """Products with different prices produce different price-ratio features."""
    pids = ["P0001", "P0002"]
    node = _make_node(
        pids,
        list_prices={"P0001": 10.0, "P0002": 20.0},
        base_prices={"P0001": 18.0, "P0002": 18.0},
    )
    market = _make_market()

    obs = encode_set_observation(node, market, active_subset=pids, initial_cash=50_000.0)

    assert obs[0, ROW_PRICE_RATIO] != pytest.approx(obs[1, ROW_PRICE_RATIO])


def test_cash_feature_normalised_by_initial_cash():
    """Cash feature = node.cash / initial_cash."""
    pids = ["P0001"]
    node = _make_node(pids, cash=25_000.0)
    market = _make_market()

    obs = encode_set_observation(node, market, active_subset=pids, initial_cash=50_000.0)

    assert obs[0, ROW_CASH] == pytest.approx(0.5, abs=1e-6)


def test_contention_features_zero_when_no_proposed():
    """Contention features are 0 when no proposed quantities are provided."""
    pids = ["P0001", "P0002"]
    node = _make_node(pids, cash=10_000.0, capacity=200.0)
    market = _make_market()

    obs = encode_set_observation(
        node, market, active_subset=pids, initial_cash=10_000.0
        # no proposed_quantities arg → defaults to zeros
    )
    for i in range(len(pids)):
        assert obs[i, ROW_CONTENTION_QTY] == pytest.approx(0.0)
        assert obs[i, ROW_CONTENTION_COST] == pytest.approx(0.0)


def test_contention_qty_feature_reflects_proposal():
    """Contention qty feature = total_proposed / free_space, clamped at 1.0."""
    pids = ["P0001", "P0002"]
    node = _make_node(pids, inventory={pid: 0 for pid in pids}, cash=10_000.0, capacity=200.0)
    market = _make_market()
    # total free space = 200 units; propose 100 total
    proposed = {"P0001": 60.0, "P0002": 40.0}  # sum = 100
    unit_prices = {"P0001": 1.0, "P0002": 1.0}

    obs = encode_set_observation(
        node, market, active_subset=pids, initial_cash=10_000.0,
        proposed_quantities=proposed, unit_prices=unit_prices,
    )
    # free_space = 200 - 0 - 0 = 200; contention = 100/200 = 0.5
    for i in range(len(pids)):
        assert obs[i, ROW_CONTENTION_QTY] == pytest.approx(0.5, abs=1e-5)


def test_contention_cost_feature_reflects_proposal():
    """Contention cost feature = total_estimated_cost / cash, clamped at 1.0."""
    pids = ["P0001", "P0002"]
    cash = 1000.0
    node = _make_node(pids, cash=cash, capacity=10_000.0)
    market = _make_market()
    proposed = {"P0001": 100.0, "P0002": 100.0}  # 200 units total
    unit_prices = {"P0001": 2.0, "P0002": 3.0}  # cost = 200 + 300 = 500

    obs = encode_set_observation(
        node, market, active_subset=pids, initial_cash=cash,
        proposed_quantities=proposed, unit_prices=unit_prices,
    )
    # cost = 500; cash = 1000; contention = 500/1000 = 0.5
    for i in range(len(pids)):
        assert obs[i, ROW_CONTENTION_COST] == pytest.approx(0.5, abs=1e-5)


def test_contention_features_clamped_at_one():
    """Contention features are clamped to [0, 1] even when demand exceeds resources."""
    pids = ["P0001"]
    node = _make_node(pids, inventory={"P0001": 0}, cash=100.0, capacity=10.0)
    market = _make_market()
    # Propose 1000 units in a 10-unit space → contention_qty = clamped to 1.0
    proposed = {"P0001": 1000.0}
    unit_prices = {"P0001": 5.0}  # cost = 5000 >> cash = 100

    obs = encode_set_observation(
        node, market, active_subset=pids, initial_cash=100.0,
        proposed_quantities=proposed, unit_prices=unit_prices,
    )
    assert obs[0, ROW_CONTENTION_QTY] == pytest.approx(1.0, abs=1e-5)
    assert obs[0, ROW_CONTENTION_COST] == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_encode_deterministic():
    """Same inputs → identical tensor on two calls."""
    K = 5
    pids = [f"P{i:04d}" for i in range(K)]
    node = _make_node(pids)
    market = _make_market(step=42)

    obs1 = encode_set_observation(node, market, active_subset=pids, initial_cash=50_000.0)
    obs2 = encode_set_observation(node, market, active_subset=pids, initial_cash=50_000.0)
    np.testing.assert_array_equal(obs1, obs2)


# ---------------------------------------------------------------------------
# Action decode: mask honoured
# ---------------------------------------------------------------------------


def test_decode_only_emits_active_products():
    """decode_set_action emits orders/prices only for active (non-padded) products."""
    K = 3
    pids = [f"P{i:04d}" for i in range(K)]
    base_prices = {pid: 18.0 for pid in pids}
    node = _make_node(pids, capacity=5000.0, inventory={pid: 0 for pid in pids})
    effective_rate = {pid: 10.0 for pid in pids}

    # All-ones action tensor
    action = np.ones((K_MAX, 3), dtype=np.float32)

    result = decode_set_action(
        action, node, active_subset=pids, base_prices=base_prices,
        effective_rate=effective_rate, supplier_ids=None,
    )

    # Only active pids should appear in the dicts
    assert set(result["list_price"].keys()) == set(pids)
    assert set(result["order"].keys()) <= set(pids)


def test_decode_no_orders_for_padded_slots():
    """Padded rows produce zero orders regardless of action values."""
    K = 2
    pids = ["P0001", "P0002"]
    base_prices = {pid: 18.0 for pid in pids}
    node = _make_node(pids, capacity=5000.0)
    effective_rate = {pid: 10.0 for pid in pids}

    # Set padded slots to +1 (would generate large orders if decoded)
    action = np.zeros((K_MAX, 3), dtype=np.float32)
    action[K:, :] = 1.0  # padded rows all +1

    result = decode_set_action(
        action, node, active_subset=pids, base_prices=base_prices,
        effective_rate=effective_rate,
    )

    # Output keys should only be active pids — no extra keys
    assert set(result["list_price"].keys()) == set(pids)


def test_decode_price_bounds():
    """Decoded prices in [0.5*MSRP, 1.5*MSRP] for any action in [-1, 1]."""
    K = 3
    pids = [f"P{i:04d}" for i in range(K)]
    msrps = {pid: 20.0 for pid in pids}
    node = _make_node(pids, base_prices=msrps, capacity=5000.0)

    rng = np.random.default_rng(42)
    for _ in range(20):
        action = rng.uniform(-1.0, 1.0, size=(K_MAX, 3)).astype(np.float32)
        result = decode_set_action(
            action, node, active_subset=pids, base_prices=msrps,
        )
        for pid in pids:
            p = result["list_price"][pid]
            assert 0.5 * msrps[pid] <= p <= 1.5 * msrps[pid] + 1e-9, (
                f"{pid}: price={p} outside [{0.5*msrps[pid]}, {1.5*msrps[pid]}]"
            )


def test_decode_action_result_keys():
    """decode_set_action returns dict with 'order', 'list_price', 'min_order_imposed'."""
    pids = ["P0001"]
    node = _make_node(pids)
    action = np.zeros((K_MAX, 3), dtype=np.float32)
    result = decode_set_action(action, node, active_subset=pids, base_prices={"P0001": 18.0})
    assert "order" in result
    assert "list_price" in result
    assert "min_order_imposed" in result


# ---------------------------------------------------------------------------
# Round-trip through Arbiter: per-pid dict only for active products
# ---------------------------------------------------------------------------


def test_roundtrip_arbiter_produces_only_active_pids():
    """Encode observation → decode action → Arbiter → result has only active pids."""
    from src.rl.arbiter import allocate

    K = 4
    pids = [f"P{i:04d}" for i in range(K)]
    base_prices = {pid: 18.0 for pid in pids}
    unit_prices = {pid: 5.0 for pid in pids}  # purchase prices
    node = _make_node(
        pids,
        capacity=2000.0,
        inventory={pid: 10 for pid in pids},
        cash=10_000.0,
        base_prices=base_prices,
    )
    market = _make_market(step=10)
    effective_rate = {pid: 5.0 for pid in pids}

    # Encode
    obs = encode_set_observation(
        node, market, active_subset=pids, initial_cash=10_000.0
    )
    assert obs.shape == (K_MAX, F)

    # Use a uniform action tensor
    action = np.zeros((K_MAX, 3), dtype=np.float32)

    # Decode
    result = decode_set_action(
        action, node, active_subset=pids, base_prices=base_prices,
        effective_rate=effective_rate, supplier_ids=["F_P0001"],
    )

    # The decoded result has per-pid proposals
    order_dict = result["order"]
    proposed = {
        pid: sum(qty for _, qty in lines)
        for pid, lines in order_dict.items()
        if lines
    }

    # Run through Arbiter
    per_sku_headroom = {pid: 1000 for pid in pids}
    allocations = allocate(
        proposed={pid: float(sum(qty for _, qty in lines)) for pid, lines in order_dict.items()},
        per_sku_headroom=per_sku_headroom,
        global_free_space=1000,
        cash_budget=10_000.0,
        unit_prices=unit_prices,
    )

    # Only active pids should appear
    assert set(allocations.keys()) <= set(pids), (
        f"Arbiter output contains unexpected pids: {set(allocations.keys()) - set(pids)}"
    )
    # No padded pids sneak through
    for i in range(K, K_MAX):
        padded_pid = f"P{i:04d}"
        assert padded_pid not in allocations


def test_roundtrip_arbiter_zero_for_padded():
    """After encode → decode → Arbiter, padded product slots are not allocated."""
    from src.rl.arbiter import allocate

    # K=2 active, K_MAX-2 padded
    pids = ["P0001", "P0002"]
    base_prices = {pid: 18.0 for pid in pids}
    node = _make_node(
        pids,
        capacity=1000.0,
        inventory={pid: 0 for pid in pids},
        cash=5_000.0,
        base_prices=base_prices,
    )

    action = np.ones((K_MAX, 3), dtype=np.float32)  # max orders

    result = decode_set_action(
        action, node, active_subset=pids, base_prices=base_prices,
        effective_rate={pid: 10.0 for pid in pids},
    )

    # propose only from active pids
    proposals = {
        pid: float(sum(qty for _, qty in lines))
        for pid, lines in result["order"].items()
    }
    assert set(proposals.keys()) == set(pids)

    allocations = allocate(
        proposed=proposals,
        per_sku_headroom={pid: 1000 for pid in pids},
        global_free_space=900,
        cash_budget=5_000.0,
        unit_prices={pid: 5.0 for pid in pids},
    )
    # All pids in allocations should be active
    assert set(allocations.keys()).issubset(set(pids))


# ---------------------------------------------------------------------------
# OBS_LAYOUT_VERSION single source of truth
# ---------------------------------------------------------------------------


def test_layout_version_is_single_source():
    """OBS_LAYOUT_VERSION is defined exactly in set_encoder and is an integer."""
    import src.rl.set_encoder as mod
    assert hasattr(mod, "OBS_LAYOUT_VERSION")
    assert isinstance(mod.OBS_LAYOUT_VERSION, int)
    assert mod.OBS_LAYOUT_VERSION >= 1


def test_layout_version_exported_in_all():
    """OBS_LAYOUT_VERSION appears in __all__ if the module defines __all__."""
    import src.rl.set_encoder as mod
    if hasattr(mod, "__all__"):
        assert "OBS_LAYOUT_VERSION" in mod.__all__


# ---------------------------------------------------------------------------
# Existing flat codec untouched (regression guard)
# ---------------------------------------------------------------------------


def test_old_encoder_still_importable():
    """The old encoders module still exports encode_observation."""
    from src.rl.encoders import encode_observation, decode_action, N_PER_SKU
    assert N_PER_SKU == 18


def test_old_encoder_still_works():
    """encode_observation from old module still returns correct shape."""
    from src.rl.encoders import encode_observation, N_PER_SKU, N_GLOBAL, observation_dim

    pids = ["P0001", "P0002"]
    store = SimpleNamespace(
        active_items=pids,
        inventory={p: 10 for p in pids},
        prices={p: 15.0 for p in pids},
        base_prices={p: 18.0 for p in pids},
        costs={p: 6.0 for p in pids},
        pending={p: 0 for p in pids},
        capacity=300.0,
        balance=20_000.0,
        activation_tick={p: 0 for p in pids},
    )
    market = SimpleNamespace(
        step=0,
        date=SimpleNamespace(month=3),
        params=SimpleNamespace(season_months={}),
    )

    class _Reg:
        def stage(self, pid):
            return "maturity"
        def seasonality(self, pid):
            return "all_season"

    obs = encode_observation(store, market, _Reg(), step=0, slot_perm=[0, 1], K_active=2)
    assert obs.shape == (observation_dim(2),)
    assert obs.dtype == np.float32
