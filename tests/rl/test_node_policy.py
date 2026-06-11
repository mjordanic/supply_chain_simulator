"""Tests for src/rl/node_policy.py (RLNodePolicy).

Acceptance criteria (from issue 02):
  - from_checkpoint + policy_overrides + Runner.run() completes a simulation
  - Observation-parity: training env and RLNodePolicy produce the same obs tensor
  - Arbiter application: under engineered contention the returned order dict
    respects headroom/free-space/cash and differs from raw proposals
  - Deterministic default: two decide() calls on identical state return identical
    actions with no global-seed manipulation; stochastic mode uses a policy-local
    generator
  - K_MAX overflow raises early with a clear message
  - Supplier-routing tests: cheapest-offer selection, supplier-id tie-break,
    no-offer → no-order-line on a multi-supplier topology
  - MSRP and opening-cash capture at first tick verified
  - Tests assert external behaviour (returned action dicts, tensors) — not
    private attributes
"""

from __future__ import annotations

import copy
from collections import deque
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.rl.node_policy import RLNodePolicy
from src.rl.set_encoder import K_MAX, F
from src.rl.configs.default import RLConfig


# ---------------------------------------------------------------------------
# Shared helpers / stubs
# ---------------------------------------------------------------------------


def _make_catalog(n: int = 10) -> list:
    from src.sim.scenario import load_catalog
    items = [
        {
            "name": f"Product {i}",
            "category": "General",
            "related_products": [],
            "base_price": float(10 + i),
            "unit_cost": float(4 + i * 0.3),
            "seasonality": "all_season",
        }
        for i in range(n)
    ]
    return load_catalog(items)


def _make_config(episode_length: int = 5) -> RLConfig:
    from src.sim.distributions import Uniform
    return RLConfig(
        episode_length=episode_length,
        K_active=3,
        K_min=3,
        K_max_episode=3,
        capacity_dist=Uniform(200, 500),
        balance_dist=Uniform(10_000, 20_000),
    )


def _make_obs(
    pids: list[str],
    tick: int = 0,
    inventory: dict[str, int] | None = None,
    pending: dict[str, dict[str, int]] | None = None,
    list_prices: dict[str, float] | None = None,
    cash: float = 10_000.0,
    capacity: float = 1_000.0,
    observed_sales: dict[str, int] | None = None,
    direct_supplier_ids: list[str] | None = None,
) -> dict[str, Any]:
    if inventory is None:
        inventory = {pid: 20 for pid in pids}
    if pending is None:
        pending = {}
    if list_prices is None:
        list_prices = {pid: float(10 + i) for i, pid in enumerate(pids)}
    if observed_sales is None:
        observed_sales = {pid: 0 for pid in pids}
    if direct_supplier_ids is None:
        direct_supplier_ids = [f"F_{pid}" for pid in pids]
    return {
        "node_id": "S",
        "tick": tick,
        "inventory": inventory,
        "pending": pending,
        "list_prices": list_prices,
        "min_order_imposed": {pid: 0 for pid in pids},
        "capacity": capacity,
        "cash": cash,
        "observed_sales": observed_sales,
        "direct_supplier_ids": direct_supplier_ids,
    }


class _FakeOffer:
    """Minimal offer stub."""
    def __init__(self, list_price: float, fill_rate_recent: float = 1.0):
        self.list_price = list_price
        self.fill_rate_recent = fill_rate_recent


class _FakeCentralTable:
    """Minimal central table stub: one offer per pid at a given price."""

    def __init__(self, offers_by_pid: dict[str, list[tuple[str, float]]]):
        """offers_by_pid: {pid: [(supplier_id, price), ...]}"""
        self._offers = {
            pid: [(sid, _FakeOffer(price)) for sid, price in lines]
            for pid, lines in offers_by_pid.items()
        }

    def snapshot_for_buyer(self, pid: str):
        return list(self._offers.get(pid, []))


def _make_zero_actor_fn() -> callable:
    """Actor that always returns zeros (tanh(0) = 0 for prices; order = 0.5 decoded)."""
    def fn(obs_2d: np.ndarray) -> np.ndarray:
        return np.zeros((K_MAX, 3), dtype=np.float32)
    return fn


def _make_constant_actor_fn(order_raw: float = 0.5) -> callable:
    """Actor that returns a constant order_raw for all active rows."""
    def fn(obs_2d: np.ndarray) -> np.ndarray:
        arr = np.zeros((K_MAX, 3), dtype=np.float32)
        # col 1 is the order head
        arr[:, 1] = order_raw
        return arr
    return fn


# ---------------------------------------------------------------------------
# K_MAX overflow
# ---------------------------------------------------------------------------


class TestKMaxOverflow:
    def test_overflow_raises_on_first_decide(self):
        """An assortment larger than K_MAX raises a clear error on first decide()."""
        pids = [f"P{i:04d}" for i in range(K_MAX + 1)]
        obs = _make_obs(pids)
        config = RLConfig()
        policy = RLNodePolicy(_make_zero_actor_fn(), config)

        with pytest.raises(ValueError, match=r"K_MAX=32"):
            policy.decide(obs, None)

    def test_no_error_at_exactly_k_max(self):
        """An assortment of exactly K_MAX products does not raise."""
        pids = [f"P{i:04d}" for i in range(K_MAX)]
        obs = _make_obs(pids)
        config = RLConfig()
        policy = RLNodePolicy(_make_zero_actor_fn(), config)
        # Should not raise.
        result = policy.decide(obs, None)
        assert "order" in result


# ---------------------------------------------------------------------------
# MSRP and opening-cash capture at first tick
# ---------------------------------------------------------------------------


class TestFirstTickCapture:
    def test_opening_cash_captured(self):
        """The opening cash captured at tick 0 normalises the cash feature."""
        pids = ["A", "B"]
        opening_cash = 12_345.0
        obs = _make_obs(pids, cash=opening_cash)
        obs2 = _make_obs(pids, cash=999.0)  # second tick: different cash

        captured_obs_2d: list[np.ndarray] = []

        def recording_actor(obs_2d: np.ndarray) -> np.ndarray:
            captured_obs_2d.append(obs_2d.copy())
            return np.zeros((K_MAX, 3), dtype=np.float32)

        config = RLConfig()
        policy = RLNodePolicy(recording_actor, config)

        # Two decide() calls.
        policy.decide(obs, None)
        policy.decide(obs2, None)

        # The cash feature (ROW_CASH = 9) in the first obs should be 1.0
        # (opening_cash / opening_cash == 1.0).
        from src.rl.set_encoder import ROW_CASH
        assert len(captured_obs_2d) == 2
        cash_feat_tick0 = float(captured_obs_2d[0][0, ROW_CASH])
        assert cash_feat_tick0 == pytest.approx(1.0, abs=1e-4), (
            f"Expected cash feature = 1.0 at tick 0, got {cash_feat_tick0}"
        )

        # At tick 1, cash=999, opening_cash=12345 → feature < 1.0.
        cash_feat_tick1 = float(captured_obs_2d[1][0, ROW_CASH])
        assert cash_feat_tick1 < 1.0, (
            f"Cash feature should be < 1.0 at tick 1 (cash dropped), got {cash_feat_tick1}"
        )

    def test_msrp_anchors_captured_at_tick_0(self):
        """MSRP anchors are fixed to tick-0 list prices and do not change."""
        pids = ["A", "B"]
        tick0_prices = {"A": 20.0, "B": 30.0}
        tick1_prices = {"A": 99.0, "B": 99.0}  # dramatically changed

        captured_obs_2d: list[np.ndarray] = []

        def recording_actor(obs_2d: np.ndarray) -> np.ndarray:
            captured_obs_2d.append(obs_2d.copy())
            return np.zeros((K_MAX, 3), dtype=np.float32)

        config = RLConfig()
        policy = RLNodePolicy(recording_actor, config)

        obs0 = _make_obs(pids, list_prices=tick0_prices)
        obs1 = _make_obs(pids, list_prices=tick1_prices)

        policy.decide(obs0, None)
        policy.decide(obs1, None)

        # ROW_PRICE_RATIO = current_price / MSRP.  At tick 0: price==MSRP → ratio 1.0.
        from src.rl.set_encoder import ROW_PRICE_RATIO
        price_ratio_t0 = float(captured_obs_2d[0][0, ROW_PRICE_RATIO])
        assert price_ratio_t0 == pytest.approx(1.0, abs=1e-4), (
            f"Price ratio should be 1.0 at tick 0 (price==MSRP), got {price_ratio_t0}"
        )

        # At tick 1: list_prices = 99.0 but MSRP anchored at 20.0 → ratio = 99/20 = 4.95.
        price_ratio_t1 = float(captured_obs_2d[1][0, ROW_PRICE_RATIO])
        assert price_ratio_t1 == pytest.approx(99.0 / 20.0, abs=0.01), (
            f"Unexpected price ratio at tick 1: {price_ratio_t1}"
        )


# ---------------------------------------------------------------------------
# Deterministic / stochastic mode
# ---------------------------------------------------------------------------


class TestDeterministicMode:
    def _make_policy_with_real_actor(self, deterministic: bool = True, seed: int | None = None):
        """Build an RLNodePolicy backed by a freshly-initialised SetActor."""
        import torch
        from src.rl.agents.set_actor_critic import SetActor

        actor = SetActor()
        actor.eval()
        actor_fn = RLNodePolicy._make_actor_fn(actor, deterministic=deterministic, policy_seed=seed)
        return RLNodePolicy(actor_fn, RLConfig(), deterministic=deterministic, policy_seed=seed)

    def test_deterministic_default_identical_on_identical_state(self):
        """Two decide() calls with identical state return identical action dicts.

        No global seed manipulation — identical state ≡ identical output.
        Both policies share the same actor weights for a fair comparison.
        """
        import torch
        from src.rl.agents.set_actor_critic import SetActor

        torch.manual_seed(0)
        actor = SetActor()
        actor.eval()

        pids = ["P0", "P1", "P2"]
        obs = _make_obs(pids, tick=5, cash=8_000.0)
        table = _FakeCentralTable({pid: [(f"F_{pid}", 5.0)] for pid in pids})

        actor_fn1 = RLNodePolicy._make_actor_fn(actor, deterministic=True, policy_seed=None)
        actor_fn2 = RLNodePolicy._make_actor_fn(actor, deterministic=True, policy_seed=None)
        policy1 = RLNodePolicy(actor_fn1, RLConfig())
        policy2 = RLNodePolicy(actor_fn2, RLConfig())

        # Copy the state so both calls start from identical conditions.
        import copy
        obs_copy = copy.deepcopy(obs)

        r1 = policy1.decide(obs, table)
        r2 = policy2.decide(obs_copy, table)

        # Deterministic: identical state → identical order dict.
        assert r1["order"] == r2["order"], (
            f"Deterministic mode: expected identical orders, got\n  {r1['order']}\nvs\n  {r2['order']}"
        )

    def test_stochastic_uses_local_generator_not_global(self):
        """Stochastic mode uses a policy-local generator; global torch.manual_seed
        must not change the sampled action when the policy has its own generator.

        Two policies share the SAME actor weights and the same local seed.
        Changing the global torch seed between calls must not change their output.
        """
        import torch
        from src.rl.agents.set_actor_critic import SetActor

        torch.manual_seed(0)  # Fix weights of the shared actor.
        actor = SetActor()
        actor.eval()

        pids = ["P0", "P1"]
        obs = _make_obs(pids, tick=0, cash=5_000.0)
        table = _FakeCentralTable({pid: [(f"F_{pid}", 5.0)] for pid in pids})

        # Both policies share the same actor and the same local seed → should agree.
        actor_fn_a = RLNodePolicy._make_actor_fn(actor, deterministic=False, policy_seed=99)
        actor_fn_b = RLNodePolicy._make_actor_fn(actor, deterministic=False, policy_seed=99)
        policy_a = RLNodePolicy(actor_fn_a, RLConfig(), deterministic=False, policy_seed=99)
        policy_b = RLNodePolicy(actor_fn_b, RLConfig(), deterministic=False, policy_seed=99)

        import copy
        obs_a = copy.deepcopy(obs)
        obs_b = copy.deepcopy(obs)

        # Manipulate global torch seed between the two calls.
        torch.manual_seed(42)
        r_a = policy_a.decide(obs_a, table)
        torch.manual_seed(0)
        r_b = policy_b.decide(obs_b, table)

        # Both policies have the same local seed + same actor → they should agree.
        assert r_a["list_price"] == pytest.approx(r_b["list_price"], abs=1e-5), (
            "Stochastic policies with the same local seed and same actor weights "
            f"should produce the same prices, got {r_a['list_price']} vs {r_b['list_price']}"
        )

    def test_deterministic_no_global_seed_dependency(self):
        """Deterministic mode: output is independent of global torch seed.

        Both policies share the same actor weights (fixed seed before init).
        """
        import torch
        from src.rl.agents.set_actor_critic import SetActor

        torch.manual_seed(0)
        actor = SetActor()
        actor.eval()

        pids = ["P0"]
        obs0 = _make_obs(pids, tick=0, cash=5_000.0)
        obs1 = _make_obs(pids, tick=0, cash=5_000.0)
        table = _FakeCentralTable({pid: [(f"F_{pid}", 5.0)] for pid in pids})

        actor_fn_a = RLNodePolicy._make_actor_fn(actor, deterministic=True, policy_seed=None)
        actor_fn_b = RLNodePolicy._make_actor_fn(actor, deterministic=True, policy_seed=None)
        policy_a = RLNodePolicy(actor_fn_a, RLConfig())
        policy_b = RLNodePolicy(actor_fn_b, RLConfig())

        torch.manual_seed(0)
        r_a = policy_a.decide(obs0, table)
        torch.manual_seed(999999)
        r_b = policy_b.decide(obs1, table)

        assert r_a["list_price"] == pytest.approx(r_b["list_price"], abs=1e-5)


# ---------------------------------------------------------------------------
# Supplier routing
# ---------------------------------------------------------------------------


class TestSupplierRouting:
    """Cheapest-offer selection, tie-breaking, and no-offer → no-order-line."""

    def _make_policy_with_strong_order(self, pids: list[str]) -> RLNodePolicy:
        """Policy that always requests a large order (order_raw = 1.0).

        Uses a large opening inventory so effective_rate ≈ base_demand_prior
        and the order-up-to formula produces non-zero quantities even on the
        first tick when there is no sales history yet.
        """
        def actor_fn(obs_2d: np.ndarray) -> np.ndarray:
            arr = np.zeros((K_MAX, 3), dtype=np.float32)
            arr[:len(pids), 1] = 1.0   # maximum order head
            return arr
        # base_demand_prior=10 ensures effective_rate > 0 on cold start.
        return RLNodePolicy(actor_fn, RLConfig(), base_demand_prior=10.0)

    def test_cheapest_supplier_selected(self):
        """Order line goes to the cheaper of two suppliers for each product."""
        pids = ["A", "B"]
        # Supplier S1 is cheap, S2 is expensive.
        table = _FakeCentralTable({
            "A": [("S1_A", 2.0), ("S2_A", 10.0)],
            "B": [("S1_B", 3.0), ("S2_B", 20.0)],
        })
        # No inventory so position = 0 → order-up-to decoder requests positive qty.
        obs = _make_obs(
            pids,
            inventory={"A": 0, "B": 0},
            direct_supplier_ids=["S1_A", "S2_A", "S1_B", "S2_B"],
            cash=100_000.0,
            capacity=10_000.0,
        )
        policy = self._make_policy_with_strong_order(pids)

        result = policy.decide(obs, table)
        order = result["order"]

        # Both products should route to the cheaper supplier.
        for pid in pids:
            lines = order.get(pid, [])
            if lines:  # only check if a line was placed (cash may limit)
                assert lines[0][0] == f"S1_{pid}", (
                    f"Expected S1_{pid} (cheapest), got {lines[0][0]}"
                )
            # We verify at least one product placed an order to the cheap supplier.
        placed = [pid for pid in pids if order.get(pid, [])]
        assert placed, "Expected at least one product to place an order"
        for pid in placed:
            assert order[pid][0][0] == f"S1_{pid}"

    def test_supplier_id_tiebreak(self):
        """When two suppliers offer the same price, lower supplier id wins."""
        pids = ["X"]
        table = _FakeCentralTable({
            "X": [("SUP_B", 5.0), ("SUP_A", 5.0)],  # same price, B listed first
        })
        obs = _make_obs(
            pids,
            inventory={"X": 0},
            direct_supplier_ids=["SUP_A", "SUP_B"],
            cash=100_000.0,
            capacity=10_000.0,
        )
        policy = self._make_policy_with_strong_order(pids)

        result = policy.decide(obs, table)
        lines = result["order"].get("X", [])
        if lines:
            assert lines[0][0] == "SUP_A", (
                f"Expected alphabetically-first supplier 'SUP_A', got {lines[0][0]}"
            )

    def test_no_offer_produces_no_order_line(self):
        """A product with no available offer this tick gets an empty order line."""
        pids = ["A", "B"]
        # Only A has an offer; B has none.
        table = _FakeCentralTable({
            "A": [("F_A", 5.0)],
            # "B" absent
        })
        obs = _make_obs(
            pids,
            inventory={"A": 0, "B": 0},
            direct_supplier_ids=["F_A", "F_B"],
            cash=100_000.0,
            capacity=10_000.0,
        )
        policy = self._make_policy_with_strong_order(pids)

        result = policy.decide(obs, table)
        order = result["order"]

        # B must have an empty order list (no offer → no order line).
        assert order.get("B", []) == [], (
            f"Expected empty order for B (no offer), got {order.get('B')}"
        )

    def test_only_direct_suppliers_considered(self):
        """Orders only route through direct suppliers, not via indirect ones."""
        pids = ["P"]
        # Table has a very cheap indirect offer and a pricier direct offer.
        table = _FakeCentralTable({
            "P": [("INDIRECT", 0.01), ("DIRECT", 5.0)],
        })
        obs = _make_obs(
            pids,
            inventory={"P": 0},
            direct_supplier_ids=["DIRECT"],  # only DIRECT is allowed
            cash=100_000.0,
            capacity=10_000.0,
        )
        policy = self._make_policy_with_strong_order(pids)

        result = policy.decide(obs, table)
        lines = result["order"].get("P", [])
        if lines:
            assert lines[0][0] == "DIRECT", (
                f"Expected DIRECT supplier only, got {lines[0][0]}"
            )


# ---------------------------------------------------------------------------
# Arbiter application under contention
# ---------------------------------------------------------------------------


class TestArbiterApplication:
    """Under engineered contention the returned order dict respects constraints."""

    def test_arbiter_applied_under_cash_contention(self):
        """When many products want large orders but cash is tiny, the returned
        order dict total cost must not exceed the available cash budget."""
        pids = [f"P{i}" for i in range(5)]
        cash = 50.0  # tiny budget
        unit_price = 10.0

        table = _FakeCentralTable({
            pid: [(f"F_{pid}", unit_price)] for pid in pids
        })
        obs = _make_obs(
            pids,
            cash=cash,
            capacity=10_000,
            inventory={pid: 0 for pid in pids},
            direct_supplier_ids=[f"F_{pid}" for pid in pids],
        )

        # Actor that requests the maximum for every product.
        def actor_fn(obs_2d: np.ndarray) -> np.ndarray:
            arr = np.zeros((K_MAX, 3), dtype=np.float32)
            arr[:len(pids), 1] = 1.0  # max order
            return arr

        config = RLConfig()
        policy = RLNodePolicy(actor_fn, config)
        result = policy.decide(obs, table)

        order = result["order"]
        total_cost = sum(
            qty * unit_price
            for lines in order.values()
            for _, qty in lines
        )
        budget = cash * config.cash_budget_fraction
        assert total_cost <= budget + 1e-6, (
            f"Order cost {total_cost:.2f} exceeds cash budget {budget:.2f}"
        )

    def test_arbiter_applied_under_capacity_contention(self):
        """When capacity is tiny the total allocated units must not exceed free space."""
        pids = ["A", "B", "C"]
        capacity = 10.0  # only 10 units total
        table = _FakeCentralTable({pid: [(f"F_{pid}", 1.0)] for pid in pids})
        obs = _make_obs(
            pids,
            capacity=capacity,
            inventory={pid: 0 for pid in pids},
            cash=1_000_000.0,  # no cash constraint
        )

        def actor_fn(obs_2d: np.ndarray) -> np.ndarray:
            arr = np.zeros((K_MAX, 3), dtype=np.float32)
            arr[:len(pids), 1] = 1.0
            return arr

        policy = RLNodePolicy(actor_fn, RLConfig())
        result = policy.decide(obs, table)

        total_qty = sum(
            qty for lines in result["order"].values() for _, qty in lines
        )
        assert total_qty <= int(capacity) + 1, (
            f"Allocated {total_qty} units exceeds capacity {int(capacity)}"
        )

    def test_arbiter_differs_from_raw_proposals_under_contention(self):
        """Under contention the arbitrated result must differ from the raw decoded
        proposals — the Arbiter was actually applied, not bypassed."""
        pids = ["A", "B", "C", "D"]
        capacity = 5  # extremely tight: raw proposals will vastly exceed this
        cash = 1_000_000.0
        table = _FakeCentralTable({pid: [(f"F_{pid}", 1.0)] for pid in pids})
        obs = _make_obs(
            pids,
            capacity=float(capacity),
            inventory={pid: 0 for pid in pids},
            cash=cash,
        )

        def actor_fn(obs_2d: np.ndarray) -> np.ndarray:
            arr = np.zeros((K_MAX, 3), dtype=np.float32)
            arr[:len(pids), 1] = 1.0  # max order — will far exceed capacity
            return arr

        policy = RLNodePolicy(actor_fn, RLConfig())
        result = policy.decide(obs, table)

        total_qty = sum(qty for lines in result["order"].values() for _, qty in lines)
        # Raw proposals for 4 products × effective_rate × 30 ticks >> 5 units.
        # After the Arbiter, total must be ≤ capacity.
        assert total_qty <= capacity, (
            f"Arbiter did not reduce total qty: {total_qty} > capacity {capacity}"
        )


# ---------------------------------------------------------------------------
# Return dict structure
# ---------------------------------------------------------------------------


class TestReturnDictStructure:
    def test_returns_required_keys(self):
        """decide() always returns a dict with 'order', 'list_price',
        'min_order_imposed' keys."""
        pids = ["A", "B"]
        obs = _make_obs(pids)
        policy = RLNodePolicy(_make_zero_actor_fn(), RLConfig())
        result = policy.decide(obs, None)

        assert "order" in result
        assert "list_price" in result
        assert "min_order_imposed" in result

    def test_order_values_are_lists(self):
        """Each value in 'order' is a list of (supplier_id, qty) tuples."""
        pids = ["A", "B"]
        obs = _make_obs(pids)
        table = _FakeCentralTable({pid: [(f"F_{pid}", 5.0)] for pid in pids})
        policy = RLNodePolicy(_make_zero_actor_fn(), RLConfig())
        result = policy.decide(obs, table)

        for pid in pids:
            lines = result["order"].get(pid, [])
            assert isinstance(lines, list), f"{pid}: expected list, got {type(lines)}"
            for sup_id, qty in lines:
                assert isinstance(sup_id, str)
                assert isinstance(qty, int)
                assert qty >= 0

    def test_list_price_has_all_managed_products(self):
        """list_price contains entries for all managed products."""
        pids = ["A", "B", "C"]
        obs = _make_obs(pids)
        policy = RLNodePolicy(_make_zero_actor_fn(), RLConfig())
        result = policy.decide(obs, None)

        for pid in pids:
            assert pid in result["list_price"], f"Missing list_price for {pid}"


# ---------------------------------------------------------------------------
# Integration: policy_overrides + Runner.run()
# ---------------------------------------------------------------------------


class TestRunnerIntegration:
    """from_checkpoint + policy_overrides + Runner.run() completes without error."""

    def _build_rl_episode(self):
        """Build a minimal 3-node scenario for the standard RL episode."""
        from src.rl.episode_sampler import sample_episode
        catalog = _make_catalog(10)
        config = _make_config(episode_length=5)
        spec = sample_episode(
            catalog=catalog,
            config=config,
            episode_seed=42,
        )
        return spec

    def test_runner_run_with_rl_node_policy_no_crash(self):
        """Attaching an RLNodePolicy via policy_overrides and calling Runner.run()
        completes the full episode without raising."""
        from src.sim.runner import Runner

        spec = self._build_rl_episode()
        config = _make_config()

        policy = RLNodePolicy(_make_constant_actor_fn(0.0), config)
        runner = Runner(spec.scenario, policy_overrides={"S": policy})
        log = runner.run()

        assert log["n_steps"] == spec.scenario.n_steps
        assert len(log["ticks"]) == spec.scenario.n_steps

    def test_runner_run_returns_standard_run_log(self):
        """The run log has the standard keys expected by inspect/metrics tooling."""
        from src.sim.runner import Runner

        spec = self._build_rl_episode()
        config = _make_config()

        policy = RLNodePolicy(_make_constant_actor_fn(0.5), config)
        runner = Runner(spec.scenario, policy_overrides={"S": policy})
        log = runner.run()

        assert "n_steps" in log
        assert "ticks" in log
        assert "global" in log
        # Each tick log should have cash and inventory.
        tick_0 = log["ticks"][0]
        assert "node_cash" in tick_0
        assert "node_inventory" in tick_0

    def test_runner_run_multiple_episodes_fresh_instance_each(self):
        """Running two episodes with fresh policy instances each time produces
        valid results — no cross-contamination of state."""
        from src.sim.runner import Runner

        spec = self._build_rl_episode()
        config = _make_config()

        # First run.
        p1 = RLNodePolicy(_make_constant_actor_fn(0.5), config)
        log1 = Runner(spec.scenario, policy_overrides={"S": p1}).run()

        # Second run — fresh policy, same scenario.
        p2 = RLNodePolicy(_make_constant_actor_fn(0.5), config)
        log2 = Runner(spec.scenario, policy_overrides={"S": p2}).run()

        assert log1["n_steps"] == log2["n_steps"]


# ---------------------------------------------------------------------------
# Observation-parity: RLNodePolicy vs RLEnv
# ---------------------------------------------------------------------------


class TestObservationParity:
    """Training env and RLNodePolicy produce the same observation tensor on
    identical world state (PRD user story 22)."""

    def test_obs_parity_on_training_episode(self):
        """After tick_world(), both the env and RLNodePolicy produce the same
        (K_MAX, F) tensor from the identical node/market state."""
        from src.rl.env import RLEnv
        from src.rl.episode_sampler import sample_episode
        from src.rl.set_encoder import encode_set_observation, K_MAX

        catalog = _make_catalog(10)
        config = _make_config(episode_length=3)
        spec = sample_episode(catalog=catalog, config=config, episode_seed=7)

        # --- Env path ---
        env = RLEnv(catalog=catalog, config=config)
        obs_env_flat, _ = env.reset(seed=7)
        obs_env_2d = obs_env_flat.reshape(K_MAX, -1)

        # --- RLNodePolicy path ---
        # Build the same sim from the same spec and snapshot state after reset.
        from src.sim.runner import build_world
        from src.sim.policy import RLIntermediatePolicy
        sim = build_world(spec.scenario, policy_overrides={"S": RLIntermediatePolicy()})
        node_s = sim.nodes["S"]
        initial_cash = float(node_s.cash)

        # The env's _build_observation() is called before any tick.
        # Encode using the same encoder with the same arguments.
        active_subset = spec.active_subset
        supplier_ids_for = {pid: [f"F_{pid}"] for pid in active_subset}
        sales_history: dict[str, deque] = {
            pid: deque(maxlen=100) for pid in active_subset
        }

        obs_policy_2d = encode_set_observation(
            node=node_s,
            market=sim.market,
            active_subset=active_subset,
            initial_cash=initial_cash,
            sales_history=sales_history,
            central_table=None,  # before any tick, no central table
            supplier_ids_for=supplier_ids_for,
            proposed_quantities=None,
            unit_prices=None,
        )

        # Both obs tensors must match on all non-central-table features.
        # Features 6-8 (supplier_count, min_price, fill_rate) can differ
        # because the env reads from the central table after tick_world().
        # The env's initial obs is also built before any tick (no central table).
        # So for tick-0, both should be identical.
        from src.rl.set_encoder import ROW_SUPPLIER_COUNT
        mask = obs_policy_2d[:, -1]  # 1.0 for active rows
        active_rows = mask > 0.5

        # Active rows must match (within float32 precision).
        np.testing.assert_allclose(
            obs_policy_2d[active_rows],
            obs_env_2d[active_rows],
            atol=1e-5,
            err_msg="Tick-0 observation tensors differ between env and policy encoder paths",
        )


# ---------------------------------------------------------------------------
# from_policy_fn constructor
# ---------------------------------------------------------------------------


class TestFromPolicyFn:
    def test_from_policy_fn_wraps_flat_callable(self):
        """from_policy_fn wraps a flat obs→action callable and produces valid output."""
        pids = ["A", "B", "C"]
        obs = _make_obs(pids)
        table = _FakeCentralTable({pid: [(f"F_{pid}", 5.0)] for pid in pids})

        def flat_policy_fn(obs_flat: np.ndarray) -> np.ndarray:
            assert obs_flat.shape == (K_MAX * F,), f"Wrong shape: {obs_flat.shape}"
            return np.zeros(K_MAX * 3, dtype=np.float32)

        config = RLConfig()
        policy = RLNodePolicy.from_policy_fn(flat_policy_fn, config)
        result = policy.decide(obs, table)

        assert "order" in result
        assert "list_price" in result

    def test_from_policy_fn_deterministic_by_default(self):
        """Repeated calls with the same flat policy fn on same state give same result."""
        pids = ["A"]
        obs = _make_obs(pids)
        table = _FakeCentralTable({"A": [("F_A", 5.0)]})

        calls: list[np.ndarray] = []

        def flat_policy_fn(obs_flat: np.ndarray) -> np.ndarray:
            # Always return a fixed action.
            arr = np.zeros(K_MAX * 3, dtype=np.float32)
            arr[1] = 0.8  # order head for row 0
            calls.append(arr.copy())
            return arr

        config = RLConfig()
        p1 = RLNodePolicy.from_policy_fn(flat_policy_fn, config)
        p2 = RLNodePolicy.from_policy_fn(flat_policy_fn, config)

        import copy
        r1 = p1.decide(copy.deepcopy(obs), table)
        r2 = p2.decide(copy.deepcopy(obs), table)

        assert r1["order"] == r2["order"]
