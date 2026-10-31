"""Run isolation: repeated runs from one loaded Scenario must not share policy state.

Regression for the policy-state leak on multi-shop setups: ``build_world``
deep-copied nodes but re-attached the Scenario's own ``NodeInstance.policy``
objects, so a stateful policy on a non-overridden node (``shop-2``) carried
history from one run into the next.  Same seed + knobs must give identical
results regardless of what ran earlier in the process.
"""

from __future__ import annotations

from pathlib import Path

from src.sim.policy_registry import build_policy
from src.sim.runner import Runner
from src.sim.setup_io import _derive_node_seed, load_setup

_TWO_SHOPS = Path(__file__).resolve().parents[2] / "setups" / "two_factories_two_shops"


def _run_family_on_shop1(scenario, name: str, params: dict) -> dict:
    shop1 = next(ni.node for ni in scenario.nodes if ni.node.id == "shop-1")
    policy = build_policy(
        name,
        params,
        node=shop1,
        edges=scenario.edges,
        policy_seed=_derive_node_seed(scenario.world_seed, "shop-1", "policy"),
    )
    return Runner(scenario, policy_overrides={"shop-1": policy}).run()


def test_family_result_independent_of_earlier_runs_on_same_scenario():
    family_a = ("periodic_order_up_to", {"cover_horizon_ticks": 14, "review_interval": 5})
    family_b = ("order_up_to", {"cover_horizon_ticks": 14})

    alone = _run_family_on_shop1(load_setup(_TWO_SHOPS), *family_a)

    shared = load_setup(_TWO_SHOPS)
    _run_family_on_shop1(shared, *family_b)
    after_b = _run_family_on_shop1(shared, *family_a)

    assert after_b["global"] == alone["global"]
    assert after_b["ticks"] == alone["ticks"]


def test_repeated_runner_on_same_scenario_is_identical():
    scenario = load_setup(_TWO_SHOPS)
    first = Runner(scenario).run()
    second = Runner(scenario).run()

    assert second["ticks"] == first["ticks"]
