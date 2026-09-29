"""Synthetic transition log.

Seams, from the issue acceptance criteria:

1. Filled quantity. The recorded action is the quantity Allocation filled.
   The request is its own column. Offer availability, buyer cash, and buyer
   capacity each make the two columns differ.
2. Arrival-age profile. In-transit units are a vector indexed by age in
   ticks, length ``delivery_lag``, not a summed scalar. Age 0 is empty at
   decision time.
3. Manifest. Writing the parquet also writes a sidecar with the episode
   parameters, catalog identity, behaviour mixture, realised non-zero-order
   fraction, and git SHA.
4. Mix and shape. Half the episodes are published-default ``OrderUpToPolicy``
   and half are ``ExploratoryPolicy``. Defaults are 365 ticks, K sampled
   from 1 to 30, a sampled nominal delivery lag, ``holding_rate`` 0.01, the tuner order fee, 64 episodes of
   each behavior, seed offset ``14_000_000``. Factories stay on
   ``StaticFactoryPolicy``. The package does not import the RL stack.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.jepa.log import (
    EPISODES_EACH,
    LOG_SEED_OFFSET,
    build_log_overrides,
    build_log_scenario,
    catalog_identity,
    collect_episode_transitions,
    collect_transition_log,
    collection_config,
)
from src.jepa.policy import ExploratoryPolicy
from src.sim.policy import IntermediatePolicy, OrderUpToPolicy, StaticFactoryPolicy
from src.tuning.config import TuningConfig


def _as_window(value) -> list[int]:
    return [int(x) for x in list(value)]


def _as_profile(value) -> list[int]:
    return [int(x) for x in list(value)]


class _FixedRequest(IntermediatePolicy):
    """Request the same quantity from the first direct supplier, with no self-clamp."""

    def __init__(self, qty: int) -> None:
        super().__init__(policy_seed=0)
        self.qty = int(qty)

    def decide(self, obs_intermediate, central_table):
        del central_table
        inventory = dict(obs_intermediate.get("inventory") or {})
        suppliers = list(obs_intermediate.get("direct_supplier_ids") or [])
        supplier = suppliers[0]
        return {
            "order": {pid: [(supplier, self.qty)] for pid in inventory},
            "list_price": dict(obs_intermediate.get("list_prices") or {}),
            "min_order_imposed": {pid: 0 for pid in inventory},
            "promotions": {},
        }


def _clamp_scenario(
    *,
    factory_on_hand: int,
    factory_price: float,
    shop_cash: float,
    shop_capacity: int,
    shop_on_hand: int,
    n_steps: int = 2,
    lead_time: int = 2,
):
    """One factory, shop ``S``, and a sink that buys nothing.

    Opening factory stock is the offer. The sink's demand is zero so the
    shop's on-hand and cash are unchanged when Allocation runs.
    """
    from datetime import datetime

    from src.sim.distributions import Constant
    from src.sim.episode_sampler import (
        default_disruption_params,
        default_lifecycle_params,
        default_market_params,
    )
    from src.sim.graph import EdgeSpec
    from src.sim.node import DemandSinkNode, FactoryNode, IntermediateNode
    from src.sim.scenario import NodeInstance, Scenario, load_catalog

    catalog = load_catalog(
        [
            {
                "name": "Widget",
                "category": "test",
                "related_products": [],
                "base_price": 10.0,
                "unit_cost": factory_price,
                "seasonality": "all_season",
            }
        ]
    )
    pid = catalog[0].product_id
    factory = FactoryNode(
        id="F",
        region="US",
        init_seed=1,
        produces_product_id=pid,
        unit_cost=factory_price,
        capacity_per_tick=0,
        inventory=factory_on_hand,
        list_price=factory_price,
        cash=0.0,
    )
    shop = IntermediateNode(
        id="S",
        region="US",
        init_seed=2,
        carried_products={pid},
        capacity=shop_capacity,
        tags=["shop"],
        inventory={pid: shop_on_hand},
        pending={},
        list_prices={pid: 10.0},
        min_order_imposed={pid: 0},
        cash=shop_cash,
    )
    sink = DemandSinkNode(
        id="D",
        region="US",
        init_seed=3,
        product_id=pid,
        demand_dist=Constant(0),
        income_rate=0.0,
        cash=0.0,
    )
    return Scenario(
        catalog=catalog,
        market=default_market_params(),
        disruption=default_disruption_params(),
        item_lifecycle=default_lifecycle_params(),
        n_steps=n_steps,
        start_date=datetime(2024, 1, 1),
        world_seed=1,
        nodes=[
            NodeInstance(node=factory, init_seed=1),
            NodeInstance(node=shop, init_seed=2),
            NodeInstance(node=sink, init_seed=3),
        ],
        edges=[
            EdgeSpec(supplier_id="F", buyer_id="S", default_lead_time=lead_time),
            EdgeSpec(supplier_id="S", buyer_id="D", default_lead_time=1),
        ],
    )


def _filled_on_the_only_tick(scenario) -> tuple[int, int]:
    frame = collect_episode_transitions(
        scenario,
        _FixedRequest(20),
        behavior="over_order",
        episode_seed=1,
        delivery_lag=2,
    )
    assert len(frame) == 1
    row = frame.iloc[0]
    return int(row["requested_qty"]), int(row["filled_qty"])


def test_filled_quantity_differs_from_the_request_when_allocation_clamps():
    """Allocation, not the emitted action, is the recorded action.

    The request is 20. Each world binds a different ``execute_buy`` clamp,
    in ADR 0012 order: offer availability, then buyer cash, then buyer capacity.
    """
    requested, filled = _filled_on_the_only_tick(
        _clamp_scenario(
            factory_on_hand=4,
            factory_price=1.0,
            shop_cash=1_000_000.0,
            shop_capacity=10_000,
            shop_on_hand=0,
        )
    )
    assert requested == 20
    assert filled == 4

    requested, filled = _filled_on_the_only_tick(
        _clamp_scenario(
            factory_on_hand=1_000,
            factory_price=10.0,
            shop_cash=25.0,
            shop_capacity=10_000,
            shop_on_hand=0,
        )
    )
    assert requested == 20
    assert filled == 2

    requested, filled = _filled_on_the_only_tick(
        _clamp_scenario(
            factory_on_hand=1_000,
            factory_price=1.0,
            shop_cash=1_000_000.0,
            shop_capacity=5,
            shop_on_hand=2,
        )
    )
    assert requested == 20
    assert filled == 3


def test_in_transit_is_an_arrival_age_profile():
    """Index ``age`` is units filled that many ticks ago and still in transit.

    Decide runs before Allocation, so age 0 is empty. With lead time 3, an
    order filled at tick t is in the profile at ages 1 and 2, and has arrived
    before the observation at tick t + 3. A summed scalar cannot tell tick 2
    (everything arrives in two ticks) from a pipeline whose units land sooner.
    """
    request = 7
    lag = 3
    scenario = _clamp_scenario(
        factory_on_hand=1_000,
        factory_price=1.0,
        shop_cash=1_000_000.0,
        shop_capacity=10_000,
        shop_on_hand=0,
        n_steps=5,
        lead_time=lag,
    )
    frame = collect_episode_transitions(
        scenario,
        _FixedRequest(request),
        behavior="over_order",
        episode_seed=1,
        delivery_lag=lag,
    )
    profiles = {
        int(row.tick): _as_profile(row.in_transit) for row in frame.itertuples(index=False)
    }
    nxt = {
        int(row.tick): _as_profile(row.next_in_transit) for row in frame.itertuples(index=False)
    }
    assert profiles == {
        1: [0, 0, 0],
        2: [0, request, 0],
        3: [0, request, request],
        4: [0, request, request],
    }
    assert nxt[1] == profiles[2]
    assert nxt[3] == [0, request, request]
    assert sum(profiles[2]) == request
    assert profiles[2] != [request]


def test_smoke_log_is_half_order_up_to_and_half_exploratory():
    lag = 3
    df = collect_transition_log(
        n_episodes_each=1,
        episode_length=3,
        k_active=2,
        delivery_lag=lag,
        seed_offset=14_000_000,
    )

    assert set(df["behavior"]) == {"order_up_to", "exploratory"}
    counts = df.groupby("behavior")["episode_seed"].nunique()
    assert int(counts["order_up_to"]) == 1
    assert int(counts["exploratory"]) == 1
    assert set(df.loc[df["behavior"] == "order_up_to", "episode_seed"]) == {14_000_000}
    assert set(df.loc[df["behavior"] == "exploratory", "episode_seed"]) == {14_000_001}

    required = {
        "episode_seed",
        "behavior",
        "tick",
        "product_id",
        "on_hand",
        "in_transit",
        "sales_window",
        "cash",
        "capacity",
        "requested_qty",
        "filled_qty",
        "next_on_hand",
        "next_in_transit",
        "next_sales",
    }
    assert required <= set(df.columns)
    assert "market" not in df.columns
    assert not any("disruption" in column for column in df.columns)

    # Last tick has no next observation, so each episode contributes
    # (length - 1) ticks × K rows.
    assert len(df) == 2 * 2 * (3 - 1)
    assert df.groupby("episode_seed")["product_id"].nunique().eq(2).all()

    for (_, _), group in df.groupby(["episode_seed", "product_id"]):
        group = group.sort_values("tick")
        windows = [_as_window(value) for value in group["sales_window"]]
        assert all(len(window) == lag for window in windows)
        on_hand = [int(v) for v in group["on_hand"]]
        nxt_on_hand = [int(v) for v in group["next_on_hand"]]
        profiles = [_as_profile(value) for value in group["in_transit"]]
        nxt_transit = [_as_profile(value) for value in group["next_in_transit"]]
        nxt_sales = [int(v) for v in group["next_sales"]]
        assert all(len(profile) == lag for profile in profiles)
        assert nxt_on_hand[:-1] == on_hand[1:]
        assert nxt_transit[:-1] == profiles[1:]
        for index in range(len(group) - 1):
            assert windows[index + 1] == (windows[index] + [nxt_sales[index]])[-lag:]

    for (_, _), group in df.groupby(["episode_seed", "tick"]):
        assert group["cash"].nunique() == 1
        assert group["capacity"].nunique() == 1
        transit = sum(sum(_as_profile(value)) for value in group["in_transit"])
        free = int(group["capacity"].iloc[0]) - int(group["on_hand"].sum()) - transit
        assert int(group["filled_qty"].sum()) <= max(0, free)
        assert (group["filled_qty"] >= 0).all()
        assert (group["filled_qty"] <= group["requested_qty"]).all()


def test_published_collector_matches_the_tuner_episode_shape():
    cfg = collection_config()
    plain = TuningConfig()
    assert cfg.episode_length == 365 == plain.episode_length
    assert cfg.K_active is None and plain.K_active is None
    assert cfg.K_min == 1 == plain.K_min
    assert cfg.K_max_episode == 30 == plain.K_max_episode
    assert cfg.delivery_lag is None and plain.delivery_lag is None
    assert cfg.holding_rate == 0.01 == plain.holding_rate
    assert cfg.order_fee == plain.order_fee
    assert EPISODES_EACH == 64
    assert LOG_SEED_OFFSET == 14_000_000
    assert LOG_SEED_OFFSET > plain.holdout_seed_offset
    assert LOG_SEED_OFFSET >= plain.seed_offset + plain.n_search_seeds
    assert LOG_SEED_OFFSET >= plain.holdout_seed_offset + plain.n_holdout_seeds

    sizes = {
        len(build_log_scenario(cfg, LOG_SEED_OFFSET + i).active_subset)
        for i in range(24)
    }
    assert min(sizes) >= cfg.K_min
    assert max(sizes) <= cfg.K_max_episode
    assert min(sizes) < max(sizes)
    nominal = {
        build_log_scenario(cfg, LOG_SEED_OFFSET + i).delivery_lag
        for i in range(24)
    }
    assert min(nominal) >= 2
    assert max(nominal) <= 6
    assert min(nominal) < max(nominal)
    spec = build_log_scenario(cfg, LOG_SEED_OFFSET)
    shop = next(instance.node for instance in spec.scenario.nodes if instance.node.id == "S")
    assert shop.holding_rate == 0.01
    assert shop.order_fee == plain.order_fee
    assert spec.scenario.n_steps == 365
    assert cfg.K_min <= len(spec.active_subset) <= cfg.K_max_episode
    pid = spec.active_subset[0]
    lags = [
        edge.default_lead_time
        for edge in spec.scenario.edges
        if edge.buyer_id == "S" and edge.supplier_id.startswith(f"F_{pid}_")
    ]
    assert len(lags) >= 2
    assert len(set(lags)) == len(lags)
    assert spec.scenario.world_seed != LOG_SEED_OFFSET


def test_factories_stay_static_and_order_up_to_keeps_published_defaults():
    spec = build_log_scenario(
        collection_config(episode_length=2, k_active=2),
        episode_seed=1,
    )
    exploratory = build_log_overrides(
        spec.scenario, behavior="exploratory", policy_seed=1, delivery_lag=3
    )
    order_up_to = build_log_overrides(
        spec.scenario, behavior="order_up_to", policy_seed=1, delivery_lag=3
    )
    from src.sim.node import FactoryNode

    factories = [
        instance.node
        for instance in spec.scenario.nodes
        if isinstance(instance.node, FactoryNode)
    ]
    by_pid: dict[str, list[str]] = {}
    for node in factories:
        by_pid.setdefault(node.produces_product_id, []).append(node.id)
    for pid in spec.active_subset:
        assert len(by_pid[pid]) >= 2
        for factory_id in by_pid[pid]:
            assert isinstance(exploratory[factory_id], StaticFactoryPolicy)
            assert isinstance(order_up_to[factory_id], StaticFactoryPolicy)
    assert isinstance(exploratory["S"], ExploratoryPolicy)
    assert type(order_up_to["S"]) is OrderUpToPolicy
    bare = OrderUpToPolicy()
    got = order_up_to["S"]
    assert got.cover_horizon_ticks == bare.cover_horizon_ticks
    assert got.safety_lead_pct_of_lag == bare.safety_lead_pct_of_lag
    assert got.min_qty == bare.min_qty == 0
    assert got.per_supplier_min_order_floor == bare.per_supplier_min_order_floor == 0
    assert got.delivery_lag == bare.delivery_lag


def test_each_half_calls_its_own_policy(monkeypatch):
    calls = {"exploratory": 0, "order_up_to": 0}
    exploratory = ExploratoryPolicy.decide
    order_up_to = OrderUpToPolicy.decide

    def exploratory_decide(self, obs, table):
        calls["exploratory"] += 1
        return exploratory(self, obs, table)

    def order_up_to_decide(self, *args, **kwargs):
        calls["order_up_to"] += 1
        return order_up_to(self, *args, **kwargs)

    monkeypatch.setattr(ExploratoryPolicy, "decide", exploratory_decide)
    monkeypatch.setattr(OrderUpToPolicy, "decide", order_up_to_decide)
    collect_transition_log(
        n_episodes_each=1,
        episode_length=2,
        k_active=2,
        seed_offset=14_000_000,
    )
    assert calls["exploratory"] == 2
    assert calls["order_up_to"] == 2


def test_collector_builds_a_static_factory_per_factory_node(monkeypatch):
    from src.sim.node import FactoryNode

    spec = build_log_scenario(
        collection_config(episode_length=2, k_active=2, delivery_lag=3),
        episode_seed=1,
    )
    n_factories = sum(
        1
        for instance in spec.scenario.nodes
        if isinstance(instance.node, FactoryNode)
    )
    built: list[int] = []
    original = StaticFactoryPolicy.__init__

    def wrapped(self, *args, **kwargs):
        built.append(int(kwargs["capacity_per_tick"]))
        original(self, *args, **kwargs)

    monkeypatch.setattr(StaticFactoryPolicy, "__init__", wrapped)
    collect_transition_log(
        n_episodes_each=1,
        episode_length=2,
        k_active=2,
        delivery_lag=3,
        seed_offset=14_000_000,
    )
    assert n_factories >= 2
    assert len(built) == n_factories * 2


def test_collecting_twice_repeats_the_log():
    kwargs = dict(
        n_episodes_each=1,
        episode_length=3,
        k_active=2,
        seed_offset=14_000_000,
    )
    first = collect_transition_log(**kwargs)
    second = collect_transition_log(**kwargs)
    assert first.equals(second)


def test_collector_writes_parquet(tmp_path):
    path = tmp_path / "log" / "transitions.parquet"
    frame = collect_transition_log(
        n_episodes_each=1,
        episode_length=2,
        k_active=2,
        delivery_lag=3,
        seed_offset=14_000_000,
        path=path,
    )
    assert path.is_file()
    loaded = pd.read_parquet(path)
    assert len(loaded) == len(frame)
    assert set(loaded["behavior"]) == {"order_up_to", "exploratory"}
    assert len(_as_profile(loaded["in_transit"].iloc[0])) == 3
    assert "filled_qty" in loaded.columns
    assert "requested_qty" in loaded.columns


def test_collector_writes_a_manifest_beside_the_log(tmp_path):
    from src.tuning.episode import make_synthetic_catalog

    path = tmp_path / "log" / "transitions.parquet"
    frame = collect_transition_log(
        n_episodes_each=1,
        episode_length=2,
        k_active=2,
        delivery_lag=3,
        seed_offset=14_000_000,
        path=path,
    )
    sidecar = path.with_suffix(".manifest.json")
    manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    config = collection_config(episode_length=2, k_active=2, delivery_lag=3)
    catalog = make_synthetic_catalog(config.active_catalog_size())
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    nonzero = float((frame["filled_qty"] > 0).mean()) if len(frame) else 0.0

    assert manifest["episode_count"] == 2
    assert manifest["seed_min"] == 14_000_000
    assert manifest["seed_max"] == 14_000_001
    assert manifest["episode_length"] == 2
    assert manifest["K_active"] == 2
    assert manifest["delivery_lag"] == 3
    assert manifest["holding_rate"] == config.holding_rate
    assert manifest["order_fee"] == config.order_fee
    assert manifest["catalog_identity"] == catalog_identity(catalog)
    assert manifest["behavior_mixture"] == [
        {"behavior": "order_up_to", "episodes": 1, "policy": "OrderUpToPolicy"},
        {"behavior": "exploratory", "episodes": 1, "policy": "ExploratoryPolicy"},
    ]
    assert manifest["git_sha"] == sha
    assert manifest["nonzero_order_fraction"] == nonzero


def test_log_package_does_not_import_the_rl_stack():
    for path in Path("src/jepa").glob("*.py"):
        text = path.read_text()
        assert "src.rl" not in text
        assert "optuna" not in text

    # Collection stays on OrderUpTo and the exploratory policy. The comparison
    # entry point is the module that scores the other textbook policies.
    collected = (Path("src/jepa") / "log.py").read_text() + (Path("src/jepa") / "policy.py").read_text()
    assert "Periodic" not in collected
    assert "ReorderPoint" not in collected

    script = (
        "import src.jepa, sys\n"
        "bad = [name for name in sys.modules if name == 'src.rl' or name.startswith('src.rl.')]\n"
        "assert not bad, bad\n"
    )
    subprocess.check_call([sys.executable, "-c", script])
