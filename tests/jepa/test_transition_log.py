"""Synthetic transition log.

Seams, from the issue acceptance criteria:

1. Filled quantity. The recorded action is the quantity Allocation filled.
   The request is its own column. Offer availability, buyer cash, and buyer
   capacity each make the two columns differ.
2. Arrival-age profile. In-transit units are a vector indexed by age in
   ticks, width ``ARRIVAL_AGE_SLOTS``, not a summed scalar and not the
   episode's nominal lag. Age 0 is empty at decision time. The sales
   window length is ``SALES_WINDOW``, the encoder's knob.
3. Manifest. Writing the parquet also writes a sidecar with the episode
   parameters, catalog identity, behaviour mixture, realised non-zero-order
   fraction, and git SHA.
4. Mix and shape. Every textbook family is sampled from the tuning search
   spaces, plus exploratory, never-order, and a heavier over-order arm.
   Planning depth varies by episode. Defaults are 365 ticks, K sampled
   from 1 to 30, a sampled nominal delivery lag, one ``holding_rate`` of
   0.01, the tuner order fee, a base of 64 episodes per arm (over-order
   weighted higher), seed offset ``14_000_000``. Factories stay on
   ``StaticFactoryPolicy``. The package does not import the RL stack.
5. Coverage. The manifest's realised non-zero fill is materially above
   13%, and the log visits low-inventory, high-inventory, capacity-bound,
   and cash-bound states.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.jepa.dynamics import ARRIVAL_AGE_SLOTS, SALES_WINDOW, arrival_age_profile
from src.jepa.log import (
    BEHAVIORS,
    COVER_HORIZON_TICKS,
    EPISODES_EACH,
    LOG_SEED_OFFSET,
    behavior_schedule,
    build_log_overrides,
    build_log_scenario,
    catalog_identity,
    collect_episode_transitions,
    collect_transition_log,
    collection_config,
    episodes_for,
    log_buyer,
)
from src.jepa.policy import ExploratoryPolicy, NeverOrderPolicy, OverOrderPolicy
from src.sim.policy import (
    IntermediatePolicy,
    OrderUpToPolicy,
    PeriodicOrderUpToPolicy,
    PeriodicReorderPolicy,
    ReorderPointPolicy,
    StaticFactoryPolicy,
)
from src.tuning.config import TuningConfig
from src.tuning.search_spaces import sample_textbook_policy


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
    def _aged(prefix: list[int]) -> list[int]:
        return prefix + [0] * (ARRIVAL_AGE_SLOTS - len(prefix))

    assert profiles == {
        1: _aged([0, 0, 0]),
        2: _aged([0, request, 0]),
        3: _aged([0, request, request]),
        4: _aged([0, request, request]),
    }
    assert nxt[1] == profiles[2]
    assert nxt[3] == _aged([0, request, request])
    assert sum(profiles[2]) == request
    assert profiles[2] != [request]


def test_arrival_profile_is_not_clipped_to_the_nominal_lag():
    """A slower edge stays visible after the nominal lag has passed.

    The log used to size the profile to the episode's nominal lag, so a
    unit still in transit on a longer edge disappeared from the vector.
    """
    request = 4
    nominal = 2
    lead = 6
    scenario = _clamp_scenario(
        factory_on_hand=1_000,
        factory_price=1.0,
        shop_cash=1_000_000.0,
        shop_capacity=10_000,
        shop_on_hand=0,
        n_steps=6,
        lead_time=lead,
    )
    frame = collect_episode_transitions(
        scenario,
        _FixedRequest(request),
        behavior="over_order",
        episode_seed=1,
        delivery_lag=nominal,
    )
    profiles = {
        int(row.tick): _as_profile(row.in_transit) for row in frame.itertuples(index=False)
    }
    # The tick-1 fill is age 3 at tick 4, past the nominal lag and still short of the edge.
    assert profiles[4][3] == request
    assert len(profiles[4]) == ARRIVAL_AGE_SLOTS
    assert all(len(_as_window(value)) == SALES_WINDOW for value in frame["sales_window"])


def test_decision_time_lots_match_the_logged_arrival_profile():
    """The profile on the observation is the profile written to the log."""
    request = 7
    lag = 3
    recorded: list[tuple[int, dict]] = []

    class _Recording(_FixedRequest):
        def decide(self, obs_intermediate, central_table):
            recorded.append(
                (int(obs_intermediate["tick"]), dict(obs_intermediate.get("in_transit_lots") or {}))
            )
            return super().decide(obs_intermediate, central_table)

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
        _Recording(request),
        behavior="over_order",
        episode_seed=1,
        delivery_lag=lag,
    )
    by_tick = {tick: lots for tick, lots in recorded}
    for row in frame.itertuples(index=False):
        pid = str(row.product_id)
        lots = by_tick[int(row.tick)].get(pid, [])
        assert _as_profile(row.in_transit) == arrival_age_profile(lots, int(row.tick))


def test_smoke_log_covers_every_mixture_arm():
    lag = 3
    df = collect_transition_log(
        n_episodes_each=1,
        episode_length=3,
        k_active=2,
        delivery_lag=lag,
        seed_offset=14_000_000,
    )

    schedule = behavior_schedule(1, 14_000_000)
    assert set(df["behavior"]) == set(BEHAVIORS)
    counts = df.groupby("behavior")["episode_seed"].nunique()
    for behavior, seed in schedule:
        assert int(counts[behavior]) == episodes_for(behavior, 1)
        assert seed in set(df.loc[df["behavior"] == behavior, "episode_seed"])

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
        "delivery_lag",
        "next_on_hand",
        "next_in_transit",
        "next_sales",
    }
    assert required <= set(df.columns)
    assert "market" not in df.columns
    assert not any("disruption" in column for column in df.columns)

    # Last tick has no next observation, so each episode contributes
    # (length - 1) ticks × K rows.
    assert len(df) == len(schedule) * 2 * (3 - 1)
    assert df.groupby("episode_seed")["product_id"].nunique().eq(2).all()

    for (_, _), group in df.groupby(["episode_seed", "product_id"]):
        group = group.sort_values("tick")
        windows = [_as_window(value) for value in group["sales_window"]]
        assert all(len(window) == SALES_WINDOW for window in windows)
        assert set(int(value) for value in group["delivery_lag"]) == {lag}
        on_hand = [int(v) for v in group["on_hand"]]
        nxt_on_hand = [int(v) for v in group["next_on_hand"]]
        profiles = [_as_profile(value) for value in group["in_transit"]]
        nxt_transit = [_as_profile(value) for value in group["next_in_transit"]]
        nxt_sales = [int(v) for v in group["next_sales"]]
        assert all(len(profile) == ARRIVAL_AGE_SLOTS for profile in profiles)
        assert nxt_on_hand[:-1] == on_hand[1:]
        assert nxt_transit[:-1] == profiles[1:]
        for index in range(len(group) - 1):
            assert windows[index + 1] == (windows[index] + [nxt_sales[index]])[-SALES_WINDOW:]

    for (_, _), group in df.groupby(["episode_seed", "tick"]):
        assert group["cash"].nunique() == 1
        assert group["capacity"].nunique() == 1
        # Allocation clamps on on-hand, not on the in-transit pipeline.
        free = int(group["capacity"].iloc[0]) - int(group["on_hand"].sum())
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


def test_factories_stay_static_and_order_up_to_uses_the_sampled_space():
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
    sampled, params = sample_textbook_policy("order_up_to", 1)
    got = order_up_to["S"]
    assert type(got) is type(sampled) is OrderUpToPolicy
    assert got.cover_horizon_ticks == params["cover_horizon_ticks"]
    assert got.safety_lead_pct_of_lag == params["safety_lead_pct_of_lag"]
    assert got.min_qty == 0
    assert got.per_supplier_min_order_floor == 0


def test_each_arm_calls_its_own_policy(monkeypatch):
    classes = {
        "order_up_to": OrderUpToPolicy,
        "reorder_point": ReorderPointPolicy,
        "periodic_order_up_to": PeriodicOrderUpToPolicy,
        "periodic_reorder": PeriodicReorderPolicy,
        "exploratory": ExploratoryPolicy,
        "never_order": NeverOrderPolicy,
        "over_order": OverOrderPolicy,
    }
    calls = {name: 0 for name in classes}
    for name, cls in classes.items():
        original = cls.decide

        def counted(self, *args, _name=name, _original=original, **kwargs):
            calls[_name] += 1
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(cls, "decide", counted)

    collect_transition_log(
        n_episodes_each=1,
        episode_length=2,
        k_active=2,
        seed_offset=14_000_000,
    )
    assert calls == {name: episodes_for(name, 1) * 2 for name in classes}


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
    assert len(built) == n_factories * len(behavior_schedule(1, 14_000_000))


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
    assert set(loaded["behavior"]) == set(BEHAVIORS)
    assert len(_as_profile(loaded["in_transit"].iloc[0])) == ARRIVAL_AGE_SLOTS
    assert int(loaded["delivery_lag"].iloc[0]) == 3
    assert "filled_qty" in loaded.columns
    assert "requested_qty" in loaded.columns


def test_collector_writes_a_manifest_beside_the_log(tmp_path):
    from src.tuning.episode import make_synthetic_catalog, partition_catalog

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
    full = make_synthetic_catalog(config.active_catalog_size())
    catalog, evaluation = partition_catalog(full)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    nonzero = float((frame["filled_qty"] > 0).mean()) if len(frame) else 0.0

    schedule = behavior_schedule(1, 14_000_000)
    assert manifest["episode_count"] == len(schedule)
    assert manifest["seed_min"] == 14_000_000
    assert manifest["seed_max"] == 14_000_000 + len(schedule) - 1
    assert manifest["episode_length"] == 2
    assert manifest["K_active"] == 2
    assert manifest["delivery_lag"] == 3
    assert manifest["holding_rate"] == config.holding_rate
    assert isinstance(manifest["holding_rate"], float)
    assert manifest["order_fee"] == config.order_fee
    assert manifest["catalog_identity"] == catalog_identity(catalog)
    assert manifest["evaluation_catalog_identity"] == catalog_identity(evaluation)
    assert manifest["training_sku_count"] == len(catalog)
    assert manifest["evaluation_sku_count"] == len(evaluation)
    training_ids = {ware.product_id for ware in catalog}
    evaluation_ids = {ware.product_id for ware in evaluation}
    assert training_ids.isdisjoint(evaluation_ids)
    mixture = {row["behavior"]: row for row in manifest["behavior_mixture"]}
    assert list(mixture) == list(BEHAVIORS)
    for behavior in BEHAVIORS:
        arm = mixture[behavior]
        assert arm["episodes"] == episodes_for(behavior, 1)
        assert len(arm["hyperparameters"]) == episodes_for(behavior, 1)
        for hyper in arm["hyperparameters"]:
            _policy, expected = log_buyer(behavior, hyper["episode_seed"], 3)
            assert set(hyper) == set(expected)
            for key, value in expected.items():
                assert hyper[key] == pytest.approx(value)
    assert manifest["git_sha"] == sha
    assert manifest["nonzero_order_fraction"] == nonzero
    assert set(manifest["state_coverage"]) == {
        "low_inventory",
        "high_inventory",
        "capacity_bound",
        "cash_bound",
    }


def test_log_package_does_not_import_the_rl_stack():
    for path in Path("src/jepa").glob("*.py"):
        text = path.read_text()
        assert "src.rl" not in text
        assert "optuna" not in text

    script = (
        "import src.jepa, sys\n"
        "bad = [name for name in sys.modules if name == 'src.rl' or name.startswith('src.rl.')]\n"
        "assert not bad, bad\n"
    )
    subprocess.check_call([sys.executable, "-c", script])


def _stored(policy, key: str):
    """Attribute on the buyer, or on the inner textbook policy it wraps."""
    if hasattr(policy, key):
        return getattr(policy, key)
    inner = getattr(policy, "_inner", None)
    if inner is not None and hasattr(inner, key):
        return getattr(inner, key)
    raise AssertionError(f"{type(policy).__name__} dropped {key}")


def test_textbook_buyers_match_the_tuning_search_spaces():
    """The log does not invent ranges. It calls the space factories."""
    for behavior in (
        "order_up_to",
        "reorder_point",
        "periodic_order_up_to",
        "periodic_reorder",
    ):
        for seed in (0, 1, 14_000_000):
            policy, hyper = log_buyer(behavior, seed, delivery_lag=4)
            sampled, params = sample_textbook_policy(behavior, seed)
            assert type(policy) is type(sampled)
            assert hyper["planning_depth"] == params["cover_horizon_ticks"]
            for key, value in params.items():
                assert hyper[key] == value
                assert _stored(policy, key) == _stored(sampled, key) == value


def test_planning_depth_varies_across_log_episodes():
    depths: dict[str, set[int]] = {behavior: set() for behavior in BEHAVIORS}
    for behavior, seed in behavior_schedule(8, LOG_SEED_OFFSET):
        _policy, hyper = log_buyer(behavior, seed, delivery_lag=3)
        depths[behavior].add(int(hyper["planning_depth"]))
    for behavior in ("order_up_to", "reorder_point", "exploratory", "over_order"):
        assert len(depths[behavior]) > 1
    assert depths["never_order"] == {0}
    low, high = COVER_HORIZON_TICKS
    assert all(low <= depth <= high for depth in depths["exploratory"])
    assert all(low <= depth <= high for depth in depths["order_up_to"])
    assert all(depth > high for depth in depths["over_order"])


def _cell(value: object) -> object:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def test_streamed_collect_matches_the_in_memory_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    kwargs = {
        "n_episodes_each": 1,
        "episode_length": 3,
        "k_active": 2,
        "delivery_lag": 3,
        "seed_offset": 14_000_000,
    }
    eager_path = tmp_path / "eager.parquet"
    eager = collect_transition_log(path=eager_path, **kwargs)
    monkeypatch.setattr("src.jepa.log._STREAM_EPISODE_MIN", 1)
    stream_path = tmp_path / "stream.parquet"
    returned = collect_transition_log(path=stream_path, **kwargs)
    assert returned.empty
    loaded = pd.read_parquet(stream_path)
    assert list(loaded.columns) == list(eager.columns)
    assert len(loaded) == len(eager)
    for column in eager.columns:
        assert [_cell(value) for value in loaded[column]] == [_cell(value) for value in eager[column]]
    eager_manifest = json.loads(eager_path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    stream_manifest = json.loads(stream_path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert stream_manifest == eager_manifest


def test_published_log_uses_one_holding_rate():
    cfg = collection_config()
    assert cfg.holding_rate == 0.01
    rates = set()
    for offset in range(5):
        spec = build_log_scenario(cfg, LOG_SEED_OFFSET + offset)
        shop = next(instance.node for instance in spec.scenario.nodes if instance.node.id == "S")
        rates.add(shop.holding_rate)
    assert rates == {0.01}


@pytest.fixture(scope="module")
def mixture_log(tmp_path_factory):
    """Two episodes of each arm, eight of over-order, on the capacity-hitting seed block."""
    path = tmp_path_factory.mktemp("mixture") / "transitions.parquet"
    frame = collect_transition_log(
        n_episodes_each=2,
        episode_length=80,
        k_active=2,
        delivery_lag=3,
        seed_offset=LOG_SEED_OFFSET,
        path=path,
    )
    manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    return frame, manifest


def test_nonzero_order_fraction_is_materially_above_the_old_log(mixture_log):
    frame, manifest = mixture_log
    realised = float((frame["filled_qty"] > 0).mean())
    assert manifest["nonzero_order_fraction"] == pytest.approx(realised)
    assert realised > 0.18


def test_state_coverage_is_asserted_on_the_collected_log(mixture_log):
    frame, manifest = mixture_log
    coverage = manifest["state_coverage"]
    assert coverage["low_inventory"] == pytest.approx(float((frame["on_hand"] == 0).mean()))
    for key in ("low_inventory", "high_inventory", "capacity_bound", "cash_bound"):
        assert coverage[key] > 0.0
