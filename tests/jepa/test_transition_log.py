"""Synthetic transition log.

Seams, from the issue acceptance criteria:

1. Mix. Half the episodes are published-default ``OrderUpToPolicy`` and half
   are ``ExploratoryPolicy``.
2. Schema. Each row is one SKU at one tick: on-hand, in-transit, a sales
   window of length ``delivery_lag``, cash, capacity, the granted quantity,
   and the next on-hand, in-transit, and sales.
3. Published shape. Defaults are 365 ticks, five SKUs, lead time 3,
   ``holding_rate`` 0.01, the tuner default order fee, 64 episodes of each
   behavior, and seed offset ``14_000_000``. Smoke runs pass a shorter
   episode and a smaller count.
4. Factories on those episodes stay on ``StaticFactoryPolicy``. The package
   does not import the RL stack.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.jepa.log import (
    EPISODES_EACH,
    LOG_SEED_OFFSET,
    build_log_overrides,
    build_log_scenario,
    collect_transition_log,
    collection_config,
)
from src.jepa.policy import ExploratoryPolicy
from src.sim.policy import OrderUpToPolicy, StaticFactoryPolicy
from src.tuning.config import TuningConfig


def _as_window(value) -> list[int]:
    return [int(x) for x in list(value)]


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
        "granted_qty",
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
        nxt_transit = [int(v) for v in group["next_in_transit"]]
        nxt_sales = [int(v) for v in group["next_sales"]]
        assert nxt_on_hand[:-1] == on_hand[1:]
        assert nxt_transit[:-1] == [int(v) for v in group["in_transit"]][1:]
        for index in range(len(group) - 1):
            assert windows[index + 1] == (windows[index] + [nxt_sales[index]])[-lag:]

    for (_, _), group in df.groupby(["episode_seed", "tick"]):
        assert group["cash"].nunique() == 1
        assert group["capacity"].nunique() == 1
        free = int(group["capacity"].iloc[0]) - int(group["on_hand"].sum()) - int(
            group["in_transit"].sum()
        )
        assert int(group["granted_qty"].sum()) <= max(0, free)
        assert (group["granted_qty"] >= 0).all()


def test_published_collector_matches_the_tuner_episode_shape():
    cfg = collection_config()
    plain = TuningConfig()
    assert cfg.episode_length == 365 == plain.episode_length
    assert cfg.K_active == 5 == plain.K_active
    assert cfg.delivery_lag == 3 == plain.delivery_lag
    assert cfg.holding_rate == 0.01 == plain.holding_rate
    assert cfg.order_fee == plain.order_fee
    assert EPISODES_EACH == 64
    assert LOG_SEED_OFFSET == 14_000_000
    assert LOG_SEED_OFFSET > plain.holdout_seed_offset
    assert LOG_SEED_OFFSET >= plain.seed_offset + plain.n_search_seeds
    assert LOG_SEED_OFFSET >= plain.holdout_seed_offset + plain.n_holdout_seeds

    spec = build_log_scenario(cfg, LOG_SEED_OFFSET)
    shop = next(instance.node for instance in spec.scenario.nodes if instance.node.id == "S")
    assert shop.holding_rate == 0.01
    assert shop.order_fee == plain.order_fee
    assert spec.scenario.n_steps == 365
    assert len(spec.active_subset) == 5
    lags = [edge.default_lead_time for edge in spec.scenario.edges if edge.buyer_id == "S"]
    assert lags and set(lags) == {3}
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
    for pid in spec.active_subset:
        assert isinstance(exploratory[f"F_{pid}"], StaticFactoryPolicy)
        assert isinstance(order_up_to[f"F_{pid}"], StaticFactoryPolicy)
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


def test_collector_builds_a_static_factory_per_sku(monkeypatch):
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
        seed_offset=14_000_000,
    )
    assert len(built) == 4


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
        seed_offset=14_000_000,
        path=path,
    )
    assert path.is_file()
    loaded = pd.read_parquet(path)
    assert len(loaded) == len(frame)
    assert set(loaded["behavior"]) == {"order_up_to", "exploratory"}


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
