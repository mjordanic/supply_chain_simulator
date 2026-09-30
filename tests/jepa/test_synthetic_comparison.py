"""Synthetic comparison tables for the latent replenishment planner.

Seams, from the issue acceptance criteria:

1. Pass rules. The in-distribution rule passes when the learned arm beats
   the tuned anchor and the bootstrap interval of that paired difference
   excludes zero. A tie fails. The retarget rule passes when the frozen
   latent planner beats the frozen textbook policy on that paired profit.
   Smoke runs check that the booleans match these rules. They do not
   require the published thresholds to pass.
2. Arms. A smoke run returns a finite paired ``net_profit / initial_cash``
   for the latent planner, the direct planner, permutation priority,
   textbook fair-share, the never-order, random-cover, and least-squares
   floors, the oracle ceiling, published-default ``OrderUpToPolicy``, and
   the other three textbook policies at published defaults. Both protocols
   carry the floors, the oracle, and the learned arms.
3. Columns. Order volume, service level, stockout rate, and realised fill
   rate sit on every arm. Tightness is a per-seed column. The paired
   comparison is the per-seed difference. Arms correlating above 0.99 are
   a warning and are omitted from the written table.
4. Rates. The in-distribution smoke uses ``holding_rate = 0.01``. The
   retarget smoke scores the same weights at ``holding_rate = 0.05`` and
   includes the textbook policy from the ``0.01`` anchor, left frozen.
5. Factories. An ``OrderUpToPolicy`` factory is the ``0.01`` anchor on
   both protocols. A second factory is the ``0.05`` ceiling and is not a
   pass bar. With no anchor factory, the anchor is the published default.
   Anchor and planner share routing, the minimum-order floor, and the
   supplier's unit cost. No tuning study runs.
6. Holdout defaults. The in-distribution seed count is 32 and the offset
   is ``13_000_000``. Unseen-SKU starts at ``15_000_000`` and
   unseen-disruption at ``16_000_000``. Smoke tests override the seed count.
   The three blocks stay disjoint. Retarget reuses the in-distribution seeds.
7. The RL stack is not imported. The publish gate is unchanged.
"""

from __future__ import annotations

import inspect
import json
import math
import subprocess
import sys
from dataclasses import replace
from unittest.mock import patch

import pandas as pd
import torch

from src.jepa.eval import (
    HOLDOUT_SEED_OFFSET,
    LEARNED_ARMS,
    N_HOLDOUT_SEEDS,
    UNSEEN_DISRUPTION_SEED_OFFSET,
    UNSEEN_SKU_SEED_OFFSET,
    ComparisonTable,
    SyntheticComparison,
    collinear_report,
    compare_synthetic,
    harsh_disruption_params,
    pearson,
    in_distribution_passes,
    learned_beats_never_order,
    paired_interval,
    retarget_passes,
    write_synthetic_comparison,
)
from src.jepa.log import collect_transition_log
from src.jepa.plan import fit_direct, fit_latent
from src.sim.distributions import Constant
from src.sim.episode_sampler import default_lifecycle_params
from src.sim.scenario import load_catalog
from src.sim.policy import (
    IntermediatePolicy,
    OrderUpToPolicy,
    PeriodicOrderUpToPolicy,
    PeriodicReorderPolicy,
    ReorderPointPolicy,
)
from src.tuning.config import TuningConfig
from src.tuning.episode import make_synthetic_catalog, partition_catalog, sample_episode
from src.tuning.evaluator import evaluate_policy_normalised


def _smoke_config() -> TuningConfig:
    return TuningConfig(
        episode_length=21,
        K_active=2,
        delivery_lag=2,
        capacity_dist=Constant(5_000),
        balance_dist=Constant(1_000_000.0),
        init_stock_pct_dist=Constant(0.05),
        order_fee=50.0,
    )


def _fitted(lag: int = 2):
    frame = collect_transition_log(
        n_episodes_each=1,
        episode_length=lag + 3,
        k_active=2,
        delivery_lag=lag,
        seed_offset=14_000_000,
    )
    latent = fit_latent(frame, delivery_lag=lag, epochs=1, seed=0, policy_seed=3, budget=4)
    direct = fit_direct(frame, delivery_lag=lag, epochs=1, seed=1, policy_seed=5, budget=4)
    return latent, direct, frame


def test_in_distribution_fails_when_the_paired_interval_covers_zero():
    # Paired gaps of +1 and -1. The interval around that mean covers zero.
    assert in_distribution_passes([1.0, 0.0], [0.0, 1.0]) is False


def test_in_distribution_fails_a_loss_even_when_the_interval_excludes_zero():
    # The old 95% rule passed this. The interval is the point -0.4.
    assert in_distribution_passes([9.6, 9.6], [10.0, 10.0]) is False
    assert in_distribution_passes([9.0, 9.0], [10.0, 10.0]) is False


def test_in_distribution_passes_when_the_interval_lies_above_zero():
    assert in_distribution_passes([1.2, 1.4], [1.0, 1.0]) is True


def test_retarget_passes_only_when_the_frozen_planner_beats_the_frozen_textbook():
    assert retarget_passes([1.0, 1.2], [1.0, 1.0]) is True
    assert retarget_passes([1.0, 1.0], [1.0, 1.0]) is False
    assert retarget_passes([0.5, 0.5], [1.0, 1.0]) is False


def test_smoke_comparison_returns_paired_profit_for_every_arm():
    latent, direct, frame = _fitted()
    latent_weights = {key: value.clone() for key, value in latent.dynamics.state_dict().items()}
    direct_weights = {key: value.clone() for key, value in direct.dynamics.state_dict().items()}
    config = _smoke_config()
    offset = 13_000_000
    result = compare_synthetic(
        latent,
        direct,
        n_seeds=2,
        seed_offset=offset,
        config=config,
        frame=frame,
    )

    learned = {"latent", "direct", "permutation", "fair_share"}
    floors = {"never_order", "random_cover", "least_squares", "oracle"}
    textbooks = {
        "anchor",
        "order_up_to",
        "reorder_point",
        "periodic_order_up_to",
        "periodic_reorder",
    }
    assert set(result.in_distribution.per_seed) == learned | floors | textbooks
    assert set(result.unseen_sku.per_seed) == learned | floors | textbooks
    assert set(result.unseen_disruption.per_seed) == learned | floors | textbooks
    assert set(result.retarget.per_seed) == learned | floors | {"anchor"}
    assert result.in_distribution.holding_rate == 0.01
    assert result.unseen_sku.holding_rate == 0.01
    assert result.unseen_disruption.holding_rate == 0.01
    assert result.retarget.holding_rate == 0.05
    assert result.in_distribution.seeds == (offset, offset + 1)
    assert result.retarget.seeds == (offset, offset + 1)
    assert result.unseen_sku.seeds == (UNSEEN_SKU_SEED_OFFSET, UNSEEN_SKU_SEED_OFFSET + 1)
    assert result.unseen_disruption.seeds == (
        UNSEEN_DISRUPTION_SEED_OFFSET,
        UNSEEN_DISRUPTION_SEED_OFFSET + 1,
    )
    assert set(result.in_distribution.seeds).isdisjoint(result.unseen_sku.seeds)
    assert set(result.in_distribution.seeds).isdisjoint(result.unseen_disruption.seeds)
    assert set(result.unseen_sku.seeds).isdisjoint(result.unseen_disruption.seeds)

    for table in (
        result.in_distribution,
        result.unseen_sku,
        result.unseen_disruption,
        result.retarget,
    ):
        assert table.collinear_warnings == collinear_report(table.per_seed)[0]
        assert len(table.tightness) == len(table.cash_tightness) == 2
        assert len(table.paired_difference) == 2
        assert table.paired_difference == tuple(
            left - right
            for left, right in zip(table.per_seed["latent"], table.per_seed["anchor"], strict=True)
        )
        for name, series in table.per_seed.items():
            assert len(series) == 2
            assert all(math.isfinite(value) for value in series)
            for column in (
                table.order_volume,
                table.service_level,
                table.stockout_rate,
                table.fill_rate,
            ):
                assert name in column
                assert len(column[name]) == 2
            assert all(math.isfinite(value) for value in table.order_volume[name])
            assert all(math.isfinite(value) for value in table.service_level[name])
            assert all(math.isfinite(value) for value in table.stockout_rate[name])
        assert table.order_volume["never_order"] == (0.0, 0.0)

    assert result.in_distribution.per_seed["anchor"] == result.in_distribution.per_seed["order_up_to"]

    training, evaluation = partition_catalog(make_synthetic_catalog(config.K_catalog))
    at_one = [
        sample_episode(training, replace(config, holding_rate=0.01), offset + i) for i in range(2)
    ]
    at_five = [
        sample_episode(training, replace(config, holding_rate=0.05), offset + i) for i in range(2)
    ]
    unseen = [
        sample_episode(
            evaluation,
            replace(config, holding_rate=0.01),
            UNSEEN_SKU_SEED_OFFSET + i,
        )
        for i in range(2)
    ]
    shocked = [
        sample_episode(
            training,
            replace(config, holding_rate=0.01),
            UNSEEN_DISRUPTION_SEED_OFFSET + i,
            disruption_params=harsh_disruption_params(),
        )
        for i in range(2)
    ]
    published_arms = {
        "order_up_to": OrderUpToPolicy,
        "reorder_point": ReorderPointPolicy,
        "periodic_order_up_to": PeriodicOrderUpToPolicy,
        "periodic_reorder": PeriodicReorderPolicy,
    }
    for label, policy_cls in published_arms.items():
        assert result.in_distribution.per_seed[label] == _series(policy_cls, at_one)
    assert result.unseen_sku.per_seed["order_up_to"] == _series(OrderUpToPolicy, unseen)
    assert result.unseen_disruption.per_seed["order_up_to"] == _series(
        OrderUpToPolicy, shocked
    )
    frozen = _series(OrderUpToPolicy, at_five)
    assert result.retarget.per_seed["anchor"] == frozen
    assert all(
        left < right
        for left, right in zip(frozen, result.in_distribution.per_seed["order_up_to"], strict=True)
    )

    low, high = paired_interval(
        result.in_distribution.per_seed["latent"],
        result.in_distribution.per_seed["anchor"],
    )
    assert result.in_distribution_paired_interval == (low, high)
    assert result.in_distribution_passes == in_distribution_passes(
        result.in_distribution.per_seed["latent"],
        result.in_distribution.per_seed["anchor"],
    )
    assert result.retarget_passes == retarget_passes(
        result.retarget.per_seed["latent"],
        result.retarget.per_seed["anchor"],
    )
    assert isinstance(result.in_distribution_passes, bool)
    assert isinstance(result.retarget_passes, bool)

    for key, value in latent_weights.items():
        assert torch.equal(latent.dynamics.state_dict()[key], value)
    for key, value in direct_weights.items():
        assert torch.equal(direct.dynamics.state_dict()[key], value)


class _FixedOrder(IntermediatePolicy):
    """Buys ``qty`` of every SKU from one of its factories, every tick."""

    def __init__(self, qty: int) -> None:
        super().__init__(policy_seed=0)
        self.qty = qty

    def decide(self, obs_intermediate, central_table):
        inventory = obs_intermediate.get("inventory") or {}
        pids = sorted(inventory)
        direct = obs_intermediate.get("direct_supplier_ids") or []
        prices = obs_intermediate.get("base_prices") or obs_intermediate.get("list_prices") or {}
        order = {}
        for pid in pids:
            suppliers = sorted(
                sid for sid in direct if str(sid).startswith(f"F_{pid}_")
            )
            if suppliers:
                order[pid] = [(suppliers[0], self.qty)]
        return {
            "order": order,
            "list_price": {pid: float(prices.get(pid, 0.0)) for pid in pids},
            "min_order_imposed": {pid: 0 for pid in pids},
            "promotions": {},
        }


def _series(factory, specs) -> tuple[float, ...]:
    kpis = evaluate_policy_normalised(factory, specs)
    return tuple(
        float(profit) / max(1e-9, float(cash))
        for profit, cash in zip(
            kpis["per_seed_net_profit"],
            kpis["per_seed_initial_cash"],
            strict=True,
        )
    )


def test_anchor_and_ceiling_factories_are_scored_without_a_study():
    latent, direct, frame = _fitted()
    config = _smoke_config()
    offset = 13_000_000
    training, _evaluation = partition_catalog(make_synthetic_catalog(config.K_catalog))
    at_one = [
        sample_episode(training, replace(config, holding_rate=0.01), offset + i) for i in range(2)
    ]
    at_five = [
        sample_episode(training, replace(config, holding_rate=0.05), offset + i) for i in range(2)
    ]

    def anchor() -> _FixedOrder:
        return _FixedOrder(5)

    def ceiling() -> _FixedOrder:
        return _FixedOrder(40)

    result = compare_synthetic(
        latent,
        direct,
        anchor_factory=anchor,
        ceiling_factory=ceiling,
        n_seeds=2,
        seed_offset=offset,
        config=config,
        frame=frame,
    )

    assert result.in_distribution.per_seed["anchor"] == _series(anchor, at_one)
    assert result.retarget.per_seed["anchor"] == _series(anchor, at_five)
    assert result.in_distribution.per_seed["anchor"] != result.in_distribution.per_seed["order_up_to"]
    assert "ceiling" not in result.in_distribution.per_seed
    assert result.retarget.per_seed["ceiling"] == _series(ceiling, at_five)
    assert result.retarget.per_seed["ceiling"] != result.retarget.per_seed["anchor"]
    assert result.retarget_passes == retarget_passes(
        result.retarget.per_seed["latent"],
        result.retarget.per_seed["anchor"],
    )


def test_published_defaults_are_the_tuner_holdout():
    params = inspect.signature(compare_synthetic).parameters
    holdout = TuningConfig()
    assert params["n_seeds"].default == holdout.n_holdout_seeds == 32
    assert params["seed_offset"].default == holdout.holdout_seed_offset == 13_000_000
    assert N_HOLDOUT_SEEDS == 32
    assert HOLDOUT_SEED_OFFSET == 13_000_000


def test_a_collinear_chain_keeps_an_arm_distinct_from_the_survivor():
    """Oracle can match least squares and the learned arm, while those two differ.

    The learned arm stays in the file. Dropping it because it matches the
    omitted oracle would hide the comparison the ceiling exists to make.
    """
    least_squares = (0.9417, -1.3966, -0.6797, 0.3705, -1.0163, -0.0721, 0.1792, -0.8311)
    oracle = (0.8370, -1.3811, -0.6003, 0.3187, -1.0430, 0.0595, 0.1345, -0.8722)
    latent = (1.0293, -1.5036, -0.5365, 0.1585, -1.0908, 0.1798, 0.2322, -0.9443)
    warnings, dropped = collinear_report(
        {"least_squares": least_squares, "oracle": oracle, "latent": latent}
    )
    assert "latent" not in dropped
    assert "oracle" in dropped
    assert any("least_squares" in line and "oracle" in line for line in warnings)
    assert any("oracle" in line and "latent" in line for line in warnings)
    assert pearson(least_squares, latent) <= 0.99


def test_correlation_of_minus_one_is_still_a_distinct_comparison():
    warnings, dropped = collinear_report(
        {"latent": (1.0, 2.0, 3.0), "anchor": (3.0, 2.0, 1.0)}
    )
    assert warnings == ()
    assert dropped == ()


def test_collinear_arms_are_a_warning_and_omitted_from_the_written_table(tmp_path):
    profits = {
        "latent": (0.4, 0.6),
        "direct": (0.4, 0.6),
        "anchor": (1.0, 0.0),
    }
    warnings, dropped = collinear_report(profits)
    assert dropped == ("direct",)
    table = ComparisonTable(
        holding_rate=0.01,
        seeds=(1, 2),
        per_seed=profits,
        tightness=(0.5, 1.5),
        cash_tightness=(0.8, 1.2),
        order_volume={name: (1.0, 2.0) for name in profits},
        service_level={name: (0.5, 0.5) for name in profits},
        stockout_rate={name: (0.1, 0.2) for name in profits},
        fill_rate={name: (1.0, 0.5) for name in profits},
        paired_difference=(-0.6, 0.6),
        paired_interval=(0.1, 0.2),
        collinear_warnings=warnings,
        dropped_arms=dropped,
    )
    retarget = ComparisonTable(
        holding_rate=0.05,
        seeds=(1, 2),
        per_seed={"latent": (0.0, 1.0), "anchor": (0.0, 0.0)},
    )
    from src.jepa.dynamics import RolloutAccuracy as Accuracy

    result = SyntheticComparison(
        in_distribution=table,
        retarget=retarget,
        in_distribution_passes=False,
        retarget_passes=True,
        in_distribution_paired_interval=(0.1, 0.2),
        unseen_sku=ComparisonTable(
            holding_rate=0.01,
            seeds=(15_000_000, 15_000_001),
            per_seed={"latent": (0.2, 0.3), "anchor": (0.1, 0.1)},
        ),
        unseen_disruption=ComparisonTable(
            holding_rate=0.01,
            seeds=(16_000_000, 16_000_001),
            per_seed={"latent": (0.2, 0.3), "anchor": (0.1, 0.1)},
        ),
        rollout_accuracy=Accuracy(steps=(1, 2), latent=(0.6, 0.4), direct=(0.2, 0.2), ols=(0.1, 0.1)),
    )
    write_synthetic_comparison(result, tmp_path)
    frame = pd.read_parquet(tmp_path / "in_distribution.parquet")
    sku_frame = pd.read_parquet(tmp_path / "unseen_sku.parquet")
    shock_frame = pd.read_parquet(tmp_path / "unseen_disruption.parquet")
    assert "latent" in frame.columns and "anchor" in frame.columns
    assert "direct" not in frame.columns
    assert "direct_order_volume" not in frame.columns
    assert "latent_order_volume" in frame.columns
    assert "latent_service_level" in frame.columns
    assert "latent_stockout_rate" in frame.columns
    assert "latent_fill_rate" in frame.columns
    assert frame["tightness"].tolist() == [0.5, 1.5]
    assert frame["cash_tightness"].tolist() == [0.8, 1.2]
    assert frame["latent_minus_anchor"].tolist() == [-0.6, 0.6]
    assert frame["rollout_r2_latent"].tolist() == [0.5, 0.5]
    assert frame["rollout_r2_direct"].tolist() == [0.2, 0.2]
    assert frame["rollout_r2_ols"].tolist() == [0.1, 0.1]
    written = json.loads((tmp_path / "comparison_warnings.json").read_text(encoding="utf-8"))
    assert any("direct" in line and "latent" in line for line in written["in_distribution"])
    assert "direct" in result.in_distribution.per_seed
    assert not (tmp_path / "pooled.parquet").exists()
    assert set(frame["seed"]).isdisjoint(sku_frame["seed"])
    assert set(sku_frame["seed"]).isdisjoint(shock_frame["seed"])


def test_oracle_sells_the_recent_rate_and_lands_the_grant_after_the_lead_time():
    from src.jepa.dynamics import ARRIVAL_AGE_SLOTS, OracleDynamics, sku_features

    features = [
        sku_features(
            on_hand=10,
            in_transit=[0] * ARRIVAL_AGE_SLOTS,
            sales_window=[4] * 28,
            cash=100,
            capacity=50,
            delivery_lag=2,
        )
    ]
    rollout = OracleDynamics(2).open_rollout(features)
    on_hand, profile, sales = rollout.step([3.0])[0]
    assert sales == 4.0
    assert on_hand == 6.0
    assert profile[1] == 3.0
    on_hand, profile, sales = rollout.step([0.0])[0]
    assert sales == 4.0
    assert on_hand == 6.0 + 3.0 - 4.0


def test_anchor_and_planner_share_routing_floor_and_supplier_unit_cost():
    """The comparison anchor and the planner price the pilot at the offer."""
    from src.jepa.dynamics import OracleDynamics
    from src.jepa.plan import ReplenishmentPlanner
    from src.sim.central_table import CentralTable, Offer
    from src.sim.policy import MIN_ORDER_FLOOR, OPENING_BUDGET_PCT, OrderUpToPolicy

    anchor = OrderUpToPolicy()
    planner = ReplenishmentPlanner(
        OracleDynamics(3),
        policy_seed=0,
        delivery_lag=3,
        budget=4,
        order_fee=0.0,
    )
    assert anchor.routing_strategy is None
    assert planner.routing_strategy is None
    assert anchor.per_supplier_min_order_floor == planner.per_supplier_min_order_floor == MIN_ORDER_FLOOR

    price = 11.0
    cash = 1100.0
    expected = int(OPENING_BUDGET_PCT * cash / price)
    assert expected != int(OPENING_BUDGET_PCT * cash / anchor.unit_cost)
    table = CentralTable()
    table.publish("F", "A", Offer(available_qty=10_000, list_price=price, min_order=0))
    obs = {
        "tick": 0,
        "inventory": {"A": 0},
        "pending": {},
        "observed_sales": {"A": 0},
        "capacity": 100_000,
        "cash": cash,
        "base_prices": {"A": 20.0},
        "list_prices": {"A": 20.0},
        "direct_supplier_ids": {"F"},
    }
    planner_qty = sum(qty for _sid, qty in planner.decide(obs, table)["order"].get("A", []))
    anchor_qty = sum(qty for _sid, qty in anchor.decide(obs, table)["order"].get("A", []))
    assert planner_qty == anchor_qty == expected


def test_learned_arms_beat_the_never_order_floor_and_a_duplicate_is_a_warning():
    """One fast seller, empty shelf, no order fee. Never-order sells nothing.

    The published catalog's demand runs from about 0.5 to 40, and a sampled
    dead lifecycle makes a purchase lose to never-order for a reason other
    than refusing to buy. This world pins two SKUs at demand 12, one in each
    catalog partition, with the inert lifecycle, and sets the order fee to
    zero because the published fee of 50 exceeds the margin of a
    tightness-sized opening order. One SKU per episode makes permutation
    the same buyer as the density arm, so the comparison must warn instead
    of treating it as an ablation.
    """
    catalog = load_catalog(
        [
            {
                "name": "Product 0",
                "category": "General",
                "related_products": [],
                "base_price": 10.0,
                "unit_cost": 5.0,
                "seasonality": "all_season",
                "demand_mean": 12.0,
                "demand_cv": 0.05,
            },
            {
                "name": "Product 1",
                "category": "General",
                "related_products": [],
                "base_price": 10.0,
                "unit_cost": 5.0,
                "seasonality": "all_season",
                "demand_mean": 12.0,
                "demand_cv": 0.05,
            },
        ]
    )
    original_sample = sample_episode

    def _inert_lifecycle(catalog, config, seed, **kwargs):
        kwargs.setdefault("lifecycle_params", default_lifecycle_params())
        return original_sample(catalog, config, seed, **kwargs)

    latent, direct, frame = _fitted()
    config = TuningConfig(
        episode_length=40,
        K_active=1,
        K_catalog=1,
        delivery_lag=2,
        init_stock_pct_dist=Constant(0.0),
        order_fee=0.0,
    )
    with (
        patch("src.jepa.eval.make_synthetic_catalog", return_value=catalog),
        patch("src.jepa.eval.sample_episode", side_effect=_inert_lifecycle),
    ):
        result = compare_synthetic(
            latent,
            direct,
            n_seeds=2,
            seed_offset=13_000_000,
            config=config,
            frame=frame,
        )
    assert learned_beats_never_order(result.in_distribution)
    assert learned_beats_never_order(result.unseen_sku)
    assert learned_beats_never_order(result.retarget)
    for name in LEARNED_ARMS:
        assert name in result.in_distribution.per_seed
    assert "permutation" in result.in_distribution.dropped_arms
    assert any(
        "permutation" in warning and "latent" in warning
        for warning in result.in_distribution.collinear_warnings
    )
    assert all(math.isfinite(value) for value in result.in_distribution.tightness)
    assert all(math.isnan(value) for value in result.in_distribution.fill_rate["never_order"])


def test_comparison_does_not_import_the_rl_stack():
    script = (
        "import src.jepa.eval, sys\n"
        "bad = [name for name in sys.modules if name == 'src.rl' or name.startswith('src.rl.')]\n"
        "assert not bad, bad\n"
    )
    subprocess.check_call([sys.executable, "-c", script])
