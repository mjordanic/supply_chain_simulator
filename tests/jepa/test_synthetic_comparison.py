"""Synthetic comparison tables for the latent replenishment planner.

Seams, from the issue acceptance criteria:

1. Pass rules. The in-distribution rule passes when the paired interval of
   ``net_profit / initial_cash`` against the anchor covers zero, or the
   latent mean is at least 95% of the anchor mean. The retarget rule
   passes when the frozen latent planner beats the frozen textbook policy
   on that paired profit. Smoke runs check that the booleans match these
   rules. They do not require the published thresholds to pass.
2. Arms. A smoke run returns a finite paired ``net_profit / initial_cash``
   on one held-out seed for the latent planner, the direct planner, latent
   covers with textbook fair-share, published-default ``OrderUpToPolicy``,
   and the other three textbook policies at published defaults.
3. Rates. The in-distribution smoke uses ``holding_rate = 0.01``. The
   retarget smoke scores the same weights at ``holding_rate = 0.05`` and
   includes the textbook policy from the ``0.01`` anchor, left frozen.
4. Factories. An ``OrderUpToPolicy`` factory is the ``0.01`` anchor on
   both protocols. A second factory is the ``0.05`` ceiling and is not a
   pass bar. With no anchor factory, the anchor is the published default.
   No tuning study runs.
5. Holdout defaults. The entry point's published seed count is 32 and the
   offset is ``13_000_000``. Smoke tests override the seed count.
6. The RL stack is not imported.
"""

from __future__ import annotations

import inspect
import math
import subprocess
import sys
from dataclasses import replace

import torch

from src.jepa.eval import (
    HOLDOUT_SEED_OFFSET,
    N_HOLDOUT_SEEDS,
    compare_synthetic,
    in_distribution_passes,
    paired_interval,
    retarget_passes,
)
from src.jepa.log import collect_transition_log
from src.jepa.plan import fit_direct, fit_latent
from src.sim.distributions import Constant
from src.sim.policy import (
    IntermediatePolicy,
    OrderUpToPolicy,
    PeriodicOrderUpToPolicy,
    PeriodicReorderPolicy,
    ReorderPointPolicy,
)
from src.tuning.config import TuningConfig
from src.tuning.episode import make_synthetic_catalog, sample_episode
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
    return latent, direct


def test_in_distribution_passes_when_the_paired_interval_covers_zero():
    # Paired gaps of +1 and -1. The interval around that mean covers zero.
    assert in_distribution_passes([1.0, 0.0], [0.0, 1.0]) is True


def test_in_distribution_passes_when_the_mean_is_at_least_95_percent_of_the_anchor():
    # Constant gap: the interval sits below zero, and 9.6 is 96% of 10.
    assert in_distribution_passes([9.6, 9.6], [10.0, 10.0]) is True


def test_in_distribution_fails_when_the_gap_misses_zero_and_the_mean_ratio():
    # 9.0 is 90% of 10, and the interval is the point -1.
    assert in_distribution_passes([9.0, 9.0], [10.0, 10.0]) is False


def test_retarget_passes_only_when_the_frozen_planner_beats_the_frozen_textbook():
    assert retarget_passes([1.0, 1.2], [1.0, 1.0]) is True
    assert retarget_passes([1.0, 1.0], [1.0, 1.0]) is False
    assert retarget_passes([0.5, 0.5], [1.0, 1.0]) is False


def test_smoke_comparison_returns_paired_profit_for_every_arm():
    latent, direct = _fitted()
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
    )

    in_dist_arms = {
        "latent",
        "direct",
        "fair_share",
        "anchor",
        "order_up_to",
        "reorder_point",
        "periodic_order_up_to",
        "periodic_reorder",
    }
    assert set(result.in_distribution.per_seed) == in_dist_arms
    assert set(result.retarget.per_seed) == {"latent", "anchor"}
    assert result.in_distribution.holding_rate == 0.01
    assert result.retarget.holding_rate == 0.05
    assert result.in_distribution.seeds == (offset, offset + 1)
    assert result.retarget.seeds == (offset, offset + 1)

    for table in (result.in_distribution, result.retarget):
        for series in table.per_seed.values():
            assert len(series) == 2
            assert all(math.isfinite(value) for value in series)

    assert result.in_distribution.per_seed["anchor"] == result.in_distribution.per_seed["order_up_to"]

    catalog = make_synthetic_catalog(config.K_catalog)
    at_one = [
        sample_episode(catalog, replace(config, holding_rate=0.01), offset + i) for i in range(2)
    ]
    at_five = [
        sample_episode(catalog, replace(config, holding_rate=0.05), offset + i) for i in range(2)
    ]
    published_arms = {
        "order_up_to": OrderUpToPolicy,
        "reorder_point": ReorderPointPolicy,
        "periodic_order_up_to": PeriodicOrderUpToPolicy,
        "periodic_reorder": PeriodicReorderPolicy,
    }
    for label, policy_cls in published_arms.items():
        assert result.in_distribution.per_seed[label] == _series(policy_cls, at_one)
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
    """Buys ``qty`` of every SKU from its factory, every tick."""

    def __init__(self, qty: int) -> None:
        super().__init__(policy_seed=0)
        self.qty = qty

    def decide(self, obs_intermediate, central_table):
        inventory = obs_intermediate.get("inventory") or {}
        pids = sorted(inventory)
        prices = obs_intermediate.get("base_prices") or obs_intermediate.get("list_prices") or {}
        return {
            "order": {pid: [(f"F_{pid}", self.qty)] for pid in pids},
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
    latent, direct = _fitted()
    config = _smoke_config()
    offset = 13_000_000
    catalog = make_synthetic_catalog(config.K_catalog)
    at_one = [
        sample_episode(catalog, replace(config, holding_rate=0.01), offset + i) for i in range(2)
    ]
    at_five = [
        sample_episode(catalog, replace(config, holding_rate=0.05), offset + i) for i in range(2)
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


def test_comparison_does_not_import_the_rl_stack():
    script = (
        "import src.jepa.eval, sys\n"
        "bad = [name for name in sys.modules if name == 'src.rl' or name.startswith('src.rl.')]\n"
        "assert not bad, bad\n"
    )
    subprocess.check_call([sys.executable, "-c", script])
