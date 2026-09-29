"""Demand-side episode distribution.

Public seam: ``src.tuning.episode.sample_episode`` and the run log it
produces. Lifecycle draws, assortment size, and catalog spread are
properties of that seam and stay deterministic in ``episode_seed``.
"""

from __future__ import annotations

from random import Random

from src.sim.distributions import Constant
from src.sim.node import DemandSinkNode
from src.sim.runner import Runner
from src.sim.scenario import CANONICAL_STAGES, load_catalog
from src.tuning.config import TuningConfig
from src.sim.episode_sampler import default_lifecycle_params
from src.tuning.episode import make_synthetic_catalog, sample_episode


def _plain_catalog(n: int = 6):
    return load_catalog(
        [
            {
                "name": f"Product {i}",
                "category": "General",
                "related_products": [],
                "base_price": float(10 + i),
                "unit_cost": float(4 + i),
                "seasonality": "all_season",
            }
            for i in range(n)
        ]
    )


def test_episode_samples_stage_change_probs_and_per_sku_initial_stages():
    """Stage-change probabilities and per-SKU initial stages vary by seed.

    The caller's catalog stays untouched. The same seed reproduces the
    same draws.
    """
    catalog = _plain_catalog()
    config = TuningConfig(K_active=4, episode_length=5)
    tables = []
    saw_distinct_stages = False

    for seed in range(12):
        spec = sample_episode(catalog, config, episode_seed=seed)
        again = sample_episode(catalog, config, episode_seed=seed)

        assert all(ware.init_stage is None for ware in catalog)
        assert all(ware.stage_change_probs is None for ware in catalog)

        stages = [ware.init_stage for ware in spec.scenario.catalog]
        probs = spec.scenario.item_lifecycle.default_stage_change_probs
        assert stages == [ware.init_stage for ware in again.scenario.catalog]
        assert probs == again.scenario.item_lifecycle.default_stage_change_probs
        assert set(stages) <= set(CANONICAL_STAGES)
        assert any(float(value) > 0.0 for value in probs.values())
        if len(set(stages)) > 1:
            saw_distinct_stages = True
        tables.append(tuple(sorted((key, float(value)) for key, value in probs.items())))

    assert saw_distinct_stages
    assert len(set(tables)) > 1


def _stage_paths(spec) -> list[tuple[str, ...]]:
    log = Runner(spec.scenario).run()
    series = log["global"]["lifecycle"]
    expected = spec.scenario.n_steps + 1
    return [tuple(series[pid]) for pid in spec.active_subset if len(series[pid]) == expected]


def _change_ticks(path: tuple[str, ...]) -> tuple[int, ...]:
    return tuple(i for i in range(len(path) - 1) if path[i] != path[i + 1])


def test_lifecycle_stages_move_and_two_skus_diverge():
    """In one episode, a stage changes and the two SKUs do not share change ticks.

    Different initial stages alone do not count. Paths are a function of
    the episode seed: a second run matches the first. An explicit
    zero-transition lifecycle stays at maturity.
    """
    catalog = _plain_catalog()
    config = TuningConfig(K_active=2, episode_length=180)
    independent = False

    for seed in range(12):
        spec = sample_episode(catalog, config, episode_seed=seed)
        paths = _stage_paths(spec)
        assert len(paths) == 2
        assert all(set(path) <= set(CANONICAL_STAGES) for path in paths)
        ticks = [_change_ticks(path) for path in paths]
        if any(ticks) and ticks[0] != ticks[1]:
            independent = True

    assert independent

    spec = sample_episode(catalog, config, episode_seed=0)
    assert _stage_paths(spec) == _stage_paths(spec)

    inert = sample_episode(
        catalog,
        config,
        episode_seed=0,
        lifecycle_params=default_lifecycle_params(),
    )
    assert all(set(path) == {"maturity"} for path in _stage_paths(inert))


def test_k_active_is_sampled_across_a_wide_range_and_can_be_pinned():
    """Default episodes draw K from 1 through 30. An explicit K stays fixed.

    The same seed draws the same assortment.
    """
    catalog = _plain_catalog(40)
    config = TuningConfig(episode_length=2)
    sizes = [
        len(sample_episode(catalog, config, episode_seed=seed).active_subset)
        for seed in range(80)
    ]
    assert min(sizes) >= 1
    assert max(sizes) <= 30
    assert min(sizes) <= 5
    assert max(sizes) >= 25

    pinned = TuningConfig(episode_length=2, K_active=4)
    assert len(sample_episode(catalog, pinned, episode_seed=1).active_subset) == 4

    first = sample_episode(catalog, config, episode_seed=3).active_subset
    second = sample_episode(catalog, config, episode_seed=3).active_subset
    assert first == second


def _sample_moments(dist, n: int = 120) -> tuple[float, float]:
    rng = Random(0)
    draws = [float(dist.sample(rng)) for _ in range(n)]
    mean = sum(draws) / n
    var = sum((value - mean) ** 2 for value in draws) / n
    cv = (var ** 0.5) / mean if mean > 0 else 0.0
    return mean, cv


def test_synthetic_catalog_is_large_and_heterogeneous():
    """The default synthetic catalog spans demand, margin, volatility, and lifecycle.

    Spread is measured on the episode the sampler actually builds, including
    draws from the sink demand distributions, not on the generator's comments.
    """
    catalog = make_synthetic_catalog()
    assert 500 <= len(catalog) <= 1000
    assert len(catalog) == TuningConfig().K_catalog

    config = TuningConfig(
        episode_length=1,
        K_active=len(catalog),
        capacity_dist=Constant(1_000),
        balance_dist=Constant(1_000_000),
        init_stock_pct_dist=Constant(0.0),
    )
    spec = sample_episode(catalog, config, episode_seed=1)
    wares = spec.scenario.catalog

    margins = [
        (ware.base_price - ware.unit_cost) / ware.base_price for ware in wares
    ]
    assert min(margins) > 0.0
    assert max(margins) - min(margins) >= 0.5

    sinks = [
        instance.node
        for instance in spec.scenario.nodes
        if isinstance(instance.node, DemandSinkNode)
    ]
    assert len(sinks) == len(catalog)
    moments = [_sample_moments(sink.demand_dist) for sink in sinks]
    means = [mean for mean, _cv in moments]
    cvs = [cv for _mean, cv in moments if cv > 0]
    assert max(means) / min(means) >= 10.0
    assert max(cvs) / min(cvs) >= 3.0

    authored = {
        ware.product_id: float(ware.stage_change_probs["maturity"])
        for ware in catalog
    }
    rates = [float(ware.stage_change_probs["maturity"]) for ware in wares]
    assert len(rates) == len(wares)
    assert max(rates) / min(rates) >= 5.0
    ratios = [
        float(ware.stage_change_probs["maturity"]) / authored[ware.product_id]
        for ware in wares
    ]
    assert max(ratios) - min(ratios) < 1e-9
    scales = {round(ratios[0], 5)}
    for seed in (2, 3, 4):
        other = sample_episode(catalog, config, episode_seed=seed)
        ware = other.scenario.catalog[0]
        scales.add(round(float(ware.stage_change_probs["maturity"]) / authored[ware.product_id], 5))
    assert len(scales) > 1
    assert len({ware.init_stage for ware in wares}) > 1
