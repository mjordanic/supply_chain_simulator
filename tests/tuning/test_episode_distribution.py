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


def _factories(spec, pid: str):
    from src.sim.node import FactoryNode

    return [
        instance.node
        for instance in spec.scenario.nodes
        if isinstance(instance.node, FactoryNode) and instance.node.produces_product_id == pid
    ]


def _lead(spec, factory_id: str) -> int:
    for edge in spec.scenario.edges:
        if edge.supplier_id == factory_id and edge.buyer_id == "S":
            return int(edge.default_lead_time)
    raise AssertionError(factory_id)


def _rate(node) -> int:
    cap = node.capacity_per_tick
    return int(cap.value if hasattr(cap, "value") else cap)


def test_each_sku_has_several_suppliers_with_markup_and_a_speed_tradeoff():
    """One SKU has several factories, each listing above unit cost.

    The shortest lead time is the dearest offer. The same seed repeats
    the offers. A pinned nominal lag is the fastest supplier's lead time.
    """
    catalog = _plain_catalog()
    config = TuningConfig(
        K_active=2,
        episode_length=4,
        delivery_lag=3,
        capacity_dist=Constant(5_000),
    )
    spec = sample_episode(catalog, config, episode_seed=4)
    again = sample_episode(catalog, config, episode_seed=4)
    assert spec.delivery_lag == 3

    for pid in spec.active_subset:
        factories = _factories(spec, pid)
        assert len(factories) >= 2
        offers = [
            (_lead(spec, node.id), node.list_price, node.unit_cost, _rate(node), node.inventory)
            for node in factories
        ]
        assert offers == [
            (_lead(again, node.id), node.list_price, node.unit_cost, _rate(node), node.inventory)
            for node in _factories(again, pid)
        ]
        assert all(price > cost for _lead_time, price, cost, _prod, _opening in offers)
        leads = [lead for lead, *_rest in offers]
        assert len(set(leads)) == len(leads)
        fastest = min(offers, key=lambda item: item[0])
        dearest = max(offers, key=lambda item: item[1])
        assert fastest[1] == dearest[1]
        assert fastest[0] == spec.delivery_lag


def test_supplier_output_tracks_demand_rather_than_node_capacity():
    """Two catalogs that differ only in demand scale differ in factory output.

    Both episodes pin the same shop capacity. Factory production and
    opening stock are not multiples of that capacity.
    """
    def catalog(mean: float):
        return load_catalog(
            [
                {
                    "name": "Widget",
                    "category": "General",
                    "related_products": [],
                    "base_price": 12.0,
                    "unit_cost": 5.0,
                    "seasonality": "all_season",
                    "demand_mean": mean,
                    "demand_cv": 0.1,
                }
            ]
        )

    config = TuningConfig(
        K_active=1,
        episode_length=8,
        delivery_lag=3,
        capacity_dist=Constant(5_000),
        balance_dist=Constant(50_000),
    )
    low = sample_episode(catalog(80.0), config, episode_seed=2)
    high = sample_episode(catalog(320.0), config, episode_seed=2)
    pid = low.active_subset[0]
    low_prod = sum(_rate(node) for node in _factories(low, pid))
    high_prod = sum(_rate(node) for node in _factories(high, pid))
    low_open = sum(node.inventory for node in _factories(low, pid))
    high_open = sum(node.inventory for node in _factories(high, pid))
    assert low_prod > 0
    ratio = high_prod / low_prod
    assert abs(ratio - 4.0) / 4.0 < 0.15, ratio
    if low_open > 0:
        open_ratio = high_open / low_open
        assert abs(open_ratio - 4.0) / 4.0 < 0.15, open_ratio
    assert low_prod != 5_000 * 4
    assert low_open != 5_000 * 2
    assert high_prod != 5_000 * 4


def test_delivery_lag_is_sampled_per_episode_and_can_be_pinned():
    """Unpinned episodes draw a nominal lag. An explicit lag stays fixed."""
    catalog = _plain_catalog(8)
    config = TuningConfig(K_active=1, episode_length=2)
    lags = [sample_episode(catalog, config, episode_seed=seed).delivery_lag for seed in range(40)]
    assert min(lags) == 2
    assert max(lags) == 6
    assert len(set(lags)) >= 4

    pinned = TuningConfig(K_active=1, episode_length=2, delivery_lag=5)
    spec = sample_episode(catalog, pinned, episode_seed=1)
    assert spec.delivery_lag == 5
    leads = [_lead(spec, node.id) for node in _factories(spec, spec.active_subset[0])]
    assert min(leads) == 5
    assert len(set(leads)) == len(leads)


def test_upstream_stockouts_happen_and_fill_rate_varies():
    """A demand-sized order is not a guaranteed arrival.

    The plain catalog's expected demand is 5 units per tick. The buyer
    spreads that 5 across every supplier of the SKU and does not trim
    the request to the published offer. Shop capacity and cash are large,
    and the lifecycle is inert, so a partial fill is upstream. The run
    log records the realised fill rate, and that rate is not the same
    on every seed.
    """
    from src.sim.policy import IntermediatePolicy, StaticFactoryPolicy

    class _DemandOrder(IntermediatePolicy):
        def decide(self, obs, table):
            inventory = dict(obs.get("inventory") or {})
            direct = obs.get("direct_supplier_ids")
            order = {}
            for pid in inventory:
                offers = [
                    (supplier_id, offer)
                    for supplier_id, offer in table.snapshot_for_buyer(pid)
                    if direct is None or supplier_id in direct
                ]
                if not offers:
                    continue
                ranked = sorted(offers, key=lambda item: item[1].list_price)
                base, remainder = divmod(5, len(ranked))
                lines = []
                for index, (supplier_id, _offer) in enumerate(ranked):
                    qty = base + (1 if index < remainder else 0)
                    if qty > 0:
                        lines.append((supplier_id, qty))
                if lines:
                    order[pid] = lines
            prices = dict(obs.get("list_prices") or {})
            return {
                "order": order,
                "list_price": {pid: float(prices.get(pid, 0.0)) for pid in inventory},
                "min_order_imposed": {pid: 0 for pid in inventory},
            }

    catalog = _plain_catalog()
    config = TuningConfig(
        K_active=2,
        episode_length=36,
        delivery_lag=3,
        capacity_dist=Constant(100_000),
        balance_dist=Constant(1_000_000),
        init_stock_pct_dist=Constant(0.0),
    )
    stockouts = 0
    rates: list[float] = []
    paid_above_cost = False

    for seed in range(16):
        spec = sample_episode(
            catalog,
            config,
            episode_seed=seed,
            lifecycle_params=default_lifecycle_params(),
        )
        costs = {
            instance.node.id: float(instance.node.unit_cost)
            for instance in spec.scenario.nodes
            if instance.node.id.startswith("F_")
        }
        overrides = {
            node.id: StaticFactoryPolicy(
                capacity_per_tick=_rate(node),
                unit_cost=float(node.unit_cost),
            )
            for instance in spec.scenario.nodes
            if instance.node.id.startswith("F_")
            for node in [instance.node]
        }
        overrides["S"] = _DemandOrder(policy_seed=0)
        log = Runner(spec.scenario, policy_overrides=overrides).run()
        rate = log["global"]["supplier_fill_rate"]
        assert rate is not None
        rates.append(float(rate))
        short = False
        for tick in log["ticks"]:
            for flow in tick["node_flows"]:
                if str(flow["node_id"]).startswith("F_") and int(flow["demand"]) > int(flow["sales"]):
                    short = True
            for purchase in tick["purchases"]:
                supplier_id = str(purchase["supplier_id"])
                qty = int(purchase["qty_filled"])
                if supplier_id in costs and qty > 0:
                    unit = float(purchase["cash_paid"]) / qty
                    if unit > costs[supplier_id] + 1e-9:
                        paid_above_cost = True
        if short:
            stockouts += 1

    fraction = stockouts / 16
    assert fraction >= 0.20, (fraction, rates)
    assert max(rates) - min(rates) >= 0.10, rates
    assert paid_above_cost

