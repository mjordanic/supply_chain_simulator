"""Unseen-SKU and unseen-disruption protocols.

Seams, from the issue acceptance criteria:

1. Catalog partition. Training SKUs and evaluation SKUs are disjoint, and
   the transition log's catalog is the training partition.
2. Disruption. Training episodes use the default regime. A harsher regime
   is absent from those episodes and is the unseen-disruption protocol.
3. Tables. In-distribution, unseen-SKU, and unseen-disruption are three
   tables. Each has the floors, the oracle ceiling, and the operational
   columns. They are not one pooled table.
4. Seeds. The published seed range of every protocol is disjoint from the
   others and is written into the log manifest.
5. Gate. A profit table the rollout gate refuses is recorded and not written.
"""

from __future__ import annotations

import json

from src.jepa.eval import harsh_disruption_params, protocol_episodes, published_protocol_ranges
from src.jepa.log import build_log_scenario, collect_transition_log
from src.sim.episode_sampler import default_disruption_params
from src.tuning.config import TuningConfig
from src.tuning.episode import make_synthetic_catalog, partition_catalog


def test_training_skus_and_evaluation_skus_do_not_overlap():
    catalog = make_synthetic_catalog()
    training, evaluation = partition_catalog(catalog)
    training_ids = {ware.product_id for ware in training}
    evaluation_ids = {ware.product_id for ware in evaluation}
    config = TuningConfig()

    assert training_ids.isdisjoint(evaluation_ids)
    assert training_ids | evaluation_ids == {ware.product_id for ware in catalog}
    assert len(training_ids) >= config.K_max_episode
    assert len(evaluation_ids) >= config.K_max_episode


def test_transition_log_catalog_is_the_training_partition():
    config = TuningConfig(episode_length=2, K_active=2, K_catalog=40, delivery_lag=2)
    training, evaluation = partition_catalog(make_synthetic_catalog(config.active_catalog_size()))
    training_ids = {ware.product_id for ware in training}
    evaluation_ids = {ware.product_id for ware in evaluation}
    spec = build_log_scenario(config, 14_000_000)
    logged = {ware.product_id for ware in spec.scenario.catalog}

    assert logged == training_ids
    assert logged.isdisjoint(evaluation_ids)
    assert set(spec.active_subset) <= training_ids

    frame = collect_transition_log(
        n_episodes_each=1,
        episode_length=2,
        k_active=2,
        delivery_lag=2,
        seed_offset=14_000_000,
    )
    published = partition_catalog(make_synthetic_catalog(TuningConfig().active_catalog_size()))
    published_training = {ware.product_id for ware in published[0]}
    published_evaluation = {ware.product_id for ware in published[1]}
    assert set(frame["product_id"]) <= published_training
    assert set(frame["product_id"]).isdisjoint(published_evaluation)


def test_harsher_disruption_is_held_out_of_training_episodes():
    mild = default_disruption_params().to_dict()
    harsh = harsh_disruption_params().to_dict()
    assert harsh["event_prob"] > mild["event_prob"]
    assert harsh["severity"]["value"] > mild["severity"]["value"]
    assert harsh["duration"]["value"] > mild["duration"]["value"]
    assert set(mild["types"]) < set(harsh["types"])

    config = TuningConfig(episode_length=4, K_active=2, K_catalog=40, delivery_lag=2)
    training, evaluation = partition_catalog(make_synthetic_catalog(config.K_catalog))
    training_ids = {ware.product_id for ware in training}
    evaluation_ids = {ware.product_id for ware in evaluation}
    episodes = protocol_episodes(config, n_seeds=2)
    log_spec = build_log_scenario(config, 14_000_000)

    assert log_spec.scenario.disruption.to_dict() == mild
    for spec in episodes["in_distribution"] + episodes["unseen_sku"]:
        assert spec.scenario.disruption.to_dict() == mild
    for spec in episodes["unseen_disruption"]:
        assert spec.scenario.disruption.to_dict() == harsh
        assert {ware.product_id for ware in spec.scenario.catalog} == training_ids
    for spec in episodes["unseen_sku"]:
        assert {ware.product_id for ware in spec.scenario.catalog} == evaluation_ids
        assert set(spec.active_subset).isdisjoint(training_ids)
    for spec in episodes["in_distribution"]:
        assert {ware.product_id for ware in spec.scenario.catalog} == training_ids
        assert set(spec.active_subset).isdisjoint(evaluation_ids)


def _spans_are_disjoint(spans: list[tuple[int, int]]) -> bool:
    for index, (left_lo, left_hi) in enumerate(spans):
        for right_lo, right_hi in spans[index + 1 :]:
            if not (left_hi < right_lo or right_hi < left_lo):
                return False
    return True


def test_protocol_seed_ranges_are_disjoint_and_recorded_in_the_manifest(tmp_path):
    from src.jepa.log import EPISODES_EACH, LOG_SEED_OFFSET, behavior_schedule

    published = published_protocol_ranges()
    train_seeds = [seed for _behavior, seed in behavior_schedule(EPISODES_EACH, LOG_SEED_OFFSET)]
    published_spans = [(min(train_seeds), max(train_seeds))]
    published_spans.extend(
        (int(row["seed_min"]), int(row["seed_max"])) for row in published.values()
    )
    assert _spans_are_disjoint(published_spans)
    assert published["in_distribution"]["catalog"] == "training"
    assert published["unseen_sku"]["catalog"] == "evaluation"
    assert published["unseen_disruption"]["disruption"] == "harsh"
    assert published["in_distribution"]["disruption"] == "default"

    path = tmp_path / "transitions.parquet"
    collect_transition_log(
        n_episodes_each=1,
        episode_length=2,
        k_active=2,
        delivery_lag=2,
        seed_offset=LOG_SEED_OFFSET,
        path=path,
    )
    manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    protocols = manifest["protocols"]
    assert protocols["training"]["seed_min"] == manifest["seed_min"]
    assert protocols["training"]["seed_max"] == manifest["seed_max"]
    assert protocols["training"]["disruption"] == "default"
    for name, row in published.items():
        assert protocols[name] == row
    recorded = [
        (int(protocols[name]["seed_min"]), int(protocols[name]["seed_max"]))
        for name in ("training", "in_distribution", "unseen_sku", "unseen_disruption")
    ]
    assert _spans_are_disjoint(recorded)
    assert manifest["harsh_disruption"]["event_prob"] > manifest["disruption"]["event_prob"]
    assert manifest["harsh_disruption"]["severity"]["value"] > manifest["disruption"]["severity"]["value"]
    assert manifest["harsh_disruption"]["duration"]["value"] > manifest["disruption"]["duration"]["value"]
