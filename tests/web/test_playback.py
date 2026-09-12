"""Playback frames from a Run Log — public-behavior tests."""

from __future__ import annotations

from pathlib import Path

from src.sim.setup_io import load_setup
from src.web.compare import apply_knobs, run_crn_overlay
from src.web.dag import animated_dag, dag_figure
from src.web.playback import playback_frames

_CHAIN = Path(__file__).resolve().parent.parent.parent / "setups" / "three_node_chain"


def test_playback_frames_cover_every_tick_with_nodes_purchases_rejections():
    scenario = apply_knobs(
        load_setup(_CHAIN),
        world_seed=42,
        n_steps=8,
        demand_mean=10.0,
        cycle_amp=0.03,
        disruption=False,
    )
    log = run_crn_overlay(
        scenario,
        ["order_up_to"],
        {"cover_horizon_ticks": 10, "Q": 30, "review_interval": 7},
    )["order_up_to"]

    frames = playback_frames(log, scenario)
    assert len(frames) == 8
    first = frames[0]
    assert first["tick"] == log["ticks"][0]["tick"]
    assert set(first["nodes"]) == {"factory-1", "shop-1", "sink-1"}
    shop = first["nodes"]["shop-1"]
    assert "cash" in shop
    assert "inventory" in shop
    assert shop["node_type"] == "intermediate"
    assert isinstance(first["purchases"], list)
    assert isinstance(first["rejections"], int)
    assert first["rejections"] >= 0
    # Factory inventory is the _total snapshot, not an empty dict.
    assert first["nodes"]["factory-1"]["inventory"] >= 0


def test_dag_figure_labels_shop_1_and_draws_edges():
    scenario = apply_knobs(
        load_setup(_CHAIN),
        world_seed=42,
        n_steps=4,
        demand_mean=10.0,
        cycle_amp=0.03,
        disruption=False,
    )
    log = run_crn_overlay(
        scenario,
        ["order_up_to"],
        {"cover_horizon_ticks": 10, "Q": 30, "review_interval": 7},
    )["order_up_to"]
    frame = playback_frames(log, scenario)[0]
    fig = dag_figure(scenario, frame)
    labels = []
    for trace in fig.data:
        if trace.text is None:
            continue
        if isinstance(trace.text, str):
            labels.append(trace.text)
        else:
            labels.extend(str(t) for t in trace.text)
    blob = " ".join(labels)
    assert "shop-1" in blob
    assert "factory-1" in blob
    assert "sink-1" in blob
    # One scatter per edge (line) plus node markers — at least 2 edges in the chain.
    n_line = sum(1 for t in fig.data if getattr(t, "mode", None) and "lines" in t.mode)
    assert n_line >= 2
    legend_names = [t.name for t in fig.data if t.showlegend]
    assert "Factory (produces)" in legend_names
    assert "Order filled this tick" in legend_names


def test_animated_dag_has_one_frame_per_tick_and_skip_button():
    scenario = apply_knobs(
        load_setup(_CHAIN),
        world_seed=42,
        n_steps=5,
        demand_mean=10.0,
        cycle_amp=0.03,
        disruption=False,
    )
    log = run_crn_overlay(
        scenario,
        ["order_up_to"],
        {"cover_horizon_ticks": 10, "Q": 30, "review_interval": 7},
    )["order_up_to"]
    frames = playback_frames(log, scenario)
    fig = animated_dag(scenario, frames)
    assert len(fig.frames) == 5
    labels = [btn["label"] for menu in fig.layout.updatemenus for btn in menu.buttons]
    assert "Play" in labels
    assert "Skip to end" in labels
