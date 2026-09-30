"""Published model capacity and the scaling curves.

Seams, from the issue acceptance criteria:

1. Size. The published latent model, embedding width and depth together,
   has between half a million and two million parameters. The count
   includes the frozen EMA copy of the encoder.
2. Direct. The published direct predictor is scaled to that trainable
   count, so latent-versus-direct is not a capacity gap.
3. The constructor default stays the reduced width. The publish-gate
   test trains that width and still expects the table to stay closed.
4. Capacity curve. Held-out rollout accuracy, from the planning-horizon
   harness, is reported against parameter count.
5. Data curve. The same accuracy is reported against episode count.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from src.jepa.dynamics import (
    PUBLISHED_DEPTH,
    PUBLISHED_DIRECT_HIDDEN,
    PUBLISHED_EMBED_DIM,
    REDUCED_DEPTH,
    REDUCED_EMBED_DIM,
    DirectDynamics,
    LatentDynamics,
    parameter_count,
    trainable_parameter_count,
)
from src.jepa.plan import data_scaling_curve, model_capacity_curve


def test_published_latent_model_is_between_half_a_million_and_two_million_parameters():
    model = LatentDynamics(
        delivery_lag=3,
        embed_dim=PUBLISHED_EMBED_DIM,
        depth=PUBLISHED_DEPTH,
    )
    count = parameter_count(model)
    assert 500_000 <= count <= 2_000_000


def test_published_direct_predictor_matches_latent_trainable_capacity():
    """Latent-versus-direct is not a wider net against a narrower one."""
    latent = LatentDynamics(
        delivery_lag=3,
        embed_dim=PUBLISHED_EMBED_DIM,
        depth=PUBLISHED_DEPTH,
    )
    direct = DirectDynamics(
        delivery_lag=3,
        hidden_dim=PUBLISHED_DIRECT_HIDDEN,
        depth=PUBLISHED_DEPTH,
    )
    trainable = trainable_parameter_count(latent)
    direct_count = parameter_count(direct)
    ratio = direct_count / trainable
    assert 0.5 <= ratio <= 2.0


def test_default_constructor_stays_the_reduced_width():
    model = LatentDynamics(delivery_lag=3)
    assert model.embed_dim == REDUCED_EMBED_DIM
    assert model.depth == REDUCED_DEPTH
    assert parameter_count(model) < 500_000


def _frame(*, episodes: int, ticks: int, lag: int) -> pd.DataFrame:
    rows = []
    for episode in range(episodes):
        on_hand = float(2 + episode)
        for tick in range(ticks):
            filled = float(1 + (tick + episode) % 4)
            next_on_hand = 0.4 * on_hand + filled
            rows.append(
                {
                    "episode_seed": episode,
                    "product_id": "A",
                    "tick": tick,
                    "on_hand": on_hand,
                    "in_transit": [0.0] * 10,
                    "sales_window": [float((tick + offset) % 3) for offset in range(4)],
                    "cash": 40.0,
                    "capacity": 20.0,
                    "filled_qty": filled,
                    "next_on_hand": next_on_hand,
                    "next_in_transit": [0.0] * 10,
                    "next_sales": 0.2 * on_hand + 0.1 * filled,
                    "delivery_lag": lag,
                }
            )
            on_hand = next_on_hand
    return pd.DataFrame(rows)


def test_capacity_curve_reports_rollout_accuracy_against_parameter_count(tmp_path: Path):
    lag = 2
    frame = _frame(episodes=4, ticks=12, lag=lag)
    curve = model_capacity_curve(
        frame,
        ((4, 2), (8, 2)),
        delivery_lag=lag,
        epochs=1,
        seed=0,
        model_unroll_steps=3,
        artifact_dir=tmp_path,
    )
    assert [point.parameters for point in curve] == sorted(
        point.parameters for point in curve
    )
    assert curve[0].parameters < curve[1].parameters
    for point in curve:
        assert point.parameters == parameter_count(
            LatentDynamics(
                delivery_lag=lag,
                model_unroll_steps=3,
                embed_dim=point.embed_dim,
                depth=point.depth,
            )
        )
        assert list(point.accuracy.steps) == [1, 2, 3]
        assert all(math.isfinite(value) for value in point.accuracy.latent)
        assert all(math.isfinite(value) for value in point.accuracy.direct)
        assert all(math.isfinite(value) for value in point.accuracy.ols)


def test_data_scaling_curve_reports_rollout_accuracy_against_episode_count(tmp_path: Path):
    lag = 2
    frame = _frame(episodes=6, ticks=12, lag=lag)
    curve = data_scaling_curve(
        frame,
        (3, 6),
        embed_dim=4,
        depth=2,
        delivery_lag=lag,
        epochs=1,
        seed=0,
        model_unroll_steps=3,
        artifact_dir=tmp_path,
    )
    assert [point.episodes for point in curve] == [3, 6]
    for point in curve:
        assert list(point.accuracy.steps) == [1, 2, 3]
        assert all(math.isfinite(value) for value in point.accuracy.latent)


def test_fit_seed_fixes_initial_weights_and_the_fit(tmp_path: Path):
    import torch

    from src.jepa.plan import fit_latent

    lag = 2
    frame = _frame(episodes=4, ticks=10, lag=lag)
    first = fit_latent(
        frame,
        delivery_lag=lag,
        epochs=1,
        seed=0,
        embed_dim=4,
        depth=2,
        model_unroll_steps=3,
        artifact_dir=tmp_path / "a",
    )
    torch.randn(32)
    second = fit_latent(
        frame,
        delivery_lag=lag,
        epochs=1,
        seed=0,
        embed_dim=4,
        depth=2,
        model_unroll_steps=3,
        artifact_dir=tmp_path / "b",
    )
    assert torch.equal(
        first.dynamics.encoder.net[0].weight,
        second.dynamics.encoder.net[0].weight,
    )


def test_checkpoint_reloads_the_width_and_depth_it_was_saved_with(tmp_path: Path):
    from src.jepa.plan import fit_direct, fit_latent, load_planners, save_planners

    lag = 2
    frame = _frame(episodes=2, ticks=6, lag=lag)
    latent = fit_latent(
        frame,
        delivery_lag=lag,
        epochs=0,
        seed=0,
        embed_dim=16,
        depth=3,
    )
    direct = fit_direct(
        frame,
        delivery_lag=lag,
        epochs=0,
        seed=0,
        hidden_dim=20,
        depth=3,
    )
    save_planners(latent, direct, tmp_path)
    loaded, loaded_direct = load_planners(tmp_path)
    assert loaded.dynamics.embed_dim == 16
    assert loaded.dynamics.depth == 3
    assert loaded_direct.dynamics.hidden_dim == 20
    assert loaded_direct.dynamics.depth == 3
    assert parameter_count(loaded.dynamics) == parameter_count(latent.dynamics)
    assert parameter_count(loaded_direct.dynamics) == parameter_count(direct.dynamics)


def test_fit_command_writes_the_published_capacity(tmp_path: Path):
    import json

    from src.jepa.commands import main
    from src.jepa.plan import load_planners

    lag = 2
    log = tmp_path / "transitions.parquet"
    _frame(episodes=2, ticks=6, lag=lag).to_parquet(log, index=False)
    destination = tmp_path / "checkpoints"
    main([
        "fit",
        "--log", str(log),
        "--output", str(destination),
        "--epochs", "0",
        "--delivery-lag", str(lag),
    ])
    meta = json.loads((destination / "planner.json").read_text(encoding="utf-8"))
    assert meta["embed_dim"] == PUBLISHED_EMBED_DIM
    assert meta["depth"] == PUBLISHED_DEPTH
    assert meta["direct_hidden"] == PUBLISHED_DIRECT_HIDDEN
    latent, direct = load_planners(destination)
    assert 500_000 <= parameter_count(latent.dynamics) <= 2_000_000
    assert parameter_count(direct.dynamics) == meta["direct_parameters"]


def test_tracked_docs_record_the_published_configuration():
    from src.jepa.commands import SCALE_COMMAND

    count = parameter_count(
        LatentDynamics(
            delivery_lag=3,
            embed_dim=PUBLISHED_EMBED_DIM,
            depth=PUBLISHED_DEPTH,
        )
    )
    rendered = f"{count:,}"
    context = Path("CONTEXT.md").read_text()
    section = context.split("**Latent replenishment planner**", 1)[1].split(
        "**Direct replenishment predictor**", 1
    )[0]
    assert str(PUBLISHED_EMBED_DIM) in section
    assert str(PUBLISHED_DEPTH) in section
    assert rendered in section
    readme = Path("README.md").read_text()
    latent = readme.split("## Latent replenishment", 1)[1].split("\n## ", 1)[0]
    assert SCALE_COMMAND in latent
    assert rendered in latent


def test_scale_command_writes_both_curves(tmp_path: Path):
    import json

    from src.jepa.commands import main

    lag = 2
    log = tmp_path / "transitions.parquet"
    _frame(episodes=6, ticks=12, lag=lag).to_parquet(log, index=False)
    output = tmp_path / "scaling"
    main([
        "scale",
        "--log", str(log),
        "--output", str(output),
        "--epochs", "1",
        "--delivery-lag", str(lag),
        "--sizes", "4x2,8x2",
        "--episodes", "3,6",
        "--embed-dim", "4",
        "--depth", "2",
        "--model-unroll-steps", "3",
    ])
    capacity = json.loads((output / "capacity.json").read_text(encoding="utf-8"))
    data = json.loads((output / "data.json").read_text(encoding="utf-8"))
    counts = [point["parameters"] for point in capacity["points"]]
    assert counts == sorted(counts)
    assert counts[0] < counts[1]
    assert [point["episodes"] for point in data["points"]] == [3, 6]
    assert all(point["latent_mean"] is not None for point in capacity["points"])
    assert all(point["latent_mean"] is not None for point in data["points"])
    assert data["embed_dim"] == 4
    assert data["depth"] == 2
