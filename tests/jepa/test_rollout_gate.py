"""Planning-horizon unroll and the publish gate.

Seams, from the issue acceptance criteria:

1. Unroll. The open-loop loss trains for ``model_unroll_steps``. That
   length defaults to the planning horizon and is not ``delivery_lag``.
2. Report. Held-out rollout R² is one number per step from one step
   through that horizon, for the latent model, the direct predictor,
   and ordinary least squares on the same inputs.
3. Gate. A profit table is written only when the latent curve beats
   that least-squares baseline. An undertrained fit does not write it.
4. Budget. The curve that clears the gate is a multi-epoch fit on a
   reduced episode count, not ``epochs=1``.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from src.jepa.dynamics import (
    ARRIVAL_AGE_SLOTS,
    FIT_BATCH_SIZE,
    LatentDynamics,
    RolloutAccuracy,
    held_out_rollout_accuracy,
    rollout_beats_baseline,
)
from src.jepa.eval import (
    ComparisonTable,
    RolloutGateError,
    SyntheticComparison,
    publish_synthetic_comparison,
    write_gated_profit_table,
)
from src.jepa.plan import (
    DEFAULT_HORIZON_MULTIPLE,
    fit_direct,
    fit_latent,
    load_planners,
    save_planners,
)


def _frame(*, episodes: int, ticks: int, lag: int) -> pd.DataFrame:
    rows = []
    for episode in range(episodes):
        for tick in range(ticks):
            rows.append(
                {
                    "episode_seed": episode,
                    "product_id": "A",
                    "tick": tick,
                    "on_hand": float(tick % 7),
                    "in_transit": [float((tick + age) % 3) for age in range(10)],
                    "sales_window": [float((tick + offset) % 4) for offset in range(3)],
                    "cash": 80.0,
                    "capacity": 40,
                    "filled_qty": float((tick + 1) % 6),
                    "next_on_hand": float((tick + 2) % 7),
                    "next_in_transit": [float((tick + age) % 3) for age in range(10)],
                    "next_sales": float(tick % 3),
                    "delivery_lag": lag,
                }
            )
    return pd.DataFrame(rows)


def test_open_loop_unroll_trains_for_model_unroll_steps_not_the_lead_time(tmp_path: Path):
    lag = 2
    unroll = 5
    ticks = 20
    frame = _frame(episodes=1, ticks=ticks, lag=lag)
    model = LatentDynamics(delivery_lag=lag, model_unroll_steps=unroll)
    model.fit(frame, epochs=1, seed=0, artifact_dir=tmp_path)
    import json

    curve = json.loads((tmp_path / "latent.json").read_text(encoding="utf-8"))["curve"]
    examples = ticks - unroll
    assert examples != ticks - lag
    assert curve[0]["steps"] == math.ceil(examples / FIT_BATCH_SIZE)


def test_default_unroll_is_the_planning_horizon():
    lag = 3
    model = LatentDynamics(delivery_lag=lag)
    assert model.model_unroll_steps == DEFAULT_HORIZON_MULTIPLE * lag
    assert model.model_unroll_steps != lag


def test_held_out_rollout_accuracy_covers_each_step_of_the_horizon(tmp_path: Path):
    lag = 2
    unroll = 4
    frame = _frame(episodes=6, ticks=unroll + 8, lag=lag)
    latent = fit_latent(
        frame,
        delivery_lag=lag,
        model_unroll_steps=unroll,
        epochs=1,
        seed=0,
        artifact_dir=tmp_path / "latent",
    )
    direct = fit_direct(
        frame,
        delivery_lag=lag,
        model_unroll_steps=unroll,
        epochs=1,
        seed=0,
        artifact_dir=tmp_path / "direct",
    )
    report = held_out_rollout_accuracy(latent.dynamics, direct.dynamics, frame)
    assert list(report.steps) == list(range(1, unroll + 1))
    assert len(report.latent) == len(report.direct) == len(report.ols) == unroll
    assert all(math.isfinite(value) for value in report.latent)
    assert all(math.isfinite(value) for value in report.direct)
    assert all(math.isfinite(value) for value in report.ols)


def _linear_frame(*, episodes: int = 8, ticks: int = 16, lag: int = 2) -> pd.DataFrame:
    """On-hand and sales are linear in the observation, so least squares fits them."""
    rows = []
    for episode in range(episodes):
        on_hand = float(2 + episode)
        profile = [0.0] * ARRIVAL_AGE_SLOTS
        window = [0.0] * 4
        for tick in range(ticks):
            filled = float(1 + (tick + episode) % 4)
            next_on_hand = 0.4 * on_hand + filled
            next_sales = 0.2 * on_hand + 0.1 * filled
            next_profile = [0.0] * ARRIVAL_AGE_SLOTS
            next_profile[1] = filled
            rows.append(
                {
                    "episode_seed": episode,
                    "product_id": "A",
                    "tick": tick,
                    "on_hand": on_hand,
                    "in_transit": list(profile),
                    "sales_window": list(window),
                    "cash": 40.0,
                    "capacity": 20.0,
                    "filled_qty": filled,
                    "next_on_hand": next_on_hand,
                    "next_in_transit": next_profile,
                    "next_sales": next_sales,
                    "delivery_lag": lag,
                }
            )
            on_hand = next_on_hand
            profile = next_profile
            window = (window + [next_sales])[-4:]
    return pd.DataFrame(rows)


def test_undertrained_model_does_not_publish_a_profit_table(tmp_path: Path):
    lag = 2
    frame = _linear_frame(lag=lag)
    latent = fit_latent(
        frame,
        delivery_lag=lag,
        model_unroll_steps=4,
        epochs=1,
        seed=0,
        artifact_dir=tmp_path / "latent",
    )
    direct = fit_direct(
        frame,
        delivery_lag=lag,
        model_unroll_steps=4,
        epochs=1,
        seed=0,
        artifact_dir=tmp_path / "direct",
    )
    with pytest.raises(RolloutGateError):
        publish_synthetic_comparison(latent, direct, frame, tmp_path / "out")
    assert not (tmp_path / "out" / "in_distribution.parquet").exists()
    assert not (tmp_path / "out" / "retarget.parquet").exists()
    assert (tmp_path / "out" / "rollout_accuracy.json").is_file()


def test_real_budget_on_a_reduced_log_scores_the_planning_horizon(tmp_path: Path):
    """The gate is a multi-epoch fit, not ``epochs=1``, on fewer episodes than the study."""
    import json

    from src.jepa.log import collect_transition_log

    lag = 2
    horizon = DEFAULT_HORIZON_MULTIPLE * lag
    epochs = 6
    frame = collect_transition_log(
        n_episodes_each=1,
        episode_length=horizon + 6,
        k_active=1,
        delivery_lag=lag,
        seed_offset=14_000_000,
    )
    latent = fit_latent(
        frame,
        delivery_lag=lag,
        epochs=epochs,
        seed=0,
        artifact_dir=tmp_path / "latent",
    )
    direct = fit_direct(
        frame,
        delivery_lag=lag,
        epochs=epochs,
        seed=0,
        artifact_dir=tmp_path / "direct",
    )
    assert latent.dynamics.model_unroll_steps == horizon
    assert horizon != lag
    report = held_out_rollout_accuracy(latent.dynamics, direct.dynamics, frame)
    assert list(report.steps) == list(range(1, horizon + 1))
    assert all(math.isfinite(value) for value in (*report.latent, *report.ols))
    curve = json.loads((tmp_path / "latent" / "latent.json").read_text(encoding="utf-8"))
    assert len(curve["curve"]) > 1
    # This width still loses to least squares on a reduced log. The table stays
    # withheld; a larger model is a later ticket. The budget is what makes the
    # refusal a measurement rather than an epochs=1 shape check.
    assert not rollout_beats_baseline(report)
    with pytest.raises(RolloutGateError):
        publish_synthetic_comparison(latent, direct, frame, tmp_path / "out")
    assert not (tmp_path / "out" / "in_distribution.parquet").exists()


def test_passing_rollout_accuracy_writes_the_profit_table(tmp_path: Path):
    accuracy = RolloutAccuracy(
        steps=(1, 2),
        latent=(0.6, 0.5),
        direct=(0.4, 0.3),
        ols=(0.2, 0.1),
    )
    result = SyntheticComparison(
        in_distribution=ComparisonTable(
            holding_rate=0.01,
            seeds=(1,),
            per_seed={"latent": (0.2,), "anchor": (0.1,)},
        ),
        retarget=ComparisonTable(
            holding_rate=0.05,
            seeds=(1,),
            per_seed={"latent": (0.2,), "anchor": (0.0,)},
        ),
        in_distribution_passes=True,
        retarget_passes=True,
        in_distribution_paired_interval=(0.0, 0.0),
    )
    write_gated_profit_table(result, tmp_path, accuracy)
    assert (tmp_path / "in_distribution.parquet").is_file()
    assert (tmp_path / "retarget.parquet").is_file()


def test_checkpoint_keeps_its_own_unroll_length(tmp_path: Path):
    lag = 2
    frame = _frame(episodes=2, ticks=8, lag=lag)
    latent = fit_latent(
        frame,
        delivery_lag=lag,
        model_unroll_steps=5,
        epochs=1,
        seed=0,
        artifact_dir=tmp_path / "fit",
    )
    direct = fit_direct(
        frame,
        delivery_lag=lag,
        model_unroll_steps=5,
        epochs=1,
        seed=0,
        artifact_dir=tmp_path / "fit-direct",
    )
    save_planners(latent, direct, tmp_path / "ckpt")
    loaded, _direct = load_planners(tmp_path / "ckpt")
    assert loaded.dynamics.model_unroll_steps == 5
    assert loaded.dynamics.model_unroll_steps != lag

