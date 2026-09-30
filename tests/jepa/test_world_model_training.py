"""World-model training.

Seams, from the issue acceptance criteria:

1. Epoch. One epoch is a shuffled pass over the training rows. A pass
   that does not fit in one batch takes more than one gradient step.
2. Holdout. Validation episodes and training episodes are disjoint, and
   rows from a held-out episode are not in the training steps.
3. Early stopping. Fitting halts before the epoch budget when the holdout
   loss stops improving. The curve records that holdout.
4. Target encoder. After a fit that runs its budget, the EMA target has
   moved with the online encoder. The decay matches the step count.
5. Direct predictor. The same epoch, holdout, and early-stopping contract.
6. Probe target. Next in-transit is the arrival-age profile. A fit places
   units on the true age when every row has the same pipeline sum.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest
import torch

from src.jepa.dynamics import (
    ARRIVAL_AGE_SLOTS,
    EMA_INIT_SHARE,
    FIT_ARTIFACT_DIR,
    FIT_BATCH_SIZE,
    DirectDynamics,
    LatentDynamics,
    sku_features,
)


def _frame(*, episodes: int, ticks: int, lag: int) -> pd.DataFrame:
    rows = []
    for episode in range(episodes):
        for tick in range(ticks):
            profile = [0.0] * ARRIVAL_AGE_SLOTS
            profile[1] = float((tick + episode) % 5)
            rows.append(
                {
                    "episode_seed": episode,
                    "product_id": "A",
                    "tick": tick,
                    "on_hand": float(tick % 7),
                    "in_transit": list(profile),
                    "sales_window": [float((tick + offset) % 4) for offset in range(3)],
                    "cash": 80.0,
                    "capacity": 40,
                    "filled_qty": float((tick + 1) % 6),
                    "next_on_hand": float((tick + 2) % 7),
                    "next_in_transit": list(profile),
                    "next_sales": float(tick % 3),
                    "delivery_lag": lag,
                }
            )
    return pd.DataFrame(rows)


def _model(kind: str, lag: int) -> DirectDynamics | LatentDynamics:
    if kind == "direct":
        return DirectDynamics(delivery_lag=lag)
    if kind == "latent":
        return LatentDynamics(delivery_lag=lag)
    raise ValueError(kind)


def _read(directory: Path, kind: str) -> dict:
    name = "direct.json" if kind == "direct" else "latent.json"
    return json.loads((directory / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("kind", ["direct", "latent"])
def test_one_epoch_is_a_pass_over_minibatches(tmp_path: Path, kind: str):
    lag = 2
    horizon = _unroll(kind, lag)
    ticks = FIT_BATCH_SIZE + horizon + 2
    examples = ticks - horizon
    frame = _frame(episodes=1, ticks=ticks, lag=lag)
    _model(kind, lag).fit(frame, epochs=1, seed=0, artifact_dir=tmp_path)
    curve = _read(tmp_path, kind)["curve"]
    assert [row["epoch"] for row in curve] == [1]
    assert curve[0]["steps"] == math.ceil(examples / FIT_BATCH_SIZE)
    assert curve[0]["steps"] > 1
    assert curve[0]["val_loss"] is None


def _state(module: DirectDynamics | LatentDynamics) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


def _same(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left)


@pytest.mark.parametrize("kind", ["direct", "latent"])
def test_shuffle_follows_the_fit_seed(tmp_path: Path, kind: str):
    lag = 2
    frame = _frame(episodes=1, ticks=FIT_BATCH_SIZE + lag + 3, lag=lag)
    torch.manual_seed(0)
    first = _model(kind, lag)
    second = _model(kind, lag)
    second.load_state_dict(first.state_dict())
    third = _model(kind, lag)
    third.load_state_dict(first.state_dict())
    first.fit(frame, epochs=2, seed=0, artifact_dir=tmp_path / "a")
    second.fit(frame, epochs=2, seed=0, artifact_dir=tmp_path / "b")
    third.fit(frame, epochs=2, seed=1, artifact_dir=tmp_path / "c")
    assert _same(_state(first), _state(second))
    assert not _same(_state(first), _state(third))


def _unroll(kind: str, lag: int) -> int:
    """Direct training is one step. The latent unroll is its own length."""
    if kind == "direct":
        return 1
    return LatentDynamics(delivery_lag=lag).model_unroll_steps


def _examples(frame: pd.DataFrame, episodes: list[int], *, kind: str, lag: int) -> int:
    horizon = _unroll(kind, lag)
    subset = frame[frame["episode_seed"].isin(episodes)]
    total = 0
    for _, group in subset.groupby(["episode_seed", "product_id"], sort=False):
        n_rows = len(group)
        if n_rows >= horizon + 1:
            total += n_rows - horizon
    return total


@pytest.mark.parametrize("kind", ["direct", "latent"])
def test_validation_split_holds_out_whole_episodes(tmp_path: Path, kind: str):
    lag = 2
    # Eight training episodes of this many windows fit in one batch.
    # Ten episodes do not, so holding episodes out changes the step count.
    windows = 28
    ticks = _unroll(kind, lag) + windows
    episodes = 10
    frame = _frame(episodes=episodes, ticks=ticks, lag=lag)
    _model(kind, lag).fit(frame, epochs=1, seed=0, artifact_dir=tmp_path)
    artifact = _read(tmp_path, kind)
    train = set(artifact["train_episodes"])
    held = set(artifact["val_episodes"])
    assert train.isdisjoint(held)
    assert train | held == set(range(episodes))
    assert held
    train_steps = math.ceil(_examples(frame, sorted(train), kind=kind, lag=lag) / FIT_BATCH_SIZE)
    all_steps = math.ceil(_examples(frame, list(range(episodes)), kind=kind, lag=lag) / FIT_BATCH_SIZE)
    assert artifact["curve"][0]["steps"] == train_steps
    assert train_steps < all_steps
    assert isinstance(artifact["curve"][0]["val_loss"], float)
    if kind == "latent":
        probes = json.loads((tmp_path / "probes.json").read_text(encoding="utf-8"))
        assert probes["train_episodes"] == artifact["train_episodes"]
        assert probes["val_episodes"] == artifact["val_episodes"]
        assert probes["curve"][0]["steps"] == math.ceil(
            _examples(frame, sorted(train), kind="direct", lag=lag) / FIT_BATCH_SIZE
        )


def _pattern(tick: int, scale: float) -> tuple[float, list[float], float]:
    profile = [scale * float((tick + age) % 5) for age in range(ARRIVAL_AGE_SLOTS)]
    return scale * float(tick % 6), profile, scale * float((tick + 2) % 4)


def _conflict_frame(train_ids: set[int], val_ids: set[int], *, ticks: int, lag: int) -> pd.DataFrame:
    rows = []
    for episode in sorted(train_ids | val_ids):
        scale = -1.0 if episode in val_ids else 1.0
        for tick in range(ticks):
            on_hand, profile, sales = _pattern(tick, scale)
            rows.append(
                {
                    "episode_seed": episode,
                    "product_id": "A",
                    "tick": tick,
                    "on_hand": float(tick % 5),
                    "in_transit": [float((tick + age) % 3) for age in range(ARRIVAL_AGE_SLOTS)],
                    "sales_window": [1.0, float(tick % 3)],
                    "cash": 25.0,
                    "capacity": 40,
                    "filled_qty": float(tick % 5),
                    "next_on_hand": on_hand,
                    "next_in_transit": profile,
                    "next_sales": sales,
                    "delivery_lag": lag,
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("kind", ["direct", "probes"])
def test_early_stopping_halts_when_the_holdout_stops_improving(tmp_path: Path, kind: str):
    lag = 2
    neutral = _frame(episodes=4, ticks=8, lag=lag)
    peek = tmp_path / "peek"
    _model("direct" if kind == "direct" else "latent", lag).fit(
        neutral,
        epochs=1,
        seed=0,
        artifact_dir=peek,
    )
    peeked = _read(peek, "direct" if kind == "direct" else "latent")
    train_ids = set(peeked["train_episodes"])
    val_ids = set(peeked["val_episodes"])
    frame = _conflict_frame(train_ids, val_ids, ticks=12, lag=lag)
    fitted = tmp_path / "fitted"
    model = DirectDynamics(delivery_lag=lag) if kind == "direct" else LatentDynamics(delivery_lag=lag)
    model.fit(frame, epochs=8, seed=0, lr=1e-2, patience=1, artifact_dir=fitted)
    name = "direct.json" if kind == "direct" else "probes.json"
    artifact = json.loads((fitted / name).read_text(encoding="utf-8"))
    assert artifact["stopped_early"] is True
    assert len(artifact["curve"]) == 2
    assert artifact["best_epoch"] == 1
    assert artifact["curve"][1]["val_loss"] > artifact["curve"][0]["val_loss"]


def _flat(module: torch.nn.Module) -> torch.Tensor:
    return torch.cat([parameter.detach().reshape(-1) for parameter in module.parameters()])


def test_target_encoder_leaves_its_initialisation(tmp_path: Path):
    """One pass walks the target off init, including the epoch early stopping keeps."""
    lag = 2
    frame = _frame(episodes=4, ticks=30, lag=lag)
    torch.manual_seed(0)
    model = LatentDynamics(delivery_lag=lag)
    before = _flat(model.target_encoder)
    online_before = _flat(model.encoder)
    model.fit(frame, epochs=8, seed=0, lr=1e-2, artifact_dir=tmp_path)
    artifact = _read(tmp_path, "latent")
    assert artifact["val_episodes"]
    first_pass = artifact["curve"][0]["steps"]
    assert first_pass >= 1
    assert artifact["ema_tau"] ** first_pass == pytest.approx(EMA_INIT_SHARE, rel=1e-4)
    online_move = torch.linalg.vector_norm(_flat(model.encoder) - online_before)
    target_move = torch.linalg.vector_norm(_flat(model.target_encoder) - before)
    assert float(online_move) > 1e-3
    assert float(target_move) > 0.25 * float(online_move)


def test_direct_fit_places_units_on_the_true_arrival_age(tmp_path: Path):
    lag = 2
    rows = []
    for tick in range(80):
        action = 1 + (tick % 4)
        profile = [0.0] * ARRIVAL_AGE_SLOTS
        profile[action] = 10.0
        rows.append(
            {
                "episode_seed": 0,
                "product_id": "A",
                "tick": tick,
                "on_hand": 1.0,
                "in_transit": [0.0] * ARRIVAL_AGE_SLOTS,
                "sales_window": [1.0, 1.0],
                "cash": 10.0,
                "capacity": 20,
                "filled_qty": float(action),
                "next_on_hand": 1.0,
                "next_in_transit": profile,
                "next_sales": 1.0,
                "delivery_lag": lag,
            }
        )
    torch.manual_seed(0)
    model = DirectDynamics(delivery_lag=lag)
    model.fit(pd.DataFrame(rows), epochs=80, seed=0, lr=1e-2, artifact_dir=tmp_path)
    features = sku_features(
        on_hand=1.0,
        in_transit=[0.0] * ARRIVAL_AGE_SLOTS,
        sales_window=[1.0, 1.0],
        cash=10.0,
        capacity=20,
        delivery_lag=lag,
    )
    for action in (1, 2, 3, 4):
        _on_hand, profile, _sales = model.predict_next([features], [float(action)])[0]
        assert len(profile) == ARRIVAL_AGE_SLOTS
        rest = sum(profile) - profile[action]
        assert profile[action] > rest


def test_latent_decode_is_the_arrival_age_profile(tmp_path: Path):
    frame = _frame(episodes=1, ticks=6, lag=2)
    model = LatentDynamics(delivery_lag=2)
    model.fit(frame, epochs=1, seed=0, artifact_dir=tmp_path)
    features = sku_features(
        on_hand=1.0,
        in_transit=[0.0, 2.0],
        sales_window=[1.0],
        cash=10.0,
        capacity=20,
        delivery_lag=2,
    )
    _on_hand, profile, _sales = model.predict_next([features], [1.0])[0]
    assert len(profile) == ARRIVAL_AGE_SLOTS


def test_fit_artifact_directory_is_under_runs():
    assert FIT_ARTIFACT_DIR.parts == ("runs", "jepa-replenishment", "fit")
