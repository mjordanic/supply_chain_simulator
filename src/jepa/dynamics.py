"""Per-SKU latent dynamics and the direct quantity predictor.

The encoder reads on-hand, an in-transit arrival-age profile, a sales
window of length ``SALES_WINDOW``, cash, capacity, and the nominal
delivery lead time. That window is the encoder's own knob: it does not
follow the sampled lead time, and neither does the feature dimension.
The Market multiplier, the disruption identity, and the calendar are
not features. An action-conditioned predictor unrolls open-loop for
``model_unroll_steps`` steps. That length is the planning horizon
(``PLANNING_HORIZON_MULTIPLE * delivery_lag``) unless it is set on its
own. The latent loss matches a stopped target
embedding at each step, plus an anti-collapse penalty. Probes read a
stopped embedding and regress next on-hand, the next arrival-age
profile, and sales over the step. Probe losses do not update the encoder.

``fit(epochs=N)`` takes N shuffled minibatch passes over the training
episodes. A fraction of episodes is held out whole. Early stopping
watches that holdout and restores the best epoch. The loss curve is
written under ``runs/jepa-replenishment/fit/``. The EMA decay is set from the
minibatches in one pass, so the epoch early stopping keeps has already
left only ``EMA_INIT_SHARE`` of the target encoder's initial weights.
"""

from __future__ import annotations

import copy
import json
import math
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

# The publish-gate's reduced run. Width 32 and two linear layers are the
# model that still loses to least squares at that budget. The published
# fit uses the wider, deeper constants below.
REDUCED_EMBED_DIM = 32
REDUCED_DEPTH = 2
REDUCED_DIRECT_HIDDEN = 32

# Chosen from the capacity curve at learning rate 1e-4. Width 256 and
# depth 4 is the peak held-out latent rollout inside the half-million to
# two-million band; wider models on that curve scored lower. 1e-3 stops
# at the first epoch on this log, so the published fit does not use it.
PUBLISHED_EMBED_DIM = 256
PUBLISHED_DEPTH = 4
PUBLISHED_LR = 1e-4

FIT_BATCH_SIZE = 256
# The eager path builds every window as a Python list. Past this many log
# rows that list does not fit beside the frame. Smoke logs stay eager.
_EAGER_ROW_LIMIT = 50_000
_ROLLOUT_CHUNK = 8_192
VAL_EPISODE_FRACTION = 0.2
EARLY_STOP_PATIENCE = 5
EMA_INIT_SHARE = 0.05
FIT_ARTIFACT_DIR = Path("runs/jepa-replenishment/fit")

# Four weeks of daily sales. The longest sampled edge is a nominal lag
# of 6 plus the slow supplier's offset of 4, so this sits well above
# the delivery lead time and does not move when that lag is sampled.
SALES_WINDOW = 28

# Age 0 .. lead-1. The longest study edge is 10, so a unit stays in
# the profile until it arrives. Index is age in ticks, not a slot
# counted back from today.
ARRIVAL_AGE_SLOTS = 10

# The planner scores this many lead times. The open-loop unroll
# defaults to that horizon and can be set apart from the lead time.
PLANNING_HORIZON_MULTIPLE = 4


def _fill_ratio(frame: pd.DataFrame) -> float:
    """Share of requested units Allocation actually filled.

    A zero request is not a fill. Missing columns, or a log that never
    ordered, leave the rate at 1 so the planner still imagines a whole arrival.
    """
    if frame.empty or "requested_qty" not in frame.columns or "filled_qty" not in frame.columns:
        return 1.0
    requested = frame["requested_qty"].to_numpy(dtype=float)
    filled = frame["filled_qty"].to_numpy(dtype=float)
    positive = requested > 0
    if not bool(positive.any()):
        return 1.0
    ratio = float(filled[positive].sum() / requested[positive].sum())
    if ratio < 0.0:
        return 0.0
    if ratio > 1.0:
        return 1.0
    return ratio


def feature_dim(sales_window_length: int = SALES_WINDOW) -> int:
    """Width of one encoder vector. Independent of the sampled lead time."""
    return 1 + ARRIVAL_AGE_SLOTS + max(0, int(sales_window_length)) + 3


def target_dim() -> int:
    """Width of the quantity head: on-hand, arrival-age profile, sales."""
    return 2 + ARRIVAL_AGE_SLOTS


def arrival_age_profile(
    lots: Sequence[tuple[int, int, int]],
    tick: int,
    length: int = ARRIVAL_AGE_SLOTS,
) -> list[int]:
    """Units still on the way, indexed by age.

    ``lots`` are ``(order_tick, qty, lead)`` filled on earlier ticks.
    Age 0 is empty at decision time, because ``decide`` runs before
    this tick's fill. A unit of age ``a`` is in transit while ``a < lead``.
    """
    profile = [0] * max(0, int(length))
    for order_tick, qty, lead in lots:
        age = int(tick) - int(order_tick)
        if 0 < age < int(lead) and age < len(profile):
            profile[age] += int(qty)
    return profile


def padded_sales_window(values: Sequence[Any], length: int = SALES_WINDOW) -> list[float]:
    """Oldest-first sales window. Missing older ticks are zeros on the left."""
    length = max(0, int(length))
    window = [float(value) for value in list(values)[-length:]]
    if len(window) < length:
        window = [0.0] * (length - len(window)) + window
    return window


def _profile_slots(values: Sequence[Any], length: int) -> list[float]:
    """Age-indexed profile. Missing older ages are zeros on the right."""
    length = max(0, int(length))
    profile = [float(value) for value in list(values)[:length]]
    if len(profile) < length:
        profile.extend([0.0] * (length - len(profile)))
    return profile


def _as_profile(value: Any, length: int) -> list[float]:
    """A sequence is an arrival-age profile. A scalar is the pipeline total.

    Ages of a bare total are unknown. Age 0 stays empty, matching a
    decision-time profile, and the total sits at age 1.
    """
    length = max(0, int(length))
    try:
        items = list(value)
    except TypeError:
        items = None
    if items is None:
        profile = [0.0] * length
        if profile:
            profile[1 if length > 1 else 0] = float(value)
        return profile
    return _profile_slots(items, length)


def sku_features(
    *,
    on_hand: float,
    in_transit: Any,
    sales_window: Sequence[float] | Sequence[int],
    cash: float,
    capacity: float,
    delivery_lag: float,
    sales_window_length: int = SALES_WINDOW,
) -> list[float]:
    """Observation vector the encoder is allowed to see.

    Layout: on-hand, the arrival-age profile (age 0 first, width
    ``ARRIVAL_AGE_SLOTS``), the sales window padded or trimmed to
    ``sales_window_length`` (oldest first), then cash, capacity, and
    the nominal delivery lead time. Nothing else — not a Market
    multiplier, a disruption label, or a calendar tick.
    """
    profile = _as_profile(in_transit, ARRIVAL_AGE_SLOTS)
    window = padded_sales_window(sales_window, sales_window_length)
    return [
        float(on_hand),
        *profile,
        *window,
        float(cash),
        float(capacity),
        float(delivery_lag),
    ]


def log_row_features(
    row: Mapping[str, Any],
    *,
    delivery_lag: int,
    sales_window_length: int = SALES_WINDOW,
) -> list[float]:
    """Encoder vector for one transition-log row.

    Extra columns — Market multiplier, disruption identity, tick, date —
    are ignored. A missing lead time falls back to ``delivery_lag``.
    """
    return sku_features(
        on_hand=float(row["on_hand"]),
        in_transit=row["in_transit"],
        sales_window=list(row["sales_window"]),
        cash=float(row["cash"]),
        capacity=float(row["capacity"]),
        delivery_lag=_row_lead(row, delivery_lag),
        sales_window_length=sales_window_length,
    )


def _row_lead(row: Mapping[str, Any], fallback: int) -> float:
    raw = row.get("delivery_lag", fallback)
    if raw is None:
        return float(fallback)
    value = float(raw)
    if math.isnan(value):
        return float(fallback)
    return value


def _groups(frame: pd.DataFrame):
    if frame.empty:
        return
    grouped = frame.groupby(["episode_seed", "product_id"], sort=True)
    for _, group in grouped:
        yield group.sort_values("tick", kind="mergesort")


def _targets(row: dict) -> list[float]:
    """On-hand, the next arrival-age profile, and sales. Not the pipeline sum."""
    profile = _as_profile(row["next_in_transit"], ARRIVAL_AGE_SLOTS)
    return [float(row["next_on_hand"]), *profile, float(row["next_sales"])]


def _batch_count(n_rows: int) -> int:
    if n_rows <= 0:
        return 0
    return (int(n_rows) + FIT_BATCH_SIZE - 1) // FIT_BATCH_SIZE


def _ema_decay(n_steps: int) -> float:
    """Decay that leaves ``EMA_INIT_SHARE`` of the initial target after ``n_steps``."""
    if n_steps <= 0:
        return 1.0
    return float(EMA_INIT_SHARE ** (1.0 / float(n_steps)))


def _generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def _holdout_episodes(
    episode_ids: Sequence[int],
    generator: torch.Generator,
) -> tuple[set[int], set[int]]:
    """Train and validation episode ids. One episode stays entirely on one side."""
    unique = sorted({int(episode) for episode in episode_ids})
    if len(unique) < 2:
        return set(unique), set()
    n_val = int(round(VAL_EPISODE_FRACTION * len(unique)))
    n_val = min(max(n_val, 1), len(unique) - 1)
    order = torch.randperm(len(unique), generator=generator).tolist()
    held = {unique[index] for index in order[:n_val]}
    return set(unique) - held, held


def _by_episode(frame: pd.DataFrame, episodes: set[int]) -> pd.DataFrame:
    if frame.empty or not episodes:
        return frame.iloc[0:0]
    return frame.loc[frame["episode_seed"].isin(episodes)]


class _PackedWindows:
    """Open-loop windows stored as float32 memmaps.

    Indexing a batch copies that batch into a tensor. The full array stays
    on disk so a published log can be fit without holding every window in RAM.
    """

    def __init__(
        self,
        obs: np.memmap,
        act: np.memmap,
        *,
        action_mean: float,
        action_std: float,
    ) -> None:
        self.obs = obs
        self.act = act
        self.action_mean = float(action_mean)
        self.action_std = float(action_std)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(size) for size in self.obs.shape)

    def batch(self, index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        idx = index.detach().cpu().numpy()
        obs = torch.tensor(np.asarray(self.obs[idx]), dtype=torch.float32)
        act = torch.tensor(np.asarray(self.act[idx]), dtype=torch.float32)
        return obs, act

    def as_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(np.asarray(self.obs), dtype=torch.float32),
            torch.tensor(np.asarray(self.act), dtype=torch.float32),
        )


def _window_dir() -> Path:
    path = Path("runs/jepa-replenishment/fit")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _discard_window_dir(directory: Path) -> None:
    for child in directory.glob("*"):
        child.unlink(missing_ok=True)
    directory.rmdir()


def _population_std(total_sq: float, total: float, count: int) -> tuple[float, float]:
    if count <= 0:
        return 0.0, 1.0
    mean = total / count
    var = max(total_sq / count - mean * mean, 0.0)
    return mean, max(math.sqrt(var), 1e-6)


def _moment_tensors(total: np.ndarray, total_sq: np.ndarray, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    if count <= 0:
        width = 0 if total is None else int(total.shape[0])
        return torch.zeros(width), torch.ones(width)
    mean = total / count
    var = np.maximum(total_sq / count - mean * mean, 0.0)
    std = np.maximum(np.sqrt(var), 1e-6)
    return (
        torch.tensor(mean, dtype=torch.float32),
        torch.tensor(std, dtype=torch.float32),
    )


def _snapshot(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


def _write_fit_artifact(directory: str | Path | None, name: str, payload: dict[str, Any]) -> None:
    destination = FIT_ARTIFACT_DIR if directory is None else Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, allow_nan=False)
    (destination / name).write_text(text + "\n", encoding="utf-8")


def _fit_record(
    train_ids: set[int],
    val_ids: set[int],
    trained: Mapping[str, Any],
    **extra: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "train_episodes": sorted(train_ids),
        "val_episodes": sorted(val_ids),
        "batch_size": FIT_BATCH_SIZE,
        "curve": list(trained["curve"]),
        "stopped_early": bool(trained["stopped_early"]),
        "best_epoch": trained["best_epoch"],
    }
    payload.update(extra)
    return payload


def _train_passes(
    *,
    n_train: int,
    epochs: int,
    generator: torch.Generator,
    optimizer: torch.optim.Optimizer,
    loss_on: Callable[[torch.Tensor], torch.Tensor],
    val_loss: Callable[[], float | None],
    after_step: Callable[[], None] | None,
    snapshot: Callable[[], dict[str, torch.Tensor]],
    restore: Callable[[Mapping[str, torch.Tensor]], None],
    patience: int,
) -> dict[str, Any]:
    """One epoch is one shuffled pass. Stop when the holdout stops improving."""
    curve: list[dict[str, Any]] = []
    if n_train <= 0 or epochs <= 0:
        return {"curve": curve, "stopped_early": False, "best_epoch": None}
    best_loss = math.inf
    best_epoch: int | None = None
    best_state: dict[str, torch.Tensor] | None = None
    wait = 0
    stopped = False
    steps = 0
    tolerated = max(int(patience), 1)
    for epoch in range(int(epochs)):
        order = torch.randperm(n_train, generator=generator)
        total = 0.0
        seen = 0
        for start in range(0, n_train, FIT_BATCH_SIZE):
            index = order[start : start + FIT_BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_on(index)
            loss.backward()
            optimizer.step()
            if after_step is not None:
                after_step()
            steps += 1
            batch = int(index.numel())
            total += float(loss.detach()) * batch
            seen += batch
        measured = val_loss()
        curve.append(
            {
                "epoch": epoch + 1,
                "train_loss": total / seen,
                "val_loss": measured,
                "steps": steps,
            }
        )
        if measured is None:
            best_epoch = epoch + 1
            continue
        if measured < best_loss:
            best_loss = measured
            best_epoch = epoch + 1
            best_state = snapshot()
            wait = 0
            continue
        wait += 1
        if wait >= tolerated:
            stopped = True
            break
    if best_state is not None:
        restore(best_state)
    return {"curve": curve, "stopped_early": stopped, "best_epoch": best_epoch}


class _EmptyRollout:
    def step(self, granted: list[float]) -> list[tuple[float, tuple[float, ...], float]]:
        return []


def _anti_collapse(embedding: torch.Tensor) -> torch.Tensor:
    """Keep per-dimension variance off the floor and punish correlation.

    A one-row batch has no variance to estimate; the penalty is zero
    but still tied to the embedding so the graph stays intact.
    """
    if embedding.shape[0] < 2:
        return embedding.sum() * 0.0
    centered = embedding - embedding.mean(dim=0, keepdim=True)
    variance = torch.relu(1.0 - centered.std(dim=0, unbiased=False)).mean()
    covariance = (centered.transpose(0, 1) @ centered) / embedding.shape[0]
    off_diagonal = covariance - torch.diag(torch.diag(covariance))
    return variance + off_diagonal.pow(2).mean()


def _mlp(in_dim: int, hidden: int, out_dim: int, depth: int) -> nn.Sequential:
    """``depth`` linear layers. ReLU sits between them, not after the last."""
    if depth < 1:
        raise ValueError("depth must be at least 1")
    layers: list[nn.Module] = []
    fan_in = in_dim
    for index in range(depth):
        fan_out = out_dim if index == depth - 1 else hidden
        layers.append(nn.Linear(fan_in, fan_out))
        if index != depth - 1:
            layers.append(nn.ReLU())
        fan_in = fan_out
    return nn.Sequential(*layers)


def parameter_count(module: nn.Module) -> int:
    """Every parameter, including a frozen EMA encoder."""
    return sum(parameter.numel() for parameter in module.parameters())


def trainable_parameter_count(module: nn.Module) -> int:
    """Parameters the optimizer is allowed to step."""
    return sum(
        parameter.numel() for parameter in module.parameters() if parameter.requires_grad
    )


class _Encoder(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int, depth: int) -> None:
        super().__init__()
        self.net = _mlp(in_dim, embed_dim, embed_dim, depth)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class _Predictor(nn.Module):
    def __init__(self, embed_dim: int, depth: int) -> None:
        super().__init__()
        self.net = _mlp(embed_dim + 1, embed_dim, embed_dim, depth)

    def forward(self, embedding: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([embedding, action], dim=-1))


class _Probe(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.head = nn.Linear(embed_dim, target_dim())

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.head(embedding)


class _DirectNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, depth: int) -> None:
        super().__init__()
        self.net = _mlp(in_dim + 1, hidden_dim, target_dim(), depth)

    def forward(self, features: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([features, action], dim=-1))


class _ScaledDynamics(nn.Module):
    def __init__(self, delivery_lag: int, model_unroll_steps: int | None = None) -> None:
        super().__init__()
        self.delivery_lag = int(delivery_lag)
        if model_unroll_steps is None:
            steps = PLANNING_HORIZON_MULTIPLE * self.delivery_lag
        else:
            steps = int(model_unroll_steps)
        if steps < 1:
            raise ValueError("model_unroll_steps must be at least 1")
        self.model_unroll_steps = steps
        self.train_episodes: set[int] = set()
        self.val_episodes: set[int] = set()
        dim = feature_dim()
        self.register_buffer("x_mean", torch.zeros(dim))
        self.register_buffer("x_std", torch.ones(dim))
        self.register_buffer("y_mean", torch.zeros(target_dim()))
        self.register_buffer("y_std", torch.ones(target_dim()))
        self.register_buffer("a_mean", torch.zeros(1))
        self.register_buffer("a_std", torch.ones(1))
        # Imagined orders are conditioned on this. 1 means the request arrived whole.
        self.register_buffer("fill_rate", torch.ones(()))

    def _set_stats(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        if features.numel():
            self.x_mean.copy_(features.mean(dim=0))
            self.x_std.copy_(features.std(dim=0, unbiased=False).clamp(min=1e-6))
        if actions.numel():
            self.a_mean.fill_(float(actions.mean()))
            self.a_std.fill_(max(float(actions.std(unbiased=False)), 1e-6))
        if targets.numel():
            self.y_mean.copy_(targets.mean(dim=0))
            self.y_std.copy_(targets.std(dim=0, unbiased=False).clamp(min=1e-6))

    def _norm_x(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.x_mean) / self.x_std

    def _norm_a(self, action: torch.Tensor) -> torch.Tensor:
        return (action - self.a_mean) / self.a_std

    def _denorm_y(self, targets: torch.Tensor) -> torch.Tensor:
        return targets * self.y_std + self.y_mean

    def _norm_y(self, targets: torch.Tensor) -> torch.Tensor:
        return (targets - self.y_mean) / self.y_std

    def _as_quantities(
        self, normalized: torch.Tensor
    ) -> list[tuple[float, tuple[float, ...], float]]:
        """On-hand, the arrival-age profile, and sales. The planner sums the profile."""
        decoded = self._denorm_y(normalized)
        rows: list[tuple[float, tuple[float, ...], float]] = []
        for row in decoded:
            values = [float(value) for value in row]
            rows.append((values[0], tuple(values[1:-1]), values[-1]))
        return rows

    def open_rollout(self, features: list[list[float]]) -> Any:
        """Start an open-loop unroll from the decision-time observation.

        Later steps see the predicted embedding (latent) or the predicted
        inventory (direct). They do not see cash after the holding charge,
        so a change in holding rate cannot change decoded quantities except
        by changing the granted quantity the planner feeds in.
        """
        raise NotImplementedError

    def _features(self, row: dict) -> list[float]:
        return log_row_features(row, delivery_lag=self.delivery_lag)


class LatentDynamics(_ScaledDynamics):
    """Encoder, EMA target, open-loop predictor, and stopped probes."""

    def __init__(
        self,
        delivery_lag: int,
        model_unroll_steps: int | None = None,
        *,
        embed_dim: int = REDUCED_EMBED_DIM,
        depth: int = REDUCED_DEPTH,
    ) -> None:
        super().__init__(delivery_lag, model_unroll_steps)
        if int(embed_dim) < 1:
            raise ValueError("embed_dim must be at least 1")
        if int(depth) < 1:
            raise ValueError("depth must be at least 1")
        self.embed_dim = int(embed_dim)
        self.depth = int(depth)
        dim = feature_dim()
        self.encoder = _Encoder(dim, self.embed_dim, self.depth)
        self.predictor = _Predictor(self.embed_dim, self.depth)
        self.probes = _Probe(self.embed_dim)
        self.target_encoder = copy.deepcopy(self.encoder)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)

    def fit(
        self,
        frame: pd.DataFrame,
        *,
        epochs: int,
        seed: int,
        lr: float = 1e-3,
        artifact_dir: str | Path | None = None,
        patience: int = EARLY_STOP_PATIENCE,
    ) -> None:
        torch.manual_seed(seed)
        self.fill_rate.fill_(_fill_ratio(frame))
        if epochs <= 0 or frame.empty:
            return
        generator = _generator(seed)
        train_ids, val_ids = _holdout_episodes(frame["episode_seed"].tolist(), generator)
        self.train_episodes = set(train_ids)
        self.val_episodes = set(val_ids)
        if len(frame) > _EAGER_ROW_LIMIT:
            directory = Path(tempfile.mkdtemp(prefix="jepa-windows-", dir=_window_dir()))
            try:
                trained, decay = self._fit_latent_packed(
                    frame,
                    train_ids,
                    val_ids,
                    directory,
                    epochs=epochs,
                    lr=lr,
                    generator=generator,
                    patience=patience,
                )
            finally:
                _discard_window_dir(directory)
        else:
            train_frame = _by_episode(frame, train_ids)
            val_frame = _by_episode(frame, val_ids)
            observations, actions = self._windows(train_frame)
            val_observations, val_actions = self._windows(val_frame)
            _probe_x, probe_y = self._probe_pairs(train_frame)
            self._set_stats(self._all_features(train_frame), actions.reshape(-1), probe_y)
            trained, decay = self._fit_latent(
                observations,
                actions,
                val_observations,
                val_actions,
                epochs=epochs,
                lr=lr,
                generator=generator,
                patience=patience,
            )
        _write_fit_artifact(
            artifact_dir,
            "latent.json",
            _fit_record(
                train_ids,
                val_ids,
                trained,
                patience=int(patience),
                ema_tau=decay,
                model_unroll_steps=self.model_unroll_steps,
            ),
        )
        self.fit_probes(
            frame,
            epochs=epochs,
            lr=lr,
            seed=seed,
            artifact_dir=artifact_dir,
            patience=patience,
        )

    def fit_probes(
        self,
        frame: pd.DataFrame,
        *,
        epochs: int,
        lr: float = 1e-3,
        seed: int = 0,
        artifact_dir: str | Path | None = None,
        patience: int = EARLY_STOP_PATIENCE,
    ) -> None:
        """Regress the quantity head from a stopped target embedding.

        The encoder is not in the optimizer and the embedding is
        detached, so this step cannot change encoder parameters.
        """
        if epochs <= 0 or frame.empty:
            return
        generator = _generator(seed)
        train_ids, val_ids = _holdout_episodes(frame["episode_seed"].tolist(), generator)
        if len(frame) > _EAGER_ROW_LIMIT:
            print(f"jepa fit probes rows {len(frame)}", flush=True)
            train_x, train_y = self._probe_pairs_in(frame, train_ids)
            val_x, val_y = self._probe_pairs_in(frame, val_ids)
        else:
            train_x, train_y = self._probe_pairs(_by_episode(frame, train_ids))
            val_x, val_y = self._probe_pairs(_by_episode(frame, val_ids))
        trained = self._fit_probe_passes(
            train_x,
            train_y,
            val_x,
            val_y,
            epochs=epochs,
            lr=lr,
            generator=generator,
            patience=patience,
        )
        _write_fit_artifact(
            artifact_dir,
            "probes.json",
            _fit_record(train_ids, val_ids, trained, patience=int(patience)),
        )

    def _fit_probe_passes(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        val_features: torch.Tensor,
        val_targets: torch.Tensor,
        *,
        epochs: int,
        lr: float,
        generator: torch.Generator,
        patience: int,
    ) -> dict[str, Any]:
        n_train = int(features.shape[0])
        if n_train <= 0 or epochs <= 0:
            return {"curve": [], "stopped_early": False, "best_epoch": None}
        features_n = self._norm_x(features)
        targets_n = self._norm_y(targets)
        if val_features.numel():
            val_features_n = self._norm_x(val_features)
            val_targets_n = self._norm_y(val_targets)
        else:
            val_features_n = val_features
            val_targets_n = val_targets
        optimizer = torch.optim.Adam(self.probes.parameters(), lr=lr)
        self.encoder.eval()
        self.target_encoder.eval()
        self.probes.train()

        def loss_on(index: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                embedding = self.target_encoder(features_n[index])
            return nn.functional.mse_loss(self.probes(embedding), targets_n[index])

        def measured() -> float | None:
            if val_features_n.numel() == 0:
                return None
            self.probes.eval()
            n_val = int(val_features_n.shape[0])
            with torch.no_grad():
                if n_val <= _EAGER_ROW_LIMIT:
                    embedding = self.target_encoder(val_features_n)
                    loss = float(nn.functional.mse_loss(self.probes(embedding), val_targets_n))
                else:
                    total = 0.0
                    for start, end in _chunks(n_val):
                        embedding = self.target_encoder(val_features_n[start:end])
                        batch = float(
                            nn.functional.mse_loss(
                                self.probes(embedding), val_targets_n[start:end]
                            )
                        )
                        total += batch * (end - start)
                    loss = total / n_val
            self.probes.train()
            return loss

        return _train_passes(
            n_train=n_train,
            epochs=epochs,
            generator=generator,
            optimizer=optimizer,
            loss_on=loss_on,
            val_loss=measured,
            after_step=None,
            snapshot=lambda: _snapshot(self),
            restore=self.load_state_dict,
            patience=patience,
        )

    @torch.no_grad()
    def predict_next(
        self,
        features: list[list[float]],
        granted: list[float],
    ) -> list[tuple[float, tuple[float, ...], float]]:
        """On-hand, the next arrival-age profile, and sales over the step."""
        if not features:
            return []
        self.encoder.eval()
        self.predictor.eval()
        self.probes.eval()
        observation = self._norm_x(torch.tensor(features, dtype=torch.float32))
        action = self._norm_a(torch.tensor(granted, dtype=torch.float32).unsqueeze(-1))
        embedding = self.predictor(self.encoder(observation), action)
        return self._as_quantities(self.probes(embedding))

    def open_rollout(self, features: list[list[float]]) -> Any:
        if not features:
            return _EmptyRollout()
        return _LatentRollout(self, features)

    def _fit_latent(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        val_observations: torch.Tensor,
        val_actions: torch.Tensor,
        *,
        epochs: int,
        lr: float,
        generator: torch.Generator,
        patience: int,
    ) -> tuple[dict[str, Any], float]:
        n_train = int(observations.shape[0])
        # One pass, not the epoch budget. Early stopping keeps an earlier
        # epoch, and that epoch has to have walked the target off its init.
        decay = _ema_decay(_batch_count(n_train))
        self._ema_decay = decay
        if epochs <= 0 or n_train == 0:
            return {"curve": [], "stopped_early": False, "best_epoch": None}, decay
        observations_n = self._norm_x(observations)
        actions_n = self._norm_a(actions)
        if val_observations.numel():
            val_obs_n = self._norm_x(val_observations)
            val_act_n = self._norm_a(val_actions)
        else:
            val_obs_n = val_observations
            val_act_n = val_actions
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.predictor.parameters()),
            lr=lr,
        )
        self.encoder.train()
        self.predictor.train()

        def loss_on(index: torch.Tensor) -> torch.Tensor:
            return self._open_loop_loss(observations_n[index], actions_n[index])

        def measured() -> float | None:
            if val_obs_n.numel() == 0:
                return None
            with torch.no_grad():
                return float(self._rollout_mse(val_obs_n, val_act_n))

        trained = _train_passes(
            n_train=n_train,
            epochs=epochs,
            generator=generator,
            optimizer=optimizer,
            loss_on=loss_on,
            val_loss=measured,
            after_step=self._ema,
            snapshot=lambda: _snapshot(self),
            restore=self.load_state_dict,
            patience=patience,
        )
        return trained, decay

    def _fit_latent_packed(
        self,
        frame: pd.DataFrame,
        train_ids: set[int],
        val_ids: set[int],
        directory: Path,
        *,
        epochs: int,
        lr: float,
        generator: torch.Generator,
        patience: int,
    ) -> tuple[dict[str, Any], float]:
        """Train from memmap windows. The loss is the eager open-loop loss."""
        observations = self._windows_packed(frame, directory, "train", train_ids)
        validation = self._windows_packed(frame, directory, "val", val_ids)
        x_mean, x_std = self._feature_moments(frame, train_ids)
        y_mean, y_std = self._target_moments(frame, train_ids)
        self.x_mean.copy_(x_mean)
        self.x_std.copy_(x_std)
        self.y_mean.copy_(y_mean)
        self.y_std.copy_(y_std)
        self.a_mean.fill_(observations.action_mean)
        self.a_std.fill_(observations.action_std)
        n_train = int(observations.shape[0])
        print(f"jepa fit latent windows {n_train}", flush=True)
        decay = _ema_decay(_batch_count(n_train))
        self._ema_decay = decay
        if epochs <= 0 or n_train == 0:
            return {"curve": [], "stopped_early": False, "best_epoch": None}, decay
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.predictor.parameters()),
            lr=lr,
        )
        self.encoder.train()
        self.predictor.train()
        n_val = int(validation.shape[0])

        def loss_on(index: torch.Tensor) -> torch.Tensor:
            obs, act = observations.batch(index)
            return self._open_loop_loss(self._norm_x(obs), self._norm_a(act))

        def measured() -> float | None:
            if n_val == 0:
                print("jepa fit latent epoch val_loss=none", flush=True)
                return None
            total = 0.0
            weight = 0
            with torch.no_grad():
                for start in range(0, n_val, _ROLLOUT_CHUNK):
                    end = min(start + _ROLLOUT_CHUNK, n_val)
                    index = torch.arange(start, end)
                    obs, act = validation.batch(index)
                    batch = end - start
                    total += float(self._rollout_mse(self._norm_x(obs), self._norm_a(act))) * batch
                    weight += batch
            value = total / weight
            print(f"jepa fit latent epoch val_loss={value}", flush=True)
            return value

        trained = _train_passes(
            n_train=n_train,
            epochs=epochs,
            generator=generator,
            optimizer=optimizer,
            loss_on=loss_on,
            val_loss=measured,
            after_step=self._ema,
            snapshot=lambda: _snapshot(self),
            restore=self.load_state_dict,
            patience=patience,
        )
        return trained, decay

    def _rollout_mse(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Mean open-loop distance to the stopped target encoder.

        ``observations`` are normalized. The mean, not the sum, so a longer
        planning horizon does not rescale the update.
        """
        horizon = actions.shape[1]
        embedding = self.encoder(observations[:, 0])
        loss = observations.new_zeros(())
        for step in range(horizon):
            embedding = self.predictor(embedding, actions[:, step : step + 1])
            target = self.target_encoder(observations[:, step + 1]).detach()
            loss = loss + nn.functional.mse_loss(embedding, target)
        return loss / max(horizon, 1)

    def _open_loop_loss(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Unroll the predictor for ``H`` steps without feeding true states back.

        ``observations`` is ``[batch, H + 1, features]`` and already
        normalized. The target at each step is the stopped target encoder.
        """
        flat = observations.reshape(-1, observations.shape[-1])
        return self._rollout_mse(observations, actions) + _anti_collapse(self.encoder(flat))

    def _ema(self) -> None:
        tau = float(self._ema_decay)
        with torch.no_grad():
            for online, target in zip(
                self.encoder.parameters(),
                self.target_encoder.parameters(),
                strict=True,
            ):
                target.mul_(tau).add_(online, alpha=1.0 - tau)

    def _windows(self, frame: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        horizon = self.model_unroll_steps
        sequences: list[list[list[float]]] = []
        actions: list[list[float]] = []
        if horizon < 1:
            empty_x = torch.zeros(0, 1, feature_dim())
            return empty_x, torch.zeros(0, 1)
        for group in _groups(frame):
            rows = list(group.to_dict("records"))
            features = [self._features(row) for row in rows]
            if len(rows) < horizon + 1:
                continue
            for start in range(len(rows) - horizon):
                sequences.append(features[start : start + horizon + 1])
                actions.append(
                    [float(rows[start + step]["filled_qty"]) for step in range(horizon)]
                )
        if not sequences:
            return (
                torch.zeros(0, horizon + 1, feature_dim()),
                torch.zeros(0, horizon),
            )
        return (
            torch.tensor(sequences, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.float32),
        )

    def _windows_packed(self, frame: pd.DataFrame, directory: Path, name: str, episodes: set[int]) -> _PackedWindows:
        """Same window order as ``_windows``, written to ``directory``."""
        horizon = int(self.model_unroll_steps)
        dim = feature_dim()
        count = 0
        for group in _groups(frame):
            if int(group["episode_seed"].iloc[0]) not in episodes:
                continue
            length = len(group)
            if length >= horizon + 1:
                count += length - horizon
        obs = np.memmap(
            directory / f"{name}-obs.dat",
            dtype=np.float32,
            mode="w+",
            shape=(count, horizon + 1, dim),
        )
        act = np.memmap(
            directory / f"{name}-act.dat",
            dtype=np.float32,
            mode="w+",
            shape=(count, horizon),
        )
        cursor = 0
        action_sum = 0.0
        action_sq = 0.0
        action_n = 0
        for group in _groups(frame):
            if int(group["episode_seed"].iloc[0]) not in episodes:
                continue
            rows = list(group.to_dict("records"))
            if len(rows) < horizon + 1:
                continue
            features = [self._features(row) for row in rows]
            n = len(rows) - horizon
            block = np.asarray(
                [features[start : start + horizon + 1] for start in range(n)],
                dtype=np.float32,
            )
            actions = np.asarray(
                [
                    [float(rows[start + step]["filled_qty"]) for step in range(horizon)]
                    for start in range(n)
                ],
                dtype=np.float32,
            )
            obs[cursor : cursor + n] = block
            act[cursor : cursor + n] = actions
            cursor += n
            action_sum += float(actions.sum(dtype=np.float64))
            action_sq += float(np.square(actions.astype(np.float64)).sum())
            action_n += int(actions.size)
        obs.flush()
        act.flush()
        mean, std = _population_std(action_sq, action_sum, action_n)
        return _PackedWindows(obs, act, action_mean=mean, action_std=std)

    def _feature_moments(self, frame: pd.DataFrame, episodes: set[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Population mean and std of every training row, matching ``_all_features``."""
        total: np.ndarray | None = None
        total_sq: np.ndarray | None = None
        count = 0
        for group in _groups(frame):
            if int(group["episode_seed"].iloc[0]) not in episodes:
                continue
            for row in group.to_dict("records"):
                feat = np.asarray(self._features(row), dtype=np.float64)
                if total is None or total_sq is None:
                    total = np.zeros_like(feat)
                    total_sq = np.zeros_like(feat)
                total += feat
                total_sq += feat * feat
                count += 1
        assert total is not None and total_sq is not None
        return _moment_tensors(total, total_sq, count)

    def _target_moments(self, frame: pd.DataFrame, episodes: set[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Population mean and std of probe targets. The last row of a group has no pair."""
        total: np.ndarray | None = None
        total_sq: np.ndarray | None = None
        count = 0
        width = target_dim()
        for group in _groups(frame):
            if int(group["episode_seed"].iloc[0]) not in episodes:
                continue
            rows = list(group.to_dict("records"))
            for row in rows[:-1]:
                values = np.asarray(_targets(row), dtype=np.float64)
                if total is None or total_sq is None:
                    total = np.zeros(width, dtype=np.float64)
                    total_sq = np.zeros(width, dtype=np.float64)
                total += values
                total_sq += values * values
                count += 1
        if total is None or total_sq is None:
            return torch.zeros(width), torch.ones(width)
        return _moment_tensors(total, total_sq, count)

    def _probe_pairs(self, frame: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        if len(frame) > _EAGER_ROW_LIMIT:
            return self._probe_pairs_compact(frame)
        features: list[list[float]] = []
        targets: list[list[float]] = []
        for group in _groups(frame):
            rows = list(group.to_dict("records"))
            for index in range(len(rows) - 1):
                features.append(self._features(rows[index + 1]))
                targets.append(_targets(rows[index]))
        if not features:
            return torch.zeros(0, feature_dim()), torch.zeros(0, target_dim())
        return torch.tensor(features, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

    def _probe_pairs_in(
        self, frame: pd.DataFrame, episodes: set[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._probe_pairs_compact(frame, episodes)

    def _probe_pairs_compact(
        self, frame: pd.DataFrame, episodes: set[int] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count = 0
        for group in _groups(frame):
            if episodes is not None and int(group["episode_seed"].iloc[0]) not in episodes:
                continue
            length = len(group)
            if length >= 2:
                count += length - 1
        dim = feature_dim()
        width = target_dim()
        features = np.empty((count, dim), dtype=np.float32)
        targets = np.empty((count, width), dtype=np.float32)
        cursor = 0
        for group in _groups(frame):
            if episodes is not None and int(group["episode_seed"].iloc[0]) not in episodes:
                continue
            rows = list(group.to_dict("records"))
            n = len(rows) - 1
            if n <= 0:
                continue
            features[cursor : cursor + n] = np.asarray(
                [self._features(rows[index + 1]) for index in range(n)],
                dtype=np.float32,
            )
            targets[cursor : cursor + n] = np.asarray(
                [_targets(rows[index]) for index in range(n)],
                dtype=np.float32,
            )
            cursor += n
        return torch.from_numpy(features), torch.from_numpy(targets)

    def _all_features(self, frame: pd.DataFrame) -> torch.Tensor:
        rows = [
            self._features(row)
            for group in _groups(frame)
            for row in group.to_dict("records")
        ]
        if not rows:
            return torch.zeros(0, feature_dim())
        return torch.tensor(rows, dtype=torch.float32)

class DirectDynamics(_ScaledDynamics):
    """Predict on-hand, the arrival-age profile, and sales from the observation.

    No latent target and no EMA encoder. Training is the same minibatch
    passes, episode holdout, and early stopping as the latent fit. The
    planner around it is the same search and the same ration.
    """

    def __init__(
        self,
        delivery_lag: int,
        model_unroll_steps: int | None = None,
        *,
        hidden_dim: int = REDUCED_DIRECT_HIDDEN,
        depth: int = REDUCED_DEPTH,
    ) -> None:
        super().__init__(delivery_lag, model_unroll_steps)
        if int(hidden_dim) < 1:
            raise ValueError("hidden_dim must be at least 1")
        if int(depth) < 1:
            raise ValueError("depth must be at least 1")
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.net = _DirectNet(feature_dim(), self.hidden_dim, self.depth)

    def fit(
        self,
        frame: pd.DataFrame,
        *,
        epochs: int,
        seed: int,
        lr: float = 1e-3,
        artifact_dir: str | Path | None = None,
        patience: int = EARLY_STOP_PATIENCE,
    ) -> None:
        torch.manual_seed(seed)
        self.fill_rate.fill_(_fill_ratio(frame))
        if epochs <= 0 or frame.empty:
            return
        generator = _generator(seed)
        train_ids, val_ids = _holdout_episodes(frame["episode_seed"].tolist(), generator)
        self.train_episodes = set(train_ids)
        self.val_episodes = set(val_ids)
        if len(frame) > _EAGER_ROW_LIMIT:
            print(f"jepa fit direct rows {len(frame)}", flush=True)
            features, actions, targets = self._pairs_in(frame, train_ids)
            val_features, val_actions, val_targets = self._pairs_in(frame, val_ids)
        else:
            features, actions, targets = self._pairs(_by_episode(frame, train_ids))
            val_features, val_actions, val_targets = self._pairs(_by_episode(frame, val_ids))
        self._set_stats(features, actions, targets)
        trained = self._fit_passes(
            features,
            actions,
            targets,
            val_features,
            val_actions,
            val_targets,
            epochs=epochs,
            lr=lr,
            generator=generator,
            patience=patience,
        )
        _write_fit_artifact(
            artifact_dir,
            "direct.json",
            _fit_record(
                train_ids,
                val_ids,
                trained,
                patience=int(patience),
                model_unroll_steps=self.model_unroll_steps,
            ),
        )

    def _fit_passes(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        targets: torch.Tensor,
        val_features: torch.Tensor,
        val_actions: torch.Tensor,
        val_targets: torch.Tensor,
        *,
        epochs: int,
        lr: float,
        generator: torch.Generator,
        patience: int,
    ) -> dict[str, Any]:
        n_train = int(features.shape[0])
        if n_train <= 0 or epochs <= 0:
            return {"curve": [], "stopped_early": False, "best_epoch": None}
        features_n = self._norm_x(features)
        actions_n = self._norm_a(actions)
        targets_n = self._norm_y(targets)
        if val_features.numel():
            val_features_n = self._norm_x(val_features)
            val_actions_n = self._norm_a(val_actions)
            val_targets_n = self._norm_y(val_targets)
        else:
            val_features_n = val_features
            val_actions_n = val_actions
            val_targets_n = val_targets
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.net.train()

        def loss_on(index: torch.Tensor) -> torch.Tensor:
            return nn.functional.mse_loss(
                self.net(features_n[index], actions_n[index]),
                targets_n[index],
            )

        def measured() -> float | None:
            if val_features_n.numel() == 0:
                return None
            self.net.eval()
            n_val = int(val_features_n.shape[0])
            with torch.no_grad():
                if n_val <= _EAGER_ROW_LIMIT:
                    loss = float(
                        nn.functional.mse_loss(
                            self.net(val_features_n, val_actions_n),
                            val_targets_n,
                        )
                    )
                else:
                    total = 0.0
                    for start, end in _chunks(n_val):
                        batch = float(
                            nn.functional.mse_loss(
                                self.net(val_features_n[start:end], val_actions_n[start:end]),
                                val_targets_n[start:end],
                            )
                        )
                        total += batch * (end - start)
                    loss = total / n_val
            self.net.train()
            return loss

        return _train_passes(
            n_train=n_train,
            epochs=epochs,
            generator=generator,
            optimizer=optimizer,
            loss_on=loss_on,
            val_loss=measured,
            after_step=None,
            snapshot=lambda: _snapshot(self),
            restore=self.load_state_dict,
            patience=patience,
        )

    @torch.no_grad()
    def predict_next(
        self,
        features: list[list[float]],
        granted: list[float],
    ) -> list[tuple[float, tuple[float, ...], float]]:
        """On-hand, the next arrival-age profile, and sales over the step."""
        if not features:
            return []
        self.net.eval()
        observation = self._norm_x(torch.tensor(features, dtype=torch.float32))
        action = self._norm_a(torch.tensor(granted, dtype=torch.float32).unsqueeze(-1))
        return self._as_quantities(self.net(observation, action))

    def open_rollout(self, features: list[list[float]]) -> Any:
        if not features:
            return _EmptyRollout()
        return _DirectRollout(self, features)

    def _pairs(
        self, frame: pd.DataFrame
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(frame) > _EAGER_ROW_LIMIT:
            packed = _compact_step_arrays(frame, self._features)
            dim = feature_dim()
            if packed is None:
                return torch.zeros(0, dim), torch.zeros(0, 1), torch.zeros(0, target_dim())
            features, actions, targets = packed
            return (
                torch.from_numpy(features),
                torch.from_numpy(actions).unsqueeze(-1),
                torch.from_numpy(targets),
            )
        features: list[list[float]] = []
        actions: list[float] = []
        targets: list[list[float]] = []
        for group in _groups(frame):
            rows = list(group.to_dict("records"))
            for index in range(len(rows) - 1):
                current = rows[index]
                features.append(self._features(current))
                actions.append(float(current["filled_qty"]))
                targets.append(_targets(current))
        dim = feature_dim()
        if not features:
            return torch.zeros(0, dim), torch.zeros(0, 1), torch.zeros(0, target_dim())
        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(targets, dtype=torch.float32),
        )

    def _pairs_in(
        self, frame: pd.DataFrame, episodes: set[int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        packed = _compact_step_arrays(frame, self._features, episodes)
        dim = feature_dim()
        if packed is None:
            return torch.zeros(0, dim), torch.zeros(0, 1), torch.zeros(0, target_dim())
        features, actions, targets = packed
        return (
            torch.from_numpy(features),
            torch.from_numpy(actions).unsqueeze(-1),
            torch.from_numpy(targets),
        )


def direct_hidden_matching(embed_dim: int, depth: int) -> int:
    """Hidden width whose parameter count is closest to the latent trainable count.

    Same depth as the latent model. The EMA copy is not extra capacity, so
    the direct net is matched to the parameters the latent optimizer steps.
    """
    target = trainable_parameter_count(
        LatentDynamics(delivery_lag=1, embed_dim=int(embed_dim), depth=int(depth))
    )

    def count_at(width: int) -> int:
        return parameter_count(
            DirectDynamics(delivery_lag=1, hidden_dim=width, depth=int(depth))
        )

    hi = 1
    while count_at(hi) < target and hi < 8192:
        hi *= 2
    left, right = 1, hi
    while left < right:
        mid = (left + right) // 2
        if count_at(mid) < target:
            left = mid + 1
        else:
            right = mid
    if left == 1:
        return 1
    under = left - 1
    if abs(count_at(under) - target) <= abs(count_at(left) - target):
        return under
    return left


# Resolved once so the published fit and the curves share one width.
PUBLISHED_DIRECT_HIDDEN = direct_hidden_matching(PUBLISHED_EMBED_DIM, PUBLISHED_DEPTH)


class _LatentRollout:
    """Carry the predicted embedding. Do not re-encode decoded inventory."""

    def __init__(self, dynamics: LatentDynamics, features: list[list[float]]) -> None:
        self._dynamics = dynamics
        dynamics.encoder.eval()
        dynamics.predictor.eval()
        dynamics.probes.eval()
        observation = dynamics._norm_x(torch.tensor(features, dtype=torch.float32))
        with torch.no_grad():
            self._embedding = dynamics.encoder(observation)

    @torch.no_grad()
    def step(self, granted: list[float]) -> list[tuple[float, tuple[float, ...], float]]:
        dynamics = self._dynamics
        action = dynamics._norm_a(torch.tensor(granted, dtype=torch.float32).unsqueeze(-1))
        self._embedding = dynamics.predictor(self._embedding, action)
        return dynamics._as_quantities(dynamics.probes(self._embedding))


def _advance_observation(
    raw: torch.Tensor,
    on_hand: torch.Tensor,
    sales: torch.Tensor,
    granted: torch.Tensor,
) -> None:
    """Write a decoded step back into the observation the direct model sees.

    On-hand and the newest sales replace the decision-time values. The
    pipeline ages from the grant and the lead time already in the vector.
    Cash, capacity, and that lead time stay put.
    """
    raw[:, 0] = on_hand
    profile_end = 1 + ARRIVAL_AGE_SLOTS
    profile = raw[:, 1:profile_end]
    shifted = torch.zeros_like(profile)
    if ARRIVAL_AGE_SLOTS > 1:
        shifted[:, 1] = granted
    lead = raw[:, -1]
    for age in range(1, ARRIVAL_AGE_SLOTS - 1):
        dest = age + 1
        keep = lead > float(dest)
        shifted[keep, dest] = profile[keep, age]
    raw[:, 1:profile_end] = shifted
    length = SALES_WINDOW
    if length <= 0:
        return
    start = profile_end
    end = profile_end + length
    window = raw[:, start:end].clone()
    if length > 1:
        raw[:, start : end - 1] = window[:, 1:]
    raw[:, end - 1] = sales


class _DirectRollout:
    """Feed predicted on-hand and sales forward, and age the pipeline.

    Cash, capacity, and the delivery lead time stay at the decision-time
    observation, so a later holding charge cannot change the next decoded
    quantities. The decoded in-transit value is an arrival-age profile;
    this rollout does not write that profile back. The next
    observation keeps ages: this step's grant becomes age 1, older cohorts
    move up one tick, and a cohort whose next age is not below the nominal
    lead drops out.
    """

    def __init__(self, dynamics: DirectDynamics, features: list[list[float]]) -> None:
        self._dynamics = dynamics
        self._raw = torch.tensor(features, dtype=torch.float32)

    @torch.no_grad()
    def step(self, granted: list[float]) -> list[tuple[float, tuple[float, ...], float]]:
        quantities = self._dynamics.predict_next(self._raw.tolist(), granted)
        if self._raw.numel() == 0 or not quantities:
            return quantities
        _advance_observation(
            self._raw,
            torch.tensor([row[0] for row in quantities], dtype=torch.float32),
            torch.tensor([row[2] for row in quantities], dtype=torch.float32),
            torch.tensor(granted, dtype=torch.float32),
        )
        return quantities


@dataclass(frozen=True)
class RolloutAccuracy:
    """Held-out per-step rollout R² from step 1 through the planning horizon.

    Each number is the mean of the on-hand, arrival-age profile, and sales
    R² at that step. A target with no held-out variance is left out of the mean.
    """

    steps: tuple[int, ...]
    latent: tuple[float, ...]
    direct: tuple[float, ...]
    ols: tuple[float, ...]


def rollout_beats_baseline(accuracy: RolloutAccuracy) -> bool:
    """Latent mean R² over the horizon is strictly above ordinary least squares."""
    if not accuracy.latent or len(accuracy.latent) != len(accuracy.ols):
        return False
    if any(math.isnan(value) for value in (*accuracy.latent, *accuracy.ols)):
        return False
    learned = sum(accuracy.latent) / len(accuracy.latent)
    baseline = sum(accuracy.ols) / len(accuracy.ols)
    return learned > baseline


def _r2(pred: torch.Tensor, true: torch.Tensor) -> float | None:
    if true.shape[0] < 2 or true.shape[1] == 0:
        return None
    total = (true - true.mean(dim=0)).pow(2).sum()
    if float(total) <= 1e-8:
        return None
    residual = (pred - true).pow(2).sum()
    return float(1.0 - residual / total)


def _step_scores(pred: torch.Tensor, true: torch.Tensor) -> tuple[float, ...]:
    scores: list[float] = []
    for step in range(pred.shape[1]):
        groups = (
            _r2(pred[:, step, 0:1], true[:, step, 0:1]),
            _r2(pred[:, step, 1:-1], true[:, step, 1:-1]),
            _r2(pred[:, step, -1:], true[:, step, -1:]),
        )
        finite = [value for value in groups if value is not None]
        scores.append(float("nan") if not finite else sum(finite) / len(finite))
    return tuple(scores)


def _compact_step_arrays(
    frame: pd.DataFrame,
    feature_of: Callable[[dict], list[float]],
    episodes: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """One-step pairs as float32 arrays. ``None`` when the frame has no pair."""
    count = 0
    for group in _groups(frame):
        if episodes is not None and int(group["episode_seed"].iloc[0]) not in episodes:
            continue
        length = len(group)
        if length >= 2:
            count += length - 1
    if count == 0:
        return None
    features = np.empty((count, feature_dim()), dtype=np.float32)
    actions = np.empty((count,), dtype=np.float32)
    targets = np.empty((count, target_dim()), dtype=np.float32)
    cursor = 0
    for group in _groups(frame):
        if episodes is not None and int(group["episode_seed"].iloc[0]) not in episodes:
            continue
        rows = list(group.to_dict("records"))
        n = len(rows) - 1
        if n <= 0:
            continue
        features[cursor : cursor + n] = np.asarray(
            [feature_of(rows[index]) for index in range(n)],
            dtype=np.float32,
        )
        actions[cursor : cursor + n] = np.asarray(
            [float(rows[index]["filled_qty"]) for index in range(n)],
            dtype=np.float32,
        )
        targets[cursor : cursor + n] = np.asarray(
            [_targets(rows[index]) for index in range(n)],
            dtype=np.float32,
        )
        cursor += n
    return features, actions, targets


def _score_windows(
    frame: pd.DataFrame,
    episodes: set[int],
    horizon: int,
    feature_of: Callable[[dict], list[float]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if len(frame) > _EAGER_ROW_LIMIT:
        return _score_windows_compact(frame, episodes, horizon, feature_of)
    initials: list[list[float]] = []
    actions: list[list[float]] = []
    targets: list[list[list[float]]] = []
    for group in _groups(frame):
        rows = list(group.to_dict("records"))
        if not rows or int(rows[0]["episode_seed"]) not in episodes:
            continue
        if len(rows) < horizon + 1:
            continue
        features = [feature_of(row) for row in rows]
        for start in range(len(rows) - horizon):
            initials.append(features[start])
            actions.append([float(rows[start + step]["filled_qty"]) for step in range(horizon)])
            targets.append([_targets(rows[start + step]) for step in range(horizon)])
    if len(initials) < 2:
        return None
    return (
        torch.tensor(initials, dtype=torch.float32),
        torch.tensor(actions, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
    )


def _score_windows_compact(
    frame: pd.DataFrame,
    episodes: set[int],
    horizon: int,
    feature_of: Callable[[dict], list[float]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    count = 0
    for group in _groups(frame):
        if int(group["episode_seed"].iloc[0]) not in episodes:
            continue
        length = len(group)
        if length >= horizon + 1:
            count += length - horizon
    if count < 2:
        return None
    width = target_dim()
    initials = np.empty((count, feature_dim()), dtype=np.float32)
    actions = np.empty((count, horizon), dtype=np.float32)
    targets = np.empty((count, horizon, width), dtype=np.float32)
    cursor = 0
    for group in _groups(frame):
        if int(group["episode_seed"].iloc[0]) not in episodes:
            continue
        rows = list(group.to_dict("records"))
        if len(rows) < horizon + 1:
            continue
        features = [feature_of(row) for row in rows]
        n = len(rows) - horizon
        initials[cursor : cursor + n] = np.asarray(
            [features[start] for start in range(n)],
            dtype=np.float32,
        )
        actions[cursor : cursor + n] = np.asarray(
            [
                [float(rows[start + step]["filled_qty"]) for step in range(horizon)]
                for start in range(n)
            ],
            dtype=np.float32,
        )
        targets[cursor : cursor + n] = np.asarray(
            [[_targets(rows[start + step]) for step in range(horizon)] for start in range(n)],
            dtype=np.float32,
        )
        cursor += n
    return torch.from_numpy(initials), torch.from_numpy(actions), torch.from_numpy(targets)


def _one_step_pairs(
    frame: pd.DataFrame,
    episodes: set[int],
    feature_of: Callable[[dict], list[float]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if len(frame) > _EAGER_ROW_LIMIT:
        packed = _compact_step_arrays(frame, feature_of, episodes)
        if packed is None:
            return None
        features, actions, targets = packed
        return torch.from_numpy(features), torch.from_numpy(actions), torch.from_numpy(targets)
    features: list[list[float]] = []
    actions: list[float] = []
    targets: list[list[float]] = []
    for group in _groups(frame):
        rows = list(group.to_dict("records"))
        if not rows or int(rows[0]["episode_seed"]) not in episodes:
            continue
        for index in range(len(rows) - 1):
            features.append(feature_of(rows[index]))
            actions.append(float(rows[index]["filled_qty"]))
            targets.append(_targets(rows[index]))
    if not features:
        return None
    return (
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(actions, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
    )


def _fit_ols(
    features: torch.Tensor,
    actions: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Least squares of the targets on the varying inputs and an intercept.

    A constant column is already the intercept, so it is left out of the
    design. That is still ordinary least squares on the same inputs.
    """
    keep = features.std(dim=0, unbiased=False) > 1e-6
    if not bool(keep.any()):
        keep = torch.zeros(features.shape[1], dtype=torch.bool)
        keep[0] = True
    ones = torch.ones(features.shape[0], 1)
    design = torch.cat([features[:, keep], actions.unsqueeze(-1), ones], dim=1)
    solution = torch.linalg.lstsq(design, targets).solution
    return solution, keep


def _chunks(n_rows: int) -> list[tuple[int, int]]:
    return [
        (start, min(start + _ROLLOUT_CHUNK, n_rows))
        for start in range(0, n_rows, _ROLLOUT_CHUNK)
    ]


def _ols_rollout(
    solution: torch.Tensor,
    keep: torch.Tensor,
    initial: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    if int(initial.shape[0]) > _ROLLOUT_CHUNK:
        parts = [
            _ols_rollout(solution, keep, initial[start:end], actions[start:end])
            for start, end in _chunks(int(initial.shape[0]))
        ]
        return torch.cat(parts, dim=0)
    raw = initial.clone()
    steps: list[torch.Tensor] = []
    for step in range(actions.shape[1]):
        granted = actions[:, step]
        ones = torch.ones(raw.shape[0], 1)
        design = torch.cat([raw[:, keep], granted.unsqueeze(-1), ones], dim=1)
        pred = design @ solution
        steps.append(pred)
        _advance_observation(raw, pred[:, 0], pred[:, -1], granted)
    return torch.stack(steps, dim=1)


def _shift_pipeline(profile: list[float], granted: float, lead: int) -> tuple[list[float], float]:
    """Age the arrival profile one tick. Units that leave it have arrived.

    Matches the observation advance: the grant is age 1, older cohorts
    move up while their next age is still below the nominal lead, and
    age 0 (empty at decision time) drops out.
    """
    slots = len(profile)
    shifted = [0.0] * slots
    if slots > 1:
        shifted[1] = float(granted)
    for age in range(1, max(0, slots - 1)):
        dest = age + 1
        if dest < slots and lead > dest:
            shifted[dest] = float(profile[age])
    arriving = sum(profile) + float(granted) - sum(shifted)
    return shifted, arriving


def _oracle_state(feat: list[float], fallback_lag: int) -> tuple[float, list[float], float, int]:
    on_hand = float(feat[0]) if feat else 0.0
    profile = [float(value) for value in feat[1 : 1 + ARRIVAL_AGE_SLOTS]]
    start = 1 + ARRIVAL_AGE_SLOTS
    window = [float(value) for value in feat[start : start + SALES_WINDOW]]
    lead = int(fallback_lag)
    if feat:
        parsed = int(round(float(feat[-1])))
        if parsed >= 1:
            lead = parsed
    recent = window[-max(1, lead) :] if window else []
    rate = sum(recent) / len(recent) if recent else 0.0
    return on_hand, profile, rate, lead


class OracleDynamics:
    """Inventory recursion used as the planner's ceiling.

    Arrivals follow the arrival-age profile and the nominal lead time.
    Sales are stock after those arrivals, capped by the mean of the most
    recent lead-time of sales. ``fill_rate`` is 1, so the grant the
    planner feeds in is the arrival. Demand is fixed when the rollout
    opens; later imagined sales do not revise it.
    """

    def __init__(self, delivery_lag: int) -> None:
        self.delivery_lag = int(delivery_lag)
        self.fill_rate = 1.0

    def open_rollout(self, features: list[list[float]]) -> Any:
        if not features:
            return _EmptyRollout()
        return _OracleRollout(features, self.delivery_lag)


class _OracleRollout:
    def __init__(self, features: list[list[float]], fallback_lag: int) -> None:
        self.rows = [_oracle_state(feat, fallback_lag) for feat in features]

    def step(self, granted: list[float]) -> list[tuple[float, tuple[float, ...], float]]:
        predicted: list[tuple[float, tuple[float, ...], float]] = []
        for index, qty in enumerate(granted):
            on_hand, profile, rate, lead = self.rows[index]
            shifted, arriving = _shift_pipeline(profile, float(qty), lead)
            stock = on_hand + arriving
            sales = min(max(stock, 0.0), max(rate, 0.0))
            next_on_hand = stock - sales
            self.rows[index] = (next_on_hand, shifted, rate, lead)
            predicted.append((next_on_hand, tuple(shifted), sales))
        return predicted


class LeastSquaresDynamics:
    """One-step least squares rolled forward like the direct predictor.

    Fit on the encoder inputs and the filled quantity. ``fill_rate`` is
    the log's realised fill, the same fraction the learned planner uses.
    """

    def __init__(
        self,
        solution: torch.Tensor,
        keep: torch.Tensor,
        *,
        delivery_lag: int,
        fill_rate: float,
    ) -> None:
        self.solution = solution
        self.keep = keep
        self.delivery_lag = int(delivery_lag)
        self.fill_rate = float(fill_rate)

    def open_rollout(self, features: list[list[float]]) -> Any:
        if not features:
            return _EmptyRollout()
        return _LeastSquaresRollout(self, features)


class _LeastSquaresRollout:
    def __init__(self, dynamics: LeastSquaresDynamics, features: list[list[float]]) -> None:
        self._dynamics = dynamics
        self._raw = torch.tensor(features, dtype=torch.float32)

    @torch.no_grad()
    def step(self, granted: list[float]) -> list[tuple[float, tuple[float, ...], float]]:
        dynamics = self._dynamics
        action = torch.tensor(granted, dtype=torch.float32)
        ones = torch.ones(self._raw.shape[0], 1)
        design = torch.cat([self._raw[:, dynamics.keep], action.unsqueeze(-1), ones], dim=1)
        pred = design @ dynamics.solution
        _advance_observation(self._raw, pred[:, 0], pred[:, -1], action)
        rows: list[tuple[float, tuple[float, ...], float]] = []
        for row in pred:
            values = [float(value) for value in row]
            rows.append((values[0], tuple(values[1:-1]), values[-1]))
        return rows


def fit_least_squares_dynamics(
    frame: pd.DataFrame,
    *,
    delivery_lag: int,
    train_episodes: set[int] | None = None,
) -> LeastSquaresDynamics:
    """Least squares on the same inputs the rollout gate uses.

    ``train_episodes`` is the latent fit's training split when the caller
    has one. An empty frame, or a split with no one-step pairs, predicts
    zeros so the planner still has a model to score.
    """
    feature_of = lambda row: log_row_features(row, delivery_lag=int(delivery_lag))
    if train_episodes is None and not frame.empty and "episode_seed" in frame.columns:
        episodes: set[int] | None = {int(seed) for seed in frame["episode_seed"].tolist()}
    else:
        episodes = train_episodes
    pairs = _one_step_pairs(frame, episodes, feature_of) if episodes else None
    fill_rate = _fill_ratio(frame)
    if pairs is None:
        width = feature_dim()
        keep = torch.zeros(width, dtype=torch.bool)
        keep[0] = True
        solution = torch.zeros(int(keep.sum()) + 2, target_dim())
        return LeastSquaresDynamics(
            solution,
            keep,
            delivery_lag=delivery_lag,
            fill_rate=fill_rate,
        )
    features, actions, targets = pairs
    solution, keep = _fit_ols(features, actions, targets)
    return LeastSquaresDynamics(
        solution.detach(),
        keep.detach(),
        delivery_lag=delivery_lag,
        fill_rate=fill_rate,
    )


@torch.no_grad()
def _latent_rollout_quantities(
    model: LatentDynamics,
    initial: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    if int(initial.shape[0]) > _ROLLOUT_CHUNK:
        parts = [
            _latent_rollout_quantities(model, initial[start:end], actions[start:end])
            for start, end in _chunks(int(initial.shape[0]))
        ]
        return torch.cat(parts, dim=0)
    model.encoder.eval()
    model.predictor.eval()
    model.probes.eval()
    embedding = model.encoder(model._norm_x(initial))
    steps: list[torch.Tensor] = []
    for step in range(actions.shape[1]):
        action = model._norm_a(actions[:, step : step + 1])
        embedding = model.predictor(embedding, action)
        steps.append(model._denorm_y(model.probes(embedding)))
    return torch.stack(steps, dim=1)


@torch.no_grad()
def _direct_rollout_quantities(
    model: DirectDynamics,
    initial: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    if int(initial.shape[0]) > _ROLLOUT_CHUNK:
        parts = [
            _direct_rollout_quantities(model, initial[start:end], actions[start:end])
            for start, end in _chunks(int(initial.shape[0]))
        ]
        return torch.cat(parts, dim=0)
    model.net.eval()
    raw = initial.clone()
    steps: list[torch.Tensor] = []
    for step in range(actions.shape[1]):
        granted = actions[:, step]
        pred = model._denorm_y(model.net(model._norm_x(raw), model._norm_a(granted.unsqueeze(-1))))
        steps.append(pred)
        _advance_observation(raw, pred[:, 0], pred[:, -1], granted)
    return torch.stack(steps, dim=1)


def _missing_report(horizon: int) -> RolloutAccuracy:
    missing = tuple(float("nan") for _ in range(horizon))
    return RolloutAccuracy(
        steps=tuple(range(1, horizon + 1)),
        latent=missing,
        direct=missing,
        ols=missing,
    )


def held_out_rollout_accuracy(
    latent: LatentDynamics,
    direct: DirectDynamics,
    frame: pd.DataFrame,
) -> RolloutAccuracy:
    """Per-step held-out rollout R² for the latent model, the direct predictor, and OLS.

    The episode split is the one the latent fit held out. OLS is fit on
    that split's training rows, from the same encoder inputs and filled
    quantity, and rolled forward the way the direct predictor is.
    """
    horizon = int(latent.model_unroll_steps)
    if latent.train_episodes or latent.val_episodes:
        train_ids = set(latent.train_episodes)
        val_ids = set(latent.val_episodes)
    elif frame.empty or "episode_seed" not in frame.columns:
        return _missing_report(horizon)
    else:
        train_ids, val_ids = _holdout_episodes(
            frame["episode_seed"].tolist(),
            _generator(0),
        )
    feature_of = latent._features
    windows = _score_windows(frame, val_ids, horizon, feature_of) if val_ids else None
    pairs = _one_step_pairs(frame, train_ids, feature_of) if train_ids else None
    if windows is None or pairs is None:
        return _missing_report(horizon)
    initial, actions, targets = windows
    features, pair_actions, pair_targets = pairs
    solution, keep = _fit_ols(features, pair_actions, pair_targets)
    return RolloutAccuracy(
        steps=tuple(range(1, horizon + 1)),
        latent=_step_scores(_latent_rollout_quantities(latent, initial, actions), targets),
        direct=_step_scores(_direct_rollout_quantities(direct, initial, actions), targets),
        ols=_step_scores(_ols_rollout(solution, keep, initial, actions), targets),
    )
