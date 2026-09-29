"""Per-SKU latent dynamics and the direct quantity predictor.

The encoder reads on-hand, an in-transit arrival-age profile, a sales
window of length ``SALES_WINDOW``, cash, capacity, and the nominal
delivery lead time. That window is the encoder's own knob: it does not
follow the sampled lead time, and neither does the feature dimension.
The Market multiplier, the disruption identity, and the calendar are
not features. An action-conditioned predictor unrolls open-loop for
``H = delivery_lag`` steps. The latent loss matches a stopped target
embedding at each step, plus an anti-collapse penalty. Probes read a
stopped embedding and regress next on-hand, next in-transit, and sales
over the step. Probe losses do not update the encoder.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
import torch
from torch import nn

_EMBED_DIM = 32
_EMA_TAU = 0.99

# Four weeks of daily sales. The longest sampled edge is a nominal lag
# of 6 plus the slow supplier's offset of 4, so this sits well above
# the delivery lead time and does not move when that lag is sampled.
SALES_WINDOW = 28

# Age 0 .. lead-1. The longest study edge is 10, so a unit stays in
# the profile until it arrives. Index is age in ticks, not a slot
# counted back from today.
ARRIVAL_AGE_SLOTS = 10


def feature_dim(sales_window_length: int = SALES_WINDOW) -> int:
    """Width of one encoder vector. Independent of the sampled lead time."""
    return 1 + ARRIVAL_AGE_SLOTS + max(0, int(sales_window_length)) + 3


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


def _units(value: Any) -> float:
    """Sum an arrival-age profile. Probes still regress that total."""
    return float(sum(float(part) for part in value))


def _targets(row: dict) -> list[float]:
    return [
        float(row["next_on_hand"]),
        _units(row["next_in_transit"]),
        float(row["next_sales"]),
    ]


class _EmptyRollout:
    def step(self, granted: list[float]) -> list[tuple[float, float, float]]:
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


class _Encoder(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class _Predictor(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim + 1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, embedding: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([embedding, action], dim=-1))


class _Probe(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.head = nn.Linear(embed_dim, 3)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.head(embedding)


class _DirectNet(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + 1, _EMBED_DIM),
            nn.ReLU(),
            nn.Linear(_EMBED_DIM, 3),
        )

    def forward(self, features: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([features, action], dim=-1))


class _ScaledDynamics(nn.Module):
    def __init__(self, delivery_lag: int) -> None:
        super().__init__()
        self.delivery_lag = int(delivery_lag)
        dim = feature_dim()
        self.register_buffer("x_mean", torch.zeros(dim))
        self.register_buffer("x_std", torch.ones(dim))
        self.register_buffer("y_mean", torch.zeros(3))
        self.register_buffer("y_std", torch.ones(3))
        self.register_buffer("a_mean", torch.zeros(1))
        self.register_buffer("a_std", torch.ones(1))

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

    def _as_quantities(self, normalized: torch.Tensor) -> list[tuple[float, float, float]]:
        decoded = self._denorm_y(normalized)
        return [tuple(float(value) for value in row) for row in decoded]

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

    def __init__(self, delivery_lag: int) -> None:
        super().__init__(delivery_lag)
        dim = feature_dim()
        self.encoder = _Encoder(dim, _EMBED_DIM)
        self.predictor = _Predictor(_EMBED_DIM)
        self.probes = _Probe(_EMBED_DIM)
        self.target_encoder = copy.deepcopy(self.encoder)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)

    def fit(self, frame: pd.DataFrame, *, epochs: int, seed: int, lr: float = 1e-3) -> None:
        torch.manual_seed(seed)
        observations, actions = self._windows(frame)
        _probe_x, probe_y = self._probe_pairs(frame)
        feature_rows = self._all_features(frame)
        self._set_stats(feature_rows, actions.reshape(-1), probe_y)
        self._fit_latent(observations, actions, epochs=epochs, lr=lr)
        self.fit_probes(frame, epochs=epochs, lr=lr)

    def fit_probes(self, frame: pd.DataFrame, *, epochs: int, lr: float = 1e-3) -> None:
        """Regress the three quantities from a stopped target embedding.

        The encoder is not in the optimizer and the embedding is
        detached, so this step cannot change encoder parameters.
        """
        features, targets = self._probe_pairs(frame)
        if features.numel() == 0 or epochs <= 0:
            return
        optimizer = torch.optim.Adam(self.probes.parameters(), lr=lr)
        self.encoder.eval()
        self.target_encoder.eval()
        self.probes.train()
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                embedding = self.target_encoder(self._norm_x(features))
            loss = nn.functional.mse_loss(self.probes(embedding), self._norm_y(targets))
            loss.backward()
            optimizer.step()

    @torch.no_grad()
    def predict_next(
        self,
        features: list[list[float]],
        granted: list[float],
    ) -> list[tuple[float, float, float]]:
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
        *,
        epochs: int,
        lr: float,
    ) -> None:
        if epochs <= 0 or observations.numel() == 0:
            return
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.predictor.parameters()),
            lr=lr,
        )
        observations_n = self._norm_x(observations)
        actions_n = self._norm_a(actions)
        self.encoder.train()
        self.predictor.train()
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = self._open_loop_loss(observations_n, actions_n)
            loss.backward()
            optimizer.step()
            self._ema()

    def _open_loop_loss(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Unroll the predictor for ``H`` steps without feeding true states back.

        ``observations`` is ``[batch, H + 1, features]`` and already
        normalized. The target at each step is the stopped target encoder.
        """
        horizon = actions.shape[1]
        embedding = self.encoder(observations[:, 0])
        loss = observations.new_zeros(())
        for step in range(horizon):
            embedding = self.predictor(embedding, actions[:, step : step + 1])
            target = self.target_encoder(observations[:, step + 1]).detach()
            loss = loss + nn.functional.mse_loss(embedding, target)
        flat = observations.reshape(-1, observations.shape[-1])
        return loss + _anti_collapse(self.encoder(flat))

    def _ema(self) -> None:
        with torch.no_grad():
            for online, target in zip(
                self.encoder.parameters(),
                self.target_encoder.parameters(),
            ):
                target.mul_(_EMA_TAU).add_(online, alpha=1.0 - _EMA_TAU)

    def _windows(self, frame: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        horizon = self.delivery_lag
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

    def _probe_pairs(self, frame: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        features: list[list[float]] = []
        targets: list[list[float]] = []
        for group in _groups(frame):
            rows = list(group.to_dict("records"))
            for index in range(len(rows) - 1):
                features.append(self._features(rows[index + 1]))
                targets.append(_targets(rows[index]))
        if not features:
            return torch.zeros(0, feature_dim()), torch.zeros(0, 3)
        return torch.tensor(features, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

    def _all_features(self, frame: pd.DataFrame) -> torch.Tensor:
        rows = [
            self._features(row)
            for group in _groups(frame)
            for row in group.to_dict("records")
        ]
        if not rows:
            return torch.zeros(0, feature_dim())
        return torch.tensor(rows, dtype=torch.float32)

    def _norm_y(self, targets: torch.Tensor) -> torch.Tensor:
        return (targets - self.y_mean) / self.y_std


class DirectDynamics(_ScaledDynamics):
    """Predict the three quantities from the observation and the granted qty.

    No latent target and no EMA encoder. The planner around it is the
    same shooter and the same ration as the latent arm.
    """

    def __init__(self, delivery_lag: int) -> None:
        super().__init__(delivery_lag)
        self.net = _DirectNet(feature_dim())

    def fit(self, frame: pd.DataFrame, *, epochs: int, seed: int, lr: float = 1e-3) -> None:
        torch.manual_seed(seed)
        features, actions, targets = self._pairs(frame)
        self._set_stats(features, actions, targets)
        if features.numel() == 0 or epochs <= 0:
            return
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        features_n = self._norm_x(features)
        actions_n = self._norm_a(actions)
        targets_n = (targets - self.y_mean) / self.y_std
        self.net.train()
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(self.net(features_n, actions_n), targets_n)
            loss.backward()
            optimizer.step()

    @torch.no_grad()
    def predict_next(
        self,
        features: list[list[float]],
        granted: list[float],
    ) -> list[tuple[float, float, float]]:
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
            return torch.zeros(0, dim), torch.zeros(0, 1), torch.zeros(0, 3)
        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(targets, dtype=torch.float32),
        )


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
    def step(self, granted: list[float]) -> list[tuple[float, float, float]]:
        dynamics = self._dynamics
        action = dynamics._norm_a(torch.tensor(granted, dtype=torch.float32).unsqueeze(-1))
        self._embedding = dynamics.predictor(self._embedding, action)
        return dynamics._as_quantities(dynamics.probes(self._embedding))


class _DirectRollout:
    """Feed predicted on-hand and sales forward, and age the pipeline.

    Cash, capacity, and the delivery lead time stay at the decision-time
    observation, so a later holding charge cannot change the next decoded
    quantities. The probe still emits in-transit as a sum. The next
    observation keeps ages: this step's grant becomes age 1, older cohorts
    move up one tick, and a cohort whose next age is not below the nominal
    lead drops out.
    """

    def __init__(self, dynamics: DirectDynamics, features: list[list[float]]) -> None:
        self._dynamics = dynamics
        self._raw = torch.tensor(features, dtype=torch.float32)

    @torch.no_grad()
    def step(self, granted: list[float]) -> list[tuple[float, float, float]]:
        quantities = self._dynamics.predict_next(self._raw.tolist(), granted)
        if self._raw.numel() == 0 or not quantities:
            return quantities
        self._raw[:, 0] = torch.tensor([row[0] for row in quantities])
        self._age_pipeline(granted)
        length = SALES_WINDOW
        profile_end = 1 + ARRIVAL_AGE_SLOTS
        if length > 0:
            start = profile_end
            end = profile_end + length
            window = self._raw[:, start:end].clone()
            if length > 1:
                self._raw[:, start : end - 1] = window[:, 1:]
            self._raw[:, end - 1] = torch.tensor([row[2] for row in quantities])
        return quantities

    def _age_pipeline(self, granted: list[float]) -> None:
        profile_end = 1 + ARRIVAL_AGE_SLOTS
        profile = self._raw[:, 1:profile_end]
        shifted = torch.zeros_like(profile)
        if ARRIVAL_AGE_SLOTS > 1:
            shifted[:, 1] = torch.tensor(granted, dtype=torch.float32)
        lead = self._raw[:, -1]
        for age in range(1, ARRIVAL_AGE_SLOTS - 1):
            dest = age + 1
            keep = lead > float(dest)
            shifted[keep, dest] = profile[keep, age]
        self._raw[:, 1:profile_end] = shifted
