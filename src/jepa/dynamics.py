"""Per-SKU latent dynamics and the direct quantity predictor.

The encoder reads on-hand, in-transit, a sales window of length
``delivery_lag``, cash, and capacity. The Market multiplier and the
disruption type are not features. An action-conditioned predictor
unrolls open-loop for ``H = delivery_lag`` steps. The latent loss
matches a stopped target embedding at each step, plus an anti-collapse
penalty. Probes read a stopped embedding and regress next on-hand,
next in-transit, and sales over the step. Probe losses do not update
the encoder.
"""

from __future__ import annotations

import copy
from typing import Any

import pandas as pd
import torch
from torch import nn

_EMBED_DIM = 32
_EMA_TAU = 0.99


def sku_features(
    *,
    on_hand: float,
    in_transit: float,
    sales_window: list[float] | list[int],
    cash: float,
    capacity: float,
    delivery_lag: int,
) -> list[float]:
    """Observation vector the encoder is allowed to see.

    Layout: on-hand, in-transit, the sales window padded or trimmed to
    ``delivery_lag`` (oldest first), then cash and capacity. Nothing
    else — in particular not a Market multiplier or a disruption label.
    """
    lag = max(0, int(delivery_lag))
    window = [float(value) for value in list(sales_window)]
    window = window[-lag:] if lag else []
    if len(window) < lag:
        window = [0.0] * (lag - len(window)) + window
    return [float(on_hand), float(in_transit), *window, float(cash), float(capacity)]


def _feature_dim(delivery_lag: int) -> int:
    return 2 + max(0, int(delivery_lag)) + 2


def _groups(frame: pd.DataFrame):
    if frame.empty:
        return
    grouped = frame.groupby(["episode_seed", "product_id"], sort=True)
    for _, group in grouped:
        yield group.sort_values("tick", kind="mergesort")


def _units(value: Any) -> float:
    """Sum an arrival-age profile. The encoder still reads that total."""
    return float(sum(float(part) for part in value))


def _row_features(row: dict, delivery_lag: int) -> list[float]:
    return sku_features(
        on_hand=float(row["on_hand"]),
        in_transit=_units(row["in_transit"]),
        sales_window=list(row["sales_window"]),
        cash=float(row["cash"]),
        capacity=float(row["capacity"]),
        delivery_lag=delivery_lag,
    )


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
        dim = _feature_dim(self.delivery_lag)
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


class LatentDynamics(_ScaledDynamics):
    """Encoder, EMA target, open-loop predictor, and stopped probes."""

    def __init__(self, delivery_lag: int) -> None:
        super().__init__(delivery_lag)
        dim = _feature_dim(self.delivery_lag)
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
            empty_x = torch.zeros(0, 1, _feature_dim(self.delivery_lag))
            return empty_x, torch.zeros(0, 1)
        for group in _groups(frame):
            rows = list(group.to_dict("records"))
            features = [_row_features(row, horizon) for row in rows]
            if len(rows) < horizon + 1:
                continue
            for start in range(len(rows) - horizon):
                sequences.append(features[start : start + horizon + 1])
                actions.append(
                    [float(rows[start + step]["filled_qty"]) for step in range(horizon)]
                )
        if not sequences:
            return (
                torch.zeros(0, horizon + 1, _feature_dim(horizon)),
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
                features.append(_row_features(rows[index + 1], self.delivery_lag))
                targets.append(_targets(rows[index]))
        if not features:
            return torch.zeros(0, _feature_dim(self.delivery_lag)), torch.zeros(0, 3)
        return torch.tensor(features, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

    def _all_features(self, frame: pd.DataFrame) -> torch.Tensor:
        rows = [
            _row_features(row, self.delivery_lag)
            for group in _groups(frame)
            for row in group.to_dict("records")
        ]
        if not rows:
            return torch.zeros(0, _feature_dim(self.delivery_lag))
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
        self.net = _DirectNet(_feature_dim(self.delivery_lag))

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
                features.append(_row_features(current, self.delivery_lag))
                actions.append(float(current["filled_qty"]))
                targets.append(_targets(current))
        dim = _feature_dim(self.delivery_lag)
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
    """Feed predicted on-hand, in-transit, and sales forward.

    Cash and capacity stay at the decision-time observation, so a later
    holding charge cannot change the next decoded quantities.
    """

    def __init__(self, dynamics: DirectDynamics, features: list[list[float]]) -> None:
        self._dynamics = dynamics
        self._raw = torch.tensor(features, dtype=torch.float32)
        self._lag = dynamics.delivery_lag

    @torch.no_grad()
    def step(self, granted: list[float]) -> list[tuple[float, float, float]]:
        quantities = self._dynamics.predict_next(self._raw.tolist(), granted)
        if self._raw.numel() == 0 or not quantities:
            return quantities
        self._raw[:, 0] = torch.tensor([row[0] for row in quantities])
        self._raw[:, 1] = torch.tensor([row[1] for row in quantities])
        lag = self._lag
        if lag > 0:
            start = 2
            end = 2 + lag
            window = self._raw[:, start:end].clone()
            if lag > 1:
                self._raw[:, start : end - 1] = window[:, 1:]
            self._raw[:, end - 1] = torch.tensor([row[2] for row in quantities])
        return quantities
