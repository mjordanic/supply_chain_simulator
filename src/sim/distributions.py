"""Typed Distribution objects for lazily-sampled config values.

Each Distribution consumes a caller-supplied ``Random`` instance so the
caller controls determinism. JSON round-trip via ``to_dict`` / ``from_dict``
replaces the previous lambda-string config encoding.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from random import Random
from typing import Any, Sequence


class Distribution(ABC):
    """Abstract base for lazily-sampled values."""

    @abstractmethod
    def sample(self, rng: Random) -> Any:
        """Draw one sample using ``rng``."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dict tagged with a ``type`` key."""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Distribution":
        return distribution_from_dict(d)


@dataclass(frozen=True)
class Uniform(Distribution):
    low: float
    high: float

    def sample(self, rng: Random) -> float:
        return rng.uniform(self.low, self.high)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "uniform", "low": self.low, "high": self.high}


@dataclass(frozen=True)
class Normal(Distribution):
    mean: float
    std: float
    clip: tuple[float, float] | None = None

    def sample(self, rng: Random) -> float:
        x = rng.gauss(self.mean, self.std)
        if self.clip is not None:
            lo, hi = self.clip
            if x < lo:
                return lo
            if x > hi:
                return hi
        return x

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": "normal", "mean": self.mean, "std": self.std}
        if self.clip is not None:
            out["clip"] = [self.clip[0], self.clip[1]]
        return out


@dataclass(frozen=True)
class Choice(Distribution):
    options: Sequence[Any]
    weights: Sequence[float] | None = None

    def __post_init__(self) -> None:
        # Freeze sequences so equality is structural and instances stay hashable-ish.
        object.__setattr__(self, "options", tuple(self.options))
        if self.weights is not None:
            object.__setattr__(self, "weights", tuple(self.weights))
            if len(self.weights) != len(self.options):
                raise ValueError("Choice: weights length must match options length")

    def sample(self, rng: Random) -> Any:
        if self.weights is None:
            return rng.choice(list(self.options))
        return rng.choices(list(self.options), weights=list(self.weights), k=1)[0]

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": "choice", "options": list(self.options)}
        if self.weights is not None:
            out["weights"] = list(self.weights)
        return out


@dataclass(frozen=True)
class Constant(Distribution):
    value: Any

    def sample(self, rng: Random) -> Any:
        return self.value

    def to_dict(self) -> dict[str, Any]:
        return {"type": "constant", "value": self.value}


_REGISTRY: dict[str, type[Distribution]] = {
    "uniform": Uniform,
    "normal": Normal,
    "choice": Choice,
    "constant": Constant,
}


def distribution_from_dict(d: dict[str, Any]) -> Distribution:
    if "type" not in d:
        raise ValueError(f"Distribution: missing 'type' key in {d!r}")
    type_name = d["type"]
    if type_name not in _REGISTRY:
        raise ValueError(
            f"Distribution: unknown distribution type {type_name!r} "
            f"(known: {sorted(_REGISTRY)})"
        )
    payload = {k: v for k, v in d.items() if k != "type"}
    if type_name == "normal" and "clip" in payload:
        clip = payload["clip"]
        payload["clip"] = (clip[0], clip[1])
    return _REGISTRY[type_name](**payload)
