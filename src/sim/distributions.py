"""Typed Distribution objects for lazily-sampled config values.

Scenario authors often want a field to take "some random value drawn at
sampling time" rather than a hardcoded scalar — e.g. ``severity`` on
``DisruptionParams`` may be a ``Normal(mean=1.0, std=0.2)`` rather than
a fixed ``1.0``. The Distribution hierarchy here lets a parameter dataclass
declare ``severity: Distribution`` and have the consumer call
``severity.sample(rng)`` whenever it actually needs a number.

Two responsibilities live here:

1. The ``Distribution`` ABC plus four concrete shapes (``Constant``,
   ``Uniform``, ``Normal``, ``Choice``).
2. JSON round-trip (``to_dict`` / ``distribution_from_dict``) so a
   ``Scenario`` can be persisted and reloaded without losing
   stochasticity information. Each subclass tags its dict with a
   ``"type"`` key registered in ``_REGISTRY``.

Every ``sample`` takes a caller-supplied ``Random`` instance so the
caller controls determinism. This is what replaces the previous
lambda-string config encoding, which couldn't be serialised cleanly.
"""

from __future__ import annotations

# Pure-stdlib imports throughout — distributions are lightweight and
# get imported by both the sim package and the LLM/world-builder, so
# we avoid any heavy dependency at this layer.
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field  # noqa: F401  (kept for downstream re-export hygiene)
from random import Random
from typing import Any, Sequence


class Distribution(ABC):
    """Abstract base for lazily-sampled values.

    Concrete subclasses implement ``sample(rng)`` (draw one value) and
    ``to_dict()`` (serialise with a discriminator). ``from_dict``
    deserialises via the type-name registry below.
    """

    @abstractmethod
    def sample(self, rng: Random) -> Any:
        """Draw one sample using ``rng``."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dict tagged with a ``type`` key."""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Distribution":
        """Dispatch to the right subclass via the ``type`` discriminator."""
        return distribution_from_dict(d)


@dataclass(frozen=True)
class Uniform(Distribution):
    """Continuous uniform distribution over ``[low, high]`` (inclusive)."""

    # Lower bound of the support.
    low: float
    # Upper bound of the support; expected ``high >= low`` (not enforced
    # so callers can choose to validate at the param-dataclass layer).
    high: float

    def sample(self, rng: Random) -> float:
        """Draw one float in ``[low, high]`` from the caller's ``rng``."""
        return rng.uniform(self.low, self.high)

    def to_dict(self) -> dict[str, Any]:
        """JSON shape: ``{type: 'uniform', low, high}``."""
        return {"type": "uniform", "low": self.low, "high": self.high}


@dataclass(frozen=True)
class Normal(Distribution):
    """Gaussian ``N(mean, std)`` with optional ``clip = (lo, hi)`` bounds."""

    # Mean of the underlying Gaussian.
    mean: float
    # Standard deviation. ``std == 0`` collapses to a degenerate point
    # mass at ``mean`` — equivalent to ``Constant`` but uses one extra
    # ``rng`` draw, so prefer ``Constant`` if the cost matters for CRN.
    std: float
    # Optional inclusive clip range. ``None`` ⇒ unbounded Gaussian.
    clip: tuple[float, float] | None = None

    def sample(self, rng: Random) -> float:
        """Draw one ``N(mean, std)`` sample, optionally clipped."""
        # One RNG draw — ``rng.gauss`` consumes one underlying state
        # advance, which keeps determinism predictable.
        x = rng.gauss(self.mean, self.std)
        if self.clip is not None:
            # Unpack clip bounds once and apply min/max — explicit
            # comparisons avoid the extra ``min(max(x, lo), hi)`` call
            # so the no-clip path stays a single function-call faster.
            lo, hi = self.clip
            if x < lo:
                return lo
            if x > hi:
                return hi
        return x

    def to_dict(self) -> dict[str, Any]:
        """JSON shape: ``{type: 'normal', mean, std, clip?: [lo, hi]}``."""
        out: dict[str, Any] = {"type": "normal", "mean": self.mean, "std": self.std}
        if self.clip is not None:
            # JSON has no native tuple — serialise as a 2-element list
            # and recover the tuple shape on deserialise.
            out["clip"] = [self.clip[0], self.clip[1]]
        return out


@dataclass(frozen=True)
class Choice(Distribution):
    """Discrete distribution over a fixed list of options.

    Supports optional non-uniform ``weights``. Anything hashable can be
    used as an option (strings, ints, floats, even small tuples).
    """

    # The fixed option set. Stored as a tuple after ``__post_init__`` so
    # the dataclass stays hashable and immutable.
    options: Sequence[Any]
    # Optional per-option weight. ``None`` ⇒ uniform.
    weights: Sequence[float] | None = None

    def __post_init__(self) -> None:
        # Freeze sequences so equality is structural and instances stay
        # hashable-ish — frozen dataclasses with mutable list members
        # would break hashing.
        object.__setattr__(self, "options", tuple(self.options))
        if self.weights is not None:
            object.__setattr__(self, "weights", tuple(self.weights))
            # Length parity is a structural error — fail loudly at
            # construction rather than silently misweighting samples.
            if len(self.weights) != len(self.options):
                raise ValueError("Choice: weights length must match options length")

    def sample(self, rng: Random) -> Any:
        """Return one option, weighted if ``weights`` was supplied."""
        if self.weights is None:
            # ``rng.choice`` consumes one underlying draw; ``list(...)``
            # is needed because ``rng.choice`` rejects tuples in some
            # older stdlib versions.
            return rng.choice(list(self.options))
        # ``rng.choices`` accepts the unfrozen sequences directly; ``k=1``
        # yields a singleton list — unwrap for the scalar return type.
        return rng.choices(list(self.options), weights=list(self.weights), k=1)[0]

    def to_dict(self) -> dict[str, Any]:
        """JSON shape: ``{type: 'choice', options, weights?}``."""
        out: dict[str, Any] = {"type": "choice", "options": list(self.options)}
        if self.weights is not None:
            out["weights"] = list(self.weights)
        return out


@dataclass(frozen=True)
class Constant(Distribution):
    """Degenerate distribution that always returns the same value.

    Used as a "no randomness" sentinel everywhere a typed
    ``Distribution`` is required — preferred over ``Normal(x, 0)``
    because it skips an RNG draw and stays CRN-clean.
    """

    # The fixed value to return. Any type (the field is typed ``Any``).
    value: Any

    def sample(self, rng: Random) -> Any:
        """Return ``value``. Does **not** consume an ``rng`` draw."""
        return self.value

    def to_dict(self) -> dict[str, Any]:
        """JSON shape: ``{type: 'constant', value}``."""
        return {"type": "constant", "value": self.value}


@dataclass(frozen=True)
class LogUniform(Distribution):
    """Log-uniform distribution over ``[low, high]`` (inclusive).

    Draws are uniform in log-space: ``sample(rng)`` returns
    ``exp(rng.uniform(log(low), log(high)))``, yielding a value that
    is uniform on a multiplicative scale rather than an additive one.

    Typical use: domain-randomisation parameters that span orders of
    magnitude, e.g. ``LogUniform(100, 10_000)`` for SKU capacity or
    ``LogUniform(10_000, 1_000_000)`` for opening balance.

    Requires ``low > 0`` and ``high > low`` — validated at construction
    so a typo in a hand-edited scenario JSON fails loudly.
    """

    # Lower bound of the support (strictly positive).
    low: float
    # Upper bound of the support (strictly greater than ``low``).
    high: float

    def __post_init__(self) -> None:
        if self.low <= 0:
            raise ValueError(
                f"LogUniform: low must be > 0, got low={self.low!r}"
            )
        if self.high <= self.low:
            raise ValueError(
                f"LogUniform: high must be > low, got low={self.low!r}, high={self.high!r}"
            )

    def sample(self, rng: Random) -> float:
        """Draw one log-uniformly distributed float in ``[low, high]``."""
        return math.exp(rng.uniform(math.log(self.low), math.log(self.high)))

    def to_dict(self) -> dict[str, Any]:
        """JSON shape: ``{type: 'log_uniform', low, high}``."""
        return {"type": "log_uniform", "low": self.low, "high": self.high}


# Discriminator registry consumed by ``distribution_from_dict`` and by
# ``scenario._deserialize`` to decide whether a JSON dict should be
# parsed as a ``Distribution`` or kept as a plain dict.
_REGISTRY: dict[str, type[Distribution]] = {
    "uniform": Uniform,
    "normal": Normal,
    "choice": Choice,
    "constant": Constant,
    "log_uniform": LogUniform,
}


def distribution_from_dict(d: dict[str, Any]) -> Distribution:
    """Rebuild the right ``Distribution`` subclass from a tagged dict.

    Inverse of ``Distribution.to_dict``. Raises ``ValueError`` on a
    missing or unknown ``type`` discriminator so a typo in a hand-edited
    scenario JSON fails loudly rather than silently becoming a plain
    dict at runtime.
    """
    if "type" not in d:
        raise ValueError(f"Distribution: missing 'type' key in {d!r}")
    # The discriminator field selects which subclass to instantiate.
    type_name = d["type"]
    if type_name not in _REGISTRY:
        raise ValueError(
            f"Distribution: unknown distribution type {type_name!r} "
            f"(known: {sorted(_REGISTRY)})"
        )
    # Strip the discriminator and pass the remaining fields as kwargs.
    payload = {k: v for k, v in d.items() if k != "type"}
    if type_name == "normal" and "clip" in payload:
        # Recover the tuple shape lost in JSON round-trip — see
        # ``Normal.to_dict`` for the matching serialise step.
        clip = payload["clip"]
        payload["clip"] = (clip[0], clip[1])
    return _REGISTRY[type_name](**payload)
