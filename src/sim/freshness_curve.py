"""Pure-function freshness multiplier (issue 03).

Encapsulates the per-(store, product) hype curve

    m(τ) = 1 + α · exp(−τ / β)

where ``τ`` is ticks since the product was last activated in the store,
``α`` controls peak hype magnitude, and ``β`` controls how quickly the
hype decays.

Pure function, no internal state. ``Store.freshness_multiplier``
delegates here. The single ``multiplier`` entry point is the swap-point
for a future Bass-shaped diffusion or polynomial curve — keep it
isolated from ``Store`` / ``Market`` / ``Registry`` so the math is
testable on its own.
"""

from __future__ import annotations

import math


def multiplier(
    alpha: float,
    decay: float,
    ticks_since_activation: int | float,
) -> float:
    """Return ``1 + α · exp(−τ / β)``.

    ``alpha == 0`` short-circuits to ``1.0`` so callers can disable the
    hype curve per-product without worrying about the decay parameter.
    """
    if alpha == 0.0:
        return 1.0
    return 1.0 + alpha * math.exp(-float(ticks_since_activation) / decay)


__all__ = ["multiplier"]
