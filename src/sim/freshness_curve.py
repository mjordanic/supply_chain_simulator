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

# ``math.exp`` is the only external dependency — pure stdlib so this
# module is safe to import from anywhere without dragging in
# numerical-library side effects.
import math


def multiplier(
    alpha: float,
    decay: float,
    ticks_since_activation: int | float,
) -> float:
    """Return ``1 + α · exp(−τ / β)``.

    ``alpha == 0`` short-circuits to ``1.0`` so callers can disable the
    hype curve per-product without worrying about the decay parameter.

    Parameters
    ----------
    alpha
        Peak hype amplitude. ``0`` means "no hype" (e.g. staple).
    decay
        Hype decay length in ticks (``β`` in the formula). Must be
        positive — division by zero is the caller's bug.
    ticks_since_activation
        Number of ticks since the most recent activation in this store
        (``τ``).

    Returns
    -------
    float
        Multiplicative demand factor; ``1.0`` means "no effect".
    """
    # Short-circuit on the staple override. Saves an ``exp`` call and
    # — more importantly — makes the contract "alpha == 0 ⇒ multiplier
    # identically 1" exact regardless of decay or τ.
    if alpha == 0.0:
        return 1.0
    # Standard exponential decay: at τ = 0 the multiplier peaks at
    # ``1 + α``; as τ → ∞ it asymptotes back to 1.
    return 1.0 + alpha * math.exp(-float(ticks_since_activation) / decay)


__all__ = ["multiplier"]
