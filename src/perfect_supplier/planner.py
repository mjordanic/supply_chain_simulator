"""Model-predictive buyer for the perfect-supplier shop.

Each day the planner asks a forecaster for demand quantiles over the next
``HORIZON`` days and draws ``n_paths`` demand paths from them. Every candidate
is a joint reorder-point plan ``(cover R, service quantile z)``. On any
imagined day where some product's inventory position is below the
``z``-quantile of its demand over the next ``lag`` days (what must be on hand
or in transit before an order placed tomorrow can arrive), the shop orders
every product up to the ``z``-quantile of its demand over ``R + lag - 1`` days.
All products order on the same day, so one order fee covers the shop. Today's
order is the winning plan's day-0 order, which is zero when no product is
below its reorder point.

Inventory, the pipeline, cash, holding, the fee, and the shelf ration are the
simulator's own recursion, run on every path. A candidate's score is the mean
change in equity over the horizon, stock left at the end valued at unit cost.
Paths are shared by all candidates, so their differences are not sampling noise
between candidates.

Knobs (the defaults are the original planner):

- ``rho``: path coherence. Each product's path shares one Gaussian shock
  across days with equicorrelation ``rho`` in normal-score space (a Gaussian
  copula), so every day keeps its marginal quantiles while lead-time sums
  widen; a learned forecast's errors are mostly level errors shared by days.
- ``service_max``: extends the service grid with 0.998, 0.999 and 0.9995 up to
  this cap; the per-day newsvendor ratio reaches 0.9993 here.
- ``level_paths``: draw that many separate paths only for the reorder and
  order-up-to quantiles, so a high service level is not the largest of 64 sums.
  The candidates are still scored on the ``n_paths`` paths.

The forecaster's ``levels`` (default ``QUANTILES``) are its quantile levels.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from src.perfect_supplier.forecast import HORIZON, QUANTILES
from src.perfect_supplier.sim import Decision, Obs

REVIEWS = (1, 2, 3, 4, 5, 7, 10, 14, 21)
SERVICE = (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995)
TAIL_SERVICE = (0.998, 0.999, 0.9995)


def sample_paths(
    quantiles: np.ndarray, n_paths: int, rng: np.random.Generator, levels: np.ndarray = QUANTILES, rho: float = 0.0
) -> np.ndarray:
    """Demand paths ``(n_paths, H, P)`` by inverse-CDF on the piecewise-linear quantiles at ``levels``.

    With ``rho == 0`` days are drawn independently; otherwise through a
    Gaussian copula with equicorrelation ``rho`` across the days of a product.
    Levels below the first and above the last quantile take the end values.
    """
    H, P, Q = quantiles.shape
    if rho == 0:
        u = rng.random((n_paths, H, P))
    else:
        z = np.sqrt(rho) * rng.standard_normal((n_paths, 1, P)) + np.sqrt(1 - rho) * rng.standard_normal((n_paths, H, P))
        u = norm.cdf(z)
    idx = np.clip(np.searchsorted(levels, u) - 1, 0, Q - 2)
    lo, hi = levels[idx], levels[idx + 1]
    w = np.clip((u - lo) / (hi - lo), 0, 1)
    qh = np.arange(H)[None, :, None]
    qp = np.arange(P)[None, None, :]
    v_lo = quantiles[qh, qp, idx]
    v_hi = quantiles[qh, qp, idx + 1]
    return np.rint(v_lo + w * (v_hi - v_lo))


def _ration(want: np.ndarray, free: np.ndarray, budget: np.ndarray, unit_cost: float) -> np.ndarray:
    """:func:`src.perfect_supplier.sim.ration` on every (candidate, path) at once.

    ``want`` is ``(C, M, P)``; ``free`` and ``budget`` are ``(C, M)``.
    """
    limit = np.minimum(np.maximum(free, 0), np.maximum(np.floor(budget / unit_cost + 1e-9), 0))
    total = want.sum(axis=2)
    over = total > limit
    if not over.any():
        return want
    # Only the few (candidate, path) rows over their limit are rationed.
    w, lim = want[over], limit[over]
    share = w * (lim / np.maximum(total[over], 1))[:, None]
    out = np.floor(share)
    rest = lim - out.sum(axis=1)
    # Largest remainders first; ties go to the lower product index, as in the sim.
    order = np.argsort(-(share - out), axis=1, kind="stable")
    rank = np.argsort(order, axis=1, kind="stable")
    out += rank < rest[:, None]
    want = want.copy()
    want[over] = out
    return want


class Planner:
    def __init__(
        self,
        forecaster,
        name: str | None = None,
        *,
        n_paths: int = 64,
        reviews: tuple[int, ...] = REVIEWS,
        service: tuple[float, ...] = SERVICE,
        seed: int = 0,
        rho: float = 0.0,
        service_max: float | None = None,
        level_paths: int = 0,
    ):
        self.forecaster = forecaster
        self.name = name or f"planner_{forecaster.name}"
        self.n_paths = n_paths
        self.seed = seed
        self.rho, self.level_paths = rho, level_paths
        if service_max is not None:
            service = tuple(z for z in (*service, *TAIL_SERVICE) if z <= service_max)
        cands = [(r, z) for r in reviews for z in service]
        self.review = np.array([c[0] for c in cands])
        self.z = np.array([c[1] for c in cands])

    def decide(self, obs: Obs) -> Decision:
        rng = np.random.default_rng([self.seed, obs.day])
        quantiles = self.forecaster(obs)  # (H, P, Q)
        levels = getattr(self.forecaster, "levels", QUANTILES)
        paths = sample_paths(quantiles, self.n_paths, rng, levels, self.rho)  # (M, H, P)
        extra = sample_paths(quantiles, self.level_paths, rng, levels, self.rho) if self.level_paths else None
        score, first_order = self.imagine(obs, paths, extra)
        best = int(np.argmax(score))
        orders_today = first_order.sum(axis=1) > 0
        info = {
            "review": int(self.review[best]),
            "service": float(self.z[best]),
            "order_today": bool(orders_today[best]),
            "score": float(score[best]),
            "score_best_wait": float(score[~orders_today].max()) if (~orders_today).any() else np.nan,
            "score_best_order": float(score[orders_today].max()) if orders_today.any() else np.nan,
            "forecast_mean_7d": quantiles[:7, :, list(levels).index(0.5)].sum(axis=0),
        }
        return Decision(first_order[best], info)

    def _levels(self, paths: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
        """Reorder points and order-up-to levels, each ``(C, H, P)``.

        Row ``h`` is day ``t + h`` (row 0 is today); path row 0 is tomorrow.
        The reorder point covers days ``t + h + 1 .. t + h + lag`` and the
        order-up-to level covers ``t + h + 1 .. t + h + R + lag - 1``.
        """
        M, H, P = paths.shape
        csum = np.concatenate([np.zeros((M, 1, P)), np.cumsum(paths, axis=1)], axis=1)  # (M, H+1, P)
        rows = np.arange(H)

        def quantiles(span: int) -> dict[float, np.ndarray]:
            end = np.minimum(rows + span, H)
            covered = csum[:, end, :] - csum[:, rows, :]  # (M, H, P)
            # Short windows near the horizon end: scale up to the full span.
            days = (end - rows)[None, :, None]
            covered = np.sort(covered * (span / np.maximum(days, 1)), axis=0)
            out = {}
            for z in np.unique(self.z):
                # Empirical z-quantile, linear between order statistics.
                pos = z * (M - 1)
                lo = int(np.floor(pos))
                hi = min(lo + 1, M - 1)
                out[z] = covered[lo] + (pos - lo) * (covered[hi] - covered[lo])
            return out

        reorder = np.empty((len(self.review), H, P))
        up_to = np.empty((len(self.review), H, P))
        at_lag = quantiles(lag)
        for r in np.unique(self.review):
            at_cover = quantiles(int(r) + lag - 1)
            for z in np.unique(self.z):
                k = (self.review == r) & (self.z == z)
                reorder[k] = at_lag[z]
                up_to[k] = np.maximum(at_cover[z], at_lag[z])
        return reorder, up_to

    def imagine(
        self, obs: Obs, paths: np.ndarray, level_paths: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mean imagined equity change per candidate, and each candidate's order today.

        The reorder and order-up-to quantiles come from ``level_paths`` when given, else from ``paths``.
        """
        M, H, P = paths.shape
        C, L = len(self.review), obs.delivery_lag
        reorder, targets = self._levels(paths if level_paths is None else level_paths, L)  # (C, H, P) each
        on_hand = np.broadcast_to(obs.on_hand.astype(float), (C, M, P)).copy()
        # Sums over products as a matrix product: much faster than a reduce over a short last
        # axis, and exact, since every summand is a whole number of units.
        ones = np.ones(P)
        # Arrivals by horizon row: row k lands on day t + 1 + k.
        arriving = np.zeros((H + L + 1, C, M, P))
        arriving[: L - 1] = obs.pipeline[:, None, None, :].astype(float)
        in_transit = arriving.sum(axis=0)
        cash = np.full((C, M), float(obs.cash))
        start_equity = obs.cash + obs.unit_cost * float(obs.position.sum())
        first_order = np.zeros((C, P), dtype=np.int64)
        demand = np.ascontiguousarray(paths.transpose(1, 0, 2))  # (H, M, P)
        for h in range(H):
            # Day t + h: rows h >= 1 sell first; row 0 (today) has already sold.
            if h > 0:
                arrived = arriving[h - 1]
                on_hand += arrived
                in_transit -= arrived
                sold = np.minimum(demand[h - 1], on_hand)
                on_hand -= sold
                cash += obs.price * (sold @ ones)
            position = on_hand + in_transit  # (C, M, P)
            due = ((position < reorder[:, h, None, :]) @ ones) > 0  # (C, M)
            if due.any():
                want = np.maximum(np.rint(targets[:, h, None, :] - position), 0) * due[:, :, None]
                qty = _ration(want, obs.capacity - position @ ones, cash - obs.order_fee, obs.unit_cost)
                units = qty @ ones
                cash -= obs.unit_cost * units + obs.order_fee * (units > 0)
                arriving[h + L - 1] += qty
                in_transit += qty
                if h == 0:
                    # Today's order is the same on every path; take path 0.
                    first_order = qty[:, 0, :].astype(np.int64)
            cash -= obs.holding_rate * obs.unit_cost * (on_hand @ ones)
        stock = (on_hand + in_transit) @ ones
        equity = cash + obs.unit_cost * stock
        return (equity - start_equity).mean(axis=1), first_order
