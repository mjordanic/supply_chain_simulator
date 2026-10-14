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
"""

from __future__ import annotations

import numpy as np

from src.perfect_supplier.forecast import HORIZON, QUANTILES
from src.perfect_supplier.sim import Decision, Obs

REVIEWS = (1, 2, 3, 4, 5, 7, 10, 14, 21)
SERVICE = (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995)


def sample_paths(quantiles: np.ndarray, n_paths: int, rng: np.random.Generator) -> np.ndarray:
    """Demand paths ``(n_paths, H, P)`` by inverse-CDF on the piecewise-linear quantiles.

    Days are drawn independently. Levels below the first and above the last
    quantile take the end values.
    """
    H, P, Q = quantiles.shape
    u = rng.random((n_paths, H, P))
    idx = np.clip(np.searchsorted(QUANTILES, u) - 1, 0, Q - 2)
    lo, hi = QUANTILES[idx], QUANTILES[idx + 1]
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
    share = want * (limit / np.maximum(total, 1))[:, :, None]
    out = np.floor(share)
    rest = limit - out.sum(axis=2)
    # Largest remainders first; ties go to the lower product index, as in the sim.
    order = np.argsort(-(share - out), axis=2, kind="stable")
    rank = np.argsort(order, axis=2, kind="stable")
    out += rank < rest[:, :, None]
    return np.where(over[:, :, None], out, want)


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
    ):
        self.forecaster = forecaster
        self.name = name or f"planner_{forecaster.name}"
        self.n_paths = n_paths
        self.seed = seed
        cands = [(r, z) for r in reviews for z in service]
        self.review = np.array([c[0] for c in cands])
        self.z = np.array([c[1] for c in cands])

    def decide(self, obs: Obs) -> Decision:
        rng = np.random.default_rng([self.seed, obs.day])
        quantiles = self.forecaster(obs)  # (H, P, Q)
        paths = sample_paths(quantiles, self.n_paths, rng)  # (M, H, P)
        score, first_order = self.imagine(obs, paths)
        best = int(np.argmax(score))
        orders_today = first_order.sum(axis=1) > 0
        info = {
            "review": int(self.review[best]),
            "service": float(self.z[best]),
            "order_today": bool(orders_today[best]),
            "score": float(score[best]),
            "score_best_wait": float(score[~orders_today].max()) if (~orders_today).any() else np.nan,
            "score_best_order": float(score[orders_today].max()) if orders_today.any() else np.nan,
            "forecast_mean_7d": quantiles[:7, :, QUANTILES.tolist().index(0.5)].sum(axis=0),
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

    def imagine(self, obs: Obs, paths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mean imagined equity change per candidate, and each candidate's order today."""
        M, H, P = paths.shape
        C, L = len(self.review), obs.delivery_lag
        reorder, targets = self._levels(paths, L)  # (C, H, P) each
        on_hand = np.broadcast_to(obs.on_hand.astype(float), (C, M, P)).copy()
        # Arrivals by horizon row: row k lands on day t + 1 + k.
        arriving = np.zeros((C, M, H + L + 1, P))
        arriving[:, :, : L - 1, :] = obs.pipeline[None, None].astype(float)
        in_transit = arriving.sum(axis=2)
        cash = np.full((C, M), float(obs.cash))
        start_equity = obs.cash + obs.unit_cost * float(obs.position.sum())
        first_order = np.zeros((C, P), dtype=np.int64)
        demand = paths[None]  # (1, M, H, P)
        for h in range(H):
            # Day t + h: rows h >= 1 sell first; row 0 (today) has already sold.
            if h > 0:
                arrived = arriving[:, :, h - 1, :]
                on_hand += arrived
                in_transit -= arrived
                sold = np.minimum(demand[:, :, h - 1, :], on_hand)
                on_hand -= sold
                cash += obs.price * sold.sum(axis=2)
            position = on_hand + in_transit  # (C, M, P)
            due = (position < reorder[:, h, None, :]).any(axis=2)  # (C, M)
            if due.any():
                want = np.maximum(np.rint(targets[:, h, None, :] - position), 0) * due[:, :, None]
                qty = _ration(want, obs.capacity - position.sum(axis=2), cash - obs.order_fee, obs.unit_cost)
                units = qty.sum(axis=2)
                cash -= obs.unit_cost * units + obs.order_fee * (units > 0)
                arriving[:, :, h + L - 1, :] += qty
                in_transit += qty
                if h == 0:
                    # Today's order is the same on every path; take path 0.
                    first_order = qty[:, 0, :].astype(np.int64)
            cash -= obs.holding_rate * obs.unit_cost * on_hand.sum(axis=2)
        stock = (on_hand + in_transit).sum(axis=2)
        equity = cash + obs.unit_cost * stock
        return (equity - start_equity).mean(axis=1), first_order
