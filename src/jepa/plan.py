"""Fit a per-SKU predictor and plan replenishment orders with it.

``decide`` spends a fixed budget of cover proposals. The published
search is factored cross-entropy: a categorical over ``COVERS`` for each
SKU, updated from the elite joint samples. ``search="uniform"`` draws
that same budget of joint covers independently and is the baseline the
factored search has to beat. Priority is not part of the draw.
``ration="priority"`` sorts SKUs by the marginal imagined profit of one
unit of shared capacity. ``ration="permutation"`` draws one order from
``policy_rng``. ``ration="fair_share"`` uses the textbook split. The
winner is the proposal with the highest imagined score over
``horizon_multiple * delivery_lag`` steps (``horizon_multiple`` defaults
to 4). Each step's profit is revenue minus purchase cost minus holding
cost minus the order fee. Ending inventory position — on-hand plus
in-transit — is then credited at unit cost, so unsold stock is not
confiscated at the window boundary. There is no stockout penalty and no
learned reward head. Imagined cash uses the holding rate under test and
is the cash the next ration sees. The terminal credit is not cash.
Imagined arrivals and imagined purchase cash follow the predictor's fill
rate; the emitted order stays the full grant.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, Mapping

import pandas as pd
import torch

from src.jepa.dynamics import (
    ARRIVAL_AGE_SLOTS,
    PLANNING_HORIZON_MULTIPLE,
    REDUCED_DEPTH,
    REDUCED_DIRECT_HIDDEN,
    REDUCED_EMBED_DIM,
    DirectDynamics,
    LatentDynamics,
    RolloutAccuracy,
    arrival_age_profile,
    direct_hidden_matching,
    held_out_rollout_accuracy,
    parameter_count,
    sku_features,
)
from src.jepa.policy import COVERS, _flatten_pending, _priority_ration, _suppliers_and_costs
from src.sim.policy import (
    MIN_ORDER_FLOOR,
    OPENING_BUDGET_PCT,
    IntermediatePolicy,
    MultiSupplierTextbookPolicy,
    _allocate_two_pass_fair_share,
    _estimate_rate,
    cold_start_pilot_qty,
    direct_offers,
)

DEFAULT_BUDGET = 64
DEFAULT_HORIZON_MULTIPLE = PLANNING_HORIZON_MULTIPLE
_CEM_ELITE_FRACTION = 0.25
_CEM_SMOOTHING = 0.7


def _cem_waves(budget: int, n_covers: int) -> list[int]:
    """Split a proposal budget into cross-entropy waves.

    A budget too small to try every cover is a single wave. A budget of
    at least four covers per wave uses four waves; otherwise two.
    """
    if budget <= 0:
        return []
    if budget <= n_covers:
        return [budget]
    n_waves = 4 if budget >= n_covers * 4 else 2
    base, extra = divmod(budget, n_waves)
    return [base + (1 if index < extra else 0) for index in range(n_waves)]


def _mix_cover_weights(
    weights: dict[str, list[float]],
    elites: list[dict[str, int]],
) -> dict[str, list[float]]:
    """Pull each SKU's categorical toward the covers in the elite set."""
    index_of = {cover: index for index, cover in enumerate(COVERS)}
    n_covers = len(COVERS)
    updated: dict[str, list[float]] = {}
    for pid, old in weights.items():
        counts = [0.0] * n_covers
        for covers in elites:
            counts[index_of[int(covers[pid])]] += 1.0
        total = float(len(elites))
        mixed = [
            (1.0 - _CEM_SMOOTHING) * previous + _CEM_SMOOTHING * (count / total)
            for previous, count in zip(old, counts, strict=True)
        ]
        scale = sum(mixed)
        updated[pid] = [value / scale for value in mixed]
    return updated


def _in_transit_units(value: Any) -> float:
    """Pipeline units. An arrival-age profile is summed; a bare total is itself."""
    if isinstance(value, (int, float)):
        return float(value)
    return float(sum(float(part) for part in value))


def _decision_profile(obs: Mapping[str, Any], pid: str, pending_units: int) -> list[int] | int:
    """Arrival-age profile at ``decide`` time, or the pending total if ages are unknown.

    ``in_transit_lots`` is the buyer's own pipeline, ``(order_tick, qty, lead)``.
    A bare total is placed by ``sku_features`` (age 0 empty, total at age 1).
    """
    lots = (obs.get("in_transit_lots") or {}).get(pid)
    if lots:
        return arrival_age_profile(lots, int(obs.get("tick") or 0), ARRIVAL_AGE_SLOTS)
    return int(pending_units)


class ReplenishmentPlanner(IntermediatePolicy):
    """Ordering-only buyer that scores proposals with a frozen predictor.

    ``ration="priority"`` ranks SKUs by marginal imagined profit per unit
    of free capacity and fills in that order. ``ration="permutation"``
    draws one priority from ``policy_rng`` and is the ablation.
    ``ration="fair_share"`` splits with the textbook allocator. All three
    search covers the same way. ``search="cem"`` is factored
    cross-entropy. ``search="uniform"`` is joint uniform draws of the
    same budget. ``retarget`` builds another planner on the same weights
    at a new holding rate and does not fit. Each recorded proposal keeps
    every draw and ``chosen``, the covers and priority that won. Cold
    start matches the textbook anchor: a SKU with no history, zero
    position, and a zero censored-sales rate is ordered
    ``cold_start_pilot_qty`` at ``OPENING_BUDGET_PCT``. The supplier
    split is cheapest-first with ``MIN_ORDER_FLOOR``. The opening
    order's fee and purchase use those routed lines. Later imagined
    steps still price the cheapest offer, because the offer book is
    not rolled forward. The imagined transition is conditioned on the
    predictor's fill rate.
    """

    def __init__(
        self,
        dynamics: LatentDynamics | DirectDynamics,
        *,
        policy_seed: int | None = None,
        delivery_lag: int | None = None,
        budget: int = DEFAULT_BUDGET,
        holding_rate: float = 0.01,
        order_fee: float = 50.0,
        ration: str = "priority",
        horizon_multiple: int = DEFAULT_HORIZON_MULTIPLE,
        search: str = "cem",
    ) -> None:
        super().__init__(policy_seed=policy_seed)
        if ration not in ("priority", "permutation", "fair_share"):
            raise ValueError(f"unknown ration {ration!r}")
        if search not in ("cem", "uniform"):
            raise ValueError(f"unknown search {search!r}")
        if int(horizon_multiple) < 1:
            raise ValueError("horizon_multiple must be at least 1")
        self.dynamics = dynamics
        self.delivery_lag = int(dynamics.delivery_lag if delivery_lag is None else delivery_lag)
        self.budget = int(budget)
        self.holding_rate = float(holding_rate)
        self.order_fee = float(order_fee)
        self.ration = ration
        self.search = search
        self.horizon_multiple = int(horizon_multiple)
        self.min_order = 0
        self.per_supplier_min_order_floor = MIN_ORDER_FLOOR
        self.routing_strategy = None
        self.opening_budget_pct = OPENING_BUDGET_PCT
        self.sales_log: dict[str, list[int]] = {}
        self.proposals: list[dict[str, Any]] = []

    def retarget(self, holding_rate: float) -> ReplenishmentPlanner:
        """Same weights and ``policy_seed``, new holding rate. Does not fit."""
        return ReplenishmentPlanner(
            self.dynamics,
            policy_seed=self.policy_seed,
            delivery_lag=self.delivery_lag,
            budget=self.budget,
            holding_rate=holding_rate,
            order_fee=self.order_fee,
            ration=self.ration,
            horizon_multiple=self.horizon_multiple,
            search=self.search,
        )

    def fit_probes(self, frame: pd.DataFrame, *, epochs: int) -> None:
        fit = getattr(self.dynamics, "fit_probes", None)
        if fit is None:
            raise TypeError("this predictor has no probes")
        fit(frame, epochs=epochs)

    def decide(
        self,
        obs_intermediate: Mapping[str, Any],
        central_table: Any,
    ) -> dict[str, Any]:
        inventory = {
            pid: int(qty)
            for pid, qty in dict(obs_intermediate.get("inventory") or {}).items()
        }
        pids = sorted(inventory)
        pending = _flatten_pending(obs_intermediate.get("pending") or {})
        observed = dict(obs_intermediate.get("observed_sales") or {})
        known = set(self.sales_log)
        for pid in pids:
            self.sales_log.setdefault(pid, []).append(int(observed.get(pid, 0)))
        rates_now = _estimate_rate(
            {pid: self.sales_log[pid] for pid in pids},
            max(1, self.delivery_lag),
            {pid: [] for pid in pids},
        )
        cold = {
            pid
            for pid in pids
            if pid not in known
            and inventory[pid] + pending.get(pid, 0) == 0
            and float(rates_now.get(pid, 0.0)) == 0.0
        }

        cash = float(obs_intermediate.get("cash") or 0.0)
        capacity = int(obs_intermediate.get("capacity") or 0)
        profiles = {
            pid: _decision_profile(obs_intermediate, pid, pending.get(pid, 0)) for pid in pids
        }
        suppliers, unit_costs = _suppliers_and_costs(pids, obs_intermediate, central_table)
        base = obs_intermediate.get("base_prices") or obs_intermediate.get("list_prices") or {}
        prices = {pid: float(base.get(pid, 0.0)) for pid in pids}

        direct = obs_intermediate.get("direct_supplier_ids")
        delivery_lags = obs_intermediate.get("delivery_lags") or {}
        priority = self._priority(
            pids,
            inventory,
            pending,
            cash,
            capacity,
            suppliers,
            unit_costs,
            prices,
            cold,
            direct,
            central_table,
            profiles,
            delivery_lags,
        )
        draws: list[dict[str, Any]] = []
        best_score: float | None = None
        chosen_covers: dict[str, int] = {}
        opening_order: dict[str, list[tuple[str, int]]] = {}

        def consider(covers: dict[str, int]) -> float:
            nonlocal best_score, chosen_covers, opening_order
            total, _grant, order = self._score(
                pids,
                covers,
                priority,
                inventory,
                pending,
                cash,
                capacity,
                suppliers,
                unit_costs,
                prices,
                cold,
                direct,
                central_table,
                profiles,
                delivery_lags,
            )
            draws.append({"covers": dict(covers), "priority": list(priority)})
            if best_score is None or total > best_score:
                best_score = total
                chosen_covers = dict(covers)
                opening_order = order
            return total

        if pids and self.budget > 0:
            if self.search == "uniform":
                self._uniform_covers(pids, consider)
            else:
                self._cem_covers(pids, consider)

        chosen_priority = list(priority)

        order = {pid: lines for pid, lines in opening_order.items() if lines}
        action = {
            "order": order,
            "list_price": prices,
            "min_order_imposed": {pid: 0 for pid in pids},
            "promotions": {},
        }
        self.proposals.append(
            {
                "tick": int(obs_intermediate.get("tick", len(self.proposals))),
                "draws": draws,
                "chosen": {"covers": chosen_covers, "priority": chosen_priority},
                "score": None if best_score is None else float(best_score),
                "action": action,
            }
        )
        return action

    def _score(
        self,
        pids: list[str],
        covers: Mapping[str, int],
        priority: list[str],
        inventory: Mapping[str, int],
        pending: Mapping[str, int],
        cash: float,
        capacity: int,
        suppliers: Mapping[str, str | None],
        unit_costs: Mapping[str, float],
        prices: Mapping[str, float],
        cold: set[str],
        direct: Any,
        central_table: Any,
        profiles: Mapping[str, list[int] | int],
        delivery_lags: Mapping[str, float],
        opening_desired: Mapping[str, int] | None = None,
    ) -> tuple[float, dict[str, int], dict[str, list[tuple[str, int]]]]:
        on_hand = {pid: int(inventory.get(pid, 0)) for pid in pids}
        in_transit = {pid: int(pending.get(pid, 0)) for pid in pids}
        sales_hist = {pid: list(self.sales_log.get(pid, [])) for pid in pids}
        cash_now = float(cash)
        total = 0.0
        opening_grant: dict[str, int] | None = None
        opening_order: dict[str, list[tuple[str, int]]] = {}
        # One encode of the decision-time window. Later steps advance the
        # predicted embedding (or, for the direct arm, the predicted
        # inventory) and do not re-read cash, so the holding rate changes
        # the score and the ration but not the transition itself.
        rollout = self.dynamics.open_rollout(
            [
                sku_features(
                    on_hand=on_hand[pid],
                    in_transit=profiles.get(pid, 0),
                    sales_window=sales_hist[pid],
                    cash=cash_now,
                    capacity=capacity,
                    delivery_lag=float(delivery_lags.get(pid, self.delivery_lag)),
                )
                for pid in pids
            ]
        )
        fill_rate = self._fill_rate()
        for _step in range(self.horizon_multiple * self.delivery_lag):
            holding = self.holding_rate * sum(
                on_hand[pid] * float(unit_costs.get(pid, 0.0)) for pid in pids
            )
            rates = _estimate_rate(
                {pid: sales_hist[pid] for pid in pids},
                max(1, self.delivery_lag),
                {pid: [] for pid in pids},
            )
            free_space = max(0, int(capacity) - sum(on_hand.values()) - sum(in_transit.values()))
            if opening_desired is not None:
                # A one-unit probe. Later steps stay quiet so the gap is
                # the marginal unit, and the ration does not zero it out
                # before it can be scored.
                shipped = {
                    pid: int(opening_desired.get(pid, 0)) if _step == 0 else 0 for pid in pids
                }
            else:
                desired: dict[str, int] = {}
                for pid in pids:
                    if suppliers.get(pid) is None:
                        desired[pid] = 0
                        continue
                    position = on_hand[pid] + in_transit[pid]
                    if _step == 0 and pid in cold:
                        desired[pid] = cold_start_pilot_qty(
                            cash,
                            len(pids),
                            float(unit_costs.get(pid, 0.0)),
                            free_space,
                            self.opening_budget_pct,
                        )
                        continue
                    target = round(
                        int(covers.get(pid, 0)) * float(rates.get(pid, 0.0)) * self.delivery_lag
                    )
                    desired[pid] = max(0, int(target) - position)
                spendable = max(0.0, cash_now - holding)
                granted = self._ration(desired, priority, free_space, spendable, unit_costs)
                shipped = {pid: int(granted.get(pid, 0)) for pid in pids}
            purchase = 0.0
            used: set[str] = set()
            if _step == 0:
                routed = {
                    pid: self._place(pid, shipped[pid], direct, central_table) for pid in pids
                }
                opening_order = {
                    pid: [(sid, qty) for sid, qty, _price in lines]
                    for pid, lines in routed.items()
                    if lines
                }
                for pid, lines in routed.items():
                    shipped[pid] = sum(qty for _sid, qty, _price in lines)
                    # Cash moves with the fill. The request above is what
                    # the order emits; Allocation is what the log trained on.
                    purchase += sum(qty * price for _sid, qty, price in lines) * fill_rate
                    used.update(sid for sid, qty, _price in lines if qty > 0)
            else:
                for pid in pids:
                    qty = shipped[pid]
                    purchase += qty * float(unit_costs.get(pid, 0.0)) * fill_rate
                    supplier_id = suppliers.get(pid)
                    if qty > 0 and supplier_id is not None:
                        used.add(supplier_id)
            predicted = rollout.step([float(shipped[pid]) * fill_rate for pid in pids])
            revenue = 0.0
            for pid, (next_on_hand, next_in_transit, next_sales) in zip(pids, predicted, strict=True):
                revenue += max(0.0, float(next_sales)) * float(prices.get(pid, 0.0))
                on_hand[pid] = max(0, int(round(next_on_hand)))
                in_transit[pid] = max(0, int(round(_in_transit_units(next_in_transit))))
                sales_hist[pid].append(max(0, int(round(next_sales))))
            fee = self.order_fee * len(used)
            profit = revenue - purchase - holding - fee
            total += profit
            if opening_grant is None:
                opening_grant = dict(shipped)
            cash_now = cash_now + profit
        # Purchase cost is charged when the order is imagined. Crediting
        # unit cost on ending on-hand and in-transit puts that cash back,
        # so the window does not confiscate stock that has not sold yet.
        # The credit is not cash the next ration can spend.
        total += sum(
            (on_hand[pid] + in_transit[pid]) * float(unit_costs.get(pid, 0.0))
            for pid in pids
        )
        return total, opening_grant or {pid: 0 for pid in pids}, opening_order

    def _fill_rate(self) -> float:
        """Fraction of a granted unit the imagined transition treats as filled."""
        rate = getattr(self.dynamics, "fill_rate", 1.0)
        if isinstance(rate, torch.Tensor):
            return float(rate.detach().item())
        return float(rate)

    def _priority(
        self,
        pids: list[str],
        inventory: Mapping[str, int],
        pending: Mapping[str, int],
        cash: float,
        capacity: int,
        suppliers: Mapping[str, str | None],
        unit_costs: Mapping[str, float],
        prices: Mapping[str, float],
        cold: set[str],
        direct: Any,
        central_table: Any,
        profiles: Mapping[str, list[int] | int],
        delivery_lags: Mapping[str, float],
    ) -> list[str]:
        if self.ration == "permutation":
            # Own stream so the cover search sees the same ``policy_rng``
            # draws as the density and fair-share arms.
            if not pids:
                return []
            return list(Random(self.policy_seed).sample(list(pids), len(pids)))
        if self.ration == "fair_share":
            return list(pids)
        return self._density_order(
            pids,
            inventory,
            pending,
            cash,
            capacity,
            suppliers,
            unit_costs,
            prices,
            cold,
            direct,
            central_table,
            profiles,
            delivery_lags,
        )

    def _density_order(
        self,
        pids: list[str],
        inventory: Mapping[str, int],
        pending: Mapping[str, int],
        cash: float,
        capacity: int,
        suppliers: Mapping[str, str | None],
        unit_costs: Mapping[str, float],
        prices: Mapping[str, float],
        cold: set[str],
        direct: Any,
        central_table: Any,
        profiles: Mapping[str, list[int] | int],
        delivery_lags: Mapping[str, float],
    ) -> list[str]:
        """SKUs sorted by imagined profit of one capacity unit, highest first.

        One granted unit consumes one unit of the shared pool, so the
        horizon profit of that unit is already a density. Equal densities
        break toward the earlier product id and do not touch ``policy_rng``.
        """
        if not pids:
            return []
        ranked: list[tuple[float, str]] = []
        for pid in pids:
            grant = {other: 0 for other in pids}
            grant[pid] = 1
            profit, _granted, _order = self._score(
                pids,
                {other: 0 for other in pids},
                list(pids),
                inventory,
                pending,
                cash,
                capacity,
                suppliers,
                unit_costs,
                prices,
                cold,
                direct,
                central_table,
                profiles,
                delivery_lags,
                opening_desired=grant,
            )
            ranked.append((profit, pid))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [pid for _profit, pid in ranked]

    def _uniform_covers(self, pids: list[str], consider: Any) -> None:
        for _ in range(self.budget):
            consider({pid: int(self.policy_rng.choice(COVERS)) for pid in pids})

    def _cem_covers(self, pids: list[str], consider: Any) -> None:
        n_covers = len(COVERS)
        weights = {pid: [1.0 / n_covers] * n_covers for pid in pids}
        first_wave = True
        for population in _cem_waves(self.budget, n_covers):
            scored: list[tuple[float, dict[str, int]]] = []
            # Score each cover once with every SKU on that cover, before
            # the categorical moves. Later samples are draws from it.
            n_constant = n_covers if first_wave and population >= n_covers else 0
            first_wave = False
            for index in range(population):
                if index < n_constant:
                    cover = int(COVERS[index])
                    covers = {pid: cover for pid in pids}
                else:
                    covers = {
                        pid: int(self.policy_rng.choices(COVERS, weights=weights[pid], k=1)[0])
                        for pid in pids
                    }
                scored.append((consider(covers), covers))
            elite_count = max(1, int(population * _CEM_ELITE_FRACTION))
            elites = [
                covers
                for _score, covers in sorted(scored, key=lambda item: item[0], reverse=True)[:elite_count]
            ]
            weights = _mix_cover_weights(weights, elites)

    def _place(
        self,
        pid: str,
        qty: int,
        direct: Any,
        central_table: Any,
    ) -> list[tuple[str, int, float]]:
        """Route ``qty`` the way the textbook anchor does, with each line's price."""
        offers = direct_offers(pid, direct, central_table)
        if qty <= 0 or not offers:
            return []
        splits = MultiSupplierTextbookPolicy._split_across_suppliers(
            pid,
            qty,
            [supplier_id for supplier_id, _offer in offers],
            central_table,
            buyer_min_order_floor=self.per_supplier_min_order_floor,
            routing_strategy=self.routing_strategy,
        )
        price_of = {supplier_id: float(offer.list_price) for supplier_id, offer in offers}
        return [(supplier_id, line_qty, price_of[supplier_id]) for supplier_id, line_qty in splits]

    def _ration(
        self,
        desired: dict[str, int],
        priority: list[str],
        free_space: int,
        cash: float,
        unit_costs: Mapping[str, float],
    ) -> dict[str, int]:
        if self.ration == "fair_share":
            return _allocate_two_pass_fair_share(
                desired,
                {pid: float(unit_costs.get(pid, 0.0)) for pid in desired},
                free_space,
                cash,
                min_qty=0,
                pilot_pids=set(),
            )
        return _priority_ration(desired, priority, free_space, cash, unit_costs)


def _unroll_steps(
    delivery_lag: int,
    model_unroll_steps: int | None,
    horizon_multiple: int,
) -> int:
    """Planning horizon, unless ``model_unroll_steps`` is set on its own."""
    if model_unroll_steps is None:
        return int(horizon_multiple) * int(delivery_lag)
    return int(model_unroll_steps)


def fit_latent(
    frame: pd.DataFrame,
    *,
    delivery_lag: int = 3,
    epochs: int = 20,
    seed: int = 0,
    lr: float = 1e-3,
    artifact_dir: str | Path | None = None,
    model_unroll_steps: int | None = None,
    embed_dim: int = REDUCED_EMBED_DIM,
    depth: int = REDUCED_DEPTH,
    **planner_kwargs: Any,
) -> ReplenishmentPlanner:
    """Fit the latent predictor on a transition log and wrap it as a Policy.

    The default width and depth are the reduced model. The published fit
    command passes the published capacity instead. ``seed`` fixes the
    initial weights and the shuffled passes.
    """
    torch.manual_seed(seed)
    multiple = int(planner_kwargs.pop("horizon_multiple", DEFAULT_HORIZON_MULTIPLE))
    steps = _unroll_steps(delivery_lag, model_unroll_steps, multiple)
    dynamics = LatentDynamics(
        delivery_lag=delivery_lag,
        model_unroll_steps=steps,
        embed_dim=embed_dim,
        depth=depth,
    )
    dynamics.fit(frame, epochs=epochs, seed=seed, lr=lr, artifact_dir=artifact_dir)
    return ReplenishmentPlanner(
        dynamics,
        delivery_lag=delivery_lag,
        horizon_multiple=multiple,
        **planner_kwargs,
    )


def fit_direct(
    frame: pd.DataFrame,
    *,
    delivery_lag: int = 3,
    epochs: int = 20,
    seed: int = 0,
    lr: float = 1e-3,
    artifact_dir: str | Path | None = None,
    model_unroll_steps: int | None = None,
    hidden_dim: int = REDUCED_DIRECT_HIDDEN,
    depth: int = REDUCED_DEPTH,
    **planner_kwargs: Any,
) -> ReplenishmentPlanner:
    """Fit the direct predictor on a transition log and wrap it as a Policy.

    Training stays one-step. ``model_unroll_steps`` is the horizon the
    rollout report scores, the same planning horizon as the latent fit.
    The default width is the reduced predictor. The published fit passes
    the hidden width matched to the published latent model. ``seed`` fixes
    the initial weights and the shuffled passes.
    """
    torch.manual_seed(seed)
    multiple = int(planner_kwargs.pop("horizon_multiple", DEFAULT_HORIZON_MULTIPLE))
    steps = _unroll_steps(delivery_lag, model_unroll_steps, multiple)
    dynamics = DirectDynamics(
        delivery_lag=delivery_lag,
        model_unroll_steps=steps,
        hidden_dim=hidden_dim,
        depth=depth,
    )
    dynamics.fit(frame, epochs=epochs, seed=seed, lr=lr, artifact_dir=artifact_dir)
    return ReplenishmentPlanner(
        dynamics,
        delivery_lag=delivery_lag,
        horizon_multiple=multiple,
        **planner_kwargs,
    )


@dataclass(frozen=True)
class CapacityPoint:
    """One width and depth on the model-capacity curve.

    ``accuracy`` is the held-out per-step rollout from the planning-horizon
    harness. ``parameters`` counts the latent model, including the frozen
    EMA encoder. The direct predictor at this point uses ``direct_hidden``.
    """

    embed_dim: int
    depth: int
    parameters: int
    direct_hidden: int
    direct_parameters: int
    accuracy: RolloutAccuracy


@dataclass(frozen=True)
class DataPoint:
    """One episode count on the data-scaling curve."""

    episodes: int
    accuracy: RolloutAccuracy


def _episode_prefix(frame: pd.DataFrame, episodes: int) -> pd.DataFrame:
    """The first ``episodes`` seeds, in seed order. Later counts contain earlier ones."""
    seeds = sorted(int(seed) for seed in frame["episode_seed"].unique())
    count = int(episodes)
    if count < 1:
        raise ValueError("episode count must be at least 1")
    if count > len(seeds):
        raise ValueError(
            f"episode count {count} exceeds the {len(seeds)} episodes in the log"
        )
    keep = set(seeds[:count])
    return frame.loc[frame["episode_seed"].isin(keep)].reset_index(drop=True)


def model_capacity_curve(
    frame: pd.DataFrame,
    sizes: Sequence[tuple[int, int]],
    *,
    delivery_lag: int,
    epochs: int,
    seed: int = 0,
    model_unroll_steps: int | None = None,
    artifact_dir: str | Path | None = None,
    lr: float = 1e-3,
) -> tuple[CapacityPoint, ...]:
    """Held-out rollout accuracy at each ``(embed_dim, depth)``.

    The direct predictor at each point is widened to the latent trainable
    count, at the same depth. Both fits use the planning-horizon harness.
    """
    root = None if artifact_dir is None else Path(artifact_dir)
    points: list[CapacityPoint] = []
    for embed_dim, depth in sizes:
        width = int(embed_dim)
        layers = int(depth)
        hidden = direct_hidden_matching(width, layers)
        tag = f"capacity-{width}x{layers}"
        latent = fit_latent(
            frame,
            delivery_lag=delivery_lag,
            epochs=epochs,
            seed=seed,
            lr=lr,
            artifact_dir=None if root is None else root / tag / "latent",
            model_unroll_steps=model_unroll_steps,
            embed_dim=width,
            depth=layers,
        )
        direct = fit_direct(
            frame,
            delivery_lag=delivery_lag,
            epochs=epochs,
            seed=seed,
            lr=lr,
            artifact_dir=None if root is None else root / tag / "direct",
            model_unroll_steps=model_unroll_steps,
            hidden_dim=hidden,
            depth=layers,
        )
        accuracy = held_out_rollout_accuracy(latent.dynamics, direct.dynamics, frame)
        point = CapacityPoint(
            embed_dim=width,
            depth=layers,
            parameters=parameter_count(latent.dynamics),
            direct_hidden=hidden,
            direct_parameters=parameter_count(direct.dynamics),
            accuracy=accuracy,
        )
        points.append(point)
        print(
            f"capacity {width}x{layers} parameters={point.parameters} "
            f"latent_mean={_mean_r2(accuracy.latent)} "
            f"direct_mean={_mean_r2(accuracy.direct)} "
            f"ols_mean={_mean_r2(accuracy.ols)}",
            flush=True,
        )
    return tuple(points)


def data_scaling_curve(
    frame: pd.DataFrame,
    episode_counts: Sequence[int],
    *,
    embed_dim: int,
    depth: int,
    delivery_lag: int,
    epochs: int,
    seed: int = 0,
    model_unroll_steps: int | None = None,
    artifact_dir: str | Path | None = None,
    lr: float = 1e-3,
    hidden_dim: int | None = None,
) -> tuple[DataPoint, ...]:
    """Held-out rollout accuracy at each nested episode count.

    Each count refits on the prefix of episode seeds. The direct predictor
    uses ``hidden_dim`` when given, otherwise the width matched to this
    latent size.
    """
    width = int(embed_dim)
    layers = int(depth)
    hidden = direct_hidden_matching(width, layers) if hidden_dim is None else int(hidden_dim)
    root = None if artifact_dir is None else Path(artifact_dir)
    points: list[DataPoint] = []
    for episodes in episode_counts:
        count = int(episodes)
        subset = _episode_prefix(frame, count)
        tag = f"data-{count}"
        latent = fit_latent(
            subset,
            delivery_lag=delivery_lag,
            epochs=epochs,
            seed=seed,
            lr=lr,
            artifact_dir=None if root is None else root / tag / "latent",
            model_unroll_steps=model_unroll_steps,
            embed_dim=width,
            depth=layers,
        )
        direct = fit_direct(
            subset,
            delivery_lag=delivery_lag,
            epochs=epochs,
            seed=seed,
            lr=lr,
            artifact_dir=None if root is None else root / tag / "direct",
            model_unroll_steps=model_unroll_steps,
            hidden_dim=hidden,
            depth=layers,
        )
        accuracy = held_out_rollout_accuracy(latent.dynamics, direct.dynamics, subset)
        points.append(DataPoint(episodes=count, accuracy=accuracy))
        print(
            f"data episodes={count} latent_mean={_mean_r2(accuracy.latent)} "
            f"direct_mean={_mean_r2(accuracy.direct)} ols_mean={_mean_r2(accuracy.ols)}",
            flush=True,
        )
    return tuple(points)


def _json_number(value: float) -> float | None:
    if math.isnan(value) or math.isinf(value):
        return None
    return float(value)


def _mean_r2(values: tuple[float, ...]) -> float | None:
    if not values or any(math.isnan(value) or math.isinf(value) for value in values):
        return None
    return float(sum(values) / len(values))


def _accuracy_payload(accuracy: RolloutAccuracy) -> dict[str, Any]:
    return {
        "steps": list(accuracy.steps),
        "latent": [_json_number(value) for value in accuracy.latent],
        "direct": [_json_number(value) for value in accuracy.direct],
        "ols": [_json_number(value) for value in accuracy.ols],
        "latent_mean": _mean_r2(accuracy.latent),
        "direct_mean": _mean_r2(accuracy.direct),
        "ols_mean": _mean_r2(accuracy.ols),
    }


def write_capacity_curve(
    points: Sequence[CapacityPoint],
    path: str | Path,
    **meta: Any,
) -> Path:
    """Write the capacity curve. The directory is a run artifact, not a commit."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **meta,
        "points": [
            {
                "embed_dim": point.embed_dim,
                "depth": point.depth,
                "parameters": point.parameters,
                "direct_hidden": point.direct_hidden,
                "direct_depth": point.depth,
                "direct_parameters": point.direct_parameters,
                **_accuracy_payload(point.accuracy),
            }
            for point in points
        ]
    }
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination


def write_data_curve(
    points: Sequence[DataPoint],
    path: str | Path,
    *,
    embed_dim: int,
    depth: int,
    parameters: int,
    direct_hidden: int,
    direct_parameters: int,
    **meta: Any,
) -> Path:
    """Write the data-scaling curve for one model size."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **meta,
        "embed_dim": int(embed_dim),
        "depth": int(depth),
        "parameters": int(parameters),
        "direct_hidden": int(direct_hidden),
        "direct_parameters": int(direct_parameters),
        "points": [
            {"episodes": point.episodes, **_accuracy_payload(point.accuracy)}
            for point in points
        ],
    }
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination


def save_planners(
    latent: ReplenishmentPlanner,
    direct: ReplenishmentPlanner,
    directory: str | Path,
) -> Path:
    """Write both fitted predictors. The published directory is gitignored."""
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(latent.dynamics.state_dict(), destination / "latent.pt")
    torch.save(direct.dynamics.state_dict(), destination / "direct.pt")
    meta = {
        "delivery_lag": int(latent.delivery_lag),
        "budget": int(latent.budget),
        "holding_rate": float(latent.holding_rate),
        "order_fee": float(latent.order_fee),
        "horizon_multiple": int(latent.horizon_multiple),
        "model_unroll_steps": int(latent.dynamics.model_unroll_steps),
        "train_episodes": sorted(int(episode) for episode in latent.dynamics.train_episodes),
        "val_episodes": sorted(int(episode) for episode in latent.dynamics.val_episodes),
        "latent_policy_seed": latent.policy_seed,
        "direct_policy_seed": direct.policy_seed,
        "embed_dim": int(latent.dynamics.embed_dim),
        "depth": int(latent.dynamics.depth),
        "direct_hidden": int(direct.dynamics.hidden_dim),
        "direct_depth": int(direct.dynamics.depth),
        "latent_parameters": parameter_count(latent.dynamics),
        "direct_parameters": parameter_count(direct.dynamics),
    }
    (destination / "planner.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return destination


def load_planners(directory: str | Path) -> tuple[ReplenishmentPlanner, ReplenishmentPlanner]:
    """Reload the planners ``save_planners`` wrote. Does not fit."""
    destination = Path(directory)
    meta = json.loads((destination / "planner.json").read_text(encoding="utf-8"))
    lag = int(meta["delivery_lag"])
    shared = {
        "delivery_lag": lag,
        "budget": int(meta["budget"]),
        "holding_rate": float(meta["holding_rate"]),
        "order_fee": float(meta["order_fee"]),
        "horizon_multiple": int(meta.get("horizon_multiple", DEFAULT_HORIZON_MULTIPLE)),
    }
    steps = meta.get("model_unroll_steps")
    if steps is None:
        steps = int(shared["horizon_multiple"]) * lag
    embed_dim = int(meta.get("embed_dim", REDUCED_EMBED_DIM))
    depth = int(meta.get("depth", REDUCED_DEPTH))
    latent_dynamics = LatentDynamics(
        delivery_lag=lag,
        model_unroll_steps=int(steps),
        embed_dim=embed_dim,
        depth=depth,
    )
    latent_dynamics.load_state_dict(
        torch.load(destination / "latent.pt", weights_only=True)
    )
    latent_dynamics.train_episodes = {int(episode) for episode in meta.get("train_episodes", [])}
    latent_dynamics.val_episodes = {int(episode) for episode in meta.get("val_episodes", [])}
    direct_dynamics = DirectDynamics(
        delivery_lag=lag,
        model_unroll_steps=int(steps),
        hidden_dim=int(meta.get("direct_hidden", REDUCED_DIRECT_HIDDEN)),
        depth=int(meta.get("direct_depth", depth)),
    )
    direct_dynamics.load_state_dict(
        torch.load(destination / "direct.pt", weights_only=True)
    )
    latent = ReplenishmentPlanner(
        latent_dynamics,
        policy_seed=meta["latent_policy_seed"],
        **shared,
    )
    direct = ReplenishmentPlanner(
        direct_dynamics,
        policy_seed=meta["direct_policy_seed"],
        **shared,
    )
    return latent, direct
