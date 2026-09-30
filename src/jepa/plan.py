"""Fit a per-SKU predictor and plan replenishment orders with it.

``decide`` draws a fixed budget of cover-and-priority proposals from
``policy_rng``, rations each one, and keeps the proposal whose imagined
score is highest over ``horizon_multiple * delivery_lag`` steps
(``horizon_multiple`` defaults to 4). Each step's profit is revenue
minus purchase cost minus holding cost minus the order fee. Ending
inventory position — on-hand plus in-transit — is then credited at unit
cost, so unsold stock is not confiscated at the window boundary. There
is no stockout penalty and no learned reward head. Imagined cash uses
the holding rate under test and is the cash the next ration sees. The
terminal credit is not cash.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import torch

from src.jepa.dynamics import (
    ARRIVAL_AGE_SLOTS,
    DirectDynamics,
    LatentDynamics,
    arrival_age_profile,
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
DEFAULT_HORIZON_MULTIPLE = 4


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

    ``ration="priority"`` is the learned ration. ``ration="fair_share"``
    shoots covers from the same grid and splits them with the textbook
    fair-share allocator. ``retarget`` builds another planner on the
    same weights at a new holding rate and does not fit. Each recorded
    proposal keeps every draw and ``chosen``, the covers and priority
    that won. Cold start matches the textbook anchor: a SKU with no
    history, zero position, and a zero censored-sales rate is ordered
    ``cold_start_pilot_qty`` at ``OPENING_BUDGET_PCT``. The supplier
    split is cheapest-first with ``MIN_ORDER_FLOOR``. The opening
    order's fee and purchase use those routed lines. Later imagined
    steps still price the cheapest offer, because the offer book is
    not rolled forward.
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
    ) -> None:
        super().__init__(policy_seed=policy_seed)
        if ration not in ("priority", "fair_share"):
            raise ValueError(f"unknown ration {ration!r}")
        if int(horizon_multiple) < 1:
            raise ValueError("horizon_multiple must be at least 1")
        self.dynamics = dynamics
        self.delivery_lag = int(dynamics.delivery_lag if delivery_lag is None else delivery_lag)
        self.budget = int(budget)
        self.holding_rate = float(holding_rate)
        self.order_fee = float(order_fee)
        self.ration = ration
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

        draws: list[dict[str, Any]] = []
        scored: list[tuple[float, dict[str, int], dict[str, list[tuple[str, int]]]]] = []
        for _ in range(self.budget):
            covers = {pid: int(self.policy_rng.choice(COVERS)) for pid in pids}
            if self.ration == "priority" and pids:
                priority = list(self.policy_rng.sample(pids, len(pids)))
            else:
                priority = list(pids)
            draws.append({"covers": dict(covers), "priority": list(priority)})
            scored.append(
                self._score(
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
                    obs_intermediate.get("direct_supplier_ids"),
                    central_table,
                    profiles,
                    obs_intermediate.get("delivery_lags") or {},
                )
            )

        chosen_covers: dict[str, int] = {}
        chosen_priority: list[str] = []
        opening_order: dict[str, list[tuple[str, int]]] = {}
        if scored:
            winner = max(range(len(scored)), key=lambda index: scored[index][0])
            opening_order = scored[winner][2]
            chosen_covers = dict(draws[winner]["covers"])
            chosen_priority = list(draws[winner]["priority"])

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
                    purchase += sum(qty * price for _sid, qty, price in lines)
                    used.update(sid for sid, _qty, _price in lines)
            else:
                for pid in pids:
                    qty = shipped[pid]
                    purchase += qty * float(unit_costs.get(pid, 0.0))
                    supplier_id = suppliers.get(pid)
                    if qty > 0 and supplier_id is not None:
                        used.add(supplier_id)
            predicted = rollout.step([float(shipped[pid]) for pid in pids])
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


def fit_latent(
    frame: pd.DataFrame,
    *,
    delivery_lag: int = 3,
    epochs: int = 20,
    seed: int = 0,
    lr: float = 1e-3,
    artifact_dir: str | Path | None = None,
    **planner_kwargs: Any,
) -> ReplenishmentPlanner:
    """Fit the latent predictor on a transition log and wrap it as a Policy."""
    dynamics = LatentDynamics(delivery_lag=delivery_lag)
    dynamics.fit(frame, epochs=epochs, seed=seed, lr=lr, artifact_dir=artifact_dir)
    return ReplenishmentPlanner(dynamics, delivery_lag=delivery_lag, **planner_kwargs)


def fit_direct(
    frame: pd.DataFrame,
    *,
    delivery_lag: int = 3,
    epochs: int = 20,
    seed: int = 0,
    lr: float = 1e-3,
    artifact_dir: str | Path | None = None,
    **planner_kwargs: Any,
) -> ReplenishmentPlanner:
    """Fit the direct predictor on a transition log and wrap it as a Policy."""
    dynamics = DirectDynamics(delivery_lag=delivery_lag)
    dynamics.fit(frame, epochs=epochs, seed=seed, lr=lr, artifact_dir=artifact_dir)
    return ReplenishmentPlanner(dynamics, delivery_lag=delivery_lag, **planner_kwargs)


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
        "latent_policy_seed": latent.policy_seed,
        "direct_policy_seed": direct.policy_seed,
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
    latent_dynamics = LatentDynamics(delivery_lag=lag)
    latent_dynamics.load_state_dict(
        torch.load(destination / "latent.pt", weights_only=True)
    )
    direct_dynamics = DirectDynamics(delivery_lag=lag)
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
