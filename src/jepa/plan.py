"""Fit a per-SKU predictor and plan replenishment orders with it.

``decide`` draws a fixed budget of cover-and-priority proposals from
``policy_rng``, rations each one, and keeps the proposal whose imagined
engine ``net_profit`` is highest over the lead-time horizon. Imagined
profit is revenue minus purchase cost minus holding cost minus the
order fee. There is no stockout penalty and no learned reward head.
Imagined cash uses the holding rate under test and is the cash the next
ration sees.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import torch

from src.jepa.dynamics import DirectDynamics, LatentDynamics, sku_features
from src.jepa.policy import COVERS, _flatten_pending, _priority_ration, _suppliers_and_costs
from src.sim.policy import IntermediatePolicy, _allocate_two_pass_fair_share, _estimate_rate

DEFAULT_BUDGET = 64


class ReplenishmentPlanner(IntermediatePolicy):
    """Ordering-only buyer that scores proposals with a frozen predictor.

    ``ration="priority"`` is the learned ration. ``ration="fair_share"``
    shoots covers from the same grid and splits them with the textbook
    fair-share allocator. ``retarget`` builds another planner on the
    same weights at a new holding rate and does not fit. Each recorded
    proposal keeps every draw and ``chosen``, the covers and priority
    that won.
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
    ) -> None:
        super().__init__(policy_seed=policy_seed)
        if ration not in ("priority", "fair_share"):
            raise ValueError(f"unknown ration {ration!r}")
        self.dynamics = dynamics
        self.delivery_lag = int(dynamics.delivery_lag if delivery_lag is None else delivery_lag)
        self.budget = int(budget)
        self.holding_rate = float(holding_rate)
        self.order_fee = float(order_fee)
        self.ration = ration
        self.min_order = 0
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
        for pid in pids:
            self.sales_log.setdefault(pid, []).append(int(observed.get(pid, 0)))

        cash = float(obs_intermediate.get("cash") or 0.0)
        capacity = int(obs_intermediate.get("capacity") or 0)
        suppliers, unit_costs = _suppliers_and_costs(pids, obs_intermediate, central_table)
        base = obs_intermediate.get("base_prices") or obs_intermediate.get("list_prices") or {}
        prices = {pid: float(base.get(pid, 0.0)) for pid in pids}

        draws: list[dict[str, Any]] = []
        scored: list[tuple[float, dict[str, int]]] = []
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
                )
            )

        granted = {pid: 0 for pid in pids}
        chosen_covers: dict[str, int] = {}
        chosen_priority: list[str] = []
        if scored:
            winner = max(range(len(scored)), key=lambda index: scored[index][0])
            granted = scored[winner][1]
            chosen_covers = dict(draws[winner]["covers"])
            chosen_priority = list(draws[winner]["priority"])

        order: dict[str, list[tuple[str, int]]] = {}
        for pid in pids:
            qty = int(granted.get(pid, 0))
            supplier_id = suppliers.get(pid)
            if qty > 0 and supplier_id is not None:
                order[pid] = [(supplier_id, qty)]
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
    ) -> tuple[float, dict[str, int]]:
        on_hand = {pid: int(inventory.get(pid, 0)) for pid in pids}
        in_transit = {pid: int(pending.get(pid, 0)) for pid in pids}
        sales_hist = {pid: list(self.sales_log.get(pid, [])) for pid in pids}
        cash_now = float(cash)
        total = 0.0
        opening_grant: dict[str, int] | None = None
        # One encode of the decision-time window. Later steps advance the
        # predicted embedding (or, for the direct arm, the predicted
        # inventory) and do not re-read cash, so the holding rate changes
        # the score and the ration but not the transition itself.
        rollout = self.dynamics.open_rollout(
            [
                sku_features(
                    on_hand=on_hand[pid],
                    in_transit=in_transit[pid],
                    sales_window=sales_hist[pid],
                    cash=cash_now,
                    capacity=capacity,
                    delivery_lag=self.delivery_lag,
                )
                for pid in pids
            ]
        )
        for _step in range(self.delivery_lag):
            holding = self.holding_rate * sum(
                on_hand[pid] * float(unit_costs.get(pid, 0.0)) for pid in pids
            )
            rates = _estimate_rate(
                {pid: sales_hist[pid] for pid in pids},
                max(1, self.delivery_lag),
                {pid: [] for pid in pids},
            )
            desired: dict[str, int] = {}
            for pid in pids:
                if suppliers.get(pid) is None:
                    desired[pid] = 0
                    continue
                position = on_hand[pid] + in_transit[pid]
                target = round(
                    int(covers.get(pid, 0)) * float(rates.get(pid, 0.0)) * self.delivery_lag
                )
                desired[pid] = max(0, int(target) - position)
            free_space = max(0, int(capacity) - sum(on_hand.values()) - sum(in_transit.values()))
            spendable = max(0.0, cash_now - holding)
            granted = self._ration(desired, priority, free_space, spendable, unit_costs)
            predicted = rollout.step([float(granted.get(pid, 0)) for pid in pids])
            revenue = 0.0
            purchase = 0.0
            used: set[str] = set()
            for pid, (next_on_hand, next_in_transit, next_sales) in zip(pids, predicted, strict=True):
                revenue += max(0.0, float(next_sales)) * float(prices.get(pid, 0.0))
                qty = int(granted.get(pid, 0))
                purchase += qty * float(unit_costs.get(pid, 0.0))
                supplier_id = suppliers.get(pid)
                if qty > 0 and supplier_id is not None:
                    used.add(supplier_id)
                on_hand[pid] = max(0, int(round(next_on_hand)))
                in_transit[pid] = max(0, int(round(next_in_transit)))
                sales_hist[pid].append(max(0, int(round(next_sales))))
            fee = self.order_fee * len(used)
            profit = revenue - purchase - holding - fee
            total += profit
            if opening_grant is None:
                opening_grant = {pid: int(granted.get(pid, 0)) for pid in pids}
            cash_now = cash_now + profit
        return total, opening_grant or {pid: 0 for pid in pids}

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
    **planner_kwargs: Any,
) -> ReplenishmentPlanner:
    """Fit the latent predictor on a transition log and wrap it as a Policy."""
    dynamics = LatentDynamics(delivery_lag=delivery_lag)
    dynamics.fit(frame, epochs=epochs, seed=seed, lr=lr)
    return ReplenishmentPlanner(dynamics, delivery_lag=delivery_lag, **planner_kwargs)


def fit_direct(
    frame: pd.DataFrame,
    *,
    delivery_lag: int = 3,
    epochs: int = 20,
    seed: int = 0,
    lr: float = 1e-3,
    **planner_kwargs: Any,
) -> ReplenishmentPlanner:
    """Fit the direct predictor on a transition log and wrap it as a Policy."""
    dynamics = DirectDynamics(delivery_lag=delivery_lag)
    dynamics.fit(frame, epochs=epochs, seed=seed, lr=lr)
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
