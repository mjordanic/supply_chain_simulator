"""Gymnasium-compatible RL environment wrapping the supply-chain simulator.

``RLEnv`` exposes the graph-engine's two-phase tick API through the Gymnasium
``Env`` interface (``reset`` / ``step`` / ``observation_space`` /
``action_space``).

The RL episode is a degenerate multi-product graph:
  FactoryNode("F_<pid>") × K → IntermediateNode("S") → DemandSinkNode("D_<pid>") × K

K is sampled per episode from [config.K_min, config.K_max_episode], padded to K_MAX.

Observation space: ``Box(shape=(K_MAX * F,))`` — the set encoder ``(K_MAX, F)``
tensor flattened to 1-D.
Action space: ``Box(low=-1, high=1, shape=(K_MAX * 3,))`` — the set action
``(K_MAX, 3)`` tensor flattened to 1-D.

Step path:
  tick_world() → encode (K_MAX, F) obs → decode (K_MAX, 3) action →
  Arbiter (resolve contention) → per-pid order dict →
  RLIntermediatePolicy.set_pending_action() → tick_decide_and_settle()

The Arbiter is the sole within-tick contention resolver.
"""

from __future__ import annotations

from collections import deque
from random import Random
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.rl.configs.default import RLConfig
from src.rl.episode_sampler import RLEpisodeSpec, sample_episode
from src.rl.set_encoder import (
    K_MAX,
    F,
    encode_set_observation,
    decode_set_action,
)
from src.rl.arbiter import allocate as _arbiter_allocate
from src.sim.policy import RLIntermediatePolicy
from src.sim.runner import Simulation, build_world
from src.sim.scenario import (
    DisruptionParams,
    ItemLifecycleParams,
    MarketParams,
    Ware,
)


class RLEnv(gym.Env):
    """Gymnasium env wrapping the supply-chain graph simulator for RL training.

    Parameters
    ----------
    catalog:
        The full product universe.
    config:
        An ``RLConfig`` instance. Defaults to ``RLConfig()`` (PRD defaults).
    market_params, lifecycle_params, disruption_params:
        Optional simulator parameter overrides.

    Gymnasium interface
    -------------------
    - ``reset(seed=None) → (obs, info)``
    - ``step(action) → (obs, reward, terminated, truncated, info)``
    - ``observation_space``: ``Box`` of shape ``(K_MAX * F,)``
    - ``action_space``: ``Box`` of shape ``(K_MAX * 3,)``, bounds ``[-1, 1]``
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        catalog: list[Ware],
        config: RLConfig | None = None,
        *,
        market_params: MarketParams | None = None,
        lifecycle_params: ItemLifecycleParams | None = None,
        disruption_params: DisruptionParams | None = None,
    ) -> None:
        super().__init__()

        self.catalog = catalog
        self.config: RLConfig = config if config is not None else RLConfig()
        self._market_params = market_params
        self._lifecycle_params = lifecycle_params
        self._disruption_params = disruption_params

        # Observation: (K_MAX, F) flattened.
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(K_MAX * F,),
            dtype=np.float32,
        )

        # Action: (K_MAX, 3) flattened, bounds [-1, 1].
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(K_MAX * 3,),
            dtype=np.float32,
        )

        # Per-env RNG used only to draw episode seeds when reset(seed=None).
        self._env_rng: Random = Random()

        # Live episode state (populated by reset).
        self._episode_spec: RLEpisodeSpec | None = None
        self._sim: Simulation | None = None
        self._rl_policy: RLIntermediatePolicy | None = None

        # Step counter within the current episode.
        self._step_count: int = 0

        # Opening cash of the intermediate node "S" — used to normalise.
        self._initial_cash: float = 1.0

        # Per-tick sales history for the rolling-mean feature.
        self._sales_history: dict[str, deque] = {}

        # Active product ids for the current episode (length K ≤ K_MAX).
        self._active_subset: tuple[str, ...] = ()

    # ------------------------------------------------------------------ reset

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Start a fresh episode and return the initial observation."""
        super().reset(seed=seed)

        if seed is not None:
            self._env_rng = Random(seed)
            episode_seed = seed
        else:
            episode_seed = self._env_rng.randint(0, 2**31 - 1)

        # Sample a fully deterministic episode specification.
        spec = sample_episode(
            catalog=self.catalog,
            config=self.config,
            episode_seed=episode_seed,
            market_params=self._market_params,
            disruption_params=self._disruption_params,
            lifecycle_params=self._lifecycle_params,
        )
        self._episode_spec = spec
        self._active_subset = spec.active_subset

        scenario = spec.scenario

        # Build the RLIntermediatePolicy shim.
        self._rl_policy = RLIntermediatePolicy()

        # Construct the Simulation bundle.
        self._sim = build_world(scenario, policy_overrides={"S": self._rl_policy})

        # Get a reference to node "S".
        node_s = self._sim.nodes["S"]

        # Record opening cash for cash-normalisation.
        self._initial_cash = float(node_s.cash)

        # Reset per-tick state.
        self._step_count = 0
        self._sales_history = {pid: deque(maxlen=100) for pid in self._active_subset}

        obs = self._build_observation()
        return obs, {}

    # ------------------------------------------------------------------ step

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advance one simulation tick and return the RL transition.

        Step path:
        1. tick_world() — advance market, events, lifecycle.
        2. Reshape action to (K_MAX, 3) and decode via set decoder.
        3. Run Arbiter to resolve capacity/cash contention.
        4. Set arbitrated action on RLIntermediatePolicy.
        5. tick_decide_and_settle().
        6. Reward = cash delta of node "S".
        7. Build next observation.

        Parameters
        ----------
        action:
            Numpy array of shape ``(K_MAX * 3,)`` with values in ``[-1, 1]``.
        """
        if self._sim is None:
            raise RuntimeError("RLEnv.step() called before reset().")

        sim = self._sim
        node_s = sim.nodes["S"]
        rl_policy = self._rl_policy
        active_subset = self._active_subset

        cash_before: float = float(node_s.cash)

        # --- Phase 1: advance the world, publish offers ---
        current_tick = sim.tick_world()
        central_table = getattr(sim, "_current_table", None)

        # --- Compute unit prices (for Arbiter cash budget) ---
        unit_prices: dict[str, float] = {}
        if central_table is not None:
            for pid in active_subset:
                offers = central_table.snapshot_for_buyer(pid)
                allowed = {f"F_{pid}"}
                offers = [(sid, o) for sid, o in offers if sid in allowed]
                if offers:
                    unit_prices[pid] = float(min(o.list_price for _, o in offers))
                else:
                    unit_prices[pid] = float(getattr(node_s, "costs", {}).get(pid, 1.0))
        else:
            for pid in active_subset:
                unit_prices[pid] = float(getattr(node_s, "costs", {}).get(pid, 1.0))

        # --- Decode action ---
        action_arr = np.asarray(action, dtype=np.float32).reshape(K_MAX, 3)
        base_prices = {
            pid: float(node_s.list_prices.get(pid, 1.0))
            for pid in active_subset
        }
        supplier_ids = [f"F_{pid}" for pid in active_subset]

        from src.rl.encoders import compute_effective_rate
        base_demand_prior = self._get_base_demand_prior()
        effective_rate = compute_effective_rate(self._sales_history, base_demand_prior)

        decoded = decode_set_action(
            action=action_arr,
            node=node_s,
            active_subset=active_subset,
            base_prices=base_prices,
            effective_rate=effective_rate,
            supplier_ids=supplier_ids,
        )

        # --- Arbiter: resolve contention ---
        raw_orders = decoded.get("order", {})
        proposed: dict[str, float] = {}
        for pid in active_subset:
            lines = raw_orders.get(pid, [])
            proposed[pid] = float(sum(qty for _, qty in lines))

        # Compute per-SKU headroom and global free space.
        inventory = dict(node_s.inventory)
        raw_pending = getattr(node_s, "pending", {})
        if raw_pending and isinstance(next(iter(raw_pending.values()), None), dict):
            pending: dict[str, int] = {}
            for sup_pend in raw_pending.values():
                for pid, qty in sup_pend.items():
                    pending[pid] = pending.get(pid, 0) + qty
        else:
            pending = dict(raw_pending) if raw_pending else {}

        capacity_val = float(getattr(node_s, "capacity", max(1, len(active_subset) * 100)))
        total_inv = sum(float(v) for v in inventory.values())
        total_pend = sum(float(v) for v in pending.values())
        global_free_space = max(0, int(capacity_val - total_inv - total_pend))

        per_sku_headroom: dict[str, int] = {}
        for pid in active_subset:
            inv_pid = float(inventory.get(pid, 0))
            pend_pid = float(pending.get(pid, 0))
            per_sku_headroom[pid] = max(0, int(capacity_val - inv_pid - pend_pid))

        cash_budget = float(node_s.cash) * self.config.cash_budget_fraction

        priorities: dict[str, float] = {}
        for i, pid in enumerate(active_subset):
            priorities[pid] = float(action_arr[i, 2])

        allocated = _arbiter_allocate(
            proposed=proposed,
            per_sku_headroom=per_sku_headroom,
            global_free_space=global_free_space,
            cash_budget=cash_budget,
            unit_prices=unit_prices,
            priorities=priorities,
            mode=self.config.arbiter_mode,
        )

        # Build arbitrated order dict.
        order_dict: dict[str, list] = {}
        for pid in active_subset:
            qty = allocated.get(pid, 0)
            order_dict[pid] = [(f"F_{pid}", qty)] if qty > 0 else []

        action_dict = {
            "order": order_dict,
            "list_price": decoded.get("list_price", {}),
            "min_order_imposed": decoded.get("min_order_imposed", {}),
        }
        rl_policy.set_pending_action(action_dict)

        # --- Phase 2: demand-pull walk ---
        sim.tick_decide_and_settle(current_tick)

        # Reward = cash delta.
        cash_after: float = float(node_s.cash)
        reward: float = cash_after - cash_before

        # Update rolling sales history.
        last_sales = sim._tick_sales.get("S", {})
        for pid in active_subset:
            qty_sold = last_sales.get(pid, 0)
            if pid in self._sales_history:
                self._sales_history[pid].append(qty_sold)

        # Build next observation (pass proposed/prices for contention features).
        self._step_count += 1
        obs = self._build_observation(proposed_quantities=proposed, unit_prices=unit_prices)

        terminated: bool = self._step_count >= self.config.episode_length
        truncated: bool = False

        info: dict[str, Any] = {
            "step": self._step_count,
            "cash": cash_after,
            "active_products": list(active_subset),
            "inventory": dict(node_s.inventory),
            "per_product": {
                pid: {
                    "inventory": node_s.inventory.get(pid, 0),
                    "list_price": node_s.list_prices.get(pid, 0.0),
                    "allocated": allocated.get(pid, 0),
                }
                for pid in active_subset
            },
        }

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ private

    def _get_base_demand_prior(self) -> float:
        """Return a base demand prior from the scenario market params."""
        spec = self._episode_spec
        if spec is None:
            return 1.0
        from src.sim.distributions import Distribution
        market_base_demand = getattr(spec.scenario.market, "base_demand", None)
        if isinstance(market_base_demand, Distribution):
            prior_rng = Random(spec.world_seed + 1)
            return float(market_base_demand.sample(prior_rng))
        elif market_base_demand is not None:
            return float(market_base_demand)
        return 1.0

    def _build_observation(
        self,
        *,
        proposed_quantities: dict[str, float] | None = None,
        unit_prices: dict[str, float] | None = None,
    ) -> np.ndarray:
        """Encode the current state into the (K_MAX, F) obs tensor (flattened)."""
        sim = self._sim
        node_s = sim.nodes["S"]
        central_table = getattr(sim, "_current_table", None)
        supplier_ids_for = {pid: [f"F_{pid}"] for pid in self._active_subset}

        obs_2d = encode_set_observation(
            node=node_s,
            market=sim.market,
            active_subset=self._active_subset,
            initial_cash=self._initial_cash,
            sales_history=self._sales_history,
            central_table=central_table,
            supplier_ids_for=supplier_ids_for,
            proposed_quantities=proposed_quantities,
            unit_prices=unit_prices,
        )
        return obs_2d.reshape(-1)

    def render(self) -> None:
        """No-op render."""
        pass

    def close(self) -> None:
        """Release any resources held by the env."""
        pass


__all__ = ["RLEnv"]
