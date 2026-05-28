"""Gymnasium-compatible RL environment wrapping the supply-chain simulator.

``RLEnv`` exposes the graph-engine's two-phase tick API through the Gymnasium
``Env`` interface (``reset`` / ``step`` / ``observation_space`` /
``action_space``).

The RL episode is a degenerate 3-node graph:
  FactoryNode("F_<pid>") → IntermediateNode("S") → DemandSinkNode("D_<pid>")

The trainable node is "S". ``RLIntermediatePolicy`` is attached to it via
``policy_overrides_by_id={"S": rl_policy}`` in ``build_world``.

Design decisions
----------------
- One ``reset()`` call produces a fully independent episode via
  ``episode_sampler.sample_episode``. The episode seed is derived from the
  user-supplied ``seed`` argument (or drawn from the env's own RNG when
  ``seed=None``).
- ``reset()`` constructs the ``Simulation`` bundle via
  ``build_world(scenario, policy_overrides={"S": RLIntermediatePolicy()})``.
- The tick order inside ``step()`` uses the two-phase sim API:
  ``sim.tick_world()`` → encode obs (from node S + central_table) →
  run actor → decode → ``RLIntermediatePolicy.set_pending_action(...)`` →
  ``sim.tick_decide_and_settle()``.
- Reward each tick = ``S.cash`` after tick − ``S.cash`` before tick.
- The env owns its own ``env_rng`` used only for drawing episode seeds
  when ``reset(seed=None)`` is called.
"""

from __future__ import annotations

from collections import deque
from random import Random
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.rl.configs.default import RLConfig
from src.rl.encoders import (
    compute_effective_rate,
    decode_action,
    encode_observation,
    observation_dim,
    action_dim,
)
from src.rl.episode_sampler import RLEpisodeSpec, sample_episode
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
        The full product universe built via ``load_catalog``.
    base_template:
        A template object carrying non-episodic knobs (delivery_lag,
        holding_rate, order_fee). The ``StoreTemplate`` format is still
        accepted for backward-compatibility; only ``delivery_lag``,
        ``holding_rate``, and ``order_fee`` are consumed.
    config:
        An ``RLConfig`` instance. Defaults to ``RLConfig()`` (PRD defaults).
    market_params, lifecycle_params, disruption_params:
        Optional simulator parameter overrides.

    Gymnasium interface
    -------------------
    - ``reset(seed=None) → (obs, info)``
    - ``step(action) → (obs, reward, terminated, truncated, info)``
    - ``observation_space``: ``Box`` of shape ``(observation_dim(K_active),)``
    - ``action_space``: ``Box`` of shape ``(action_dim(K_active),)``, bounds ``[-1, 1]``
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        catalog: list[Ware],
        base_template: Any,
        config: RLConfig | None = None,
        *,
        market_params: MarketParams | None = None,
        lifecycle_params: ItemLifecycleParams | None = None,
        disruption_params: DisruptionParams | None = None,
    ) -> None:
        super().__init__()

        self.catalog = catalog
        self.base_template = base_template
        self.config: RLConfig = config if config is not None else RLConfig()
        self._market_params = market_params
        self._lifecycle_params = lifecycle_params
        self._disruption_params = disruption_params

        K = self.config.K_active

        obs_dim = observation_dim(K)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        act_dim = action_dim(K)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(act_dim,),
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

        # Per-tick sales history for the rolling-mean feature in encoders.
        self._sales_history: dict[str, deque] = {}

        # Market-derived demand prior for cold-start ordering.
        self._base_demand_prior: float = 1.0

        # Effective demand rate per pid.
        self._effective_rate: dict[str, float] = {}

        # Slot permutation for the current episode.
        self._slot_perm: tuple[int, ...] = tuple(range(K))

        # Ordered list of active product ids for this episode.
        self._active_subset: tuple[str, ...] = ()

        # Supplier ids per product (for action decoding).
        self._supplier_ids_for: dict[str, list[str]] = {}

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
            base_template=self.base_template,
            config=self.config,
            episode_seed=episode_seed,
            market_params=self._market_params,
            disruption_params=self._disruption_params,
            lifecycle_params=self._lifecycle_params,
        )
        self._episode_spec = spec
        self._slot_perm = spec.slot_permutation
        self._active_subset = spec.active_subset

        scenario = spec.scenario

        # Build the RLIntermediatePolicy shim.
        self._rl_policy = RLIntermediatePolicy()

        # Construct the Simulation bundle.
        # policy_overrides={"S": self._rl_policy} attaches the RL shim to
        # the intermediate node "S".
        self._sim = build_world(scenario, policy_overrides={"S": self._rl_policy})

        # Get a reference to node "S".
        node_s = self._sim.nodes["S"]

        # Record opening cash for the cash-normalisation feature.
        self._initial_cash = float(node_s.cash)

        # Derive the market-demand prior for cold-start ordering.
        from src.sim.distributions import Distribution

        market_base_demand = getattr(scenario.market, "base_demand", None)
        if isinstance(market_base_demand, Distribution):
            prior_rng = Random(scenario.world_seed + 1)
            self._base_demand_prior = float(market_base_demand.sample(prior_rng))
        elif market_base_demand is not None:
            self._base_demand_prior = float(market_base_demand)
        else:
            self._base_demand_prior = 1.0

        # Reset per-tick state.
        self._step_count = 0
        self._sales_history = {pid: deque(maxlen=100) for pid in self._active_subset}

        # Build the supplier_ids_for mapping (factory per product).
        self._supplier_ids_for = {
            pid: [f"F_{pid}"] for pid in self._active_subset
        }

        # Compute effective_rate for the initial observation.
        self._effective_rate = compute_effective_rate(
            self._sales_history, self._base_demand_prior
        )

        obs = self._build_observation()
        return obs, {}

    # ------------------------------------------------------------------ step

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advance one simulation tick and return the RL transition.

        Uses the two-phase graph-engine tick API:

        1. ``sim.tick_world()`` — advance market, events, lifecycle;
           publish offers to the central table.
        2. Encode observation from node "S" + central table.
        3. Decode action and set it on ``RLIntermediatePolicy`` via
           ``set_pending_action``.
        4. ``sim.tick_decide_and_settle()`` — run the full phase cascade
           (node S's policy.decide() returns the pending action).
        5. Compute reward from cash delta on node "S".
        6. Build next observation and check for episode termination.

        Parameters
        ----------
        action:
            Numpy array of shape ``(2*K,)`` with values in ``[-1, 1]``.
        """
        if self._sim is None:
            raise RuntimeError("RLEnv.step() called before reset().")

        sim = self._sim
        node_s = sim.nodes["S"]
        rl_policy = self._rl_policy

        cash_before: float = float(node_s.cash)

        # --- Phase 1: advance the world, publish offers ---
        current_tick = sim.tick_world()

        # Get the central table (published by tick_world).
        central_table = getattr(sim, "_current_table", None)

        # --- Encode observation (uses current central_table state) ---
        self._effective_rate = compute_effective_rate(
            self._sales_history, self._base_demand_prior
        )

        obs_pre = encode_observation(
            node=node_s,
            market=sim.market,
            registry=sim.item_registry,
            step=self._step_count,
            slot_perm=self._slot_perm,
            K_active=self.config.K_active,
            central_table=central_table,
            active_subset=self._active_subset,
            supplier_ids_for=self._supplier_ids_for,
            initial_cash=self._initial_cash,
            sales_history=self._sales_history,
            effective_rate=self._effective_rate,
            max_inventory_lt=self.config.max_inventory_lt,
        )

        # --- Decode action and set it on the RLIntermediatePolicy shim ---
        action_vec = np.asarray(action, dtype=np.float32)

        # Build per-product supplier id list (one factory per product).
        # For the decode, we want the actual supplier id for this product.
        all_supplier_ids = [f"F_{pid}" for pid in self._active_subset]

        action_dict = decode_action(
            action_vec,
            self._slot_perm,
            node_s,
            self.config.K_active,
            {pid: float(node_s.list_prices.get(pid, 1.0)) for pid in self._active_subset},
            supplier_ids=all_supplier_ids,
            active_subset=self._active_subset,
            effective_rate=self._effective_rate,
            target_centre_lead_times=self.config.target_centre_lead_times,
            target_half_span_lead_times=self.config.target_half_span_lead_times,
            target_max_lead_times=self.config.target_max_lead_times,
        )
        rl_policy.set_pending_action(action_dict)

        # --- Phase 2: run phase cascade (S's policy.decide() called here) ---
        sim.tick_decide_and_settle(current_tick)

        # Reward = cash delta of node "S" for this tick.
        cash_after: float = float(node_s.cash)
        reward: float = cash_after - cash_before

        # Update rolling sales history from intermediate node's sales.
        # The runner tracks _last_tick_sales for IntermediateNode suppliers.
        last_sales = sim._last_tick_sales.get("S", {})
        for pid in self._active_subset:
            qty = last_sales.get(pid, 0)
            if pid in self._sales_history:
                self._sales_history[pid].append(qty)

        # --- Build next observation ---
        self._step_count += 1
        obs = self._build_observation()

        # --- Termination check ---
        terminated: bool = self._step_count >= self.config.episode_length
        truncated: bool = False

        # --- info ---
        info: dict[str, Any] = {
            "step": self._step_count,
            "cash": cash_after,
            "active_products": list(self._active_subset),
            "inventory": dict(node_s.inventory),
            "per_product": {
                pid: {
                    "inventory": node_s.inventory.get(pid, 0),
                    "list_price": node_s.list_prices.get(pid, 0.0),
                }
                for pid in self._active_subset
            },
        }

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ private helpers

    def _build_observation(self) -> np.ndarray:
        """Encode the current node-S / market / registry state into the obs tensor."""
        sim = self._sim
        node_s = sim.nodes["S"]
        central_table = getattr(sim, "_current_table", None)
        return encode_observation(
            node=node_s,
            market=sim.market,
            registry=sim.item_registry,
            step=self._step_count,
            slot_perm=self._slot_perm,
            K_active=self.config.K_active,
            central_table=central_table,
            active_subset=self._active_subset,
            supplier_ids_for=self._supplier_ids_for,
            initial_cash=self._initial_cash,
            sales_history=self._sales_history,
            effective_rate=self._effective_rate if self._effective_rate else None,
            max_inventory_lt=self.config.max_inventory_lt,
        )

    def render(self) -> None:
        """No-op render (no visual output supported)."""
        pass

    def close(self) -> None:
        """Release any resources held by the env (none currently)."""
        pass


__all__ = ["RLEnv"]
