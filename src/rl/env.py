"""Gymnasium-compatible RL environment wrapping the supply-chain simulator.

``RLEnv`` exposes the simulator's ``Market`` / ``EventEngine`` /
``ItemRegistry`` / ``Store`` subsystems through the Gymnasium ``Env``
interface (``reset`` / ``step`` / ``observation_space`` / ``action_space``).

Design decisions
----------------
- One ``reset()`` call produces a fully independent episode via
  ``episode_sampler.sample_episode``. The episode seed is derived from the
  user-supplied ``seed`` argument (or drawn from the env's own RNG when
  ``seed=None``).
- The tick order inside ``step()`` mirrors ``Runner.run()`` exactly so the
  world simulation is bit-identical to a Runner-based run given the same
  world seed. The Runner is NOT imported or modified here.
- Promotions are disabled for the lifetime of an episode (``RLPolicy``
  always returns ``promotions={}``, and ``assortment`` is frozen via
  ``activate=[]`` / ``deactivate=[]``).
- The reward each tick is the sum of ``(revenue − total_cost)`` over the
  active SKUs, which equals the per-tick balance delta attributable to
  active-SKU decisions.
- The env owns its own ``env_rng`` used only for drawing episode seeds
  when ``reset(seed=None)`` is called. World stochasticity flows through
  ``world_rng`` inside the subsystems, untouched by this RNG.
"""

from __future__ import annotations

import math
from collections import deque
from random import Random
from typing import Any, Sequence

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
from src.rl.episode_sampler import EpisodeSpec, sample_episode
from src.sim.event_engine import EventEngine
from src.sim.item_registry import ItemRegistry
from src.sim.market import Market
from src.sim.policy import RLPolicy
from src.sim.scenario import (
    DisruptionParams,
    ItemLifecycleParams,
    MarketParams,
    StoreInstance,
    Ware,
)
from src.sim.store import Store


def _make_delivery_callback(store: Store, pid: str, qty: int):
    """Bind ``(store, pid, qty)`` into a zero-arg delivery callback.

    Same pattern as ``Runner._make_delivery_callback`` — a factory keeps
    each callback's closure independent across loop iterations.
    """

    def _callback() -> None:
        store.deliver(pid, qty)

    return _callback


class RLEnv(gym.Env):
    """Gymnasium env wrapping the supply-chain simulator for RL training.

    Parameters
    ----------
    catalog:
        The full product universe built via ``load_catalog``. Must contain
        at least ``config.K_active`` products.
    base_template:
        A ``StoreTemplate`` carrying non-episodic knobs (region, delivery
        lag, holding rate, order fee). Episodic fields are overridden by
        ``sample_episode`` each reset.
    config:
        An ``RLConfig`` instance. Defaults to ``RLConfig()`` (PRD defaults).
    market_params, lifecycle_params, disruption_params:
        Optional simulator parameter overrides. When ``None`` the sampler's
        own defaults are used.

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

        # Observation space: flat float32 vector.
        obs_dim = observation_dim(K)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Action space: 2*K continuous in [-1, 1].
        act_dim = action_dim(K)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(act_dim,),
            dtype=np.float32,
        )

        # Per-env RNG used only to draw episode seeds when reset(seed=None).
        # World stochasticity goes through world_rng inside the subsystems.
        self._env_rng: Random = Random()

        # Live episode state (populated by reset).
        self._episode_spec: EpisodeSpec | None = None
        self._market: Market | None = None
        self._event_engine: EventEngine | None = None
        self._item_registry: ItemRegistry | None = None
        self._store: Store | None = None
        self._rl_policy: RLPolicy | None = None

        # Step counter within the current episode.
        self._step_count: int = 0

        # Opening balance — used to normalise the cash feature in observations.
        self._initial_cash: float = 1.0

        # Per-tick sales history for the rolling-mean feature in encoders.
        self._sales_history: dict[str, deque] = {}

        # Market-derived demand prior for the cold-start tick.  Set by reset()
        # from market.params.base_demand; positive float.
        self._base_demand_prior: float = 1.0

        # Slot permutation for the current episode (set by reset).
        self._slot_perm: tuple[int, ...] = tuple(range(K))

    # ------------------------------------------------------------------ reset

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Start a fresh episode and return the initial observation.

        Parameters
        ----------
        seed:
            Episode seed. Determines the assortment, capacity, balance,
            world seed, and slot permutation for this episode. When
            ``None``, the env draws a seed from its own internal RNG.
        options:
            Unused; present for Gymnasium API compliance.

        Returns
        -------
        obs:
            Initial observation tensor of shape ``(observation_dim(K),)``.
        info:
            Empty dict at reset (no per-tick accounting yet).
        """
        super().reset(seed=seed)

        # Seed the env's own RNG when the caller supplies a seed — this
        # makes the env fully reproducible when reset(seed=s) is called.
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

        scenario = spec.scenario

        # Build world subsystems in the same order as Runner.__init__.
        from random import Random as _Random

        world_rng = _Random(scenario.world_seed)

        self._item_registry = ItemRegistry(
            scenario.item_lifecycle,
            scenario.catalog,
            world_rng,
        )
        self._market = Market(
            scenario.market,
            world_rng,
            scenario.start_date,
            registry=self._item_registry,
        )
        self._event_engine = EventEngine(scenario.disruption, world_rng)

        # Build the RLPolicy shim.
        self._rl_policy = RLPolicy()

        # Build the single Store from the episode's StoreInstance.
        # The episode has exactly one store in its scenario.
        si: StoreInstance = scenario.stores[0]
        self._store = Store(
            si.template,
            si.init_seed,
            self._rl_policy,
            scenario.catalog,
            freshness_alpha=self._item_registry.default_freshness_alpha,
            freshness_decay=self._item_registry.default_freshness_decay,
            item_registry=self._item_registry,
        )

        # Record opening balance for the cash-normalisation feature.
        self._initial_cash = float(self._store.balance)

        # Derive the market-demand prior for cold-start ordering.
        # scenario.market is a MarketParams; its base_demand field is always
        # a Distribution (see MarketParams dataclass).  Sample it using a
        # CRN-disjoint RNG stream (world_seed + 1) so the world RNG does not
        # advance and market draws remain bit-identical to a Runner-based run.
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
        self._sales_history = {pid: deque(maxlen=100) for pid in [w.product_id for w in self.catalog]}

        obs = self._build_observation()
        return obs, {}

    # ------------------------------------------------------------------ step

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advance one simulation tick and return the RL transition.

        The tick order mirrors ``Runner.run()`` exactly:

        1. ``market.tick()``
        2. ``event_engine.tick(market)``
        3. ``item_registry.tick()``
        4. Set pending action on ``RLPolicy`` via ``encoders.decode_action``
        5. ``store.decide(store.observe(...))``
        6. Dispatch orders (same as ``Runner._dispatch_orders``)
        7. Settle demand (same as ``Runner._process_demand``), accumulate reward
        8. Build next observation via ``encoders.encode_observation``
        9. Increment step counter; terminate when step == episode_length.

        Parameters
        ----------
        action:
            Numpy array of shape ``(2*K,)`` with values in ``[-1, 1]``.

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        if self._store is None:
            raise RuntimeError("RLEnv.step() called before reset().")

        store = self._store
        market = self._market
        event_engine = self._event_engine
        item_registry = self._item_registry
        rl_policy = self._rl_policy

        # Balance before this tick — used to compute reward later.
        balance_before: float = float(store.balance)

        # --- 1. market.tick() ---
        market.tick()

        # --- 2. event_engine.tick(market) ---
        event_engine.tick(market)

        # --- 3. item_registry.tick() ---
        item_registry.tick()

        # --- 4. Decode action and set it on the RLPolicy shim ---
        # Compute effective rate once per tick; shared by decoder (and later
        # the encoder once slice 5 lands — do NOT pass to encode_observation yet).
        effective_rate = compute_effective_rate(
            self._sales_history, self._base_demand_prior
        )

        action_vec = np.asarray(action, dtype=np.float32)
        action_dict = decode_action(
            action_vec,
            self._slot_perm,
            store,
            self.config.K_active,
            store.base_prices,
            effective_rate=effective_rate,
            target_centre_lead_times=self.config.target_centre_lead_times,
            target_half_span_lead_times=self.config.target_half_span_lead_times,
            target_max_lead_times=self.config.target_max_lead_times,
        )
        # Freeze assortment and disable promotions for the episode.
        action_dict["activate"] = []
        action_dict["deactivate"] = []
        action_dict["promotions"] = {}
        rl_policy.set_pending_action(action_dict)

        # --- 5. store.decide(store.observe(...)) ---
        obs_dict = store.observe(market.current_step(), item_registry)
        decisions = store.decide(obs_dict)

        # --- 6. Dispatch orders ---
        self._dispatch_orders(store, decisions)

        # --- 7. Settle demand, accumulate reward ---
        self._process_demand(store, decisions)

        # Reward = balance delta attributable to this tick (over ALL catalog
        # products that settled). This equals total profit for the tick.
        balance_after: float = float(store.balance)
        reward: float = balance_after - balance_before

        # Update rolling sales history for the observation encoder.
        for pid, qty in store.sales.items():
            if pid in self._sales_history:
                self._sales_history[pid].append(qty)

        # --- 8. Build next observation ---
        self._step_count += 1
        obs = self._build_observation()

        # --- 9. Termination check ---
        terminated: bool = self._step_count >= self.config.episode_length
        truncated: bool = False

        # --- info: per-SKU traces for business-metric computation ---
        active_set = set(store.active_items)
        info: dict[str, Any] = {
            "step": self._step_count,
            "balance": balance_after,
            "active_products": list(store.active_items),
            "per_sku": {
                pid: {
                    "sales": store.sales.get(pid, 0),
                    "demand": store.demand.get(pid, 0),
                    "inventory": store.inventory.get(pid, 0),
                    "revenue": store.revenue.get(pid, 0.0),
                    "total_cost": store.total_cost.get(pid, 0.0),
                    "holding_cost": store.holding_cost.get(pid, 0.0),
                    "price": store.prices.get(pid, 0.0),
                    "is_active": pid in active_set,
                }
                for pid in store.inventory
            },
        }

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ private helpers

    def _build_observation(self) -> np.ndarray:
        """Encode the current store/market/registry state into the obs tensor."""
        return encode_observation(
            self._store,
            self._market,
            self._item_registry,
            step=self._step_count,
            slot_perm=self._slot_perm,
            K_active=self.config.K_active,
            initial_cash=self._initial_cash,
            sales_history=self._sales_history,
        )

    def _dispatch_orders(self, store: Store, action: dict) -> None:
        """Schedule delivery callbacks for positive-qty orders.

        Verbatim port of ``Runner._dispatch_orders``.
        """
        orders = action.get("order", {})
        if not orders:
            return
        current_step = self._market.current_step()
        supply = self._market.market_state[store.region]["market_supply"]
        supply_factor = max(self._market.params.supply_factor_min, supply)
        for pid, qty in orders.items():
            if qty <= 0:
                continue
            base_lead = store.delivery_lags[pid]
            adjusted_lead = int(base_lead / supply_factor)
            arrival_time = current_step + adjusted_lead
            self._event_engine.schedule(
                event_type="order_arrival",
                delay=arrival_time,
                callback=_make_delivery_callback(store, pid, qty),
            )

    def _process_demand(self, store: Store, action: dict) -> None:
        """Sample realised demand and settle accounting for all catalog products.

        Verbatim port of ``Runner._process_demand``. Every product gets a
        ``world_rng`` draw regardless of active status — this is
        load-bearing for CRN across episodes sharing the same world seed.
        """
        prices = action.get("price", {})
        orders = action.get("order", {})
        current_step = self._market.current_step()
        for pid in list(store.inventory.keys()):
            price = prices.get(pid, store.prices[pid])
            order_qty = orders.get(pid, 0)
            demand = self._market.sample_demand(
                pid, store, price, current_step=current_step
            )
            store.settle(pid, demand=demand, price=price, order_qty=order_qty)
            store.prices[pid] = price

    def render(self) -> None:
        """No-op render (no visual output supported)."""
        pass

    def close(self) -> None:
        """Release any resources held by the env (none currently)."""
        pass


__all__ = ["RLEnv"]
