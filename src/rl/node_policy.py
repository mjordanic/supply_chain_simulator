"""RLNodePolicy — a first-class IntermediatePolicy that wraps a trained RL agent.

Attaching a trained checkpoint to any intermediate node is a one-liner::

    policy_overrides = {"<node-id>": RLNodePolicy.from_checkpoint(path)}
    Runner(scenario, policy_overrides=policy_overrides).run()

Inside ``decide()`` the policy is fully self-contained: it builds the
``(K_MAX, F)`` observation tensor from the engine-provided observation dict and
central-table snapshot, runs the ``SetActor``, decodes the action, applies the
Arbiter via the shared ``arbitrate_orders`` helper, and returns the standard
intermediate action dict.

Design constraints (from PRD Implementation Decisions):
- ``from_checkpoint`` validates layout version via the existing loader.
- Deterministic by default (squashed-Gaussian distribution means).
- Opt-in stochastic mode via ``deterministic=False``; uses a policy-local
  ``torch.Generator``, never global torch state.
- First-tick capture: opening cash + MSRP anchors (tick-0 list prices).
- Rolling sales history accumulated from ``observed_sales``.
- Contention features fed to next observation from last-tick proposals/prices.
- Managed product set: sorted list_prices keys at first ``decide()``.
- Hard error if assortment exceeds K_MAX.
- Supplier routing: cheapest current offer among direct suppliers, tie-break
  by supplier id; no offer → no order line.
- Single-run statefulness: one instance per run, no reset method.

Torch is imported lazily so the module can be imported without torch installed
when tests only exercise the integration plumbing.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.rl.arbiter import arbitrate_orders
from src.rl.set_encoder import (
    K_MAX,
    encode_set_observation,
    decode_set_action,
)
from src.sim.policy import IntermediatePolicy


__all__ = ["RLNodePolicy"]


class RLNodePolicy(IntermediatePolicy):
    """Runner-compatible RL policy that wraps a trained ``SetActor``.

    One instance per simulation run (single-run statefulness).  There is no
    ``reset()`` method; create a fresh instance for each ``Runner.run()`` call.

    Parameters
    ----------
    actor_fn:
        Callable ``(obs_tensor: np.ndarray of shape (K_MAX, F)) → action_arr:
        np.ndarray of shape (K_MAX, 3)``.  This is the only coupling to the
        neural-network code.
    config:
        An object exposing at least ``.arbiter_mode`` and
        ``.cash_budget_fraction``.  Typically an ``RLConfig`` instance.
    deterministic:
        When ``True`` (default) the distribution mean is used directly.
        When ``False`` a sample is drawn using a policy-local ``torch.Generator``
        seeded from ``policy_seed``.
    policy_seed:
        Seed for the policy-local ``torch.Generator`` used in stochastic mode.
        Ignored when ``deterministic=True``.
    base_demand_prior:
        Floor demand rate used as the cold-start prior before sales history
        accumulates.  Defaults to ``1.0``.
    managed_products:
        Optional explicit list of product ids to manage.  When ``None`` (the
        default), the managed set defaults to the node's assortment (sorted
        ``list_prices`` keys) captured at the first ``decide()`` call.
    """

    def __init__(
        self,
        actor_fn: Callable[[np.ndarray], np.ndarray],
        config: Any,
        *,
        deterministic: bool = True,
        policy_seed: int | None = None,
        base_demand_prior: float = 1.0,
        managed_products: Sequence[str] | None = None,
    ) -> None:
        super().__init__(policy_seed=policy_seed)
        self._actor_fn = actor_fn
        self._config = config
        self._deterministic = deterministic
        self._policy_seed = policy_seed
        self._base_demand_prior = base_demand_prior
        self._managed_products_override: tuple[str, ...] | None = (
            tuple(managed_products) if managed_products is not None else None
        )

        # State initialised on first decide() call.
        self._active_subset: tuple[str, ...] | None = None
        self._initial_cash: float | None = None
        # MSRP anchors: tick-0 list prices (catalog MSRP).
        self._msrp_anchors: dict[str, float] = {}
        # Rolling per-tick sales history (deque of ints per pid).
        self._sales_history: dict[str, deque] = {}
        # Last-tick proposals and unit prices for contention features.
        self._last_proposals: dict[str, float] | None = None
        self._last_unit_prices: dict[str, float] | None = None

        # Stochastic: policy-local torch.Generator, created lazily.
        self._torch_gen: Any | None = None

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        config: Any | None = None,
        deterministic: bool = True,
        policy_seed: int | None = None,
        base_demand_prior: float = 1.0,
        managed_products: Sequence[str] | None = None,
    ) -> "RLNodePolicy":
        """Load a trained checkpoint and return a ready-to-use ``RLNodePolicy``.

        Validates the stored ``layout_version`` against the current encoder
        constant; raises ``ValueError`` on mismatch.

        Parameters
        ----------
        path:
            Path to a checkpoint written by ``src.rl.checkpoint.save()``.
        config:
            ``RLConfig`` instance.  When ``None``, a default ``RLConfig()`` is
            used.
        deterministic:
            When ``True`` (default) actions are the squashed-Gaussian means.
        policy_seed:
            Seed for the policy-local generator in stochastic mode.
        base_demand_prior:
            Floor demand rate (cold-start prior).
        managed_products:
            Optional explicit product subset to manage.
        """
        import torch
        from src.rl.agents.set_actor_critic import SetActor
        from src.rl.checkpoint import load as _load
        from src.rl.configs.default import RLConfig

        bundle = _load(path)
        actor = SetActor()
        actor.load_state_dict(bundle["state_dict"])
        actor.eval()

        if config is None:
            cfg_dict = bundle.get("config", {})
            if cfg_dict:
                # Reconstruct from the saved config dict where fields match.
                valid_fields = {f.name for f in RLConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
                filtered = {k: v for k, v in cfg_dict.items() if k in valid_fields}
                try:
                    config = RLConfig(**filtered)
                except Exception:
                    config = RLConfig()
            else:
                config = RLConfig()

        actor_fn = cls._make_actor_fn(actor, deterministic=deterministic, policy_seed=policy_seed)
        return cls(
            actor_fn=actor_fn,
            config=config,
            deterministic=deterministic,
            policy_seed=policy_seed,
            base_demand_prior=base_demand_prior,
            managed_products=managed_products,
        )

    @classmethod
    def from_policy_fn(
        cls,
        policy_fn: Callable[[np.ndarray], np.ndarray],
        config: Any,
        *,
        deterministic: bool = True,
        policy_seed: int | None = None,
        base_demand_prior: float = 1.0,
        managed_products: Sequence[str] | None = None,
    ) -> "RLNodePolicy":
        """Wrap a raw policy callable for use with the Runner.

        Used by the eval harness (issue 03) so ``evaluate()`` keeps its
        ``(policy_fn, ...)`` signature while running through the standard
        Runner + Arbiter path.

        Parameters
        ----------
        policy_fn:
            Callable ``(obs: np.ndarray shape (K_MAX * F,)) → action: np.ndarray
            shape (K_MAX * 3,)`` in flat layout (eval harness convention), OR
            ``(obs: np.ndarray shape (K_MAX, F)) → action: np.ndarray shape
            (K_MAX, 3)`` in matrix layout.  The wrapper auto-detects by ndim.
        config:
            ``RLConfig`` (or any object with ``arbiter_mode`` /
            ``cash_budget_fraction``).
        """
        def _actor_fn(obs_2d: np.ndarray) -> np.ndarray:
            # obs_2d: (K_MAX, F)
            obs_flat = obs_2d.reshape(-1)
            raw = policy_fn(obs_flat)
            arr = np.asarray(raw, dtype=np.float32)
            if arr.ndim == 1:
                return arr.reshape(K_MAX, 3)
            return arr

        return cls(
            actor_fn=_actor_fn,
            config=config,
            deterministic=deterministic,
            policy_seed=policy_seed,
            base_demand_prior=base_demand_prior,
            managed_products=managed_products,
        )

    # ------------------------------------------------------------------
    # IntermediatePolicy.decide()
    # ------------------------------------------------------------------

    def decide(
        self,
        obs_intermediate: Mapping[str, Any],
        central_table: Any,
    ) -> dict[str, Any]:
        """Return the arbitrated order/price/min_order dict for one tick.

        Parameters
        ----------
        obs_intermediate:
            The intermediate observation dict injected by the runner; contains
            at minimum: ``inventory``, ``pending``, ``capacity``, ``cash``,
            ``tick``, ``list_prices``, ``observed_sales``,
            ``direct_supplier_ids``.
        central_table:
            Live ``CentralTable`` snapshot from the runner.
        """
        # --- First-tick initialisation ---
        tick: int = int(obs_intermediate.get("tick", 0))
        if self._active_subset is None:
            self._first_tick_init(obs_intermediate)

        active_subset = self._active_subset  # type: ignore[assignment]

        # --- Accumulate sales history ---
        observed_sales: dict[str, int] = obs_intermediate.get("observed_sales", {})
        for pid in active_subset:
            self._sales_history[pid].append(observed_sales.get(pid, 0))

        # --- Compute unit prices from central table ---
        direct_supplier_ids: list[str] = list(
            obs_intermediate.get("direct_supplier_ids", [])
        )
        unit_prices = self._get_unit_prices(
            active_subset, central_table, direct_supplier_ids, obs_intermediate
        )

        # --- Build (K_MAX, F) observation tensor ---
        from src.rl.encoders import compute_effective_rate

        effective_rate = compute_effective_rate(
            self._sales_history, self._base_demand_prior
        )

        # Build a minimal node-like object for encode_set_observation.
        _node_proxy = _NodeProxy(obs_intermediate, self._msrp_anchors)

        # supplier_ids_for: direct suppliers per product.
        supplier_ids_for: dict[str, list[str]] = {
            pid: direct_supplier_ids for pid in active_subset
        }

        # Build a minimal market-like object for the step field.
        _market_proxy = _MarketProxy(tick)

        obs_2d: np.ndarray = encode_set_observation(
            node=_node_proxy,
            market=_market_proxy,
            active_subset=active_subset,
            initial_cash=float(self._initial_cash),  # type: ignore[arg-type]
            sales_history=self._sales_history,
            central_table=central_table,
            supplier_ids_for=supplier_ids_for,
            proposed_quantities=self._last_proposals,
            unit_prices=self._last_unit_prices,
        )
        # obs_2d: (K_MAX, F)

        # --- Run the actor ---
        action_arr: np.ndarray = self._actor_fn(obs_2d)
        # action_arr: (K_MAX, 3)

        # --- Decode action (supplier id is a placeholder; real routing below) ---
        base_prices = dict(self._msrp_anchors)
        decoded = decode_set_action(
            action=action_arr,
            node=_node_proxy,
            active_subset=active_subset,
            base_prices=base_prices,
            effective_rate=effective_rate,
            supplier_ids=None,  # placeholder; we reassign after arbitration
        )

        # --- Build raw_orders for the Arbiter (supplier id doesn't matter here;
        #     arbitrate_orders sums per-product proposals before allocating) ---
        raw_orders_for_arbiter = decoded.get("order", {})

        # --- Arbiter: allocate quantities subject to headroom/space/cash ---
        priorities: dict[str, float] = {}
        for i, pid in enumerate(active_subset):
            priorities[pid] = float(action_arr[i, 2])

        arbitrated_orders_fkey = arbitrate_orders(
            raw_orders=raw_orders_for_arbiter,
            active_subset=active_subset,
            node=_node_proxy,
            unit_prices=unit_prices,
            priorities=priorities,
            config=self._config,
        )
        # arbitrated_orders_fkey uses "F_<pid>" convention from arbitrate_orders.
        # Replace with the cheapest real supplier.

        # --- Supplier routing: assign each allocated qty to cheapest direct supplier ---
        final_orders = self._route_arbitrated_to_cheapest_supplier(
            arbitrated_orders_fkey,
            active_subset,
            central_table,
            direct_supplier_ids,
        )

        # --- Store proposals for next-tick contention features ---
        # proposals = decoded quantities (before arbiter), for parity with env.
        self._last_proposals = {
            pid: float(sum(qty for _, qty in raw_orders_for_arbiter.get(pid, [])))
            for pid in active_subset
        }
        self._last_unit_prices = dict(unit_prices)

        return {
            "order": final_orders,
            "list_price": decoded.get("list_price", {}),
            "min_order_imposed": decoded.get("min_order_imposed", {}),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _first_tick_init(self, obs: Mapping[str, Any]) -> None:
        """Capture opening cash, MSRP anchors, and managed product set."""
        list_prices: dict[str, float] = dict(obs.get("list_prices", {}))

        if self._managed_products_override is not None:
            active_subset = tuple(self._managed_products_override)
        else:
            active_subset = tuple(sorted(list_prices.keys()))

        if len(active_subset) > K_MAX:
            raise ValueError(
                f"RLNodePolicy: managed product set has {len(active_subset)} products "
                f"but the encoder supports at most K_MAX={K_MAX}.  "
                "Pass a smaller 'managed_products' list or use a scenario with "
                "fewer products on this node."
            )

        self._active_subset = active_subset
        self._initial_cash = float(obs.get("cash", 1.0))
        # MSRP anchors: tick-0 list prices equal catalog MSRPs by convention.
        self._msrp_anchors = {pid: float(list_prices.get(pid, 1.0)) for pid in active_subset}
        # Initialise rolling sales history deques.
        self._sales_history = {pid: deque(maxlen=100) for pid in active_subset}

    def _get_unit_prices(
        self,
        active_subset: tuple[str, ...],
        central_table: Any,
        direct_supplier_ids: list[str],
        obs: Mapping[str, Any],
    ) -> dict[str, float]:
        """Extract cheapest offer prices from central table for each product."""
        unit_prices: dict[str, float] = {}
        list_prices: dict[str, float] = dict(obs.get("list_prices", {}))

        for pid in active_subset:
            if central_table is not None:
                offers = central_table.snapshot_for_buyer(pid)
                if direct_supplier_ids:
                    allowed = set(direct_supplier_ids)
                    offers = [(sid, o) for sid, o in offers if sid in allowed]
                if offers:
                    unit_prices[pid] = float(min(o.list_price for _, o in offers))
                    continue
            # Fallback: use current list price as cost proxy.
            unit_prices[pid] = float(list_prices.get(pid, 1.0))

        return unit_prices

    def _route_arbitrated_to_cheapest_supplier(
        self,
        arbitrated: dict[str, list[tuple[str, int]]],
        active_subset: tuple[str, ...],
        central_table: Any,
        direct_supplier_ids: list[str],
    ) -> dict[str, list[tuple[str, int]]]:
        """Reassign each arbitrated order line to the cheapest current direct supplier.

        The arbiter returns ``F_{pid}`` convention placeholders; we replace them
        with the real cheapest supplier from the central table.

        Ties broken by supplier id (ascending, deterministic).
        No offer available → empty order list for that product.
        """
        routed: dict[str, list[tuple[str, int]]] = {}

        for pid in active_subset:
            lines = arbitrated.get(pid, [])
            total_qty = sum(qty for _, qty in lines)

            if total_qty <= 0:
                routed[pid] = []
                continue

            cheapest_supplier: str | None = None
            if central_table is not None:
                offers = central_table.snapshot_for_buyer(pid)
                if direct_supplier_ids:
                    allowed = set(direct_supplier_ids)
                    offers = [(sid, o) for sid, o in offers if sid in allowed]
                if offers:
                    offers_sorted = sorted(
                        offers,
                        key=lambda x: (float(x[1].list_price), x[0]),
                    )
                    cheapest_supplier = offers_sorted[0][0]

            if cheapest_supplier is None:
                # No offer from any direct supplier this tick → no order line.
                routed[pid] = []
            else:
                routed[pid] = [(cheapest_supplier, total_qty)]

        return routed

    @staticmethod
    def _make_actor_fn(
        actor: Any,
        *,
        deterministic: bool,
        policy_seed: int | None,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Wrap a ``SetActor`` into a numpy-in / numpy-out callable."""
        import torch

        if not deterministic:
            gen = torch.Generator()
            if policy_seed is not None:
                gen.manual_seed(policy_seed)
        else:
            gen = None

        def _fn(obs_2d: np.ndarray) -> np.ndarray:
            # obs_2d: (K_MAX, F)
            obs_t = torch.from_numpy(obs_2d).float().unsqueeze(0)  # (1, K_MAX, F)
            mask = obs_t[0, :, -1]  # last feature = ROW_MASK
            mask_t = mask.unsqueeze(0)  # (1, K_MAX)

            with torch.no_grad():
                dist = actor.get_distribution(obs_t, mask_t)
                if deterministic:
                    # Squashed-Gaussian distribution means: tanh applied to the
                    # pre-squash mean (dist.mean is the raw pre-tanh mean tensor).
                    action_t = torch.tanh(dist.mean)  # (1, K_MAX, 3)
                    # Zero out padded rows to match the sample() convention.
                    action_t = action_t * mask_t.unsqueeze(-1)
                else:
                    action_t = dist.sample(generator=gen)  # (1, K_MAX, 3)

            return action_t.squeeze(0).numpy()  # (K_MAX, 3)

        return _fn


# ------------------------------------------------------------------
# Lightweight proxy objects
# ------------------------------------------------------------------


class _NodeProxy:
    """Duck-typed node proxy built from an intermediate obs dict.

    Provides the attribute interface expected by ``encode_set_observation``
    and ``decode_set_action`` without requiring an actual ``IntermediateNode``.
    """

    __slots__ = (
        "inventory",
        "pending",
        "list_prices",
        "base_prices",
        "costs",
        "capacity",
        "cash",
        "carried_products",
    )

    def __init__(
        self,
        obs: Mapping[str, Any],
        msrp_anchors: dict[str, float],
    ) -> None:
        self.inventory: dict[str, int] = dict(obs.get("inventory", {}))
        self.pending: dict[str, dict[str, int]] = {
            k: dict(v) for k, v in obs.get("pending", {}).items()
        }
        self.list_prices: dict[str, float] = dict(obs.get("list_prices", {}))
        self.base_prices: dict[str, float] = dict(msrp_anchors)
        self.costs: dict[str, float] = {}
        self.capacity: float = float(obs.get("capacity", 1.0))
        self.cash: float = float(obs.get("cash", 0.0))
        self.carried_products: set[str] = set(self.inventory.keys())


class _MarketProxy:
    """Minimal market proxy exposing only the ``step`` attribute."""

    __slots__ = ("step",)

    def __init__(self, step: int) -> None:
        self.step = step
