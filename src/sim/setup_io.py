"""Setup I/O — the only module that knows the on-disk format.

``load_setup(dir) -> Scenario`` reads ``catalog.csv`` and ``setup.yaml``
from a setup directory, validates both files, builds typed Node/EdgeSpec
objects, derives all seeds from ``world_seed``, resolves policies via the
registry, and returns an in-memory ``Scenario``.

Seed derivation
---------------
All randomness derives from a single ``world_seed`` in ``setup.yaml``:

    init_seed[node_id]   = _derive_node_seed(world_seed, node_id)
    policy_seed[node_id] = _derive_node_seed(world_seed, node_id, "policy")

``_derive_node_seed`` uses a FNV-1a-32 hash of the string arguments
XORed with world_seed via the same multiply-add-mask style as the
existing episode-sampler sub-seed derivation (ADR 0016).

Validation errors
-----------------
A single clear ``ValueError`` is raised for:
- Unknown policy name
- Missing required node / edge field
- Malformed Distribution tag
- Duplicate node id
- Graph errors (cycles, unreachable nodes, same-level links)

``write_setup`` is deliberately out of scope for this slice —
it lands with the LLM generator (issue 04). The reproducibility
snapshot here is a verbatim file copy.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml  # pyyaml — already a transitive dep


# ---------------------------------------------------------------------------
# Seed derivation
# ---------------------------------------------------------------------------

_FNV_PRIME_32 = 0x01000193
_FNV_OFFSET_32 = 0x811C9DC5
_MASK_32 = 0xFFFF_FFFF


def _fnv1a_32(s: str) -> int:
    """FNV-1a 32-bit hash of a UTF-8 string."""
    h = _FNV_OFFSET_32
    for byte in s.encode():
        h ^= byte
        h = (h * _FNV_PRIME_32) & _MASK_32
    return h


def _derive_node_seed(world_seed: int, node_id: str, purpose: str = "") -> int:
    """Derive a deterministic 32-bit seed for a (world_seed, node_id[, purpose]) triple.

    Uses the same multiply-add-mask style as ``episode_sampler._derive_seed``.
    The node_id (and optional purpose suffix) are hashed via FNV-1a-32 to
    produce a per-node prime-like multiplier, then mixed with world_seed.
    """
    key = node_id if not purpose else f"{node_id}:{purpose}"
    node_hash = _fnv1a_32(key)
    # Mix: (world_seed * node_hash + node_hash) & mask — ensures different
    # (world_seed, key) pairs produce different seeds.
    prime = node_hash | 1  # force odd so it's co-prime to 2^32
    return (world_seed * prime + node_hash) & _MASK_32


# ---------------------------------------------------------------------------
# Distribution parsing from YAML tagged form
# ---------------------------------------------------------------------------

def _parse_distribution(raw: Any, field_path: str) -> Any:
    """Parse a tagged distribution dict or pass through a scalar.

    Tagged form: ``{kind: normal, mean: 10, std: 2}`` (YAML) maps to
    the existing ``kind`` taxonomy (constant|uniform|normal|choice|log_uniform).

    Scalars (int/float/bool/str) are returned unchanged.

    Raises ``ValueError`` for malformed tags.
    """
    from src.sim.distributions import (
        Constant, Uniform, Normal, Choice, LogUniform,
        Distribution,
    )

    if isinstance(raw, (int, float, bool)):
        return raw
    if isinstance(raw, Distribution):
        return raw
    if not isinstance(raw, dict):
        raise ValueError(
            f"{field_path}: expected a scalar or tagged distribution dict, "
            f"got {type(raw).__name__!r} ({raw!r})"
        )
    if "kind" not in raw:
        raise ValueError(
            f"{field_path}: distribution dict missing required 'kind' key "
            f"(valid: constant, uniform, normal, choice, log_uniform). Got: {raw!r}"
        )
    kind = raw["kind"]
    rest = {k: v for k, v in raw.items() if k != "kind"}
    try:
        if kind == "constant":
            return Constant(value=rest["value"])
        elif kind == "uniform":
            return Uniform(low=float(rest["low"]), high=float(rest["high"]))
        elif kind == "normal":
            clip = rest.get("clip")
            if clip is not None:
                clip = (float(clip[0]), float(clip[1]))
            return Normal(
                mean=float(rest["mean"]),
                std=float(rest["std"]),
                clip=clip,
            )
        elif kind == "choice":
            weights = rest.get("weights")
            return Choice(options=list(rest["options"]), weights=weights)
        elif kind == "log_uniform":
            return LogUniform(low=float(rest["low"]), high=float(rest["high"]))
        else:
            raise ValueError(
                f"{field_path}: unknown distribution kind {kind!r} "
                f"(valid: constant, uniform, normal, choice, log_uniform)"
            )
    except KeyError as exc:
        raise ValueError(
            f"{field_path}: malformed distribution {raw!r} — missing key {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# catalog.csv parsing
# ---------------------------------------------------------------------------

_CATALOG_REQUIRED = ("product_id", "name", "category", "base_price", "unit_cost", "seasonality")


def _parse_catalog(csv_path: Path) -> list[Any]:
    """Parse ``catalog.csv`` into a list of ``Ware`` objects.

    Expected columns:
        product_id, name, category, base_price, unit_cost, seasonality,
        related_products (optional), init_stock_share (optional)

    ``related_products`` is a string-encoded list of ``pid:weight`` pairs
    separated by ``;`` (e.g. ``P0001:0.5;P0002:0.3``). Empty string = none.
    """
    from src.sim.scenario import Ware

    if not csv_path.is_file():
        raise ValueError(f"catalog.csv not found at {csv_path}")

    wares: list[Ware] = []
    seen_ids: set[str] = set()

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path}: empty catalog.csv")
        missing_cols = set(_CATALOG_REQUIRED) - set(reader.fieldnames)
        if missing_cols:
            raise ValueError(
                f"{csv_path}: catalog.csv missing required columns: "
                f"{sorted(missing_cols)}"
            )

        for row_num, row in enumerate(reader, start=2):
            pid = row["product_id"].strip()
            if not pid:
                raise ValueError(f"{csv_path} row {row_num}: product_id is empty")
            if pid in seen_ids:
                raise ValueError(
                    f"{csv_path} row {row_num}: duplicate product_id {pid!r}"
                )
            seen_ids.add(pid)

            # Parse related_products: "P0001:0.5;P0002:0.3" → [(P0001, 0.5), ...]
            related_raw = row.get("related_products", "").strip()
            related: list[tuple[str, float]] = []
            if related_raw:
                for pair in related_raw.split(";"):
                    pair = pair.strip()
                    if not pair:
                        continue
                    if ":" not in pair:
                        raise ValueError(
                            f"{csv_path} row {row_num}: malformed related_products "
                            f"pair {pair!r} (expected 'pid:weight')"
                        )
                    rel_pid, rel_weight = pair.rsplit(":", 1)
                    try:
                        related.append((rel_pid.strip(), float(rel_weight)))
                    except ValueError:
                        raise ValueError(
                            f"{csv_path} row {row_num}: invalid weight in "
                            f"related_products pair {pair!r}"
                        )

            # init_stock_share — optional, default 1.0
            init_stock_share_raw = row.get("init_stock_share", "").strip()
            init_stock_share: float | None = None
            if init_stock_share_raw:
                try:
                    init_stock_share = float(init_stock_share_raw)
                except ValueError:
                    raise ValueError(
                        f"{csv_path} row {row_num}: invalid init_stock_share "
                        f"{init_stock_share_raw!r}"
                    )

            try:
                wares.append(
                    Ware(
                        product_id=pid,
                        name=row["name"].strip(),
                        category=row["category"].strip(),
                        base_price=float(row["base_price"]),
                        unit_cost=float(row["unit_cost"]),
                        seasonality=row["seasonality"].strip(),
                        related_products=related,
                        init_stage=None,
                        stage_change_probs=None,
                        freshness_alpha=None,
                        freshness_decay=None,
                        init_stock_share=init_stock_share,
                    )
                )
            except (ValueError, KeyError) as exc:
                raise ValueError(
                    f"{csv_path} row {row_num}: failed to parse row — {exc}"
                ) from exc

    if not wares:
        raise ValueError(f"{csv_path}: catalog.csv contains no product rows")
    return wares


# ---------------------------------------------------------------------------
# setup.yaml parsing helpers
# ---------------------------------------------------------------------------

def _require(d: dict, key: str, path: str) -> Any:
    """Return ``d[key]`` or raise a clear ``ValueError``."""
    if key not in d:
        raise ValueError(f"{path}: missing required field '{key}'")
    return d[key]


def _parse_market(raw: dict, path: str) -> Any:
    """Parse the ``market:`` block into a ``MarketParams`` instance."""
    from src.sim.scenario import MarketParams

    def _field(key: str) -> Any:
        return _require(raw, key, f"{path}.{key}")

    # Distribution-valued fields
    cycle_len = _parse_distribution(_field("cycle_len"), f"{path}.cycle_len")
    cycle_amp = _parse_distribution(_field("cycle_amp"), f"{path}.cycle_amp")
    peak_factor = _parse_distribution(_field("peak_factor"), f"{path}.peak_factor")
    off_factor = _parse_distribution(_field("off_factor"), f"{path}.off_factor")
    trend = _parse_distribution(_field("trend"), f"{path}.trend")
    demand_shock = _parse_distribution(_field("demand_shock"), f"{path}.demand_shock")
    supply_shock = _parse_distribution(_field("supply_shock"), f"{path}.supply_shock")
    base_demand = _parse_distribution(_field("base_demand"), f"{path}.base_demand")
    severity_raw = raw.get("severity")

    # stage_multipliers: optional, default to empty dict (lifecycle removed)
    stage_multipliers_raw = raw.get("stage_multipliers", {})
    stage_multipliers = {
        k: _parse_distribution(v, f"{path}.stage_multipliers.{k}")
        for k, v in stage_multipliers_raw.items()
    }

    cross_factor_range_raw = _field("cross_factor_range")
    if isinstance(cross_factor_range_raw, (list, tuple)) and len(cross_factor_range_raw) == 2:
        cross_factor_range = (float(cross_factor_range_raw[0]), float(cross_factor_range_raw[1]))
    else:
        raise ValueError(
            f"{path}.cross_factor_range: expected [lo, hi] pair, "
            f"got {cross_factor_range_raw!r}"
        )

    return MarketParams(
        cycle_len=cycle_len,
        cycle_amp=cycle_amp,
        init_demand=float(_field("init_demand")),
        init_supply=float(_field("init_supply")),
        peak_factor=peak_factor,
        off_factor=off_factor,
        season_months=dict(_field("season_months")),
        regions=list(_field("regions")),
        correlation=float(_field("correlation")),
        trend_update_interval=int(_field("trend_update_interval")),
        min_value=float(_field("min_value")),
        max_value=float(_field("max_value")),
        stage_multipliers=stage_multipliers,
        price_elasticity=float(_field("price_elasticity")),
        promo_multiplier=float(_field("promo_multiplier")),
        demand_factor_min=float(_field("demand_factor_min")),
        supply_factor_min=float(_field("supply_factor_min")),
        cross_inv_lo=float(_field("cross_inv_lo")),
        cross_inv_hi=float(_field("cross_inv_hi")),
        cross_factor_range=cross_factor_range,
        trend=trend,
        demand_shock=demand_shock,
        supply_shock=supply_shock,
        base_demand=base_demand,
    )


def _parse_disruption(raw: dict, path: str) -> Any:
    """Parse the ``disruption:`` block into a ``DisruptionParams`` instance."""
    from src.sim.scenario import DisruptionParams

    return DisruptionParams(
        event_prob=float(_require(raw, "event_prob", path)),
        types=list(_require(raw, "types", path)),
        regions=list(_require(raw, "regions", path)),
        severity=_parse_distribution(
            _require(raw, "severity", path), f"{path}.severity"
        ),
        duration=_parse_distribution(
            _require(raw, "duration", path), f"{path}.duration"
        ),
    )


def _parse_factory_node(raw: dict, path: str, world_seed: int) -> Any:
    """Parse a factory node dict into a ``FactoryNode`` instance."""
    from src.sim.node import FactoryNode

    node_id = _require(raw, "id", path)
    region = _require(raw, "region", path)
    capacity_per_tick = _parse_distribution(
        _require(raw, "capacity_per_tick", path), f"{path}.capacity_per_tick"
    )
    unit_cost = float(_require(raw, "unit_cost", path))
    list_price = float(_require(raw, "list_price", path))
    inventory = int(raw.get("inventory", 0))
    init_seed = _derive_node_seed(world_seed, node_id)

    return FactoryNode(
        id=node_id,
        region=region,
        init_seed=init_seed,
        produces_product_id=_require(raw, "produces_product_id", path),
        unit_cost=unit_cost,
        capacity_per_tick=capacity_per_tick,
        inventory=inventory,
        list_price=list_price,
        cash=float(raw.get("cash", 0.0)),
    )


def _parse_intermediate_node(raw: dict, path: str, world_seed: int) -> Any:
    """Parse an intermediate node dict into an ``IntermediateNode`` instance."""
    from src.sim.node import IntermediateNode
    from src.sim.distributions import Distribution

    node_id = _require(raw, "id", path)
    region = _require(raw, "region", path)
    carried_products = set(_require(raw, "carried_products", path))
    capacity_raw = _require(raw, "capacity", path)
    capacity = _parse_distribution(capacity_raw, f"{path}.capacity")
    # If capacity is a Distribution, sample it with the node's init_seed
    # to get the scalar capacity.
    init_seed = _derive_node_seed(world_seed, node_id)
    if isinstance(capacity, Distribution):
        from random import Random
        capacity = int(capacity.sample(Random(init_seed)))

    tags = list(raw.get("tags", []))
    inventory_raw = raw.get("inventory", {})
    inventory = {k: int(v) for k, v in inventory_raw.items()} if inventory_raw else {}
    list_prices_raw = raw.get("list_prices", {})
    list_prices = {k: float(v) for k, v in list_prices_raw.items()} if list_prices_raw else {}
    min_order_imposed_raw = raw.get("min_order_imposed", {})
    min_order_imposed = {k: int(v) for k, v in min_order_imposed_raw.items()} if min_order_imposed_raw else {}

    return IntermediateNode(
        id=node_id,
        region=region,
        init_seed=init_seed,
        carried_products=carried_products,
        capacity=capacity,
        tags=tags,
        inventory=inventory,
        list_prices=list_prices,
        min_order_imposed=min_order_imposed,
        cash=float(raw.get("cash", 10000.0)),
    )


def _parse_demand_sink_node(raw: dict, path: str, world_seed: int) -> Any:
    """Parse a demand_sink node dict into a ``DemandSinkNode`` instance."""
    from src.sim.node import DemandSinkNode

    node_id = _require(raw, "id", path)
    region = _require(raw, "region", path)
    product_id = _require(raw, "product_id", path)
    demand_dist = _parse_distribution(
        _require(raw, "demand_dist", path), f"{path}.demand_dist"
    )
    income_rate = float(_require(raw, "income_rate", path))
    cash = float(raw.get("cash", 10000.0))
    init_seed = _derive_node_seed(world_seed, node_id)

    return DemandSinkNode(
        id=node_id,
        region=region,
        init_seed=init_seed,
        product_id=product_id,
        demand_dist=demand_dist,
        income_rate=income_rate,
        cash=cash,
    )


def _parse_node(raw: dict, index: int, world_seed: int) -> Any:
    """Dispatch to the right node parser based on the ``type`` field."""
    path = f"nodes[{index}]"
    node_type = _require(raw, "type", path)
    if node_type == "factory":
        return _parse_factory_node(raw, path, world_seed)
    elif node_type == "intermediate":
        return _parse_intermediate_node(raw, path, world_seed)
    elif node_type == "demand_sink":
        return _parse_demand_sink_node(raw, path, world_seed)
    else:
        raise ValueError(
            f"{path}: unknown node type {node_type!r} "
            f"(valid: factory, intermediate, demand_sink)"
        )


def _parse_edge(raw: dict, index: int, node_ids: set[str]) -> Any:
    """Parse one edge dict into an ``EdgeSpec``."""
    from src.sim.graph import EdgeSpec

    path = f"edges[{index}]"
    supplier = _require(raw, "supplier", path)
    buyer = _require(raw, "buyer", path)
    lead_time = int(_require(raw, "lead_time", path))

    # Validate node refs exist
    for ref, label in ((supplier, "supplier"), (buyer, "buyer")):
        if ref not in node_ids:
            raise ValueError(
                f"{path}: {label} node {ref!r} not found in nodes list"
            )

    per_product = raw.get("per_product_lead_time")
    if per_product is not None:
        per_product = {k: int(v) for k, v in per_product.items()}

    return EdgeSpec(
        supplier_id=supplier,
        buyer_id=buyer,
        default_lead_time=lead_time,
        per_product_lead_time=per_product,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def load_setup(setup_dir: str | Path) -> Any:
    """Load a setup directory into an in-memory ``Scenario``.

    Parameters
    ----------
    setup_dir:
        Path to a directory containing ``catalog.csv`` and ``setup.yaml``.

    Returns
    -------
    ``Scenario`` — fully populated, all policies resolved, all seeds derived.

    Raises
    ------
    ValueError
        On any validation error (missing fields, unknown policy, malformed
        distribution, duplicate id, graph topology errors).
    FileNotFoundError
        If ``setup_dir``, ``catalog.csv``, or ``setup.yaml`` do not exist.
    """
    from src.sim.graph import EdgeSpec, build_graph
    from src.sim.policy_registry import build_policy
    from src.sim.scenario import (
        DisruptionParams,
        ItemLifecycleParams,
        MarketParams,
        NodeInstance,
        Scenario,
        Ware,
    )

    setup_dir = Path(setup_dir)
    if not setup_dir.is_dir():
        raise FileNotFoundError(f"Setup directory not found: {setup_dir}")

    catalog_path = setup_dir / "catalog.csv"
    yaml_path = setup_dir / "setup.yaml"

    if not catalog_path.is_file():
        raise FileNotFoundError(f"catalog.csv not found in {setup_dir}")
    if not yaml_path.is_file():
        raise FileNotFoundError(f"setup.yaml not found in {setup_dir}")

    # --- Load YAML -------------------------------------------------------
    with yaml_path.open(encoding="utf-8") as f:
        try:
            doc = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError(f"setup.yaml: YAML parse error — {exc}") from exc

    if not isinstance(doc, dict):
        raise ValueError("setup.yaml: top-level must be a mapping")

    # --- Run section -----------------------------------------------------
    run_raw = _require(doc, "run", "setup.yaml")
    n_steps = int(_require(run_raw, "n_steps", "setup.yaml.run"))
    world_seed = int(_require(run_raw, "world_seed", "setup.yaml.run"))
    start_date_raw = _require(run_raw, "start_date", "setup.yaml.run")
    if isinstance(start_date_raw, str):
        try:
            start_date = datetime.fromisoformat(start_date_raw)
        except ValueError as exc:
            raise ValueError(
                f"setup.yaml.run.start_date: invalid ISO date {start_date_raw!r} — {exc}"
            ) from exc
    elif isinstance(start_date_raw, datetime):
        start_date = start_date_raw
    else:
        # YAML may parse a bare date as datetime.date
        from datetime import date
        if isinstance(start_date_raw, date):
            start_date = datetime(start_date_raw.year, start_date_raw.month, start_date_raw.day)
        else:
            raise ValueError(
                f"setup.yaml.run.start_date: expected ISO date string, "
                f"got {type(start_date_raw).__name__!r}"
            )

    # --- catalog.csv -----------------------------------------------------
    catalog: list[Ware] = _parse_catalog(catalog_path)

    # --- Validate duplicate node ids first (before building nodes) -------
    nodes_raw = _require(doc, "nodes", "setup.yaml")
    if not isinstance(nodes_raw, list):
        raise ValueError("setup.yaml.nodes: must be a list")
    seen_node_ids: set[str] = set()
    for i, nr in enumerate(nodes_raw):
        nid = _require(nr, "id", f"nodes[{i}]")
        if nid in seen_node_ids:
            raise ValueError(
                f"setup.yaml.nodes[{i}]: duplicate node id {nid!r}"
            )
        seen_node_ids.add(nid)

    # --- Parse nodes -----------------------------------------------------
    raw_nodes: list[Any] = [_parse_node(nr, i, world_seed) for i, nr in enumerate(nodes_raw)]

    # --- Parse edges -----------------------------------------------------
    edges_raw = _require(doc, "edges", "setup.yaml")
    if not isinstance(edges_raw, list):
        raise ValueError("setup.yaml.edges: must be a list")
    edges: list[EdgeSpec] = [
        _parse_edge(er, i, seen_node_ids) for i, er in enumerate(edges_raw)
    ]

    # --- Validate DAG topology (cycles, unreachable, same-level) ----------
    try:
        build_graph(list(seen_node_ids), edges)
    except ValueError as exc:
        raise ValueError(f"setup.yaml graph topology error: {exc}") from exc

    # --- Market + Disruption ---------------------------------------------
    market_raw = _require(doc, "market", "setup.yaml")
    market_params: MarketParams = _parse_market(market_raw, "setup.yaml.market")

    disruption_raw = _require(doc, "disruption", "setup.yaml")
    disruption_params: DisruptionParams = _parse_disruption(
        disruption_raw, "setup.yaml.disruption"
    )

    # --- ItemLifecycleParams (legacy compat — minimal defaults) ----------
    # The setup-files model has no lifecycle; we construct a minimal valid
    # ItemLifecycleParams so existing Scenario/DataExporter code still works.
    from src.sim.distributions import Constant
    item_lifecycle = ItemLifecycleParams(
        stages=["introduction", "growth", "maturity", "decline", "dead"],
        init_stage="maturity",
        default_stage_change_probs={
            "introduction": 0.0,
            "growth": 0.0,
            "maturity": 0.0,
            "decline": 0.0,
            "dead": 0.0,
        },
        default_freshness_alpha=0.0,
        default_freshness_decay=1.0,
        default_init_stock_share=1.0,
    )

    # --- Resolve policies and build NodeInstance list --------------------
    node_instances: list[NodeInstance] = []
    for i, (raw_node_dict, node) in enumerate(zip(nodes_raw, raw_nodes)):
        policy_spec = raw_node_dict.get("policy")
        policy = None
        if policy_spec is not None:
            policy_name = _require(policy_spec, "name", f"nodes[{i}].policy")
            policy_params = dict(policy_spec.get("params", {}))
            policy_seed = _derive_node_seed(world_seed, node.id, "policy")
            try:
                policy = build_policy(
                    policy_name,
                    policy_params,
                    node=node,
                    edges=edges,
                    policy_seed=policy_seed,
                )
            except ValueError as exc:
                raise ValueError(
                    f"nodes[{i}] (id={node.id!r}): {exc}"
                ) from exc

        node_instances.append(
            NodeInstance(
                node=node,
                init_seed=node.init_seed,
                policy=policy,
            )
        )

    # --- Assemble Scenario -----------------------------------------------
    return Scenario(
        catalog=catalog,
        market=market_params,
        disruption=disruption_params,
        item_lifecycle=item_lifecycle,
        stores=[],  # no legacy stores in setup-dir scenarios
        nodes=node_instances,
        edges=edges,
        n_steps=n_steps,
        start_date=start_date,
        world_seed=world_seed,
    )


__all__ = ["load_setup", "_derive_node_seed"]
