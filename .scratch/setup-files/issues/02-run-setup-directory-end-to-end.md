# 02 — Run a hand-authored setup directory end-to-end

Status: ready-for-agent

## Parent

`.scratch/setup-files/PRD.md`

## What to build

The core tracer bullet: make `uv run python main.py run <setup-dir>` load a directory of
human-authored data files, run the simulation deterministically, and write outputs — with no
Python scenario file involved.

A *simulation setup* is a directory containing:

- `catalog.csv` — one row per product, columns
  `product_id, name, category, base_price, unit_cost, seasonality, related_products, init_stock_share`.
  `related_products` is a string-encoded list of `pid:weight` pairs separated by `;` (empty =
  none). Prices are scalars.
- `setup.yaml` — top-level keys `market`, `disruption`, `nodes`, `edges`, `run`.
  - `nodes[]`: `id`, `type` (`factory|intermediate|demand_sink`), `region`, type-specific fields
    (factory: `produces_product_id`, `unit_cost`, `capacity_per_tick`, `inventory`, `list_price`;
    intermediate: `carried_products`, `capacity`, `tags`, `inventory` map, `list_prices` map,
    `min_order_imposed` map; demand_sink: `product_id`, `demand_dist`, `income_rate`, `cash`),
    and an optional `policy: {name, params}`.
  - `edges[]`: `supplier`, `buyer`, `lead_time` (optional `per_product_lead_time` map).
  - `run`: `n_steps`, `start_date`, `world_seed`.
  - `Distribution`-valued fields use a tagged form `{kind: normal, mean: 10, std: 2}` matching
    the existing taxonomy (`constant|uniform|normal|choice|log_uniform`).

Deliver, as one end-to-end path:

- **Setup I/O (deep module):** `load_setup(dir) -> Scenario` parses both files, validates,
  builds typed `Node`/`EdgeSpec` and the in-memory `Scenario` (`catalog, market, disruption,
  nodes, edges, n_steps, start_date, world_seed`), and resolves policies. It is the only module
  that knows the on-disk format.
- **Determinism:** `world_seed` is the only seed in the file. The loader derives each node's
  `init_seed = derive(world_seed, node_id)` and `policy_seed = derive(world_seed, node_id,
  "policy")` using the existing `_derive_seed` hashing; `allocation_rng` continues to derive
  from `world_seed` (ADR 0016).
- **Policy registry + fact injection (deep module):** a `{name: class}` registry of stable
  snake_case names (`order_up_to`, `reorder_point`, `periodic_order_up_to`, `periodic_reorder`,
  `single_supplier`, `static_factory`, `default_demand_sink`). `build_policy(name, params, *,
  node, edges, policy_seed)` resolves the class, injects node/edge facts (capacity, `unit_cost`,
  `list_prices`, supplier ids, lead times) and `policy_seed`, then splats remaining `params` as
  kwargs. Policy constructors are refactored so injected facts arrive via injection, not as
  re-declared required kwargs; `params` carries only true hyperparameters.
- **CLI:** `run <setup-dir> [--output DIR]` (default output `data/<setup-dir-name>/`). The old
  "pass a `scenarios/*.py` path" entry point is removed.
- **Output:** node static table written as `nodes.parquet`; the input `catalog.csv` + `setup.yaml`
  are copied verbatim into `config/` of the output folder as the reproducibility snapshot.
  Timeseries, run log, and overview PNG unchanged.
- **Validation errors:** a single clear error for an unknown policy name, a missing required
  node/edge field, a malformed `Distribution` tag, a duplicate node id, and graph errors
  (cycles / unreachable nodes / same-level supplier links — delegated to `build_graph`).
- A committed no-API-key example setup directory under `setups/` (e.g. a 3-node chain) that
  runs immediately.

`write_setup` (Scenario → files) and its round-trip are NOT part of this slice — the snapshot
here is a verbatim file copy. `write_setup` lands with the LLM generator (issue 04).

## Acceptance criteria

- [ ] `uv run python main.py run setups/<example>` runs end-to-end and writes outputs including `nodes.parquet` and a verbatim `config/` snapshot of the two input files.
- [ ] `catalog.csv` (incl. `related_products` parsing) and `setup.yaml` load into a typed `Scenario`; `build_graph` accepts the resulting DAG.
- [ ] All seeds derive from a single `world_seed`; running the identical setup directory twice produces a bit-identical run log.
- [ ] Each policy is selected by name from the registry; injected node/edge facts reach the constructed policy and match the node/edge; `params` holds only hyperparameters.
- [ ] Malformed setups (unknown policy name, missing required field, bad `Distribution` tag, duplicate id, cyclic graph) each raise a single clear, actionable error.
- [ ] A committed example setup under `setups/` runs with no API key; a regression/snapshot test covers it end-to-end.

## Blocked by

- Issue 01 (simplified demand core / relocated CRN loop)
