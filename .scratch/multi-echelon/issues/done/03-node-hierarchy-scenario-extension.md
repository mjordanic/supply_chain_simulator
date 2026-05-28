# Node hierarchy + Scenario extension + NodePolicy ABCs + allocation sub-seed

Status: ready-for-agent

## Parent

`.scratch/multi-echelon/PRD.md`

## What to build

Land the type skeleton that the graph engine will plug into. After this slice, a scenario can declare a graph topology and round-trip through serialisation, but no graph engine is yet running — the legacy `Store` engine continues unchanged.

`Node` ABC + three subclasses (`FactoryNode`, `IntermediateNode`, `DemandSinkNode`) carry the fields named in the PRD's "Node hierarchy" section. `Scenario` gains optional `nodes` and `edges` fields and an `is_graph` property; `StoreInstance`/`stores` stay as the default. `NodeInstance(node, init_seed, policy)` with policy omitted from JSON (mirrors `StoreInstance`).

`policy.py` gets `NodePolicy` ABC (carries `policy_seed`, `policy_rng`) plus three type-paired subclass ABCs (`FactoryPolicy`, `IntermediatePolicy`, `DemandSinkPolicy`). Existing `Policy` classes untouched.

`observation.py` is created with per-node-type observation builder stubs; `allocation.py` is created as an empty stub (real impl in issue 7). `episode_sampler.py` gains the `"allocation"` entry in `_SUB_SEED_PARAMS` — not yet consumed.

`Scenario.nodes_df()` and `Scenario.edges_df()` join the existing DataFrame inspection methods.

This slice is additive — every existing test stays green.

## Acceptance criteria

- [ ] `src/sim/node.py` exports `Node` ABC + `FactoryNode`, `IntermediateNode`, `DemandSinkNode` with fields per PRD
- [ ] `Node` carries `id`, `region`, `policy`, `init_seed`, computed `level`
- [ ] Per-class fields match the PRD spec (factory: `produces_product_id`, `unit_cost`, `capacity_per_tick`, `inventory`, `list_price`; intermediate: `carried_products`, `capacity`, `tags`, `inventory`, `pending`, `list_prices`, `min_order_imposed`; sink: `product_id`, `demand_dist`, `income_rate`, `cash`, `activation_tick`)
- [ ] `src/sim/observation.py` exists with per-node-type builder stubs
- [ ] `src/sim/allocation.py` exists as empty stub
- [ ] `src/sim/policy.py` adds `NodePolicy` + `FactoryPolicy` + `IntermediatePolicy` + `DemandSinkPolicy` ABCs; existing classes untouched
- [ ] `src/sim/scenario.py` adds optional `nodes`, `edges`, `is_graph` property, `NodeInstance(node, init_seed, policy)`, `nodes_df()`, `edges_df()`; `StoreInstance`/`stores` remain the default; `NodeInstance.policy` omitted from JSON
- [ ] `src/sim/episode_sampler.py` adds `"allocation": (0xA24B_AED4, 0x0000_0007)` to `_SUB_SEED_PARAMS` (unused so far)
- [ ] `tests/sim/test_node_construction.py` covers: per-class init via `init_seed` is bit-identical regardless of attached policy (extends `test_two_stores_same_init_seed_identical_step0` to all three node types)
- [ ] `tests/sim/test_scenario_graph_roundtrip.py` covers: `Scenario` with `nodes`/`edges` survives `to_dict`/`from_dict` round-trip; `NodeInstance.policy` omitted as specified
- [ ] `uv run pytest tests/sim -x` is green; no existing test broken

## Blocked by

- `.scratch/multi-echelon/issues/01-graph-deep-module.md`
- `.scratch/multi-echelon/issues/02-central-table-deep-module.md`
