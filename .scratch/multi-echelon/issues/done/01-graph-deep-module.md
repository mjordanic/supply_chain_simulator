# Graph deep module + DAG validation

Status: ready-for-agent

## Parent

`.scratch/multi-echelon/PRD.md`

## What to build

Land the `Graph` deep module: a pure structural representation of the typed-node DAG that becomes the world. No runtime behaviour yet; the module is a self-contained library of topology operations consumed by later phases.

The module owns DAG validation, level computation (longest path from any factory), and topology queries (`suppliers_of`, `buyers_of`, `lead_time`). Edges carry per-edge default lead time with optional per-product overrides — today's `Store.delivery_lags` becomes a fallback.

`build_graph(nodes, edges) -> Graph` is the authoring entry point and raises on cycles, unreachable nodes, and same-level supplier links. Echelon levels are computed deterministically.

This slice is additive — no existing module is modified, no existing test is touched. The legacy `Store` engine continues to run unchanged.

## Acceptance criteria

- [ ] `src/sim/graph.py` exports `Graph`, `EdgeSpec`, `build_graph`, `validate_dag`, `compute_levels`
- [ ] `EdgeSpec` carries `supplier_id`, `buyer_id`, `default_lead_time`, optional `per_product_lead_time: dict[str, int] | None`
- [ ] `Graph` exposes `suppliers_of(buyer_id)`, `buyers_of(supplier_id)`, `lead_time(supplier_id, buyer_id, pid)` honouring per-product overrides
- [ ] `validate_dag` rejects cycles, unreachable nodes, and same-level supplier links with clear errors
- [ ] `compute_levels` returns `dict[node_id, int]` matching longest-path-from-any-factory
- [ ] `tests/sim/test_graph_dag.py` covers: cycle rejection, same-level link rejection, unreachable-node rejection, level computation on a multi-tier topology, per-product lead time override resolution
- [ ] `uv run pytest tests/sim -x` is green; no existing test modified

## Blocked by

None — can start immediately
