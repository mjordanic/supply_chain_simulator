# 03 — Topology scaffolder

Status: ready-for-agent

## Parent

`.scratch/setup-files/PRD.md`

## What to build

A deterministic scaffolder so a modeller with a large catalog doesn't have to hand-write
hundreds of sinks and factories. It is the successor to `world_to_graph`, decoupled from the
retired `StoreTemplate`.

`scaffold_topology(catalog, spec) -> {nodes, edges}` emits a starter `nodes:`/`edges:` block
where `spec` carries shop count, `sink_density`, region, and default node sizes. The output is
YAML-serializable and identical in shape to hand-authored topology, so scaffolded and
hand-written setups are indistinguishable and both re-loadable via `load_setup`.

Exposed as `uv run python main.py scaffold <catalog.csv> [options] --out <setup.yaml>`, which
emits or merges a `nodes:`/`edges:` block into the target `setup.yaml`.

## Acceptance criteria

- [ ] `scaffold_topology(catalog, spec)` returns a `{nodes, edges}` structure that `build_graph` accepts as a valid DAG.
- [ ] Sink-per-product / shop-count / `sink_density` expansion is correct for the given spec.
- [ ] Output is deterministic (same catalog + spec ⇒ identical YAML) and re-loadable via `load_setup`.
- [ ] `main.py scaffold <catalog.csv> --out <setup.yaml>` writes/merges a `nodes:`/`edges:` block in the same format a human would author.

## Blocked by

- Issue 02 (Scenario / node-edge types, `load_setup`, `build_graph` wiring)
