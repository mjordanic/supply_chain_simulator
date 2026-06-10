# 01 — Type-based graph validation (lateral edges legal)

Status: done

## Parent

`.scratch/lateral-links-scheduling/PRD.md` (ADR 0018)

## What to build

Replace level-based DAG validation with **type-based** validation so the modeller can author
lateral `intermediate→intermediate` edges and the graph rejects only genuinely illegal edges.

- Drop the BFS same-level check from `validate_dag`.
- Add **type rules** on edge endpoints: supplier ∈ {factory, intermediate}, buyer ∈
  {intermediate, sink}. Concretely this rejects `factory→sink` (flow must pass through ≥1
  intermediate), `→factory` (factories don't buy), and `sink→*` (sinks don't sell). These edges
  were previously silently mishandled at runtime, so rejecting them at authoring time is strictly
  safer.
- Keep the cycle (3-colour DFS), self-loop, and unreachable-node checks. The union of all edges
  must remain acyclic.
- `validate_dag` / `build_graph` gain a `node_types: dict[str, str]` parameter (id →
  `Node._node_type`, values `"factory"` / `"intermediate"` / `"demand_sink"`). graph.py must stay
  free of any `src.sim` import — types arrive as plain strings.
- `build_world` constructs `node_types` from `scenario.nodes` and passes it through.
- Each validation error names the rule it violated so a modeller can fix the topology quickly.

Scheduling is **not** touched in this slice — the runner stays on the existing cascade. A lateral
graph will *build* here; correct lateral settlement lands in issue 02. So this slice's runtime
guarantee is only that existing (non-lateral) scenarios still run unchanged.

## Acceptance criteria

- [ ] `build_graph` / `validate_dag` accept an `intermediate→intermediate` (lateral) DAG.
- [ ] `factory→sink`, `→factory`, and `sink→*` edges each raise a clear error naming the violated rule.
- [ ] `factory→intermediate` and `intermediate→sink` edges are accepted.
- [ ] Cycles, self-loops, and unreachable nodes are still rejected.
- [ ] `validate_dag` / `build_graph` take `node_types: dict[str, str]`; graph.py imports nothing from `src.sim`.
- [ ] `build_world` constructs `node_types` from the scenario and existing chain setups build & run unchanged (regression).
- [ ] Unit tests on the pure validator cover each accept/reject case above.

## Blocked by

- None — can start immediately.
