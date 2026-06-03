# Notebook `05-topology-gallery` (headline, NEW)

Status: ready-for-agent

## Parent

`.scratch/notebooks-multi-echelon-rewrite/PRD.md`

## What to build

A brand-new `05-topology-gallery` notebook — the headline deliverable of the refactor. The
multi-echelon engine's most important added capability (multiple topologies) is currently shown
nowhere; every example scenario tops out at three echelon levels. This notebook builds, draws, runs
live, and compares five graph shapes on KPIs.

Each graph is drawn with `networkx.multipartite_layout` keyed on echelon level (`Simulation.levels`
/ `Node.level`, populated by `build_world`), run live, and compared on KPIs (e.g. system cash
growth, service level, profit per dollar). The five shapes:

1. **Linear chain (3-tier)** — `1F → 1S → 1 sink`. Reference shape; explains echelon levels + tick cascade.
2. **Multi-supplier contention** — `2F → 2S → 2 sinks` with shared suppliers; demonstrates the
   central table + FCFS allocator under contention (`fill_rate_recent` drops, cheapest-first routing).
3. **Deep 4-tier (new)** — `factory → warehouse/DC → shops → sinks` (four echelon levels). The
   genuine multi-echelon demonstration no current scenario shows.
4. **Fan-out fleet** — few factories → several shops → many per-product sinks (the `llm_world_20` shape).
5. **Two disconnected sub-graphs** — two independent factory→shop→sink chains sharing no nodes in
   one `Scenario`; valid because `build_graph` rejects cycles / same-level supplier links / nodes
   unreachable-from-any-factory, **not** disconnected components.

Topologies 1, 2, and the fleet reuse existing `scenarios/*.py`
(`example_two_factories_two_shops.py` for contention, the 3-node chain, the fleet shape). The 4-tier
and disconnected graphs are built with **inline `build_*()` helper functions inside the notebook** —
no new `scenarios/*.py` files, no `DataExporter` involvement. KPI extraction goes through
`src/sim/inspect.py` so the showcase stays thin over the shared, tested parser.

## Acceptance criteria

- [ ] `notebooks/05-topology-gallery.ipynb` builds, draws, runs live, and compares all five topologies above
- [ ] Each graph is drawn with `networkx.multipartite_layout` keyed on echelon level (`Node.level` / `Simulation.levels`)
- [ ] The 4-tier and two-disconnected-sub-graph topologies are defined by inline `build_*()` helpers in the notebook (no new `scenarios/*.py`); the disconnected case runs to completion in one `Scenario`
- [ ] Topologies are compared on KPIs (e.g. system cash growth, service level, profit per dollar), with KPI series derived via `src/sim/inspect.py`
- [ ] The contention topology visibly exercises the central table / FCFS allocator (e.g. `fill_rate_recent` under contention)
- [ ] `%matplotlib inline`; committed with embedded executed outputs; modest DPI
- [ ] `jupyter nbconvert --to notebook --execute` runs the notebook with zero cell errors and a rendered figure for every plotting cell

## Blocked by

- `.scratch/notebooks-multi-echelon-rewrite/issues/01-inspect-deep-module.md`
