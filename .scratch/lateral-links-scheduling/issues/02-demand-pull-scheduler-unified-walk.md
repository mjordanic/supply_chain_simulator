# 02 — Demand-pull topological scheduler + unified walk

Status: ready-for-agent

## Parent

`.scratch/lateral-links-scheduling/PRD.md` (ADR 0018)

## What to build

Replace the echelon-level cascade with a **demand-pull (reverse-topological) walk over the full
edge set**, so any legal topology — including lateral links — settles by supply dependency rather
than by shuffle.

- **Reverse-topo scheduler (pure deep module in graph.py).** A pure function builds the schedule
  structure from `(nodes, edges)`: in-degree over the **reversed** edge set so a node becomes
  "ready" only once all its downstream buyers are processed (sinks / graph-terminals start ready).
  The structure yields successive ready-sets of mutually-incomparable nodes. It is deterministic
  and unit-testable with no `Simulation`. The `allocation_rng` shuffle is a separate seam applied
  to each ready-set to produce the per-tick total order — injected, not embedded.
- **Unified walk (`_run_demand_pull_schedule` in runner.py).** One helper replaces the duplicated
  buyer loops in `tick()` and `tick_decide_and_settle()`. It walks ready-sets in demand-pull order
  (shuffling each via `allocation_rng` per ADR 0016), runs each node's observe→decide→`execute_buy`
  per line, then the factory `produce` step.
- **Cache topology once.** The schedule structure is computed in `build_world` (topology is static)
  and cached on `Simulation`; only the ready-set shuffle re-draws each tick, preserving the CRN draw
  cadence.
- **`level` demotion.** Remove the level-bucket scheduling and the `Simulation.levels` scheduling
  dict. `compute_levels` (longest-path) is retained **display-only** for `inspect.py`; `node.level`
  is still set but functionally inert.
- **RL seam unchanged.** The `tick_world` / `tick_decide_and_settle` action-injection boundary is
  preserved exactly.

This slice still reads the existing `prev_tick_sales` signal — the demand-signal swap to
`observed_sales` lands in issue 03. This is a **behavior change, not a pure refactor**: the old code
ran *push* (level-1 intermediates first, sinks last) despite ADR 0014's demand-pull prose, so
realised numbers and golden outputs will shift. ADR 0018 is authoritative; regenerate goldens.

## Acceptance criteria

- [ ] Pure scheduler unit tests: a strict chain yields singleton ready-sets (shuffle is a no-op); a lateral edge A→B forces A after B; sinks/graph-terminals are first and factories last; same topology + seed → identical order, different seed → different peer order.
- [ ] `tick()` and `tick_decide_and_settle()` both delegate to one `_run_demand_pull_schedule`; the duplicated cascade is gone.
- [ ] The schedule structure is computed once at `build_world` and cached; per-tick only the `allocation_rng` ready-set shuffle re-draws.
- [ ] A lateral graph settles by supply dependency (not by shuffle); a strict chain is unaffected in shape.
- [ ] `Simulation.levels` scheduling is removed; `compute_levels` remains and `inspect.py` still renders a depth hint.
- [ ] The RL three-node training chain remains valid and the `tick_world`/`tick_decide_and_settle` seam is unchanged.
- [ ] An end-to-end run on a lateral graph is bit-deterministic from the seed; goldens regenerated.

## Blocked by

- Issue 01 (type-based validation — needed to author/build the lateral graph the integration test runs on)
