# ADRs 0011–0016 (Proposed status)

Status: ready-for-agent

## Parent

`.scratch/multi-echelon/PRD.md`

## What to build

Author six new ADRs in `docs/adr/` capturing the locked-in design decisions of the multi-echelon rewrite. Each ADR lands with `Status: Proposed`. They will be ratified to `Accepted` in issue 14 after the implementation is complete.

The decisions themselves are already pinned (see the "Locked-in design decisions" section of the implementation plan); this slice just persists them as ADRs:

- ADR 0011 — Multi-echelon graph as the core sim model (supersedes single-`Store` semantics)
- ADR 0012 — Central table + sequential FCFS allocation contract
- ADR 0013 — Cash flow conservation across nodes
- ADR 0014 — Tick phasing as upward cascade by echelon level
- ADR 0015 — Demand-sinks as the demand source; `Market` shrinks to a multiplier engine (re-grounds ADRs 0001–0003 onto the sink path)
- ADR 0016 — RNG/CRN extension: the `allocation` sub-seed stream

Match the format and tone of existing ADRs in `docs/adr/`. Cross-link the ADRs to each other where decisions reference each other (e.g. 0012 → 0016 for the allocation_rng, 0013 → 0012 for payment timing).

## Acceptance criteria

- [ ] `docs/adr/0011-multi-echelon-graph.md` exists with `Status: Proposed` and reflects PRD's node hierarchy + graph decisions
- [ ] `docs/adr/0012-central-table-fcfs-allocation.md` exists with `Status: Proposed` and reflects the live central table + sequential FCFS contract
- [ ] `docs/adr/0013-cash-flow-conservation.md` exists with `Status: Proposed` and reflects the buyer−/seller+/sink-creates/factory-zero-margin/holding-and-fee-to-void model
- [ ] `docs/adr/0014-tick-phasing-cascade.md` exists with `Status: Proposed` and reflects the upward-by-echelon cascade phasing
- [ ] `docs/adr/0015-demand-sinks-market-multiplier.md` exists with `Status: Proposed` and re-grounds ADRs 0001–0003 onto the sink path
- [ ] `docs/adr/0016-allocation-rng-sub-seed.md` exists with `Status: Proposed` and reflects the new `allocation` sub-seed in `_SUB_SEED_PARAMS`
- [ ] ADRs cross-link where decisions reference each other
- [ ] Format matches existing ADRs in `docs/adr/`

## Blocked by

- `.scratch/multi-echelon/issues/03-node-hierarchy-scenario-extension.md`
