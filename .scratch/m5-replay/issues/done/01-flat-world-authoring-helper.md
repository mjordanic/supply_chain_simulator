# 01: Flat-world authoring helper + identity gate

Status: done

## Parent

`.scratch/m5-replay/PRD.md` (see also ADR 0020 and `.scratch/m5-replay/STUDY.md`)

## What to build

A factory helper that authors a "flat" world: market, lifecycle, and freshness configured
so the full ADR 0015 demand multiplier chain (`market.demand_multiplier × stage_multiplier
× freshness`) is exactly 1.0 on every tick, for every product and region. This is the
off-switch that makes pure demand replay possible — replayed series pass through
unmodified — while leaving the chain live for what-if scenarios that want synthetic
effects on top.

The **gate**: a test must prove the chain reaches 1.0 bit-exact over a full run (not
approximately). If some component structurally cannot reach exactly 1.0, fall back to the
pre-agreed alternative in ADR 0020 — a node-level bypass flag on the replay sink — and
record which branch was taken (in the commit body and as a note on ADR 0020). Do not
invent a third design.

The helper is pure authoring convenience: it produces ordinary market/lifecycle parameter
objects (zero cycle amplitude, `Constant(0)` shocks and trend, disruptions off, inert
stage/freshness multipliers). No engine changes expected.

Test prior art: the existing market multiplier tests (per the PRD's Testing Decisions) —
the gate test follows their pattern, asserting multiplier values over a run.

## Acceptance criteria

- [ ] A one-call helper produces market + lifecycle params whose multiplier chain is identity
- [ ] Gate test: over a multi-region, multi-product, full-length run, every demand multiplier value equals exactly 1.0 (bit-exact, no tolerance)
- [ ] Flat world composes with an ordinary stochastic scenario: a sink's demand under the flat world equals its raw `demand_dist` draws
- [ ] If the gate cannot be satisfied, the ADR 0020 fallback (node bypass flag) is implemented instead and the choice is recorded
- [ ] Tests assert external behavior (multiplier values, sink demand), not internals

## Blocked by

None - can start immediately
