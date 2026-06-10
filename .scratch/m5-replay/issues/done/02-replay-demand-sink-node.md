# 02: ReplayDemandSinkNode — Python-authored replay end-to-end

Status: done

## Parent

`.scratch/m5-replay/PRD.md` (see also ADR 0020 and `.scratch/m5-replay/STUDY.md`)

## What to build

The tracer bullet: a `ReplayDemandSinkNode` subclass of the demand sink node whose
demand target replaces the stochastic `demand_dist` draw with `series[tick]`, keeping
the full ADR 0015 multiplier chain applied on top. Authoring is Python-only in this slice
(setup-dir serialization is issue 03). After this slice, a scenario author can wire a
hardcoded series into a sink, run a graph scenario, and watch the sink demand exactly
that series under a flat world (issue 01).

Contract details (all decided in ADR 0020):

- **CRN draws burned, not skipped**: the override consumes exactly as many `world_rng`
  draws as the base class's one-draw-per-catalog-product-per-tick loop, then discards
  them, so mixed replay/stochastic graphs keep all other streams bit-identical.
- **Fail fast**: series length ≥ `n_steps` validated at build time; no wraparound, no
  padding.
- **Rounding** of `series[tick] × multiplier` follows the existing base-class behavior.
- The subclass is visible as its own node type in the scenario's node roster inspection.

## Acceptance criteria

- [ ] Under a flat world (issue 01), per-tick sink consumption equals the replay series exactly, end to end through a real `Runner.run()`
- [ ] With a non-flat market, replayed demand scales by the same multiplier chain as a stochastic sink would
- [ ] CRN-paired test: a graph mixing one replay sink with stochastic sinks yields bit-identical trajectories for the stochastic parts versus the same graph with the replay sink replaced by a stochastic sink (same seeds)
- [ ] A series shorter than `n_steps` raises at build time with a clear error
- [ ] In-sim lost sales against replayed demand appear in the existing rejection log (`unmet_demand`)
- [ ] Tests follow existing CRN/runner test prior art and assert outcomes, not RNG internals

## Blocked by

- `01-flat-world-authoring-helper.md`
