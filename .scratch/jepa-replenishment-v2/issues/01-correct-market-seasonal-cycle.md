# 01: Correct the Market seasonal cycle and re-establish CRN baselines

**What to build:** A world whose regional demand responds to its own seed instead
of following a fixed script. Today `Market` adds the seasonal cycle to the demand
*level* each tick, so the level integrates the sine wave and slams into its
clamps: demand sits at the ceiling for roughly ticks 10–190, collapses between
190 and 207, and sits at the floor for the remaining 158 ticks. Across 128
episodes the collapse tick has median 207 and only 13 distinct values. Fifty-three
percent of every episode is a dead world and the world seed barely matters.

After this ticket the demand level is a purely stochastic clamped process and the
seasonal cycle is a read-time multiplier on top of it, so `cycle_amp` means a
bounded seasonal swing, different world seeds produce materially different demand
paths, and the level no longer parks against a clamp.

This is a wide change: it alters CRN identity for a fixed `world_seed`, so every
stored baseline, golden value and tuned study value that depends on it moves. The
ticket therefore includes sweeping the suite and re-establishing those baselines,
because the correction and the re-baselining cannot land green separately.
Re-baselining only — reinforcement-learning policies are not retrained and RL
results are not re-published.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

**Prior art:** existing `Market` unit tests; the `Market` and `CRN-paired eval`
entries in `CONTEXT.md`; ADR 0003 (CRN demand for all products) and ADR 0015
(demand sinks market multiplier) for the surrounding decisions.

**Artifact homes:** the new ADR goes in `docs/adr/` (tracked). The blast-radius
report goes in the ticket's completion notes and in the ADR's consequences
section, not in an untracked run directory.

- [ ] The regional demand level evolves as level × trend + shock, clamped to
      `[min_value, max_value]`, with no cycle term in the update.
- [ ] The emitted demand factor is the level multiplied by
      `1 + cycle_amp * sin(2π * step / cycle_len)`, applied at read time.
- [ ] The seasonal multiplier is not stored into the region's market state, so
      `EventEngine` disruption shocks continue to land on the level.
- [ ] `MarketParams` rejects `cycle_amp >= 1.0` with a clear error; the emitted
      demand factor is strictly positive for every admissible parameterisation.
- [ ] Over a long run the level spends no extended stretch pinned at either
      clamp, and aggregate demand does not collapse to a floor partway through an
      episode.
- [ ] `demand_shock` dispersion and the `trend` distribution are widened so that
      two different world seeds produce materially different demand paths;
      "materially different" is asserted by a concrete divergence measure, not
      by eyeball.
- [ ] An ADR records the defect, the correction, the `cycle_amp < 1` invariant,
      and that existing CRN baselines are superseded.
- [ ] The full test suite is swept; every CRN-dependent baseline outside
      `src/jepa/` that moved is re-established without retraining RL policies.
- [ ] The blast radius is reported: which suites moved and by how much.
- [ ] Any notebook, README section or `CONTEXT.md` text invalidated by this
      change is updated in this ticket.
