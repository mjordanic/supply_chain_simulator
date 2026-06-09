# PRD: M5 real-demand replay mode

Status: ready-for-agent

Companion documents: `.scratch/m5-replay/STUDY.md` (feasibility study, decision log) and
ADR 0020 (replay-demand-sinks decision record). The study is the richer narrative; this
PRD is the implementation contract.

## Problem Statement

The simulator runs entirely on synthetic demand. Every conclusion drawn from it — which
textbook policy wins, what the tuner converges to, how multi-echelon topologies behave
under stress — is a conclusion about a synthetic world. There is no way to ask the most
basic credibility question: **could these policies have served a real demand stream, and
at what cost?** Without real data, the simulator's results can't be sanity-checked against
reality, and demonstrations of the framework lack a believable anchor.

## Solution

An optional **replay demand mode**: demand sinks that stream an observed per-tick demand
series instead of sampling a synthetic distribution, with the M5 dataset (Walmart daily
unit sales, 3,049 items × 10 stores × 1,941 days) as the first supported dataset.

The engine gains a generic, dataset-agnostic capability: a replay sink node type that is a
first-class citizen of the scenario format — authorable in a setup directory, composable
with any topology, any policy, and (optionally) every synthetic effect the simulator
already has (seasonality, disruptions, promotions, future price elasticity). Pure replay —
demand exactly as observed — is achieved by authoring a "flat" world where the multiplier
chain is identity.

All M5 specifics live in a separate adapter package that converts the raw Kaggle files
into a standard setup directory, derives catalog parameters (reference prices, categories,
regions) from the data, exposes a data-quality report, and preserves everything not yet
wired (calendar events, SNAP flags, launch dates) on disk for future use. Observed selling
prices can optionally be replayed through a wrapper policy. An example notebook
demonstrates the full path: raw files → quality report → multi-echelon scenario → run →
proof of exact replay.

Declared semantics, documented and accepted: **observed sales = true demand** (M5 records
sales, which are stockout-censored; the quality report flags suspected out-of-stock runs
so slice selection can avoid them).

## User Stories

1. As a simulation author, I want a replay demand sink node type, so that a sink can demand exactly an observed real-world series instead of a synthetic draw.
2. As a simulation author, I want replay sinks authorable in `setup.yaml` alongside ordinary nodes, so that replay scenarios remain complete, reloadable, diffable setup-dir artifacts (ADR 0017).
3. As a simulation author, I want the demand series stored in a tidy parquet file inside the setup directory, so that series data round-trips through `load_setup` like every other scenario input.
4. As a simulation author, I want a flat-world authoring helper that makes the demand multiplier chain exactly 1.0, so that I can show demand exactly as observed in the data.
5. As a simulation author, I want the multiplier chain (seasonality, regional shocks, disruptions, lifecycle, freshness) to still apply to replayed demand when I configure it, so that I can run what-if experiments (promo, pandemic, price elasticity) on top of real demand.
6. As a simulation author, I want build/load-time validation that every replay series covers `n_steps`, so that a too-short series fails fast instead of silently wrapping or padding.
7. As a researcher running CRN-paired comparisons, I want replay sinks to consume the same `world_rng` draws as stochastic sinks, so that mixing replay and stochastic sinks in one graph keeps all other streams bit-identical.
8. As a policy researcher, I want to evaluate any textbook policy (and tuned variants) against real M5 demand, so that I can check whether conclusions drawn on synthetic worlds hold on real data.
9. As a policy researcher, I want in-sim lost sales against replayed demand to be measurable through the existing rejection log, so that "could policy X have served this demand?" has a quantitative answer.
10. As a policy researcher, I want sink purchasing power guaranteed non-binding in adapter-generated scenarios, so that the sim's cash mechanics never silently censor the replayed demand I am evaluating against.
11. As a data analyst, I want an M5 adapter that converts the raw Kaggle files into a setup directory, so that going from download to runnable scenario is one function call.
12. As a data analyst, I want to select a slice (items × stores × date window) when building a scenario, so that I can scale from a handful of fast movers in one store to multiple stores in a state.
13. As a data analyst, I want a per-item data-quality report (price coverage, fill counts, zero-sale share, longest zero run, launch date), so that I can judge slice quality and avoid censoring-suspect items.
14. As a data analyst, I want weekly M5 prices expanded to daily series via the Walmart week calendar, so that downstream consumers never deal with `wm_yr_wk` numbering.
15. As a data analyst, I want price gaps forward-filled and leading gaps back-filled, with fill counts surfaced in the quality report, so that fabricated prices are visible rather than silent.
16. As a simulation author, I want catalog `base_price` derived from observed prices (median over the slice) and `category` / `region` derived from the M5 hierarchy, so that catalog parameters are grounded in data rather than invented.
17. As a simulation author, I want `unit_cost` defaulted to a documented fraction of `base_price` and overridable, so that the invented cost fiction is explicit and consistent.
18. As a simulation author, I want an opt-in price replay wrapper policy that overrides a shop's list prices with observed daily prices while delegating ordering to any inner policy, so that I can reproduce real revenue dynamics when I want them and price freely when I don't.
19. As a future maintainer, I want calendar events, SNAP flags, and launch dates preserved on disk in the emitted setup directory, so that a later calibration effort (fitting synthetic generators to M5) needs no re-ingestion.
20. As a notebook reader, I want an end-to-end example notebook (raw files → quality report → multi-echelon CA scenario → run → exact-replay assertion → event-overlay plots), so that the feature is demonstrated and self-verified.
21. As a new user, I want a README committed next to the M5 adapter explaining what it is, that I must download the raw files from Kaggle myself and where to place them, and how to run the adapter, so that I can get started without reading source code.
22. As a tuning user, I want replay scenarios to be ordinary fixed-`world_seed` scenarios, so that the Optuna tuner runs on them unchanged.
23. As a repo maintainer, I want the sim core to know nothing about M5 and the adapter to import only public sim interfaces, so that the sibling-consumer architecture (ADR 0010) is preserved.
24. As a repo maintainer, I want raw and generated M5 data git-ignored, so that the repository never redistributes Kaggle data of unclear licensing.

## Implementation Decisions

All decisions below were grilled and agreed in the session recorded in
`.scratch/m5-replay/STUDY.md`; rejected alternatives are documented in ADR 0020.

- **Scope is evaluation-only replay.** Calibrating synthetic generators to M5, bootstrap
  demand sampling, and RL *training* on real data are explicitly deferred (replay is one
  fixed trace — useless as a training distribution, fine for evaluation).
- **`ReplayDemandSinkNode`, a subclass of the demand sink node**, overrides only the
  demand-target computation: the stochastic `demand_dist` draw is replaced by
  `series[tick]`; the full ADR 0015 multiplier chain (market multiplier × lifecycle stage
  × freshness) still applies. Rejected alternatives: a cursor-based `Distribution`
  subclass (no tick in the `sample` contract; catalog loop advances cursors wrongly) and
  an optional series field on the base sink (half-state, hidden from the node roster).
- **Pure replay via flat authoring, not a node flag.** A factory helper authors a world
  whose multiplier chain is identity (zero cycle amplitude, `Constant(0)` shocks/trend,
  disruptions off, inert lifecycle/freshness). **Gate:** a test must prove the chain hits
  exactly 1.0 bit-exact; if any component cannot, fall back to a node-level bypass flag
  (the documented fallback, not the default).
- **CRN draws burned, not skipped.** The replay override consumes exactly as many
  `world_rng` draws as the base class's one-draw-per-catalog-product-per-tick loop
  (ADR 0003/0015) and discards them, so mixed replay/stochastic graphs stay CRN-clean.
- **Series contract:** length ≥ `n_steps` validated at build/load; no wraparound, no
  padding; rounding of `series × multiplier` follows existing base-class behavior.
- **Setup-dir native serialization (extends ADR 0017).** `setup.yaml` grows a
  `sink_replay` node type whose spec references a `series_id`; `load_setup` resolves it
  from a tidy demand parquet (`series_id, tick, qty`) in the same directory and constructs
  replay sinks. `start_date` from the M5 calendar; tick 0 = window start; `n_steps` =
  window length. One day = one tick.
- **Price replay is policy turf.** `PriceReplayPolicy(inner, prices)` wraps any
  intermediate policy: delegates `decide()`, overwrites the `list_price` portion of the
  decision from per-tick daily arrays. Opt-in by construction. Policies are attached in
  Python, so no YAML schema change; the price parquet rides along in the setup dir.
- **Catalog derivation:** `base_price` = median observed price over the slice (catalog
  reference is per-item, not per-store); `unit_cost` = documented default fraction of
  base_price, author-overridable; M5 state → node `region`; M5 dept → `Ware.category`;
  `Ware.seasonality` left default (dead under a flat market).
- **Sink economics:** adapter sets sink `income_rate` so the affordability cap never
  binds; the example notebook asserts zero `insufficient_cash` rejections at sinks.
- **M5 adapter is a sibling package** (same pattern as the tuning/RL/LLM layers,
  ADR 0010): raw-file parsing, slice selection, weekly→daily price expansion via the
  calendar's `wm_yr_wk` mapping, ffill then leading bfill for price gaps, per-item quality
  report, setup-dir emission. The sim core imports nothing from it.
- **README requirement:** a committed README ships next to the adapter in the datasets
  package (not inside the data folder — that folder is git-ignored, so a README there
  would be untracked), explaining the adapter's purpose, that raw files must be downloaded
  from Kaggle by the user (files are git-ignored; redistribution rights unclear), where to
  place them, and how to invoke the adapter and example notebook. Written as part of the
  adapter issue.
- **Preserved but not wired:** calendar events and SNAP flags are *not* mapped to the
  event engine (their effect is already embedded in the series; mapping would
  double-count); launch dates are *not* mapped to lifecycle stage. Both are emitted to
  disk (calendar parquet, quality report) for a future calibration effort.
- **Deferred refactor:** a first-class `DemandSource` protocol plus per-product RNG
  streams is the documented future path (revisit when committing to calibrated/bootstrap
  demand for RL training); the subclass design collapses into it mechanically.

## Testing Decisions

Good tests here assert **external behavior**: what demand a sink emits, what a loaded
scenario contains, what the adapter writes — never private attributes or draw-by-draw RNG
internals (CRN cleanliness is asserted by comparing *outcomes* of paired runs, the
established pattern in the existing CRN/runner tests).

All five code modules get dedicated unit tests (user's explicit choice):

1. **Replay sink:** under a flat world, per-tick consumption equals the series exactly;
   a graph mixing one replay sink with stochastic sinks produces bit-identical stochastic
   trajectories to the same graph with the replay sink swapped for any other sink
   (CRN-paired run comparison); too-short series raises at build time.
2. **Flat authoring helper:** the gate test — multiplier chain ≡ 1.0 bit-exact over a
   full run, across regions and ticks.
3. **Setup IO extension:** write → `load_setup` → run roundtrip for a replay scenario;
   prior art: the existing setup-example end-to-end test.
4. **Price replay wrapper:** delegation (inner policy's order decisions pass through
   untouched) and override (posted offers carry the observed price for the tick).
5. **M5 adapter:** small hand-built fixture files (not real M5 data) covering weekly→daily
   expansion, ffill/bfill behavior and counts, quality-report fields, slice filtering,
   and emitted setup-dir validity (loadable by `load_setup`).

Prior art to follow: existing scenario/serialization tests, market multiplier tests,
runner CRN tests, and the setup-dir e2e test. The example notebook self-verifies with
inline assertions (exact replay, zero sink cash rejections) but is not part of the test
suite.

## Out of Scope

- Calibrating synthetic generator parameters (demand distributions, seasonality, event
  frequencies) to M5.
- Bootstrap or otherwise resampled demand sources; the `DemandSource` protocol refactor;
  per-product RNG streams.
- RL training on replayed data (evaluation against replay scenarios is fine and needs no
  new work).
- Mapping calendar events / SNAP to the event engine; mapping launch dates to lifecycle.
- Price-elastic demand and store competition (separate effort; this PRD only ships the
  `base_price`-from-data and price-replay building blocks it will reuse).
- Stockout de-censoring of M5 sales (we declare sales = demand; the quality report's
  zero-run flags are the only mitigation).
- Any upstream topology generation beyond the single example notebook (topology is the
  scenario author's job).
- Committing M5 data to the repository.

## Further Notes

- Dataset facts and the censoring discussion live in the study; keep the "observed sales
  = true demand" declaration visible in user-facing docs (README, notebook prose).
- The flat-chain gate is the one genuine technical risk; the fallback (node bypass flag)
  is pre-agreed in ADR 0020, so a gate failure changes one issue's shape, not the design.
- Issue sequencing hint: replay sink + flat helper first (engine core), then setup IO,
  then adapter + README, then price wrapper, then notebook last (it consumes everything).
