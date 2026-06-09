# M5 real-demand replay — feasibility study

Status: study complete, design agreed (grilling session 2026-06-09). Implementation not started.
Companion decision record: ADR 0020.

## Goal and scope

Add an **optional real-data demand mode** to the simulator: sinks replay observed demand
series instead of sampling synthetic distributions. First dataset: **M5** (Walmart daily
unit sales, Kaggle).

Scope decision (Q1): **evaluation-only replay**. The deliverable answers "could policy X
have served this real demand stream, at what cost?" for textbook policies and the Optuna
tuner. **Out of scope:** calibrating the synthetic generator to M5, bootstrap demand
sampling, and RL *training* on real data — replay is a single fixed trace, useless as a
training distribution. Those belong to a future "calibration" effort (scenario (b) below).

We build a **framework feature**, not one scenario: the engine gains a generic
"replay a per-tick demand series" capability that knows nothing about M5; all M5
specifics live in a sibling adapter. Upstream topology (DCs, factories, lead times,
costs) is the scenario author's job, exactly as today. One example notebook demonstrates
a multi-echelon setup.

## The dataset

M5: Walmart, 2011-01-29 → 2016-06-19 (1,941 days), 3,049 items × 10 stores, 3 states
(CA/TX/WI), hierarchy state → store → category (3) → dept (7) → item.

- `sales_train_*.csv` — daily unit sales per (item, store). **One day = one tick**: exact
  granularity match with the engine.
- `sell_prices.csv` — weekly selling price per (store, item); promos visible as dips.
  Items have no price rows before launch.
- `calendar.csv` — date ↔ day index, weekday, named events (typed), per-state SNAP flags,
  and the `wm_yr_wk` mapping needed to expand weekly prices to daily.

**What M5 lacks** (and the sim needs): unit costs, margins, inventory levels, lead times,
suppliers/topology — all invented by the scenario author.

**Censoring caveat.** M5 records *sales*, not demand; zeros can mean no demand or empty
shelf, indistinguishably. Most zeros are genuine (median item ~1 unit/day; M5 is *the*
intermittent-demand benchmark), stockout censoring is second-order. **Declared semantics:
observed sales = true demand.** Mitigation: the adapter's quality report flags long zero
runs (suspected OOS) so slices can avoid them; in-sim lost sales then measure the
*evaluated policy's* shortfall against that declared demand.

## Agreed design (the "simplest solution that fits" — recommended)

### Entity mapping

| M5 | Simulator |
|---|---|
| (item, store) sales series | one `ReplayDemandSinkNode`, single supplier edge to its store |
| store (`CA_1`, …) | `IntermediateNode` shop — the node whose policy is evaluated |
| item | `Ware` in catalog |
| state (`CA`/`TX`/`WI`) | `node.region` |
| dept (`FOODS_3`, …) | `Ware.category` |
| weekly sell price | `prices.parquet` + optional `PriceReplayPolicy` |
| calendar events / SNAP | preserved on disk (`calendar.parquet`), **not** wired to the engine |
| upstream (DC, factory, lead times, costs) | invented by the author / example notebook |

### Engine: `ReplayDemandSinkNode(DemandSinkNode)`

Subclass overriding `demand_target()` (alternatives rejected in ADR 0020):

1. **Series replaces the RNG draw, nothing else changes.**
   `demand_target = series[tick] × market.demand_multiplier × stage_multiplier ×
   freshness` — the full ADR 0015 multiplier chain still applies. Synthetic what-ifs
   (promo, disruption, future elasticity) compose on top of replayed demand with zero
   replay-specific casework.
2. **Pure replay = flat authoring, not a node flag.** A `flat` market/lifecycle authoring
   helper makes the whole multiplier chain ≡ 1.0 (cycle_amp 0, shocks/trend Constant(0),
   disruptions off, stage/freshness multipliers 1). **Implementation gate:** verify the
   chain reaches exactly 1.0 bit-exact; if some component can't, fall back to a node-level
   bypass flag.
3. **CRN preserved in mixed graphs.** The override burns exactly the same number of
   `world_rng` draws as the base catalog loop (ADR 0003/0015), then discards them, so
   stochastic sinks elsewhere in the graph stay bit-identical.
4. **Fail fast:** `len(series) >= n_steps` validated at build/load time. No wraparound,
   no padding. Rounding of `series × multiplier` follows the base class behavior.

### Prices: derived base + optional replay

- `Ware.base_price` := median observed price for the item over the slice (catalog-wide
  reference; `base_price` is not per-store).
- `unit_cost` is invented: documented default fraction of base_price (e.g. 0.7×),
  author-overridable. Consistent fiction, clearly labeled.
- **`PriceReplayPolicy(inner: IntermediatePolicy, prices)`** — wrapper that delegates
  `decide()` to any ordering policy, then overwrites the `list_price` part with the
  observed daily price. Opt-in: unwrapped policies price freely. This is also the
  `base = observed` machinery the future elasticity/replay-pricing work needs.
- Weekly → daily expansion happens in the adapter (via `wm_yr_wk`); the wrapper consumes
  plain per-tick arrays. Missing prices: forward-fill gaps, back-fill leading NaNs
  (harmless: pre-launch demand is zero); fill counts surface in the quality report.

### Sink economics

The affordability cap (`sink.cash / list_price`) must never bind, or the sim silently
censors replayed demand. The adapter sets sink `income_rate` high enough by construction;
the example notebook asserts zero `insufficient_cash` rejections at sinks (rejection log).

### Adapter: `src/datasets/m5.py` + notebook

Sibling package (ADR 0010 pattern — core imports nothing from it). Responsibilities:
Kaggle-file parsing, slice selection (items × stores × date window), weekly→daily price
expansion, ffill/bfill, `quality_report(slice)` (per item: price coverage %, fill counts,
zero-day share, longest zero run / suspected-OOS flag, launch date), and setup-dir
emission. Raw Kaggle files in `data/m5/raw/`, git-ignored (user downloads; redistribution
rights unclear); emitted artifacts also git-ignored, deterministic from raw.

`notebooks/0X_m5_replay.ipynb`: download pointer → adapter run → quality report → build
the multi-echelon example (invented CA DC → `CA_1..CA_4` shops → replay sinks) → run →
assert exact replay (consumption == series; zero sink cash rejections) → event-overlay
plots from `calendar.parquet`.

### On-disk format: setup-dir native (ADR 0017 extended)

Adapter emits one setup dir: `catalog.csv + setup.yaml + demand.parquet (+ prices.parquet
+ calendar.parquet)`. `setup.yaml` grows a `sink_replay` node type whose spec references a
`series_id`; `load_setup()` resolves it from the tidy `demand.parquet`
(`series_id, tick, qty`) and builds `ReplayDemandSinkNode`s. Replay scenarios are thus
complete, reloadable, diffable artifacts — the same interface future users author their
own topologies in. `start_date` comes from `calendar.csv`; tick 0 = window start;
`n_steps` = window length. Price replay stays Python-side (policies are attached in
Python, not YAML); `prices.parquet` just rides along in the dir.

### Preserved but not wired (v1)

- **Events/SNAP → EventEngine**: no — their effect is already embedded in the series;
  mapping them would double-count. Kept on disk for future calibration.
- **Launch dates → lifecycle `init_stage`**: no — lifecycle is inert under flat authoring.
  Recorded in the quality report.
- **`Ware.seasonality`**: default — dead field under flat market.

## Alternative considered: `DemandSource` refactor (the "major refactor" path)

Replace the sink's `demand_dist` field with a protocol — `demand(tick, world_rng) -> int`
— implemented by `StochasticDemand(dist)`, `ReplayDemand(series)`, later
`BootstrapDemand(blocks)`. Pairs with the per-product RNG streams migration already on
TODO (seed per-product streams from `(world_seed, product_id)`), which retires the
ADR 0003 catalog-loop draw-burning entirely.

**Not recommended now.** Its payoff arrives only with the calibration/bootstrap effort
(RL-grade stochastic-but-realistic demand), which is explicitly deferred. With one real
implementation it is speculative generality. The incremental design is forward-compatible:
`ReplayDemandSinkNode` overrides one method and collapses into `ReplayDemand` mechanically
when the refactor lands; only the draw-burning loop is thrown away, and that dies in the
refactor anyway. **Trigger to revisit:** committing to scenario (b) — calibrated/bootstrap
demand for RL training.

## Compatibility notes

- **Tuner:** fully compatible — trials already run fixed `world_seed` scenarios; a
  deterministic replay scenario is just a scenario. (Caution: tuning *on* one fixed trace
  is in-sample fitting; use time-split windows for honest eval.)
- **RL:** evaluation against replay scenarios is fine; training on them is not (one
  episode forever). Out of scope per Q1.

## Implementation sketch

1. `ReplayDemandSinkNode` in `src/sim/node.py` + CRN draw-burning + length validation
   (unit tests: exact replay under flat authoring; mixed-graph CRN bit-identity).
2. Flat-authoring helper + the multiplier-chain ≡ 1.0 gate test.
3. `sink_replay` node type in `setup_io.py` + `demand.parquet` reader (roundtrip test).
4. `PriceReplayPolicy` wrapper in `src/sim/policy.py` (delegation test).
5. `src/datasets/m5.py` adapter + quality report (unit tests on small fixture slices).
6. `notebooks/0X_m5_replay.ipynb` multi-echelon example with exactness assertions.
