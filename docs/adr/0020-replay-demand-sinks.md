# Replay demand sinks: subclass override, full multiplier chain, burned CRN draws

Status: Accepted (implemented 2026-06 — `src/sim/replay_demand_sink.py`, `src/sim/flat_world.py`, `src/datasets/m5.py`)

The simulator gains an optional real-data demand mode: a sink replays an observed per-tick
demand series (first dataset: M5 Walmart daily unit sales; declared semantics: observed
sales = true demand) instead of sampling `demand_dist`. Scope is evaluation-only — "could
policy X have served this real demand stream?" — not calibration or RL training.

**Decision — `ReplayDemandSinkNode(DemandSinkNode)` overrides `demand_target()` to replace
the stochastic draw with `series[tick]`, keeping everything else identical.**

**Full multiplier chain still applies.** `demand_target = series[tick] ×
market.demand_multiplier(pid, region, tick) × stage_multiplier × freshness` — the ADR 0015
composition with only the `demand_dist.sample(world_rng)` term swapped out. Real data
already embeds seasonality, events, and price response, so *pure* replay is achieved by
authoring the multiplier chain to 1.0 (flat market: `cycle_amp=0`, shocks/trend
`Constant(0)`, disruptions off; inert lifecycle/freshness), helped by a `flat` authoring
factory. Keeping the chain live means synthetic what-ifs — promo, disruption, future price
elasticity — compose multiplicatively on replayed demand with zero replay-specific
casework. Implementation gate: the flat chain must reach exactly 1.0 bit-exact; if some
component cannot, fall back to a node-level bypass flag.

**CRN draws are burned, not skipped.** The override consumes exactly as many `world_rng`
draws as the base class's one-draw-per-catalog-product-per-tick loop (ADR 0003/0015) and
discards them. A graph mixing replay sinks and stochastic sinks therefore keeps every
other stream bit-identical — paired comparison stays valid. Pure-replay graphs pay a
trivial wasted-draw cost. When the per-product RNG streams migration (ADR 0003's named
escape hatch) lands, the burning loop is deleted with it.

**Series contract.** `len(series) >= n_steps` validated at build/load time; no wraparound,
no padding. Rounding of `series × multiplier` follows base-class behavior. On disk,
replay scenarios stay setup-dir native (extends ADR 0017): `setup.yaml` grows a
`sink_replay` node type referencing a `series_id` resolved from a tidy `demand.parquet`
(`series_id, tick, qty`) in the same dir.

**Pricing rides policy turf, not the engine.** Observed prices are optionally replayed via
`PriceReplayPolicy(inner: IntermediatePolicy, prices)` — a wrapper that delegates
`decide()` to any ordering policy and overwrites the `list_price` part of the decision.
Opt-in by construction; no engine change.

Considered alternatives:

(a) **`TimeSeries` `Distribution` subclass with an internal cursor.** Rejected:
`Distribution.sample(rng)` has no `tick` argument, so the cursor is hidden mutable state,
and the ADR 0003 catalog loop samples distributions for *every* product each tick, so the
cursor would advance at the wrong times. Structural misfit with the Distribution contract.

(b) **Optional `demand_series` field on `DemandSinkNode`.** Rejected: leaves the node in a
half-state (`demand_dist` required but ignored, two demand semantics in one class) and
hides the replay/stochastic distinction from the node roster, where a subclass makes it
visible in `nodes_df()`.

(c) **Node-flag bypass of the multiplier chain (`apply_multiplier=False`).** Rejected as
the default in favour of flat authoring: a flag forks `demand_target` semantics inside one
class and offers per-sink granularity nobody asked for. Retained only as the fallback if
the flat-chain ≡ 1.0 gate fails.

(d) **First-class `DemandSource` protocol** (`StochasticDemand` / `ReplayDemand` /
`BootstrapDemand`) **plus per-product RNG streams.** Deferred, not rejected: its payoff
arrives with calibrated/bootstrap demand for RL training, which is out of scope here. The
subclass collapses into `ReplayDemand` mechanically when that refactor lands; revisit when
committing to that effort.

## Cross-references

- ADR 0003 — CRN demand for all products (draw-burning preserves its contract; per-product
  streams are its named replacement)
- ADR 0010 — sim as base for ML layers (the M5 adapter lives in a sibling package,
  `src/datasets/`, core imports nothing from it)
- ADR 0015 — demand-sinks own demand sampling (the multiplier composition this ADR keeps)
- ADR 0017 — setup files as deterministic input (extended with `sink_replay` +
  `demand.parquet`)
