# 03: Setup-dir serialization of replay scenarios

Status: ready-for-agent

## Parent

`.scratch/m5-replay/PRD.md` (see also ADR 0020 and `.scratch/m5-replay/STUDY.md`)

## What to build

Make replay scenarios first-class setup-dir artifacts (extends ADR 0017). `setup.yaml`
grows a `sink_replay` node type whose spec references a `series_id`; the demand series
live in a tidy parquet file (`series_id, tick, qty`) inside the same setup directory.
`load_setup` resolves the reference and constructs `ReplayDemandSinkNode`s; a writer
emits the parquet + YAML so a replay scenario round-trips write → load → run.

A replay scenario is then a complete, reloadable, diffable directory exactly like any
synthetic scenario — the interface future users author their own topologies in, and the
target format the M5 adapter (issue 06) emits. It must remain an ordinary fixed-
`world_seed` scenario with no special-cased loading or run path, so downstream consumers
(notably the Optuna tuner) work on it unchanged — PRD user story 22 is satisfied by this
property, not by separate work.

Validation: a `sink_replay` node whose `series_id` is missing from the parquet, or whose
series is shorter than `n_steps`, fails at load time with a clear error.

## Acceptance criteria

- [ ] `setup.yaml` supports a `sink_replay` node type referencing a `series_id`
- [ ] Demand series persist in a tidy parquet (`series_id, tick, qty`) inside the setup dir
- [ ] Roundtrip test: write a replay scenario dir → `load_setup` → `Runner.run()` → exact replay under a flat world (prior art: the existing setup-dir end-to-end test)
- [ ] Missing `series_id` or too-short series fails at load with a clear error
- [ ] Existing synthetic setup dirs load unchanged (backward compatible)

## Blocked by

- `02-replay-demand-sink-node.md`
