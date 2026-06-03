# Notebook `03-run-and-inspect-simulation`

Status: ready-for-agent

## Parent

`.scratch/notebooks-multi-echelon-rewrite/PRD.md`

## What to build

Replace the stale old-`00-check_simulated_data` notebook (which crashes on graph-engine data — it
reads the removed `run_log['stores']` key) with `03-run-and-inspect-simulation`: a live run of
`llm_world_20` whose figures reconstruct per-tier cash, inventory value, equity composition, and
P&L across **every** echelon tier (factory, warehouse/DC, shop, sink), not just the shop tier.

The notebook runs the simulation live (`Runner(scenario).run()`, sub-2-second run) and reads the
in-memory `run_log` — it does **not** read pre-exported parquet (the `timeseries.parquet` schema is
shop-only and unchanged). All per-tier/equity parsing goes through `src/sim/inspect.py` so the
notebook stays thin presentation over the shared, tested parser.

## Acceptance criteria

- [ ] `notebooks/03-run-and-inspect-simulation.ipynb` runs `llm_world_20` live and reads the in-memory `run_log` (no parquet reads for time-series)
- [ ] Per-tier cash, inventory value, equity composition, and P&L are shown for **all** tiers (factory, warehouse/DC, shop, sink), derived via `src/sim/inspect.py`
- [ ] `%matplotlib inline`; committed with embedded executed outputs; modest DPI
- [ ] `jupyter nbconvert --to notebook --execute` runs the notebook with zero cell errors and a rendered figure for every plotting cell

## Blocked by

- `.scratch/notebooks-multi-echelon-rewrite/issues/01-inspect-deep-module.md`
