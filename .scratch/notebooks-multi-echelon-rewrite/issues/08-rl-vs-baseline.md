# Notebook `09-rl-vs-baseline`

Status: ready-for-agent

## Parent

`.scratch/notebooks-multi-echelon-rewrite/PRD.md`

## What to build

Merge the old, stale `06-compare_rl_vs_baseline` and `07-rl_vs_baseline_per_product` into one
`09-rl-vs-baseline` on the graph engine. The old `06` crashes: it calls `observation_dim(config)`
(the signature is now `observation_dim(K_active: int)`) and points at the `fashion_run` checkpoint,
whose 74-dim input layer predates the central-table observation block (the current Actor expects
94 dims), so it cannot even load.

The new notebook loads the graph-engine-compatible `runs/rl_ppo` checkpoint
(`actor_step0000005000.pt`, 94-dim input) and runs the CRN-paired evaluation harness
(`src.rl.eval.evaluate`) against `OrderUpToPolicy`, showing the comparison at two resolutions in one
place:
- an **aggregate KPI / per-seed** uplift section, and
- a **per-product single-seed** overlay.

Observation/action sizing uses `observation_dim(K_active)` / `action_dim(K_active)` (int argument).
The experiment name and checkpoint path are top-of-notebook variables. The notebook states plainly
that the shipped checkpoint is **smoke-trained** (so uplift is ~0 or negative) rather than implying a
tuned win. `runs/fashion_run` (74-dim, pre-refactor) is **not** used for evaluation.

## Acceptance criteria

- [ ] `notebooks/09-rl-vs-baseline.ipynb` exists; old `06-compare_rl_vs_baseline` and `07-rl_vs_baseline_per_product` are removed
- [ ] Loads the `runs/rl_ppo` checkpoint (`actor_step0000005000.pt`, 94-dim) and evaluates via `src.rl.eval.evaluate`, CRN-paired against `OrderUpToPolicy`
- [ ] Observation/action sizing uses `observation_dim(K_active)` / `action_dim(K_active)` (int argument)
- [ ] Shows both an aggregate KPI/per-seed uplift section and a per-product single-seed overlay
- [ ] Experiment name and checkpoint path are editable top-of-notebook variables
- [ ] Notebook text states plainly that the checkpoint is smoke-trained (uplift ~0 or negative), not a tuned win
- [ ] `%matplotlib inline`; committed with embedded executed outputs; modest DPI
- [ ] `jupyter nbconvert --to notebook --execute` runs the notebook with zero cell errors and a rendered figure for every plotting cell

## Blocked by

None — can start immediately
