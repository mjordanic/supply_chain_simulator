# Notebook `08-monitor-rl-training`

Status: ready-for-agent

## Parent

`.scratch/notebooks-multi-echelon-rewrite/PRD.md`

## What to build

Rework the old `05-monitor_rl_training` into `08-monitor-rl-training`: read the graph-engine RL run's
TensorBoard event files from `runs/rl_ppo` and plot the learning, loss, and CRN-eval curves inline so
training can be monitored offline.

The experiment name is exposed as an editable variable at the top of the notebook (default
`rl_ppo`, for consistency with `09-rl-vs-baseline`), so a future longer training run drops in without
edits. TensorBoard scalar reading is env-agnostic; the notebook does not depend on `src/sim/inspect.py`.

## Acceptance criteria

- [ ] `notebooks/08-monitor-rl-training.ipynb` exists; old `05-monitor_rl_training` is removed
- [ ] Reads TensorBoard scalars from `runs/rl_ppo` and plots learning, loss, and CRN-eval curves inline
- [ ] The experiment name (default `rl_ppo`) and run path are editable top-of-notebook variables
- [ ] `%matplotlib inline`; committed with embedded executed outputs; modest DPI
- [ ] `jupyter nbconvert --to notebook --execute` runs the notebook with zero cell errors and a rendered figure for every plotting cell

## Blocked by

None — can start immediately
