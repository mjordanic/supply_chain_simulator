# RL notebooks update: train/eval and trained-agent analysis on the variable-K stack

Status: ready-for-agent

## Parent

`.scratch/variable-k-rl/PRD.md`

## What to build

Rework the two RL notebooks so they reflect and illustrate the new variable-K codebase, and
execute them end-to-end so the committed outputs show the new stack in action.

- **`notebooks/06-rl-train-and-eval.ipynb`**: drive the new train loop with a tiny smoke config
  (small number of updates / short episodes — minutes, not hours), with a markdown note pointing
  at the CLI for real training runs. Illustrate the new concepts explicitly: K sampled per
  episode (show two resets with different K), the `(K_max, F)` observation with its mask, the
  Arbiter config switch (proportional default, greedy available) and cash-budget fraction, the
  self-describing checkpoint (show the bundle keys and the layout version), and a CRN-paired
  eval against `OrderUpToPolicy` on the reduced tuple.
- **`notebooks/06a-analyze-a-trained-agent.ipynb`**: load a checkpoint through the checkpoint
  I/O module, run inference at more than one K, and adapt the existing analyses to the
  per-product row structure — including a small demonstration of implicit assortment (a product
  whose order-up-to target sits at zero is being "stopped") and of permutation invariance
  (reordering product rows reorders decisions correspondingly).

Both notebooks must run top-to-bottom with `uv run jupyter nbconvert --execute` (or equivalent)
against the merged stack; commit them with outputs as the repo's other notebooks are. Update any
stale prose cells (old layout numbers, slot-shuffle mentions, bare state-dict loading) — the
notebooks are illustrations, so prose accuracy matters as much as code.

No library code changes; if a notebook reveals an API gap, report it rather than patching
around it in hidden cells.

## Acceptance criteria

- [ ] Both notebooks execute top-to-bottom without errors on the new stack.
- [ ] Train/eval notebook demonstrates: variable K across resets, arbiter mode switch, self-describing checkpoint contents, CRN-paired uplift cell.
- [ ] Analysis notebook demonstrates: version-validated checkpoint load, inference at two different K values, implicit assortment (order-up-to ≈ 0 ⇒ stopped product), permutation-invariance check.
- [ ] No prose cell still describes slot-shuffle, fixed `K_active`, the 18-slot row, or bare state-dict checkpoints as current behaviour.
- [ ] Smoke-config runtime stays in the minutes range; CLI pointer for real runs present.
- [ ] Committed with executed outputs, consistent with the other numbered notebooks.

Prior art: the existing 06/06a notebooks for structure and tone; `05a-analyze-a-study.ipynb`
for the analysis-notebook pattern.

## Blocked by

- `06-train-loop-rewire.md`
- `07-eval-rewire.md`
