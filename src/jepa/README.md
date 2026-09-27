# `src/jepa/` — latent replenishment

Ordering-only planner for the replenishment study. It imports the simulator
and the tuner. It does not import the reinforcement-learning stack.

The published commands — 64 episodes of each behavior, the fit, the two
150-trial `OrderUpToPolicy` studies (`holding_rate` 0.01 and 0.05), the
synthetic eval on the tuner holdout, and the M5 exam — are in the root
[Latent replenishment](../../README.md#latent-replenishment) section.

[`notebooks/09-jepa-train-and-plan.ipynb`](../../notebooks/09-jepa-train-and-plan.ipynb)
walks through a smoke log, a fit, and one `decide`.
[`notebooks/10-jepa-synthetic-eval.ipynb`](../../notebooks/10-jepa-synthetic-eval.ipynb)
reads `runs/jepa-replenishment/synthetic/` and prints the eval command when
those tables are absent. Raw outputs stay under `runs/jepa-replenishment/`
and are not committed.
