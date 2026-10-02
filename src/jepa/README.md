# `src/jepa/` — latent replenishment

Ordering-only planner for the replenishment study. It imports the simulator
and the tuner. It does not import the reinforcement-learning stack.

The published commands — the behaviour mixture at `--episodes-each 64`, the fit, the
150-trial `OrderUpToPolicy` studies at holding rates 0.005, 0.01, 0.02, 0.05, and 0.1, the
synthetic eval on the three generalisation protocols and the holding-rate sweep, and the M5 exam — are in the root
[Latent replenishment](../../README.md#latent-replenishment) section.

[`notebooks/09-jepa-train-and-plan.ipynb`](../../notebooks/09-jepa-train-and-plan.ipynb)
walks through a smoke log, a fit, and one `decide`.
[`notebooks/10-jepa-synthetic-eval.ipynb`](../../notebooks/10-jepa-synthetic-eval.ipynb)
reads `runs/jepa-replenishment/synthetic/` and plots the tables, the
rollout curve, and the scaling curves. It prints the eval command when
those tables are absent. Raw outputs stay under `runs/jepa-replenishment/`
and are not committed. A fit writes its loss curve and episode holdout under
`runs/jepa-replenishment/fit/`. The published fit uses embedding width 256 and
depth 4. `python -m src.jepa scale` writes the capacity and data-scaling curves
under `runs/jepa-replenishment/scaling/`; those files are not committed. Eval
writes a profit table only when held-out rollout accuracy beats an
ordinary-least-squares baseline on the same inputs.
