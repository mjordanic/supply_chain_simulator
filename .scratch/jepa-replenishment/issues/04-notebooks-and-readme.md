# 04: Training notebook, results notebook, and README commands

**What to build:** Two interactive notebooks and a root README section that tell a reader how to collect the log, fit the planner, run the two `OrderUpToPolicy` tuning studies, and read the synthetic comparison. The training notebook shows one `decide` becoming an order. The results notebook states the success rules beside the tables, and when the run artifacts are missing it prints the command that fills them instead of plotting empty numbers.

**Blocked by:** 03 — Synthetic comparison tables.

**Status:** ready-for-agent

## Parent

`.scratch/jepa-replenishment/PRD.md`

## Stories

PRD stories 54, 55, 58, 59, 68.

## Prior art

The tuning analysis notebook, which loads an on-disk study and tells the reader the command when the artifact is absent. The M5 example notebook's skip pattern. Ticket 01's collector and ticket 03's comparison entry point.

## Artifact homes

- `notebooks/09-jepa-train-and-plan.ipynb` (tracked).
- `notebooks/10-jepa-synthetic-eval.ipynb` (tracked).
- A section in `README.md` (tracked) with the exact commands.
- A short package README in `src/jepa/` (tracked) pointing at that section.
- Notebooks read `runs/jepa-replenishment/` and do not commit it.

## Acceptance criteria

- [ ] The training notebook, run top to bottom on the smoke settings, builds a small log, fits, and shows one decision's covers, ration, and order.
- [ ] The results notebook states the in-distribution rule and the retarget rule next to the tables when `runs/jepa-replenishment/synthetic/` is present.
- [ ] When that directory is absent, the results notebook prints the published command and does not fail.
- [ ] The root README gives the exact commands for the published log (64 episodes of each behavior), the fit, the two 150-trial `OrderUpToPolicy` studies (`holding_rate` 0.01 and 0.05), and the synthetic eval on the tuner holdout.
- [ ] The README states that raw outputs stay under `runs/jepa-replenishment/` and are not committed.
