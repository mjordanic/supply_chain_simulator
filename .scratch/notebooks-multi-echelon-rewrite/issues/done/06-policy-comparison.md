# Notebook `06-policy-comparison`

Status: ready-for-agent

## Parent

`.scratch/notebooks-multi-echelon-rewrite/PRD.md`

## What to build

A `06-policy-comparison` notebook (built from the `example_paired_comparison` scenario) that compares
the four textbook reorder policies (ADR 0006) head-to-head on one shared world using Common Random
Numbers, so any difference is the policy and not seed luck:
`OrderUpToPolicy`, `ReorderPointPolicy`, `PeriodicOrderUpToPolicy`, `PeriodicReorderPolicy`.

The comparison runs on the cached single-shop `fashion_retail_20` world (20 SKUs — seasonality,
lifecycle, freshness — which meaningfully exercises the censored-sales rate estimator), CRN-paired on
shared `world_seed` / `init_seed` / allocation seed. Results are reported as mean
`net_profit / initial_cash` (profit per opening dollar) and a service-level proxy — the same metrics
the tuner optimises.

## Acceptance criteria

- [ ] `notebooks/06-policy-comparison.ipynb` compares all four textbook policies on the single-shop `fashion_retail_20` world
- [ ] Runs are CRN-paired (shared `world_seed` / `init_seed` / allocation seed) so policies see identical randomness
- [ ] Reported metrics are mean `net_profit / initial_cash` and a service-level proxy
- [ ] `%matplotlib inline`; committed with embedded executed outputs; modest DPI
- [ ] `jupyter nbconvert --to notebook --execute` runs the notebook with zero cell errors and a rendered figure for every plotting cell

## Blocked by

None — can start immediately
