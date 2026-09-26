# 03: Synthetic comparison tables

**What to build:** An evaluation entry point that scores the latent planner, the direct planner, and the latent planner's covers passed through textbook fair-share, on the same holdout seeds as published-default `OrderUpToPolicy`, an optional tuned `OrderUpToPolicy`, and the other three textbook policies at their published defaults. It reports paired `net_profit / initial_cash` for the in-distribution protocol and for the frozen holding-rate retarget, and it computes the pass rules. Ticket tests stay at smoke scale and do not launch a tuning study.

**Blocked by:** 02 — Fit the latent model and plan with it.

**Status:** ready-for-agent

## Parent

`.scratch/jepa-replenishment/PRD.md`. ADR 0009, ADR 0027.

## Stories

PRD stories 28, 29, 34–42, 64. The published 32-seed, 64-proposal, 150-trial execution is ticket 06. This ticket owns the entry point and the smoke-scale tables.

## Prior art

`evaluate_policy_normalised` and its paired CRN reporting of `net_profit / initial_cash`. `TuningConfig` holdout offset `13_000_000` and `n_holdout_seeds` 32. The four textbook policies. Business-metric aggregation from a `Runner` run log. Ticket 02's two Policies and the frozen-rate behavior.

## Artifact homes

- Smoke tables may be returned in memory.
- Published tables, written by ticket 06, land in `runs/jepa-replenishment/synthetic/` (gitignored).

## Acceptance criteria

- [ ] A smoke run on a handful of seeds returns paired `net_profit / initial_cash` for the latent planner, the direct planner, latent covers with textbook fair-share, published-default `OrderUpToPolicy`, and the other three textbook policies at published defaults.
- [ ] The in-distribution smoke uses `holding_rate = 0.01`. The retarget smoke uses the same weights at `holding_rate = 0.05` and includes the textbook policy that was scored at `0.01`, left frozen.
- [ ] The entry point accepts an already-built `OrderUpToPolicy` factory for the `0.01` anchor and another for the `0.05` ceiling. When none is passed, the anchor is the published default. This ticket does not run a 150-trial study.
- [ ] The published defaults of the entry point are the tuner holdout: 32 seeds, offset `13_000_000`. Smoke tests override the seed count.
- [ ] The result includes the in-distribution rule (paired interval against the anchor covers zero, or the latent mean is at least 95% of the anchor mean) and the retarget rule (frozen latent planner beats the frozen textbook policy on paired profit). Smoke tests check that these booleans are computed. They do not require the smoke run to pass the published thresholds.
- [ ] One held-out seed returns a finite paired profit for every arm. The RL stack is not imported.
