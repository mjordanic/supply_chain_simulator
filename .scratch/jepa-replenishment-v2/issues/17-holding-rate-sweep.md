# 17: Make the holding-rate transfer claim falsifiable

**What to build:** A transfer claim that can fail.

The frozen holding-rate retarget is the first study's headline claim, and it is a
literal no-op. Measured directly, the frozen planner places byte-identical orders
at holding rates 0.01 and 0.05 — 94 units over 7 ticks on one seed, 148 over 8 on
another, the same at both rates. The reported drop from 0.363 to 0.162 is the
world charging more for unchanged behaviour. The acceptance test for it passes on
a contrived fixture where the till empties, while the real protocol exercises
nothing, because holdout cash averages roughly 293,000 and the constraint never
binds.

The claim is worth keeping because it is the precise reason to prefer a world
model over a policy network: `holding_rate` is a reward parameter, not a dynamics
parameter. The model predicts on-hand, in-transit and sales, none of which depend
on it, so a model-based planner should re-optimise for free while the tuned
anchor's knobs are frozen and a PPO policy would need retraining. Once cash binds
and the horizon is long enough for holding cost to accumulate, the rate genuinely
changes the optimum.

**Blocked by:** 15.

**Status:** ready-for-agent

**Prior art:** existing synthetic comparison tests; the `Synthetic comparison`
and `Holding cost` entries in `CONTEXT.md`; ADR 0027 (frozen holding-rate
retarget), which this ticket amends.

- [ ] The transfer claim is evaluated as a sweep across several holding rates
      rather than a single pair.
- [ ] At each rate the frozen planner is compared against a textbook policy tuned
      at that same rate, so the ceiling is known at every point.
- [ ] A hard assertion verifies the frozen planner's *order sequence* differs
      across rates on the real holdout — not on a fixture. A no-op protocol must
      fail loudly rather than report a number.
- [ ] The model weights are not refit and no new log is collected, so the claim
      stays zero-shot.
- [ ] ADR 0027 is amended to record that the original protocol was unfalsifiable
      under the old episode distribution, and what the corrected protocol
      requires.
- [ ] Any notebook, README section or `CONTEXT.md` text invalidated by the sweep
      is updated in this ticket.
