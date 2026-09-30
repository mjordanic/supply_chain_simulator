# 15: Rebuild the comparison table with floors, a ceiling and parity checks

**What to build:** A results table in which the failures of the first study would
have been impossible to miss.

The published table had three learned arms and no floor. Nothing in it revealed
that the planner ordered 94 units a year. The three learned arms — latent, direct
and fair-share Ration — correlate at r = 0.9999 across seeds with a maximum
absolute difference of 0.039, so both advertised ablations were vacuous, yet they
were tabled as distinct results. And the evaluator already computed service level
and stockout rate per seed; both were discarded, though either would have exposed
the failure immediately.

The oracle row is the important addition. Substituting a perfect world model into
the planner scored 0.336 against the learned model's 0.333 — the measurement that
proved the planner rather than the model was broken. Making it a standing row
means that attribution is always available.

**Blocked by:** 11, 12, 13, 14.

**Status:** done

**Prior art:** existing synthetic comparison tests; the `Synthetic comparison`,
`Business metrics` and `CRN-paired eval` entries in `CONTEXT.md`. The evaluator
already returns per-seed service level and stockout rate — consume them rather
than recomputing.

**Artifact homes:** per-seed tables are run artifacts under `runs/` (gitignored);
the rendered table lands in the regenerated paper in `docs/`.

- [x] Every table carries a never-order floor, a random-cover floor and a
      least-squares-model-in-planner floor.
- [x] Every table carries an oracle-model-in-planner ceiling.
- [x] Order volume, service level, stockout rate and realised fill rate appear on
      every row alongside profit.
- [x] Any two arms whose per-seed results correlate above 0.99 are reported as a
      warning rather than tabled as a distinct comparison.
- [x] A test asserts every learned arm beats the never-order floor.
- [x] Parity assertions: anchor and planner share routing strategy,
      minimum-order floor and unit costs.
- [x] The paired comparison is expressed as per-seed differences with a bootstrap
      interval; tightness is a per-seed column so results can be read against how
      binding the episode was.
- [x] The success rule requires the learned arm to beat the tuned anchor with an
      interval excluding zero.
- [x] Held-out rollout accuracy is recorded alongside every profit number, so the
      two are never read apart.
- [x] Any notebook, README section or `CONTEXT.md` text invalidated by the new
      table shape is updated in this ticket.
