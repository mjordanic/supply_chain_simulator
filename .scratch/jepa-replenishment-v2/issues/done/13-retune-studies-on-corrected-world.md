# 13: Re-run the tuning studies on the corrected world

**What to build:** A textbook anchor tuned on the world it will be measured in,
and evidence that the rebuilt benchmark has headroom.

The published studies were run on the degenerate world. At `holding_rate = 0.01`
the 150-trial study produced a headline uplift of −9.35 with an interval spanning
zero: tuning bought nothing, because half of every episode was a dead world and
neither capacity nor cash ever bound. A benchmark where the best tuned policy
ties the published default cannot discriminate between methods, and any
comparison run on it is a coin flip.

This ticket re-tunes and, critically, checks that the corrected world is
discriminative before anything downstream depends on it.

**Blocked by:** 04, 07.

**Status:** done

**Prior art:** the `Policy tuning study` and `CRN-paired eval` entries in
`CONTEXT.md`; ADR 0009 (policy hyperparameter tuning tool). The existing study
artifacts under the previous feature's run directory show the shape of the
outputs to reproduce.

**Artifact homes:** study outputs are run artifacts under `runs/` (gitignored).
The headroom finding is quoted in the regenerated paper in `docs/`.

- [x] The tuning studies are re-run on the corrected world and corrected
      episode distribution.
- [x] The tuned anchor beats the published default by a margin whose interval
      excludes zero — if it does not, the benchmark still lacks headroom and that
      is a blocking finding, reported rather than worked around.
- [x] Studies are run at each holding rate the transfer sweep will need, so
      ticket 17 has a tuned ceiling at every point.
- [x] Study artifacts record the git SHA and the episode-distribution parameters
      they were tuned against.
- [x] Any notebook, README section or `CONTEXT.md` text invalidated by the new
      study values is updated in this ticket.
