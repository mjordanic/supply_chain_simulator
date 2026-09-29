# 18: Published run, regenerated paper, notebooks and README

**What to build:** The published result, with every number traceable to the run
that produced it, and every notebook and document that describes this study
brought back into agreement with the code.

The first study's paper hand-transcribed its "Run record" from the logs, and the
transition log carried no provenance at all — only pandas column metadata, no
sidecar, no git SHA, no generation parameters. That is how a paper drifts from
its run. Ticket 02 adds the manifest; this ticket makes the paper read from it.

Three notebooks describe this study and all of them are stale after the rebuild:
the training-and-plan walkthrough, the synthetic-evaluation results notebook, and
the M5 notebook. The M5 exam stays out of scope, but its notebook must still not
be left asserting things the code no longer does.

**Blocked by:** 15, 16, 17.

**Status:** ready-for-agent

**Prior art:** the previous feature's published-run sequence — collect, fit,
tune, eval — and the structure of its paper; the `Synthetic comparison` and
`Run Log` entries in `CONTEXT.md`.

**Artifact homes:** run outputs under `runs/` (gitignored). The paper is a
tracked file in `docs/`. Notebooks are tracked under `notebooks/`. Nothing that
must be committed may be written under `runs/` or `data/`, both of which are
ignored.

- [ ] The full published sequence runs end to end: collect, fit, tune, evaluate,
      across all three generalisation protocols and the holding-rate sweep.
- [ ] Every number in the paper is generated from run artifacts rather than
      hand-transcribed, including the run record.
- [ ] The paper reports the model's held-out rollout accuracy alongside every
      profit number.
- [ ] The paper states the training-data generation parameters, read from the
      manifest rather than retyped.
- [ ] The results notebook is updated: the new tables, the floors and oracle
      ceiling, the per-step rollout-accuracy curve, the model-capacity and
      data-scaling curves, and the holding-rate sweep, with visualisations rather
      than bare tables.
- [ ] The training-and-plan notebook is updated to the new training contract,
      observation and search.
- [ ] The M5 notebook is updated so it does not assert behaviour the code no
      longer has, even though the M5 exam stays out of scope.
- [ ] Every README section describing this study is updated, including the
      published command strings.
- [ ] `CONTEXT.md` gains the terms this rebuild introduces and corrects the
      entries the rebuild invalidates.
- [ ] A final consistency pass confirms the PRD, the tickets, the paper, the
      notebooks and the ADRs do not disagree.
