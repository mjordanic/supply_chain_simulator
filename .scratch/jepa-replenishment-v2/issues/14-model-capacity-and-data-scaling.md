# 14: Scale model capacity and measure the data-scaling curve

**What to build:** A model large enough that capacity is not the binding
constraint, and a measurement of whether more episodes help.

Today the latent model has 4,867 parameters, of which 1,536 are a frozen EMA copy
— roughly 3,331 trainable, against 232,960 training transitions. The direct
predictor has 387. More tellingly, a properly-trained latent model at this size
reaches +0.91 / +0.77 / +0.72, only matching ordinary least squares at
+0.94 / +0.75 / +0.65. On the old feature set there was no representational gap
for a joint-embedding model to close.

The richer observation from ticket 09 and the harder world from tickets 01 and
05–07 should change that. This ticket establishes whether it does, and answers
the data question by measurement rather than assertion — the simulator makes
episodes cheap, but the first study's bottleneck was optimisation and world
degeneracy, not data volume.

This ticket blocks the headline comparison, so the published table is produced by
the full-size model rather than a small one.

**Blocked by:** 11.

**Status:** ready-for-agent

**Prior art:** the `Latent replenishment planner` and `Direct replenishment
predictor` entries in `CONTEXT.md`; the rollout-accuracy harness from ticket 11
is the measurement instrument here.

**Artifact homes:** the scaling curves are run artifacts under `runs/`
(gitignored) and are read by the regenerated paper and the results notebook.

- [ ] Embedding width and depth grow so the latent model sits in the range of
      roughly half a million to two million parameters.
- [ ] The direct predictor is scaled comparably, so latent-versus-direct is not
      a capacity comparison in disguise.
- [ ] A model-capacity curve reports held-out rollout accuracy against parameter
      count.
- [ ] A data-scaling curve reports held-out rollout accuracy against episode
      count, so the question of whether more data helps is answered by
      measurement.
- [ ] The chosen published configuration is justified from those two curves.
- [ ] Any notebook, README section or `CONTEXT.md` text invalidated by the new
      model size is updated in this ticket.
