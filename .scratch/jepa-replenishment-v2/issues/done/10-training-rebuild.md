# 10: Rebuild world-model training

**What to build:** A fit that actually trains, and reports enough that an
undertrained fit is visible.

Today `fit(epochs=N)` performs exactly `N` full-batch Adam steps — no
minibatching, no inner pass over the data — so "epoch" names a single gradient
step. The published command used the default of 20. Measured on a held-out
episode split, the shipped checkpoint scores R² of −0.003 / −0.016 / +0.004 on
next on-hand, next in-transit and next sales: worse than predicting the training
mean, against +0.94 / +0.75 / +0.65 for ordinary least squares on identical
inputs. Its output standard deviation is 1.89 against a true 114.58, so it emits
a near-constant. Raising the step count to 1500 reaches +0.91 / +0.77 / +0.72.

With an EMA decay of 0.99 over 20 updates, the target encoder remains roughly
85% its random initialisation — and it supplies both the predictor's regression
target and the probe inputs, so the entire latent space is an untrained
projection.

The published `fit.log` contains only Arrow warnings. There is no loss curve, no
validation metric and no early stopping, which is why none of this was noticed.

**Blocked by:** 09.

**Status:** done

**Prior art:** existing dynamics tests — note that every one of them calls `fit`
with `epochs=1` or `epochs=2` and asserts only on shapes and determinism, which
is precisely why the suite stayed green over a model that predicts a constant.
The `Latent replenishment planner` and `Direct replenishment predictor` entries
in `CONTEXT.md`; ADR 0029 (probe stop-gradient).

**Artifact homes:** the loss curve and validation metrics are run artifacts and
live under `runs/` (gitignored), consumed later by the regenerated paper.

- [x] `epochs` means a pass over the training data, not a gradient step; the name
      matches the behaviour or the parameter is renamed.
- [x] Training is minibatched and shuffled.
- [x] A validation split is held out at episode granularity, so no episode
      contributes rows to both sides.
- [x] Early stopping runs on the validation metric.
- [x] The loss curve and validation metrics are written to a run artifact.
- [x] The EMA decay is set against the actual step count so the target encoder
      leaves its initialisation; a test asserts the target has moved materially
      from its initial weights after a real fit.
- [x] The direct predictor gets the same training treatment, so latent-versus-
      direct stays a fair comparison.
- [x] Any notebook, README section or `CONTEXT.md` text invalidated by the new
      training contract is updated in this ticket.
