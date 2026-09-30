# 11: Train the unroll at the planning horizon and gate on rollout accuracy

**What to build:** A model trained over the horizon the planner actually
consumes, and a gate that refuses to publish a profit table when the model is no
better than a linear fit.

Today the open-loop unroll trains for exactly `delivery_lag` steps while the
planner will now consume roughly four times that. Nothing measures accuracy past
step 3, and compounding error over a long open-loop rollout is the main thing
that will make a correctly-trained model still fail. This is also the axis on
which a joint-embedding model is supposed to beat a direct predictor — latent
rollouts should degrade more gracefully than decoded-state feedback — and it has
never been measured.

The gate exists because the first study published a full profit table from a
model with R² ≈ 0. Nothing in the pipeline objected.

**Blocked by:** 03, 08, 10.

**Status:** done

**Prior art:** existing dynamics tests; the `Latent replenishment planner` and
`Direct replenishment predictor` entries in `CONTEXT.md`. The ordinary-least-
squares baseline is the reference: it reaches +0.94 / +0.75 / +0.65 on the old
feature set and any learned model must beat it on the new one.

- [x] The open-loop unroll trains at the planning horizon;
      `model_unroll_steps` is its own config value rather than `delivery_lag`.
- [x] Held-out rollout accuracy is reported per step from one step to the
      planning horizon, for the latent model, the direct predictor and the
      ordinary-least-squares baseline.
- [x] A publish gate refuses to produce a profit table when held-out rollout
      accuracy fails to beat the ordinary-least-squares baseline on the same
      inputs.
- [x] The gate runs at a real training budget on a reduced episode count — not at
      `epochs=1` — so it means something.
- [x] A test demonstrates the gate fires: a deliberately undertrained model does
      not reach the results table.
- [x] Any notebook, README section or `CONTEXT.md` text invalidated by the gate
      is updated in this ticket.
