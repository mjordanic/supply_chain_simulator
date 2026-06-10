# Train loop + rollout buffer rewire: masked PPO over padded shapes

Status: ready-for-agent

## Parent

`.scratch/variable-k-rl/PRD.md` (governing decision record: ADR 0021, Decisions 1, 5, 6)

## What to build

Rewire the PPO training loop onto the masked set actor-critic and the variable-K env, so a
training run samples episodes at varying K and learns through the masked joint distribution.

- **Rollout buffer**: stores padded `(K_max, ·)` observations/actions plus the mask; masked
  slots drop out of the policy loss, value loss, and entropy bonus. Advantages/returns stay
  per-timestep scalars (shared reward, vanilla PPO — no per-pid decomposition).
- **Vectorised envs**: padded uniform shapes mean the existing vector-env machinery works
  unchanged across parallel envs whose episodes have different K.
- **Checkpointing**: training saves through the checkpoint I/O module (issue 04) — the
  self-describing dict, not a bare state-dict.
- **Config**: the fixed-K knob is replaced by the K-range and `K_max` settings; arbiter mode and
  cash-budget fraction are part of the training config snapshot stored in checkpoints.
- The legacy fixed-K actor/critic and any train-path code reading the old flat layout are
  retired here, along with their superseded tests.

PPO math is otherwise unchanged: same clipping, GAE, optimiser conventions as today.

## Acceptance criteria

- [ ] A short smoke training run (few updates, small episode length) completes end-to-end with K varying across episodes, finite losses, and no NaN.
- [ ] Masked slots verifiably contribute nothing to the loss (e.g. gradients with respect to padded-row inputs are zero in a spot check, or loss is invariant to padded-row content).
- [ ] Checkpoint written through the checkpoint I/O module; reloading it through the same module reconstructs an actor that runs inference.
- [ ] Works under the existing vector-env setup with more than one parallel env.
- [ ] Existing train-driver and PPO smoke tests updated to the new contracts (in scope per the PRD); no new dedicated train test suite.
- [ ] Full test suite green (`uv run pytest`).

Prior art: `test_ppo_smoke`, `test_train_driver` for the smoke pattern being updated.

## Blocked by

- `03-masked-set-actor-critic.md`
- `04-checkpoint-io.md`
- `05-env-sampler-rewire.md`
