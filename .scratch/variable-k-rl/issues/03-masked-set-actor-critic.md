# Masked set actor-critic with masked joint Gaussian distribution

Status: ready-for-agent

## Parent

`.scratch/variable-k-rl/PRD.md` (governing decision record: ADR 0021, Decision 1)

## What to build

The variable-K policy networks and their joint distribution, alongside the existing fixed-K
actor/critic (retired in issue 06).

- **Actor**: one shared-weight MLP applied row-wise, `(B, K_max, F) → (B, K_max, 3)`.
  Tanh-squashed means; state-independent log-std per head, as today.
- **Masked joint Gaussian distribution** (sub-module): joint log-prob is the masked sum of
  per-product diagonal Gaussian log-probs; entropy excludes masked rows; sampling is
  deterministic at fixed seed. Padded slots contribute zero to log-prob, entropy, and any
  gradient.
- **Critic**: DeepSets-style — per-product embedding MLP, masked mean-pool over active rows,
  concatenated with the global block, then a head producing scalar V. Permutation-invariant and
  K-agnostic by construction.

Permutation invariance is the structural guarantee that replaces the deleted slot-shuffle
(ADR 0004 Decision 2), so it must be proven by test, not assumed: permuting the active rows of
the input permutes the per-row actions correspondingly, leaves the joint log-prob unchanged,
and leaves the critic value unchanged.

## Acceptance criteria

- [ ] Actor maps `(B, K_max, F) → (B, K_max, 3)` with one weight set shared across rows.
- [ ] Joint log-prob equals the sum of per-row Gaussian log-probs over active rows only.
- [ ] Entropy excludes masked rows; masked slots receive zero gradient through the loss terms.
- [ ] Sampling at a fixed torch seed is deterministic.
- [ ] **Permutation invariance test**: row permutation of inputs ⇒ identically permuted per-row actions, unchanged joint log-prob, unchanged value.
- [ ] Critic output is a scalar V per batch element, invariant to K (works for K = 1 and K = 32).
- [ ] Dedicated test module for the masked distribution (one of the three pure-core modules with dedicated tests per the PRD).
- [ ] Existing fixed-K actor/critic and tests untouched (retirement in issue 06).

Prior art: current PPO agent module for hidden sizes/activation/log-std conventions;
`test_ppo_smoke` for the cheap-smoke pattern.

## Blocked by

- `02-set-encoder-decoder.md` (row feature dimension F and layout constants come from the new encoder)
