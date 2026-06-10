# Env + episode sampler rewire: variable K, superset catalog, arbiter in the step

Status: ready-for-agent

## Parent

`.scratch/variable-k-rl/PRD.md` (governing decision record: ADR 0021, Decisions 1–4)

## What to build

Rewire the Gymnasium env and the RL episode sampler onto the new codec and Arbiter, making
variable K real end-to-end: an env can be reset with K = 3 and again with K = 17, and stepping
it produces arbitrated orders for exactly the active products.

**Episode sampler:**

- K sampled per episode from the configured range (default [1, 20]); frozen within the episode
  (no mid-episode churn, ADR 0021 Decision 3).
- The degenerate training graph (K factories → one trainable intermediate → K sinks) is built
  with the intermediate's `carried_products` set to the full episode product set — the superset
  that makes implicit assortment work (order-up-to target 0 = stop).
- `slot_permutation` deleted from the RL episode spec; the RL slot sub-seed stream retired. Any
  remaining reader of the field in the RL layer is updated in this issue so the suite stays
  green (eval's CRN-tuple construction gets its full rework in issue 07, but must compile and
  pass here).

**Env:**

- Observation/action Box spaces from the new `(K_max, F)` / `(K_max, 3)` layout; mask channel
  populated from the episode's active products.
- Step path: decode action → compute cash budget (node cash × configured fraction, default 1.0,
  costed at central-table offer prices) → Arbiter (mode from config: proportional default,
  greedy switchable) → per-pid order dict to the engine. The old standalone fair-share allocator
  call is replaced by the Arbiter; the legacy allocator and flat codec are retired here along
  with their superseded tests.
- Contention aggregate features fed to the encoder from the env's view of proposals/resources.
- Reward unchanged: per-tick cash delta of the trainable node (already includes holding cost and
  order fees per ADR 0019).

Engine untouched: no dynamic `carried_products`, no listing economics. Engine-side cash clipping
in `execute_buy` remains as a backstop and is expected to be a no-op behind the Arbiter.

## Acceptance criteria

- [ ] Env resets and steps successfully across episodes with different K (e.g. 1, 5, 20) without rebuilding the env object beyond normal reset.
- [ ] Orders reach the engine only for active products; padded slots never order.
- [ ] Arbiter is the only within-tick contention resolver: under cash pressure, allocations follow the configured arbiter mode, and the engine rejection log shows no `insufficient_cash` entries for the trainable node in a smoke episode.
- [ ] `slot_permutation` no longer exists in the RL episode spec or anywhere in the RL layer.
- [ ] Trainable node carries the full episode product set from tick 0.
- [ ] Existing env and sampler tests updated to the new contracts (in scope per the PRD); no new dedicated env test suite.
- [ ] Full test suite green (`uv run pytest`).

Prior art: `test_env_smoke`, `test_episode_sampler`, `test_episode_sampler_setup_dir` for the
contracts being updated; `test_centring_sanity` for ADR 0007 head semantics it must keep
honouring.

## Blocked by

- `01-arbiter-module.md`
- `02-set-encoder-decoder.md`
