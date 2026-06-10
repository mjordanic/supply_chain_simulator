# Checkpoint I/O: self-describing checkpoints with layout-version validation

Status: done

## Parent

`.scratch/variable-k-rl/PRD.md` (governing decision record: ADR 0021, Decision 6)

## What to build

A small new checkpoint I/O module replacing the bare `actor.state_dict()` save. A checkpoint is
a dict bundling:

- the actor state-dict,
- a training config snapshot,
- the observation-layout version (sourced from the encoder's single constant, issue 02).

Loading validates the stored layout version against the encoder's current constant and raises a
descriptive error on mismatch — a stale checkpoint fails loudly instead of with a torch shape
error (or silently, once dims happen to match again).

This issue ships the module itself; the train loop starts *writing* the new format in issue 06,
and eval/notebooks *read* it in issues 07 and 09. Per the PRD's testing decision, this is not
one of the three pure-core modules with a dedicated test suite — verify through a minimal
save/load round-trip plus the version-mismatch error path, kept small.

## Acceptance criteria

- [ ] Save produces a dict containing state-dict, config snapshot, and layout version.
- [ ] Load returns the bundle when the layout version matches the encoder constant.
- [ ] Load raises a descriptive error (naming both versions) on mismatch — not a shape error.
- [ ] Loading a legacy bare state-dict file fails with a clear "legacy checkpoint, retrain" message rather than a KeyError.
- [ ] No torch-architecture coupling beyond the state-dict itself (the module does not construct networks).

Prior art: `test_train_driver` for how training artifacts are exercised today.

## Blocked by

- `02-set-encoder-decoder.md` (layout version constant lives in the encoder)
