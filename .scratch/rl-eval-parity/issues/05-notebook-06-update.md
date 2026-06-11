# 05: Notebook 06 — prose update and full re-execution

Status: ready-for-agent

## Parent

`.scratch/rl-eval-parity/PRD.md` (user story 16)

## What to build

Bring notebook 06 (train-and-eval walkthrough) in line with the new eval semantics so the
two RL notebooks do not contradict each other:

- Update the prose wherever it describes how eval works — the offline eval now runs the RL
  arm through the standard Runner with the Arbiter applied, identical to training dynamics.
- Update the "where to next" closing prose to mention checkpoint attachment via
  `RLNodePolicy` (the worked demo lives in notebook 06a).
- Fully re-execute the notebook end-to-end. It retrains its tiny demo agent — a
  minutes-scale job, expected and fine.

No structural changes to the notebook's sections; this is prose alignment plus fresh
outputs.

## Acceptance criteria

- [ ] Eval-related prose matches the rebuilt harness (no claims contradicting notebook 06a or ADR 0022)
- [ ] "Where to next" mentions attaching a checkpoint to any intermediate node
- [ ] Notebook executes end-to-end cleanly; all cell outputs are fresh (demo agent retrained)

## Prior art

- Notebook re-execution is the repo's established verification pattern for notebooks
- Notebook 06a (issue 04) — the parity statement should be consistent across both notebooks

## Blocked by

- 03-eval-harness-rebuild
