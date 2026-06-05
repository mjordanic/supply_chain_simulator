# 08 — Docs + convert example scenarios

Status: ready-for-agent

## Parent

`.scratch/setup-files/PRD.md`

## What to build

Make the documentation and the repo's runnable examples reflect the setup-files model so a new
user can get started without reading the source.

- Rewrite `README.md`: the two-stage prepare-data / run workflow; the `catalog.csv` +
  `setup.yaml` anatomy; copy-pasteable commands for `run`, `scaffold`, and LLM-generate; the
  determinism + "A/B by copying the setup directory and changing only the `policy` block" story;
  and an updated repo layout (no `World`, no stores).
- Update `CONTEXT.md` to the setup-files model and drop retired terms (`Store*`, `World`,
  life-cycle, freshness, `ItemRegistry`).
- Convert the existing `scenarios/*.py` examples into setup directories under `setups/` (a small
  hand-authored chain plus a larger converted example), so the repo's runnable examples
  demonstrate the new model. Remove the `scenarios/*.py` files they replace.
- Update `src/*/README.md` where they reference deleted concepts.

## Acceptance criteria

- [ ] `README.md` documents the two-stage workflow with copy-pasteable `run` / `scaffold` / LLM-generate commands and the determinism + A/B-by-copy story; the repo layout no longer mentions `World` or stores.
- [ ] `CONTEXT.md` describes the setup-files model with all retired terms removed.
- [ ] The example scenarios are re-expressed as setup directories under `setups/` and run via `main.py run`; the superseded `scenarios/*.py` files are removed.
- [ ] `src/*/README.md` no longer references deleted concepts.

## Blocked by

- Issue 02 (`run` command + example setup)
- Issue 03 (`scaffold` command)
- Issue 04 (LLM-generate writes setup files)
