# 04 — LLM generator writes setup files (+ `write_setup`, dir-as-cache)

Status: done

## Parent

`.scratch/setup-files/PRD.md`

## What to build

Make the LLM world generator produce a setup directory instead of the retired `World` artifact,
and add the inverse of `load_setup` that it (and any other writer) uses.

- `write_setup(scenario, dir)` writes `catalog.csv` + `setup.yaml` — the inverse of `load_setup`.
  The round-trip must be lossless for all data, including policies (reconstructable because
  `{name, params}` is serialised, unlike the old lossy `Scenario.to_json` path).
- The LLM generator writes *only data*: `catalog.csv` plus the `market:` block of `setup.yaml`.
  It never writes policies or topology — those stay cleanly separated as hand-authored /
  scaffolded experiment wiring.
- A generated setup directory acts as its own cache: re-running the generator with the same
  target name loads the existing files instead of calling the LLM again (`load_or_build` becomes
  "load the setup dir if present, else generate it").

## Acceptance criteria

- [ ] `write_setup(scenario, dir)` followed by `load_setup(dir)` is a lossless round-trip for all setup data, policies included.
- [ ] The LLM generator writes `catalog.csv` and the `market:` block of `setup.yaml`, and nothing else (no `nodes`, `edges`, or `policy` content).
- [ ] Re-running the generator against an existing setup directory loads the files without calling the LLM.
- [ ] A round-trip test asserts losslessness; generator output loads cleanly via `load_setup`.

## Blocked by

- Issue 02 (`load_setup`, `Scenario` in-memory form, setup format)
