# 07 — Remove remaining legacy: Store model, World, world_loaders, dead policy classes

Status: ready-for-agent

## Parent

`.scratch/setup-files/PRD.md`

## What to build

Complete the no-legacy-code transition by deleting the legacy `Store` model, the `World`
artifact, and the dead policy entry points now that every consumer (the runner, RL, and tuning)
has migrated to the setup-files / node-edge model. (The life-cycle/freshness demand layers were
already removed in issue 01.)

Delete:

- `src/sim/world.py`, `src/sim/world_loader.py`, `src/tuning/world_loader.py`, and the
  `world.json` cache path. The setup directory is now the single source of world data; the
  generator writes setup files directly.
- The legacy `Store` model: `StoreTemplate`, `StoreInstance`, `make_stores`, `stores_df`,
  `Scenario.stores`, `Scenario.is_graph`, `Scenario.from_world`.
- The lossy `Scenario.to_json` / `from_json` path.
- Dead policy classes (`Policy` alias, `HeuristicPolicy`, `_HeuristicPolicyOld`, `NoopPolicy`)
  and the `GraphRunner` / `build_graph_world` aliases, leaving a single `Runner` / `build_world`.

Out of scope: collapsing the textbook-policy two-hierarchy bridge (flagged in ADR 0017).

## Acceptance criteria

- [ ] `world.py`, both `world_loader`s, and the `world.json` cache path are deleted; nothing imports them.
- [ ] The `Store` model (`StoreTemplate`, `StoreInstance`, `make_stores`, `stores_df`) and the `Scenario.stores`/`is_graph`/`from_world` members are gone.
- [ ] The lossy `Scenario.to_json`/`from_json` path is removed.
- [ ] Dead policy classes and the `GraphRunner`/`build_graph_world` aliases are deleted; one `Runner`/`build_world` remains.
- [ ] The full test suite is green with no references to the deleted symbols.

## Blocked by

- Issue 02 (runner on setup files)
- Issue 05 (RL off `world_loader`)
- Issue 06 (tuning off `world_loader`)
