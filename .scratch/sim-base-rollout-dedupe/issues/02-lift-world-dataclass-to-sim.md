# 02 — Lift `World` dataclass to `src/sim/world.py`

Status: ready-for-agent

## Parent

PRD: `.scratch/sim-base-rollout-dedupe/PRD.md`

## What to build

Relocate the `World` persisted-artifact dataclass (catalog + market + store_templates + `_meta`) from `src/llm/world_builder.py` to a new `src/sim/world.py`. Keep `WorldBuilder`, `load_or_build_world`, `LLMBuildAbortedError`, the LLM-stage helpers, and the Pydantic schemas in `src/llm/world_builder.py`.

This split makes the layering clear: sim defines the persisted artifact type; llm provides one (LLM-driven) way to construct it. It is a prerequisite for issue 05 (the new `src/sim/world_loader.py` wants to import `World` from sim, not from llm).

`src/sim/world.py` contains the `World` dataclass plus its serialisation methods (`to_json`, `from_json`, `to_dict`, `from_dict`) and any helpers required for serialisation. The on-disk JSON format at `data/worlds/<archetype>/world.json` is unchanged — only the import path moves.

`src/llm/world_builder.py` gains a transitional re-export — `from src.sim.world import World as World` — so any existing `from src.llm.world_builder import World` keeps working as a safety net during the migration. Internal call sites under `src/` and `tests/` are updated to import from `src.sim.world` directly.

A new `tests/sim/test_world.py` covers (a) JSON round-trip of a small synthetic `World` and (b) every `data/worlds/<archetype>/world.json` currently checked into the repo loads via `World.from_json` without regression. Existing `World` JSON round-trip tests under `tests/llm/` migrate over.

CONTEXT.md's "WorldBuilder" architecture entry splits into two: "World" (the persisted artifact, now in `src/sim/world.py`) and "WorldBuilder" (the LLM-driven generator, still in `src/llm/world_builder.py`).

## Acceptance criteria

- [ ] `src/sim/world.py` exists; defines `World` with `to_json` / `from_json` / `to_dict` / `from_dict`.
- [ ] `src/llm/world_builder.py` no longer defines `World`; instead re-exports it via `from src.sim.world import World as World` and continues to expose `WorldBuilder`, `load_or_build_world`, `LLMBuildAbortedError`.
- [ ] `from src.llm.world_builder import World` still works (transitional re-export).
- [ ] `from src.llm.world_builder import load_or_build_world, WorldBuilder, LLMBuildAbortedError` works unchanged.
- [ ] Internal call sites under `src/` and `tests/` import `World` from `src.sim.world` (not `src.llm.world_builder`).
- [ ] `tests/sim/test_world.py` exists with the two specified tests; passes.
- [ ] Every `data/worlds/<archetype>/world.json` loads via `src.sim.world.World.from_json` without error.
- [ ] CONTEXT.md "WorldBuilder" entry is split into "World" (sim) and "WorldBuilder" (llm).
- [ ] `uv run pytest tests/sim/ tests/llm/ tests/tuning/ tests/rl/` is green.

## Blocked by

None — can start immediately.
