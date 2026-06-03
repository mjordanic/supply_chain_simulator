# 05 — Lift world loading to `src/sim/world_loader.py`; tuning + RL train consume via thin Config-adapters

Status: ready-for-agent

## Parent

PRD: `.scratch/sim-base-rollout-dedupe/PRD.md`

## What to build

Make `src/sim/world_loader.py` the canonical home for the `cache_path → archetype → synthetic fallback` resolution that both `src/tuning/world_loader.py::load_world` and `src/rl/train.py::_load_world_catalog_and_template` implement today. Both call sites collapse to ~10-line Config-adapters around the new sim function.

### New sim module

`src/sim/world_loader.py` exposes:

```python
def load_world(
    *,
    archetype: str,
    cache_path: str | None = None,
    delivery_lag: int,
    holding_rate: float,
    order_fee: float,
    K_active: int,
    synthetic_fallback: bool = True,
    synthetic_catalog_size: int = 100,
) -> tuple[list[Ware], StoreTemplate, MarketParams | None, DisruptionParams | None]:
    ...
```

Resolution order: explicit `cache_path` → `data/worlds/<archetype>/world.json` → synthetic fallback (gated by `synthetic_fallback`; when `False` and neither path resolves, raises).

Internals: imports `World` from `src.sim.world` (relying on issue 02). Applies `replace(default_disruption_params(), regions=world.market.regions)` for the disruption-params regions adjustment that both current copies do. `default_disruption_params` comes from `src.sim.episode_sampler` (issue 04).

### Tuning-side shrink

`src/tuning/world_loader.py` shrinks from ~134 lines to ~15: a Config-adapter that unpacks `TuningConfig` and calls `src.sim.world_loader.load_world`.

### RL-side rewrite

`src/rl/train.py::_load_world_catalog_and_template` becomes a ~10-line wrapper around `src.sim.world_loader.load_world` reading from `RLConfig`. Delete the private `_load_world_from_file` and `_build_synthetic_catalog` helpers in `train.py`.

### Sim-side tests

`tests/sim/test_world_loader.py` covers:

- Explicit `cache_path` resolves first when present.
- Archetype lookup (via `data/worlds/<archetype>/world.json`) resolves second.
- `synthetic_fallback=True` produces a viable 4-tuple when neither path resolves; `synthetic_fallback=False` raises.
- Disruption-params regions adjustment matches `world.market.regions`.

This test file is the only systematic exercise of the synthetic-fallback path — the end-to-end smoke runs (issue 09) all load `fashion_retail_250`, so synthetic fallback is uncovered there.

### CONTEXT.md update (inline)

Add new "World loader" architecture entry pointing at `src.sim.world_loader` and noting the `cache_path → archetype → synthetic` resolution order.

## Acceptance criteria

- [ ] `src/sim/world_loader.py` exists with the signature above; resolution order is enforced.
- [ ] `src/sim/world_loader.py` imports `World` from `src.sim.world` (not `src.llm.world_builder`).
- [ ] `src/tuning/world_loader.py` is ≤25 lines; delegates to `src.sim.world_loader.load_world`.
- [ ] `src/rl/train.py::_load_world_catalog_and_template` is ≤25 lines; delegates to `src.sim.world_loader.load_world`. The `_load_world_from_file` and `_build_synthetic_catalog` helpers in `train.py` are deleted.
- [ ] `tests/sim/test_world_loader.py` exists with the four tests above; passes.
- [ ] CONTEXT.md has new "World loader" entry.
- [ ] `uv run pytest tests/sim/ tests/tuning/ tests/rl/` is green.

## Blocked by

- `02-lift-world-dataclass-to-sim.md` — `src/sim/world_loader.py` must import `World` from `src.sim.world`.
