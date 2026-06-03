# 07 — Tier 3 two-scale paired-CRN eval runner

Status: ready-for-agent

## Parent

PRD: `.scratch/rl-scale-invariance/PRD.md`
ADR: `docs/adr/0007-rl-scale-invariance-package.md`

## What to build

Extend `src/rl/eval.py` with a two-scale paired-CRN eval runner that takes a checkpoint path and emits per-scale uplift means / 95% bootstrap CIs / cold-start qty statistics. This is the *code* for Tier 3; the actual training run and Tier 3 execution are gated on a separately-reviewed checkpoint and are explicitly out of scope for this PRD.

PRD user story 19: "As an RL researcher merging the work, I want the Tier 3 paired-CRN evaluation script reproducible from a single command, taking a checkpoint path and emitting per-scale uplift means / CIs / cold-start qty statistics, so that the validation result is auditable and re-runnable on any future checkpoint."

### Function shape

Add a new public entry point to `src/rl/eval.py` (alongside the existing `evaluate`):

```python
def evaluate_two_scale(
    checkpoint_path: str,
    *,
    catalog: list[Ware],
    base_template: StoreTemplate,
    config: RLConfig,
    n_seeds: int = 32,
    seed_offset: int | None = None,
) -> TwoScaleEvalResult:
    """Run paired-CRN evaluation against OrderUpToPolicy at two scales.

    Two held-out eval sets, disjoint from the training seed range:
      - small: capacity_dist = Uniform(150, 400); balance proportional.
      - flagship: capacity_dist = Constant(10_000); balance proportional.

    Each set runs n_seeds CRN-paired seeds (same world_seed / init_seed
    for the RL policy and OrderUpToPolicy). Per-scale results carry
    uplift mean, 95% bootstrap CI, and tick-0 cold-start qty distribution.
    """
```

`TwoScaleEvalResult` is a small dataclass with `small: ScaleResult`, `flagship: ScaleResult`. `ScaleResult` carries `mean_uplift: float`, `uplift_ci_low: float`, `uplift_ci_high: float`, `mean_cold_start_qty: float`, `cold_start_qty_p05: float`, `cold_start_qty_p95: float`, `per_seed: list[PairedSeedResult]` (for downstream auditing).

### Internal mechanics

- For each scale (`small` / `flagship`), construct a per-scale `RLConfig` derived from the input `config` by overriding `capacity_dist` and `balance_dist` (do **not** mutate the input config — return a `dataclasses.replace`'d copy).
- For each seed in `range(seed_offset + scale_offset, seed_offset + scale_offset + n_seeds)`, run paired CRN through the existing `evaluate()` helper. The `scale_offset` differs per scale so the two scales' seed spaces are disjoint from each other and from the training space (default `seed_offset = config.eval_seed_offset + 1_000_000`; `scale_offset = 0` for small, `scale_offset = 100_000` for flagship).
- Cold-start qty per seed: the qty the RL policy emits on tick 0, summed across active SKUs and divided by `K_active`. Captured by inspecting the action dict produced by `decode_action` at tick 0; either route through an info channel from `RLEnv.step` or do an out-of-band rollout of the same `(world_seed, init_seed)` and capture the first decoded action.
- 95% bootstrap CI: 1000 resamples with replacement from the per-seed paired uplifts; report the 2.5th and 97.5th percentiles.

### Smoke test

This slice ships the runner code only — Tier 3 execution requires a trained checkpoint. To prove the runner works mechanically, add a smoke test:

`tests/rl/test_two_scale_eval_smoke.py`:

`test_evaluate_two_scale_with_baseline_against_baseline_returns_near_zero_uplift`:

- Build a fake "checkpoint" by running `OrderUpToPolicy` itself as the RL policy (i.e. substitute the rollout function so it produces the same trajectory as the baseline).
- Call `evaluate_two_scale(...)` with `n_seeds = 4` for speed.
- Assert `abs(result.small.mean_uplift) < 1e-6` and `abs(result.flagship.mean_uplift) < 1e-6`.
- Assert the dataclass shape: `result.small` and `result.flagship` both have non-None `mean_cold_start_qty`, `uplift_ci_low/high`, and `len(per_seed) == 4`.

This is a mechanics test — it does not validate the eventual Tier 3 pass criteria from the PRD (mean uplift > 0, scale-invariance within 2×, cold-start in `[0.5, 2.0] × prior × target_centre`). Those criteria are checked manually once a real checkpoint exists.

### CLI wrapper

Add a `__main__` block (or a small `scripts/eval_two_scale.py` if the repo prefers a separate scripts dir) so the runner is invocable as a single command:

```
uv run python -m src.rl.eval --checkpoint runs/<name>/<step>.pt \
    --world rl_train --n-seeds 32 --seed-offset 11000000
```

Output: a small JSON report at `runs/<name>/<step>__two_scale_eval.json` with the `TwoScaleEvalResult` serialised, plus a console table summarising both scales.

### Out of scope

- Running this against a real checkpoint. Gated on retraining, which is itself gated on slices 1–6 landing.
- The actual Tier 3 pass-or-fail decision. The runner emits numbers; a human reads them against the PRD's three criteria.

## Acceptance criteria

- [ ] `src/rl/eval.py` exports `evaluate_two_scale` and `TwoScaleEvalResult` / `ScaleResult` dataclasses.
- [ ] `evaluate_two_scale` accepts a checkpoint path, runs two CRN eval sets, returns the documented result shape.
- [ ] The function does not mutate the input `RLConfig` (verified by hashing the config before / after).
- [ ] Bootstrap CI is reproducible across runs (seed the bootstrap RNG from a deterministic source — e.g. `seed_offset`).
- [ ] CLI wrapper or `scripts/eval_two_scale.py` exists and accepts `--checkpoint`, `--n-seeds`, `--seed-offset`, `--world` (or equivalent) and writes a JSON report.
- [ ] The smoke test passes under `uv run pytest tests/rl/test_two_scale_eval_smoke.py`.
- [ ] No regression in `uv run pytest tests/rl/` or `uv run pytest tests/sim/`.

## Blocked by

- `06-log-uniform-defaults-and-tier-2.md` — the runner is meaningful only on top of the full slice-1-through-5 package, and the slice-6 Tier 2 test confirms the package's behaviour before any training cycle.
