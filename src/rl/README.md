# `src/rl/` — Reinforcement learning (PPO)

A self-contained PPO training stack that wraps the simulator as a Gymnasium environment, trains a continuous-control policy on pricing and ordering decisions, and evaluates the trained policy against `OrderUpToPolicy` with Common Random Numbers (CRN) so all world variance cancels in the comparison.

For the architectural decisions behind the env (randomised assortment per episode, slot-shuffled observations, hidden market state, frozen assortment within episode) see [ADR 0004](../../docs/adr/0004-rl-training-env.md). For the scale-invariance package (order-up-to action decoder, demand-units inventory feature, log-uniform domain randomisation) see [ADR 0007](../../docs/adr/0007-rl-scale-invariance-package.md).

## Contents

1. [Layout](#layout)
2. [Observation and action layout](#observation-and-action-layout)
3. [Quickstart](#quickstart)
4. [Monitoring training](#monitoring-training)
5. [Checkpoints](#checkpoints)
6. [Comparing a trained policy against `OrderUpToPolicy`](#comparing-a-trained-policy-against-orderuptopolicy)
7. [CLI flags](#cli-flags)
8. [How the env relates to `Runner`](#how-the-env-relates-to-runner)
9. [Further reading](#further-reading)

## Layout

```
src/rl/
  configs/default.py   RLConfig — frozen dataclass; every knob the stack reads
  episode_sampler.py   RLEpisodeSpec(spec, slot_permutation); 5th seed stream over sim's sampler
  encoders.py          encode_observation() / decode_action() — single source of truth for obs/action
  env.py               RLEnv: Gymnasium wrapper; reset via build_world, step via Simulation two-phase API
  eval.py              build_eval_seeds() / evaluate() — CRN-paired RL vs Baseline
  agents/ppo.py        CleanRL-style PPO + Actor / Critic MLPs + RolloutBuffer
  train.py             argparse driver that wires the pieces together
```

`RunSlice` + `aggregate_episode` + KPI helpers live in `src/sim/metrics.py` (shared with tuning). Per ADR 0010, the per-tick state machine has exactly one implementation — `Simulation.tick_decide_and_settle()` in `src/sim/runner.py` — and the RL env, eval, and tuning rollout are all sibling consumers.

The `Actor` and `Critic` boundary in `agents/ppo.py` is the only place a future heavier model (transformer, attention-over-SKUs) needs to change; the env, encoder, sampler, and eval harness all stay the same.

## Observation and action layout

For `K_active = 5` (default), the flat observation has length `K_active * 13 + 4 = 69`:

| Index in slot | Feature | Range |
| --- | --- | --- |
| 0  | inventory / per-SKU capacity slice | [0, 1] |
| 1  | rolling 5-tick mean sales / per-SKU capacity | [0, 1] |
| 2  | pending orders / per-SKU capacity | [0, 1] |
| 3  | price / MSRP | ≈[0.5, 2] |
| 4  | MSRP / mean MSRP | ≥ 0 |
| 5  | unit_cost / MSRP | [0, 1] |
| 6-10 | lifecycle stage one-hot (introduction, growth, maturity, decline, dead) | {0, 1} |
| 11 | log1p(ticks since activation) / log1p(360) | [0, 1] |
| 12 | in-season flag for the current month | {0, 1} |

Global block (appended once after all per-SKU slots): `cash / initial_cash`, `total_inventory / capacity`, `sin(2π·step/360)`, `cos(2π·step/360)`.

Action: `2 * K_active` continuous values in `[-1, 1]`. First `K_active` are price multipliers (`[-1, 1]` → `[0.5, 1.5]` × MSRP); second `K_active` are order fractions (`[-1, 1]` → `[0, 1]` × per-SKU free space). `activate`, `deactivate`, and `promotions` are forced to empty for the duration of every episode (ADR 0004, Decision 4).

## Quickstart

The fastest end-to-end smoke run uses a synthetic catalog (no LLM, no `OPENAI_API_KEY`):

```bash
uv run python -m src.rl.train \
  --total-env-steps 5000 \
  --n-envs 2 \
  --experiment-name smoke
```

Outputs:

- `runs/smoke/events.out.tfevents.*` — TensorBoard scalars
- `runs/smoke/checkpoints/actor_step{eval_index:010d}.pt` — actor weights, saved each time the CRN eval fires

A full 1 M-step run against a cached LLM-built world:

```bash
uv run python -m src.rl.train \
  --total-env-steps 1000000 \
  --n-envs 8 \
  --experiment-name fashion_run \
  --world-cache-path data/worlds/fashion_retail_1000/world.json
```

The driver resolves the catalog in this order: explicit `--world-cache-path`, then `data/worlds/<--world-archetype>/world.json`, then falls back to a synthetic catalog of size `--k-catalog`. So you can always do a smoke run even without an LLM cache.

## Monitoring training

Two equivalent paths.

**TensorBoard** (live, browser):

```bash
uv run tensorboard --logdir runs/
```

Then open `http://localhost:6006/`. Run names (`--experiment-name`) are the top-level tag filter; you can overlay multiple runs to compare hyperparameter sweeps.

**Notebook** (offline / programmatic):

```bash
uv run jupyter notebook notebooks/05-monitor_rl_training.ipynb
```

The notebook reads the same `events.out.tfevents.*` files TensorBoard reads, surfaces every scalar tag, plots the training curves inline (loss / return / SPS), pairs the CRN eval scalars (RL vs baseline), and lists the saved checkpoints. Use it when you want to slice the curves programmatically, render to a PDF/PNG report, or skip TensorBoard entirely.

### Scalars logged every PPO update

| Tag | What it measures |
| --- | --- |
| `train/episodic_return` | Mean return over episodes that *completed* during this rollout (sum of per-tick balance deltas) |
| `train/episodic_length` | Mean episode length — should track `--episode-length` (default 180) |
| `losses/value_loss` | Critic MSE against returns (clipped) |
| `losses/policy_loss` | PPO clipped surrogate loss |
| `losses/entropy` | Mean entropy of the action distribution — should decrease slowly as the policy sharpens |
| `losses/approx_kl` | KL(new ‖ old) approximation; spikes signal big update steps |
| `losses/clipfrac` | Fraction of samples where the ratio is clipped |
| `charts/learning_rate` | Linearly annealed LR (decays to 0 by `--total-env-steps`) |
| `charts/SPS` | Steps-per-second throughput across all parallel envs |

### Scalars logged every `--eval-cadence-env-steps`

| Tag | What it measures |
| --- | --- |
| `eval/rl_return` / `eval/baseline_return` | Mean episode return on the held-out 32-seed CRN set |
| `eval/paired_uplift` | Mean (RL − baseline) **per seed** — the bottom-line metric |
| `eval/win_rate` | Fraction of seeds where RL ≥ baseline |
| `eval/rl_service_level` / `eval/baseline_service_level` | Demand fulfilled |
| `eval/rl_stockout_rate` / `eval/baseline_stockout_rate` | (active-SKU, tick) pairs with zero inventory |
| `eval/rl_inventory_turnover` / `eval/baseline_inventory_turnover` | sum(sales) / mean(inventory) |
| `eval/rl_mean_price_pct_of_msrp` / `eval/baseline_mean_price_pct_of_msrp` | Average pricing position relative to MSRP |
| `eval/rl_revenue` / `eval/baseline_revenue` | Episode revenue |
| `eval/rl_net_profit` / `eval/baseline_net_profit` | Revenue − holding − order cost − fees |

`eval/paired_uplift` is the headline number. Because the CRN eval shares `(world_seed, init_seed, capacity, balance, active subset, slot permutation)` between the RL and baseline runs for every seed, world stochasticity is fully cancelled — any observed difference is attributable to the policy alone (ADR 0003, ADR 0004). A positive *and stable* `paired_uplift` means the policy beats `OrderUpToPolicy` on identical worlds, not just on lucky draws.

## Checkpoints

`actor.state_dict()` is saved every `--eval-cadence-env-steps` under:

```
runs/<experiment-name>/checkpoints/actor_step{eval_index:010d}.pt
```

The integer in the filename is the eval-call index, not the global env step — `actor_step0000000000.pt` is the actor after the first eval, `actor_step0000000001.pt` after the second, and so on. A final checkpoint is also written at `total_env_steps` after the loop returns. The critic is intentionally discarded after training because it is not needed for inference.

Reload an actor (you must pass the same `RLConfig.K_active` you trained with):

```python
import torch
from src.rl.agents.ppo import Actor
from src.rl.encoders import observation_dim, action_dim
from src.rl.configs.default import RLConfig

cfg = RLConfig()  # change K_active here if you trained with a non-default
obs_dim = observation_dim(cfg.K_active)
act_dim = action_dim(cfg.K_active)

actor = Actor(obs_dim, act_dim)
actor.load_state_dict(torch.load("runs/smoke/checkpoints/actor_step0000000000.pt"))
actor.eval()
```

Running the policy in the simulator: wrap the actor in a `(obs_np) → action_np` closure and pass it to `evaluate(...)` (`src/rl/eval.py`), or step `RLEnv` manually:

```python
import numpy as np
import torch
from src.rl.env import RLEnv
from src.sim.scenario import load_catalog, StoreTemplate
from src.rl.configs.default import RLConfig

cfg = RLConfig()
catalog = load_catalog([
    {"name": f"Product {i}", "category": "General", "related_products": [],
     "base_price": float(10 + i % 30), "unit_cost": float(4 + i % 10),
     "seasonality": "all_season"}
    for i in range(cfg.K_catalog)
])
template = StoreTemplate(
    id="eval", region="US", capacity=200, init_balance=20_000.0,
    init_stock_pct=0.0, delivery_lag=cfg.delivery_lag,
    holding_rate=cfg.holding_rate, order_fee=cfg.order_fee,
    init_active_count=cfg.K_active,
)

env = RLEnv(catalog=catalog, base_template=template, config=cfg)
obs, _ = env.reset(seed=10_000_000)
done = False
total = 0.0
while not done:
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        action_t, _, _ = actor.get_action_and_log_prob(obs_t)
    obs, reward, terminated, truncated, info = env.step(action_t.squeeze(0).numpy())
    done = terminated or truncated
    total += reward
print(f"Episode return: {total:.2f}")
```

## Comparing a trained policy against `OrderUpToPolicy`

`notebooks/06-compare_rl_vs_baseline.ipynb` reproduces the CRN eval offline and exposes per-seed detail that the TensorBoard-aggregated scalars hide. It:

1. Loads a checkpoint into an `Actor`.
2. Builds a fixed held-out 32-seed eval set with `build_eval_seeds(...)`.
3. Runs `evaluate(...)` from `src/rl/eval.py` to compute the aggregate paired metrics — the arithmetic is identical to what `--eval` does during training.
4. Loops the same evaluator with a one-spec list per seed to recover per-seed RL and baseline returns.
5. Renders:
   - Aggregate KPI table (RL vs baseline) — service level, stockout rate, turnover, revenue, net profit, mean price % of MSRP.
   - Paired-uplift histogram + win-rate.
   - Per-seed RL vs baseline scatter on the 45° line.
   - Side-by-side KPI bar chart.
6. Lets you swap `baseline_policy_factory` for a tuned `OrderUpToPolicy(...)` variant — same CRN guarantee holds.

Use the comparison notebook to spot regimes the policy fails in (outliers below the 45° line), and to confirm that aggregate uplift is not driven by one or two lucky seeds.

**Business KPI side-by-side.** Per-KPI mean across the 32 CRN-paired held-out seeds — RL (blue) vs `OrderUpToPolicy` (orange). Rendered from a 1 M-step checkpoint trained on `fashion_retail_250` via `uv run python scripts/render_readme_rl_images.py`. Higher service level + lower stockout rate at comparable turnover and revenue → the policy ordered better; the price-multiplier panel surfaces *how* it's pricing relative to MSRP.

![RL vs OrderUpToPolicy KPIs](../../docs/images/rl_vs_baseline_kpis.png)

## CLI flags

`uv run python -m src.rl.train --help` lists every flag. The ones you tune most often:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--total-env-steps` | 1_000_000 | Total environment steps across all vector envs |
| `--n-envs` | 8 | Parallel envs in the `SyncVectorEnv` |
| `--experiment-name` | rl_ppo | TensorBoard subdir + checkpoint prefix |
| `--seed` | 0 | torch / numpy seed inside the loop |
| `--world-cache-path` | None | Explicit `world.json` path (overrides auto-lookup) |
| `--world-archetype` | rl_train | Auto-lookup key under `data/worlds/<archetype>/` |
| `--episode-length` | 180 | Ticks per episode (half-year at daily resolution) |
| `--k-active` | 5 | Active SKUs per episode |
| `--k-catalog` | 100 | Synthetic catalog size when no world cache is found |
| `--lr`, `--gamma`, `--gae-lambda`, `--clip-coef`, `--ent-coef`, `--vf-coef`, `--max-grad-norm`, `--n-steps`, `--n-epochs`, `--n-minibatches`, `--target-kl` | CleanRL defaults | Standard PPO knobs (see `RLConfig`) |
| `--eval-cadence-env-steps` | 50_000 | Run a CRN eval (and save a checkpoint) this often |
| `--n-eval-seeds` | 32 | Number of held-out CRN seeds per eval |
| `--no-eval` | off | Skip CRN eval entirely (useful for unit smoke runs) |

## How the env relates to `Runner`

`RLEnv.step()` reuses the *exact* tick order from `Runner.run()`:

1. `market.tick()`
2. `event_engine.tick(market)`
3. `item_registry.tick()`
4. Decode action → set on `RLPolicy` shim (`src/sim/policy.py`)
5. `store.decide(store.observe(...))`
6. Dispatch orders (schedule delivery callbacks with adjusted lead time)
7. Settle demand for every catalog product (CRN cleanliness — see [ADR 0003](../../docs/adr/0003-crn-demand-for-all-products.md))

The reward each tick is `balance_after − balance_before`. Total episode return equals the sum of per-tick balance deltas, which is exactly the `net_profit` metric reported by `aggregate_episode(...)`. This is why `eval/paired_uplift` is computed on `net_profit`.

`Runner` itself is not modified by anything in `src/rl/`; the same simulator powers both batch scenario runs and RL episodes.

## Further reading

- [`CONTEXT.md`](../../CONTEXT.md) — domain and architecture glossary
- [ADR 0003](../../docs/adr/0003-crn-demand-for-all-products.md) — CRN demand sampling
- [ADR 0004](../../docs/adr/0004-rl-training-env.md) — RL training env design
- [ADR 0006](../../docs/adr/0006-textbook-reorder-policy-family.md) — Textbook reorder family (the comparison anchor)
- [ADR 0007](../../docs/adr/0007-rl-scale-invariance-package.md) — RL scale-invariance package
- [ADR 0010](../../docs/adr/0010-sim-as-base-for-ml-layers.md) — `src/sim/` as the canonical home for rollout primitives
