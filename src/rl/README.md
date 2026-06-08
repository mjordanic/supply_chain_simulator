# `src/rl/` — Reinforcement learning (PPO)

A self-contained PPO training stack that wraps the simulator as a Gymnasium environment, trains a
continuous-control policy on pricing and ordering decisions, and evaluates the trained policy
against `OrderUpToPolicy` with Common Random Numbers (CRN) so all world variance cancels in the
comparison.

Key design choices baked into the env: randomised assortment per episode, slot-shuffled
observations, hidden market state, and a frozen assortment for the duration of every episode. A
scale-invariance package — order-up-to action decoder, demand-units inventory feature, log-uniform
domain randomisation over capacity and balance — lets one trained policy generalise across two
orders of magnitude in node size.

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

`RunSlice` + `aggregate_episode` + KPI helpers live in `src/sim/metrics.py` (shared with tuning).
The per-tick state machine has exactly one implementation — `Simulation.tick_decide_and_settle()`
in `src/sim/runner.py` — and the RL env, eval, and tuning rollout are all sibling consumers.

The `Actor` and `Critic` boundary in `agents/ppo.py` is the only place a future heavier model
(transformer, attention-over-SKUs) needs to change; the env, encoder, sampler, and eval harness
all stay the same.

## Observation and action layout

For `K_active = 5` (default), the flat observation has length `K_active * 18 + 4 = 94`
(`N_PER_SKU = 18`, `N_GLOBAL = 4`):

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
| 13 | demand-units inventory: `clip(inv / rate, 0, max_lt) / max_lt` | [0, 1] |
| 14 | supplier_count / max_supplier_count | [0, 1] | central-table snapshot |
| 15 | min offered price / MSRP | [0, ∞) | central-table snapshot |
| 16 | mean lead time / max lead time | [0, 1] | central-table snapshot |
| 17 | mean recent fill rate | [0, 1] | central-table snapshot |

Slots 14–17 are the **central-table snapshot block**: per product slot the env reads
`central_table.snapshot_for_buyer(pid)` for the trainable node's direct suppliers, so the agent
sees live upstream availability, price, lead time, and fill rate. When no central table is present
they default to zero.

Global block (appended once after all per-SKU slots): `cash / initial_cash`,
`total_inventory / capacity`, `sin(2π·step/360)`, `cos(2π·step/360)`.

Action: `2 * K_active` continuous values in `[-1, 1]`. First `K_active` are price multipliers
(`[-1, 1]` → `[0.5, 1.5]` × MSRP); second `K_active` are order quantities, decoded
order-up-to style (lead-times-of-demand) and split across the slot's suppliers by `decode_action`.
`activate`, `deactivate`, and `promotions` are empty for the duration of every episode — the env
locks the assortment at reset so the agent's job is pricing and ordering only.

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

A full 1 M-step run against a catalog from a setup directory:

```bash
uv run python -m src.rl.train \
  --total-env-steps 1000000 \
  --n-envs 8 \
  --experiment-name fashion_run \
  --setup-dir setups/fashion_retail
```

When `--setup-dir` is provided, the catalog and market are loaded from `catalog.csv` and the
`market:` block of `setup.yaml`. Otherwise the stack falls back to a synthetic catalog of size
`--k-catalog`.

## Monitoring training

Two equivalent paths.

**TensorBoard** (live, browser):

```bash
uv run tensorboard --logdir runs/
```

Then open `http://localhost:6006/`. Run names (`--experiment-name`) are the top-level tag filter;
you can overlay multiple runs to compare hyperparameter sweeps.

**Notebook** (offline / programmatic):

```bash
uv run jupyter notebook notebooks/06-rl-train-and-eval.ipynb
```

The notebook trains a tiny agent, then reads the same `events.out.tfevents.*` files TensorBoard
reads, surfaces every scalar tag, and plots the training curves inline (loss / return / SPS). Use
it when you want to slice the curves programmatically, render to a PDF/PNG report, or skip
TensorBoard entirely.

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

`eval/paired_uplift` is the headline number. Because the CRN eval shares
`(world_seed, init_seed, capacity, balance, active subset, slot permutation)` between the RL and
baseline runs for every seed, world stochasticity is fully cancelled — any observed difference is
attributable to the policy alone. A positive *and stable* `paired_uplift` means the policy beats
`OrderUpToPolicy` on identical worlds, not just on lucky draws.

## Checkpoints

`actor.state_dict()` is saved every `--eval-cadence-env-steps` under:

```
runs/<experiment-name>/checkpoints/actor_step{eval_index:010d}.pt
```

The integer in the filename is the eval-call index, not the global env step —
`actor_step0000000000.pt` is the actor after the first eval, `actor_step0000000001.pt` after the
second, and so on. A final checkpoint is also written at `total_env_steps` after the loop returns.
The critic is intentionally discarded after training because it is not needed for inference.

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

Running the policy against the simulator: wrap the actor in a `(obs_np) → action_np` closure and
pass it to `evaluate(...)` (`src/rl/eval.py`), or step `RLEnv` manually:

```python
import numpy as np
import torch
from src.rl.env import RLEnv
from src.rl.configs.default import RLConfig
from src.rl.episode_sampler import make_synthetic_catalog

cfg = RLConfig()
catalog = make_synthetic_catalog(cfg.K_catalog)

env = RLEnv(catalog=catalog, config=cfg)
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

`notebooks/06-rl-train-and-eval.ipynb` reproduces the CRN eval offline and exposes per-seed
detail that the TensorBoard-aggregated scalars hide. It:

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

Use the eval section to spot regimes the policy fails in (outliers below the 45° line), and to confirm that aggregate uplift is not driven by one or two lucky seeds.

**Business KPI side-by-side.** Per-KPI mean across the 32 CRN-paired held-out seeds — RL (blue)
vs `OrderUpToPolicy` (orange). Higher service level + lower stockout rate at comparable turnover
and revenue → the policy ordered better; the price-multiplier panel surfaces *how* it's pricing
relative to MSRP.

![RL vs OrderUpToPolicy KPIs](../../docs/images/rl_vs_baseline_kpis.png)

## CLI flags

`uv run python -m src.rl.train --help` lists every flag. The ones you tune most often:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--total-env-steps` | 1_000_000 | Total environment steps across all vector envs |
| `--n-envs` | 8 | Parallel envs in the `SyncVectorEnv` |
| `--experiment-name` | rl_ppo | TensorBoard subdir + checkpoint prefix |
| `--seed` | 0 | torch / numpy seed inside the loop |
| `--setup-dir` | None | Load catalog and market from this setup directory |
| `--episode-length` | 180 | Ticks per episode (half-year at daily resolution) |
| `--k-active` | 5 | Active SKUs per episode |
| `--k-catalog` | 100 | Synthetic catalog size when no setup directory is provided |
| `--lr`, `--gamma`, `--gae-lambda`, `--clip-coef`, `--ent-coef`, `--vf-coef`, `--max-grad-norm`, `--n-steps`, `--n-epochs`, `--n-minibatches`, `--target-kl` | CleanRL defaults | Standard PPO knobs (see `RLConfig`) |
| `--eval-cadence-env-steps` | 50_000 | Run a CRN eval (and save a checkpoint) this often |
| `--n-eval-seeds` | 32 | Number of held-out CRN seeds per eval |
| `--no-eval` | off | Skip CRN eval entirely (useful for unit smoke runs) |

## How the env relates to `Runner`

`reset()` builds a degenerate three-tier graph for the episode — one `FactoryNode("F_<pid>")` per
active SKU → one trainable `IntermediateNode("S")` → one `DemandSinkNode("D_<pid>")` per active
SKU — and calls `build_world(scenario, policy_overrides={"S": rl_policy})`, where `rl_policy` is
an `RLIntermediatePolicy` shim (`src/sim/policy.py`). `RLEnv.step()` drives the graph engine's
two-phase tick API in the seam between phases:

1. `sim.tick_world()` — advance market / events; publish all seller offers to the central table.
2. `encode_observation(node_S, market, registry, central_table, …)` — read the trainable node's state plus the central-table snapshot.
3. `decode_action(...)` → `RLIntermediatePolicy.set_pending_action(...)` — inject the agent's pre-decoded per-supplier order/price action.
4. `sim.tick_decide_and_settle(...)` — run the demand-pull topological walk: `S.policy.decide()` returns the pending action, the FCFS allocator settles trades against the central table, deliveries schedule at `current_tick + lead_time`, and demand sinks consume.

Demand is sampled for *every* catalog product each tick (not only the active subset), so swapping
policies leaves the world stream untouched — the CRN cleanliness property.

The reward each tick is node `S`'s `cash` delta (`cash_after − cash_before`). Total episode return
equals the sum of per-tick cash deltas, which is the `net_profit` proxy reported by
`evaluate(...)`. This is why `eval/paired_uplift` is computed on `net_profit`.

`Runner` itself is not modified by anything in `src/rl/`; the same graph simulator powers both
batch scenario runs and RL episodes. The baseline arm of the CRN eval runs `OrderUpToPolicy`
through the identical path via `build_world(scenario, policy_overrides={"S": baseline_policy})`.

## Further reading

- [`CONTEXT.md`](../../CONTEXT.md) — domain and architecture glossary
