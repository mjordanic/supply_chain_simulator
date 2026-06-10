# `src/rl/` — Reinforcement learning (PPO, variable-K)

A self-contained PPO training stack that wraps the simulator as a Gymnasium environment,
trains a continuous-control policy on pricing and ordering decisions for a variable number
of products per episode, and evaluates the trained policy against `OrderUpToPolicy` with
Common Random Numbers (CRN) so all world variance cancels in the comparison.

Key design choices baked into the stack (ADR 0021): a **shared-weight per-product actor**
applied row-wise to a `(K_max, F)` observation tensor, variable K sampled per episode from
`[1, 20]` (padded to `K_max = 32` with a mask channel), a **deterministic Arbiter** that
reconciles the joint proposal against capacity and cash budget, and structural permutation
invariance that replaces the deleted slot-shuffle. A scale-invariance package — order-up-to
action decoder, demand-units inventory feature, log-uniform domain randomisation over
capacity and balance — lets one trained policy generalise across two orders of magnitude in
node size and across K values up to `K_max = 32` without retraining.

## Contents

1. [Layout](#layout)
2. [Observation and action layout](#observation-and-action-layout)
3. [Quickstart](#quickstart)
4. [Monitoring training](#monitoring-training)
5. [Checkpoints](#checkpoints)
6. [Comparing a trained policy against `OrderUpToPolicy`](#comparing-a-trained-policy-against-orderuptopolicy)
7. [Using a trained policy in a multi-echelon graph](#using-a-trained-policy-in-a-multi-echelon-graph)
8. [CLI flags](#cli-flags)
9. [How the env relates to `Runner`](#how-the-env-relates-to-runner)
10. [Further reading](#further-reading)

## Layout

```
src/rl/
  configs/default.py        RLConfig — frozen dataclass; every knob the stack reads
  episode_sampler.py        RLEpisodeSpec; episode builder; make_synthetic_catalog
  set_encoder.py            encode_set_observation() / decode_set_action() — source of truth
                            for the (K_max, F) obs tensor and (K_max, 3) action tensor
  encoders.py               Legacy flat encoder (retained for old tests only; not used in training)
  env.py                    RLEnv: Gymnasium wrapper; reset via build_world, step via two-phase API
  eval.py                   build_eval_seeds() / evaluate() — CRN-paired RL vs Baseline
                            evaluate_two_scale() — two-scale paired-CRN CLI entry point
  checkpoint.py             save() / load() — self-describing checkpoint I/O
  arbiter.py                Arbiter: proportional and greedy capacity+cash reconciliation
  agents/
    set_actor_critic.py     SetActor / SetCritic / MaskedJointGaussian (ADR 0021)
    ppo.py                  train_ppo(); legacy flat Actor/Critic (retained for old tests)
  train.py                  argparse driver that wires the pieces together
```

`RunSlice` + `aggregate_episode` + KPI helpers live in `src/sim/metrics.py` (shared with
tuning). The per-tick state machine has exactly one implementation —
`Simulation.tick_decide_and_settle()` in `src/sim/runner.py` — and the RL env, eval, and
tuning rollout are all sibling consumers.

The `SetActor` / `SetCritic` boundary in `agents/set_actor_critic.py` is the only place a
future heavier model (set-transformer, attention-over-SKUs) needs to change; the env,
encoder, sampler, and eval harness all stay the same.

## Observation and action layout

The observation is a `(K_max, F)` float32 tensor with `K_max = 32` and `F = 16`. Active
rows carry the per-product features described below; padded rows are all-zero with `mask = 0`.

**Per-product row** (`src/rl/set_encoder.py` named constants):

| Index | Constant | Feature | Range |
|---|---|---|---|
| 0  | `ROW_INVENTORY`       | inventory / per-SKU capacity                | [0, 1] |
| 1  | `ROW_SALES`           | rolling-5-tick mean sales / capacity         | [0, 1] |
| 2  | `ROW_PENDING`         | pending orders / capacity                    | [0, 1] |
| 3  | `ROW_PRICE_RATIO`     | price / MSRP                                 | [0, ∞) |
| 4  | `ROW_MSRP_RATIO`      | MSRP / mean MSRP                             | [0, ∞) |
| 5  | `ROW_COST_RATIO`      | unit_cost / MSRP                             | [0, 1] |
| 6  | `ROW_SUPPLIER_COUNT`  | supplier_count / MAX_SUPPLIER_COUNT          | [0, 1] |
| 7  | `ROW_MIN_PRICE`       | min offered price / MSRP                     | [0, ∞) |
| 8  | `ROW_FILL_RATE`       | mean recent fill rate                        | [0, 1] |
| 9  | `ROW_CASH`            | cash / initial_cash (global, broadcast)      | [0, ∞) |
| 10 | `ROW_TOTAL_INV`       | total_inventory / capacity (global, broadcast) | [0, 1] |
| 11 | `ROW_SIN`             | sin(2π·step/360) (global, broadcast)         | [-1, 1] |
| 12 | `ROW_COS`             | cos(2π·step/360) (global, broadcast)         | [-1, 1] |
| 13 | `ROW_CONTENTION_QTY`  | Σ proposed qty / free space (contention, broadcast) | [0, 1] |
| 14 | `ROW_CONTENTION_COST` | Σ estimated order cost / cash budget (contention, broadcast) | [0, 1] |
| 15 | `ROW_MASK`            | 1.0 = active row, 0.0 = padding             | {0, 1} |

Rows 9–12 are **global features broadcast onto every active row** so the per-product row is
self-contained for the shared-weight actor. Rows 13–14 are **contention aggregates**:
`ROW_CONTENTION_QTY` tells each product how crowded the shared capacity is;
`ROW_CONTENTION_COST` how stretched the cash pool is — without these, each product row sees
only its own demand. Row 15 is the **mask channel**: the DeepSets critic mean-pools only
active rows; the `MaskedJointGaussian` excludes padded rows from the joint log-prob.

**Action**: `(K_max, 3)` continuous values in `[-1, 1]` per active row. Padded rows produce
no orders or prices in the decode output.

| Column | Meaning |
|---|---|
| 0 | price multiplier raw → `[0.5, 1.5] × MSRP` |
| 1 | order-up-to target raw → lead-time units via order-up-to decoder |
| 2 | priority scalar → used by the **greedy Arbiter** variant to rank products |

Implicit assortment: "stop carrying a product" is expressed by holding the order-up-to
target at 0 (sell down inventory) — no discrete carry/drop head, no engine
listing/delisting. The trainable node carries the full episode catalog superset in
`carried_products`. See ADR 0021 Decision 2 for the rationale.

**Arbiter** (between action decode and `execute_buy`): projects the joint order proposal
onto the feasible set defined by node capacity and the cash budget
(`node.cash × cash_budget_fraction`). Two variants behind `RLConfig.arbiter_mode`:
- `proportional` (default): scale all proposals by the binding feasibility ratio.
- `greedy`: fill in order of the priority scalar (action column 2) until capacity or cash
  is exhausted — can express "starve A to stock B", which proportional structurally cannot.

Both variants share one checkpoint format; switching modes is a one-flag CRN-paired
experiment. Engine-side clipping in `execute_buy` remains as a backstop and should be a
no-op when the Arbiter is correctly configured.

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
- `runs/smoke/checkpoints/actor_step{step:010d}.pt` — self-describing checkpoint bundles,
  saved each time the CRN eval fires and once at `total_env_steps`

A full 1 M-step run against a catalog from a setup directory:

```bash
uv run python -m src.rl.train \
  --total-env-steps 1000000 \
  --n-envs 8 \
  --experiment-name fashion_run \
  --setup-dir setups/fashion_retail
```

When `--setup-dir` is provided, the catalog and market are loaded from `catalog.csv` and
the `market:` block of `setup.yaml`. Otherwise the stack falls back to a synthetic catalog
of size `--k-catalog`.

## Monitoring training

Two equivalent paths.

**TensorBoard** (live, browser):

```bash
uv run tensorboard --logdir runs/
```

Then open `http://localhost:6006/`. Run names (`--experiment-name`) are the top-level tag
filter; you can overlay multiple runs to compare hyperparameter sweeps.

**Notebook** (offline / programmatic):

```bash
uv run jupyter notebook notebooks/06-rl-train-and-eval.ipynb
```

The notebook trains a tiny agent, reads the TensorBoard event file, surfaces every scalar
tag, and plots the training curves inline. Use it when you want to slice the curves
programmatically, render to a PDF/PNG report, or skip TensorBoard entirely.

### Scalars logged every PPO update

| Tag | What it measures |
|---|---|
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
|---|---|
| `eval/rl_return` / `eval/baseline_return` | Mean episode return on the held-out CRN set |
| `eval/paired_uplift` | Mean (RL − baseline) **per seed** — the bottom-line metric |
| `eval/win_rate` | Fraction of seeds where RL ≥ baseline |
| `eval/rl_service_level` / `eval/baseline_service_level` | Demand fulfilled |
| `eval/rl_stockout_rate` / `eval/baseline_stockout_rate` | (active-SKU, tick) pairs with zero inventory |
| `eval/rl_inventory_turnover` / `eval/baseline_inventory_turnover` | sum(sales) / mean(inventory) |
| `eval/rl_mean_price_pct_of_msrp` / `eval/baseline_mean_price_pct_of_msrp` | Average pricing position relative to MSRP |
| `eval/rl_revenue` / `eval/baseline_revenue` | Episode revenue |
| `eval/rl_net_profit` / `eval/baseline_net_profit` | Revenue − holding − order cost − fees |

`eval/paired_uplift` is the headline number. Because the CRN eval shares
`(world_seed, capacity, balance, active_subset, allocation_sub_seed)` between the RL and
baseline runs for every seed (`slot_permutation` is absent — permutation invariance is
structural), world stochasticity is fully cancelled — any observed difference is
attributable to the policy alone.

## Checkpoints

Checkpoints are **self-describing bundles** (ADR 0021 Decision 6) saved by
`src.rl.checkpoint.save()`:

```python
{
    "state_dict":     <actor state_dict()>,
    "config":         <training config as plain dict>,
    "layout_version": <OBS_LAYOUT_VERSION integer from src.rl.set_encoder>,
}
```

A checkpoint saved under the wrong observation layout fails loudly on load (layout version
mismatch) instead of with a torch shape error or silently. The checkpoint filename encodes
the global env-step count at the time of saving:

```
runs/<experiment-name>/checkpoints/actor_step{step:010d}.pt
```

Eval-cadence checkpoints are named by the global env-step count at the time of the eval
call; a final checkpoint is written at `total_env_steps`. The critic is intentionally
discarded after training — inference needs only the actor.

Reload an actor:

```python
import torch
from src.rl.agents.set_actor_critic import SetActor
from src.rl.checkpoint import load

bundle = load("runs/smoke/checkpoints/actor_step0000005000.pt")
# bundle keys: "state_dict", "config", "layout_version"
print("layout_version:", bundle["layout_version"])
print("config keys:", list(bundle["config"].keys()))

actor = SetActor()
actor.load_state_dict(bundle["state_dict"])
actor.eval()
```

`SetActor` is a `(B, K_max, F)` → `(B, K_max, 3)` shared-weight MLP; it handles any K up
to `K_max = 32` without reconfiguration. `checkpoint.load()` validates the layout version
on every call and raises `ValueError` on mismatch.

Running the policy against the simulator via `evaluate(...)` (`src/rl/eval.py`):

```python
import numpy as np
import torch
from src.rl.agents.set_actor_critic import SetActor
from src.rl.checkpoint import load
from src.rl.episode_sampler import make_synthetic_catalog
from src.rl.eval import build_eval_seeds, evaluate
from src.rl.configs.default import RLConfig
from src.sim.policy import OrderUpToPolicy

bundle = load("runs/smoke/checkpoints/actor_step0000005000.pt")
actor = SetActor()
actor.load_state_dict(bundle["state_dict"])
actor.eval()

def rl_policy_fn(obs: np.ndarray) -> np.ndarray:
    """obs shape: (K_max, F); returns action shape: (K_max, 3)."""
    t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        action, _, _ = actor.get_action_and_log_prob(t)
    return action.squeeze(0).numpy()

cfg = RLConfig()
catalog = make_synthetic_catalog(cfg.K_catalog)
eval_specs = build_eval_seeds(catalog, cfg, n_seeds=8)
metrics = evaluate(rl_policy_fn, lambda: OrderUpToPolicy(), eval_specs, config=cfg)
print(f"paired_uplift: {metrics['eval/paired_uplift']:+.1f}")
```

## Comparing a trained policy against `OrderUpToPolicy`

`notebooks/06-rl-train-and-eval.ipynb` reproduces the CRN eval offline and exposes per-seed
detail that the TensorBoard-aggregated scalars hide. It:

1. Trains a tiny smoke agent via the `src.rl.train` CLI.
2. Reads training curves from the TensorBoard event file.
3. Loads the latest checkpoint through `checkpoint.load()`, validates layout version.
4. Wraps the `SetActor` in a `(obs_tensor) → action_tensor` closure and calls
   `evaluate(...)` from `src/rl/eval.py` to compute the aggregate paired metrics.
5. Loops the same evaluator one spec at a time to recover per-seed RL and baseline returns.
6. Renders paired-uplift bars and a per-seed scatter.

`notebooks/06a-analyze-a-trained-agent.ipynb` goes deeper on a run that has multiple
eval-cadence checkpoints — model selection, bootstrap CI on the uplift, regime analysis
(does scale-invariance hold across capacity and balance ranges?), and behavioral signature.

## Using a trained policy in a multi-echelon graph

Training and eval both run on the **degenerate 3-node graph** (`FactoryNode("F_<pid>") →
IntermediateNode("S") → DemandSinkNode("D_<pid>")`) — one trainable node. The
scale-invariance package means a trained actor generalises across node size and SKU
assortment, so conceptually it should drop onto any single `IntermediateNode` in a larger
graph.

**This is not wired up yet.** Unlike a *tuned* textbook policy — which is a plain
`IntermediatePolicy` you attach with
`Runner(scenario, policy_overrides={"shop-1": tuned}).run()` (see
[`src/tuning/README.md`](../tuning/README.md#deploying-a-tuned-policy-in-a-multi-echelon-graph))
— a trained `SetActor` cannot be run through `Runner.run()` as-is.
`RLIntermediatePolicy.decide()` only returns an action pre-injected via
`set_pending_action()` inside the env's two-phase loop, and `encode_set_observation` needs
the `CentralTable` and episode-level context (initial cash, active subset, sales history)
that a node's `decide(obs_intermediate, central_table)` is never handed. Reusing a
checkpoint inside an arbitrary multi-echelon graph is a planned follow-up — see
[`TODO.md`](../../TODO.md) §8 for the two implementation options.

## CLI flags

`uv run python -m src.rl.train --help` lists every flag. The ones you tune most often:

| Flag | Default | Meaning |
|---|---|---|
| `--total-env-steps` | 1_000_000 | Total environment steps across all vector envs |
| `--n-envs` | 8 | Parallel envs in the `SyncVectorEnv` |
| `--experiment-name` | rl_ppo | TensorBoard subdir + checkpoint prefix |
| `--seed` | 0 | torch / numpy seed inside the loop |
| `--setup-dir` | None | Load catalog and market from this setup directory |
| `--episode-length` | 180 | Ticks per episode (half-year at daily resolution) |
| `--k-min` | 1 | Minimum K sampled per episode (inclusive) |
| `--k-max-episode` | 20 | Maximum K sampled per episode (inclusive) |
| `--k-catalog` | 100 | Synthetic catalog size when no setup directory is provided |
| `--arbiter-mode` | proportional | `proportional` or `greedy` — Arbiter allocation strategy |
| `--cash-budget-fraction` | 1.0 | Arbiter cash budget = `node.cash × this fraction` |
| `--lr`, `--gamma`, `--gae-lambda`, `--clip-coef`, `--ent-coef`, `--vf-coef`, `--max-grad-norm`, `--n-steps`, `--n-epochs`, `--n-minibatches`, `--target-kl` | CleanRL defaults | Standard PPO knobs (see `RLConfig`) |
| `--eval-cadence-env-steps` | 50_000 | Run a CRN eval (and save a checkpoint) this often |
| `--n-eval-seeds` | 32 | Number of held-out CRN seeds per eval |
| `--no-eval` | off | Skip CRN eval entirely (useful for unit smoke runs) |

**K-generalisation recipe**: train with `--k-max-episode 10`, then evaluate with
`--k-min 20 --k-max-episode 20` via the eval CLI — the shared-weight `SetActor` handles
any K up to `K_max = 32` without retraining:

```bash
uv run python -m src.rl.eval \
  --checkpoint runs/my_run/checkpoints/actor_step0001000000.pt \
  --k-min 20 \
  --k-max-episode 20
```

## How the env relates to `Runner`

`reset()` builds a degenerate three-tier graph for the episode — one `FactoryNode("F_<pid>")`
per active SKU → one trainable `IntermediateNode("S")` → one `DemandSinkNode("D_<pid>")` per
active SKU — and calls `build_world(scenario, policy_overrides={"S": rl_policy})`, where
`rl_policy` is an `RLIntermediatePolicy` shim (`src/sim/policy.py`). `RLEnv.step()` drives
the graph engine's two-phase tick API in the seam between phases:

1. `sim.tick_world()` — advance market / events; publish all seller offers to the central
   table.
2. `encode_set_observation(node_S, market, ...)` — encode the trainable node's state into a
   `(K_max, F)` observation tensor.
3. `SetActor.get_action_and_log_prob(obs)` → `decode_set_action(...)` → Arbiter reconciles
   against capacity + cash budget → `RLIntermediatePolicy.set_pending_action(...)`.
4. `sim.tick_decide_and_settle(...)` — run the demand-pull topological walk: `S.policy.decide()`
   returns the pending action, the FCFS allocator settles trades against the central table,
   deliveries schedule at `current_tick + lead_time`, and demand sinks consume.

Demand is sampled for *every* catalog product each tick (not only the active subset), so
swapping policies leaves the world stream untouched — the CRN cleanliness property.

The reward each tick is node `S`'s `cash` delta (`cash_after − cash_before`). Total episode
return equals the sum of per-tick cash deltas, which is the `net_profit` proxy reported by
`evaluate(...)`. This is why `eval/paired_uplift` is computed on `net_profit`.

`Runner` itself is not modified by anything in `src/rl/`; the same graph simulator powers
both batch scenario runs and RL episodes.

## Further reading

- [`CONTEXT.md`](../../CONTEXT.md) — domain and architecture glossary
- [ADR 0021](../../docs/adr/0021-variable-k-shared-weight-policy.md) — variable-K design
  decisions (shared-weight actor, implicit assortment, Arbiter, self-describing checkpoints)
- [ADR 0004](../../docs/adr/0004-rl-training-env.md) — original RL env design; ADR 0021
  supersedes its slot-shuffle and assortment-head decisions
