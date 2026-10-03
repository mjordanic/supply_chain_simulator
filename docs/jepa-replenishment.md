# Latent replenishment

## Method

The learned buyer is ordering only. `decide` sets `list_price` from
`base_price`, emits no promotions, and sets the minimum-order floor to
zero. Inventory position is on-hand plus in-transit. The order before
the ration is `max(0, round(cover × rate × delivery_lag) − position)`
on every tick. The rate is the textbook censored-sales estimator over
`delivery_lag`. Covers are chosen from `{0, 1, 2, 4}`.

One learned buyer sits on the intermediate that faces the sinks.
Factories stay on `StaticFactoryPolicy`. There is no second learned
echelon. The anchor and the planner share cheapest-first routing and
the same `per_supplier_min_order_floor`.

The world model is per SKU. It is conditioned on the quantity
Allocation filled. The planner imagines that quantity as the grant
times the textbook arms' fill rate, and the emitted order stays the full grant.
The encoder sees on-hand, an arrival-age
profile, a sales window of its own length rather than the delivery
lead time, cash, capacity, and that lead time. The Market
multiplier, the disruption identity, and the calendar are not
features. An
exponential-moving-average target encoder supplies the regression
target. The predictor rolls its embedding forward for the planning
horizon, `model_unroll_steps`, which defaults to
`horizon_multiple * delivery_lag` and is not the delivery lead time.
The latent loss is the distance to that stopped
target at each step. An anti-collapse penalty keeps embedding variance
off the floor and penalizes cross-dimension correlation. Probes decode
next on-hand, the next arrival-age profile, and sales over the step.
The planner sums that profile for inventory position. Probe losses
update the probes only. The planner's profit calculation does not
update the encoder. Imagined profit is revenue minus order cost minus
holding cost minus order fees, which is the engine's `net_profit`,
plus a terminal credit of ending inventory position at unit cost.
That credit is not cash the next ration can spend.
There is no stockout penalty and no learned reward head.

Each tick scores `budget` cover proposals. That budget is the
`budget` line in the run record. The search is factored
cross-entropy over a per-SKU cover, seeded by `policy_seed`. Priority
is the marginal imagined profit of one unit of shared capacity. A
greedy fill grants units in that order until free capacity or cash is
exhausted. `ration="permutation"` draws one order from `policy_rng`
and is the ablation. `ration="fair_share"` keeps the cover search
and splits with textbook fair-share. The winner is the proposal with
the highest imagined `net_profit`. The same `policy_seed` places the
same orders on a repeated run. Search randomness is `policy_seed`.
`OrderUpToPolicy` keeps fair-share. Engine Allocation stays
`execute_buy`. The direct predictor uses the same search and the same
ration, with no latent and no EMA target.

The training log mixes all four textbook families, with hyperparameters
sampled from the tuning search spaces, plus the exploratory policy and
never-order and over-order arms. Planning depth varies across episodes.
The log is collected at a single holding rate, on the training
partition of the catalog. Untuned textbook policies
stay at published defaults on the holdout tables. The in-distribution
protocol uses the tuner holdout at `holding_rate = 0.01`, the training
SKUs, and the default disruption. The anchor is an
`OrderUpToPolicy` tuned at that rate. Unseen SKU uses the evaluation
partition and its own seed block. Unseen disruption uses the training
SKUs and a harsher disruption regime the log never samples.
The holding-rate sweep keeps the predictor
trained at `holding_rate = 0.01`. Imagined cash is rolled forward at
each sweep rate and fed back into the ration. The weights are
not refit, and behavior data is not recollected. At each rate the
ceiling is the `OrderUpToPolicy` tuned at that same rate. The sweep
writes no curve when the frozen planner's order sequence is identical
across those rates.

Every comparison table carries a never-order floor, a random-cover
floor, and a least-squares model in the planner, plus an oracle model
in the planner as a ceiling. Order volume, service level, stockout
rate, and realised fill rate are reported on every arm. Held-out
rollout R² for the latent model, the direct predictor, and ordinary
least squares is printed beside those profit means.

## Caveats

Sales are censored. The censored-sales rate treats a stockout as low
demand, so a cover learned from that rate is not a cover of true
demand. Lost sales appear only as revenue that did not arrive.

On the M5 arm, `unit_cost` is 40% of median price. That figure is the
adapter's documented fiction, not a Walmart cost. Both ordering
policies would face the same fiction, inside one `PriceReplayPolicy`,
under `flat_world`.

## Success rules

In-distribution rule (holding_rate 0.01, tuner holdout): pass if the learned arm beats the tuned anchor on paired net_profit / initial_cash and the bootstrap interval of latent minus that anchor excludes zero.

Holding-rate sweep (0.005, 0.01, 0.02, 0.05, 0.1; same weights, no new log): at each rate the frozen planner is scored against the OrderUpToPolicy tuned at that rate. The run reports no curve when the frozen planner's order sequence is identical across those rates on the tuner holdout.

The textbook policy tuned at each sweep rate is reported and is not a
pass bar. An M5 row is not a gate. A profit table is not written when
held-out latent rollout R² fails to beat ordinary least squares on the
same inputs.

## Where this draft does not belong

The claim worth writing down is whether the frozen planner's orders
change across holding rates, and the split M5 result, against a
textbook policy tuned at the same rate, with the direct predictor and
the fair-share arm in the table. When the sweep refuses because those
orders are identical, that refusal is the result for that claim.
Venues that want a structural inventory proof, and venues that want a
PPO or Dreamer bake-off, are the wrong homes for this draft.

## Run record

Training-data parameters are copied from `log/transitions.manifest.json`.
Episode counts below the manifest are copied from the transition log.
Planner fields are copied from `checkpoints/planner.json`.
Each study line is copied from that directory's `study.json`.
`best_value` is that file's search objective on the search seeds.
It is not a holdout mean. The transition log omits the final tick of
each episode, because that tick has no next state.

Training-data generation parameters:
manifest episode_count 896
manifest seed_min 14000000
manifest seed_max 14000895
manifest episode_length 365
manifest K_active null
manifest K_min 1
manifest K_max_episode 30
manifest delivery_lag null
manifest holding_rate 0.01
manifest order_fee 50.0
manifest catalog_identity "8e66d1343f47b8e4cf65894deaf0f7ef2db4ee9f322bbf5e3f8a9e49ed8aca72"
manifest evaluation_catalog_identity "971cb35429d8611d7c4ded34e790b4d16b9672bbf08871e59f39e4693bcb3911"
manifest training_sku_count 600
manifest evaluation_sku_count 150
manifest git_sha "32e0309b468af938eab38c4de8109a1b80a1d543"
manifest nonzero_order_fraction 0.1464101181547187
manifest disruption {"duration": {"type": "constant", "value": 3}, "event_prob": 0.05, "regions": ["US"], "severity": {"type": "constant", "value": 0.01}, "types": ["natural_disaster", "economic_crisis"]}
manifest harsh_disruption {"duration": {"type": "constant", "value": 21}, "event_prob": 0.2, "regions": ["US"], "severity": {"type": "constant", "value": 0.2}, "types": ["natural_disaster", "economic_crisis", "pandemic", "political_unrest"]}
manifest protocols {"in_distribution": {"catalog": "training", "disruption": "default", "n_seeds": 32, "seed_max": 13000031, "seed_min": 13000000}, "training": {"catalog": "training", "disruption": "default", "n_seeds": 896, "seed_max": 14000895, "seed_min": 14000000}, "unseen_disruption": {"catalog": "training", "disruption": "harsh", "n_seeds": 32, "seed_max": 16000031, "seed_min": 16000000}, "unseen_sku": {"catalog": "evaluation", "disruption": "default", "n_seeds": 32, "seed_max": 15000031, "seed_min": 15000000}}
manifest state_coverage {"capacity_bound": 0.023328701772820037, "cash_bound": 0.3840061001415583, "high_inventory": 0.20743265611162445, "low_inventory": 0.41810221999989244}
manifest mixture order_up_to episodes 64 policy OrderUpToPolicy
manifest mixture reorder_point episodes 64 policy ReorderPointPolicy
manifest mixture periodic_order_up_to episodes 64 policy PeriodicOrderUpToPolicy
manifest mixture periodic_reorder episodes 64 policy PeriodicReorderPolicy
manifest mixture exploratory episodes 64 policy ExploratoryPolicy
manifest mixture never_order episodes 64 policy NeverOrderPolicy
manifest mixture over_order episodes 512 policy OverOrderPolicy
Per-episode mixture hyperparameters stay in the manifest sidecar and are not retyped here.

exploratory_episodes 64
never_order_episodes 64
order_up_to_episodes 64
over_order_episodes 512
periodic_order_up_to_episodes 64
periodic_reorder_episodes 64
reorder_point_episodes 64
log_seed_min 14000000
log_seed_max 14000895
products_per_episode 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30
recorded_tick_min 1
recorded_tick_max 364
recorded_tick_count 364
delivery_lag 3
budget 64
fit_holding_rate 0.010000
order_fee 50.000000
study order_up_to_holding_0_005 holding_rate 0.005
study order_up_to_holding_0_005 n_trials 150
study order_up_to_holding_0_005 n_search_seeds 16
study order_up_to_holding_0_005 n_holdout_seeds 32
study order_up_to_holding_0_005 episode_length 365
study order_up_to_holding_0_005 seed_offset 12000000
study order_up_to_holding_0_005 holdout_seed_offset 13000000
study order_up_to_holding_0_005 best_value 1.9102880646287577
study order_up_to_holding_0_005 git_sha 6fda2d43ce638d5e8f3788d299eb3816d2e6371b
study order_up_to_holding_0_005 best_params {"cover_horizon_ticks": 12, "safety_lead_pct_of_lag": 0.05025909338623214, "stockout_safety_bonus_pct_of_lag": 0.19108043943834024}
study order_up_to_holding_0_01 holding_rate 0.01
study order_up_to_holding_0_01 n_trials 150
study order_up_to_holding_0_01 n_search_seeds 16
study order_up_to_holding_0_01 n_holdout_seeds 32
study order_up_to_holding_0_01 episode_length 365
study order_up_to_holding_0_01 seed_offset 12000000
study order_up_to_holding_0_01 holdout_seed_offset 13000000
study order_up_to_holding_0_01 best_value 1.0641955949975148
study order_up_to_holding_0_01 git_sha 6fda2d43ce638d5e8f3788d299eb3816d2e6371b
study order_up_to_holding_0_01 best_params {"cover_horizon_ticks": 12, "safety_lead_pct_of_lag": 0.16993287200409285, "stockout_safety_bonus_pct_of_lag": 0.09706576917593086}
study order_up_to_holding_0_02 holding_rate 0.02
study order_up_to_holding_0_02 n_trials 150
study order_up_to_holding_0_02 n_search_seeds 16
study order_up_to_holding_0_02 n_holdout_seeds 32
study order_up_to_holding_0_02 episode_length 365
study order_up_to_holding_0_02 seed_offset 12000000
study order_up_to_holding_0_02 holdout_seed_offset 13000000
study order_up_to_holding_0_02 best_value -0.5221367401283703
study order_up_to_holding_0_02 git_sha 6fda2d43ce638d5e8f3788d299eb3816d2e6371b
study order_up_to_holding_0_02 best_params {"cover_horizon_ticks": 7, "safety_lead_pct_of_lag": 0.12627988145408958, "stockout_safety_bonus_pct_of_lag": 0.06893640026208986}
study order_up_to_holding_0_05 holding_rate 0.05
study order_up_to_holding_0_05 n_trials 150
study order_up_to_holding_0_05 n_search_seeds 16
study order_up_to_holding_0_05 n_holdout_seeds 32
study order_up_to_holding_0_05 episode_length 365
study order_up_to_holding_0_05 seed_offset 12000000
study order_up_to_holding_0_05 holdout_seed_offset 13000000
study order_up_to_holding_0_05 best_value -4.994853343225005
study order_up_to_holding_0_05 git_sha 6fda2d43ce638d5e8f3788d299eb3816d2e6371b
study order_up_to_holding_0_05 best_params {"cover_horizon_ticks": 3, "safety_lead_pct_of_lag": 0.217201280660895, "stockout_safety_bonus_pct_of_lag": 0.24606803342018368}
study order_up_to_holding_0_1 holding_rate 0.1
study order_up_to_holding_0_1 n_trials 150
study order_up_to_holding_0_1 n_search_seeds 16
study order_up_to_holding_0_1 n_holdout_seeds 32
study order_up_to_holding_0_1 episode_length 365
study order_up_to_holding_0_1 seed_offset 12000000
study order_up_to_holding_0_1 holdout_seed_offset 13000000
study order_up_to_holding_0_1 best_value -12.513578848373466
study order_up_to_holding_0_1 git_sha 6fda2d43ce638d5e8f3788d299eb3816d2e6371b
study order_up_to_holding_0_1 best_params {"cover_horizon_ticks": 3, "safety_lead_pct_of_lag": 0.21533775713818853, "stockout_safety_bonus_pct_of_lag": 0.23836659081391773}
in_distribution_seeds 32
in_distribution_seed_min 13000000
in_distribution_seed_max 13000031
unseen_sku_seeds 32
unseen_sku_seed_min 15000000
unseen_sku_seed_max 15000031
unseen_disruption_seeds 32
unseen_disruption_seed_min 16000000
unseen_disruption_seed_max 16000031
holding_rate_sweep_seeds 160
holding_rate_sweep_seed_min 13000000
holding_rate_sweep_seed_max 13000031
Means, intervals, and rollout R² below are printed to 6 decimal places.

## Rollout accuracy

Held-out rollout accuracy, mean R² over the planning horizon:
rollout_r2_latent_mean 0.432036
rollout_r2_direct_mean 0.742028
rollout_r2_ols_mean -0.110273

Per-step rollout R²:
step 1 latent 0.368218 direct 0.919635 ols -0.113529
step 2 latent 0.451315 direct 0.784160 ols -0.112925
step 3 latent 0.457716 direct 0.769770 ols -0.112317
step 4 latent 0.454601 direct 0.752939 ols -0.111704
step 5 latent 0.450153 direct 0.742439 ols -0.111108
step 6 latent 0.446080 direct 0.732054 ols -0.110522
step 7 latent 0.439383 direct 0.723257 ols -0.109952
step 8 latent 0.433938 direct 0.712832 ols -0.109379
step 9 latent 0.429128 direct 0.703989 ols -0.108821
step 10 latent 0.423438 direct 0.694810 ols -0.108236
step 11 latent 0.418161 direct 0.688063 ols -0.107675
step 12 latent 0.412300 direct 0.680391 ols -0.107112

## In-distribution

In-distribution rule (holding_rate 0.01, tuner holdout): pass if the learned arm beats the tuned anchor on paired net_profit / initial_cash and the bootstrap interval of latent minus that anchor excludes zero.

The in-distribution claim is that rule.

Held-out rollout accuracy, mean R² over the planning horizon:
rollout_r2_latent_mean 0.432036
rollout_r2_direct_mean 0.742028
rollout_r2_ols_mean -0.110273

| arm | mean net_profit / initial_cash |
| --- | --- |
| never_order | 0.348522 |
| random_cover | 0.305812 |
| oracle | 0.912643 |
| anchor | 1.021222 |
| order_up_to | 0.835168 |
| periodic_order_up_to | -0.030309 |

| arm | mean order volume | mean service level | mean stockout rate | mean fill rate |
| --- | --- | --- | --- | --- |
| never_order | 0.000000 | 1.000000 | 0.585385 | nan |
| random_cover | 13997.250000 | 1.000000 | 0.246437 | 0.479018 |
| oracle | 3177.187500 | 1.000000 | 0.512749 | 1.000000 |
| anchor | 12898.656250 | 1.000000 | 0.155463 | 1.000000 |
| order_up_to | 13612.562500 | 1.000000 | 0.134528 | 0.999909 |
| periodic_order_up_to | 14101.906250 | 1.000000 | 0.166642 | 0.999028 |

in_distribution_passes unavailable; latent or anchor was omitted from the parquet as a collinear arm. See comparison_warnings.json.

## Unseen SKU

Held-out rollout accuracy, mean R² over the planning horizon:
rollout_r2_latent_mean 0.432036
rollout_r2_direct_mean 0.742028
rollout_r2_ols_mean -0.110273

| arm | mean net_profit / initial_cash |
| --- | --- |
| never_order | -0.331425 |
| random_cover | 0.031361 |
| oracle | 0.385389 |
| anchor | 0.595378 |
| order_up_to | 0.414247 |
| periodic_order_up_to | -0.611453 |

| arm | mean order volume | mean service level | mean stockout rate | mean fill rate |
| --- | --- | --- | --- | --- |
| never_order | 0.000000 | 0.968750 | 0.560799 | nan |
| random_cover | 12417.875000 | 0.968750 | 0.241893 | 0.438648 |
| oracle | 3554.500000 | 0.968750 | 0.476367 | 1.000000 |
| anchor | 10952.625000 | 0.968750 | 0.157641 | 0.999883 |
| order_up_to | 11535.062500 | 0.968750 | 0.134117 | 0.999926 |
| periodic_order_up_to | 12191.437500 | 0.968750 | 0.127555 | 0.995359 |

## Unseen disruption

Training SKUs, and a harsher disruption regime the training log
never samples. This protocol is not future work.

Held-out rollout accuracy, mean R² over the planning horizon:
rollout_r2_latent_mean 0.432036
rollout_r2_direct_mean 0.742028
rollout_r2_ols_mean -0.110273

| arm | mean net_profit / initial_cash |
| --- | --- |
| never_order | -0.556331 |
| random_cover | -0.479452 |
| oracle | -0.380010 |
| anchor | -0.359757 |
| periodic_order_up_to | -0.628451 |

| arm | mean order volume | mean service level | mean stockout rate | mean fill rate |
| --- | --- | --- | --- | --- |
| never_order | 0.000000 | 1.000000 | 0.351368 | nan |
| random_cover | 4397.468750 | 1.000000 | 0.065798 | 0.691967 |
| oracle | 918.437500 | 1.000000 | 0.297757 | 1.000000 |
| anchor | 3286.187500 | 1.000000 | 0.066800 | 1.000000 |
| periodic_order_up_to | 4430.750000 | 1.000000 | 0.032903 | 0.999758 |

## Holding-rate sweep

Holding-rate sweep (0.005, 0.01, 0.02, 0.05, 0.1; same weights, no new log): at each rate the frozen planner is scored against the OrderUpToPolicy tuned at that rate. The run reports no curve when the frozen planner's order sequence is identical across those rates on the tuner holdout.

Held-out rollout accuracy, mean R² over the planning horizon:
rollout_r2_latent_mean 0.432036
rollout_r2_direct_mean 0.742028
rollout_r2_ols_mean -0.110273

sweep_rates (0.005, 0.01, 0.02, 0.05, 0.1)

### holding_rate 0.005

| arm | mean net_profit / initial_cash |
| --- | --- |
| ceiling | 1.351098 |
| latent | 0.604627 |

paired_interval -1.306742 -0.279629

### holding_rate 0.01

| arm | mean net_profit / initial_cash |
| --- | --- |
| ceiling | 1.021222 |
| latent | 0.381458 |

paired_interval -1.185105 -0.184952

### holding_rate 0.02

| arm | mean net_profit / initial_cash |
| --- | --- |
| ceiling | 0.425762 |
| latent | -0.072456 |

paired_interval -1.019959 -0.043683

### holding_rate 0.05

| arm | mean net_profit / initial_cash |
| --- | --- |
| ceiling | -1.385574 |
| latent | -1.406850 |

paired_interval -0.344415 0.426941

### holding_rate 0.1

| arm | mean net_profit / initial_cash |
| --- | --- |
| ceiling | -3.547822 |
| latent | nan |

paired_interval nan nan


## Comparison warnings

Copied from `synthetic/comparison_warnings.json`.

{
  "in_distribution": [
    "never_order and least_squares correlate at 1.0000 across seeds and are not a distinct comparison",
    "never_order and latent correlate at 0.9987 across seeds and are not a distinct comparison",
    "never_order and direct correlate at 0.9956 across seeds and are not a distinct comparison",
    "never_order and permutation correlate at 0.9987 across seeds and are not a distinct comparison",
    "never_order and fair_share correlate at 0.9987 across seeds and are not a distinct comparison",
    "least_squares and latent correlate at 0.9987 across seeds and are not a distinct comparison",
    "least_squares and direct correlate at 0.9956 across seeds and are not a distinct comparison",
    "least_squares and permutation correlate at 0.9987 across seeds and are not a distinct comparison",
    "least_squares and fair_share correlate at 0.9987 across seeds and are not a distinct comparison",
    "order_up_to and reorder_point correlate at 0.9980 across seeds and are not a distinct comparison",
    "order_up_to and periodic_reorder correlate at 0.9942 across seeds and are not a distinct comparison",
    "reorder_point and periodic_reorder correlate at 0.9929 across seeds and are not a distinct comparison",
    "latent and direct correlate at 0.9971 across seeds and are not a distinct comparison",
    "latent and permutation correlate at 1.0000 across seeds and are not a distinct comparison",
    "latent and fair_share correlate at 1.0000 across seeds and are not a distinct comparison",
    "direct and permutation correlate at 0.9972 across seeds and are not a distinct comparison",
    "direct and fair_share correlate at 0.9972 across seeds and are not a distinct comparison",
    "permutation and fair_share correlate at 1.0000 across seeds and are not a distinct comparison"
  ],
  "unseen_sku": [
    "never_order and least_squares correlate at 1.0000 across seeds and are not a distinct comparison",
    "never_order and latent correlate at 0.9997 across seeds and are not a distinct comparison",
    "never_order and direct correlate at 0.9917 across seeds and are not a distinct comparison",
    "never_order and permutation correlate at 0.9996 across seeds and are not a distinct comparison",
    "never_order and fair_share correlate at 0.9997 across seeds and are not a distinct comparison",
    "least_squares and latent correlate at 0.9997 across seeds and are not a distinct comparison",
    "least_squares and direct correlate at 0.9917 across seeds and are not a distinct comparison",
    "least_squares and permutation correlate at 0.9996 across seeds and are not a distinct comparison",
    "least_squares and fair_share correlate at 0.9997 across seeds and are not a distinct comparison",
    "anchor and periodic_reorder correlate at 0.9906 across seeds and are not a distinct comparison",
    "order_up_to and reorder_point correlate at 0.9974 across seeds and are not a distinct comparison",
    "reorder_point and periodic_reorder correlate at 0.9905 across seeds and are not a distinct comparison",
    "latent and direct correlate at 0.9929 across seeds and are not a distinct comparison",
    "latent and permutation correlate at 1.0000 across seeds and are not a distinct comparison",
    "latent and fair_share correlate at 1.0000 across seeds and are not a distinct comparison",
    "direct and permutation correlate at 0.9927 across seeds and are not a distinct comparison",
    "direct and fair_share correlate at 0.9928 across seeds and are not a distinct comparison",
    "permutation and fair_share correlate at 0.9999 across seeds and are not a distinct comparison"
  ],
  "unseen_disruption": [
    "never_order and least_squares correlate at 1.0000 across seeds and are not a distinct comparison",
    "never_order and latent correlate at 0.9971 across seeds and are not a distinct comparison",
    "never_order and direct correlate at 0.9976 across seeds and are not a distinct comparison",
    "never_order and permutation correlate at 0.9971 across seeds and are not a distinct comparison",
    "never_order and fair_share correlate at 0.9971 across seeds and are not a distinct comparison",
    "least_squares and latent correlate at 0.9971 across seeds and are not a distinct comparison",
    "least_squares and direct correlate at 0.9976 across seeds and are not a distinct comparison",
    "least_squares and permutation correlate at 0.9971 across seeds and are not a distinct comparison",
    "least_squares and fair_share correlate at 0.9971 across seeds and are not a distinct comparison",
    "anchor and order_up_to correlate at 0.9965 across seeds and are not a distinct comparison",
    "anchor and reorder_point correlate at 0.9972 across seeds and are not a distinct comparison",
    "anchor and periodic_reorder correlate at 0.9972 across seeds and are not a distinct comparison",
    "order_up_to and reorder_point correlate at 0.9987 across seeds and are not a distinct comparison",
    "order_up_to and periodic_reorder correlate at 0.9976 across seeds and are not a distinct comparison",
    "reorder_point and periodic_reorder correlate at 0.9990 across seeds and are not a distinct comparison",
    "latent and direct correlate at 0.9991 across seeds and are not a distinct comparison",
    "latent and permutation correlate at 1.0000 across seeds and are not a distinct comparison",
    "latent and fair_share correlate at 1.0000 across seeds and are not a distinct comparison",
    "direct and permutation correlate at 0.9991 across seeds and are not a distinct comparison",
    "direct and fair_share correlate at 0.9991 across seeds and are not a distinct comparison",
    "permutation and fair_share correlate at 1.0000 across seeds and are not a distinct comparison"
  ],
  "holding_rate_sweep": [
    "holding_rate 0.1: ceiling and latent correlate at 0.9984 across seeds and are not a distinct comparison"
  ]
}

## M5

The published command is:

```bash
uv run jupyter nbconvert --to notebook --execute --output-dir runs/jepa-replenishment notebooks/11-jepa-m5.ipynb
```

ADR 0028: choose five items on 2015-01-01 through 2015-12-31 only. Rank FOODS_3 at CA_1 by shortest longest_zero_run, then highest mean sales. Drop a zero run longer than 7 days. If fewer than five items survive, widen to all FOODS departments at CA_1 under the same rule. The test window is 2016-01-01 through 2016-04-24.

The graph is one `StaticFactoryPolicy` factory feeding `shop-CA_1`
at lead time 3, and one replay sink per item. `flat_world` is on.
Both the learned policy and `OrderUpToPolicy` are wrapped in the same
`PriceReplayPolicy`. Zero-shot scores the synthetic weights and does
not fit on either window. Adaptation refits the same architecture on
reward-free transitions from the 2015 window only. The test window is
2016. The windows are not pooled. The textbook anchor is tuned on 2015
replay profit only. Sales are censored, and `unit_cost` is a fiction:
40% of median price. Neither row is a pass/fail gate.

PINNED_ITEM_IDS is empty until a quality report runs.

This run has no `m5/exam.parquet`, so there is no zero-shot row
and no adaptation row.

## Future work

Future work, deferred rather than forgotten:

- pricing policies, with ordering held fixed
- a joint multi-SKU model
- policy control of engine Allocation
