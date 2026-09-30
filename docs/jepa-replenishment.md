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
echelon.

The world model is per SKU. It is conditioned on the quantity
Allocation filled. The planner imagines that quantity as the grant
times the log's fill rate, and the emitted order stays the full grant.
The encoder sees on-hand, an arrival-age
profile, a sales window of its own length rather than the delivery
lead time, cash, capacity, and that lead time. The Market
multiplier, the disruption identity, and the calendar are not
features. An
exponential-moving-average target encoder supplies the regression
target. The predictor rolls its embedding forward for the edge lead
time, open-loop, and the latent loss is the distance to that stopped
target at each step. An anti-collapse penalty keeps embedding variance
off the floor and penalizes cross-dimension correlation. Probes decode
next on-hand, the next arrival-age profile, and sales over the step.
The planner sums that profile for inventory position. Probe losses
update the probes only. The planner's profit calculation does not
update the encoder. Imagined profit is revenue minus order cost minus
holding cost minus order fees, which is the engine's `net_profit`.
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
The log is collected at a single holding rate. Untuned textbook policies
stay at published defaults on the holdout table. The in-distribution protocol uses
the tuner holdout at `holding_rate = 0.01`. The anchor is an
`OrderUpToPolicy` tuned at that rate. The retarget keeps the predictor
trained at `holding_rate = 0.01`. Imagined cash is rolled forward at
`holding_rate = 0.05` and fed back into the ration. The weights are
not refit, and behavior data is not recollected. An `OrderUpToPolicy`
tuned at `holding_rate = 0.05` is reported as a ceiling.

## Caveats

Sales are censored. The censored-sales rate treats a stockout as low
demand, so a cover learned from that rate is not a cover of true
demand. Lost sales appear only as revenue that did not arrive.

On the M5 arm, `unit_cost` is 40% of median price. That figure is the
adapter's documented fiction, not a Walmart cost. Both ordering
policies would face the same fiction, inside one `PriceReplayPolicy`,
under `flat_world`.

## Success rules

In-distribution rule (holding_rate 0.01, tuner holdout): pass if the paired interval of latent minus the anchor covers zero, or the latent mean is at least 95% of the anchor mean.

Retarget rule (holding_rate 0.05, same weights, no new log): pass if the frozen planner beats the frozen OrderUpToPolicy from the 0.01 study on paired net_profit / initial_cash. The OrderUpToPolicy tuned at 0.05 is a ceiling and is not a pass bar.

The retuned textbook policy is reported and is not a pass bar. An M5
row is not a gate.

## Where this draft does not belong

The claim worth writing down is the frozen holding-rate retarget and
the split M5 result, against a tuned textbook anchor, with the direct
predictor and the fair-share arm in the table. Venues that want a
structural inventory proof, and venues that want a PPO or Dreamer
bake-off, are the wrong homes for this draft.

## Run record

These lines are copied from the log, `checkpoints/planner.json`, and
each `study.json`. `best_value` is that file's search objective on the
search seeds. It is not the holdout table below. The transition log
omits the final tick of each episode, because that tick has no next
state. `episode_length` is the length stored on the study.

exploratory_episodes 64
order_up_to_episodes 64
log_seed_min 14000000
log_seed_max 14000127
products_per_episode 5
recorded_tick_min 1
recorded_tick_max 364
recorded_tick_count 364
delivery_lag 3
budget 64
fit_holding_rate 0.010000
order_fee 50.000000
study order_up_to_holding_0_005 n_trials 150
study order_up_to_holding_0_005 n_search_seeds 16
study order_up_to_holding_0_005 n_holdout_seeds 32
study order_up_to_holding_0_005 episode_length 365
study order_up_to_holding_0_005 seed_offset 12000000
study order_up_to_holding_0_005 holdout_seed_offset 13000000
study order_up_to_holding_0_005 best_value 1.9102880646287577
study order_up_to_holding_0_005 git_sha 6fda2d43ce638d5e8f3788d299eb3816d2e6371b
study order_up_to_holding_0_005 best_params {"cover_horizon_ticks": 12, "safety_lead_pct_of_lag": 0.05025909338623214, "stockout_safety_bonus_pct_of_lag": 0.19108043943834024}
study order_up_to_holding_0_01 n_trials 150
study order_up_to_holding_0_01 n_search_seeds 16
study order_up_to_holding_0_01 n_holdout_seeds 32
study order_up_to_holding_0_01 episode_length 365
study order_up_to_holding_0_01 seed_offset 12000000
study order_up_to_holding_0_01 holdout_seed_offset 13000000
study order_up_to_holding_0_01 best_value 1.0641955949975148
study order_up_to_holding_0_01 git_sha 6fda2d43ce638d5e8f3788d299eb3816d2e6371b
study order_up_to_holding_0_01 best_params {"cover_horizon_ticks": 12, "safety_lead_pct_of_lag": 0.16993287200409285, "stockout_safety_bonus_pct_of_lag": 0.09706576917593086}
study order_up_to_holding_0_02 n_trials 150
study order_up_to_holding_0_02 n_search_seeds 16
study order_up_to_holding_0_02 n_holdout_seeds 32
study order_up_to_holding_0_02 episode_length 365
study order_up_to_holding_0_02 seed_offset 12000000
study order_up_to_holding_0_02 holdout_seed_offset 13000000
study order_up_to_holding_0_02 best_value -0.5221367401283703
study order_up_to_holding_0_02 git_sha 6fda2d43ce638d5e8f3788d299eb3816d2e6371b
study order_up_to_holding_0_02 best_params {"cover_horizon_ticks": 7, "safety_lead_pct_of_lag": 0.12627988145408958, "stockout_safety_bonus_pct_of_lag": 0.06893640026208986}
study order_up_to_holding_0_05 n_trials 150
study order_up_to_holding_0_05 n_search_seeds 16
study order_up_to_holding_0_05 n_holdout_seeds 32
study order_up_to_holding_0_05 episode_length 365
study order_up_to_holding_0_05 seed_offset 12000000
study order_up_to_holding_0_05 holdout_seed_offset 13000000
study order_up_to_holding_0_05 best_value -4.994853343225005
study order_up_to_holding_0_05 git_sha 6fda2d43ce638d5e8f3788d299eb3816d2e6371b
study order_up_to_holding_0_05 best_params {"cover_horizon_ticks": 3, "safety_lead_pct_of_lag": 0.217201280660895, "stockout_safety_bonus_pct_of_lag": 0.24606803342018368}
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
retarget_seeds 32
retarget_seed_min 13000000
retarget_seed_max 13000031
Means and the paired interval below are printed to 6 decimal places.

## In-distribution

In-distribution rule (holding_rate 0.01, tuner holdout): pass if the paired interval of latent minus the anchor covers zero, or the latent mean is at least 95% of the anchor mean.

The in-distribution claim is that rule.

Per-seed rows stay in `synthetic/in_distribution.parquet`.
The `anchor` column is the tuned `OrderUpToPolicy` when that study
was passed to eval. `order_up_to` is the published-default policy.

| arm | mean net_profit / initial_cash |
| --- | --- |
| latent | 0.363801 |
| direct | 0.366432 |
| fair_share | 0.362479 |
| order_up_to | 3.034964 |
| reorder_point | 3.035251 |
| periodic_order_up_to | 2.693331 |
| periodic_reorder | 2.893932 |
| anchor | 3.031103 |

paired_interval -3.897324 -1.743538
in_distribution_passes fail

## Retarget

Retarget rule (holding_rate 0.05, same weights, no new log): pass if the frozen planner beats the frozen OrderUpToPolicy from the 0.01 study on paired net_profit / initial_cash. The OrderUpToPolicy tuned at 0.05 is a ceiling and is not a pass bar.

Per-seed rows stay in `synthetic/retarget.parquet`.
`ceiling` is the `OrderUpToPolicy` tuned at holding rate 0.05.
It is reported and it is not a pass bar.

| arm | mean net_profit / initial_cash |
| --- | --- |
| latent | 0.162079 |
| anchor | 2.473266 |
| ceiling | 2.551653 |

retarget_passes fail

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
- a harsher held-out disruption, kept out of the training episodes
