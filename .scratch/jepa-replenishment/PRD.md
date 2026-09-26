# PRD: Latent replenishment planner

Status: ready-for-agent

Related ADRs:

- [ADR 0006](../../docs/adr/0006-textbook-reorder-policy-family.md) — Textbook reorder family. `OrderUpToPolicy` is the comparison anchor. Pricing stays out of the comparison.
- [ADR 0009](../../docs/adr/0009-policy-hyperparameter-tuning-tool.md) — Optuna tuner. The published anchor is a tuned `OrderUpToPolicy`, scored as mean `net_profit / initial_cash`.
- [ADR 0010](../../docs/adr/0010-sim-as-base-for-ml-layers.md) — New ML layers are sibling consumers of the simulator. They do not fork the tick loop and they do not import the tuner or the RL stack.
- [ADR 0019](../../docs/adr/0019-charge-holding-order-fee-and-flow-logged-metrics.md) — Holding cost and order fee are charged to the intermediate. Profit is derived from the run log.
- [ADR 0020](../../docs/adr/0020-replay-demand-sinks.md) — Replay sinks are an exam. Widened for this study by ADR 0026.
- [ADR 0024](../../docs/adr/0024-ordering-only-learned-policy.md) — The learned Policy orders only. Pricing policies are later work.
- [ADR 0025](../../docs/adr/0025-priority-ration.md) — The learned Policy rations by priority. Engine Allocation is unchanged.
- [ADR 0026](../../docs/adr/0026-replay-window-adaptation.md) — Zero-shot M5 is the headline. An earlier-window fit is a separate row.
- [ADR 0027](../../docs/adr/0027-frozen-holding-rate-retarget.md) — The holding-rate retarget re-scores a frozen predictor.
- [ADR 0028](../../docs/adr/0028-m5-slice.md) — One `CA_1` shop, five FOODS items chosen on 2015, factory lead time 3.
- [ADR 0029](../../docs/adr/0029-probe-stop-gradient.md) — Probe losses do not train the encoder.

## Problem Statement

The simulator can score a textbook **Policy** on synthetic episodes and on replayed Walmart sales, but it cannot answer the question this study exists to ask: if a reward-free model learns, from the simulator alone, how one shop's inventory moves, can that model plan replenishment orders that hold up against a tuned `OrderUpToPolicy`, adapt when holding cost changes without a new fit, and serve a real demand stream?

The PPO stack is not a usable answer. It does not show a real improvement and would have to be restructured before it could be a baseline. This study does not use it.

A hand-rolled comparison would also miss the point of the simulator. The planner has to be an `IntermediatePolicy`, attached the same way every other policy is attached, and scored with the profit the engine already charges. Otherwise a win is a win about a private reward, not about the simulator.

## Solution

Add a latent replenishment planner. It is an `IntermediatePolicy` on the one intermediate that faces the sinks. Upstream factories stay on `StaticFactoryPolicy`.

A per-SKU joint-embedding predictive model is trained offline on simulator rollouts. It never sees the **Market** multiplier or which disruption is active. It predicts its own future embedding given the quantity that SKU was granted. Probes decode on-hand, in-transit, and sales. The planner scores the engine's `net_profit` from those quantities, searches a small cover-and-priority action, and rations scarce capacity and cash by priority. `OrderUpToPolicy` keeps its fair-share ration, so a gap can be a better **ration** rather than a different price.

Three published protocols, plus the ablations inside them:

1. **In-distribution.** Tuner holdout seeds, `holding_rate = 0.01`. Success is a paired gap against Optuna-tuned `OrderUpToPolicy` whose confidence interval covers zero, or a planner mean of at least 95% of that anchor. A win is not required.
2. **Holding-rate retarget.** Same seeds, `holding_rate = 0.05`. The predictor stays frozen. Imagined cash uses the new rate and feeds the ration. Success is beating the `OrderUpToPolicy` that was tuned at `0.01` and then left frozen. An `OrderUpToPolicy` retuned at `0.05` is the ceiling. Matching it is not required.
3. **M5 replay.** `shop-CA_1`, one synthetic factory, `flat_world`, both ordering policies wrapped in the same `PriceReplayPolicy`. The headline model never trained on Walmart sales. A second row refits the predictor on calendar 2015 and is tested on 2016-01-01 through 2016-04-24. Both rows are reported either way.

Notebooks explain the process and read the result tables. The root README states the exact commands. A markdown paper draft states the method from these decisions and fills result tables only from run artifacts.

## User Stories

1. As an inventory researcher, I want a learned replenishment Policy that orders only, so that a gap against `OrderUpToPolicy` is a replenishment result rather than a pricing result.
2. As an inventory researcher, I want that Policy to set `list_price` from `base_price` on every product, so that both arms face the synthetic price elasticity the same way.
3. As an inventory researcher, I want the Policy to emit no promotions and no learned minimum-order floor, so that the action is a cover and a ration.
4. As an inventory researcher, I want each SKU's order to be `max(0, cover − position)` every tick, with position equal to on-hand plus in-transit, so that the action is base-stock in lead times of recent demand.
5. As an inventory researcher, I want "recent demand" to be the textbook censored-sales rate over `delivery_lag`, so that a cover of two lead times means the same quantity the textbook family means.
6. As an inventory researcher, I want the cover grid to be `{0, 1, 2, 4}` lead times, so that planning stays inside the action set the training log contains.
7. As an inventory researcher, I want the learned Policy to ration free capacity and cash by priority, so that a fast seller can take the whole pool when space is tight.
8. As an inventory researcher, I want `OrderUpToPolicy` to keep fair-share, so that the textbook arm is unchanged.
9. As an inventory researcher, I want engine **Allocation** to stay `execute_buy`, so that the policy does not take over supplier fill.
10. As an inventory researcher, I want one learned buyer, the intermediate facing the sinks, so that the study uses the tuner graph instead of inventing a new topology.
11. As an inventory researcher, I want factories to stay on `StaticFactoryPolicy`, so that production is not a second learned problem.
12. As an inventory researcher, I want the world model to be per SKU, so that a joint model of the whole assortment is not required for version 1.
13. As an inventory researcher, I want the model conditioned on the quantity the ration actually granted, so that it learns the transition of a feasible order.
14. As an inventory researcher, I want the encoder to see on-hand, in-transit, a sales window of length `delivery_lag`, cash, and capacity, so that it lives on the observation a Policy already receives.
15. As an inventory researcher, I want the **Market** multiplier and the disruption identity kept out of that observation, so that the model has to infer the demand regime from sales.
16. As an inventory researcher, I want the predictor to roll its own embedding forward for the edge lead time, open-loop, so that the training loss matches the plan the policy will make.
17. As an inventory researcher, I want an anti-collapse penalty on the embeddings, so that a constant embedding cannot drive the prediction loss to zero.
18. As an inventory researcher, I want probes that decode next on-hand, next in-transit, and sales over the step, so that profit uses the engine's accounting identity.
19. As an inventory researcher, I want probe losses to update the probes only, so that the encoder remains a latent predictor and the direct-predictor ablation stays a real control.
20. As an inventory researcher, I want the planner's profit calculation kept out of the encoder's gradients, so that the model is not trained as a reward model.
21. As an inventory researcher, I want imagined profit to be revenue minus order cost minus holding cost minus order fees, so that the score is the engine's `net_profit`.
22. As an inventory researcher, I want no separate stockout penalty, so that lost sales appear only as revenue that did not arrive.
23. As an inventory researcher, I want `order_fee` left at the tuner default, so that the retarget lever is `holding_rate` alone.
24. As an inventory researcher, I want `decide` to draw 64 proposals from `policy_rng`, each a cover per SKU plus a priority order, so that the search finishes.
25. As an inventory researcher, I want the winning proposal's granted quantities to be the order, so that the engine receives one feasible ration.
26. As an inventory researcher, I want the same `policy_seed` to place the same orders on a repeated run, so that paired comparisons are reproducible.
27. As an inventory researcher, I want the world seed kept out of the search, so that the policy's randomness is not the world's randomness.
28. As an inventory researcher, I want a direct predictor of on-hand, in-transit, and sales, with the same shooter and the same ration, so that a win can be attributed to the latent objective.
29. As an inventory researcher, I want a fair-share arm that takes the latent model's covers and rations them with the textbook fair-share, so that a gain can be credited to the priority ration rather than to the cover.
30. As an inventory researcher, I want the synthetic training log to be half published-default `OrderUpToPolicy` and half random covers from that grid with random priorities, so that the log contains both sensible inventories and non-fair-share rations.
31. As an inventory researcher, I want that log collected on the tuner episode shape, 365 ticks, five active SKUs, lead time 3, so that training and evaluation share one world.
32. As an inventory researcher, I want training-log seeds disjoint from the tuner holdout seeds, so that the published eval is not a replay of the training episodes.
33. As an inventory researcher, I want no Optuna-tuned policy and no periodic or `(s, Q)` policy inside the training log, so that collection does not wait on a study and the log matches the action the planner emits.
34. As an inventory researcher, I want the in-distribution protocol on the tuner holdout seeds at `holding_rate = 0.01`, so that the sanity check uses the instrument the repo already trusts.
35. As an inventory researcher, I want the comparison anchor to be an Optuna-tuned `OrderUpToPolicy`, so that the planner is not praised for beating an untuned default.
36. As an inventory researcher, I want the published-default `OrderUpToPolicy` in the same table, so that the value of tuning is visible.
37. As an inventory researcher, I want the other three textbook policies at their published defaults, not retuned, so that the ladder is cheap and still present.
38. As an inventory researcher, I want the in-distribution success rule to be a paired confidence interval covering zero or a mean of at least 95% of the tuned anchor, so that a small gap is a pass and a win is not required.
39. As an inventory researcher, I want the retarget to keep the predictor trained at `holding_rate = 0.01`, so that the model is not allowed to see the new economy.
40. As an inventory researcher, I want imagined cash rolled forward at `holding_rate = 0.05` and fed back into the ration, so that a cover the till can no longer afford is not proposed.
41. As an inventory researcher, I want the retarget success rule to be the frozen planner beating the frozen tuned `OrderUpToPolicy` on paired `net_profit / initial_cash`, so that the claim is adaptation without a new fit.
42. As an inventory researcher, I want an `OrderUpToPolicy` retuned at `holding_rate = 0.05` reported as a ceiling, so that the table shows how much a fresh search can still gain.
43. As an inventory researcher, I want both M5 numbers, zero-shot and adaptation, so that transfer and "served this store's history" are not collapsed into one claim.
44. As an inventory researcher, I want the M5 windows kept unpooled, so that the test year is not in the adaptation fit.
45. As an inventory researcher, I want the M5 graph to be one factory feeding `shop-CA_1` at lead time 3, so that the horizon matches the synthetic training horizon.
46. As an inventory researcher, I want five FOODS items chosen on 2015 only, shortest zero-run then highest mean sales, dropping a zero-run longer than 7 days, so that censored series are not the exam.
47. As an inventory researcher, I want the pool widened to all FOODS departments at `CA_1` only when `FOODS_3` yields fewer than five items, so that the rule is deterministic.
48. As an inventory researcher, I want those five ids pinned after the first quality report, so that a later run does not reshuffle the slice.
49. As an inventory researcher, I want the adaptation window to be 2015-01-01 through 2015-12-31 and the test window to be 2016-01-01 through 2016-04-24, so that the test fits both M5 sales files.
50. As an inventory researcher, I want `unit_cost` left at 40% of median price, so that the cost fiction stays the adapter's documented default.
51. As an inventory researcher, I want `flat_world` on the M5 arm, so that replayed demand is not multiplied by a synthetic seasonality the series already contains.
52. As an inventory researcher, I want both the learned Policy and `OrderUpToPolicy` wrapped in the same `PriceReplayPolicy` on the M5 arm, so that both charge the observed Walmart price.
53. As an inventory researcher, I want the M5 textbook anchor tuned only on the 2015 window, so that the test window is not used to pick its hyperparameters.
54. As a notebook reader, I want a training notebook that shows how the log is built, how the model is fit, and how one `decide` becomes an order, so that I can rerun the process interactively.
55. As a notebook reader, I want a synthetic-results notebook that loads the in-distribution and retarget tables and states the success rules next to the numbers, so that I do not have to reconstruct the claim from a parquet file.
56. As a notebook reader, I want an M5 notebook that runs the slice rule, the zero-shot row, and the adaptation row when the raw files are present, so that the real-data arm is the same kind of walkthrough.
57. As a notebook reader without the Kaggle files, I want the M5 notebook to skip with the download command and the slice rule, so that opening it is not an error.
58. As a notebook reader whose full study has not been run, I want the results notebook to say which command fills the tables, so that empty artifacts are explained rather than plotted as zeros.
59. As a new user, I want the root README to give the exact commands for the log, the fit, the two tuning studies, the synthetic eval, and the M5 eval, so that I can run the experiment without reading the tickets.
60. As a paper reader, I want a markdown draft whose method section matches these decisions, so that the writeup cannot drift from the study that was run.
61. As a paper reader, I want result tables copied from run artifacts and nowhere else, so that the draft never contains invented improvements.
62. As a paper reader, I want the M5 section to keep the zero-shot row and the adaptation row separate, and to state that sales are censored and `unit_cost` is a fiction, so that the real-data claim stays honest.
63. As a paper reader, I want future work to name pricing policies, a joint multi-SKU model, and policy control of engine Allocation, so that those omissions are recorded as deferred rather than forgotten.
64. As a CI maintainer, I want ticket tests to be smoke runs on a tiny log, so that the suite does not launch Optuna or a 365-tick shooting eval.
65. As a CI maintainer, I want ticket tests to pass without `data/m5`, so that the Kaggle files are not a dependency of the build.
66. As a repo maintainer, I want the full synthetic protocols run once, at the end, writing the tables the notebook and the paper read, so that the published numbers exist and are not re-computed inside every ticket.
67. As a repo maintainer, I want raw logs, checkpoints, and per-seed tables under the ignored runs directory, so that bulky artifacts are not committed.
68. As a repo maintainer, I want the paper markdown and the notebooks tracked, so that the method and the walkthrough stay in the repository.
69. As a repo maintainer, I want the new package to import the simulator and not the RL stack, so that a broken PPO trainer cannot become a dependency of this study.
70. As a repo maintainer, I want the PPO stack left untouched, so that this study does not become a restructuring of reinforcement learning.

## Implementation Decisions

- **One learned buyer.** The Policy attaches to the single intermediate on the tuner graph (factories, that intermediate, one sink per active product). The M5 graph is one `StaticFactoryPolicy` factory, `shop-CA_1`, and one `ReplayDemandSinkNode` per pinned item. No second learned echelon. No lateral link.
- **Ordering only.** `decide` returns `list_price` from `base_price`, an empty promotion map, and a zero minimum-order floor. On M5, both ordering policies are wrapped in one `PriceReplayPolicy` under `flat_world`. Pricing as an action is future work (ADR 0024).
- **Cover action.** Inventory position is on-hand plus in-transit. The rate is the textbook censored-sales estimator over `delivery_lag`. The order before the ration is `max(0, round(cover × rate × delivery_lag) − position)`, applied every tick. Covers are chosen from `{0, 1, 2, 4}`.
- **Priority ration.** Each proposal carries a priority order over the active SKUs. A greedy fill grants units in that order until free capacity or cash is exhausted. The world model sees the granted quantity. `OrderUpToPolicy` continues to use fair-share. Engine Allocation is not modified. A joint latent over all SKUs is future work (ADR 0025).
- **What is trained.** A per-SKU encoder maps the observation window to an embedding. An exponential-moving-average target encoder supplies the regression target. An action-conditioned predictor unrolls open-loop for `H = delivery_lag` steps. The latent loss is the distance to the stopped target at each step. An anti-collapse penalty keeps embedding variance off the floor and penalizes cross-dimension correlation. Probes read a stopped embedding and regress next on-hand, next in-transit, and step sales. Probe losses and the profit arithmetic do not update the encoder (ADR 0029).
- **Observation window.** Per SKU: on-hand, in-transit, and the last `delivery_lag` observed sales. Broadcast onto every SKU: cash and capacity. The Market multiplier and the disruption type are not features.
- **Profit used for planning.** Over the horizon, revenue minus purchase cost minus `holding_rate × on-hand × unit_cost` minus the per-supplier order fee, using the probed quantities and the rate under test. Cash is updated with that same identity and becomes the cash the next imagined ration sees. No learned reward head. No stockout penalty.
- **Search.** Sixty-four proposals per tick, drawn from `policy_rng` seeded by `policy_seed`. Each proposal is one cover per SKU and one priority permutation. The winner is the proposal with the highest imagined `net_profit`. The direct-predictor arm uses the same shooter. The fair-share arm shoots covers only and rations with textbook fair-share.
- **Training log.** Half the episodes run published-default `OrderUpToPolicy`. Half run the random cover-and-priority behavior. Episode shape matches the tuner: 365 ticks, `K_active = 5`, `delivery_lag = 3`, `holding_rate = 0.01`, default `order_fee`. Published log size is 64 episodes of each behavior. Log seeds use offset `14_000_000`, disjoint from the tuner search offset `12_000_000` and holdout offset `13_000_000`.
- **Direct predictor.** Same probes' targets, predicted from the observation window and the granted quantity with no latent and no EMA target. Same planner, same search budget, same ration.
- **In-distribution eval.** Tuner holdout seeds (`n_holdout_seeds = 32`, offset `13_000_000`), `holding_rate = 0.01`. Arms: latent planner, direct planner, latent covers with fair-share, Optuna-tuned `OrderUpToPolicy`, published-default `OrderUpToPolicy`, and the other three textbook policies at published defaults. Report paired `net_profit / initial_cash` and a paired confidence interval. Pass if the interval against the tuned anchor covers zero, or the latent mean is at least 95% of the tuned mean.
- **Retarget eval.** Same holdout seeds, `holding_rate = 0.05`, weights from the `0.01` fit, no new log (ADR 0027). Arms: frozen latent planner, the `OrderUpToPolicy` tuned at `0.01`, and an `OrderUpToPolicy` tuned at `0.05`. Pass if the frozen planner beats the frozen textbook policy on paired profit. The retuned textbook policy is reported and is not a pass/fail bar.
- **Tuning studies.** Two studies, both `OrderUpToPolicy`, both through the existing tuner. One at `holding_rate = 0.01`, one at `0.05`. Published-run trial count is the tuner default (150). The other three textbook policies are not tuned.
- **M5.** Selection and dates are ADR 0028. Zero-shot uses the synthetic weights. Adaptation refits the same architecture on reward-free transitions collected on the 2015 replay graph, from the same two behavior policies, and is tested on the 2016 window. The textbook anchor for both rows is `OrderUpToPolicy` tuned on 2015 replay profit only. `unit_cost_fraction = 0.4`. Item ids are written into the M5 notebook and the paper the first time the quality report runs.
- **Smoke versus published.** Ticket tests use a handful of ticks, two episodes, and a search budget of four proposals. The published seed counts, 64-proposal search, and 150-trial studies run once in the final milestone.
- **PPO.** No import, no baseline, no checkpoint, no change to the RL package.

## Testing Decisions

Good tests check behavior visible on a `Runner` run log and on `net_profit / initial_cash`. They do not assert latent width, EMA decay, or loss values.

The seam is the one the tuner already uses. Build a scenario, attach a Policy with `policy_overrides`, run it, and read the run log and the business metrics. Prior art is the tuner evaluator tests, the textbook policy tests, the determinism tests, and the M5 replay tests that skip or fake a tiny series rather than reading Kaggle files.

Smoke tests, all without `data/m5`:

- A learned Policy on a tiny episode returns orders, sets `list_price` from `base_price`, and grants quantities that fit in capacity and cash.
- Two runs with the same `policy_seed` and the same world seed place the same orders. Two runs that differ only in `policy_seed` may differ. Changing the world seed alone does not change the sequence of proposals.
- On a fixture where the till still covers every proposal at `holding_rate = 0.01` and empties under `0.05`, a frozen Policy places a different order at the higher rate.
- A tiny log produced by `Runner` can be fit, and the resulting Policy still satisfies the order-dict checks. One held-out seed returns a finite paired profit.
- Fitting the probes does not change encoder parameters. This is the only test that looks inside the module, because ADR 0029 is invisible on the run log.
- The direct predictor and the fair-share arm both return a valid order dict through the same seam.

The published synthetic run and the M5 notebook are not ticket tests. The M5 notebook's skip path, given a missing raw directory, is tested by asserting it names the download and the slice rule rather than raising.

## Out of Scope

- Any comparison with the PPO stack, and any change to that stack.
- Pricing as an action, promotions, or a learned minimum-order floor. Recorded as future work.
- A joint latent model of every SKU, and any policy control of engine Allocation.
- A second learned echelon, lateral links, or the M5 demo's three-shop distribution-centre graph.
- A harsher held-out disruption protocol. Default episodes already draw mild disruptions.
- Periodic and `(s, Q)` policies inside the training log. They remain evaluation-only, at published defaults.
- Refitting or recollecting data at `holding_rate = 0.05`.
- Pooling the 2015 and 2016 M5 windows, or training the headline model on Walmart sales.
- Committing raw M5 files, generated setup dirs under `data/`, or run artifacts under `runs/`.
- A learned reward head, a stockout penalty, or probe gradients into the encoder.
- Claiming a pass on M5. Both rows are reported. Neither is a gate.

## Further Notes

### Artifact homes

| Artifact | Location | Tracked |
|---|---|---|
| Latent replenishment package, including its README | `src/jepa/` | yes |
| Root README section with the exact commands | `README.md` | yes |
| Training walkthrough notebook | `notebooks/09-jepa-train-and-plan.ipynb` | yes |
| Synthetic in-distribution and retarget notebook | `notebooks/10-jepa-synthetic-eval.ipynb` | yes |
| M5 notebook | `notebooks/11-jepa-m5.ipynb` | yes |
| Paper draft | `docs/jepa-replenishment.md` | yes |
| Logs, checkpoints, tuning studies, per-seed tables | `runs/jepa-replenishment/` | no (`runs/` is gitignored) |
| Raw M5 files and emitted setup dirs | `data/m5/` and `data/m5_jepa/` | no (`data/` is gitignored) |

The final milestone copies summary tables into `docs/jepa-replenishment.md` from `runs/jepa-replenishment/`. It does not invent a cell. If the M5 raw files are absent, that section of the paper contains the command and the ADR 0028 slice rule, and the notebook skips.

### Publication note

The claim worth writing down is the frozen holding-rate retarget and the split M5 result, against a tuned textbook anchor, with the direct predictor and the fair-share arm in the table. An in-distribution win over tuned `OrderUpToPolicy` is not the claim. Venues that want a structural inventory proof, and venues that want a PPO or Dreamer bake-off, are the wrong homes for this draft. The draft should say that in one paragraph so a later submission does not quietly add those baselines.

### Future work, to appear in the paper

Pricing policies, with ordering held fixed. A joint multi-SKU dynamics model. Policy control of engine Allocation. A harsher disruption protocol held out of training.
