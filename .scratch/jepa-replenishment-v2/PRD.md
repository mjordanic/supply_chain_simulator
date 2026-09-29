# Latent replenishment, second attempt

Status: ready-for-agent

Supersedes the study published in `docs/jepa-replenishment.md` and the
feature folder `.scratch/jepa-replenishment/`.

## Problem Statement

The first latent-replenishment study reported that the learned buyer earned
`net_profit / initial_cash` of 0.364 against a tuned `OrderUpToPolicy` anchor at
3.031, and both pass rules failed. Those numbers are not a result about the
model. An audit established that the experiment could not distinguish a good
world model from a bad one, so nothing about the architecture can be concluded
from it.

The audit found three independent defects, each individually fatal.

**The planner's scoring horizon equals the delivery lead time.** An order is
charged to imagined cash inside the window while the units it buys arrive at or
after the window's edge, so the profit-maximising proposal is always the one
that orders least. Substituting a perfect world model — exact inventory
recursion with correct arrival timing — into the unmodified planner scores
0.336 against the learned model's 0.333 on the same seeds. The world model is
irrelevant to the outcome. The learned buyer placed orders on 7 of 365 ticks,
94 units for a year, against roughly 8,900 units per year for the random
behaviour policy that generated its own training data. A search that loses by
8x to uniform sampling from its own action set has a broken objective, not a
weak model.

**The world models are untrained.** `fit(epochs=N)` performs exactly `N`
full-batch Adam steps; there is no minibatching and no inner pass over the
data, so "epoch" names a single gradient step. The published command used the
default of 20. Measured on a held-out episode split, the shipped checkpoint
scores R² of −0.003 / −0.016 / +0.004 on next on-hand, next in-transit and next
sales — worse than predicting the training mean, and against +0.94 / +0.75 /
+0.65 for ordinary least squares on identical inputs. Its output standard
deviation is 1.89 against a true 114.58, so it emits a near-constant. Raising
the step count to 1500 reaches +0.91 / +0.77 / +0.72. With `_EMA_TAU = 0.99`
over 20 updates the target encoder remains roughly 85% its random
initialisation, and that random projection supplies both the predictor's
regression target and the probe inputs.

**The world is degenerate and the benchmark has no headroom.** `Market` adds the
seasonal cycle to the demand *level* each tick rather than modulating it, so the
level integrates the sine wave and pins against its clamps. Demand sits at the
`max_value` ceiling for roughly ticks 10–190, collapses between 190 and 207, and
sits at the `demand_factor_min` floor for the remaining 158 ticks. Across 128
episodes the collapse tick has median 207 and only 13 distinct values, so the
world seed barely matters. Fifty-three percent of every training episode is a
dead world. Consequently capacity and cash never bind, service level is 1.000,
random covers score 3.004 against the tuned anchor's 3.031, and a 150-trial
Optuna study bought a headline uplift of −9.35 with a confidence interval
spanning zero. No method can be shown to beat any other on a benchmark where
tuning buys nothing.

Three further defects make the reported table unreadable even setting the above
aside. The three learned arms — latent, direct, and fair-share ration —
correlate at r = 0.9999 across seeds with a maximum absolute difference of
0.039, so both advertised ablations are vacuous. The frozen holding-rate
retarget, which the paper names as its headline claim, is a literal no-op: the
frozen planner places byte-identical orders at holding rates 0.01 and 0.05, and
the reported drop from 0.363 to 0.162 is the world charging more for unchanged
behaviour. And the anchor is handed an advantage the learned buyer does not
get — `MultiSupplierTextbookPolicy` reasons with `unit_cost = 1.0` against a
true supplier price near 11, which disables its internal budget check and lets
it over-request and rely on the Allocation clamp, while the planner self-clamps
honestly.

Underneath all of it sits a process gap: `fit` reports no loss, writes no
validation metric and has no early stopping; the published `fit.log` contains
only Arrow warnings; and every one of the 1,191 tests calls `fit` with
`epochs=1` or `epochs=2` and asserts on shapes and determinism. The suite was
fully green with a model that predicts a constant.

## Solution

Rebuild the study so that it can produce a result, and gate it so that it cannot
silently produce a non-result again.

Repair the world first. Separate the stochastic demand level from the
deterministic seasonal modulation in `Market` so the cycle stops entering the
integrator, and turn on the per-SKU item lifecycle so that nonstationarity is
independent across SKUs rather than shared and deterministic. Size the
`IntermediateNode` capacity pool and opening cash *relative to* episode demand
so tightness is the sampled quantity and both constraints genuinely bind. These
three changes are what create the headroom the study needs: a world where the
textbook censored-sales estimator provably lags, and where a shared capacity
pool forces real cross-SKU Ration decisions.

Repair the planner. Decouple the scoring horizon from the delivery lead time and
add a terminal value on ending inventory position, so that an order's
continuation value is represented rather than truncated away. Replace random
shooting with a factored search whose priority ordering is computed from
marginal profit density rather than drawn at random.

Repair the training. Make `epochs` mean a pass over the data, add minibatching,
a validation split, early stopping and a loss curve, and train the open-loop
unroll at the horizon the planner actually consumes. Promote held-out per-step
rollout accuracy to a gated, first-class metric that must clear a
least-squares baseline before any profit number is published.

Repair the measurement. Put mandatory floor rows and an oracle-model ceiling in
the comparison table, report operational columns alongside profit, flag arms
that agree to r > 0.99 instead of tabling them as distinct results, and make
the holding-rate transfer claim falsifiable by asserting that the frozen
policy's order sequence actually changes.

The claim the rebuilt study is built to support: **a learned world model plus a
planner beats a tuned textbook policy on a replenishment task where the
textbook family demonstrably leaves money on the table, and transfers to a
changed cost structure, to unseen SKUs and to an unseen disruption regime
without refitting.**

## User Stories

1. As an inventory researcher, I want the `Market` seasonal cycle to modulate
   the demand level rather than accumulate into it, so that `cycle_amp` means a
   bounded seasonal swing instead of an unbounded drift.
2. As an inventory researcher, I want the regional demand level to remain a
   clamped stochastic process, so that demand can never go negative.
3. As an inventory researcher, I want `MarketParams` to reject a `cycle_amp` of
   1.0 or greater, so that the positivity guarantee the old additive form
   silently relied on becomes an enforced invariant.
4. As an inventory researcher, I want the seasonal multiplier applied when the
   demand factor is read rather than stored into the region's market state, so
   that `EventEngine` disruption shocks continue to land on the level.
5. As an inventory researcher, I want the corrected demand level to spend no
   long stretch pinned at either clamp, so that the world seed materially
   changes the demand path.
6. As an inventory researcher, I want `demand_shock` dispersion and the trend
   factor to produce visibly different demand paths across world seeds, so that
   CRN-paired eval compares policies on genuinely different worlds.
7. As an inventory researcher, I want an ADR recording the `Market` cycle
   defect, the correction, and the fact that it supersedes existing CRN
   baselines, so that a future reader understands why tuned study values moved.
8. As an inventory researcher, I want every policy tuning study re-run after the
   `Market` correction, so that the anchor is tuned on the world it is measured
   in.
9. As an inventory researcher, I want per-SKU item lifecycle stage transitions
   enabled with per-episode sampled transition probabilities and initial stages,
   so that SKUs enter, grow, mature, decline and die on independent schedules.
10. As an inventory researcher, I want lifecycle-driven nonstationarity to be
    independent across SKUs, so that a shared Market wave cannot be mistaken for
    per-SKU signal and so that cross-SKU allocation tension is real.
11. As an inventory researcher, I want the `IntermediateNode` capacity pool sized
    as a sampled tightness multiple of episode demand rather than drawn
    independently, so that capacity binds in a controlled fraction of episodes.
12. As an inventory researcher, I want opening cash sized as a sampled tightness
    multiple of episode demand valued at unit cost, so that the purchasing budget
    binds in a controlled fraction of episodes.
13. As an inventory researcher, I want episode tightness reported as a per-seed
    column, so that a result can be read as a function of how binding the
    episode was.
14. As an inventory researcher, I want the delivery lead time sampled per
    episode, so that the study covers a range of supply contracts rather than a
    single lag of three.
15. As an inventory researcher, I want `K_active` sampled per episode across a
    wide range, so that the size of the assortment competing for the shared
    capacity pool varies.
16. As an inventory researcher, I want a larger and genuinely heterogeneous
    synthetic catalog, varying in demand scale, margin, volatility and lifecycle
    behaviour, so that SKU identity carries information the model must generalise
    across.
17. As an inventory researcher, I want the planner's scoring horizon expressed as
    a named multiple of the delivery lead time rather than equal to it, so that
    an order's revenue falls inside the window that scores it.
18. As an inventory researcher, I want a terminal value on ending inventory
    position at the end of the scoring horizon, so that the window boundary stops
    being a cliff and the horizon multiple stops being delicately tuned.
19. As an inventory researcher, I want the planner to order a sensible quantity
    under known-good conditions, so that a silent refusal to order is caught as a
    behavioural failure rather than surfacing as a low profit number.
20. As an inventory researcher, I want the proposal search to scale to a large
    assortment, so that sampling covers and a priority permutation at random does
    not become hopeless as `K_active` grows.
21. As an inventory researcher, I want the Ration priority order computed from
    marginal profit density under the shared capacity pool, so that the allocation
    reflects the knapsack structure of the problem.
22. As an inventory researcher, I want a random-permutation priority arm retained
    as an ablation, so that the question ADR 0025 asks — whether the priority
    Ration beats textbook fair-share — stays answerable.
23. As an inventory researcher, I want `epochs` to mean a pass over the training
    data, so that the published command trains the model rather than taking
    twenty gradient steps.
24. As an inventory researcher, I want minibatched, shuffled training, so that
    optimisation is not a full-batch descent on a quarter of a million rows.
25. As an inventory researcher, I want a validation split held out by episode, so
    that the reported training metric is not measured on the training rows.
26. As an inventory researcher, I want early stopping on the validation metric, so
    that the number of passes does not have to be guessed.
27. As an inventory researcher, I want the loss curve and validation metrics
    written to a run artifact, so that an undertrained fit is visible in the log
    rather than invisible.
28. As an inventory researcher, I want the open-loop unroll trained at the horizon
    the planner consumes, so that the planner does not operate on extrapolation.
29. As an inventory researcher, I want held-out rollout accuracy reported per step
    from one to the planning horizon, so that compounding open-loop error is
    visible rather than assumed away.
30. As an inventory researcher, I want a gate that refuses to publish a profit
    table when the model fails to beat an ordinary-least-squares baseline on
    held-out rollout accuracy, so that an untrained model cannot reach the results
    table again.
31. As an inventory researcher, I want the encoder's sales window length to be its
    own knob rather than the delivery lead time, so that a three-tick window is
    not silently imposed on a nonstationary world.
32. As an inventory researcher, I want the encoder's feature dimension to stay
    fixed as the delivery lead time varies, so that one fitted model serves every
    episode.
33. As an inventory researcher, I want the delivery lead time exposed to the
    encoder, so that the learned buyer knows the supply contract the textbook
    family already reads from its observation.
34. As an inventory researcher, I want in-transit units represented by their
    arrival-age profile rather than summed to a scalar, so that the model can
    know when units land instead of being structurally unable to predict it.
35. As an inventory researcher, I want the Market multiplier, the disruption
    identity and the calendar to stay out of the encoder's inputs, so that the
    premise of ADR 0024 survives the rebuild.
36. As an inventory researcher, I want the EMA target encoder to receive enough
    updates to leave its initialisation, so that the regression target and the
    probe inputs are a learned representation.
37. As an inventory researcher, I want the model's parameter count sized to the
    data volume, so that capacity rather than a three-thousand-parameter ceiling
    limits what can be learned.
38. As an inventory researcher, I want a reported data-scaling curve, so that the
    question of whether more episodes help is measured rather than asserted.
39. As an inventory researcher, I want the transition log generated by a mixture
    of the whole `TextbookReorderPolicy` family with hyperparameters sampled from
    the existing tuning search spaces, so that behaviour coverage comes from a
    principled distribution rather than two hand-picked points.
40. As an inventory researcher, I want the `ExploratoryPolicy` and deliberately
    poor arms — never-order and over-order — included in the behaviour mixture, so
    that the log covers states a competent policy never visits.
41. As an inventory researcher, I want behaviour policies to vary their own
    planning depth across log episodes, so that the log contains orders made under
    many different planning horizons.
42. As an inventory researcher, I want the fraction of transitions carrying a
    non-zero order reported for the log, so that action coverage is a measured
    property rather than an assumption.
43. As an inventory researcher, I want the published transition log collected at a
    single holding rate, so that the holding-rate transfer claim stays a zero-shot
    claim.
44. As an inventory researcher, I want the evaluation catalog partitioned so that
    training SKUs and evaluation SKUs never overlap, so that a per-SKU model is
    shown to generalise rather than to memorise.
45. As an inventory researcher, I want a harsher disruption regime held entirely
    out of the training episodes and reported as its own row, so that transfer to
    an unseen shock is measured.
46. As an inventory researcher, I want the in-distribution, unseen-SKU and
    unseen-disruption protocols reported as separate rows rather than pooled, so
    that each transfer claim stands alone.
47. As an inventory researcher, I want a never-order floor row in every comparison
    table, so that a policy that refuses to order is immediately visible.
48. As an inventory researcher, I want a random-cover floor row, so that any arm
    failing to beat uniform sampling from its own action set is visible.
49. As an inventory researcher, I want a least-squares-model-in-planner row, so
    that the learned model has to justify itself against the cheapest possible
    predictor.
50. As an inventory researcher, I want an oracle-model-in-planner ceiling row, so
    that a shortfall can be attributed to the planner or to the model rather than
    left ambiguous.
51. As an inventory researcher, I want order volume, service level and stockout
    rate reported on every row, so that an operational catastrophe cannot hide
    behind a profit average.
52. As an inventory researcher, I want any two arms whose per-seed results
    correlate above 0.99 reported as a warning, so that a vacuous ablation is
    named rather than presented as a comparison.
53. As an inventory researcher, I want the textbook anchor to reason with the true
    supplier unit cost, so that its internal budget check is not disabled while
    the learned planner self-clamps honestly.
54. As an inventory researcher, I want the cold-start behaviour of the anchor and
    the planner to be stated and matched, so that an opening-order advantage is a
    deliberate choice rather than an accident.
55. As an inventory researcher, I want the holding-rate transfer claim evaluated
    as a sweep over several rates rather than a single pair, so that the claim is
    a curve rather than one point.
56. As an inventory researcher, I want the frozen planner compared at each rate
    against a textbook policy tuned at that same rate, so that the ceiling is
    known at every point of the sweep.
57. As an inventory researcher, I want a hard assertion that the frozen planner's
    order sequence differs across holding rates on the real holdout, so that a
    no-op protocol fails loudly instead of reporting a number.
58. As an inventory researcher, I want ADR 0027 amended to record that the
    original retarget protocol was unfalsifiable under the old episode
    distribution, so that the correction is traceable.
59. As an inventory researcher, I want the paired comparison expressed as per-seed
    differences with a bootstrap interval, so that the heavy-tailed per-seed
    distribution does not drive the headline.
60. As an inventory researcher, I want the success rule to require the learned arm
    to beat the tuned anchor with an interval excluding zero, so that the study
    states a win rather than a tie.
61. As an inventory researcher, I want the published run to record the fitted
    model's held-out rollout accuracy alongside every profit number, so that the
    two are never read apart.
62. As an inventory researcher, I want the results document regenerated from run
    artifacts rather than hand-transcribed, so that the paper cannot drift from
    the run.
63. As an inventory researcher, I want `CONTEXT.md` updated with the terms this
    rebuild introduces, so that the glossary stays the shared vocabulary.

## Implementation Decisions

### Market demand process

The regional demand level becomes a purely stochastic clamped process; the
seasonal cycle becomes a read-time multiplier. In the update step the level
evolves as level times trend plus shock, clamped to `[min_value, max_value]`.
When the demand factor is read, the emitted value is the level multiplied by
`1 + cycle_amp * sin(2 * pi * step / cycle_len)`.

Positivity holds because the level is clamped strictly above zero and the
seasonal multiplier stays positive for any `cycle_amp` below 1.0. That bound was
implicit in the old additive form and becomes an enforced validation on
`MarketParams`. Applying the seasonal factor at read time rather than storing it
keeps `Market.state()` as the level, which is what `EventEngine` writes
disruption shocks into.

`demand_shock` dispersion and the `trend` distribution are widened so that the
world seed produces materially different demand paths. The exact values are
chosen to satisfy the episode-distribution invariants below rather than pinned
in advance.

This changes CRN identity for a fixed `world_seed`. Every tuned study, golden
value and stored baseline that depends on it moves. That is recorded in a new
ADR, and every policy tuning study used by this feature is re-run.

### Item lifecycle

`ItemLifecycleParams` stage-change probabilities and per-SKU initial stages are
sampled per episode rather than left at zero and `maturity`. The existing
`stage_multipliers` supply the demand effect. Transitions are drawn per SKU from
the episode's own sub-seed so that SKUs move independently.

### Episode distribution

Capacity and opening cash stop being drawn independently of demand. Each episode
samples a tightness factor; the `IntermediateNode` capacity pool is that factor
times aggregate episode demand times a cover horizon, and opening cash is that
factor times aggregate episode demand valued at unit cost. The tightness range
is chosen so that a controlled fraction of episodes bind on capacity, on cash,
and on neither. Tightness is recorded on the episode spec and surfaced as a
per-seed column.

The delivery lead time and `K_active` are sampled per episode. The synthetic
catalog grows to roughly 500–1000 SKUs with heterogeneous demand scale, margin,
volatility and lifecycle behaviour.

### Planner

The scoring horizon becomes `horizon_multiple * delivery_lag` with
`horizon_multiple` a named planner knob, defaulting to roughly 4. A terminal
value on ending inventory position is added at the end of the horizon so that
the boundary carries continuation value.

Random shooting is replaced by a factored cross-entropy search over per-SKU
covers. The Ration priority order is computed from marginal imagined profit per
unit of the shared capacity pool rather than drawn as a permutation. A
random-permutation arm and the textbook fair-share arm are both retained so the
ADR 0025 question stays answerable.

### World model

`epochs` means a pass over the data. Training is minibatched and shuffled, with
an episode-level validation split, early stopping on validation rollout
accuracy, and a written loss curve. The open-loop unroll is trained at the
planning horizon; `model_unroll_steps` is its own config value, not
`delivery_lag`.

The encoder's sales window length becomes its own knob, set well above the
delivery lead time so a nonstationary regime is identifiable. Feature dimension
becomes fixed and independent of the sampled lead time. The encoder gains the
delivery lead time and the in-transit arrival-age profile; both are contract
terms or self-knowledge rather than latent world state, so the ADR 0024 premise
that the Market multiplier, the disruption identity and the calendar stay hidden
is preserved and is restated explicitly.

Embedding width and depth grow so the model sits in the range of roughly half a
million to two million parameters. The EMA target encoder's decay is set against
the actual step count so the target leaves its initialisation.

### Transition log

The behaviour mixture is drawn from the whole `TextbookReorderPolicy` family with
hyperparameters sampled from the existing tuning search spaces, plus
`ExploratoryPolicy`, plus never-order and over-order arms. Behaviour policies
vary their own planning depth across episodes. The log records the realised
fraction of transitions carrying a non-zero order. The published log is collected
at a single holding rate.

### Evaluation

The catalog is partitioned so that training SKUs and evaluation SKUs are
disjoint. Three protocols are reported as separate tables: in-distribution,
unseen SKUs, and an unseen harsher disruption regime kept out of training.

Every table carries never-order, random-cover and least-squares-model floors and
an oracle-model ceiling, plus order volume, service level and stockout rate on
every row. Arms correlating above 0.99 across seeds are reported as a warning.

The textbook anchor is constructed with the true supplier unit cost. Cold-start
behaviour for anchor and planner is stated and matched.

The holding-rate transfer becomes a sweep across several rates, each against a
textbook policy tuned at that rate, with a hard assertion that the frozen
planner's order sequence differs across rates on the real holdout. ADR 0027 is
amended.

The headline metric stays `net_profit / initial_cash` for consistency with the
existing policy tuning study objective, reported as per-seed paired differences
with a bootstrap interval. The success rule requires the learned arm to beat the
tuned anchor with an interval excluding zero.

A gate refuses to produce a profit table when held-out rollout accuracy fails to
beat an ordinary-least-squares baseline on the same inputs.

### Documentation

A new ADR records the `Market` cycle correction and its CRN consequences. ADR
0027 is amended. ADR 0024's hidden-state premise is restated to name exactly what
the encoder may and may not see. `CONTEXT.md` gains the terms this rebuild
introduces.

## Testing Decisions

A good test here asserts an external, observable property — a distributional
invariant of sampled episodes, a held-out accuracy number, a behavioural
property of a Policy's emitted action, or the shape and content of a published
table. It does not assert on network internals, parameter values, or the number
of gradient steps taken. The failure mode this feature exists to prevent is a
green suite over a model that predicts a constant, so tests that only check
tensor shapes and determinism are explicitly not sufficient.

Four seams, all of them existing module boundaries. No new seams.

**`Market` demand process.** Invariants over a long simulated run: the emitted
demand factor is strictly positive; the level spends no long stretch pinned at
either clamp; the realised seasonal swing matches `cycle_amp`; a `cycle_amp` at
or above 1.0 is rejected; different world seeds produce materially different
demand paths. Prior art: existing `Market` unit tests.

**`sample_episode`.** Distributional properties over many sampled episodes:
aggregate demand does not collapse to a floor partway through an episode;
capacity and cash each bind in a controlled fraction of episodes and neither
binds in another; lifecycle stages actually move and move independently across
SKUs; delivery lead time and `K_active` vary. Prior art: existing tuning episode
tests.

**`LatentDynamics` fit through to held-out rollout accuracy.** The model-quality
gate. Held-out per-step rollout R² from one step to the planning horizon must
beat an ordinary-least-squares baseline on the same inputs. This is the test
that fails today and must pass before any profit number is published. It runs at
a real training budget on a reduced episode count, not at `epochs=1`.

**`compare_synthetic`.** End-to-end table content: the floor and ceiling rows are
present and populated; operational columns are present on every row; arms
correlating above 0.99 raise the warning; the never-order floor is beaten by
every learned arm. Prior art: existing synthetic comparison tests.

Smoke configurations stay short — few ticks, few episodes, small search budget —
except for the model-quality gate, which needs a real training budget to mean
anything.

## Out of Scope

The M5 replay exam. It was never run in the first study and it depends on
everything above being sound first. `docs/adr/0026` and `docs/adr/0028` are
untouched.

Pricing policies. The learned buyer stays ordering-only, emitting `list_price`
from `base_price` with no promotions, per ADR 0024.

Policy control of engine Allocation. The Ration stays a buyer-side decision and
`execute_buy` is unchanged.

A second learned echelon. One learned buyer on the `IntermediateNode` facing the
sinks; factories stay on `StaticFactoryPolicy`.

Multi-supplier routing and upstream scarcity. Considered and deliberately
deferred — factories remain amply stocked, so routing stays exercised by the
textbook family but is not a source of headroom for this study.

A PPO or Dreamer bake-off.

Rewriting the reinforcement-learning stack to match the corrected `Market`. The
ADR records that RL baselines move; re-establishing them is separate work.

## Further Notes

The audit's diagnostic scripts live in `/tmp/jepa_diag/` and are not committed.
The three findings they establish — planner is model-blind at 0.336 with a
perfect model, training reaches R² ≈ 0 at the published step count, and demand
collapses at tick 207 in 128 of 128 episodes — are the acceptance evidence that
this rebuild must invalidate.

The `Market` correction is the highest-leverage change and also the most
disruptive, because it moves every CRN baseline in the repository. It is
sequenced first deliberately: tuning the anchor before the world is corrected
would produce an anchor tuned for a degenerate world.

Ordinary least squares on the existing eight inputs reaches R² of +0.94 / +0.75 /
+0.65, and a properly trained latent model reaches +0.91 / +0.77 / +0.72 — it
only matches the linear baseline. That is evidence the old feature set had no
representational gap for a joint-embedding model to close, which is why the
encoder input changes (longer window, arrival-age profile, sampled lead time)
matter as much as the training changes. The `next_in_transit` ceiling near 0.75
is structural: arrival timing is not recoverable from a summed in-transit
scalar.
