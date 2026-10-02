# Learned policy rations by priority; engine Allocation stays put

Status: Accepted

`execute_buy` is **Allocation**: it clamps one requested quantity against the supplier's stock, the buyer's cash, and remaining capacity. The textbook family's `_allocate_two_pass_fair_share` is a different step. It splits one intermediate's free capacity and cash across SKUs before an order exists. Calling both Allocation hides which one a policy is allowed to choose.

**Decision — The learned policy owns the split, and the engine still owns Allocation.** Name the split a **ration**. Each SKU proposes a cover in lead times of recent demand and a priority. A greedy fill walks that priority order and grants units until space or cash is exhausted. Dynamics stay per SKU, conditioned on the quantity that SKU was granted. `OrderUpToPolicy` keeps fair-share, so a gap between them can be a better ration under contention. Training rollouts must include non-fair-share rations. Do not replace `execute_buy`.

**Future work.** A joint latent model of every SKU at once, and any policy control of engine Allocation, are out of this study. Pricing stays out under ADR 0024.

**Why priority fill rather than a search over the joint order?** `K` is sampled per episode from 1 through 30. A free combination of covers is already thousands of candidates at a handful of SKUs and grows worse as K rises. Priority plus greedy fill is one low-dimensional action per SKU and still lets a fast seller take the whole pool when space is tight, which equal fair-share will not do.

**Amendment — 2026-09-30, assortment.** This paragraph used to say tuning episodes ship a fixed `K_active` of 5. That was the first study's episode. The corrected distribution samples `K`. The search reason is unchanged.

**Amendment — 2026-09-30.** The planner computes that priority from the marginal imagined profit of one unit of shared capacity. `ration="permutation"` draws one order from `policy_rng` and is the ablation. `ration="fair_share"` stays, so the comparison is three-way. Covers are chosen by factored cross-entropy, not by a joint random draw. Engine Allocation is unchanged.
