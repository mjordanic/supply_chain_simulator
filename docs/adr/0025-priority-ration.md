# Learned policy rations by priority; engine Allocation stays put

Status: Accepted

`execute_buy` is **Allocation**: it clamps one requested quantity against the supplier's stock, the buyer's cash, and remaining capacity. The textbook family's `_allocate_two_pass_fair_share` is a different step. It splits one intermediate's free capacity and cash across SKUs before an order exists. Calling both Allocation hides which one a policy is allowed to choose.

**Decision — The learned policy owns the split, and the engine still owns Allocation.** Name the split a **ration**. Each SKU proposes a cover in lead times of recent demand and a priority. A greedy fill walks that priority order and grants units until space or cash is exhausted. Dynamics stay per SKU, conditioned on the quantity that SKU was granted. `OrderUpToPolicy` keeps fair-share, so a gap between them can be a better ration under contention. Training rollouts must include non-fair-share rations. Do not replace `execute_buy`.

**Future work.** A joint latent model of every SKU at once, and any policy control of engine Allocation, are out of this study. Pricing stays out under ADR 0024.

**Why priority fill rather than a search over the joint order?** Tuning episodes ship `K_active = 5`. A free combination of covers is already thousands of candidates and grows worse if K rises. Priority plus greedy fill is one low-dimensional action per SKU and still lets a fast seller take the whole pool when space is tight, which equal fair-share will not do.
