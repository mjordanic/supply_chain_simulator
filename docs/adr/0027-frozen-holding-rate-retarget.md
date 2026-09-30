# Holding-rate retarget re-scores a frozen predictor

Status: Accepted, amended. The single-rate protocol below was unfalsifiable. The sweep in the amendment is the protocol.

Holding cost is `holding_rate × closing on-hand × unit_cost`, charged to the intermediate's cash (ADR 0019). The latent predictor decodes on-hand, in-transit, and sales. Those quantities move with the granted order, not with the holding rate. The rate changes the score, and it changes later cash, which changes which later rations still fit.

**Decision — On the retarget, the predictor stays the one trained at `holding_rate = 0.01`.** Imagined trajectories are re-scored with `holding_rate = 0.05`. Cash is rolled forward with that rate and fed back into the priority ration, so a cover that no longer fits is not proposed. Behavior data is not recollected at the new rate, and the weights are not refit. The `OrderUpToPolicy` tuned at `0.01` stays frozen beside it. An `OrderUpToPolicy` retuned at `0.05` is the ceiling, reported separately. Matching that ceiling is not the success criterion. Success is the frozen planner beating the frozen textbook policy on paired `net_profit / initial_cash`.

**Why not refit?** A refit on rollouts collected at `0.05` would let the model see the new economy while the frozen textbook policy cannot. That spends the claim the retarget exists to test.

## Amendment — the single-rate protocol was unfalsifiable

The original protocol compared one pair, `holding_rate` 0.01 against 0.05, and read a drop in paired profit as the frozen planner adapting. Under the old episode distribution the tuner holdout's opening cash was large enough that the cash constraint never bound. Measured on that holdout, the frozen planner placed the same orders at both rates. The profit gap was the world charging more for unchanged behaviour. A fixture that emptied the till made an acceptance test pass without exercising the holdout, so the claim could not fail.

The corrected protocol:

- Sweeps holding rates 0.005, 0.01, 0.02, 0.05, and 0.1 on the tuner holdout.
- At each rate, scores the frozen planner — weights from the 0.01 fit — against the `OrderUpToPolicy` tuned at that same rate. Those tuned policies are the existing 150-trial studies. The sweep does not launch another study.
- Does not refit the model and does not collect a new transition log.
- Before reporting any sweep profit, checks that the frozen planner's order sequence on that holdout is not the same at every rate.
- If the sequences match, the run raises `UnfalsifiableTransfer` and writes no sweep table.

Matching the rate-tuned textbook policy is not the success criterion. The order-sequence check is what makes the claim able to fail. Imagined cash still uses the rate under test and still feeds the ration. The predictor still does not see the holding rate except through the quantity it is asked to roll forward.
