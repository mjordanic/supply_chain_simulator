# Holding-rate retarget re-scores a frozen predictor

Status: Accepted

Holding cost is `holding_rate × closing on-hand × unit_cost`, charged to the intermediate's cash (ADR 0019). The latent predictor decodes on-hand, in-transit, and sales. Those quantities move with the granted order, not with the holding rate. The rate changes the score, and it changes later cash, which changes which later rations still fit.

**Decision — On the retarget, the predictor stays the one trained at `holding_rate = 0.01`.** Imagined trajectories are re-scored with `holding_rate = 0.05`. Cash is rolled forward with that rate and fed back into the priority ration, so a cover that no longer fits is not proposed. Behavior data is not recollected at the new rate, and the weights are not refit. The `OrderUpToPolicy` tuned at `0.01` stays frozen beside it. An `OrderUpToPolicy` retuned at `0.05` is the ceiling, reported separately. Matching that ceiling is not the success criterion. Success is the frozen planner beating the frozen textbook policy on paired `net_profit / initial_cash`.

**Why not refit?** A refit on rollouts collected at `0.05` would let the model see the new economy while the frozen textbook policy cannot. That spends the claim the retarget exists to test.
