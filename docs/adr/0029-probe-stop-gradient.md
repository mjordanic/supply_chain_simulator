# Probe losses do not train the encoder

Status: Accepted

The planner scores `net_profit` from decoded on-hand, in-transit, and sales. Those decoders are probes on the latent. If their loss trains the encoder, the encoder becomes a multi-task regressor of the same targets the direct-predictor ablation is trained to emit, and a win can no longer be attributed to the latent objective.

**Decision — Probe losses update the probes only.** The encoder is trained by the multi-step latent prediction loss over the lead-time horizon and by the anti-collapse penalty. Probes read a stopped embedding and match the next on-hand, the next in-transit arrival-age profile, and the sales over the step. The planner sums that profile when it scores inventory position; the sum is not the regression target. The planner's profit calculation is not differentiated into the encoder. The direct predictor is trained to emit those same targets and has no latent target.

**In-transit target.** A summed pipeline cannot time arrivals. The regression target is the arrival-age profile on both the probes and the direct predictor.

**Why stop the probes?** An embedding that predicts the future latent and still decodes inventory badly should lose to the direct predictor. Shared gradients would hide that failure by reshaping the embedding until the probes succeed.
