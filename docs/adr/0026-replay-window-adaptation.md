# Replay stays an exam; one adaptation row may fit an earlier window

Status: Accepted — widens ADR 0020 for the latent replenishment study only.

ADR 0020 makes a `ReplayDemandSinkNode` an exam: could this policy have served the recorded stream? It rules out calibration and RL training. Fitting a world model on the same series you then score is a third use. The engine contract is unchanged. `demand_target` is still the series times the multiplier chain, and CRN draws are still burned.

**Decision — Two M5 numbers, never one pooled window.** The headline model is trained on synthetic rollouts only and evaluated zero-shot on a later replay window. A second row refits the same reward-free predictor on an earlier window of that series and is tested on the later window. The two windows are not concatenated. The adaptation row is not RL training and does not use the PPO stack. Both arms still wrap the ordering policy in the same `PriceReplayPolicy` under `flat_world`.

**Why allow the second row?** Intermittent, stockout-censored retail sales can make a pure transfer number unreadable. The adaptation row says whether the simulator's policy interface can serve the stream once the predictor has seen that store's history. The zero-shot row says whether synthetic pretraining transferred. Reporting only the better of the two would hide which claim held.
