# A simpler replenishment experiment with one perfect supplier

Status: Accepted

The published JEPA buyer (ADRs 0024–0032) did not beat a tuned order-up-to policy. Its latent model had to learn multi-supplier routing, partial fills, lifecycles, and disruptions as well as demand, and its held-out rollout R² was 0.43. That world was too rich for the model and too slow to simulate enough episodes to see why.

**Decision — Run the learned buyer in a world where only demand is uncertain, and learn only demand.**

1. **World.** One shop, four always-active products, one perfect supplier (an order arrives in full after the delivery lag). Cost 10, price 20, one fee of 50 per order day. Per world: lag 2–7, holding rate, demand CV, product means, a weekday profile, and level shifts (see `CONTEXT.md`, *Perfect-supplier world*). Shelf and cash scale with demand.
2. **Simulator.** `src/perfect_supplier/sim.py` is a lean numpy loop in the engine's tick order. `tests/perfect_supplier/test_engine_parity.py` runs the four textbook rules on it and on `src/sim` against the same replayed demand; on-hand matches exactly and cash to 1e-9. The engine charges one fee per supplier edge, so that test uses fee 0.
3. **Censoring.** Policies see sales and a sold-out flag, never demand. The forecaster's loss is censored pinball: a sold-out day bounds demand from below.
4. **Learned part.** A JEPA forecaster (context encoder, EMA target encoder over the next 35 days, predictor, VICReg terms) with a quantile head trained jointly. This departs from ADR 0029's stop-gradient probes: the head is the output the planner uses, so its loss trains the encoder. A direct ablation drops the latent loss.
5. **Planner.** Exact inventory, cash, fee, and shelf recursion over 64 sampled demand paths; it chooses a joint reorder plan `(cover, service quantile)` by mean imagined change in equity over 35 days. The same planner with moving-average and oracle forecasts brackets the forecast's value.
6. **Baselines.** Each textbook family is tuned once (Optuna, 200 trials) across 64 tune worlds, including a longer demand window than the engine's lag-day default. A practitioner tunes once; the learned buyer adapts per world.
7. **Score.** Change in equity over `(price − cost) × total demand` (margin capture), paired by world across arms, 95% bootstrap intervals.

## Consequences

The experiment runs thousands of episodes in minutes and stores every day × product row (`src/perfect_supplier/data.py`). Results do not transfer to multi-supplier worlds; that claim needs its own run. The engine path (`src/jepa/`) is unchanged.

## Cross-references

- ADR 0019 — equity and the fee and holding mechanics
- ADR 0029 — stop-gradient probes, not followed here
- ADR 0031, 0032 — the supply-side and tightness sampling this experiment removes
