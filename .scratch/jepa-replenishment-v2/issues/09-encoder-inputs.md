# 09: Encoder inputs

**What to build:** An observation the encoder can actually infer the regime from,
without handing it the latent state the study is premised on withholding.

Today the sales window length is hardwired to the delivery lead time — three
ticks — which cannot identify a seasonal level or a lifecycle stage. Feature
dimension is `2 + delivery_lag + 2`, so a sampled lead time would reshape the
model. In-transit units are a summed scalar, so arrival timing is not recoverable
and `next_in_transit` accuracy is capped near 0.75 for every method including
ordinary least squares. And the lead time itself is hidden from the learned
buyer while `TextbookReorderPolicy` reads it directly from its observation and
uses it to set the reorder point — the same unfair-asymmetry class as the
`unit_cost` defect in ticket 04.

The ADR 0024 premise survives: the Market multiplier, the disruption identity and
the calendar stay out. Lead time and pipeline age are contract terms and
self-knowledge, not latent world state.

**Blocked by:** 02, 06.

**Status:** ready-for-agent

**Prior art:** existing dynamics tests; the `Latent replenishment planner`,
`Observation` and `Market` entries in `CONTEXT.md`; ADR 0024 (ordering-only
learned policy) for the hidden-state premise, ADR 0029 (probe stop-gradient) for
what the probes may update.

- [ ] The encoder's sales window length is its own knob, set well above the
      delivery lead time.
- [ ] Feature dimension is fixed and independent of the sampled delivery lag, so
      one fitted model serves every episode.
- [ ] The delivery lead time is an encoder feature.
- [ ] In-transit units enter as an arrival-age profile rather than a summed
      scalar, and a test shows this lifts held-out `next_in_transit` accuracy
      above the 0.75 ceiling that the summed scalar imposes.
- [ ] The Market multiplier, the disruption identity and the calendar are absent
      from the encoder's inputs, asserted by a test.
- [ ] ADR 0024's hidden-state premise is restated to name exactly what the
      encoder may and may not see, and to record that routing stays outside the
      learned action.
- [ ] Any notebook, README section or `CONTEXT.md` text invalidated by the new
      observation is updated in this ticket.
