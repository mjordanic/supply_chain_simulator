# 08: Behaviour mixture for the transition log

**What to build:** A training log whose state-action coverage comes from a
principled distribution rather than two hand-picked policies.

Today half the episodes run published-default `OrderUpToPolicy` and half run
`ExploratoryPolicy`, and only 13% of transitions carry a non-zero order. Three
of the four textbook families never appear. What a world model needs from its
behaviour policy is *coverage* — states and actions a competent policy never
visits — not behaviour quality.

**Blocked by:** 02, 07.

**Status:** done

**Prior art:** existing transition-log tests; the `Transition log`,
`Exploratory replenishment policy`, `TextbookReorderPolicy family` and
`Policy tuning study` entries in `CONTEXT.md`. The tuning search spaces already
define the hyperparameter distributions this ticket samples from — reuse them
rather than inventing new ranges.

- [x] The behaviour mixture draws from all four `TextbookReorderPolicy` families
      with hyperparameters sampled from the existing tuning search spaces.
- [x] `ExploratoryPolicy` and deliberately poor arms — never-order and
      over-order — are included, so the log covers states a competent policy
      never reaches.
- [x] Behaviour policies vary their own planning depth across log episodes.
- [x] The realised fraction of transitions carrying a non-zero order is reported
      in the manifest and is materially higher than the current 13%.
- [x] State coverage is reported: the log visits low-inventory, high-inventory,
      capacity-bound and cash-bound states, asserted rather than assumed.
- [x] The published log is collected at a single holding rate, so the
      holding-rate transfer claim stays a zero-shot claim.
- [x] The manifest records the mixture composition and the sampled
      hyperparameters.
- [x] Any notebook, README section or `CONTEXT.md` text invalidated by the new
      mixture is updated in this ticket.
