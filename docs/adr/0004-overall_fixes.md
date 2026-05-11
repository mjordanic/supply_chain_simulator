remove init demand / suplly from LLM world generator, set default to max-min/2

should market state be observable?

Reorder in eventEngine. Should be:
2 (drop expired)
3 (create new ones)
1 apply
4 callbacks


_event_payload (what if there are multiple events?)

store.needs_init_order - should it be part of policy to order more?

try to move self.needs_init_order from store to policy.
same for promotion_cooldown. perhaps it can be passed by saving time step at which something happened
what about active products? should store keep this info?

promo duration @policy.py line 285 should be sampled as discount at line 283
also 337, 338

don't understand well correlated prices

policy 521 hardcoded 6

policy 506 - only one product can be activated by policy in one step. Should we increase this? Perhaps use n products as a parameter of the policy? Number of deactivated is uncapped.

Why don't we place order after activating? move _review_catalog at the top of the tasks line 216


runner 275, price is not changed within the function. soesn't make sense
