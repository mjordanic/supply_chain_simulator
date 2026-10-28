# T1 gate G1 (300 dev worlds)

Rule (fixed before the run): G1 passes when the known-model Bayes planner's mean dev margin capture beats both the protocol reference (the textbook arm with the best tune score) and the best textbook arm on dev by >= 1.00 pt.

| arm − reference | reference | diff, pts [95% BCa] | win rate |
| --- | --- | --- | --- |
| planner_bayes − by tune score | `tuned_param_eoq_joint` | +0.838 [+0.758, +0.927] | 98% |
| planner_bayes − best on dev | `ctx_ets_eoq_joint` | +0.748 [+0.680, +0.825] | 99% |
| planner_oracle − by tune score | `tuned_param_eoq_joint` | +1.406 [+1.270, +1.562] | 100% |
| planner_oracle − best on dev | `ctx_ets_eoq_joint` | +1.315 [+1.191, +1.463] | 100% |

Mean margin capture: `planner_oracle` 0.9632, `planner_bayes` 0.9575, `ctx_ets_eoq_joint` 0.9500, `ctx_eoq_joint` 0.9494, `tuned_ets_eoq_joint` 0.9492, `tuned_param_eoq_joint` 0.9491, `tuned_eoq_joint` 0.9490, `tuned_eoq` 0.9321, `tuned_reorder_point` 0.9191, `tuned_periodic_reorder` 0.9180, `tuned_order_up_to` 0.9177, `tuned_periodic_order_up_to` 0.9158, `never_order` 0.0113

**G1 fails.**
