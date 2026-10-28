# T1 gate G1 (300 dev worlds)

Rule (fixed before the run): G1 passes when the known-model Bayes planner's mean dev margin capture beats both the protocol reference (the textbook arm with the best tune score) and the best textbook arm on dev by >= 1.00 pt.

| arm − reference | reference | diff, pts [95% BCa] | win rate |
| --- | --- | --- | --- |
| planner_bayes − by tune score | `tuned_param_eoq_joint` | +1.011 [+0.909, +1.125] | 95% |
| planner_bayes − best on dev | `ctx_ets_eoq_joint` | +0.942 [+0.845, +1.062] | 95% |
| planner_oracle − by tune score | `tuned_param_eoq_joint` | +2.162 [+1.955, +2.407] | 100% |
| planner_oracle − best on dev | `ctx_ets_eoq_joint` | +2.093 [+1.892, +2.338] | 100% |

Mean margin capture: `planner_oracle` 0.9635, `planner_bayes` 0.9520, `ctx_ets_eoq_joint` 0.9425, `tuned_param_eoq_joint` 0.9418, `tuned_ets_eoq_joint` 0.9416, `ctx_eoq_joint` 0.9415, `tuned_eoq_joint` 0.9411, `tuned_eoq` 0.9263, `tuned_reorder_point` 0.9106, `tuned_periodic_reorder` 0.9092, `tuned_order_up_to` 0.9087, `tuned_periodic_order_up_to` 0.9051, `never_order` 0.0106

**G1 fails.**
