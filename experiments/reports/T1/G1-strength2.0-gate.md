# T1 gate G1 (300 dev worlds)

Rule (fixed before the run): G1 passes when the known-model Bayes planner's mean dev margin capture beats both the protocol reference (the textbook arm with the best tune score) and the best textbook arm on dev by >= 1.00 pt.

| arm − reference | reference | diff, pts [95% BCa] | win rate |
| --- | --- | --- | --- |
| planner_bayes − by tune score | `tuned_ets_eoq_joint` | +1.376 [+1.239, +1.524] | 94% |
| planner_bayes − best on dev | `ctx_ets_eoq_joint` | +1.286 [+1.147, +1.444] | 96% |
| planner_oracle − by tune score | `tuned_ets_eoq_joint` | +3.155 [+2.846, +3.518] | 100% |
| planner_oracle − best on dev | `ctx_ets_eoq_joint` | +3.064 [+2.766, +3.423] | 100% |

Mean margin capture: `planner_oracle` 0.9638, `planner_bayes` 0.9460, `ctx_ets_eoq_joint` 0.9332, `ctx_eoq_joint` 0.9330, `tuned_eoq_joint` 0.9329, `tuned_param_eoq_joint` 0.9326, `tuned_ets_eoq_joint` 0.9323, `tuned_eoq` 0.9184, `tuned_reorder_point` 0.9009, `tuned_order_up_to` 0.8981, `tuned_periodic_reorder` 0.8979, `tuned_periodic_order_up_to` 0.8902, `never_order` 0.0097

**G1 passes.**
