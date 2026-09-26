# M5 arm is one CA_1 shop and five FOODS items chosen on 2015

Status: Accepted

The M5 demo notebook builds three California shops, a distribution centre, and one factory per product over calendar 2015. This study's learned policy sits on one intermediate facing the sinks (ADR 0024). That demo graph is a different experiment. `HOUSEHOLD_1_108` at `CA_1` in the demo slice has a `longest_zero_run` of 69 days, which is censored sales rather than a demand rate the predictor should imitate.

**Decision — The M5 arm is `shop-CA_1` plus one synthetic factory with lead time 3.** Five items, chosen using 2015-01-01 through 2015-12-31 only: rank `FOODS_3` at `CA_1` by shortest `longest_zero_run`, then by highest mean sales; drop any item whose 2015 zero run is longer than 7 days. If fewer than five survive, widen to all `FOODS` departments at `CA_1` under the same rule. Pin the five ids after the first quality report. The adaptation window is that 2015 year. The test window is 2016-01-01 through 2016-04-24, which fits both `sales_train_validation` and `sales_train_evaluation`. `unit_cost` stays the adapter default, 40% of median price. `flat_world` and the same `PriceReplayPolicy` on both ordering policies, per ADR 0026.

**Why lead time 3?** The synthetic tuner episodes use `delivery_lag = 3`, and the predictor's horizon is that lead time. A shop-to-DC-to-factory chain would change the horizon the weights were trained on.
