# 07: Size capacity and cash relative to demand so they bind

**What to build:** Episodes where the shared capacity pool and the purchasing
budget actually constrain the buyer, so cross-SKU Ration decisions matter and the
benchmark has headroom.

Today `capacity` and opening `balance` are drawn independently of demand.
Measured across the published log, inventory scale varies 97× and cash 110× while
mean demand per SKU varies only 2.4×. The result is that almost every episode
sits in a "stock far exceeds demand" regime: service level is 1.000, neither
constraint binds, random covers score 3.004 against the tuned anchor's 3.031, and
a 150-trial Optuna study bought a headline uplift of −9.35 with an interval
spanning zero. On a benchmark where tuning buys nothing, no method can be shown
to beat another.

After this ticket tightness is the sampled quantity, so a controlled fraction of
episodes bind on capacity, a controlled fraction on cash, and a controlled
fraction on neither. This also conditions the `net_profit / initial_cash` metric,
whose per-seed values currently span 0.15 to 13.55 for a single fixed policy.

**Blocked by:** 05, 06.

**Status:** ready-for-agent

**Prior art:** existing tuning episode tests; the `Episode sampler`,
`IntermediateNode`, `Ration` and `Business metrics` entries in `CONTEXT.md`; ADR
0007 (RL scale-invariance package) for how scale is handled elsewhere.

- [ ] Each episode samples a tightness factor; the `IntermediateNode` capacity
      pool is that factor times aggregate episode demand times a cover horizon.
- [ ] Opening cash is sampled the same way, relative to aggregate episode demand
      valued at unit cost.
- [ ] Tightness is recorded on the episode spec and surfaced as a per-seed column
      in downstream tables.
- [ ] A test asserts capacity binds in a controlled fraction of sampled episodes,
      cash binds in a controlled fraction, and neither binds in a controlled
      fraction.
- [ ] A test asserts aggregate demand does not collapse to a floor partway
      through an episode, closing out the defect from ticket 01 at the episode
      level.
- [ ] The full episode-distribution invariant suite lands here: demand does not
      collapse, tightness binds as specified, lifecycle stages move independently,
      delivery lag and `K_active` vary, suppliers stock out and fill rate varies.
- [ ] Any notebook, README section or `CONTEXT.md` text invalidated by the new
      sizing rule is updated in this ticket.
