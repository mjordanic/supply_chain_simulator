# 06: Episode distribution, supply side

**What to build:** A supply side that can fail to deliver, so an order is not a
guaranteed arrival and lead time is not a constant.

Today every `FactoryNode` produces four times the node capacity per tick, opens
pre-stocked at twice node capacity, and lists at exactly its unit cost. Each SKU
has one supplier. Upstream stockouts never happen, suppliers earn no margin,
fill rate never moves, and routing is dead code — `_split_across_suppliers` and
the fill-rate-weighted strategy are never exercised by this study. Delivery lag
is pinned at 3 for every episode.

**Blocked by:** 01.

**Status:** ready-for-agent

**Prior art:** existing tuning episode tests and `CentralTable` tests; the
`FactoryNode`, `CentralTable`, `Allocation`, `EdgeSpec` and `Graph` entries in
`CONTEXT.md`; ADR 0011 (multi-echelon graph) and ADR 0012 (central table FCFS
allocation).

- [ ] Each SKU is supplied by several `FactoryNode`s rather than one.
- [ ] Per-supplier `list_price` carries a sampled markup over that supplier's
      unit cost, so suppliers earn margin and buyer cost is not production cost.
- [ ] Supplier production per tick and opening inventory are sampled relative to
      aggregate episode demand rather than set as multiples of node capacity.
- [ ] Upstream stockouts occur in a non-trivial fraction of episodes, and that
      fraction is asserted by a test rather than hoped for.
- [ ] Per-edge lead times differ across the suppliers of one SKU, so price and
      speed trade off.
- [ ] Delivery lag is sampled per episode rather than pinned at 3.
- [ ] Realised supplier fill rate is recorded per episode and varies across
      episodes.
- [ ] Any notebook, README section or `CONTEXT.md` text invalidated by the new
      topology is updated in this ticket.
