"""T4: ``Store`` accounting invariants (issue 05).

The four invariants from the issue:

1. ``final_balance == initial_balance + Σ revenue − Σ total_cost`` over the
   life of the store.
2. ``inventory[t] + Σ sold[0..t] == Σ delivered[0..t] + initial_stock``
   (units are conserved across settle / deliver).
3. ``pending[t]`` decreases by the dispatched ``qty`` on each ``deliver``
   call (the original semantics: pending is keyed off the order, not the
   capacity-clamped received quantity).
4. ``deliver`` clamps received units at remaining store capacity.

All tests target the public ``Store`` class directly. Issue 07 layers the
full Runner integration over these primitives; this tier only verifies
that the per-store accounting is correct in isolation.
"""

from __future__ import annotations

from src.sim.policy import NoopPolicy
from src.sim.scenario import StoreTemplate, load_catalog
from src.sim.store import Store


def _template() -> StoreTemplate:
    return StoreTemplate(
        id="t",
        region="US",
        capacity=100,
        init_balance=1000.0,
        init_stock_pct=0.5,
        delivery_lag=2,
        holding_rate=0.01,
        order_fee=10.0,
        init_active_count=2,
    )


def _catalog():
    return load_catalog(
        [
            {
                "name": "A",
                "category": "x",
                "related_products": [],
                "base_price": 10.0,
                "unit_cost": 5.0,
                "seasonality": "all_season",
            },
            {
                "name": "B",
                "category": "x",
                "related_products": [],
                "base_price": 20.0,
                "unit_cost": 8.0,
                "seasonality": "all_season",
            },
            {
                "name": "C",
                "category": "x",
                "related_products": [],
                "base_price": 15.0,
                "unit_cost": 6.0,
                "seasonality": "all_season",
            },
        ]
    )


def _make_store() -> Store:
    return Store(_template(), init_seed=1, policy=NoopPolicy(), catalog=_catalog())


def test_balance_equation_over_lifetime():
    """final_balance == initial_balance + Σ revenue − Σ total_cost."""
    s = _make_store()
    initial_balance = s.balance

    sum_revenue = 0.0
    sum_total_cost = 0.0
    pids = list(s.inventory.keys())

    # Settle a series of (demand, price, order_qty) tuples across products.
    plan = [
        (pids[0], 5, 12.0, 3),
        (pids[1], 8, 25.0, 0),
        (pids[2], 2, 14.0, 7),
        (pids[0], 1, 11.0, 0),
        (pids[1], 0, 22.0, 4),
    ]
    for pid, demand, price, order_qty in plan:
        _, revenue, _ = s.settle(pid, demand=demand, price=price, order_qty=order_qty)
        sum_revenue += revenue
        sum_total_cost += s.total_cost[pid]

    assert s.balance == initial_balance + sum_revenue - sum_total_cost


def test_inventory_unit_conservation():
    """inventory[t] + Σ sold[0..t] == Σ delivered[0..t] + initial_stock."""
    s = _make_store()
    pid = next(p for p in s.inventory if s.inventory[p] > 0)
    initial_stock = s.inventory[pid]

    sum_sold = 0
    sum_delivered = 0

    # t=0: settle some demand against initial stock.
    sold, _, _ = s.settle(pid, demand=10, price=10.0, order_qty=0)
    sum_sold += sold
    assert s.inventory[pid] + sum_sold == sum_delivered + initial_stock

    # t=1: receive a delivery.
    inv_before = s.inventory[pid]
    s.deliver(pid, 7)
    sum_delivered += s.inventory[pid] - inv_before
    assert s.inventory[pid] + sum_sold == sum_delivered + initial_stock

    # t=2: more demand than current inventory (sells out remainder).
    sold, _, _ = s.settle(pid, demand=999, price=10.0, order_qty=0)
    sum_sold += sold
    assert s.inventory[pid] + sum_sold == sum_delivered + initial_stock

    # t=3: another delivery; verify conservation still holds.
    inv_before = s.inventory[pid]
    s.deliver(pid, 5)
    sum_delivered += s.inventory[pid] - inv_before
    assert s.inventory[pid] + sum_sold == sum_delivered + initial_stock


def test_pending_decreases_by_dispatched_qty():
    """pending[pid] decreases by the dispatched qty on each delivery."""
    s = _make_store()
    pid = next(iter(s.inventory))

    s.pending[pid] = 10
    s.deliver(pid, 4)
    assert s.pending[pid] == 6

    # Second delivery further drains pending.
    s.deliver(pid, 3)
    assert s.pending[pid] == 3

    # Over-delivering clamps pending at zero (does not go negative).
    s.deliver(pid, 100)
    assert s.pending[pid] == 0


def test_delivery_clamps_at_remaining_capacity():
    """Received qty is clamped at (capacity − current total inventory)."""
    s = _make_store()
    # Drain all inventory so we know remaining capacity exactly.
    for pid in list(s.inventory):
        s.inventory[pid] = 0

    pid = next(iter(s.inventory))
    over_qty = int(s.capacity) + 50
    s.deliver(pid, over_qty)

    assert sum(s.inventory.values()) == s.capacity
    # The over-quantity that did not land in inventory was dropped.
    assert s.inventory[pid] == s.capacity


def test_delivery_clamps_when_other_inventory_present():
    """If other products already occupy capacity, the clamp respects them."""
    s = _make_store()
    pids = list(s.inventory)
    # Fill 80% of capacity with one product, then deliver to another.
    s.inventory[pids[0]] = int(s.capacity * 0.8)
    s.inventory[pids[1]] = 0
    if len(pids) > 2:
        s.inventory[pids[2]] = 0

    space = s.capacity - sum(s.inventory.values())
    s.deliver(pids[1], int(s.capacity))

    assert s.inventory[pids[1]] == space
    assert sum(s.inventory.values()) == s.capacity


def test_settle_uses_order_fee_only_when_ordering():
    """Fixed order fee is charged iff order_qty > 0."""
    s = _make_store()
    pid = next(iter(s.inventory))

    s.settle(pid, demand=0, price=10.0, order_qty=0)
    cost_no_order = s.total_cost[pid]

    s.settle(pid, demand=0, price=10.0, order_qty=1)
    cost_with_order = s.total_cost[pid]

    # Difference of order_fee plus the cost of one ordered unit.
    assert cost_with_order - cost_no_order == s.order_fee + s.costs[pid]


def test_settle_clamps_negative_demand_to_zero():
    """Negative demand collapses to zero sales (preserves original guard)."""
    s = _make_store()
    pid = next(p for p in s.inventory if s.inventory[p] > 0)
    inv_before = s.inventory[pid]

    sold, revenue, _ = s.settle(pid, demand=-5, price=10.0, order_qty=0)

    assert sold == 0
    assert revenue == 0
    assert s.inventory[pid] == inv_before


def test_init_state_independent_of_policy():
    """Two stores sharing (template, init_seed) are bit-identical at step 0."""
    template = _template()
    catalog = _catalog()

    s_a = Store(template, init_seed=42, policy=NoopPolicy(policy_seed=1), catalog=catalog)
    s_b = Store(template, init_seed=42, policy=NoopPolicy(policy_seed=999), catalog=catalog)

    assert s_a.capacity == s_b.capacity
    assert s_a.balance == s_b.balance
    assert s_a.active_items == s_b.active_items
    assert s_a.inventory == s_b.inventory


def test_activate_and_deactivate_item():
    """activate adds + flags for first order; deactivate removes from active set."""
    s = _make_store()
    catalog_pids = [w.product_id for w in _catalog()]
    inactive_pid = next(p for p in catalog_pids if p not in s.active_items)

    s.activate_item(inactive_pid)
    assert inactive_pid in s.active_items
    assert inactive_pid in s.needs_init_order

    s.deactivate_item(inactive_pid)
    assert inactive_pid not in s.active_items


def test_decide_threads_policy_decisions_into_state():
    """Policy-returned order/promotion/activate/deactivate decisions land on the store."""

    class _StubPolicy(NoopPolicy):
        def decide(self, observation):
            return {
                "order": {"P0000": 5, "P0001": 0},
                "promotions": {"P0000": {"discount": 0.1}},
                "promotion_cooldown": {"P0001": 7},
                "activate": ["P0002"],
                "deactivate": ["P0001"],
            }

    s = Store(_template(), init_seed=1, policy=_StubPolicy(), catalog=_catalog())
    s.decide({"step": 0})

    assert s.pending["P0000"] == 5
    assert "P0001" not in s.pending or s.pending["P0001"] == 0
    assert s.promotions == {"P0000": {"discount": 0.1}}
    assert s.promo_cooldown == {"P0001": 7}
    assert "P0002" in s.active_items
    assert "P0001" not in s.active_items
