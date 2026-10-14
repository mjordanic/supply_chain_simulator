import numpy as np

from src.perfect_supplier.sim import Decision, ration, simulate
from src.perfect_supplier.world import HISTORY_DAYS, demand_path, sample_world


class Fixed:
    def __init__(self, qty, every=1, name="fixed"):
        self.qty, self.every, self.name = qty, every, name

    def decide(self, obs):
        q = self.qty if obs.day % self.every == 0 else 0
        return Decision(np.full(obs.n_products, q))


def test_demand_is_the_same_path_for_every_policy():
    world = sample_world(7)
    a = simulate(world, Fixed(0, name="a")).frame
    b = simulate(world, Fixed(5, name="b")).frame
    assert (a["demand"].to_numpy() == b["demand"].to_numpy()).all()
    assert (a["demand"].to_numpy() == demand_path(world)[HISTORY_DAYS:].ravel()).all()


def test_equity_identity_holds_every_day():
    world = sample_world(3)
    f = simulate(world, Fixed(30, every=3)).frame
    day = f.groupby("day").first()
    margin = (world.price - world.unit_cost) * f.groupby("day")["sales"].sum()
    change = margin - day["fee_paid"] - day["holding_cost"]
    expected = world.initial_cash + world.unit_cost * sum(world.initial_on_hand) + change.cumsum()
    np.testing.assert_allclose(day["equity_close"], expected, rtol=1e-9)


def test_order_arrives_in_full_after_the_lag():
    world = sample_world(11)
    f = simulate(world, Fixed(1, every=1000)).frame
    first = f[f["day"] == 0]
    assert (first["filled"] == 1).all()
    lagged = f[f["day"] == world.delivery_lag]
    assert (lagged["arrivals"] == 1).all()
    assert (f[(f["day"] > 0) & (f["day"] != world.delivery_lag)]["arrivals"] == 0).all()


def test_ration_respects_shelf_and_cash():
    assert ration(np.array([10, 10]), 5, 1e9, 10.0).sum() == 5
    assert ration(np.array([10, 10]), 100, 75.0, 10.0).sum() == 7
    assert (ration(np.array([3, 4]), 100, 1e9, 10.0) == [3, 4]).all()


def test_shelf_never_overflows():
    world = sample_world(5)
    f = simulate(world, Fixed(10_000)).frame
    day = f.groupby("day")[["on_hand_open"]].sum()
    assert (day["on_hand_open"] <= world.capacity).all()


def test_variants_share_the_base_world():
    base = sample_world(9, "eval")
    one = sample_world(9, "eval", variant="one_product")
    free = sample_world(9, "eval", variant="no_fee")
    assert one.n_products == 1 and free.order_fee == 0.0
    assert (one.delivery_lag, one.cv, one.holding_rate) == (base.delivery_lag, base.cv, base.holding_rate)
    assert (demand_path(one)[:, 0] == demand_path(base)[:, 0]).all()
    assert (demand_path(free) == demand_path(base)).all()
    assert one.capacity < base.capacity and one.initial_cash < base.initial_cash


def test_order_days_count_orders_when_the_fee_is_zero():
    world = sample_world(5, variant="no_fee")
    summary = simulate(world, Fixed(1, every=7)).summary
    assert summary["fees"] == 0.0
    assert summary["order_days"] == len(range(0, world.n_days, 7))


def test_fee_is_charged_once_per_order_day():
    world = sample_world(5)
    summary = simulate(world, Fixed(3, every=4)).summary
    assert summary["order_days"] == len(range(0, world.n_days, 4))
    assert summary["fees"] == world.order_fee * summary["order_days"]
