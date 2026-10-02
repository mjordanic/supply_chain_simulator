"""Factored cover search, density-ordered ration, and fill expectation.

Seams, from the issue acceptance criteria:

1. Search. Cross-entropy over per-SKU covers beats the same number of
   uniform joint draws from ``{0, 1, 2, 4}``. The number of scored
   proposals stays at ``budget`` as the assortment grows.
2. Ration. Priority is marginal imagined profit per unit of shared
   capacity. A random permutation and textbook fair-share both remain,
   so the three-way comparison stays available.
3. Transition. Imagined arrivals follow the log's fill rate. The order
   the buyer emits is still the full grant.
"""

from __future__ import annotations

from src.jepa.plan import ReplenishmentPlanner
from src.sim.central_table import CentralTable, Offer

LAG = 2
DEMAND = 10
COST = 4.0
PRICE = 20.0


class _ShelfDynamics:
    """Units arrive after one lead time and sell at a fixed rate per SKU."""

    def __init__(self, n_skus: int, *, delivery_lag: int = LAG, demand: float = DEMAND) -> None:
        self.delivery_lag = int(delivery_lag)
        self.demand = float(demand)
        self.n_skus = int(n_skus)
        self.steps = 0
        self.seen: list[list[float]] = []

    def open_rollout(self, features: list[list[float]]) -> _ShelfRollout:
        return _ShelfRollout(self, features)


class _ShelfRollout:
    def __init__(self, dynamics: _ShelfDynamics, features: list[list[float]]) -> None:
        self.dynamics = dynamics
        self.rows = [[float(feat[0]), [0.0] * dynamics.delivery_lag] for feat in features]

    def step(self, granted: list[float]) -> list[tuple[float, float, float]]:
        self.dynamics.steps += 1
        self.dynamics.seen.append([float(qty) for qty in granted])
        predicted: list[tuple[float, float, float]] = []
        for row, qty in zip(self.rows, granted, strict=True):
            on_hand, profile = row
            arriving = float(profile.pop(0))
            profile.append(float(qty))
            on_hand += arriving
            sales = min(on_hand, self.dynamics.demand)
            row[0] = on_hand - sales
            predicted.append((row[0], float(sum(profile)), sales))
        return predicted


def _table(pids: list[str], *, cost: float = COST) -> CentralTable:
    table = CentralTable()
    for pid in pids:
        table.publish(f"F_{pid}", pid, Offer(available_qty=100_000, list_price=cost, min_order=0))
    return table


def _obs(pids: list[str], *, capacity: int, price: float = PRICE, sales: int = DEMAND) -> dict:
    return {
        "tick": 0,
        "inventory": {pid: 0 for pid in pids},
        "pending": {},
        "observed_sales": {pid: sales for pid in pids},
        "base_prices": {pid: price for pid in pids},
        "list_prices": {pid: 1.0 for pid in pids},
        "capacity": capacity,
        "cash": 10_000_000.0,
        "direct_supplier_ids": {f"F_{pid}" for pid in pids},
    }


def _units(action: dict, pid: str) -> int:
    return sum(qty for _supplier, qty in action["order"].get(pid, []))


def _planner(dynamics: _ShelfDynamics, *, budget: int, search: str, seed: int = 0) -> ReplenishmentPlanner:
    return ReplenishmentPlanner(
        dynamics,
        policy_seed=seed,
        delivery_lag=LAG,
        budget=budget,
        holding_rate=0.01,
        order_fee=0.0,
        search=search,
    )


def test_factored_search_beats_uniform_draws_from_the_same_covers():
    """Four profitable SKUs and four that lose money. No constant cover is optimal.

    A budget of 64 joint uniform draws still orders the unprofitable SKUs.
    Factored cross-entropy stops ordering them and scores higher. The four
    constant cover vectors cannot both stock the profitable SKUs and leave
    the others alone, which is the failure uniform shooting had.
    """
    good = [f"G{i}" for i in range(4)]
    bad = [f"B{i}" for i in range(4)]
    pids = good + bad
    budget = 64

    def run(search: str) -> tuple[ReplenishmentPlanner, dict]:
        planner = _planner(_ShelfDynamics(len(pids)), budget=budget, search=search, seed=0)
        obs = _obs(pids, capacity=100_000)
        obs["base_prices"] = {pid: 30.0 if pid in good else 2.0 for pid in pids}
        table = CentralTable()
        for pid in pids:
            cost = COST if pid in good else 20.0
            table.publish(
                f"F_{pid}",
                pid,
                Offer(available_qty=100_000, list_price=cost, min_order=0),
            )
        action = planner.decide(obs, table)
        assert len(planner.proposals[-1]["draws"]) == budget
        return planner, action

    cem, cem_action = run("cem")
    uniform, uniform_action = run("uniform")
    covers = cem.proposals[-1]["chosen"]["covers"]
    assert float(cem.proposals[-1]["score"]) > float(uniform.proposals[-1]["score"])
    assert len(set(covers.values())) > 1
    assert sum(_units(cem_action, pid) for pid in bad) == 0
    assert sum(_units(cem_action, pid) for pid in good) > sum(
        _units(uniform_action, pid) for pid in good
    )
    assert sum(_units(uniform_action, pid) for pid in bad) > 0


def _two_sku_planner(*, ration: str, seed: int = 0, budget: int = 4) -> ReplenishmentPlanner:
    return ReplenishmentPlanner(
        _ShelfDynamics(2),
        policy_seed=seed,
        delivery_lag=LAG,
        budget=budget,
        holding_rate=0.01,
        order_fee=0.0,
        ration=ration,
    )


def _priced_obs(prices: dict[str, float], *, capacity: int) -> tuple[dict, CentralTable]:
    pids = sorted(prices)
    obs = _obs(pids, capacity=capacity)
    obs["base_prices"] = dict(prices)
    table = CentralTable()
    for pid in pids:
        table.publish(f"F_{pid}", pid, Offer(available_qty=100_000, list_price=COST, min_order=0))
    return obs, table


def test_density_priority_serves_the_higher_margin_sku_first():
    """A is first alphabetically and the worse margin. The pool still goes to B.

    Free space holds one lead time of demand, so the first SKU in the
    priority order takes it all.
    """
    obs, table = _priced_obs({"A": 6.0, "B": PRICE}, capacity=DEMAND * LAG)
    action = _two_sku_planner(ration="priority").decide(obs, table)
    assert _units(action, "B") == DEMAND * LAG
    assert _units(action, "A") == 0
    assert action["list_price"] == {"A": 6.0, "B": PRICE}


def test_permutation_arm_can_serve_the_lower_margin_sku_first():
    """The ablation draws an order. It is not the density ranking."""
    obs, table = _priced_obs({"A": 6.0, "B": PRICE}, capacity=DEMAND * LAG)
    served_lower_margin = False
    for seed in range(40):
        planner = _two_sku_planner(ration="permutation", seed=seed)
        action = planner.decide(obs, table)
        if planner.proposals[-1]["chosen"]["priority"][0] != "A":
            continue
        assert _units(action, "A") == DEMAND * LAG
        assert _units(action, "B") == 0
        served_lower_margin = True
        break
    assert served_lower_margin


def test_fair_share_splits_a_pool_priority_gives_to_one_sku():
    """Equal margins. Priority tie-breaks to A. Fair-share still splits."""
    obs, table = _priced_obs({"A": PRICE, "B": PRICE}, capacity=DEMAND * LAG)
    priority = _two_sku_planner(ration="priority").decide(obs, table)
    fair = _two_sku_planner(ration="fair_share").decide(obs, table)
    assert _units(priority, "A") == DEMAND * LAG
    assert _units(priority, "B") == 0
    assert _units(fair, "A") > 0
    assert _units(fair, "B") > 0
    assert _units(fair, "A") + _units(fair, "B") <= DEMAND * LAG
    assert fair["order"] != priority["order"]


def test_proposal_count_stays_at_the_budget_as_the_assortment_grows():
    """Scoring does not become a joint grid when K_active reaches 30."""
    budget = 16
    horizon = 4 * LAG
    for k in (2, 30):
        dynamics = _ShelfDynamics(k)
        pids = [f"S{i}" for i in range(k)]
        planner = _planner(dynamics, budget=budget, search="cem")
        planner.decide(_obs(pids, capacity=100_000), _table(pids))
        assert len(planner.proposals[-1]["draws"]) == budget
        assert budget * horizon <= dynamics.steps <= (budget + k) * horizon


def test_imagined_transition_uses_the_fill_rate_and_the_order_stays_whole():
    """A half fill is what the rollout sees. The emitted order is the grant."""
    dynamics = _ShelfDynamics(1)
    dynamics.fill_rate = 0.5
    planner = _planner(dynamics, budget=4, search="cem")
    action = planner.decide(_obs(["A"], capacity=100_000), _table(["A"]))
    cover = planner.proposals[-1]["chosen"]["covers"]["A"]
    desired = max(0, round(cover * DEMAND * LAG))
    emitted = _units(action, "A")
    assert emitted == desired > 0
    imagined = [qty for step in dynamics.seen for qty in step]
    # The spread scores a cover of 4, whose grant is 80. Half of that arrives.
    assert max(imagined) == 0.5 * (4 * DEMAND * LAG)


def test_fit_reads_the_fill_rate_off_the_log():
    import pandas as pd

    from src.jepa.plan import fit_direct, fit_latent

    frame = pd.DataFrame(
        {
            "episode_seed": [1, 1],
            "requested_qty": [8.0, 0.0],
            "filled_qty": [2.0, 0.0],
        }
    )
    latent = fit_latent(frame, delivery_lag=LAG, epochs=0, seed=0)
    direct = fit_direct(frame, delivery_lag=LAG, epochs=0, seed=0)
    assert float(latent.dynamics.fill_rate) == 0.25
    assert float(direct.dynamics.fill_rate) == 0.25


def test_fill_rate_follows_textbook_orders_not_the_coverage_arms():
    """Over-order and exploratory requests are not the planner's grants.

    Quantity-weighting them reports a fill near zero, and the planner
    then imagines that a cover arrives empty. Textbook orders fill.
    """
    import pandas as pd

    from src.jepa.plan import fit_direct, fit_latent

    mixed = pd.DataFrame(
        {
            "episode_seed": [1, 1, 1],
            "behavior": ["order_up_to", "exploratory", "over_order"],
            "requested_qty": [10.0, 100.0, 1000.0],
            "filled_qty": [10.0, 1.0, 1.0],
        }
    )
    latent = fit_latent(mixed, delivery_lag=LAG, epochs=0, seed=0)
    assert float(latent.dynamics.fill_rate) == 1.0

    coverage_only = mixed[mixed["behavior"] != "order_up_to"]
    direct = fit_direct(coverage_only, delivery_lag=LAG, epochs=0, seed=0)
    assert abs(float(direct.dynamics.fill_rate) - (2.0 / 1100.0)) < 1e-6


def test_rollout_gate_replaces_a_crushed_fill_rate(tmp_path):
    """A checkpoint fitted on the old ratio must not keep it into the eval."""
    import pandas as pd

    from src.jepa.eval import score_rollout_gate
    from src.jepa.plan import fit_direct, fit_latent

    frame = pd.DataFrame(
        {
            "episode_seed": [1, 1],
            "behavior": ["order_up_to", "over_order"],
            "requested_qty": [4.0, 400.0],
            "filled_qty": [4.0, 1.0],
        }
    )
    latent = fit_latent(frame, delivery_lag=LAG, epochs=0, seed=0)
    direct = fit_direct(frame, delivery_lag=LAG, epochs=0, seed=0)
    latent.dynamics.fill_rate.fill_(0.001)
    direct.dynamics.fill_rate.fill_(0.001)
    try:
        score_rollout_gate(latent, direct, frame, tmp_path)
    except Exception:
        pass
    assert float(latent.dynamics.fill_rate) == 1.0
    assert float(direct.dynamics.fill_rate) == 1.0
