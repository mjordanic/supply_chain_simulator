"""M5 replay exam for the latent replenishment planner.

One ``shop-CA_1``, one ``StaticFactoryPolicy`` factory per pinned item at
lead time 3, and one replay sink per item. Selection and the two windows
are ADR 0028. Zero-shot uses synthetic weights. Adaptation refits on the
earlier window only. Neither row is a pass/fail gate.
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from src.datasets.m5 import M5Slice, load_m5_slice, quality_report

STORE_ID = "CA_1"
SHOP_ID = "shop-CA_1"
PRIMARY_DEPT = "FOODS_3"
N_ITEMS = 5
MAX_ZERO_RUN = 7
LEAD_TIME = 3
UNIT_COST_FRACTION = 0.4
SHOP_CASH = 500_000.0
HOLDING_RATE = 0.01
ORDER_FEE = 50.0
SELECTION_START = date(2015, 1, 1)
SELECTION_END = date(2015, 12, 31)
TEST_START = date(2016, 1, 1)
TEST_END = date(2016, 4, 24)

KAGGLE_DOWNLOAD = "kaggle competitions download -c m5-forecasting-accuracy"

SLICE_RULE = (
    "ADR 0028: choose five items on 2015-01-01 through 2015-12-31 only. "
    "Rank FOODS_3 at CA_1 by shortest longest_zero_run, then highest mean sales. "
    "Drop a zero run longer than 7 days. If fewer than five items survive, "
    "widen to all FOODS departments at CA_1 under the same rule. "
    "The test window is 2016-01-01 through 2016-04-24."
)


def m5_files_present(data_dir: str | Path) -> bool:
    """True when ``data_dir`` holds a sales file, ``sell_prices.csv``, and ``calendar.csv``."""
    directory = Path(data_dir)
    if not directory.is_dir():
        return False
    has_sales = any(
        path.name.startswith("sales_train") and path.suffix == ".csv"
        for path in directory.iterdir()
    )
    return has_sales and (directory / "sell_prices.csv").is_file() and (directory / "calendar.csv").is_file()


def skip_message(data_dir: str | Path) -> str | None:
    """Skip text when the raw files are absent, else ``None``.

    The text names the Kaggle download and the ADR 0028 slice rule.
    """
    if m5_files_present(data_dir):
        return None
    return (
        f"Raw M5 files not found under {data_dir}.\n"
        "Download them from Kaggle:\n"
        f"  {KAGGLE_DOWNLOAD}\n"
        "  https://www.kaggle.com/competitions/m5-forecasting-accuracy/data\n"
        "Place sales_train_validation.csv (or sales_train_evaluation.csv), "
        "sell_prices.csv, and calendar.csv in that directory.\n"
        f"{SLICE_RULE}\n"
        "Remaining cells are skipped."
    )


def _mean_sales(series: list[int]) -> float:
    if not series:
        return 0.0
    return sum(series) / len(series)


def rank_foods_items(
    m5_slice: M5Slice,
    *,
    store_id: str = STORE_ID,
    n: int = N_ITEMS,
    max_zero_run: int = MAX_ZERO_RUN,
) -> list[str]:
    """Pick ``n`` items at ``store_id`` from the slice as given.

    The caller loads 2015-01-01 through 2015-12-31. Rank ``FOODS_3`` by
    shortest ``longest_zero_run``, then highest mean sales, and drop a zero
    run longer than ``max_zero_run``. If fewer than ``n`` survive, repeat
    the same rule over every FOODS department at that store.
    """
    report = quality_report(m5_slice)
    by_item = {
        row.item_id: row
        for row in report.itertuples(index=False)
        if row.store_id == store_id
    }

    def ranked(keep) -> list[tuple[int, float, str]]:
        chosen_rows: list[tuple[int, float, str]] = []
        for item_id, row in by_item.items():
            item_dept = m5_slice.item_dept.get(item_id, "")
            if not keep(item_dept):
                continue
            zero_run = int(row.longest_zero_run)
            if zero_run > max_zero_run:
                continue
            mean_sales = _mean_sales(m5_slice.sales.get((item_id, store_id), []))
            chosen_rows.append((zero_run, -mean_sales, item_id))
        chosen_rows.sort()
        return chosen_rows

    primary = ranked(lambda dept: dept == PRIMARY_DEPT)
    chosen = primary if len(primary) >= n else ranked(lambda dept: dept.startswith("FOODS"))
    return [item_id for _run, _mean, item_id in chosen[:n]]


def list_foods_item_ids(data_dir: str | Path, *, store_id: str = STORE_ID) -> list[str]:
    """FOODS ``item_id`` values sold at ``store_id``, in file order."""
    directory = Path(data_dir)
    sales_path = next(
        path
        for path in directory.iterdir()
        if path.name.startswith("sales_train") and path.suffix == ".csv"
    )
    found: list[str] = []
    seen: set[str] = set()
    with open(sales_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("store_id") != store_id:
                continue
            if not str(row.get("dept_id", "")).startswith("FOODS"):
                continue
            item_id = row["item_id"]
            if item_id in seen:
                continue
            seen.add(item_id)
            found.append(item_id)
    return found


def load_selection_slice(data_dir: str | Path, *, store_id: str = STORE_ID) -> M5Slice:
    """FOODS items at ``store_id`` on 2015-01-01 through 2015-12-31 only."""
    item_ids = list_foods_item_ids(data_dir, store_id=store_id)
    return load_m5_slice(
        data_dir,
        item_ids,
        [store_id],
        SELECTION_START,
        SELECTION_END,
    )


def choose_items(data_dir: str | Path, *, store_id: str = STORE_ID) -> list[str]:
    """Rank FOODS items at ``store_id`` on the 2015 selection window only."""
    return rank_foods_items(load_selection_slice(data_dir, store_id=store_id), store_id=store_id)


def _median_price(m5_slice: M5Slice, item_id: str, store_id: str) -> float:
    observed = [price for price in m5_slice.prices.get((item_id, store_id), []) if price > 0.0]
    if not observed:
        return 1.0
    return float(statistics.median(observed))


def build_replay_scenario(
    m5_slice: M5Slice,
    *,
    store_id: str = STORE_ID,
    lead_time: int = LEAD_TIME,
    unit_cost_fraction: float = UNIT_COST_FRACTION,
    world_seed: int = 42,
):
    """One factory per item feeding ``shop-CA_1`` at ``lead_time``, plus one replay sink.

    ``FactoryNode`` manufactures a single product, so the upstream echelon is
    one ``StaticFactoryPolicy`` factory per pinned item and no distribution
    centre. ``unit_cost`` is ``unit_cost_fraction`` times the median observed
    price. ``flat_world`` keeps the demand multiplier at 1.
    """
    from src.sim.distributions import Constant
    from src.sim.flat_world import flat_world
    from src.sim.graph import EdgeSpec
    from src.sim.node import FactoryNode, IntermediateNode
    from src.sim.policy import StaticFactoryPolicy
    from src.sim.replay_demand_sink import ReplayDemandSinkNode
    from src.sim.scenario import DisruptionParams, NodeInstance, Scenario, Ware

    if m5_slice.start_date is None or m5_slice.end_date is None:
        raise ValueError("replay scenario needs a dated slice")
    item_ids = list(m5_slice.item_ids)
    n_steps = m5_slice.n_ticks
    region = m5_slice.store_state.get(store_id, "CA")
    market, lifecycle = flat_world(regions=[region])

    wares: list[Ware] = []
    for item_id in item_ids:
        base_price = _median_price(m5_slice, item_id, store_id)
        unit_cost = max(unit_cost_fraction * base_price, 1e-6)
        wares.append(
            Ware(
                product_id=item_id,
                name=item_id,
                category=m5_slice.item_dept.get(item_id, "FOODS"),
                base_price=base_price,
                unit_cost=unit_cost,
                seasonality="default",
                related_products=[],
                init_stage=None,
                stage_change_probs=None,
                freshness_alpha=None,
                freshness_decay=None,
                init_stock_share=1.0,
            )
        )
    ware_by_id = {ware.product_id: ware for ware in wares}

    nodes: list[NodeInstance] = []
    edges: list[EdgeSpec] = []
    for index, item_id in enumerate(item_ids):
        ware = ware_by_id[item_id]
        factory_id = f"factory-{item_id}"
        factory = FactoryNode(
            id=factory_id,
            region=region,
            init_seed=world_seed + index,
            produces_product_id=item_id,
            unit_cost=float(ware.unit_cost),
            capacity_per_tick=Constant(100_000),
            inventory=100_000,
            list_price=float(ware.unit_cost),
            cash=0.0,
        )
        nodes.append(
            NodeInstance(
                node=factory,
                init_seed=world_seed + index,
                policy=StaticFactoryPolicy(
                    capacity_per_tick=100_000,
                    unit_cost=float(ware.unit_cost),
                    policy_seed=world_seed + index,
                ),
            )
        )
        edges.append(
            EdgeSpec(
                supplier_id=factory_id,
                buyer_id=SHOP_ID,
                default_lead_time=lead_time,
            )
        )

    shop = IntermediateNode(
        id=SHOP_ID,
        region=region,
        init_seed=world_seed + len(item_ids),
        carried_products=set(item_ids),
        capacity=10_000_000,
        tags=["shop"],
        inventory={item_id: 0 for item_id in item_ids},
        list_prices={item_id: float(ware_by_id[item_id].base_price) for item_id in item_ids},
        min_order_imposed={item_id: 0 for item_id in item_ids},
        cash=SHOP_CASH,
        holding_rate=HOLDING_RATE,
        order_fee=ORDER_FEE,
    )
    nodes.append(NodeInstance(node=shop, init_seed=shop.init_seed, policy=None))

    for index, item_id in enumerate(item_ids):
        series = list(m5_slice.sales.get((item_id, store_id), [0] * n_steps))
        ware = ware_by_id[item_id]
        max_demand = max(series) if series else 1
        income_rate = float(max_demand * max(ware.base_price, 1.0) * 100)
        sink_id = f"sink-{item_id}-{store_id}"
        sink = ReplayDemandSinkNode(
            id=sink_id,
            region=region,
            init_seed=world_seed + len(item_ids) + 1 + index,
            product_id=item_id,
            series=series,
            n_steps=n_steps,
            income_rate=income_rate,
            cash=income_rate * n_steps * 2,
            demand_dist=Constant(0),
        )
        nodes.append(NodeInstance(node=sink, init_seed=sink.init_seed, policy=None))
        edges.append(EdgeSpec(supplier_id=SHOP_ID, buyer_id=sink_id, default_lead_time=0))

    return Scenario(
        catalog=wares,
        market=market,
        disruption=DisruptionParams(
            event_prob=0.0,
            types=["natural_disaster"],
            regions=[region],
            severity=Constant(0.0),
            duration=Constant(1),
        ),
        item_lifecycle=lifecycle,
        nodes=nodes,
        edges=edges,
        n_steps=n_steps,
        start_date=datetime(m5_slice.start_date.year, m5_slice.start_date.month, m5_slice.start_date.day),
        world_seed=world_seed,
    )


def replay_prices(m5_slice: M5Slice, *, store_id: str = STORE_ID) -> dict[str, list[float]]:
    """Observed daily prices for ``store_id``, one array per pinned item."""
    return {
        item_id: [float(price) for price in m5_slice.prices.get((item_id, store_id), [])]
        for item_id in m5_slice.item_ids
    }


def wrap_price_replay(inner, prices: dict[str, list[float]], n_steps: int):
    """Wrap an ordering policy so ``decide`` posts the observed price.

    The engine's first tick is observation tick 1, so the array passed to
    ``PriceReplayPolicy`` is shifted by one. ``run_shop`` also writes the
    unshifted series onto the shop before each tick, because offers are
    published from ``list_prices`` before ``decide``.
    """
    from src.sim.policy import PriceReplayPolicy

    aligned = {
        pid: ([series[0], *series] if series else [])
        for pid, series in prices.items()
    }
    return PriceReplayPolicy(inner, aligned, n_steps)


@dataclass(frozen=True)
class ShopRun:
    """One shop rollout under ``PriceReplayPolicy``."""

    policy: Any
    log: dict[str, Any]


def _run_scenario(scenario, shop_policy, prices: dict[str, list[float]]):
    """Run ``scenario`` after writing each day's observed price before publish."""
    from src.sim.runner import Runner

    runner = Runner(scenario, policy_overrides={SHOP_ID: shop_policy})
    shop = runner.nodes[SHOP_ID]
    original_tick = runner._sim.tick

    def tick() -> None:
        index = int(runner._sim.market.step)
        for product_id, series in prices.items():
            if index < len(series):
                shop.list_prices[product_id] = float(series[index])
        original_tick()

    runner._sim.tick = tick
    return runner.run()


def run_shop(m5_slice: M5Slice, ordering_policy, *, store_id: str = STORE_ID) -> ShopRun:
    """Run the replay graph with ``ordering_policy`` inside ``PriceReplayPolicy``.

    Offers are published before ``decide``, so each tick writes that day's
    observed price onto the shop first. Both the learned policy and
    ``OrderUpToPolicy`` go through this same wrapper and the same prices.
    """
    scenario = build_replay_scenario(m5_slice, store_id=store_id)
    prices = replay_prices(m5_slice, store_id=store_id)
    wrapped = wrap_price_replay(ordering_policy, prices, scenario.n_steps)
    return ShopRun(policy=wrapped, log=_run_scenario(scenario, wrapped, prices))


def collect_adaptation_log(
    m5_slice: M5Slice,
    *,
    store_id: str = STORE_ID,
    delivery_lag: int = LEAD_TIME,
):
    """Reward-free transitions on this slice from the two behavior policies.

    One trajectory of published-default ``OrderUpToPolicy`` and one of
    ``ExploratoryPolicy``. The slice is not concatenated with another window.
    """
    import pandas as pd

    from src.jepa.log import COLUMNS, _TransitionRecorder, _transitions
    from src.jepa.policy import ExploratoryPolicy
    from src.sim.policy import OrderUpToPolicy

    scenario = build_replay_scenario(m5_slice, store_id=store_id)
    prices = replay_prices(m5_slice, store_id=store_id)
    behaviors = (
        ("order_up_to", OrderUpToPolicy()),
        ("exploratory", ExploratoryPolicy(policy_seed=0, delivery_lag=delivery_lag)),
    )
    rows: list[dict[str, Any]] = []
    for episode_seed, (name, policy) in enumerate(behaviors):
        wrapped = wrap_price_replay(policy, prices, scenario.n_steps)
        recorder = _TransitionRecorder(wrapped, delivery_lag)
        _run_scenario(scenario, recorder, prices)
        rows.extend(_transitions(recorder.steps, episode_seed, name))
    return pd.DataFrame(rows, columns=list(COLUMNS))


def shop_net_profit(m5_slice: M5Slice, ordering_policy, *, store_id: str = STORE_ID) -> float:
    """Shop ``net_profit`` from one price-replay rollout of ``ordering_policy``."""
    from src.sim.inspect import closing_inventory_frame, flow_frame, purchase_frame
    from src.sim.metrics import profit_decomposition

    run = run_shop(m5_slice, ordering_policy, store_id=store_id)
    scenario = build_replay_scenario(m5_slice, store_id=store_id)
    frame = profit_decomposition(
        flow_frame(run.log),
        purchase_frame(run.log),
        closing_inventory_frame(run.log),
        scenario,
    )
    row = frame[frame["node_id"] == SHOP_ID]
    if row.empty:
        return 0.0
    return float(row["net_profit"].iloc[0])


def tune_anchor(
    m5_slice: M5Slice,
    *,
    n_trials: int,
    sampler_seed: int = 0,
    store_id: str = STORE_ID,
) -> tuple[dict[str, Any], float]:
    """Search ``order_up_to_space`` to maximise replay profit on ``m5_slice``.

    The slice is the only scenario the objective runs. A later test window
    is not an argument.
    """
    from src.tuning.search_spaces import maximise_order_up_to

    return maximise_order_up_to(
        lambda policy: shop_net_profit(m5_slice, policy, store_id=store_id),
        n_trials=n_trials,
        sampler_seed=sampler_seed,
    )


@dataclass(frozen=True)
class ReplayRow:
    """One reported M5 row. There is no pass/fail flag."""

    label: str
    planner_net_profit: float
    anchor_net_profit: float
    planner_normalised: float
    anchor_normalised: float


@dataclass(frozen=True)
class ReplayExam:
    """Zero-shot and adaptation rows on the test window.

    ``anchor_selection_profit`` is the anchor's profit on the adaptation
    window, which is the quantity the search maximised. ``anchor_params``
    rebuild an ``OrderUpToPolicy`` through ``order_up_to_from_params``.
    """

    item_ids: tuple[str, ...]
    adaptation_window: tuple[date, date]
    test_window: tuple[date, date]
    anchor_params: dict[str, Any]
    anchor_selection_profit: float
    adaptation_transition_rows: int
    rows: tuple[ReplayRow, ReplayRow]


def _windows_are_separate(adaptation: M5Slice, test: M5Slice) -> None:
    if (
        adaptation.start_date is None
        or adaptation.end_date is None
        or test.start_date is None
        or test.end_date is None
    ):
        raise ValueError("M5 windows must be dated")
    if adaptation.end_date >= test.start_date:
        raise ValueError(
            "adaptation window must end before the test window; the windows are not pooled"
        )


def _scoring_planner(planner, *, budget: int):
    """Same weights and seed, no fit. Smoke runs pass a smaller budget."""
    from src.jepa.plan import ReplenishmentPlanner

    return ReplenishmentPlanner(
        planner.dynamics,
        policy_seed=planner.policy_seed,
        delivery_lag=planner.delivery_lag,
        budget=budget,
        holding_rate=HOLDING_RATE,
        order_fee=ORDER_FEE,
        ration=planner.ration,
    )


def _row(label: str, planner_profit: float, anchor_profit: float) -> ReplayRow:
    cash = SHOP_CASH
    return ReplayRow(
        label=label,
        planner_net_profit=planner_profit,
        anchor_net_profit=anchor_profit,
        planner_normalised=planner_profit / cash,
        anchor_normalised=anchor_profit / cash,
    )


def replay_exam(
    adaptation: M5Slice,
    test: M5Slice,
    synthetic,
    *,
    n_trials: int = 150,
    epochs: int = 20,
    budget: int | None = None,
    sampler_seed: int = 0,
    fit_seed: int = 0,
) -> ReplayExam:
    """Score a zero-shot row and an adaptation row on ``test``.

    ``synthetic`` is already fit. This function does not fit it. The
    adaptation row refits the same architecture on reward-free transitions
    from ``adaptation`` only, collected with ``OrderUpToPolicy`` and
    ``ExploratoryPolicy``. The textbook anchor is tuned on ``adaptation``
    replay profit and then scored, unchanged, on ``test``. Neither row is
    a pass/fail gate.
    """
    from src.jepa.plan import DEFAULT_BUDGET, fit_latent
    from src.tuning.search_spaces import order_up_to_from_params

    _windows_are_separate(adaptation, test)
    proposal_budget = DEFAULT_BUDGET if budget is None else budget
    params, selection_profit = tune_anchor(
        adaptation,
        n_trials=n_trials,
        sampler_seed=sampler_seed,
    )
    anchor = order_up_to_from_params(params)
    anchor_on_test = shop_net_profit(test, anchor)
    zero_shot = shop_net_profit(test, _scoring_planner(synthetic, budget=proposal_budget))

    frame = collect_adaptation_log(adaptation, delivery_lag=int(synthetic.delivery_lag))
    adapted = fit_latent(
        frame,
        delivery_lag=int(synthetic.delivery_lag),
        epochs=epochs,
        seed=fit_seed,
        budget=proposal_budget,
        policy_seed=synthetic.policy_seed,
        holding_rate=HOLDING_RATE,
        order_fee=ORDER_FEE,
    )
    adaptation_profit = shop_net_profit(test, adapted)
    return ReplayExam(
        item_ids=tuple(adaptation.item_ids),
        adaptation_window=(adaptation.start_date, adaptation.end_date),
        test_window=(test.start_date, test.end_date),
        anchor_params=params,
        anchor_selection_profit=selection_profit,
        adaptation_transition_rows=len(frame),
        rows=(
            _row("zero_shot", zero_shot, anchor_on_test),
            _row("adaptation", adaptation_profit, anchor_on_test),
        ),
    )


def resolve_pinned_items(pinned: list[str], ranked: list[str]) -> list[str]:
    """Keep an existing pin. Otherwise use the rank from this quality report."""
    if pinned:
        return list(pinned)
    return list(ranked)


def pin_item_ids(notebook_path: str | Path, item_ids: list[str]) -> None:
    """Write ``item_ids`` into the notebook's ``PINNED_ITEM_IDS`` assignment.

    Call this only after ``quality_report`` has run. A later run then keeps
    the same five ids.
    """
    import nbformat

    path = Path(notebook_path)
    notebook = nbformat.read(path, as_version=4)
    rendered = "PINNED_ITEM_IDS: list[str] = " + repr(list(item_ids))
    found = False
    for cell in notebook.cells:
        if cell.cell_type != "code" or "PINNED_ITEM_IDS" not in cell.source:
            continue
        lines = []
        for line in cell.source.splitlines():
            if line.startswith("PINNED_ITEM_IDS"):
                lines.append(rendered)
                found = True
            else:
                lines.append(line)
        cell.source = "\n".join(lines) + "\n"
        break
    if not found:
        raise ValueError(f"{path} has no PINNED_ITEM_IDS assignment")
    nbformat.write(notebook, path)


def write_replay_exam(exam: ReplayExam, directory: str | Path) -> Path:
    """Write both rows. The table has no pass/fail column."""
    import pandas as pd

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [
            {
                "label": row.label,
                "planner_net_profit": row.planner_net_profit,
                "anchor_net_profit": row.anchor_net_profit,
                "planner_normalised": row.planner_normalised,
                "anchor_normalised": row.anchor_normalised,
                "test_window_start": exam.test_window[0].isoformat(),
                "test_window_end": exam.test_window[1].isoformat(),
                "adaptation_window_start": exam.adaptation_window[0].isoformat(),
                "adaptation_window_end": exam.adaptation_window[1].isoformat(),
                "anchor_selection_profit": exam.anchor_selection_profit,
            }
            for row in exam.rows
        ]
    )
    path = destination / "exam.parquet"
    frame.to_parquet(path, index=False)
    return path
