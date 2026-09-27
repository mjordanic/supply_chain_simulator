"""M5 replay exam.

Seams, from the issue acceptance criteria:

1. Skip. With the raw directory missing, the notebook names the Kaggle
   download and the ADR 0028 slice rule, and does not raise. No ``data/m5``.
2. Selection. Rank on 2015-01-01 through 2015-12-31 only: ``FOODS_3`` at
   ``CA_1``, shortest ``longest_zero_run``, then highest mean sales. Drop a
   zero run longer than 7 days. If fewer than five survive, widen to all
   FOODS departments at ``CA_1``.
3. Graph. ``StaticFactoryPolicy`` factories feed ``shop-CA_1`` at lead time
   3, one replay sink per item, ``unit_cost`` at 40% of median price,
   ``flat_world`` on. The learned policy and ``OrderUpToPolicy`` are wrapped
   in ``PriceReplayPolicy`` on the same observed prices.
4. Windows. The test window is 2016-01-01 through 2016-04-24. Zero-shot
   uses the synthetic planner and does not fit it on either window.
   Adaptation collects reward-free transitions on the earlier window only,
   from ``OrderUpToPolicy`` and ``ExploratoryPolicy``, and fits that frame.
   The windows are not pooled.
5. Anchor. ``OrderUpToPolicy`` is tuned on adaptation-window replay profit
   via ``order_up_to_space``. The test window is not an input to that search.
6. Report. Both rows are returned. Neither is a pass/fail gate.

Assumption recorded in the commit: a factory makes one product, so the
upstream echelon is one ``StaticFactoryPolicy`` factory per pinned item,
all at lead time 3, and no distribution centre.
"""

from __future__ import annotations

import io
import os
from contextlib import redirect_stdout
from datetime import date, timedelta
from pathlib import Path

import nbformat

from src.datasets.m5 import M5Slice

NOTEBOOK = Path("notebooks/11-jepa-m5.ipynb")


def _slice(items: list[dict], *, start: date = date(2015, 1, 1)) -> M5Slice:
    sales: dict[tuple[str, str], list[int]] = {}
    prices: dict[tuple[str, str], list[float]] = {}
    depts: dict[str, str] = {}
    item_ids: list[str] = []
    store_ids: list[str] = []
    for item in items:
        key = (item["item_id"], item["store_id"])
        series = list(item["sales"])
        sales[key] = series
        prices[key] = list(item.get("prices", [1.0] * len(series)))
        depts[item["item_id"]] = item["dept"]
        if item["item_id"] not in item_ids:
            item_ids.append(item["item_id"])
        if item["store_id"] not in store_ids:
            store_ids.append(item["store_id"])
    n_ticks = len(items[0]["sales"])
    return M5Slice(
        sales=sales,
        prices=prices,
        start_date=start,
        end_date=start + timedelta(days=n_ticks - 1),
        item_ids=item_ids,
        store_ids=store_ids,
        item_dept=depts,
        store_state={store_id: "CA" for store_id in store_ids},
    )


def _run_notebook() -> str:
    """Execute the notebook's code cells in-process. Magics are not used."""
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    namespace: dict = {"__name__": "__main__"}
    stdout = io.StringIO()
    previous = Path.cwd()
    try:
        with redirect_stdout(stdout):
            for cell in notebook.cells:
                if cell.cell_type != "code":
                    continue
                exec(compile(cell.source, str(NOTEBOOK), "exec"), namespace)
    finally:
        os.chdir(previous)
    return stdout.getvalue()


def test_notebook_skips_when_the_raw_directory_is_missing(tmp_path, monkeypatch):
    missing = tmp_path / "no-such-m5"
    monkeypatch.setenv("JEPA_M5_DATA_DIR", str(missing))
    output = _run_notebook()
    assert "kaggle competitions download -c m5-forecasting-accuracy" in output
    assert "ADR 0028" in output
    assert "2015-01-01" in output and "2015-12-31" in output
    assert "FOODS_3" in output and "CA_1" in output
    assert "7" in output
    assert "2016-01-01" in output and "2016-04-24" in output


def test_rank_keeps_five_foods_3_items_by_short_zero_run_then_mean_sales():
    from src.jepa.m5 import rank_foods_items

    # Eligible FOODS_3 at CA_1, plus a long zero run, another store, and another department.
    window = _slice([
        {"item_id": "FOODS_3_C", "store_id": "CA_1", "dept": "FOODS_3", "sales": [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]},
        {"item_id": "FOODS_3_A", "store_id": "CA_1", "dept": "FOODS_3", "sales": [0, 0, 10, 10, 10, 10, 10, 10, 10, 10]},
        {"item_id": "FOODS_3_B", "store_id": "CA_1", "dept": "FOODS_3", "sales": [0, 0, 1, 1, 1, 1, 1, 1, 1, 1]},
        {"item_id": "FOODS_3_F", "store_id": "CA_1", "dept": "FOODS_3", "sales": [0, 0, 0, 5, 5, 5, 5, 5, 5, 5]},
        {"item_id": "FOODS_3_G", "store_id": "CA_1", "dept": "FOODS_3", "sales": [0, 0, 0, 0, 4, 4, 4, 4, 4, 4]},
        {"item_id": "FOODS_3_H", "store_id": "CA_1", "dept": "FOODS_3", "sales": [0, 0, 0, 0, 0, 9, 9, 9, 9, 9]},
        {"item_id": "FOODS_3_D", "store_id": "CA_1", "dept": "FOODS_3", "sales": [0, 0, 0, 0, 0, 0, 0, 0, 9, 9]},
        {"item_id": "FOODS_3_OTHER", "store_id": "CA_2", "dept": "FOODS_3", "sales": [8, 8, 8, 8, 8, 8, 8, 8, 8, 8]},
        {"item_id": "HOUSEHOLD_1_X", "store_id": "CA_1", "dept": "HOUSEHOLD_1", "sales": [8, 8, 8, 8, 8, 8, 8, 8, 8, 8]},
    ])
    assert rank_foods_items(window) == [
        "FOODS_3_C",
        "FOODS_3_A",
        "FOODS_3_B",
        "FOODS_3_F",
        "FOODS_3_G",
    ]


def test_rank_widens_to_all_foods_departments_when_foods_3_is_short():
    from src.jepa.m5 import rank_foods_items

    steady = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    long_gap = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    window = _slice([
        {"item_id": "FOODS_3_A", "store_id": "CA_1", "dept": "FOODS_3", "sales": steady},
        {"item_id": "FOODS_3_B", "store_id": "CA_1", "dept": "FOODS_3", "sales": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]},
        {"item_id": "FOODS_3_C", "store_id": "CA_1", "dept": "FOODS_3", "sales": [3, 0, 3, 0, 3, 0, 3, 0, 3, 0]},
        {"item_id": "FOODS_3_LONG", "store_id": "CA_1", "dept": "FOODS_3", "sales": long_gap},
        {"item_id": "FOODS_2_A", "store_id": "CA_1", "dept": "FOODS_2", "sales": [9, 9, 9, 9, 9, 9, 9, 9, 9, 9]},
        {"item_id": "FOODS_1_A", "store_id": "CA_1", "dept": "FOODS_1", "sales": [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]},
        {"item_id": "HOUSEHOLD_1_X", "store_id": "CA_1", "dept": "HOUSEHOLD_1", "sales": steady},
    ])
    chosen = rank_foods_items(window)
    assert chosen == ["FOODS_2_A", "FOODS_1_A", "FOODS_3_A", "FOODS_3_B", "FOODS_3_C"]
    assert "FOODS_3_LONG" not in chosen
    assert "HOUSEHOLD_1_X" not in chosen


def _write_csv(path: Path, header: list[str], rows: list[dict]) -> None:
    import csv

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def _selection_fixture(tmp_path: Path) -> Path:
    """2014 zeros would drop FOODS_3_KEEP if the windows were pooled.

    2015-01-01 through 2015-01-05 is clean for that item. Four other FOODS_3
    items are clean in 2015, so a 2015-only rank stays inside FOODS_3.
    FOODS_1_WIDEN is the item a widened rank would pull in.
    """
    data_dir = tmp_path / "m5"
    data_dir.mkdir()
    dates = [date(2014, 12, 24) + timedelta(days=offset) for offset in range(13)]
    # 2014-12-24 .. 2014-12-31 (8 days) then 2015-01-01 .. 2015-01-05 (5 days).
    calendar_rows = [
        {
            "date": day.isoformat(),
            "wm_yr_wk": "11451" if day.year == 2014 else "11501",
            "weekday": "Monday",
            "wday": 2,
            "month": day.month,
            "year": day.year,
            "d": f"d_{index + 1}",
            "event_name_1": "",
            "event_type_1": "",
            "snap_CA": 0,
            "snap_TX": 0,
            "snap_WI": 0,
        }
        for index, day in enumerate(dates)
    ]
    calendar_header = [
        "date", "wm_yr_wk", "weekday", "wday", "month", "year", "d",
        "event_name_1", "event_type_1", "snap_CA", "snap_TX", "snap_WI",
    ]
    _write_csv(data_dir / "calendar.csv", calendar_header, calendar_rows)

    day_cols = [f"d_{index + 1}" for index in range(13)]
    early_zeros = [0] * 8
    clean = [5, 5, 5, 5, 5]
    items = {
        "FOODS_3_KEEP": early_zeros + clean,
        "FOODS_3_A": early_zeros + [4, 4, 4, 4, 4],
        "FOODS_3_B": early_zeros + [3, 3, 3, 3, 3],
        "FOODS_3_C": early_zeros + [2, 2, 2, 2, 2],
        "FOODS_3_D": early_zeros + [1, 1, 1, 1, 1],
        "FOODS_1_WIDEN": [9] * 13,
    }
    sales_rows = []
    for item_id, series in items.items():
        dept = "FOODS_1" if item_id.startswith("FOODS_1") else "FOODS_3"
        row = {
            "id": f"{item_id}_CA_1_validation",
            "item_id": item_id,
            "dept_id": dept,
            "cat_id": "FOODS",
            "store_id": "CA_1",
            "state_id": "CA",
        }
        for column, value in zip(day_cols, series):
            row[column] = value
        sales_rows.append(row)
    sales_header = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"] + day_cols
    _write_csv(data_dir / "sales_train_validation.csv", sales_header, sales_rows)

    price_rows = [
        {"store_id": "CA_1", "item_id": item_id, "wm_yr_wk": week, "sell_price": 2.0}
        for item_id in items
        for week in ("11451", "11501")
    ]
    _write_csv(
        data_dir / "sell_prices.csv",
        ["store_id", "item_id", "wm_yr_wk", "sell_price"],
        price_rows,
    )
    return data_dir


def test_choose_items_uses_the_2015_window_only(tmp_path):
    from src.jepa.m5 import choose_items

    chosen = choose_items(_selection_fixture(tmp_path))
    assert chosen[0] == "FOODS_3_KEEP"
    assert "FOODS_1_WIDEN" not in chosen
    assert len(chosen) == 5


def test_replay_graph_feeds_shop_ca_1_at_lead_time_3_under_flat_world():
    import pytest

    from src.jepa.m5 import SHOP_ID, build_replay_scenario
    from src.sim.node import FactoryNode, IntermediateNode
    from src.sim.policy import StaticFactoryPolicy
    from src.sim.replay_demand_sink import ReplayDemandSinkNode
    from src.sim.runner import Runner

    prices_a = [1.0, 3.0, 5.0, 7.0, 9.0]
    window = _slice([
        {
            "item_id": "FOODS_3_A",
            "store_id": "CA_1",
            "dept": "FOODS_3",
            "sales": [1, 2, 3, 4, 5],
            "prices": prices_a,
        },
        {
            "item_id": "FOODS_3_B",
            "store_id": "CA_1",
            "dept": "FOODS_3",
            "sales": [2, 2, 2, 2, 2],
            "prices": [4.0, 4.0, 4.0, 4.0, 4.0],
        },
    ])
    scenario = build_replay_scenario(window)
    factories = [ni for ni in scenario.nodes if isinstance(ni.node, FactoryNode)]
    shops = [ni for ni in scenario.nodes if isinstance(ni.node, IntermediateNode)]
    sinks = [ni for ni in scenario.nodes if isinstance(ni.node, ReplayDemandSinkNode)]

    assert [shop.node.id for shop in shops] == [SHOP_ID]
    assert {factory.node.produces_product_id for factory in factories} == {"FOODS_3_A", "FOODS_3_B"}
    assert all(isinstance(factory.policy, StaticFactoryPolicy) for factory in factories)
    assert {sink.node.product_id for sink in sinks} == {"FOODS_3_A", "FOODS_3_B"}
    factory_edges = [edge for edge in scenario.edges if edge.buyer_id == SHOP_ID]
    assert len(factory_edges) == 2
    assert {edge.default_lead_time for edge in factory_edges} == {3}
    assert all(not node.node.id.startswith("dc") for node in scenario.nodes)

    costs = {ware.product_id: ware.unit_cost for ware in scenario.catalog}
    assert costs["FOODS_3_A"] == pytest.approx(0.4 * 5.0)
    assert costs["FOODS_3_B"] == pytest.approx(0.4 * 4.0)

    log = Runner(scenario).run()
    demand = {}
    for tick_log in log["ticks"]:
        for flow in tick_log["node_flows"]:
            if flow["node_id"].startswith("sink-"):
                demand.setdefault(flow["pid"], []).append(flow["demand"])
    assert demand["FOODS_3_A"] == [1, 2, 3, 4, 5]
    assert demand["FOODS_3_B"] == [2, 2, 2, 2, 2]


def _shop_prices(log, product_id: str) -> list[float]:
    from src.jepa.m5 import SHOP_ID

    prices = []
    for tick_log in log["ticks"]:
        for flow in tick_log["node_flows"]:
            if flow["node_id"] == SHOP_ID and flow["pid"] == product_id:
                prices.append(float(flow["price"]))
    return prices


def test_both_ordering_policies_charge_the_same_observed_prices():
    import pytest

    from src.jepa.m5 import run_shop
    from src.jepa.policy import ExploratoryPolicy
    from src.sim.policy import OrderUpToPolicy, PriceReplayPolicy

    observed = [2.0, 4.0, 6.0, 8.0, 10.0]
    window = _slice([
        {
            "item_id": "FOODS_3_A",
            "store_id": "CA_1",
            "dept": "FOODS_3",
            "sales": [1, 1, 1, 1, 1],
            "prices": observed,
        },
    ])
    anchor_log = run_shop(window, OrderUpToPolicy())
    learned_log = run_shop(window, ExploratoryPolicy(policy_seed=1, delivery_lag=3))
    assert isinstance(anchor_log.policy, PriceReplayPolicy)
    assert isinstance(learned_log.policy, PriceReplayPolicy)
    assert _shop_prices(anchor_log.log, "FOODS_3_A") == pytest.approx(observed)
    assert _shop_prices(learned_log.log, "FOODS_3_A") == pytest.approx(observed)


def _window(n_ticks: int, start: date, sales: list[int] | None = None):
    series = list(sales) if sales is not None else [1] * n_ticks
    return _slice(
        [
            {
                "item_id": "FOODS_3_A",
                "store_id": "CA_1",
                "dept": "FOODS_3",
                "sales": series,
                "prices": [2.0, 4.0] * ((n_ticks + 1) // 2),
            },
            {
                "item_id": "FOODS_3_B",
                "store_id": "CA_1",
                "dept": "FOODS_3",
                "sales": series,
                "prices": [3.0] * n_ticks,
            },
        ],
        start=start,
    )


def test_adaptation_log_uses_both_behaviors_and_only_that_window():
    from src.jepa.m5 import collect_adaptation_log

    window = _window(4, date(2015, 1, 1))
    frame = collect_adaptation_log(window)
    assert set(frame["behavior"]) == {"order_up_to", "exploratory"}
    assert len(frame) == 2 * 2 * (4 - 1)


def test_anchor_search_maximises_profit_on_the_window_it_is_given():
    import pytest

    from src.jepa.m5 import shop_net_profit, tune_anchor
    from src.tuning.search_spaces import order_up_to_from_params

    window = _window(4, date(2015, 1, 1), sales=[1, 2, 3, 4])
    params, selection_profit = tune_anchor(window, n_trials=1, sampler_seed=0)
    policy = order_up_to_from_params(params)
    assert selection_profit == pytest.approx(shop_net_profit(window, policy))


def test_published_windows_follow_adr_0028():
    from src.jepa.m5 import SELECTION_END, SELECTION_START, TEST_END, TEST_START

    assert (SELECTION_START, SELECTION_END) == (date(2015, 1, 1), date(2015, 12, 31))
    assert (TEST_START, TEST_END) == (date(2016, 1, 1), date(2016, 4, 24))


def test_replay_exam_reports_both_rows_and_does_not_fit_the_synthetic_planner():
    import pytest
    import torch

    from src.jepa.log import collect_transition_log
    from src.jepa.m5 import SHOP_CASH, replay_exam, shop_net_profit
    from src.jepa.plan import fit_latent
    from src.tuning.search_spaces import order_up_to_from_params

    synthetic_log = collect_transition_log(
        n_episodes_each=1,
        episode_length=6,
        k_active=2,
        delivery_lag=3,
        seed_offset=14_000_000,
    )
    synthetic = fit_latent(
        synthetic_log,
        delivery_lag=3,
        epochs=1,
        seed=0,
        policy_seed=1,
        budget=4,
    )
    before = {key: value.detach().clone() for key, value in synthetic.dynamics.state_dict().items()}
    adaptation = _window(4, date(2015, 1, 1), sales=[1, 1, 2, 2])
    held_out = _window(3, date(2016, 1, 1), sales=[8, 8, 8])

    exam = replay_exam(
        adaptation,
        held_out,
        synthetic,
        n_trials=1,
        epochs=1,
        budget=4,
        sampler_seed=0,
    )

    assert {row.label for row in exam.rows} == {"zero_shot", "adaptation"}
    assert exam.adaptation_window == (date(2015, 1, 1), date(2015, 1, 4))
    assert exam.test_window == (date(2016, 1, 1), date(2016, 1, 3))
    assert exam.adaptation_transition_rows == 2 * 2 * (4 - 1)
    for key, value in before.items():
        assert torch.equal(value, synthetic.dynamics.state_dict()[key])

    anchor_on_adaptation = order_up_to_from_params(exam.anchor_params)
    anchor_on_test = order_up_to_from_params(exam.anchor_params)
    adaptation_profit = shop_net_profit(adaptation, anchor_on_adaptation)
    test_profit = shop_net_profit(held_out, anchor_on_test)
    assert exam.anchor_selection_profit == pytest.approx(adaptation_profit)
    assert adaptation_profit != pytest.approx(test_profit)
    for row in exam.rows:
        assert row.anchor_net_profit == pytest.approx(test_profit)
        assert row.planner_normalised == pytest.approx(row.planner_net_profit / SHOP_CASH)
        assert row.anchor_normalised == pytest.approx(row.anchor_net_profit / SHOP_CASH)
    assert not hasattr(exam, "passes")

    with pytest.raises(ValueError, match="not pooled"):
        replay_exam(adaptation, adaptation, synthetic, n_trials=1, epochs=1, budget=4)


def test_pinned_ids_stay_put_until_a_quality_report_writes_them(tmp_path):
    from src.jepa.m5 import pin_item_ids, resolve_pinned_items

    assert resolve_pinned_items(["FOODS_3_001"], ["FOODS_3_999"]) == ["FOODS_3_001"]
    assert resolve_pinned_items([], ["FOODS_3_999"]) == ["FOODS_3_999"]

    notebook_path = tmp_path / "exam.ipynb"
    notebook_path.write_text(
        '{"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": [{"cell_type": "code", '
        '"id": "pin", "metadata": {}, "execution_count": null, "outputs": [], "source": '
        '["PINNED_ITEM_IDS: list[str] = []\\n"]}]}',
        encoding="utf-8",
    )
    pin_item_ids(notebook_path, ["FOODS_3_090", "FOODS_3_586"])
    pinned = nbformat.read(notebook_path, as_version=4)
    assert pinned.cells[0].source.startswith(
        "PINNED_ITEM_IDS: list[str] = ['FOODS_3_090', 'FOODS_3_586']"
    )

    tracked = NOTEBOOK.read_text(encoding="utf-8")
    assert "PINNED_ITEM_IDS: list[str] = []" in tracked
    assert tracked.index("quality_report(selection)") < tracked.index("pin_item_ids(")


def test_written_exam_has_both_rows_and_no_pass_column(tmp_path):
    import pandas as pd

    from src.jepa.m5 import ReplayExam, ReplayRow, write_replay_exam

    exam = ReplayExam(
        item_ids=("FOODS_3_A",),
        adaptation_window=(date(2015, 1, 1), date(2015, 12, 31)),
        test_window=(date(2016, 1, 1), date(2016, 4, 24)),
        anchor_params={},
        anchor_selection_profit=1.0,
        adaptation_transition_rows=4,
        rows=(
            ReplayRow("zero_shot", 1.0, 2.0, 0.1, 0.2),
            ReplayRow("adaptation", 3.0, 2.0, 0.3, 0.2),
        ),
    )
    path = write_replay_exam(exam, tmp_path)
    frame = pd.read_parquet(path)
    assert set(frame["label"]) == {"zero_shot", "adaptation"}
    assert "pass" not in frame.columns
    assert "passes" not in frame.columns


def test_m5_exam_does_not_import_the_rl_stack():
    import subprocess
    import sys

    text = Path("src/jepa/m5.py").read_text(encoding="utf-8")
    assert "src.rl" not in text
    script = (
        "import src.jepa.m5, sys\n"
        "bad = [name for name in sys.modules if name == 'src.rl' or name.startswith('src.rl.')]\n"
        "assert not bad, bad\n"
    )
    subprocess.check_call([sys.executable, "-c", script])
