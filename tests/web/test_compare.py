"""CRN overlay helpers for the Streamlit demo — public-behavior tests."""

from __future__ import annotations

from pathlib import Path
from random import Random

from src.sim.distributions import Normal
from src.sim.inspect import flow_frame
from src.sim.node import DemandSinkNode
from src.sim.setup_io import load_setup
from src.sim.inspect import node_timeseries_df
from src.web.compare import (
    apply_knobs,
    identical_groups,
    run_crn_overlay,
    scorecard,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CHAIN = _REPO_ROOT / "setups" / "three_node_chain"


def test_apply_knobs_overlays_run_market_sink_and_disruption_in_memory():
    original = load_setup(_CHAIN)
    yaml_before = (_CHAIN / "setup.yaml").read_text(encoding="utf-8")

    overlaid = apply_knobs(
        original,
        world_seed=99,
        n_steps=40,
        demand_mean=22.0,
        cycle_amp=0.4,
        disruption=True,
    )

    assert original.world_seed == 42
    assert original.n_steps == 30
    assert original.market.cycle_amp == 0.03
    assert original.disruption.event_prob == 0.0
    assert (_CHAIN / "setup.yaml").read_text(encoding="utf-8") == yaml_before

    assert overlaid is not original
    assert overlaid.world_seed == 99
    assert overlaid.n_steps == 40
    assert overlaid.market.cycle_amp == 0.4
    assert overlaid.market.cycle_len == 20
    assert overlaid.disruption.event_prob == 0.2
    assert float(overlaid.disruption.severity.sample(Random(0))) > 0

    sinks = [ni.node for ni in overlaid.nodes if isinstance(ni.node, DemandSinkNode)]
    assert len(sinks) == 1
    dist = sinks[0].demand_dist
    assert isinstance(dist, Normal)
    assert dist.mean == 22.0
    orig_sink = next(ni.node for ni in original.nodes if isinstance(ni.node, DemandSinkNode))
    assert isinstance(orig_sink.demand_dist, Normal)
    assert orig_sink.demand_dist.mean == 10.0


def test_apply_knobs_caps_n_steps_at_sixty():
    original = load_setup(_CHAIN)
    overlaid = apply_knobs(
        original,
        world_seed=42,
        n_steps=200,
        demand_mean=10.0,
        cycle_amp=0.03,
        disruption=False,
    )
    assert overlaid.n_steps == 60
    assert overlaid.disruption.event_prob == 0.0


def _knobbed_chain():
    return apply_knobs(
        load_setup(_CHAIN),
        world_seed=42,
        n_steps=12,
        demand_mean=10.0,
        cycle_amp=0.03,
        disruption=False,
    )


_POLICY_PARAMS = {
    "cover_horizon_ticks": 10,
    "Q": 30,
    "review_interval": 7,
}


def test_run_crn_overlay_is_bit_identical_across_calls():
    scenario = _knobbed_chain()
    families = ["order_up_to", "reorder_point"]
    first = run_crn_overlay(scenario, families, _POLICY_PARAMS)
    second = run_crn_overlay(scenario, families, _POLICY_PARAMS)
    assert set(first) == set(families)
    assert first["order_up_to"]["ticks"] == second["order_up_to"]["ticks"]
    assert first["reorder_point"]["ticks"] == second["reorder_point"]["ticks"]


def test_run_crn_overlay_holds_world_fixed_and_changes_shop_1_outcomes():
    scenario = _knobbed_chain()
    logs = run_crn_overlay(
        scenario,
        ["order_up_to", "periodic_order_up_to"],
        _POLICY_PARAMS,
    )
    demand_a = logs["order_up_to"]["global"]["market_demand"]
    demand_b = logs["periodic_order_up_to"]["global"]["market_demand"]
    assert demand_a == demand_b

    cash_a = [t["node_cash"]["shop-1"] for t in logs["order_up_to"]["ticks"]]
    cash_b = [t["node_cash"]["shop-1"] for t in logs["periodic_order_up_to"]["ticks"]]
    assert cash_a != cash_b


def test_safety_fraction_moves_shop_1_without_changing_the_world():
    scenario = _knobbed_chain()
    thin = run_crn_overlay(
        scenario, ["order_up_to"], {**_POLICY_PARAMS, "safety_lead_pct_of_lag": 0.0}
    )
    fat = run_crn_overlay(
        scenario, ["order_up_to"], {**_POLICY_PARAMS, "safety_lead_pct_of_lag": 1.5}
    )
    assert (
        thin["order_up_to"]["global"]["market_demand"]
        == fat["order_up_to"]["global"]["market_demand"]
    )
    cash_thin = [t["node_cash"]["shop-1"] for t in thin["order_up_to"]["ticks"]]
    cash_fat = [t["node_cash"]["shop-1"] for t in fat["order_up_to"]["ticks"]]
    assert cash_thin != cash_fat


def test_scorecard_reports_shop_1_kpis_and_crn_delta_vs_order_up_to():
    scenario = _knobbed_chain()
    logs = run_crn_overlay(
        scenario,
        ["order_up_to", "periodic_order_up_to"],
        _POLICY_PARAMS,
    )
    card = scorecard(logs, scenario)
    assert set(card["family"]) == {"order_up_to", "periodic_order_up_to"}
    for col in (
        "fill_rate",
        "days_empty",
        "inventory_turnover",
        "net_profit",
        "rejections",
        "delta_net_profit",
    ):
        assert col in card.columns

    anchor = card.set_index("family").loc["order_up_to"]
    assert anchor["delta_net_profit"] == 0.0
    other = card.set_index("family").loc["periodic_order_up_to"]
    assert other["delta_net_profit"] == other["net_profit"] - anchor["net_profit"]
    assert (card["rejections"] >= 0).all()


def test_fill_rate_is_shop_1_sales_over_sink_demand_target():
    scenario = _knobbed_chain()
    params = {**_POLICY_PARAMS, "review_interval": 7}
    logs = run_crn_overlay(scenario, ["order_up_to", "periodic_order_up_to"], params)
    card = scorecard(logs, scenario).set_index("family")

    for family, log in logs.items():
        ff = flow_frame(log)
        sold = ff.loc[ff["node_id"] == "shop-1", "sales"].sum()
        wanted = ff.loc[ff["node_id"] == "sink-1", "demand"].sum()
        assert card.loc[family, "fill_rate"] == sold / wanted

    # A periodic policy that starves shop-1 (R=7 on a 2-tick lag) must not
    # read as perfect service.
    assert card.loc["periodic_order_up_to", "fill_rate"] < 0.5
    assert card.loc["periodic_order_up_to", "days_empty"] > 6


_TWO_SUPPLIERS = _REPO_ROOT / "setups" / "demo_two_suppliers"


def test_demo_two_suppliers_loads_and_runs_with_shop_1_focal():
    scenario = load_setup(_TWO_SUPPLIERS)
    assert len(scenario.nodes) == 4
    ids = {ni.node.id for ni in scenario.nodes}
    assert ids == {"f-lo", "f-hi", "shop-1", "sink-1"}
    assert scenario.market.cycle_len == 20
    assert scenario.disruption.event_prob == 0.0

    overlaid = apply_knobs(
        scenario,
        world_seed=scenario.world_seed,
        n_steps=8,
        demand_mean=15.0,
        cycle_amp=0.0,
        disruption=False,
    )
    logs = run_crn_overlay(overlaid, ["order_up_to"], _POLICY_PARAMS)
    assert logs["order_up_to"]["n_steps"] == 8
    last = logs["order_up_to"]["ticks"][-1]
    assert set(last["node_cash"]) == ids


_TWO_SHOPS = _REPO_ROOT / "setups" / "two_factories_two_shops"


def test_two_shops_runs_with_shop_1_focal_and_fill_rate_from_sink_1_only():
    overlaid = apply_knobs(
        load_setup(_TWO_SHOPS),
        world_seed=42,
        n_steps=20,
        demand_mean=10.0,
        cycle_amp=0.2,
        disruption=False,
    )
    logs = run_crn_overlay(overlaid, ["order_up_to"], _POLICY_PARAMS)
    ff = flow_frame(logs["order_up_to"])
    sold = ff.loc[ff["node_id"] == "shop-1", "sales"].sum()
    wanted = ff.loc[ff["node_id"] == "sink-1", "demand"].sum()
    card = scorecard(logs, overlaid)
    assert card.iloc[0]["fill_rate"] == sold / wanted


def _two_shops(**knobs):
    base = dict(world_seed=42, n_steps=30, demand_mean=10.0, cycle_amp=0.2, disruption=False)
    return apply_knobs(load_setup(_TWO_SHOPS), **{**base, **knobs})


_ALL = ["order_up_to", "reorder_point", "periodic_order_up_to", "periodic_reorder"]


def test_two_shops_row_does_not_depend_on_which_other_policies_run():
    # Regression: shop-2's setup policy object was shared across family runs,
    # so its demand history leaked from one run into the next.
    params = {**_POLICY_PARAMS, "review_interval": 3}
    together = scorecard(run_crn_overlay(_two_shops(), _ALL, params), _two_shops())
    for family in _ALL:
        alone = scorecard(run_crn_overlay(_two_shops(), [family], params), _two_shops())
        row = together.set_index("family").loc[family]
        assert alone.iloc[0]["net_profit"] == row["net_profit"]
        assert alone.iloc[0]["fill_rate"] == row["fill_rate"]


def test_scorecard_can_score_shop_2_on_the_two_shop_graph():
    scenario = _two_shops()
    logs = run_crn_overlay(scenario, ["order_up_to"], _POLICY_PARAMS)
    card = scorecard(logs, scenario, node_id="shop-2")
    ff = flow_frame(logs["order_up_to"])
    sold = ff.loc[ff["node_id"] == "shop-2", "sales"].sum()
    wanted = ff.loc[ff["node_id"] == "sink-2", "demand"].sum()
    assert card.iloc[0]["fill_rate"] == sold / wanted


def test_days_empty_counts_days_shop_1_closes_with_no_stock():
    scenario = _knobbed_chain()
    logs = run_crn_overlay(scenario, _ALL, _POLICY_PARAMS)
    card = scorecard(logs, scenario).set_index("family")
    for family, log in logs.items():
        ts = node_timeseries_df(log, scenario)
        shop = ts[ts["node_id"] == "shop-1"]
        assert card.loc[family, "days_empty"] == int((shop["inventory_total"] == 0).sum())


def test_identical_groups_names_rows_with_the_same_outcome():
    scenario = _knobbed_chain()
    params = {**_POLICY_PARAMS, "review_interval": 14}
    card = scorecard(run_crn_overlay(scenario, _ALL, params), scenario)
    groups = identical_groups(card)
    assert ["periodic_order_up_to", "periodic_reorder"] in groups
    assert all(len(g) > 1 for g in groups)


def test_web_package_does_not_import_rl_tuning_or_llm():
    web_root = _REPO_ROOT / "src" / "web"
    forbidden = ("src.rl", "src.tuning", "src.llm")
    for path in web_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for name in forbidden:
            assert f"import {name}" not in text
            assert f"from {name}" not in text
