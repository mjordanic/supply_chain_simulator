"""Streamlit CRN policy lab — sibling consumer of ``src.sim`` (ADR 0023)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.sim.inspect import node_timeseries_df
from src.sim.setup_io import load_setup
from src.web.compare import (
    DEMO_CYCLE_LEN,
    DISRUPTION_DURATION,
    DISRUPTION_ON_PROB,
    DISRUPTION_SEVERITY,
    FAMILY_LABELS,
    FOCAL_NODE_ID,
    MAX_N_STEPS,
    TEXTBOOK_FAMILIES,
    apply_knobs,
    focal_family,
    run_crn_overlay,
    scorecard,
)
from src.web.dag import animated_dag
from src.web.playback import playback_frames

GITHUB_URL = "https://github.com/mjordanic/supply_chain_simulator"

PRESETS = {
    "Linear chain (3 nodes)": _REPO_ROOT / "setups" / "three_node_chain",
    "Two suppliers (4 nodes)": _REPO_ROOT / "setups" / "demo_two_suppliers",
}

PRESET_HELP = {
    "Linear chain (3 nodes)": (
        "factory-1 → shop-1 → sink-1, one product. "
        "Lead times are on the edges (not sliders): 2 ticks factory→shop, "
        "1 tick shop→sink. The shop's reorder math uses inbound L = 2."
    ),
    "Two suppliers (4 nodes)": (
        "f-lo (cheap, 4-tick lead) and f-hi (fast but pricey, 2-tick lead) both "
        "supply shop-1 → sink-1 (1 tick). Lead times stay on those edges; "
        "the shop routes on the live offer book."
    ),
}

FAMILY_HELP = {
    "order_up_to": (
        "Every tick, if inventory position is below s, order up to S. "
        "The CRN comparison anchor — continuous review (s, S)."
    ),
    "reorder_point": (
        "Every tick, if position is below s, order a fixed batch Q. "
        "Continuous review (s, Q) — fewer, lumpier replenishments than (s, S)."
    ),
    "periodic_order_up_to": (
        "Only looks every R ticks; if reviewing, order up to S. "
        "Periodic review (R, S) — ignores the shelf between reviews."
    ),
    "periodic_reorder": (
        "Every R ticks, order up to S only if position is also below s. "
        "Periodic (R, s, S) — can skip a review when stock is still healthy."
    ),
}


def _series_figure(run_logs: dict, scenario) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
            f"{FOCAL_NODE_ID} cash (higher is better)",
            f"{FOCAL_NODE_ID} on-hand inventory",
        ),
    )
    for family, run_log in run_logs.items():
        ts = node_timeseries_df(run_log, scenario)
        shop = ts[ts["node_id"] == FOCAL_NODE_ID].sort_values("tick")
        label = FAMILY_LABELS.get(family, family)
        fig.add_trace(
            go.Scatter(
                x=shop["tick"],
                y=shop["cash"],
                name=label,
                legendgroup=family,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=shop["tick"],
                y=shop["inventory_total"],
                name=label,
                legendgroup=family,
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    fig.update_layout(
        height=460,
        margin=dict(l=50, r=20, t=40, b=80),
        legend=dict(
            title_text="Policy",
            orientation="h",
            y=-0.14,
            x=0,
            font=dict(size=11),
        ),
    )
    fig.update_yaxes(title_text="cash", row=1, col=1)
    fig.update_yaxes(title_text="units", row=2, col=1)
    fig.update_xaxes(title_text="tick", row=2, col=1)
    return fig


def _format_scorecard(card: pd.DataFrame) -> pd.DataFrame:
    out = card.copy()
    out["Policy"] = out["family"].map(lambda f: FAMILY_LABELS.get(f, f))
    out["Net profit"] = out["net_profit"].round(2)
    out["Δ vs (s, S)"] = out["delta_net_profit"].round(2)
    out["Service level"] = out["service_level"].round(3)
    out["Stockout rate"] = out["stockout_rate"].round(3)
    out["Inventory turnover"] = out["inventory_turnover"].round(2)
    out["Rejections"] = out["rejections"]
    return out[
        [
            "Policy",
            "Net profit",
            "Δ vs (s, S)",
            "Service level",
            "Stockout rate",
            "Inventory turnover",
            "Rejections",
        ]
    ]


st.set_page_config(page_title="Supply-chain policy lab", layout="wide")
st.title("Supply-chain policy lab")
st.markdown(
    f"""
This page is a **small, limited visual representation** of the simulator:
two canned graphs, four textbook inventory policies on **{FOCAL_NODE_ID}**,
and a short Common-Random-Numbers run (at most {MAX_N_STEPS} ticks).

The engine behind it can do much more — arbitrary DAGs, LLM-generated catalogs,
Optuna tuning, PPO, M5 demand replay, CLI parquet exports, and longer runs.

**Full project:** [{GITHUB_URL.removeprefix("https://")}]({GITHUB_URL})
"""
)

with st.expander("How this demo works"):
    st.markdown(
        f"""
**Compare** loads one of two setup directories (`catalog.csv` + `setup.yaml`),
overlays the sidebar knobs **in memory** (nothing on disk changes), and runs
each checked textbook family through the same `Runner` as the CLI. Only
**{FOCAL_NODE_ID}** is swapped. Factories and the demand sink keep the policies
from the setup.

### What you can change

| Sidebar | What it does in *this* demo |
|---|---|
| Preset | Picks the graph. Topology, SKUs, prices, capacities, and **lead times** come with it. |
| `world_seed` | Shared world RNG (market, demand draws, disruptions). Same seed + same knobs → bit-identical world. |
| `n_steps` | How many ticks (simulated days) to run. Capped at **{MAX_N_STEPS}** here; the CLI has no such cap. |
| Sink demand mean | Overwrites the sink's demand-distribution **mean**. Std. deviation stays as in the setup. |
| Seasonal `cycle_amp` | Amplitude of a sine wave on market demand. **Period is locked at {DEMO_CYCLE_LEN} ticks** (not a slider) so a short run can show a cycle. |
| Disruptions on | Off: no events. On: each tick has a {DISRUPTION_ON_PROB:.0%} chance of a regional shock (severity {DISRUPTION_SEVERITY}, duration {DISRUPTION_DURATION} ticks). A short run can still draw zero events. |
| Policy checkboxes | Which textbook families to attach to **{FOCAL_NODE_ID}**. Check one to run; check two or more to compare. |
| `k`, `H`, `Q`, `R` | Shop-1 policy hyperparameters (see the sidebar). |

### What is *not* a knob

**Lead time lives on the graph edges.** There is no lead-time slider. An order
placed at tick `t` arrives at `t` + that edge's lag — never the same tick.

| Preset | Graph | Inbound lags (used in s) | Shop → sink |
|---|---|---|---|
| Linear chain | factory-1 → shop-1 → sink-1 | 2 ticks | 1 tick |
| Two suppliers | f-lo and f-hi → shop-1 → sink-1 | 4 ticks (cheap) / 2 ticks (fast) | 1 tick |

Also frozen here: catalog, list prices, node cash/capacity, factory policy,
elasticity, promos, and trend. Unmet demand is a **lost sale** (a rejection),
not a backorder.

### CRN, scorecard, playback

All selected families share one `world_seed`, so scorecard differences are the
policy, not luck. The table is **{FOCAL_NODE_ID}** only. DAG playback is a
Plotly animation of an already-finished run — Play / Skip to end — not a live
`tick()` stream. Playback uses (s, S) when that family is selected, otherwise
the first checked family. Cash & inventory plots every selected family.

### Not in this UI

Custom graphs, the LLM world generator, Optuna, PPO / `RLNodePolicy`, M5
demand replay, parquet download, and the Central Table debugger. Those live
in the [GitHub repo]({GITHUB_URL}).
"""
    )

with st.sidebar:
    st.header("Setup")
    st.caption(
        "Two canned graphs. Topology and lead times are part of the preset, "
        "not sliders."
    )
    preset_name = st.selectbox(
        "Preset",
        list(PRESETS),
        help=(
            "Linear chain: factory → shop → sink, inbound lead time 2 ticks. "
            "Two suppliers: cheap/slow factory (4 ticks) and fast/pricey factory "
            "(2 ticks) into shop-1. Lead time is not a knob in this demo."
        ),
    )
    st.caption(PRESET_HELP[preset_name])
    world_seed = st.number_input(
        "world_seed",
        min_value=0,
        value=42,
        step=1,
        help=(
            "Seed for the shared world RNG (market shocks, demand draws, disruptions). "
            "Same seed + same knobs → bit-identical world. Change it to re-roll the noise. "
            "Policy choice does not consume this stream (CRN)."
        ),
    )
    n_steps = st.slider(
        "n_steps",
        min_value=1,
        max_value=MAX_N_STEPS,
        value=30,
        help=(
            f"Ticks (simulated days) in this run. This UI caps at {MAX_N_STEPS} "
            "to keep the demo cheap; `main.py run` has no such cap. "
            "Inbound lead times on these graphs are 2–4 ticks, so 30 ticks is "
            "enough to see a few replenishment cycles."
        ),
    )

    st.header("World knobs")
    st.caption(
        "These overlay the setup in memory. Lead time, catalog, and prices stay "
        f"as in the graph. Seasonal period is locked at {DEMO_CYCLE_LEN} ticks."
    )
    demand_mean = st.slider(
        "Sink demand mean",
        min_value=1.0,
        max_value=40.0,
        value=10.0,
        step=1.0,
        help=(
            "Mean of the demand sink's per-tick demand distribution (units the customer "
            "tries to buy). Overwrites the setup mean; std. deviation is unchanged. "
            "Higher mean stresses the shop: more stockouts if the policy under-orders, "
            "more inventory if it over-orders."
        ),
    )
    cycle_amp = st.slider(
        "Seasonal cycle_amp",
        min_value=0.0,
        max_value=0.8,
        value=0.2,
        step=0.05,
        help=(
            f"Amplitude of the seasonal sine wave on market demand. The period is "
            f"fixed at {DEMO_CYCLE_LEN} ticks in this demo (not a slider) so a short "
            "run can show a cycle. 0 = flat season; 0.4 is a clearly visible wave. "
            "Elasticity, promos, and trend are not exposed here."
        ),
    )
    disruption = st.checkbox(
        "Disruptions on",
        value=False,
        help=(
            f"On/off only — not a severity slider. When on, each tick has a "
            f"{DISRUPTION_ON_PROB:.0%} chance of a regional shock (severity "
            f"{DISRUPTION_SEVERITY}, duration {DISRUPTION_DURATION} ticks) that cuts "
            "demand and supply. Off = event probability 0. A given seed may still "
            "draw zero events in a short run."
        ),
    )

    st.header("Policies on shop-1")
    st.caption(
        f"Only **{FOCAL_NODE_ID}** is swapped. Check one family to run it, or two "
        "or more to compare. (s, S) uses s and S. (s, Q) uses s and Q. "
        "(R, S) uses R and S. (R, s, S) uses R, s, and S."
    )
    selected: list[str] = []
    for name in TEXTBOOK_FAMILIES:
        if st.checkbox(
            FAMILY_LABELS[name],
            value=True,
            key=f"fam_{name}",
            help=FAMILY_HELP[name],
        ):
            selected.append(name)

    st.subheader("How s and S are built")
    st.markdown(
        """
s and S are **unit counts the shop recomputes each tick** from its demand-rate
estimate. They are not typed in as stock numbers, and **L is not a slider** —
it is the inbound lead time on the graph (2 ticks on the linear chain; 4 from
f-lo / 2 from f-hi on the two-supplier graph).

- **s** (reorder point) covers lead time $L$ plus safety $k\\cdot L$.
  Raise **k** to reorder earlier.
- **S** (order-up-to) is s plus a cover horizon $H$. Raise **H = S − s**
  (ticks of demand, not a stock count) to buy a bigger batch when an order
  fires. $H$ is an EOQ cycle length — it does not depend on lead time.
- **Q** is a raw unit count, used only by (s, Q).
- **R** is ticks between reviews, used only by the periodic families.

$$s = (L + \\mathrm{round}(k\\cdot L))\\times \\mathrm{rate}$$
$$S = s + H\\times \\mathrm{rate}$$
"""
    )
    safety_pct = st.slider(
        "k — safety in s (fraction of lead time)",
        min_value=0.0,
        max_value=1.5,
        value=1.0 / 3.0,
        step=0.01,
        help=(
            "Moves s only. L comes from the graph edges, not from this slider. "
            "k = 0 means s covers exactly that inbound lead time; "
            "k = 0.33 (engine default) adds one-third of a lead time of extra stock. "
            "Used by (s, S), (s, Q), and (R, s, S). (R, S) still computes s internally "
            "but does not wait for it — it always orders up to S on a review tick."
        ),
    )
    cover_horizon = st.slider(
        "H = S − s (ticks of demand)",
        min_value=1,
        max_value=30,
        value=10,
        help=(
            "Moves S only, given s. This is one knob (the cover horizon), not two "
            "stock-level sliders. Larger → bigger orders and more inventory when the "
            "policy orders up to S. Unused by (s, Q) in this demo, because Q is set "
            "explicitly below."
        ),
    )
    q_value = st.slider(
        "Q (units)",
        min_value=1,
        max_value=80,
        value=30,
        help=(
            "Fixed order quantity. Used only by (s, Q): when position drops below s, "
            "order Q (split across suppliers), not up to S. Ignored by the other families."
        ),
    )
    review_interval = st.slider(
        "R (ticks)",
        min_value=1,
        max_value=14,
        value=7,
        help=(
            "Review interval. Used by (R, S) and (R, s, S): the shop only considers "
            "ordering every R ticks. Ignored by (s, S) and (s, Q). R = 1 is like "
            "checking every tick."
        ),
    )

    compare = st.button("Compare", type="primary")
    st.markdown(f"[Full simulator on GitHub]({GITHUB_URL})")

if compare:
    if not selected:
        st.error("Select at least one textbook policy.")
    else:
        setup_dir = PRESETS[preset_name]
        scenario = apply_knobs(
            load_setup(setup_dir),
            world_seed=int(world_seed),
            n_steps=int(n_steps),
            demand_mean=float(demand_mean),
            cycle_amp=float(cycle_amp),
            disruption=bool(disruption),
        )
        params = {
            "cover_horizon_ticks": int(cover_horizon),
            "safety_lead_pct_of_lag": float(safety_pct),
            "Q": int(q_value),
            "review_interval": int(review_interval),
        }
        with st.spinner("Running CRN overlay…"):
            logs = run_crn_overlay(scenario, selected, params)
        play = focal_family(selected)
        st.session_state["scenario"] = scenario
        st.session_state["run_logs"] = logs
        st.session_state["scorecard"] = scorecard(logs, scenario)
        st.session_state["frames"] = playback_frames(logs[play], scenario)
        st.session_state["focal"] = play

if "run_logs" in st.session_state:
    logs = st.session_state["run_logs"]
    scenario = st.session_state["scenario"]
    play = st.session_state["focal"]
    st.subheader("Scorecard (shop-1)")
    st.dataframe(
        _format_scorecard(st.session_state["scorecard"]),
        hide_index=True,
        width="stretch",
    )
    with st.expander("Scorecard columns"):
        st.markdown(
            f"""
- **Net profit** — shop-1 revenue minus order cost, holding cost, and order fees.
- **Δ vs (s, S)** — that profit minus the (s, S) order-up-to run on the **same** world.
  Positive means the row beat the CRN anchor. Blank/zero on the anchor row;
  NaN if (s, S) was not selected.
- **Service level** — units sold ÷ units demanded at shop-1 (1.0 = no lost sales).
- **Stockout rate** — fraction of (product, tick) rows where on-hand was already 0
  at decision time.
- **Inventory turnover** — units sold ÷ mean on-hand. High can mean lean *or* starved.
- **Rejections** — units requested but not filled **anywhere in the graph** this
  run (not shop-1-only): stock, cash, capacity, min-order, or unmet sink demand.
  Shop-1 service level can still be 1.0 if those misses happened upstream or at
  the sink.

Only **{FOCAL_NODE_ID}**'s policy changes, and only for this short run. DAG playback is
**{FAMILY_LABELS.get(play, play)}** (the (s, S) run when that family is selected).
Factory and sink cash are not in this table.
"""
        )

    left, right = st.columns([1, 1])
    with left:
        st.subheader("DAG playback")
        st.caption(
            "Play steps through one finished run (not a live tick). Marker size is "
            "on-hand inventory. Orange edges are units that moved this tick — a lag "
            "between order and fill is the graph's lead time, which is not a slider. "
            "Rejections in the title are lost or refused units this tick, not a cumulative count."
        )
        st.plotly_chart(
            animated_dag(scenario, st.session_state["frames"]),
            width="stretch",
        )
    with right:
        st.subheader("Cash & inventory")
        st.caption(
            "Each line is one policy on the same world (CRN). Cash is shop-1's balance; "
            "inventory is on-hand units. Periodic policies often look lumpier. "
            "Other nodes are not plotted here."
        )
        st.plotly_chart(_series_figure(logs, scenario), width="stretch")
else:
    st.info(
        "Choose a preset and knobs in the sidebar, then click **Compare**. "
        f"This is a limited demo — [full simulator on GitHub]({GITHUB_URL})."
    )
