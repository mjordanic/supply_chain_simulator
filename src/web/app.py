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
    "Two shops compete (6 nodes)": _REPO_ROOT / "setups" / "two_factories_two_shops",
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
    "Two shops compete (6 nodes)": (
        "f-lo (cheap, 4-tick lead, 30 units/tick) and f-hi (pricey, 2-tick lead) "
        "each supply both shop-1 and shop-2, so the shops compete for the cheap "
        "capacity. Only shop-1's policy changes; shop-2 keeps its setup (s, S). "
        "The demand-mean knob sets both sinks."
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
    out["Fill rate"] = out["fill_rate"].round(3)
    out["Stockout rate"] = out["stockout_rate"].round(3)
    out["Inventory turnover"] = out["inventory_turnover"].round(2)
    out["Rejections"] = out["rejections"]
    return out[
        [
            "Policy",
            "Net profit",
            "Δ vs (s, S)",
            "Fill rate",
            "Stockout rate",
            "Inventory turnover",
            "Rejections",
        ]
    ]

# Sidebar knobs and their ``st.query_params`` names: (param, cast, lo, hi, default).
# The address bar always holds the last compared knobs, so a link reproduces a run.
KNOBS = {
    "world_seed": ("seed", int, 0, 2**31 - 1, 42),
    "n_steps": ("steps", int, 1, MAX_N_STEPS, 30),
    "demand_mean": ("demand", float, 1.0, 40.0, 10.0),
    "cycle_amp": ("amp", float, 0.0, 0.8, 0.2),
    "safety_pct": ("k", float, 0.0, 1.5, 1.0 / 3.0),
    "cover_horizon": ("H", int, 1, 30, 10),
    "q_value": ("Q", int, 1, 80, 30),
    "review_interval": ("R", int, 1, 14, 3),
}


def _seed_widgets_from_link() -> None:
    """On a session's first run, start every widget from the URL, else defaults."""
    if "preset" in st.session_state:
        return
    q = st.query_params
    by_slug = {path.name: label for label, path in PRESETS.items()}
    st.session_state["preset"] = by_slug.get(q.get("preset", ""), next(iter(PRESETS)))
    for key, (param, cast, lo, hi, default) in KNOBS.items():
        try:
            value = cast(q.get(param, default))
        except ValueError:
            value = default
        st.session_state[key] = min(hi, max(lo, value))
    st.session_state["disruption"] = q.get("disrupt") == "1"
    chosen = q.get("policies")
    for name in TEXTBOOK_FAMILIES:
        st.session_state[f"fam_{name}"] = chosen is None or name in chosen.split(",")


def _write_link(selected: list[str]) -> None:
    link = {param: str(st.session_state[key]) for key, (param, *_) in KNOBS.items()}
    link["preset"] = PRESETS[st.session_state["preset"]].name
    link["disrupt"] = "1" if st.session_state["disruption"] else "0"
    link["policies"] = ",".join(selected)
    st.query_params.from_dict(link)


@st.cache_data(max_entries=64, show_spinner="Running CRN overlay…")
def _run(
    setup_dir: str,
    families: tuple[str, ...],
    world_seed: int,
    n_steps: int,
    demand_mean: float,
    cycle_amp: float,
    disruption: bool,
    params: tuple[tuple[str, float], ...],
):
    """One cached CRN comparison; the same knobs from any visitor reuse it."""
    scenario = apply_knobs(
        load_setup(Path(setup_dir)),
        world_seed=world_seed,
        n_steps=n_steps,
        demand_mean=demand_mean,
        cycle_amp=cycle_amp,
        disruption=disruption,
    )
    logs = run_crn_overlay(scenario, list(families), dict(params))
    play = focal_family(list(families))
    return (
        scenario,
        logs,
        scorecard(logs, scenario),
        playback_frames(logs[play], scenario),
        play,
    )


st.set_page_config(page_title="Supply-chain policy lab", layout="wide")
st.title("Supply-chain policy lab")
st.markdown(
    f"""
Which inventory policy makes the most money for **{FOCAL_NODE_ID}**? Four textbook
policies run on the **same random world** (Common Random Numbers), so every gap
below is the policy, not luck. Change the knobs in the sidebar and press
**Compare**; the address bar then links to that exact run.
[Full simulator on GitHub]({GITHUB_URL})
"""
)

_seed_widgets_from_link()

with st.sidebar:
    form = st.form("knobs", border=False)

with form:
    st.header("Setup")
    st.caption(
        "Three canned graphs. Topology and lead times are part of the preset, "
        "not sliders."
    )
    preset_name = st.selectbox(
        "Preset",
        list(PRESETS),
        key="preset",
        help=(
            "Linear chain: factory → shop → sink, inbound lead time 2 ticks. "
            "Two suppliers: cheap/slow factory (4 ticks) and fast/pricey factory "
            "(2 ticks) into shop-1. Two shops compete: the same two factories "
            "supply shop-1 and shop-2. Lead time is not a knob in this demo."
        ),
    )
    st.caption(PRESET_HELP[preset_name])
    world_seed = st.number_input(
        "world_seed",
        min_value=0,
        step=1,
        key="world_seed",
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
        key="n_steps",
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
        step=1.0,
        key="demand_mean",
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
        step=0.05,
        key="cycle_amp",
        help=(
            f"Amplitude of the seasonal sine wave on market demand. The period is "
            f"fixed at {DEMO_CYCLE_LEN} ticks in this demo (not a slider) so a short "
            "run can show a cycle. 0 = flat season; 0.4 is a clearly visible wave. "
            "Elasticity, promos, and trend are not exposed here."
        ),
    )
    disruption = st.checkbox(
        "Disruptions on",
        key="disruption",
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
f-lo / 2 from f-hi on the two-supplier and two-shop graphs).

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
        step=0.01,
        key="safety_pct",
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
        key="cover_horizon",
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
        key="q_value",
        help=(
            "Fixed order quantity. Used only by (s, Q): when position drops below s, "
            "order Q (split across suppliers), not up to S. Ignored by the other families."
        ),
    )
    review_interval = st.slider(
        "R (ticks)",
        min_value=1,
        max_value=14,
        key="review_interval",
        help=(
            "Review interval. Used by (R, S) and (R, s, S): the shop only considers "
            "ordering every R ticks. Ignored by (s, S) and (s, Q). R = 1 is like "
            "checking every tick. Default 3. Known engine limitation: the shop "
            "estimates demand from sales over the last L ticks only, so when R is "
            "well above the lead time the shelf is empty at review, the estimate "
            "drops to 0, and the policy stops ordering (try R = 7)."
        ),
    )

    st.form_submit_button("Compare", type="primary")

st.sidebar.markdown(f"[Full simulator on GitHub]({GITHUB_URL})")

_write_link(selected)

if not selected:
    st.error("Select at least one textbook policy.")
else:
    scenario, logs, card, frames, play = _run(
        str(PRESETS[preset_name]),
        tuple(selected),
        int(world_seed),
        int(n_steps),
        float(demand_mean),
        float(cycle_amp),
        bool(disruption),
        (
            ("cover_horizon_ticks", int(cover_horizon)),
            ("safety_lead_pct_of_lag", float(safety_pct)),
            ("Q", int(q_value)),
            ("review_interval", int(review_interval)),
        ),
    )
    st.subheader("Scorecard (shop-1)")
    st.dataframe(
        _format_scorecard(card),
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
- **Fill rate** — units shop-1 sold ÷ units its demand sink wanted (1.0 = no lost sales).
- **Stockout rate** — fraction of (product, tick) rows where on-hand was already 0
  at decision time.
- **Inventory turnover** — units sold ÷ mean on-hand. High can mean lean *or* starved.
- **Rejections** — units requested but not filled **anywhere in the graph** this
  run (not shop-1-only): stock, cash, capacity, min-order, or unmet sink demand.

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
            animated_dag(scenario, frames),
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

with st.expander("How this demo works"):
    st.markdown(
        f"""
This page is a **small, limited visual representation** of the simulator:
three canned graphs, four textbook inventory policies on **{FOCAL_NODE_ID}**,
and a short Common-Random-Numbers run (at most {MAX_N_STEPS} ticks). The engine
behind it can do much more — arbitrary DAGs, LLM-generated catalogs, Optuna
tuning, PPO, M5 demand replay, CLI parquet exports, and longer runs.

**Compare** loads one of three setup directories (`catalog.csv` + `setup.yaml`),
overlays the sidebar knobs **in memory** (nothing on disk changes), and runs
each checked textbook family through the same `Runner` as the CLI. Only
**{FOCAL_NODE_ID}** is swapped. Factories, demand sinks, and shop-2 keep the
policies from the setup. The page opens on the default run, and the address
bar always encodes the knobs of the run you are looking at.

### What you can change

| Sidebar | What it does in *this* demo |
|---|---|
| Preset | Picks the graph. Topology, SKUs, prices, capacities, and **lead times** come with it. |
| `world_seed` | Shared world RNG (market, demand draws, disruptions). Same seed + same knobs → bit-identical world. |
| `n_steps` | How many ticks (simulated days) to run. Capped at **{MAX_N_STEPS}** here; the CLI has no such cap. |
| Sink demand mean | Overwrites every sink's demand-distribution **mean**. Std. deviation stays as in the setup. |
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
| Two shops compete | f-lo and f-hi → shop-1 → sink-1, and → shop-2 → sink-2 | 4 ticks (cheap) / 2 ticks (fast) | 1 tick |

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
