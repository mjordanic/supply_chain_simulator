"""Streamlit CRN policy lab — sibling consumer of ``src.sim`` (ADR 0023)."""

from __future__ import annotations

import math
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
    FAMILY_COLORS,
    FAMILY_LABELS,
    FOCAL_NODE_ID,
    MAX_N_STEPS,
    TEXTBOOK_FAMILIES,
    apply_knobs,
    identical_groups,
    run_crn_overlay,
    scorecard,
)
from src.web.dag import animated_dag
from src.web.playback import playback_frames

GITHUB_URL = "https://github.com/mjordanic/supply_chain_simulator"
MAX_SEED = 2**31 - 1
RIVAL_NODE_ID = "shop-2"

PRESETS = {
    "Linear chain (3 nodes)": _REPO_ROOT / "setups" / "three_node_chain",
    "Two suppliers (4 nodes)": _REPO_ROOT / "setups" / "demo_two_suppliers",
    "Two shops compete (6 nodes)": _REPO_ROOT / "setups" / "two_factories_two_shops",
}

PRESET_HELP = {
    "Linear chain (3 nodes)": (
        "factory-1 → shop-1 → sink-1, one product. "
        "Lead times are on the lanes (not sliders): 2 days factory→shop, "
        "1 day shop→customers. The shop's reorder math uses inbound L = 2."
    ),
    "Two suppliers (4 nodes)": (
        "f-lo (cheap, 4-day lead) and f-hi (fast but pricey, 2-day lead) both "
        "supply shop-1 → sink-1 (1 day). Lead times stay on those lanes; "
        "the shop buys from whichever offer is best that day."
    ),
    "Two shops compete (6 nodes)": (
        "f-lo (cheap, 4-day lead, 30 units/day) and f-hi (pricey, 2-day lead) "
        "each supply both shop-1 and shop-2, so the shops compete for the cheap "
        "capacity. Only shop-1's policy changes; shop-2 keeps its setup (s, S). "
        "The demand-mean knob sets both customer groups."
    ),
}

FAMILY_SHORT = {
    "order_up_to": "(s, S)",
    "reorder_point": "(s, Q)",
    "periodic_order_up_to": "(R, S)",
    "periodic_reorder": "(R, s, S)",
}

FAMILY_HELP = {
    "order_up_to": (
        "Every day, if stock (on hand + on order) is below s, order up to S. "
        "The comparison anchor."
    ),
    "reorder_point": (
        "Every day, if stock is below s, order a fixed batch Q. "
        "Fewer, lumpier orders than (s, S)."
    ),
    "periodic_order_up_to": (
        "Only looks every R days; on those days, order up to S. "
        "Ignores the shelf between reviews."
    ),
    "periodic_reorder": (
        "Every R days, order up to S only if stock is also below s. "
        "Can skip a review when stock is still healthy."
    ),
}

NUMBER_WORDS = {2: "Two", 3: "Three", 4: "Four"}


def _series_figure(run_logs: dict, scenario) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
            f"{FOCAL_NODE_ID} cash (higher is better)",
            f"{FOCAL_NODE_ID} stock on hand",
        ),
    )
    for family, run_log in run_logs.items():
        ts = node_timeseries_df(run_log, scenario)
        shop = ts[ts["node_id"] == FOCAL_NODE_ID].sort_values("tick")
        label = FAMILY_LABELS.get(family, family)
        line = dict(color=FAMILY_COLORS[family])
        fig.add_trace(
            go.Scatter(
                x=shop["tick"],
                y=shop["cash"],
                name=label,
                legendgroup=family,
                line=line,
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
                line=line,
            ),
            row=2,
            col=1,
        )
    fig.update_layout(
        height=460,
        margin=dict(l=50, r=20, t=40, b=80),
        legend=dict(
            title_text=f"Policy on {FOCAL_NODE_ID}",
            orientation="h",
            y=-0.14,
            x=0,
            font=dict(size=11),
        ),
    )
    fig.update_yaxes(title_text="cash", row=1, col=1)
    fig.update_yaxes(title_text="units", row=2, col=1)
    fig.update_xaxes(title_text="day", row=2, col=1)
    return fig


def _format_scorecard(card: pd.DataFrame):
    out = pd.DataFrame(
        {
            "Policy": card["family"].map(FAMILY_SHORT),
            "Net profit": card["net_profit"],
            # Preformatted: st.dataframe draws a null cell as "None", ignoring na_rep.
            "Δ vs (s, S)": card["delta_net_profit"].map(
                lambda d: "—" if pd.isna(d) else f"{d:+.2f}"
            ),
            "Fill rate": card["fill_rate"],
            "Days empty": card["days_empty"],
            "Turnover": card["inventory_turnover"],
            "Units lost (network)": card["rejections"],
        }
    )
    return out.style.format(
        {
            "Net profit": "{:.2f}",
            "Fill rate": "{:.3f}",
            "Turnover": "{:.2f}",
        }
    )


def _identical_note(card: pd.DataFrame) -> str | None:
    groups = identical_groups(card)
    if not groups:
        return None
    if len(groups[0]) == len(card):
        return (
            "Every policy came out identical. Something other than the ordering rule "
            "limits shop-1 here — usually the factories cannot supply more, or a long "
            "review interval leaves the rules placing the same few orders. Not a bug."
        )
    names = [" and ".join(FAMILY_SHORT[f] for f in g) for g in groups]
    return (
        f"{'; '.join(names)} came out identical: on this world they placed exactly the "
        "same orders (common with a long review interval R). Not a bug."
    )


# Sidebar knobs and their ``st.query_params`` names: (param, cast, lo, hi, default).
# The address bar always holds the knobs of the run on screen, so a link
# reproduces it.
KNOBS = {
    "world_seed": ("seed", int, 0, MAX_SEED, 42),
    "n_steps": ("steps", int, 1, MAX_N_STEPS, 30),
    "demand_mean": ("demand", float, 1.0, 40.0, 10.0),
    "cycle_amp": ("swing", float, 0.0, 0.8, 0.2),
    "safety_pct": ("k", float, 0.0, 1.5, 1.0 / 3.0),
    "cover_horizon": ("H", int, 1, 30, 10),
    "q_value": ("Q", int, 1, 80, 30),
    "review_interval": ("R", int, 1, 14, 3),
}
MARKDOWN_SPECIALS = set("\\`*_{}[]()<>#+-.!|~:$")
# Wrapped in an invisible word joiner so GFM cannot autolink a bare
# ``https://…``, ``www.…`` or ``a@b.c`` in a link value.
AUTOLINK_MARKS = set(":.@")
WORD_JOINER = "\u2060"


def _shown(raw: str) -> str:
    """A link value as plain text in a Markdown notice, cut to a short length."""
    cut = raw if len(raw) <= 40 else raw[:40] + "…"
    out = []
    for c in cut:
        shown = "\\" + c if c in MARKDOWN_SPECIALS else c
        out.append(WORD_JOINER + shown + WORD_JOINER if c in AUTOLINK_MARKS else shown)
    return "".join(out)


def _seed_widgets_from_link() -> None:
    """On a session's first run, start every widget from the URL, else defaults.

    Anything the demo cannot use is replaced, and each replacement is named in
    ``st.session_state["link_notes"]`` so the page can say so.
    """
    if "preset" in st.session_state:
        return
    q = st.query_params
    notes: list[str] = []
    by_slug = {path.name: label for label, path in PRESETS.items()}
    default_preset = next(iter(PRESETS))
    raw = q.get("preset")
    if raw is not None and raw not in by_slug:
        notes.append(f"unknown preset “{_shown(raw)}” → {default_preset}")
    st.session_state["preset"] = by_slug.get(raw, default_preset)

    for key, (param, cast, lo, hi, default) in KNOBS.items():
        raw = q.get(param)
        if raw is None:
            st.session_state[key] = default
            continue
        try:
            value = cast(raw)
        except ValueError:
            value = None
        if value is None or not math.isfinite(value):
            notes.append(f"{param}={_shown(raw)} is not a usable number → {default:g}")
            value = default
        elif not lo <= value <= hi:
            value = min(hi, max(lo, value))
            notes.append(f"{param}={_shown(raw)} is outside {lo:g}–{hi:g} → {value:g}")
        st.session_state[key] = value

    raw = q.get("disrupt")
    if raw not in (None, "0", "1"):
        notes.append(f"disrupt={_shown(raw)} is not 0 or 1 → off")
    st.session_state["disruption"] = "On" if raw == "1" else "Off"

    raw = q.get("policies")
    chosen = list(TEXTBOOK_FAMILIES) if raw is None else raw.split(",")
    for name in chosen:
        if name and name not in TEXTBOOK_FAMILIES:
            notes.append(f"unknown policy “{_shown(name)}” ignored")
    st.session_state["policies"] = [f for f in TEXTBOOK_FAMILIES if f in chosen]
    st.session_state["link_notes"] = notes


def _write_link(selected: list[str]) -> None:
    link = {param: str(st.session_state[key]) for key, (param, *_) in KNOBS.items()}
    link["preset"] = PRESETS[st.session_state["preset"]].name
    link["disrupt"] = "1" if st.session_state["disruption"] == "On" else "0"
    link["policies"] = ",".join(selected)
    st.query_params.from_dict(link)


@st.cache_data(max_entries=64, show_spinner="Running the simulation…")
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
    """One cached comparison; the same knobs from any visitor reuse it."""
    scenario = apply_knobs(
        load_setup(Path(setup_dir)),
        world_seed=world_seed,
        n_steps=n_steps,
        demand_mean=demand_mean,
        cycle_amp=cycle_amp,
        disruption=disruption,
    )
    logs = run_crn_overlay(scenario, list(families), dict(params))
    frames = {family: playback_frames(log, scenario) for family, log in logs.items()}
    has_rival = any(ni.node.id == RIVAL_NODE_ID for ni in scenario.nodes)
    rival = scorecard(logs, scenario, node_id=RIVAL_NODE_ID) if has_rival else None
    return scenario, logs, scorecard(logs, scenario), frames, rival


st.set_page_config(page_title="Supply-chain policy lab", layout="wide")
st.title("Supply-chain policy lab")

_seed_widgets_from_link()
intro = st.empty()
if notes := st.session_state.pop("link_notes", None):
    st.warning(
        "This link had values the demo cannot use, so they were replaced: "
        + "; ".join(notes)
        + "."
    )

with st.sidebar:
    st.header("Setup")
    st.caption(
        "Three canned networks. Topology and lead times are part of the preset, "
        "not sliders. Results update as soon as you change anything."
    )
    preset_name = st.selectbox(
        "Preset",
        list(PRESETS),
        key="preset",
        help=(
            "Linear chain: factory → shop → customers, inbound lead time 2 days. "
            "Two suppliers: cheap/slow factory (4 days) and fast/pricey factory "
            "(2 days) into shop-1. Two shops compete: the same two factories "
            "supply shop-1 and shop-2. Lead time is not a knob in this demo."
        ),
    )
    st.caption(PRESET_HELP[preset_name])
    world_seed = st.number_input(
        "Seed",
        min_value=0,
        max_value=MAX_SEED,
        step=1,
        key="world_seed",
        help=(
            f"Picks the random world (demand draws, market noise, disruptions), "
            f"0 to {MAX_SEED:,}. Same seed + same settings → exactly the same "
            "world, for every policy. Change it to re-roll the luck."
        ),
    )
    n_steps = st.slider(
        "Days",
        min_value=1,
        max_value=MAX_N_STEPS,
        key="n_steps",
        help=(
            f"How many simulated days to run. This page caps at {MAX_N_STEPS} "
            "to keep it cheap; the command-line simulator has no cap. "
            "Lead times here are 2–4 days, so 30 days shows a few reorder cycles."
        ),
    )

    st.header("World")
    st.caption(
        "These change the setup in memory only. Lead time, catalog, and prices "
        f"stay as in the network. The season repeats every {DEMO_CYCLE_LEN} days."
    )
    demand_mean = st.slider(
        "Customer demand per day (mean)",
        min_value=1.0,
        max_value=40.0,
        step=1.0,
        key="demand_mean",
        help=(
            "Average units customers try to buy each day; the day-to-day spread "
            "stays as in the setup. Higher demand stresses the shop: more lost "
            "sales if a policy under-orders, more stock if it over-orders."
        ),
    )
    cycle_amp = st.slider(
        "Seasonal swing",
        min_value=0.0,
        max_value=0.8,
        step=0.05,
        key="cycle_amp",
        help=(
            f"How strongly demand rises and falls over a {DEMO_CYCLE_LEN}-day "
            "season: 0 = flat, 0.4 = clearly visible wave, 0.8 = ±80%."
        ),
    )
    disruption = st.radio(
        "Disruptions",
        ["Off", "On"],
        horizontal=True,
        key="disruption",
        help=(
            f"When on, each day has a {DISRUPTION_ON_PROB:.0%} chance that a "
            f"{DISRUPTION_DURATION}-day disruption starts. While it lasts it pushes "
            f"the demand index down by {DISRUPTION_SEVERITY} each day of the event, "
            "to its floor. Factories "
            "keep producing as usual, so it acts as a demand slump: sales and "
            "profit usually fall, and lost sales can fall too. A short run may "
            "draw no disruption at all."
        ),
    )

    st.header(f"Policies on {FOCAL_NODE_ID}")
    chosen = st.multiselect(
        "Ordering rules to compare",
        list(TEXTBOOK_FAMILIES),
        key="policies",
        format_func=FAMILY_LABELS.get,
        help="\n".join(f"- **{FAMILY_LABELS[f]}**: {FAMILY_HELP[f]}" for f in TEXTBOOK_FAMILIES),
    )
    selected = [f for f in TEXTBOOK_FAMILIES if f in chosen]

    st.subheader("How s and S are built")
    st.markdown(
        """
s and S are **unit counts the shop recomputes each day** from its recent
sales rate. **L is not a slider** — it is the inbound lead time on the
network (2 days on the linear chain; 4 from f-lo / 2 from f-hi on the
other two).

- **s** (reorder point) covers lead time $L$ plus safety $k\\cdot L$.
  Raise **k** to reorder earlier.
- **S** (order-up-to) is s plus $H$ days of sales. Raise **H** to buy
  a bigger batch each time an order fires.
- **Q** is a fixed batch in units, used only by (s, Q).
- **R** is days between reviews, used only by the periodic rules.

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
            "Moves s only. k = 0 means s covers exactly the inbound lead time; "
            "k = 0.33 (engine default) adds one-third of a lead time of extra stock. "
            "Used by (s, S), (s, Q), and (R, s, S). (R, S) always orders up to S "
            "on a review day, whatever s is."
        ),
    )
    cover_horizon = st.slider(
        "H = S − s (days of sales)",
        min_value=1,
        max_value=30,
        key="cover_horizon",
        help=(
            "Moves S only, given s. Larger → bigger orders and more stock when a "
            "rule orders up to S. Unused by (s, Q), which orders Q instead."
        ),
    )
    q_value = st.slider(
        "Q (units)",
        min_value=1,
        max_value=80,
        key="q_value",
        help=(
            "Fixed order quantity. Used only by (s, Q): when stock drops below s, "
            "order Q (split across suppliers), not up to S."
        ),
    )
    review_interval = st.slider(
        "R (days)",
        min_value=1,
        max_value=14,
        key="review_interval",
        help=(
            "Review interval. Used by (R, S) and (R, s, S): the shop only considers "
            "ordering every R days. R = 1 is like checking every day. Default 3. "
            "Known engine limitation: the shop estimates demand from sales over the "
            "last L days only, so when R is well above the lead time the shelf is "
            "empty at review, the estimate drops to 0, and the rule stops ordering "
            "(try R = 7)."
        ),
    )
    st.markdown(f"[Full simulator on GitHub]({GITHUB_URL})")

_write_link(selected)

if len(selected) > 1:
    compared = (
        f"{NUMBER_WORDS[len(selected)]} textbook ordering rules run on the "
        "**same random world**, so every gap below is the rule, not luck."
    )
elif selected:
    compared = (
        f"The {FAMILY_SHORT[selected[0]]} rule runs on one random world; tick more "
        "rules to compare them on exactly the same world."
    )
else:
    compared = "Tick at least one ordering rule in the side panel."
intro.markdown(
    f"""
Which inventory policy makes the most money for **{FOCAL_NODE_ID}**? {compared}
Change the settings in the side panel (on a phone, open it with **»** at the top
left); the address bar always links to the run on screen.
[Full simulator on GitHub]({GITHUB_URL})
"""
)

if not selected:
    st.error("Select at least one ordering rule.")
else:
    scenario, logs, card, frames, rival = _run(
        str(PRESETS[preset_name]),
        tuple(selected),
        int(world_seed),
        int(n_steps),
        float(demand_mean),
        float(cycle_amp),
        disruption == "On",
        (
            ("cover_horizon_ticks", int(cover_horizon)),
            ("safety_lead_pct_of_lag", float(safety_pct)),
            ("Q", int(q_value)),
            ("review_interval", int(review_interval)),
        ),
    )
    st.subheader(f"Scorecard ({FOCAL_NODE_ID})")
    st.dataframe(_format_scorecard(card), hide_index=True, width="stretch")
    st.caption("On a narrow screen, scroll the table sideways for every column.")
    if note := _identical_note(card):
        st.caption(note)
    if rival is not None:
        st.caption(
            f"**{RIVAL_NODE_ID}** keeps its own (s, S) rule, but it competes with "
            f"{FOCAL_NODE_ID} for the same factories, so its results move too:"
        )
        st.dataframe(
            pd.DataFrame(
                {
                    f"Policy on {FOCAL_NODE_ID}": rival["family"].map(FAMILY_SHORT),
                    f"{RIVAL_NODE_ID} net profit": rival["net_profit"],
                    f"{RIVAL_NODE_ID} fill rate": rival["fill_rate"],
                }
            ).style.format(
                {f"{RIVAL_NODE_ID} net profit": "{:.2f}", f"{RIVAL_NODE_ID} fill rate": "{:.3f}"}
            ),
            hide_index=True,
        )
    with st.expander("Scorecard columns"):
        st.markdown(
            f"""
- **Policy** — (s, S) order-up-to, (s, Q) reorder-point, (R, S) periodic
  order-up-to, (R, s, S) periodic reorder.
- **Net profit** — shop-1 revenue minus order cost, holding cost, and order fees.
  It does not value the stock left on the shelf at the end.
- **Δ vs (s, S)** — that profit minus the (s, S) run on the **same** world.
  Positive means the row beat (s, S). Shows **—** when (s, S) is not ticked.
- **Fill rate** — units shop-1 sold ÷ units its customers wanted (1.0 = no lost sales).
- **Days empty** — days on which shop-1 closed with nothing on the shelf
  (the zero points of the stock chart).
- **Turnover** — inventory turnover: units sold ÷ average stock. High can mean lean *or* starved.
- **Units lost (network)** — units asked for but not delivered **anywhere in
  the network** this run, not just at shop-1: lost customer sales, plus orders a
  supplier could not fill.

Only **{FOCAL_NODE_ID}**'s policy changes, and only for this short run.
Factory and customer cash are not in this table.
"""
        )

    best = card.loc[card["net_profit"].idxmax(), "family"]
    if st.session_state.get("picture") not in selected:
        st.session_state["picture"] = best
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Network playback")
        picture = st.radio(
            "Policy shown",
            selected,
            key="picture",
            horizontal=True,
            format_func=FAMILY_SHORT.get,
            help="Starts on the policy with the best net profit.",
        )
        st.caption(
            "One finished run, played back day by day — Play, Pause, or drag the "
            "day slider. Shop circles grow and shrink with stock on hand; factories "
            "and customers stay a fixed size. Orange lanes and numbers are units "
            "shipped that day; they reach the buyer after the lane's lead time."
        )
        st.plotly_chart(
            animated_dag(
                scenario,
                frames[picture],
                policy_label=FAMILY_LABELS[picture],
                color=FAMILY_COLORS[picture],
            ),
            width="stretch",
        )
    with right:
        st.subheader("Cash & stock")
        st.caption(
            "Each line is one policy on the same world, in the same colour as its "
            "name above the network. Periodic rules often look lumpier. "
            "Other nodes are not plotted here."
        )
        st.plotly_chart(_series_figure(logs, scenario), width="stretch")

with st.expander("How this demo works"):
    st.markdown(
        f"""
This page is a **small, limited visual representation** of the simulator:
three canned networks, four textbook ordering rules on **{FOCAL_NODE_ID}**,
and a short run (at most {MAX_N_STEPS} days). The engine behind it can do much
more — arbitrary networks, LLM-generated catalogs, Optuna tuning, PPO, M5 demand
replay, CLI parquet exports, and longer runs.

Each change loads one of three setup directories (`catalog.csv` + `setup.yaml`),
applies the side-panel settings **in memory** (nothing on disk changes), and runs
each ticked rule through the same `Runner` as the CLI. Only **{FOCAL_NODE_ID}** is
swapped. Factories, customers, and shop-2 keep the policies from the setup. The
address bar always encodes the settings of the run you are looking at.

### Same world for every rule

Every ticked rule runs on the same seed, so demand, market noise, and
disruptions are identical across rules (simulation people call this
*Common Random Numbers*). Differences in the scorecard are the rule, not luck.
A different seed is a different world, and the ranking can change.

### What is *not* a knob

**Lead time lives on the network lanes.** An order placed on day `t` arrives
on day `t` + that lane's lead time — never the same day.

| Preset | Network | Inbound lead times (used in s) | Shop → customers |
|---|---|---|---|
| Linear chain | factory-1 → shop-1 → sink-1 | 2 days | 1 day |
| Two suppliers | f-lo and f-hi → shop-1 → sink-1 | 4 days (cheap) / 2 days (fast) | 1 day |
| Two shops compete | f-lo and f-hi → shop-1 → sink-1, and → shop-2 → sink-2 | 4 days (cheap) / 2 days (fast) | 1 day |

Also frozen here: catalog, list prices, node cash/capacity, factory policy,
price elasticity, promotions, and trend. Unmet demand is a **lost sale**, not
a backorder.

### Not in this page

Custom networks, the LLM world generator, Optuna, PPO / `RLNodePolicy`, M5
demand replay, parquet download, and the Central Table debugger. Those live
in the [GitHub repo]({GITHUB_URL}).
"""
    )
