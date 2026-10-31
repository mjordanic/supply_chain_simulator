"""Plotly DAG for one playback frame — layout from display levels (ADR 0018)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import plotly.graph_objects as go

from src.sim.graph import build_graph, compute_levels
from src.web.compare import FOCAL_NODE_ID

_TYPE_COLOR = {
    "factory": "#4C72B0",
    "intermediate": "#55A868",
    "demand_sink": "#C44E52",
}

_IDLE_EDGE = "#888888"
_FILLED_EDGE = "#f4a261"


def _legend_traces() -> list[go.Scatter]:
    """Stable legend-only traces so animation frames keep the same legend."""
    return [
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=14, color=_TYPE_COLOR["factory"], line=dict(width=1, color="#111")),
            name="Factory (produces)",
            hoverinfo="skip",
            showlegend=True,
        ),
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=14, color=_TYPE_COLOR["intermediate"], line=dict(width=4, color="#111")),
            name="Shop (size = stock on hand; thick outline = compared)",
            hoverinfo="skip",
            showlegend=True,
        ),
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=14, color=_TYPE_COLOR["demand_sink"], line=dict(width=1, color="#111")),
            name="Customers (demand sink)",
            hoverinfo="skip",
            showlegend=True,
        ),
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(width=1.5, color=_IDLE_EDGE),
            name="Idle lane",
            hoverinfo="skip",
            showlegend=True,
        ),
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(width=6, color=_FILLED_EDGE),
            name="Shipped this day",
            hoverinfo="skip",
            showlegend=True,
        ),
    ]


def _positions(scenario: Any) -> dict[str, tuple[float, float]]:
    node_ids = [ni.node.id for ni in scenario.nodes]
    node_types = {ni.node.id: ni.node._node_type for ni in scenario.nodes}
    graph = build_graph(node_ids, scenario.edges, node_types=node_types)
    levels = compute_levels(graph)
    by_level: dict[int, list[str]] = defaultdict(list)
    for nid in node_ids:
        by_level[levels[nid]].append(nid)
    pos: dict[str, tuple[float, float]] = {}
    for level, ids in by_level.items():
        ids = sorted(ids)
        n = len(ids)
        for i, nid in enumerate(ids):
            y = 0.5 if n == 1 else i / (n - 1)
            pos[nid] = (float(level), float(y))
    return pos


_FIXED_SIZE = 26


def _frame_title(frame: dict[str, Any], policy_label: str, color: str | None) -> str:
    name = f"<span style='color:{color}'>{policy_label}</span>  ·  " if policy_label else ""
    return f"{name}day {frame['tick']}  ·  units lost this day {frame['rejections']}"


def dag_figure(
    scenario: Any,
    frame: dict[str, Any],
    *,
    shop_max: int | None = None,
    policy_label: str = "",
    color: str | None = None,
) -> go.Figure:
    """Scatter DAG: shop size ~ on-hand stock, orange edges ~ units shipped.

    Factories and customers keep a fixed size: a factory's stock is not what
    the demo compares, and a demand sink holds none. ``shop_max`` is the
    run-wide shop stock that maps to the largest marker.
    """
    pos = _positions(scenario)
    filled: dict[tuple[str, str], int] = {}
    for p in frame.get("purchases", []):
        key = (p["supplier_id"], p["buyer_id"])
        filled[key] = filled.get(key, 0) + int(p["qty_filled"])

    traces: list[go.Scatter] = list(_legend_traces())
    for edge in scenario.edges:
        a, b = edge.supplier_id, edge.buyer_id
        if a not in pos or b not in pos:
            continue
        qty = filled.get((a, b), 0)
        traces.append(
            go.Scatter(
                x=[pos[a][0], pos[b][0]],
                y=[pos[a][1], pos[b][1]],
                mode="lines",
                line=dict(
                    width=6 if qty > 0 else 1.5,
                    color=_FILLED_EDGE if qty > 0 else _IDLE_EDGE,
                ),
                hoverinfo="text",
                text=f"{a} → {b}" + (f"  shipped {qty} units" if qty else ""),
                showlegend=False,
            )
        )
    shipped = [
        (a, b, qty) for (a, b), qty in filled.items() if qty and a in pos and b in pos
    ]
    traces.append(
        go.Scatter(
            x=[(pos[a][0] + pos[b][0]) / 2 for a, b, _ in shipped],
            y=[(pos[a][1] + pos[b][1]) / 2 for a, b, _ in shipped],
            mode="text",
            text=[str(qty) for _, _, qty in shipped],
            textposition="top center",
            textfont=dict(color=_FILLED_EDGE, size=12),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    xs, ys, colors, sizes, texts, widths = [], [], [], [], [], []
    if shop_max is None:
        shop_max = max(
            (n["inventory"] for n in frame["nodes"].values() if n["node_type"] == "intermediate"),
            default=1,
        )
    shop_max = max(1, shop_max)
    for nid, state in frame["nodes"].items():
        if nid not in pos:
            continue
        x, y = pos[nid]
        xs.append(x)
        ys.append(y)
        colors.append(_TYPE_COLOR.get(state["node_type"], "#888888"))
        if state["node_type"] == "intermediate":
            sizes.append(12 + 36 * (state["inventory"] / shop_max) ** 0.5)
        else:
            sizes.append(_FIXED_SIZE)
        texts.append(nid)
        widths.append(4 if nid == FOCAL_NODE_ID else 1)

    traces.append(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            marker=dict(
                size=sizes,
                color=colors,
                line=dict(width=widths, color="#111111"),
            ),
            text=texts,
            textposition="bottom center",
            hovertext=[
                f"{nid}<br>cash={frame['nodes'][nid]['cash']:.1f}"
                f"<br>inventory={frame['nodes'][nid]['inventory']}"
                for nid in texts
            ],
            hoverinfo="text",
            showlegend=False,
        )
    )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=_frame_title(frame, policy_label, color),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=20, r=20, t=40, b=72),
        height=320,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.18, x=0, font=dict(size=11), title_text=""),
        showlegend=True,
    )
    return fig


def animated_dag(
    scenario: Any,
    frames: list[dict[str, Any]],
    *,
    policy_label: str = "",
    color: str | None = None,
) -> go.Figure:
    """One Plotly figure that plays, pauses, and scrubs playback frames."""
    if not frames:
        raise ValueError("frames must be non-empty")
    shop_max = max(
        n["inventory"]
        for fr in frames
        for n in fr["nodes"].values()
        if n["node_type"] == "intermediate"
    )
    style = dict(shop_max=shop_max, policy_label=policy_label, color=color)
    fig = dag_figure(scenario, frames[0], **style)
    fig.frames = [
        go.Frame(
            data=dag_figure(scenario, fr, **style).data,
            name=str(i),
            layout=go.Layout(title=_frame_title(fr, policy_label, color)),
        )
        for i, fr in enumerate(frames)
    ]
    last = str(len(frames) - 1)
    still = dict(frame=dict(duration=0, redraw=True), mode="immediate", transition=dict(duration=0))
    fig.update_layout(
        title=dict(y=0.98, yref="container", yanchor="top"),
        margin=dict(l=20, r=20, t=100, b=190),
        height=580,
        showlegend=True,
        legend=dict(
            orientation="h", y=-0.38, yanchor="top", x=0, font=dict(size=11), title_text=""
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.0,
                y=1.02,
                xanchor="left",
                yanchor="bottom",
                direction="left",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=300, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                    dict(label="Pause", method="animate", args=[[None], still]),
                    dict(
                        label="Skip to end",
                        method="animate",
                        args=[[last], still],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                x=0.0,
                y=-0.02,
                yanchor="top",
                len=1.0,
                pad=dict(t=10, b=0),
                currentvalue=dict(prefix="day ", font=dict(size=12)),
                steps=[
                    dict(label=str(fr["tick"]), method="animate", args=[[str(i)], still])
                    for i, fr in enumerate(frames)
                ],
            )
        ],
    )
    return fig
