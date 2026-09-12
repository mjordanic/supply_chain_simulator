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
            name="Shop (thick outline = compared)",
            hoverinfo="skip",
            showlegend=True,
        ),
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=14, color=_TYPE_COLOR["demand_sink"], line=dict(width=1, color="#111")),
            name="Demand sink (buys / makes cash)",
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
            name="Order filled this tick",
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


def dag_figure(scenario: Any, frame: dict[str, Any]) -> go.Figure:
    """Scatter DAG: node size ~ inventory, edge width ~ this-tick filled qty."""
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
                text=f"{a} → {b}" + (f"  filled={qty}" if qty else ""),
                showlegend=False,
            )
        )

    xs, ys, colors, sizes, texts, widths = [], [], [], [], [], []
    inventories = [n["inventory"] for n in frame["nodes"].values()] or [1]
    inv_max = max(1, max(inventories))
    for nid, state in frame["nodes"].items():
        if nid not in pos:
            continue
        x, y = pos[nid]
        xs.append(x)
        ys.append(y)
        colors.append(_TYPE_COLOR.get(state["node_type"], "#888888"))
        sizes.append(16 + 32 * (state["inventory"] / inv_max) ** 0.5)
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
        title=f"tick {frame['tick']}  ·  rejections {frame['rejections']}",
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


def animated_dag(scenario: Any, frames: list[dict[str, Any]]) -> go.Figure:
    """One Plotly figure that auto-plays playback frames in the browser."""
    if not frames:
        raise ValueError("frames must be non-empty")
    fig = dag_figure(scenario, frames[0])
    fig.frames = [
        go.Frame(
            data=dag_figure(scenario, fr).data,
            name=str(i),
            layout=go.Layout(
                title=f"tick {fr['tick']}  ·  rejections {fr['rejections']}"
            ),
        )
        for i, fr in enumerate(frames)
    ]
    last = str(len(frames) - 1)
    fig.update_layout(
        margin=dict(l=20, r=20, t=90, b=90),
        height=420,
        showlegend=True,
        legend=dict(orientation="h", y=-0.12, x=0, font=dict(size=11), title_text=""),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.0,
                y=1.18,
                xanchor="left",
                yanchor="top",
                direction="left",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=120, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                    dict(
                        label="Skip to end",
                        method="animate",
                        args=[
                            [last],
                            dict(
                                frame=dict(duration=0, redraw=True),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                ],
            )
        ]
    )
    return fig
