"""Graph deep module — typed-node DAG structural representation.

This module is a pure library of topology operations. No runtime behaviour,
no imports from other ``src.sim`` modules. It is the structural backbone that
later phases (GraphSimulation, allocation, etc.) build upon.

Public API
----------
- ``EdgeSpec``       — immutable edge descriptor with default + per-product lead times
- ``Graph``          — validated DAG with topology queries
- ``build_graph``    — authoring entry point: validates then constructs
- ``validate_dag``   — raises ``ValueError`` on cycles, unreachable nodes, illegal type-based edges
- ``compute_levels`` — returns ``dict[node_id, int]`` via longest-path-from-any-source
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EdgeSpec:
    """Directed supply edge from *supplier* to *buyer*.

    Parameters
    ----------
    supplier_id:
        Node ID of the upstream actor.
    buyer_id:
        Node ID of the downstream actor.
    default_lead_time:
        Lead-time (in ticks) used when no per-product override is present.
    per_product_lead_time:
        Optional mapping ``{product_id: lead_time_in_ticks}`` that overrides
        ``default_lead_time`` for specific products.
    """

    supplier_id: str
    buyer_id: str
    default_lead_time: int
    per_product_lead_time: dict[str, int] | None = field(default=None)

    def lead_time_for(self, pid: str) -> int:
        """Return the lead time for product *pid*, honouring per-product overrides."""
        if self.per_product_lead_time and pid in self.per_product_lead_time:
            return self.per_product_lead_time[pid]
        return self.default_lead_time


class Graph:
    """Validated DAG of typed nodes with topology-query methods.

    Construct via :func:`build_graph` (which runs validation first).
    Direct construction skips validation — use only when you own the inputs.
    """

    def __init__(self, nodes: list[str], edges: list[EdgeSpec]) -> None:
        self._nodes: frozenset[str] = frozenset(nodes)
        # supplier → {buyer} and buyer → {supplier} adjacency
        self._suppliers: dict[str, set[str]] = defaultdict(set)
        self._buyers: dict[str, set[str]] = defaultdict(set)
        # (supplier_id, buyer_id) → EdgeSpec for lead-time lookups
        self._edge_index: dict[tuple[str, str], EdgeSpec] = {}

        for edge in edges:
            self._buyers[edge.supplier_id].add(edge.buyer_id)
            self._suppliers[edge.buyer_id].add(edge.supplier_id)
            self._edge_index[(edge.supplier_id, edge.buyer_id)] = edge

    # ------------------------------------------------------------------
    # Topology queries
    # ------------------------------------------------------------------

    def suppliers_of(self, buyer_id: str) -> set[str]:
        """Return the set of direct suppliers for *buyer_id*."""
        return set(self._suppliers.get(buyer_id, set()))

    def buyers_of(self, supplier_id: str) -> set[str]:
        """Return the set of direct buyers for *supplier_id*."""
        return set(self._buyers.get(supplier_id, set()))

    def lead_time(self, supplier_id: str, buyer_id: str, pid: str) -> int:
        """Return the lead time for a specific ``(supplier_id, buyer_id, pid)`` triple.

        Per-product overrides on the edge take precedence over
        ``default_lead_time``.
        """
        edge = self._edge_index[(supplier_id, buyer_id)]
        return edge.lead_time_for(pid)

    @property
    def nodes(self) -> frozenset[str]:
        return self._nodes


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_dag(
    nodes: list[str],
    edges: list[EdgeSpec],
    *,
    node_types: dict[str, str] | None = None,
) -> None:
    """Validate that *nodes* and *edges* form a legal DAG.

    Parameters
    ----------
    nodes:
        List of node IDs in the graph.
    edges:
        List of directed supply edges.
    node_types:
        Optional mapping of ``{node_id: node_type}`` where each value is one of
        ``"factory"``, ``"intermediate"``, or ``"demand_sink"``.  When supplied,
        **type-based edge rules** are enforced instead of the old BFS same-level
        check:

        - Supplier must be ``factory`` or ``intermediate`` (sinks don't sell).
        - Buyer must be ``intermediate`` or ``demand_sink`` (factories don't buy;
          flow must pass through ≥1 intermediate so ``factory→demand_sink`` is
          illegal).

        When *None* (default), no type-based check is performed (backward-compat
        mode for callers that haven't yet plumbed node types through).

    Raises
    ------
    ValueError
        - If the graph contains a **cycle** (including self-loops).
        - If any node is **unreachable** from the connected component (isolated).
        - If *node_types* is supplied and any edge violates the type rules above.
    """
    node_set = set(nodes)

    # Build adjacency for cycle detection (supplier → buyer direction)
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    for edge in edges:
        adj[edge.supplier_id].append(edge.buyer_id)

    # ------------------------------------------------------------------
    # 1. Cycle detection via DFS with three-colour marking
    # ------------------------------------------------------------------
    WHITE, GRAY, BLACK = 0, 1, 2
    colour: dict[str, int] = {n: WHITE for n in nodes}

    def _dfs_cycle(node: str) -> None:
        colour[node] = GRAY
        for neighbour in adj[node]:
            if colour[neighbour] == GRAY:
                raise ValueError(
                    f"Cycle detected: node '{neighbour}' is reachable from itself "
                    f"(visiting '{node}' → '{neighbour}')."
                )
            if colour[neighbour] == WHITE:
                _dfs_cycle(neighbour)
        colour[node] = BLACK

    for n in nodes:
        if colour[n] == WHITE:
            _dfs_cycle(n)

    # ------------------------------------------------------------------
    # 2. Unreachable-node detection
    #    A node is "reachable / connected" if it participates in at least
    #    one edge (as supplier OR buyer) OR is the only node in the graph.
    # ------------------------------------------------------------------
    if len(nodes) > 1:
        connected: set[str] = set()
        for edge in edges:
            connected.add(edge.supplier_id)
            connected.add(edge.buyer_id)
        unreachable = node_set - connected
        if unreachable:
            raise ValueError(
                f"Unreachable (isolated) node(s) detected: {sorted(unreachable)}. "
                "Every node must participate in at least one edge."
            )

    # ------------------------------------------------------------------
    # 3. Edge-type validation (only when node_types is supplied)
    #
    # Replaces the old BFS same-level check.  Rules:
    #   - Supplier ∈ {factory, intermediate}   (sinks don't sell)
    #   - Buyer   ∈ {intermediate, demand_sink} (factories don't buy;
    #     factory→sink is forbidden — flow must pass through ≥1 intermediate)
    #
    # Lateral intermediate→intermediate edges are explicitly legal here.
    # ------------------------------------------------------------------
    if node_types is not None:
        for edge in edges:
            sup_type = node_types.get(edge.supplier_id)
            buy_type = node_types.get(edge.buyer_id)
            # Rule: sinks don't sell (demand_sink as supplier is illegal)
            if sup_type == "demand_sink":
                raise ValueError(
                    f"Illegal edge '{edge.supplier_id}' → '{edge.buyer_id}': "
                    f"supplier node '{edge.supplier_id}' is a demand_sink "
                    f"(rule: sinks don't sell)."
                )
            # Rule: factories don't buy (factory as buyer is illegal)
            if buy_type == "factory":
                raise ValueError(
                    f"Illegal edge '{edge.supplier_id}' → '{edge.buyer_id}': "
                    f"buyer node '{edge.buyer_id}' is a factory "
                    f"(rule: factories don't buy)."
                )
            # Rule: factory→sink is illegal; flow must pass through ≥1 intermediate
            if sup_type == "factory" and buy_type == "demand_sink":
                raise ValueError(
                    f"Illegal edge '{edge.supplier_id}' → '{edge.buyer_id}': "
                    f"factory→sink edge is not allowed — flow must pass through "
                    f"at least one intermediate node "
                    f"(rule: factory→sink bypasses intermediates)."
                )


# ---------------------------------------------------------------------------
# Level computation
# ---------------------------------------------------------------------------

def _compute_bfs_levels(nodes: list[str], adj: dict[str, list[str]]) -> dict[str, int]:
    """BFS shortest-path level from all source nodes (nodes with no in-edges).

    Used for same-level peer-link detection.  A node's BFS level is its
    minimum distance from any source — its "natural echelon."
    Nodes unreachable from any source (e.g. orphaned nodes that escaped
    the unreachable check) get level 0 (treated as additional sources).
    """
    # Compute in-degree
    in_degree: dict[str, int] = {n: 0 for n in nodes}
    for n in nodes:
        for neighbour in adj[n]:
            in_degree[neighbour] += 1

    level: dict[str, int] = {}
    queue: deque[str] = deque()
    for n in nodes:
        if in_degree[n] == 0:
            level[n] = 0
            queue.append(n)

    while queue:
        node = queue.popleft()
        for neighbour in adj[node]:
            if neighbour not in level:
                level[neighbour] = level[node] + 1
                queue.append(neighbour)

    # Assign level 0 to any remaining nodes (should be caught earlier)
    for n in nodes:
        if n not in level:
            level[n] = 0

    return level


def _compute_levels_from_adj(nodes: list[str], adj: dict[str, list[str]]) -> dict[str, int]:
    """Longest-path level computation on a DAG given an adjacency mapping.

    Level 0 = source nodes (no incoming edges). Level N = longest path from
    any source to this node.
    """
    # Compute in-degree
    in_degree: dict[str, int] = {n: 0 for n in nodes}
    for n in nodes:
        for neighbour in adj[n]:
            in_degree[neighbour] += 1

    # Topological BFS (Kahn's algorithm) — accumulate longest path distance
    level: dict[str, int] = {n: 0 for n in nodes}
    queue: deque[str] = deque(n for n in nodes if in_degree[n] == 0)

    while queue:
        node = queue.popleft()
        for neighbour in adj[node]:
            level[neighbour] = max(level[neighbour], level[node] + 1)
            in_degree[neighbour] -= 1
            if in_degree[neighbour] == 0:
                queue.append(neighbour)

    return level


def compute_levels(graph: Graph) -> dict[str, int]:
    """Return ``{node_id: echelon_level}`` for all nodes in *graph*.

    Level 0 = source nodes (no incoming edges, i.e. factories).
    Level N = longest-path distance from any level-0 node.
    """
    nodes = list(graph.nodes)
    adj: dict[str, list[str]] = {n: list(graph.buyers_of(n)) for n in nodes}
    return _compute_levels_from_adj(nodes, adj)


# ---------------------------------------------------------------------------
# Authoring entry point
# ---------------------------------------------------------------------------

def build_graph(
    nodes: list[str],
    edges: list[EdgeSpec],
    *,
    node_types: dict[str, str] | None = None,
) -> Graph:
    """Validate *nodes* + *edges* and return a :class:`Graph`.

    Parameters
    ----------
    nodes:
        List of node IDs.
    edges:
        List of directed supply edges.
    node_types:
        Optional ``{node_id: node_type}`` mapping (values: ``"factory"``,
        ``"intermediate"``, ``"demand_sink"``).  When supplied, type-based edge
        validation is performed (lateral ``intermediate→intermediate`` edges are
        legal; ``factory→sink``, ``→factory``, and ``sink→*`` edges are
        rejected).  When *None*, no type check is performed (backward compat).

    Raises ``ValueError`` if the topology is invalid (cycle, unreachable node,
    or illegal edge type when *node_types* is given).
    """
    validate_dag(nodes, edges, node_types=node_types)
    return Graph(nodes, edges)
