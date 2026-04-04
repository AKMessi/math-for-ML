"""Graph utilities for graph theory and computation graph lessons."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import heapq
from typing import Hashable, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
__all__ = ["Graph", "computation_graph_topology"]


@dataclass
class Graph:
    """Weighted graph backed by an adjacency dictionary.

    Args:
        directed: Whether edges are directed.

    Examples:
        >>> graph = Graph()
        >>> graph.add_edge("A", "B")
        >>> graph.bfs("A")
        ['A', 'B']
    """

    directed: bool = False
    adjacency: dict[Hashable, dict[Hashable, float]] = field(
        default_factory=lambda: defaultdict(dict)
    )

    def add_edge(
        self,
        source: Hashable,
        target: Hashable,
        *,
        weight: float = 1.0,
    ) -> None:
        """Add an edge to the graph."""

        self.adjacency[source][target] = weight
        self.adjacency.setdefault(target, {})
        if not self.directed:
            self.adjacency[target][source] = weight

    def nodes(self) -> list[Hashable]:
        """Return graph nodes in sorted string order."""

        return sorted(self.adjacency, key=str)

    def adjacency_matrix(self, order: Sequence[Hashable] | None = None) -> FloatArray:
        """Construct the adjacency matrix for a chosen node order."""

        order = list(order) if order is not None else self.nodes()
        index = {node: position for position, node in enumerate(order)}
        matrix = np.zeros((len(order), len(order)), dtype=np.float64)
        for source, neighbors in self.adjacency.items():
            for target, weight in neighbors.items():
                matrix[index[source], index[target]] = weight
        return matrix

    def degree_matrix(self, order: Sequence[Hashable] | None = None) -> FloatArray:
        """Construct the degree matrix."""

        adjacency = self.adjacency_matrix(order=order)
        degrees = adjacency.sum(axis=1)
        return np.diag(degrees)

    def laplacian_matrix(
        self,
        order: Sequence[Hashable] | None = None,
        *,
        normalized: bool = False,
    ) -> FloatArray:
        """Construct the combinatorial or normalized graph Laplacian."""

        adjacency = self.adjacency_matrix(order=order)
        degree = self.degree_matrix(order=order)
        laplacian = degree - adjacency
        if not normalized:
            return laplacian
        with np.errstate(divide="ignore"):
            inv_sqrt = np.diag(
                np.where(np.diag(degree) > 0.0, 1.0 / np.sqrt(np.diag(degree)), 0.0)
            )
        identity = np.eye(adjacency.shape[0], dtype=np.float64)
        return identity - inv_sqrt @ adjacency @ inv_sqrt

    def bfs(self, start: Hashable) -> list[Hashable]:
        """Run breadth-first search from ``start``."""

        visited = {start}
        queue: deque[Hashable] = deque([start])
        order: list[Hashable] = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in sorted(self.adjacency.get(node, {}), key=str):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return order

    def dfs(self, start: Hashable) -> list[Hashable]:
        """Run depth-first search from ``start``."""

        order: list[Hashable] = []
        visited: set[Hashable] = set()

        def visit(node: Hashable) -> None:
            visited.add(node)
            order.append(node)
            for neighbor in sorted(self.adjacency.get(node, {}), key=str):
                if neighbor not in visited:
                    visit(neighbor)

        visit(start)
        return order

    def shortest_path(self, source: Hashable, target: Hashable) -> tuple[float, list[Hashable]]:
        """Compute the shortest weighted path using Dijkstra's algorithm."""

        heap: list[tuple[float, Hashable]] = [(0.0, source)]
        distances = {source: 0.0}
        previous: dict[Hashable, Hashable] = {}

        while heap:
            distance, node = heapq.heappop(heap)
            if node == target:
                break
            if distance > distances.get(node, float("inf")):
                continue
            for neighbor, weight in self.adjacency.get(node, {}).items():
                candidate = distance + weight
                if candidate < distances.get(neighbor, float("inf")):
                    distances[neighbor] = candidate
                    previous[neighbor] = node
                    heapq.heappush(heap, (candidate, neighbor))

        if target not in distances:
            raise ValueError(f"No path from {source!r} to {target!r}.")

        path = [target]
        while path[-1] != source:
            path.append(previous[path[-1]])
        path.reverse()
        return distances[target], path

    def topological_sort(self) -> list[Hashable]:
        """Return a topological ordering of a directed acyclic graph.

        Raises:
            ValueError: If the graph is undirected or contains a cycle.
        """

        if not self.directed:
            raise ValueError("Topological sort requires a directed graph.")

        indegree = {node: 0 for node in self.adjacency}
        for neighbors in self.adjacency.values():
            for node in neighbors:
                indegree[node] = indegree.get(node, 0) + 1

        queue = deque(sorted([node for node, degree in indegree.items() if degree == 0], key=str))
        order: list[Hashable] = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in sorted(self.adjacency.get(node, {}), key=str):
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(indegree):
            raise ValueError("Graph contains a cycle.")
        return order


def computation_graph_topology(parents: dict[Hashable, Iterable[Hashable]]) -> list[Hashable]:
    """Topologically sort a computation graph described by parent links."""

    graph = Graph(directed=True)
    for node, node_parents in parents.items():
        graph.adjacency.setdefault(node, {})
        for parent in node_parents:
            graph.add_edge(parent, node)
    return graph.topological_sort()
