"""Tests for graph helpers."""

import numpy as np

from math_for_ml.graphs import Graph, computation_graph_topology


def test_adjacency_and_laplacian_have_expected_structure() -> None:
    """The graph Laplacian should have zero row sums."""

    graph = Graph()
    graph.add_edge("A", "B")
    graph.add_edge("B", "C")
    adjacency = graph.adjacency_matrix()
    laplacian = graph.laplacian_matrix()
    assert adjacency.shape == (3, 3)
    assert np.allclose(laplacian.sum(axis=1), 0.0)


def test_traversals_and_shortest_path_work() -> None:
    """Traversal orders and shortest paths should be valid."""

    graph = Graph()
    graph.add_edge("A", "B", weight=1.0)
    graph.add_edge("B", "C", weight=2.0)
    graph.add_edge("A", "C", weight=4.0)
    assert graph.bfs("A")[0] == "A"
    assert graph.dfs("A")[0] == "A"
    distance, path = graph.shortest_path("A", "C")
    assert distance == 3.0
    assert path == ["A", "B", "C"]


def test_topological_sort_on_computation_graph() -> None:
    """Parent links should yield a valid topological order."""

    graph = Graph(directed=True)
    graph.add_edge("x", "y")
    graph.add_edge("y", "z")
    order = graph.topological_sort()
    assert order == ["x", "y", "z"]
    assert computation_graph_topology({"z": ["x", "y"], "y": ["x"], "x": []})[0] == "x"
