"""This module contains tests for the ``qubo`` module."""

import networkx as nx
from dimod import BinaryQuadraticModel
from qreorder.quantum.qubo import encode_edge
from qreorder.quantum.qubo import objective
from qreorder.quantum.qubo import penalty


class TestEncodeEdge:
    """Edge test class."""

    def test_order_invariance(self) -> None:
        """Test edge invariance."""
        edge1 = (1, 2)
        edge2 = (2, 1)
        assert encode_edge(edge1) == encode_edge(edge2)

    def test_encoding(self) -> None:
        """Test dege encoding."""
        edge = (1, 2)
        bqm = BinaryQuadraticModel("BINARY")
        bqm.add_linear("x_1,2", 10)
        assert bqm == encode_edge(edge, 10)


def test_objective() -> None:
    """Test objective function."""
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (0, 2)])

    expected_bqm = BinaryQuadraticModel("BINARY")
    expected_bqm.add_linear("x_1,2", 1)

    assert objective(graph) == expected_bqm

    graph.add_edges_from([(2, 3), (1, 3)])
    expected_bqm.add_linear("x_0,3", 1)
    assert objective(graph) == expected_bqm


class TestPenalty:
    """Penalty test."""

    def test_empty_xi(self) -> None:
        """Test empty."""
        graph = nx.cycle_graph(5)
        cycle = [0, 1, 2, 3, 4]
        bqm, aux_total = penalty(cycle, graph, 0)

        expected_bqm = BinaryQuadraticModel("BINARY")
        expected_bqm.add_linear_from(
            [
                ("x_0,2", 1),
                ("x_0,3", 1),
                ("x_1,3", 1),
                ("x_1,4", 1),
                ("x_2,4", 1),
                ("z_1", -1),
                ("z_2", -2),
            ]
        )
        expected_bqm.offset = -2

        assert aux_total == 2
        assert bqm == expected_bqm**2

    def test_nonempty_xi(self) -> None:
        """Test non empty."""
        graph = nx.cycle_graph(5)
        cycle = [0, 1, 2, 3]
        bqm, aux_total = penalty(cycle, graph, 2)

        expected_bqm = BinaryQuadraticModel("BINARY")
        expected_bqm.add_linear_from([("x_0,2", 1), ("x_1,3", 1), ("x_0,3", -1), ("z_3", -1), ("z_4", -1)])

        assert aux_total == 4
        assert bqm == expected_bqm**2
