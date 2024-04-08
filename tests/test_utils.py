"""This module contains tests for the qreorder.utils module."""

import networkx as nx
import numpy as np
import pytest
import scipy.sparse as spsp
from qreorder.utils import compute_fill_in
from qreorder.utils import compute_lu_fill_in
from qreorder.utils import get_internal
from qreorder.utils import get_xi
from qreorder.utils import make_elimination_graph


@pytest.mark.parametrize("cycle_size", range(3, 10))
def test_get_internal(cycle_size: int) -> None:
    """Test internal cycles."""
    cycle = list(range(cycle_size))
    internal = get_internal(cycle)
    assert len(internal) == cycle_size
    assert len(internal.edges) == cycle_size * (cycle_size - 3) / 2
    for i in range(cycle_size):
        edge = (i, (i + 1) % cycle_size)
        assert edge not in internal


@pytest.mark.parametrize("xi_size", range(6))
def test_get_xi(xi_size: int) -> None:
    """Test get xi."""
    base_graph: nx.Graph = nx.complete_graph(5)
    base_graph.add_edge(0, 6)
    for i in range(xi_size):
        base_graph.remove_edge(i, (i + 1) % 5)

    cycle = list(range(5))

    xi = get_xi(cycle, base_graph)
    assert len(xi) == len(cycle)
    assert len(xi.edges) == xi_size


@pytest.mark.parametrize("matrix", [np.ones((5, 5)), spsp.csc_matrix(np.ones((5, 5)))])
def test_make_elimination_graph(matrix) -> None:
    """Test the elimination grpah with numpy array."""
    graph = make_elimination_graph(matrix)
    assert len(graph.edges) == 10


def test_compute_fill_in() -> None:
    """Test computation of fill-in given an ordering."""
    size = 10
    graph = nx.wheel_graph(size)
    ordering = range(size)
    lu_fill_in = compute_lu_fill_in(graph, ordering)
    fill_in = compute_fill_in(graph, ordering)
    assert lu_fill_in == fill_in
    assert lu_fill_in == 27
