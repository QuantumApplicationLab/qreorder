"""This module contains tests for the ``qubo`` module."""

import networkx as nx
import pytest
from dwave.samplers import TreeDecompositionSolver
from qreorder.quantum import BenderQUBO


@pytest.mark.parametrize("n_nodes", range(4, 9))
def test_bender_qubo_cycle_graphs(n_nodes: int) -> None:
    """Test Benders' QUBO approach to chordal completion."""
    original_graph = nx.cycle_graph(n_nodes)
    bender = BenderQUBO(original_graph)
    sampler = TreeDecompositionSolver()
    chordal_graph = bender.run(sampler)

    assert nx.is_chordal(chordal_graph)
    for edge in original_graph.edges():
        assert edge in chordal_graph.edges()
    assert chordal_graph.number_of_edges() == 2 * n_nodes - 3
