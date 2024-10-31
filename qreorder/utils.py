# SPDX-FileCopyrightText: 2024 Quantum Application Lab
#
# SPDX-License-Identifier: 	Apache-2.0
"""This module contains general utility functions."""

from __future__ import annotations
from collections.abc import Hashable
from collections.abc import Iterable
from typing import Sequence
import networkx as nx
import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import csc_matrix
from scipy.sparse import diags
from scipy.sparse import spmatrix
from scipy.sparse.linalg import splu


def get_internal(cycle: Iterable[Hashable]) -> nx.Graph:
    """Get the internal graph of the cycle.

    The are all the edges that are not part of the cycle.

    Args:
        cycle: An iterable of nodes represented by hashable object (for example a list
            of integers).

    Returns:
        The internal graph of the cycle, represented by a ``networkx`` graph.
    """
    cycle_graph = nx.cycle_graph(cycle)
    return nx.complement(cycle_graph)


def get_xi(cycle: Iterable[Hashable], base_graph: nx.Graph) -> nx.Graph:
    """Get the xi graph of the cycle.

    The xi graph of the cycle are all edges that are present in the cycle, but not in
    the `base_graph`.

    Args:
        cycle: An iterable of nodes represented by hashable object (for example a list
            of integers).
        base_graph: Networkx representation of the base graph to check against.

    Returns:
        The xi graph of the cycle, represented by a ``networkx`` graph.
    """
    xi = nx.cycle_graph(cycle)
    for edge in xi.edges:
        if edge in base_graph.edges:
            xi.remove_edge(*edge)
    return xi


def make_elimination_graph(matrix: ArrayLike | spmatrix) -> nx.Graph:
    """Create an elimination graph from the given matrix.

    Args:
        matrix: Matrix to make the elimination graph of.

    Returns:
        Networkx representation of the elimination graph.
    """
    adj_prep = (np.abs(matrix) > 0).astype("int")
    if isinstance(matrix, spmatrix):
        adj_prep -= diags(adj_prep.diagonal())
    elif isinstance(matrix, np.ndarray):
        np.fill_diagonal(adj_prep, 0)
    else:
        raise TypeError("Matrix should be ArrayLike or sp.spmatrix")
    graph = nx.from_numpy_array(adj_prep)
    assert not graph.is_directed()

    return graph


def compute_fill_in(graph: nx.Graph, ordering: Iterable[int]) -> int:
    """Compute fill-in by performing elimination.

    Note: does not require ordering of all vertices.

    Args:
        graph: graph used to perform the elimination.
        ordering: ordering to perform the elimination.

    Returns:
        Amount of fill-in.
    """
    # Make copy
    graph = graph.copy()

    # Perform elimination
    fill = 0
    for node in ordering:
        neighbors = graph.neighbors(node)

        edges_to_remove = list(graph.edges(node))
        edges_to_add = [edge for edge in nx.complete_graph(neighbors).edges if not graph.has_edge(*edge)]

        graph.remove_edges_from(edges_to_remove)
        graph.add_edges_from(edges_to_add)

        fill += len(edges_to_add)

    return fill


def compute_lu_fill_in(graph: nx.Graph, ordering: Sequence[int]) -> int:
    """Compute fill-in by performing an LU decomposition.

    Args:
        graph: graph to generate an adjacency matrix.
        ordering: ordering to permute rows and columns.

    Returns:
        Amount of fill-in.
    """
    graph.add_edges_from(zip(graph.nodes, graph.nodes))
    matrix = nx.to_scipy_sparse_array(graph, format="csc")
    matrix.setdiag(10)  # make it invertible
    size = len(ordering)
    p = csc_matrix((np.ones(size), (np.arange(size), ordering)))
    lu = splu(p @ matrix @ p.T, permc_spec="NATURAL", diag_pivot_thresh=0)
    return int(((lu.L + lu.U).count_nonzero() - matrix.count_nonzero()) / 2)
