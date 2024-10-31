# SPDX-FileCopyrightText: 2024 Quantum Application Lab
#
# SPDX-License-Identifier: 	Apache-2.0
"""This module contains classical pre-processing methods."""

from __future__ import annotations
import networkx as nx
from numpy.typing import ArrayLike
from scipy.sparse import spmatrix
from qreorder.utils import make_elimination_graph


def remove_leaves(graph: nx.Graph, verbose: bool = True) -> tuple[nx.Graph, list[int]]:
    """Iteratively remove all leaves, i.e., nodes with degree 1.

    Args:
        graph: Networkx graph to remove the leaves from.
        verbose: bool level of print (default True)

    Returns:
        Graph with leaves removed, as well as removed nodes in order.
    """
    graph = graph.copy()
    removed_nodes_order = []
    old_nodes = len(graph)
    nodes_deg1 = [node_id for node_id, degree in graph.degree if degree == 1]
    while len(nodes_deg1):
        graph.remove_nodes_from(nodes_deg1)
        removed_nodes_order += nodes_deg1
        nodes_deg1 = [node_id for node_id, degree in graph.degree if degree == 1]

    if verbose:
        print(f"{len(removed_nodes_order)} removed ({(len(removed_nodes_order))/old_nodes:.1%})")
    return graph, removed_nodes_order


def shrink_loops(graph: nx.Graph, verbose: bool = True) -> tuple[nx.Graph, int, list[int]]:
    """Iteratively remove all degree 2 nodes that lie in a cycle.

    Args:
        graph: Networkx graph to remove degree 2 nodes in cycle from.
        verbose: bool level of print (default True)

    Returns:
        Graph with all degree 2 nodes in cycle removed, the amount of fill-in, and the
        removed nodes in order.
    """
    removed_nodes_order = []
    fill_in = 0
    graph = graph.copy()
    old_nodes = len(graph)

    # Find a node with degree 2 that is part of a cycle
    node = None
    for node_id, degree in graph.degree:
        if degree == 2 and nx.cycle_basis(graph, node_id):
            node = node_id
            break

    while node is not None:
        edge_to_add = tuple(graph[node])
        if edge_to_add not in graph.edges:
            graph.add_edge(*graph[node])
            fill_in += 1
        graph.remove_node(node)
        removed_nodes_order.append(node)

        # Find a new node with degree 2 that is part of a cycle
        node = None
        for node_id, degree in graph.degree:
            if degree == 2 and nx.cycle_basis(graph, node_id):
                node = node_id
                break

    if verbose:
        print(f"{len(removed_nodes_order)} removed ({(len(removed_nodes_order))/old_nodes:.1%})")
        print(f"{fill_in} fill-in")
    return graph, fill_in, removed_nodes_order


def reduce_cliques(graph: nx.Graph, verbose: bool = True) -> tuple[nx.Graph, list[int]]:
    """Iteratively remove all nodes for which the set of its neighbours and itself form a clique.

    Args:
        graph: Networkx graph to reduce the cliques from.
        verbose: bool level of print (default True)

    Returns:
        Graph with all cliques reduced, as well as removed nodes in order.
    """
    removed_nodes_order: list[int] = []
    graph = graph.copy()
    old_nodes = len(graph)
    new_nodes = 0

    while old_nodes - new_nodes:
        graph = graph.copy()
        old_nodes = len(graph)
        cliques = [clique for clique in nx.find_cliques(graph) if len(clique) > 2]
        to_remove = []
        for clique in cliques:
            for node in clique:
                if graph.degree[node] == len(clique) - 1:
                    to_remove.append(node)
        graph.remove_nodes_from(to_remove)
        new_nodes = len(graph)

    if verbose:
        print(f"{old_nodes-new_nodes} removed ({(old_nodes-new_nodes)/old_nodes:.1%})")
    return graph, removed_nodes_order


def preprocessing(matrix: ArrayLike | spmatrix) -> tuple[nx.Graph, int, list[int]]:
    """Run the full preprocessing of a graph given a square matrix.

    Args:
        matrix: Matrix to convert to graph and run full preprocessing for.

    Returns:
        Graph instance with all leaves and degree 2 vertices in a loop removed, the
        fill-in, as well as the order of the removed nodes.
    """
    removed_nodes_order = []
    fill_in = 0
    graph = make_elimination_graph(matrix)
    graph = graph.copy()
    old_nodes = len(graph)
    new_nodes = 0
    while old_nodes - new_nodes:
        graph = graph.copy()
        old_nodes = len(graph)
        leaves_removed = remove_leaves(graph, False)
        graph = leaves_removed[0]
        removed_nodes_order += leaves_removed[1]
        shrink = shrink_loops(graph, False)
        graph = shrink[0]
        fill_in += shrink[1]
        removed_nodes_order += shrink[2]
        new_nodes = len(graph)

    return graph, fill_in, removed_nodes_order
