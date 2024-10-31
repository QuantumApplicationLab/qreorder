# SPDX-FileCopyrightText: 2024 Quantum Application Lab
#
# SPDX-License-Identifier: Apache-2.0
"""QUBO Module.

This module contains methods for creating the bender cut based QUBO formulation for the chordal completion problem.
"""

from __future__ import annotations
from collections.abc import Mapping
from collections.abc import Sequence
from math import floor
from typing import cast
import networkx as nx
import numpy as np
from dimod import Binary
from dimod import BinaryQuadraticModel
from dimod import quicksum
from qreorder.utils import get_internal
from qreorder.utils import get_xi


def encode_edge(edge: tuple[int, int], bias: float = 1) -> BinaryQuadraticModel:
    """Encode an edge into a binary variable with the given bias.

    For example, the edge (1,0) will turn into the variable x_0,1.

    Args:
        edge: Tuple of two (possible unordered) integer values representing the nodes.
        bias: Bias of the variable.

    Returns:
        A ``BinaryQuadraticModel`` with one variable with the given bias.
    """
    return BinaryQuadraticModel({f"x_{min(edge)},{max(edge)}": bias}, {}, 0, "BINARY")


def objective(graph: nx.Graph) -> BinaryQuadraticModel:
    r"""Create a QUBO formulation which minimizes the number of added edges.

    $$
    \min \sum_{e \in E_c} x_e
    $$

    Args:
        graph: ``Networkx`` representation of the graph to compute the objective for.

    Returns:
        ``BinaryQuadraticModel`` with the QUBO formulation of the objective.
    """
    comp_graph: nx.Graph = nx.complement(graph)
    return cast(BinaryQuadraticModel, quicksum(encode_edge(edge) for edge in comp_graph.edges))


def penalty(cycle: Sequence[int], graph: nx.Graph, aux_total: int) -> tuple[BinaryQuadraticModel, int]:
    r"""Create a penalty function for the Bender cut of the given cycle.

    The following bender cut is applied:
    $$
    \sum_{e\in \text{int}(C)} x_e \ge (|V(C)|-3)\left(\sum_{e\in \xi(C)}x_e+1-|\xi(C)|\right).
    $$
    Which is encoded using the penalty function $P$:
    $$
    P(\bm{x},\bm{z};C)
    =
    \left(
    \sum_{e\in \text{int}(C)} x_e - (|V(C)|-3)\sum_{e\in \xi(C)}x_e  - (|V(C)|-3)(1-|\xi(C)|)
    -\sum_{k=0}^{K-1}2^kz_k -z_{K}\left(|\text{int}(C)| - (|V(C)|-3)(1-|\xi(C)|) + 1 - 2^{K}\right)
    \right)^2,
    $$
    where $K = \lfloor\log_2|\text{int}(C)| - (|V(C)|-3)(1-|\xi(C)|)\rfloor$.

    Args:
        cycle: The cycle to make the Bender cut penalty function for. Should be an
            ``Iterable`` of nodes represented by integers.
        graph: ``Networkx`` representation of the original problem graph.
        aux_total: Number of auxiliary variables already use. This prevents using the
            same auxiliary variable in multiple penalty functions.

    Returns:
        ``BinaryQuadraticModel`` of the penalty function for the bender cut.
    """
    # Add variables for internal and xi edges
    internal = get_internal(cycle)
    xi = get_xi(cycle, graph)
    pen = quicksum(encode_edge(edge) for edge in internal.edges)
    if xi.edges:
        pen += -(len(cycle) - 3) * quicksum(encode_edge(edge) for edge in xi.edges)

    # Add constant term
    pen += -(len(cycle) - 3) * (1 - len(xi.edges))

    # Add slack
    max_slack = len(internal.edges) - (len(cycle) - 3) * (1 - len(xi.edges))
    n_aux = 1 + floor(np.log2(max_slack))
    if n_aux > 1:
        pen += -quicksum(Binary(f"z_{i+aux_total+1}", 2**i) for i in range(n_aux - 1))
    pen += Binary(f"z_{aux_total + n_aux}", -max_slack - 1 + 2 ** (n_aux - 1))

    # Square to produce the penalty term
    pen = cast(BinaryQuadraticModel, pen**2)
    return pen, aux_total + n_aux


def decode_sample(sample: Mapping[str, int]) -> list[tuple[int, int]]:
    """Decode the given sample to a list of added edges.

    Args:
        sample: Sample as retrieved by a D-Wave solver.

    Returns:
        The decoded list of edges to add to the graph. Each edge is represented by a
        2-tuple of integers.
    """
    solution = []
    for var_name, value in sample.items():
        if value and var_name.startswith("x"):
            var_name = var_name.removeprefix("x_")
            node_v, node_u = map(int, var_name.split(","))
            solution.append((node_u, node_v))
    return solution
