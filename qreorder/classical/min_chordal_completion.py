"""This module contains a minimum chordal completion solver."""

from __future__ import annotations
import logging
import warnings
from itertools import combinations
from typing import Iterator
import networkx as nx
import numpy as np
from numpy.typing import ArrayLike
from pads.LexBFS import LexBFS
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import milp
from scipy.sparse import spmatrix
from ..core import Solver
from ..preprocessing import preprocessing
from ..utils import make_elimination_graph


def is_chordal(graph: nx.Graph) -> bool:
    """Check if a graph is chordal."""
    return nx.is_chordal(graph)


def get_chordless_cycles(graph: nx.Graph, method: str = "merl") -> Iterator[list[int]]:
    """Generate chordless cycles of a graph.

    Args:
        graph: input graph.
        method: "nx" for using networkx algorithm. "merl" for using
          "Algorithm 2" as presented in
          `"A Benders Approach to the Minimum Chordal Completion Problem"
          by Bergman et al. <https://www.merl.com/publications/docs/TR2015-056.pdf>`_.

    Yields:
        A list of nodes corresponding to a chordless cycle.
    """
    if method == "nx":
        for chordless_cycle in nx.chordless_cycles(graph):
            # Skip 3-cycles
            if len(chordless_cycle) == 3:
                continue
            yield chordless_cycle
    elif method == "merl":
        nodes = list(graph)
        seen = []
        for ii, jj, kk in combinations(nodes, 3):
            for i, j, k in [(ii, jj, kk), (ii, kk, jj), (jj, ii, kk)]:
                if graph.has_edge(i, j) and graph.has_edge(j, k) and not graph.has_edge(i, k):
                    nodes_to_ignore = list(graph.neighbors(j))
                    nodes_to_ignore.append(j)
                    nodes_to_ignore.remove(i)
                    nodes_to_ignore.remove(k)
                    subgraph = nx.Graph(graph.edges())
                    subgraph.remove_nodes_from(nodes_to_ignore)
                    if nx.has_path(subgraph, i, k):
                        chordless_cycle = nx.shortest_path(subgraph, source=i, target=k) + [j]
                        if sorted(chordless_cycle) in seen:
                            continue
                        seen.append(sorted(chordless_cycle))
                        yield chordless_cycle
    else:
        raise ValueError("Unknown method.")


def _optimally_complete_to_chordal(
    graph: nx.Graph,
    method: str = "merl",
    max_n_constraints_per_iteration: int = 5000,
    max_n_constraints: int = 100000,
) -> tuple[nx.Graph, int, bool]:
    """Find minimum chordal completion of a graph.

    This function provides an implementation of the algorithm presented in
    `"A Benders Approach to the Minimum Chordal Completion Problem"
    by Bergman et al. <https://www.merl.com/publications/docs/TR2015-056.pdf>`_.

    Args:
        graph: graph to complete to chordal.
        method: method for enumerating chordless cycles ("nx": networkx,
          "merl": implementation provided in the referenced paper).
        max_n_constraints_per_iteration: maximum number of Bender cuts to add
          per iteration.
        max_n_constraints: maximum number of accumulated constraints.

    Returns:
        Tuple with three elements:
          Completed graph,
          Accumulated number of constraints,
          Status: True if chordal completion was found. False otherwise.
    """
    # Determine fill edges (i.e., those absent in the original graph)
    edges = graph.edges()
    fill_edges = nx.from_edgelist(nx.non_edges(graph)).edges()
    n_fill_edges = len(fill_edges)

    # Search for a minimum chordal completion
    accumulated_n_constraints = 0
    all_coefficients = []
    all_lower_bounds = []
    completed_graph = graph
    chordless_cycles_seen = []

    for i in range(max_n_constraints):
        # Print iteration number
        if i % 10 == 0:
            logging.info(f"Iteration {i}")

        # Find chordless cycles in current completion
        # Note: each element of `chordless_cycles` is a list of nodes along a cycle
        chordless_cycles = get_chordless_cycles(completed_graph, method=method)

        # Generate constraints for each chordless cycle
        n_constraints_for_this_iteration = 0
        for chordless_cycle in chordless_cycles:
            # Break if we've added enough constraints
            if n_constraints_for_this_iteration == max_n_constraints_per_iteration:
                break

            # Skip if we've already seen the chordless cycle
            # (in case a constraint wasn't met in a previous iteration)
            if chordless_cycle in chordless_cycles_seen:
                continue
            chordless_cycles_seen.append(chordless_cycle)

            # Keep track of number of constraints
            n_constraints_for_this_iteration += 1
            accumulated_n_constraints += 1

            # Get graph of this cycle
            cycle = nx.cycle_graph(chordless_cycle)

            # Get interior edges for this cycle
            interior_edges = nx.difference(nx.complete_graph(chordless_cycle), cycle).edges()

            # Get fill edges which are in this cycle
            fill_edges_in_cycle = nx.intersection(nx.Graph(fill_edges), cycle).edges()

            # Compute auxiliary constants for defining constraint
            constant1 = len(chordless_cycle) - 3
            constant2 = len(fill_edges_in_cycle) - 1
            constant3 = constant1 * constant2

            # Generate constraint
            coefficients = np.zeros(n_fill_edges, dtype=np.int16)
            coefficients[[fill_edge in interior_edges for fill_edge in fill_edges]] = 1
            coefficients[[fill_edge in fill_edges_in_cycle for fill_edge in fill_edges]] = -constant1
            lower_bound = -constant3

            all_coefficients.append(coefficients)
            all_lower_bounds.append(lower_bound)

        # Perform optimization to find a new completion
        A = np.stack(all_coefficients)
        idx_to_use = np.any(A, axis=0)
        A = A[:, idx_to_use]
        c = np.ones(A.shape[1])
        b_l = np.array(all_lower_bounds)
        b_u = np.full_like(b_l.astype(float), np.inf)
        bounds = Bounds(0, 1)  # binary optimization

        constraints = LinearConstraint(A, b_l, b_u)
        integrality = np.ones_like(c)
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        res_x = np.round(res.x).astype(np.int8)
        assert np.count_nonzero(res_x == 1) + np.count_nonzero(res_x == 0) == len(res_x)
        if res.status != 0 or not np.all(A @ res_x >= b_l):
            message = "Optimal solution not found by milp, better luck in the next iteration."
            warnings.warn(message, RuntimeWarning)
            logging.warning(message)
        solution_x = np.zeros(n_fill_edges, dtype=np.int8)
        solution_x[idx_to_use] = res_x

        # Complete graph
        edges_to_add = np.array(fill_edges)[solution_x.astype(bool)].tolist()
        completed_graph = nx.Graph(edges)
        completed_graph.add_edges_from(edges_to_add)

        # Check if we are done
        if is_chordal(completed_graph) or accumulated_n_constraints > max_n_constraints:
            break

    success = is_chordal(completed_graph)

    return completed_graph, accumulated_n_constraints, success


def _find_ordering_mcc(
    graph: nx.Graph,
    method: str = "merl",
    max_n_constraints_per_iteration: int = 5000,
    max_n_constraints: int = 100000,
) -> list[int]:
    """Find ordering using an algorithm for minimum chordal completion.

    Args:
        graph: graph to complete.
        method: method for enumerating chordless cycles (possible values
          are "nx" or "merl").
        max_n_constraints_per_iteration: maximum number of Bender cuts to add
          per iteration.
        max_n_constraints: maximum number of accumulated constraints.

    Returns:
        List of nodes sorted in optimal order.
    """
    # Find minimum chordal completion
    chordal_graph, n_constraints, success = _optimally_complete_to_chordal(
        graph,
        method=method,
        max_n_constraints_per_iteration=max_n_constraints_per_iteration,
        max_n_constraints=max_n_constraints,
    )
    if not success:
        message = "Graph could not be completed to chordal. " "Returning a possible ordering anyway."
        warnings.warn(message, UserWarning)
        logging.warning(message)

    logging.info(f"nr of constraints: {n_constraints}")

    # Find perfect elimination ordering
    ordering = list(LexBFS(chordal_graph))
    ordering.reverse()

    return ordering


class MinimumChordalCompletion(Solver):
    """Solver for finding ordering by minimum chordal completion."""

    def __init__(
        self,
        method: str = "merl",
        max_n_constraints_per_iteration: int = 5000,
        max_n_constraints: int = 100000,
        preprocessing: bool = False,
    ) -> None:
        """Init MinimumChordalCompletion.

        Args:
            method: method for enumerating chordless cycles (possible values
              are "nx" or "merl").
            max_n_constraints_per_iteration: maximum number of Bender cuts to add
              per iteration.
            max_n_constraints: maximum number of accumulated constraints.
            preprocessing: if ``True``, remove treelike structures and shrink cycles
              before applying the algorithm.
        """
        self.method = method
        self.max_n_constraints_per_iteration = max_n_constraints_per_iteration
        self.max_n_constraints = max_n_constraints
        self.preprocessing = preprocessing

    def get_ordering(self, matrix: ArrayLike | spmatrix) -> list[int]:  # noqa: D102
        if self.preprocessing:
            graph, _, partial_ordering = preprocessing(matrix)
        else:
            graph = make_elimination_graph(matrix)

        ordering = _find_ordering_mcc(
            graph,
            method=self.method,
            max_n_constraints_per_iteration=self.max_n_constraints_per_iteration,
            max_n_constraints=self.max_n_constraints,
        )
        if self.preprocessing:
            ordering = partial_ordering + ordering

        return ordering
