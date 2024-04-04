"""This module contains the ``BenderQUBO`` class."""

from __future__ import annotations
import warnings
from collections import Counter
from collections.abc import Iterable
from collections.abc import Mapping
from copy import deepcopy
from typing import Any
from typing import cast
import networkx as nx
import numpy as np
from dimod import Binary
from dimod import BinaryQuadraticModel
from dimod import quicksum
from dimod.core.sampler import Sampler
from .qubo import decode_sample
from .qubo import penalty


class BenderQUBO:
    """Clase to organize the Bender with QUBO.

    The ``BenderQUBO`` attempts to make a Chordal completion of the given graph using
    iteratively addend penalties based on Bender cuts.
    """

    def __init__(self, graph: nx.Graph) -> None:
        """Initialize the ``BenderQUBO`` class.

        Args:
            graph: Networkx representation of the graph.
        """
        self.graph = graph
        self._aux_total: int
        self._constraints: dict[tuple[int, ...], BinaryQuadraticModel]
        self._penalty_strengths: dict[tuple[int, ...], float]
        self._bqm: BinaryQuadraticModel
        self._run_metadata: dict[str, Any]

    def run(
        self,
        sampler: Sampler,
        sampler_kwargs: Mapping[str, Any] | None = None,
        max_iter: int = 100,
        initial_penalty_strength: float = 5,
        penalty_increase: float = 1.5,
        exclude_large_cycles: bool = False,
    ) -> nx.Graph:
        """Run the bender cut algorithm.

        Args:
            sampler: Sampler to solve each QUBO with.
            sampler_kwargs: Mapping containing the keyword arguments that the sampler
                should use.
            max_iter: Maximum number of iterations.
            initial_penalty_strength: Initial strength of the penalty function of each
                newly added bender cut. This value should be positive.
            penalty_increase: Multiplicative increase of the penalty strength for
                repeated violations. This value should be larger than 1.
            exclude_large_cycles: If ``True``, only add large cycles of minimal length
                in each iteration. This reduces the connectedness of the underlying
                QUBO at the cost of potential iterations. Default is ``False``.

        Returns:
            Networkx representation of the chordal completion of the original graph.
        """
        graph_k = self._make_graph_k([])
        if nx.is_chordal(self.graph):
            return graph_k
        self._reset_private_attributes()
        sampler_kwargs = {} if sampler_kwargs is None else sampler_kwargs
        for i in range(max_iter):
            self._update_bqm(
                graph_k,
                initial_penalty_strength,
                penalty_increase,
                exclude_large_cycles,
            )
            solution = self._sample_bqm(sampler, sampler_kwargs)
            graph_k = self._make_graph_k(solution)

            if nx.is_chordal(graph_k):
                self._run_metadata = {
                    "num_added_edges": len(solution),
                    "n_iterations": i + 1,
                    "sampler": sampler.__class__.__name__,
                    "max_iter": max_iter,
                    "initial_penalty_strength": initial_penalty_strength,
                    "penalty_increase": penalty_increase,
                    "solution": solution,
                }
                return graph_k

        warnings.warn(
            f"Graph could not be completed to chordal within {max_iter} iterations. "
            "Returning a possible ordering anyway.",
            UserWarning,
        )
        self._run_metadata = {
            "num_added_edges": len(solution),
            "n_iterations": i + 1,
            "sampler": sampler.__class__.__name__,
            "max_iter": max_iter,
            "initial_penalty_strength": initial_penalty_strength,
            "penalty_increase": penalty_increase,
            "solution": solution,
        }
        return graph_k

    def get_stats(self) -> dict[str, Any]:
        """Get some stats from the last run.

        Returns:
            Dictionary containing the stats.
        """
        strength_count = Counter(self._penalty_strengths.values())
        stats = {
            "num_constraints": len(self._constraints),
            "strength_count": dict(strength_count),
            "num_variables": self._bqm.num_variables,
            "num_interactions": self._bqm.num_interactions,
        }
        stats.update(self._run_metadata)
        return stats

    def _reset_private_attributes(self) -> None:
        """Reset the attributes of self."""
        self._bqm = BinaryQuadraticModel("BINARY")
        self._variables: set[str] = set()
        self._aux_total = 0
        self._constraints = {}
        self._penalty_strengths = {}
        self._run_metadata = {}

    def _update_bqm(
        self,
        graph_k: nx.Graph,
        initial_penalty_strength: float,
        penalty_increase: float,
        exclude_large_cycles: bool,
    ) -> None:
        """Update the BQM by looking for cycles in `graph_k`.

        If a Bender cut was already added to the BQM, increase the penalty strength with
        `penalty_increase`. If it was not yet present in the BQM, add it with an initial
        strength of `initial_penalty_strength`.

        Args:
            graph_k: Solution of the previous step. Chordless cycles will be sought in
                this graph.
            initial_penalty_strength: Initial strength of the penalty function of each
                newly added bender cut. This value should be positive.
            penalty_increase: Multiplicative increase of the penalty strength for
                repeated violations. This value should be larger than 1.
            exclude_large_cycles: If ``True``, only add large cycles of minimal length
                in each iteration. This reduces the connectedness of the underlying
                QUBO at the cost of potential iterations. Default is ``False``.
        """
        min_cycle_length = len(graph_k)
        if exclude_large_cycles:
            for cycle in nx.chordless_cycles(graph_k):
                if len(cycle) != 3:
                    min_cycle_length = min(min_cycle_length, len(cycle))

        for cycle in map(tuple, nx.chordless_cycles(graph_k)):
            # If a cycle has length of 3, then it is chordal.
            if len(cycle) == 3 or len(cycle) > min_cycle_length:
                continue

            # Create a unique representation of the cycle
            cycle = cycle[np.argmin(cycle) :] + cycle[: np.argmin(cycle)]
            # If the cycle has already been seen previously, increase penalty strength
            if cycle in self._constraints:
                self._increase_penalty_strength(cycle, penalty_increase)
            # Otherwise add it to the bqm
            else:
                self._add_new_penalty(cycle, initial_penalty_strength)

    def _extract_new_edge_variables(self, bqm: BinaryQuadraticModel) -> set[str]:
        """Look in the bqm for previously unseen edge variables."""
        new_variables = set()
        for variable in map(str, bqm.variables):
            if variable.startswith("x") and variable not in self._variables:
                new_variables.add(variable)
        return new_variables

    def _increase_penalty_strength(self, cycle: tuple[int], penalty_increase: float) -> None:
        """Incease the penalty strength.

        The penalty corresponding to `cycle` is increased by an amount relative to `penalty_increase`.

        This method should be called when a constraint is violated that has already been
        added in a previous iteration.

        This method updates the values of the attributes ``_bqm`` and
        ``_penalty_strengths``.
        """
        pen = self._constraints[cycle]
        strength = self._penalty_strengths[cycle]
        new_strength = strength * penalty_increase
        self._bqm += cast(BinaryQuadraticModel, (new_strength - strength) * pen)
        self._penalty_strengths[cycle] = new_strength

    def _add_new_penalty(self, cycle: tuple[int], initial_penalty_strength: float) -> None:
        """Add a new penalty corresponding to `cycle`.

        Also  give it an initial strength of `initial_penalty_strength`.
        an amount relative to `penalty_increase`.

        This method should be called when a constraint is violated that has not been
        seen in a previous iteration.

        This method updates the values of the attributes ``_bqm``, ``_constraints``,
        ``_penalty_strengths`` and ``_variables``.
        """
        pen, self._aux_total = penalty(cycle, self.graph, self._aux_total)
        self._constraints[cycle] = pen
        self._penalty_strengths[cycle] = initial_penalty_strength
        self._bqm += cast(BinaryQuadraticModel, initial_penalty_strength * pen)

        # Update the objective
        new_variables = self._extract_new_edge_variables(pen)
        if new_variables:
            self._bqm += cast(BinaryQuadraticModel, quicksum(map(Binary, new_variables)))
            self._variables = self._variables.union(new_variables)

    def _sample_bqm(self, sampler: Sampler, sampler_kwargs: Mapping[str, Any]) -> list[tuple[int, int]]:
        """Sample the BQM with the given `sampler`."""
        sampleset = sampler.sample(self._bqm, **sampler_kwargs)
        best_sample = sampleset.first.sample
        return decode_sample(best_sample)

    def _make_graph_k(self, edge_list: Iterable[tuple[int, int]]) -> nx.Graph:
        """Make the k-th iteration graph by adding the provided edges."""
        graph_k = deepcopy(self.graph)
        graph_k.add_edges_from(edge_list)
        return graph_k
