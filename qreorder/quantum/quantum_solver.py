"""This module contains a quantum ordering algorithm.

The quantum solver approximates solutions to the equivalent minimum chordal completion
problem using a benders' cut approach.
"""

from __future__ import annotations
from collections.abc import Mapping
from typing import Any
from dimod.core.sampler import Sampler
from dwave.samplers import SimulatedAnnealingSampler
from numpy.typing import ArrayLike
from pads.LexBFS import LexBFS
from scipy.sparse import spmatrix
from ..core import Solver
from ..preprocessing import preprocessing
from ..utils import make_elimination_graph
from .bender import BenderQUBO


class QuantumSolver(Solver):
    """Solver for finding ordering by minimum chordal completion."""

    def __init__(
        self,
        sampler: Sampler | None = None,
        sampler_kwargs: Mapping[str, Any] | None = None,
        max_iter: int = 100,
        initial_penalty_strength: float = 5,
        penalty_increase: float = 1.5,
        exclude_large_cycles: bool = False,
        preprocessing: bool = False,
    ) -> None:
        """Init of the Quantum Solver using a benders' cut algorithm.

        Args:
            sampler: ``Sampler`` to solve each QUBO with. If no ``Sampler`` is given, a
                ``SimulatedAnnealingSampler`` is used.
            sampler_kwargs: Mapping containing the keyword arguments that the
                ``Sampler`` should use.
            max_iter: Maximum number of iterations.
            initial_penalty_strength: Initial strength of the penalty function of each
                newly added bender cut. This value should be positive.
            penalty_increase: Multiplicative increase of the penalty strength for
                repeated violations. This value should be larger than 1.
            exclude_large_cycles: If ``True``, only add large cycles of minimal length
                in each iteration. This reduces the connectedness of the underlying
                QUBO at the cost of potential iterations. Default is ``False``.
            preprocessing: If ``True`` remove treelike structures and shrink cycles
                before applying the quantum algorithm. Default is ``False``.
        """
        self.sampler = SimulatedAnnealingSampler() if sampler is None else sampler
        self.sampler_kwargs = sampler_kwargs
        self.max_iter = max_iter
        self.initial_penalty_strength = initial_penalty_strength
        self.penalty_increase = penalty_increase
        self.exclude_large_cycles = exclude_large_cycles
        self.preprocessing = preprocessing

    def get_ordering(self, matrix: ArrayLike | spmatrix) -> list[int]:
        """Approximate an optimal ordering.

        Args:
            matrix: adjacency matrix.

        Returns:
            List of nodes sorted in optimal order.
        """
        if self.preprocessing:
            graph, _, partial_ordering = preprocessing(matrix)
        else:
            graph = make_elimination_graph(matrix)

        bender = BenderQUBO(graph)

        chordal_graph = bender.run(
            sampler=self.sampler,
            sampler_kwargs=self.sampler_kwargs,
            max_iter=self.max_iter,
            initial_penalty_strength=self.initial_penalty_strength,
            penalty_increase=self.penalty_increase,
            exclude_large_cycles=self.exclude_large_cycles,
        )

        # Find perfect elimination ordering
        ordering = list(LexBFS(chordal_graph))
        ordering.reverse()

        if self.preprocessing:
            ordering = partial_ordering + ordering

        return ordering
