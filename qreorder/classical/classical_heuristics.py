"""This module contains classical heuristics for optimal ordering."""

from __future__ import annotations
import networkx as nx
import numpy as np
from networkx.utils import reverse_cuthill_mckee_ordering
from numpy.typing import ArrayLike
from pads.LexBFS import LexBFS
from scipy.sparse import spmatrix
from scipy.sparse.linalg import splu
from ..core import Solver
from ..preprocessing import preprocessing
from ..utils import make_elimination_graph


class ReverseCuthillMcKee(Solver):
    """Solver for finding ordering using the reverse Cuthill-McKee heuristic."""

    def __init__(
        self,
        preprocessing: bool = False,
    ) -> None:
        """Init ReverseCuthillMcKee.

        Args:
            preprocessing: if ``True``, remove treelike structures and shrink cycles
              before applying the algorithm.
        """
        self.preprocessing = preprocessing

    def get_ordering(self, matrix: ArrayLike | spmatrix) -> list[int]:  #  noqa: D102
        if self.preprocessing:
            graph, _, partial_ordering = preprocessing(matrix)
        else:
            graph = make_elimination_graph(matrix)

        ordering = list(reverse_cuthill_mckee_ordering(graph))
        if self.preprocessing:
            ordering = partial_ordering + ordering

        return ordering


class LexicographicBFS(Solver):
    """Solver for finding ordering using lexicographic breadth first search."""

    def __init__(
        self,
        preprocessing: bool = False,
    ) -> None:
        """Init LexicographicBFS.

        Args:
            preprocessing: if ``True``, remove treelike structures and shrink cycles
              before applying the algorithm.
        """
        self.preprocessing = preprocessing

    def get_ordering(self, matrix: ArrayLike | spmatrix) -> list[int]:  #  noqa: D102
        if self.preprocessing:
            graph, _, partial_ordering = preprocessing(matrix)
        else:
            graph = make_elimination_graph(matrix)

        ordering = list(LexBFS(graph))
        ordering.reverse()
        if self.preprocessing:
            ordering = partial_ordering + ordering

        return ordering


class COLAMD(Solver):
    """Solver that uses the approximate minimum degree column ordering."""

    def __init__(
        self,
        preprocessing: bool = False,
    ) -> None:
        """Init COLAMD.

        Args:
            preprocessing: if ``True``, remove treelike structures and shrink cycles
              before applying the algorithm.
        """
        self.preprocessing = preprocessing

    def get_ordering(self, matrix: ArrayLike | spmatrix) -> list[int]:  #  noqa: D102
        if self.preprocessing:
            graph, _, partial_ordering = preprocessing(matrix)
        else:
            graph = make_elimination_graph(matrix)

        graph.add_edges_from(zip(graph.nodes, graph.nodes))
        submatrix = nx.to_scipy_sparse_array(graph, format="csc")
        submatrix.setdiag(10)  # make it invertible
        super_lu = splu(submatrix)
        ordering = np.array(graph.nodes)[super_lu.perm_c].tolist()
        if self.preprocessing:
            ordering = partial_ordering + ordering

        return ordering
