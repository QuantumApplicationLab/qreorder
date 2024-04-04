"""This module defines an interface for solvers."""

from __future__ import annotations
from abc import ABC
from abc import abstractmethod
from numpy.typing import ArrayLike


class Solver(ABC):
    """Abstract base class for solvers."""

    @abstractmethod
    def get_ordering(self, matrix: ArrayLike) -> list[int]:
        """Find optimal ordering.

        Args:
            matrix: adjacency matrix.

        Returns:
            List of nodes sorted in optimal order.
        """
