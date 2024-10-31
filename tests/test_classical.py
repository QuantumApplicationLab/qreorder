# SPDX-FileCopyrightText: 2024 Quantum Application Lab
#
# SPDX-License-Identifier: 	Apache-2.0
"""This module contains tests for classical algorithms for ordering."""

from __future__ import annotations
from typing import Sequence
import networkx as nx
import numpy as np
import pytest
from networkx.algorithms.chordal import complete_to_chordal_graph
from numpy.typing import NDArray
from pads.LexBFS import LexBFS
from qreorder.classical import COLAMD
from qreorder.classical import LexicographicBFS
from qreorder.classical import MinimumChordalCompletion
from qreorder.classical import ReverseCuthillMcKee
from qreorder.classical.min_chordal_completion import get_chordless_cycles
from qreorder.classical.min_chordal_completion import is_chordal
from qreorder.utils import compute_fill_in


def _is_unique(ordering: Sequence[int]) -> bool:
    """Check if ordering has duplicate elements."""
    return sorted(np.unique(ordering).tolist()) == sorted(ordering)


def _load_wheel_graph(size: int) -> tuple[nx.Graph, NDArray[np.float_]]:
    """Load a test wheel graph and adjacency matrix."""
    graph = nx.wheel_graph(size)
    adj: NDArray[np.float_] = nx.adjacency_matrix(graph).todense()
    return graph, adj


def _load_mesh_graph() -> tuple[nx.Graph, NDArray[np.float_]]:
    """Load a test mesh graph and adjacency matrix."""
    graph = nx.grid_graph(dim=(3, 3))
    graph = nx.convert_node_labels_to_integers(graph)
    adj: NDArray[np.float_] = nx.adjacency_matrix(graph).todense()
    return graph, adj


def test_get_ordering_rcm_without_preprocessing() -> None:
    """Test Cuthill-McKee heuristic."""
    # Get graph and adjacency matrix
    graph, adj = _load_wheel_graph(30)

    # Get ordering
    rcm = ReverseCuthillMcKee(preprocessing=False)
    ordering = rcm.get_ordering(adj)
    assert _is_unique(ordering)

    # Compute fill-in
    fill_in = compute_fill_in(graph, ordering)

    print(f"Fill-in: {fill_in}")
    print(f"Ordering: {ordering}")

    # fmt: off
    assert ordering == [
        29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
        15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 1, 0, 4, 2, 3,
    ]
    # fmt: on
    assert fill_in == 26


def test_get_ordering_rcm_with_preprocessing() -> None:
    """Test Cuthill-McKee heuristic with preprocessing."""
    # Get graph and adjacency matrix
    graph, adj = _load_mesh_graph()

    # Get ordering
    rcm = ReverseCuthillMcKee(preprocessing=True)
    ordering = rcm.get_ordering(adj)
    assert _is_unique(ordering)

    # Compute fill-in
    fill_in = compute_fill_in(graph, ordering)

    print(f"Ordering: {ordering}")
    print(f"Fill-in: {fill_in}")

    assert ordering == [0, 2, 6, 8, 1, 4, 5, 3, 7]
    assert fill_in == 5


def test_get_ordering_lbfs_without_preprocessing() -> None:
    """Test LexicographicBFS."""
    # Get graph and adjacency matrix
    graph, adj = _load_wheel_graph(30)

    # Get ordering
    rcm = LexicographicBFS(preprocessing=False)
    ordering = rcm.get_ordering(adj)
    assert _is_unique(ordering)

    # Compute fill-in
    fill_in = compute_fill_in(graph, ordering)

    print(f"Fill-in: {fill_in}")
    print(f"Ordering: {ordering}")

    # fmt: off
    assert ordering == [
        16, 15, 17, 14, 18, 13, 19, 12, 20, 11, 21, 10, 22, 9,
        23, 8, 24, 7, 25, 6, 26, 5, 27, 4, 28, 3, 29, 2, 1, 0,
    ]
    # fmt: on
    assert fill_in == 26


def test_get_ordering_lbfs_with_preprocessing() -> None:
    """Test LexicographicBFS heuristic with preprocessing."""
    # Get graph and adjacency matrix
    graph, adj = _load_mesh_graph()

    # Get ordering
    rcm = LexicographicBFS(preprocessing=True)
    ordering = rcm.get_ordering(adj)
    assert _is_unique(ordering)

    # Compute fill-in
    fill_in = compute_fill_in(graph, ordering)

    print(f"Ordering: {ordering}")
    print(f"Fill-in: {fill_in}")

    assert ordering == [0, 2, 6, 8, 7, 5, 4, 3, 1]
    assert fill_in == 5


def test_get_ordering_colamd_without_preprocessing() -> None:
    """Test COLAMD."""
    # Get graph and adjacency matrix
    graph, adj = _load_wheel_graph(30)

    # Get ordering
    rcm = COLAMD(preprocessing=False)
    ordering = rcm.get_ordering(adj)
    assert _is_unique(ordering)

    # Compute fill-in
    fill_in = compute_fill_in(graph, ordering)

    print(f"Fill-in: {fill_in}")
    print(f"Ordering: {ordering}")

    # fmt: off
    assert ordering == [
        29, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 25,
        26, 27, 28, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1,
    ]
    # fmt: on
    assert fill_in == 351


def test_get_ordering_colamd_with_preprocessing() -> None:
    """Test COLAMD heuristic with preprocessing."""
    # Get graph and adjacency matrix
    graph, adj = _load_mesh_graph()

    # Get ordering
    rcm = COLAMD(preprocessing=True)
    ordering = rcm.get_ordering(adj)
    assert _is_unique(ordering)

    # Compute fill-in
    fill_in = compute_fill_in(graph, ordering)

    print(f"Ordering: {ordering}")
    print(f"Fill-in: {fill_in}")

    assert ordering == [0, 2, 6, 8, 1, 3, 4, 5, 7]
    assert fill_in == 5


@pytest.mark.filterwarnings("error:Graph could not be completed to chordal")
def test_get_ordering_mcc_without_preprocessing() -> None:
    """Test minimum chordal completion."""
    # Get graph and adjacency matrix
    graph, adj = _load_wheel_graph(20)

    for method in ["nx", "merl"]:
        # Get ordering
        mcc = MinimumChordalCompletion(
            method=method,
            max_n_constraints_per_iteration=5000,
            max_n_constraints=100000,
            preprocessing=False,
        )
        ordering = mcc.get_ordering(adj)
        assert _is_unique(ordering)

        # Compute fill-in
        fill_in = compute_fill_in(graph, ordering)

        print(f"Ordering: {ordering}")
        print(f"Fill-in: {fill_in}")

        assert 0 in ordering[-4:]
        assert fill_in == 16


@pytest.mark.filterwarnings("error:Graph could not be completed to chordal")
def test_get_ordering_mcc_with_preprocessing() -> None:
    """Test minimum chordal completion with preprocessing."""
    # Get graph and adjacency matrix
    graph, adj = _load_mesh_graph()

    for method in ["nx", "merl"]:
        # Get ordering
        mcc = MinimumChordalCompletion(
            method=method,
            max_n_constraints_per_iteration=5000,
            max_n_constraints=100000,
            preprocessing=True,
        )
        ordering = mcc.get_ordering(adj)
        assert _is_unique(ordering)

        # Compute fill-in
        fill_in = compute_fill_in(graph, ordering)

        print(f"Ordering: {ordering}")
        print(f"Fill-in: {fill_in}")

        assert ordering == [0, 2, 6, 8, 7, 5, 4, 3, 1]
        assert fill_in == 5


def _load_chordal_graph(size: int) -> nx.Graph:
    """Load a test chordal graph."""
    # Load a graph
    graph = nx.wheel_graph(size)

    # Make it chordal
    chordal_graph, _ = complete_to_chordal_graph(graph)

    return chordal_graph


def test_compute_fill_in() -> None:
    """Check that the fill in of a chordal graph is zero."""
    # Load a graph
    graph = _load_chordal_graph(400)

    # Find perfect elimination ordering
    ordering = list(LexBFS(graph))
    ordering.reverse()
    assert _is_unique(ordering)

    # Compute fill-in
    fill_in = compute_fill_in(graph, ordering)

    # Check if the result is as expected
    assert fill_in == 0


def test_LexBFS() -> None:
    """Test the generation of an ordering based on a chordal graph."""
    # Load a graph
    graph = _load_chordal_graph(400)

    # Find perfect elimination ordering
    ordering = list(LexBFS(graph))
    ordering.reverse()
    assert _is_unique(ordering)

    # Check if the result is as expected
    # fmt: off
    assert ordering == [
        201, 200, 202, 199, 203, 198, 204, 197, 205, 196, 206, 195, 207, 194,
        208, 193, 209, 192, 210, 191, 211, 190, 212, 189, 213, 188, 214, 187,
        215, 186, 216, 185, 217, 184, 218, 183, 219, 182, 220, 181, 221, 180,
        222, 179, 223, 178, 224, 177, 225, 176, 226, 175, 227, 174, 228, 173,
        229, 172, 230, 171, 231, 170, 232, 169, 233, 168, 234, 167, 235, 166,
        236, 165, 237, 164, 238, 163, 239, 162, 240, 161, 241, 160, 242, 159,
        243, 158, 244, 157, 245, 156, 246, 155, 247, 154, 248, 153, 249, 152,
        250, 151, 251, 150, 252, 149, 253, 148, 254, 147, 255, 146, 256, 145,
        257, 144, 258, 143, 259, 142, 260, 141, 261, 140, 262, 139, 263, 138,
        264, 137, 265, 136, 266, 135, 267, 134, 268, 133, 269, 132, 270, 131,
        271, 130, 272, 129, 273, 128, 274, 127, 275, 126, 276, 125, 277, 124,
        278, 123, 279, 122, 280, 121, 281, 120, 282, 119, 283, 118, 284, 117,
        285, 116, 286, 115, 287, 114, 288, 113, 289, 112, 290, 111, 291, 110,
        292, 109, 293, 108, 294, 107, 295, 106, 296, 105, 297, 104, 298, 103,
        299, 102, 300, 101, 301, 100, 302, 99, 303, 98, 304, 97, 305, 96,
        306, 95, 307, 94, 308, 93, 309, 92, 310, 91, 311, 90, 312, 89,
        313, 88, 314, 87, 315, 86, 316, 85, 317, 84, 318, 83, 319, 82,
        320, 81, 321, 80, 322, 79, 323, 78, 324, 77, 325, 76, 326, 75,
        327, 74, 328, 73, 329, 72, 330, 71, 331, 70, 332, 69, 333, 68,
        334, 67, 335, 66, 336, 65, 337, 64, 338, 63, 339, 62, 340, 61,
        341, 60, 342, 59, 343, 58, 344, 57, 345, 56, 346, 55, 347, 54,
        348, 53, 349, 52, 350, 51, 351, 50, 352, 49, 353, 48, 354, 47,
        355, 46, 356, 45, 357, 44, 358, 43, 359, 42, 360, 41, 361, 40,
        362, 39, 363, 38, 364, 37, 365, 36, 366, 35, 367, 34, 368, 33,
        369, 32, 370, 31, 371, 30, 372, 29, 373, 28, 374, 27, 375, 26,
        376, 25, 377, 24, 378, 23, 379, 22, 380, 21, 381, 20, 382, 19,
        383, 18, 384, 17, 385, 16, 386, 15, 387, 14, 388, 13, 389, 12,
        390, 11, 391, 10, 392, 9, 393, 8, 394, 7, 395, 6, 396, 5,
        397, 4, 398, 3, 399, 2, 1, 0,
    ]
    # fmt: on


def test_is_chordal() -> None:
    """Test verification of chordal graphs."""
    graph = _load_chordal_graph(20)
    assert is_chordal(graph)
    for method in ["nx", "merl"]:
        assert len(list(get_chordless_cycles(graph, method=method))) == 0


def test_get_chordless_cycles() -> None:
    """Test generation of chordless cycles."""
    # Load graph
    graph, _ = _load_mesh_graph()

    # Test with "nx" method
    chordless_cycles = list(get_chordless_cycles(graph, method="nx"))
    print(f"Chordless cycles found with nx: {chordless_cycles}")
    # fmt: off
    assert chordless_cycles ==  [
        [3, 0, 1, 4], [3, 0, 1, 2, 5, 8, 7, 6],
        [4, 1, 2, 5], [6, 3, 4, 7], [5, 8, 7, 4],
    ]
    # fmt: on

    # Test with "merl" method
    chordless_cycles = list(get_chordless_cycles(graph, method="merl"))
    print(f"Chordless cycles found with merl: {chordless_cycles}")
    # fmt: off
    assert chordless_cycles ==  [
        [0, 3, 6, 7, 8, 5, 2, 1], [1, 4, 3, 0], [2, 5, 4, 1],
        [4, 7, 6, 3], [5, 8, 7, 4],
    ]
    # fmt: on
