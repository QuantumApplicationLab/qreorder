"""This module contains visualisation tools."""

from __future__ import annotations
from collections.abc import Hashable
from collections.abc import Mapping
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.axis import Axis


def draw_chordal_completion(
    original_graph: nx.Graph,
    chordal_graph: nx.Graph,
    ax: Axis | None = None,
    pos: Mapping[Hashable, tuple[float, float]] | None = None,
    with_labels: bool = False,
) -> None:
    """Draw the chordal completion where the completion edges are highlighted in red.

    Args:
        original_graph: Networkx representation of the original graph.
        chordal_graph: Networkx representation of the chordal completion of the
            original graph.
        ax: Optional matplotlib ``ax`` to plot on. If ``None``, a new figure with ax
            will be made.
        pos: Optional positions of the nodes.
        with_labels: Boolean value stating whether to draw the labels or not.
    """
    if ax is None:
        _, (ax) = plt.subplots()

    edge_colors = []
    for edge in chordal_graph.edges:
        if edge not in original_graph.edges:
            edge_colors.append("red")
        else:
            edge_colors.append("black")

    nx.draw(chordal_graph, ax=ax, pos=pos, edge_color=edge_colors, with_labels=with_labels)
