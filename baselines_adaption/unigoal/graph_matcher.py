"""GraphMatcher: simplified overlap computation between GoalGraph and SceneGraph.

No edges, no networkx — just set intersection on node labels to produce
a coverage ratio and lists of found/missing objects with positions.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple

from .goal_graph import GoalGraph
from .scene_graph import SceneGraph


@dataclass
class OverlapResult:
    """Result of comparing the scene graph against the goal graph.

    Attributes
    ----------
    coverage : float
        Fraction of goal objects found in the scene (0.0 to 1.0).
    found_objects : set[str]
        Goal objects that have been detected.
    missing_objects : set[str]
        Goal objects not yet detected.
    found_positions : dict[str, tuple[float, float] | None]
        World-coordinate centroids of found objects.
    """
    coverage: float = 0.0
    found_objects: Set[str] = field(default_factory=set)
    missing_objects: Set[str] = field(default_factory=set)
    found_positions: Dict[str, Optional[Tuple[float, float]]] = field(
        default_factory=dict
    )


def compute_overlap(
    goal_graph: GoalGraph,
    scene_graph: SceneGraph,
) -> OverlapResult:
    """Compute how much of the goal graph has been observed.

    Parameters
    ----------
    goal_graph : GoalGraph
        The parsed task goal.
    scene_graph : SceneGraph
        The incrementally-built scene from YOLO detections.

    Returns
    -------
    OverlapResult
    """
    goal_objects = goal_graph.all_objects  # set of lowercased labels
    scene_labels = scene_graph.get_labels()  # set of lowercased labels

    if not goal_objects:
        return OverlapResult(coverage=0.0)

    found = goal_objects & scene_labels
    missing = goal_objects - scene_labels

    coverage = len(found) / len(goal_objects)

    found_positions = {}
    for label in found:
        found_positions[label] = scene_graph.get_detection_centroid(label)

    return OverlapResult(
        coverage=coverage,
        found_objects=found,
        missing_objects=missing,
        found_positions=found_positions,
    )
