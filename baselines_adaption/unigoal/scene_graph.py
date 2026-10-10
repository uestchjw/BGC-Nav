"""SceneGraph: incremental object registry built from YOLO detections.

Maintains a flat list of detected objects with world positions, built
incrementally from YOLO-World detections. Objects with the same label
within a deduplication distance are merged (detection count increases).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# Detections of the same label within this distance (metres) are merged.
_DEDUP_DISTANCE = 10.0


@dataclass
class SceneNode:
    """A single detected object in the scene."""
    label: str
    world_x: float
    world_y: float
    confidence: float
    grid_x: int = 0
    grid_y: int = 0
    step_first_seen: int = 0
    step_last_seen: int = 0
    detection_count: int = 1


class SceneGraph:
    """Flat registry of detected objects with spatial deduplication.

    Usage
    -----
    Called from the agent's detection loop::

        for each YOLO detection:
            scene_graph.add_detection("backpack", wx, wy, conf, step)
    """

    def __init__(self, dedup_distance: float = _DEDUP_DISTANCE):
        self.nodes: List[SceneNode] = []
        self.dedup_distance = dedup_distance

    def add_detection(
        self,
        label: str,
        world_x: float,
        world_y: float,
        confidence: float,
        step: int,
        grid_x: int = 0,
        grid_y: int = 0,
    ) -> SceneNode:
        """Add or merge a detection into the scene graph.

        If an existing node with the same label is within
        ``dedup_distance`` metres, the detection is merged (running
        average position, max confidence, count incremented).
        Otherwise a new node is created.

        Returns the node (new or updated).
        """
        label_lower = label.lower()

        # Try to merge with an existing node
        best_node = None
        best_dist = float("inf")
        for node in self.nodes:
            if node.label != label_lower:
                continue
            dist = np.sqrt(
                (node.world_x - world_x) ** 2
                + (node.world_y - world_y) ** 2
            )
            if dist < self.dedup_distance and dist < best_dist:
                best_dist = dist
                best_node = node

        if best_node is not None:
            # Merge: running average position weighted by detection count
            n = best_node.detection_count
            best_node.world_x = (best_node.world_x * n + world_x) / (n + 1)
            best_node.world_y = (best_node.world_y * n + world_y) / (n + 1)
            best_node.confidence = max(best_node.confidence, confidence)
            best_node.detection_count += 1
            best_node.step_last_seen = step
            best_node.grid_x = grid_x if grid_x else best_node.grid_x
            best_node.grid_y = grid_y if grid_y else best_node.grid_y
            return best_node

        # Create new node
        new_node = SceneNode(
            label=label_lower,
            world_x=world_x,
            world_y=world_y,
            confidence=confidence,
            grid_x=grid_x,
            grid_y=grid_y,
            step_first_seen=step,
            step_last_seen=step,
            detection_count=1,
        )
        self.nodes.append(new_node)
        return new_node

    def get_labels(self) -> Set[str]:
        """Return the set of unique detected object labels."""
        return {node.label for node in self.nodes}

    def get_nodes_by_label(self, label: str) -> List[SceneNode]:
        """Return all nodes matching a label (case-insensitive)."""
        label_lower = label.lower()
        return [n for n in self.nodes if n.label == label_lower]

    def get_detection_centroid(
        self, label: str
    ) -> Optional[Tuple[float, float]]:
        """Return the average world position of all nodes with this label.

        Returns None if no detections of this label exist.
        """
        nodes = self.get_nodes_by_label(label)
        if not nodes:
            return None
        xs = [n.world_x for n in nodes]
        ys = [n.world_y for n in nodes]
        return (float(np.mean(xs)), float(np.mean(ys)))

    def get_all_positions(self) -> List[Tuple[float, float]]:
        """Return world positions of all nodes."""
        return [(n.world_x, n.world_y) for n in self.nodes]

    def get_summary(self) -> Dict[str, int]:
        """Return a {label: count} summary of all detected objects."""
        summary: Dict[str, int] = {}
        for node in self.nodes:
            summary[node.label] = summary.get(node.label, 0) + 1
        return summary
