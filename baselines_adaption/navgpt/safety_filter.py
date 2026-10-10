"""Depth-based safety filtering for NavGPT action candidates."""

from typing import Dict, List
import numpy as np


def _forward_clear(depth_img, min_safe_dist: float) -> bool:
    if depth_img is None:
        return False
    h, w = depth_img.shape
    x1, x2 = int(w * 0.35), int(w * 0.65)
    y1, y2 = int(h * 0.35), int(h * 0.75)
    roi = depth_img[y1:y2, x1:x2]
    valid = roi[roi > 0.5]
    if valid.size == 0:
        return False
    return float(np.percentile(valid, 20)) >= min_safe_dist


def _down_clear(depth_img, min_ground_clearance: float) -> bool:
    if depth_img is None:
        return True
    valid = depth_img[depth_img > 0.5]
    if valid.size == 0:
        return True
    # DepthPlanar towards the camera optical axis: smaller means close obstacle/ground.
    # For side cameras this is not used; only used when mapped to move_down check.
    return float(np.percentile(valid, 10)) > min_ground_clearance


def filter_safe_actions(
    depth_by_camera: Dict[str, np.ndarray],
    current_altitude: float,
    *,
    min_safe_dist: float = 8.0,
    min_ground_clearance: float = 4.0,
    max_altitude: float = 80.0,
) -> List[str]:
    """Return safe movement actions from depth + altitude heuristics."""

    safe = []

    if _forward_clear(depth_by_camera.get("front_camera"), min_safe_dist):
        safe.append("move_forward")
    if _forward_clear(depth_by_camera.get("back_camera"), min_safe_dist):
        safe.append("move_backward")
    if _forward_clear(depth_by_camera.get("left_camera"), min_safe_dist):
        safe.append("move_left")
    if _forward_clear(depth_by_camera.get("right_camera"), min_safe_dist):
        safe.append("move_right")

    # current_altitude: positive-up relative height
    if current_altitude < max_altitude:
        safe.append("move_up")

    # Only allow descending when the drone has enough altitude clearance.
    # No bottom camera is available, so we use altitude directly.
    if current_altitude > min_ground_clearance:
        safe.append("move_down")

    # Ensure at least one movement action is available.
    if not safe:
        safe = ["move_up"] if current_altitude < max_altitude else ["move_forward"]

    return safe
