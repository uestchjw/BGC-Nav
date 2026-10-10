"""
Frontier-based goal selection for the SemExp ASAR adaptation.

Replaces the learned RL Global Policy from the original SemExp with a
heuristic frontier-based exploration strategy, augmented with semantic
guidance (e.g. prioritise detected-but-unvisited persons).

A *frontier* is the boundary between explored free-space and unexplored
space.  The agent selects the largest frontier cluster and navigates to
its centroid.
"""

import numpy as np
import cv2
import skimage.morphology

from .utils.fmm_planner import FMMPlanner


class FrontierGoalSelector:
    """Select a long-term navigation goal using frontier exploration.

    Parameters
    ----------
    grid_size : int
    cell_size : float
    replan_dist : float
        Re-plan when the agent is within this many world-metres of the
        current goal.
    min_frontier_size : int
        Ignore frontier clusters smaller than this (in grid cells).
    fmm_step_size : int
        ``step_size`` passed to :class:`FMMPlanner`.
    fmm_stop_dist_m : float
        Stop threshold (metres) for the local FMM planner.
    """

    def __init__(
        self,
        grid_size: int = 400,
        cell_size: float = 5.0,
        replan_dist: float = 10.0,
        min_frontier_size: int = 3,
        fmm_step_size: int = 5,
        fmm_stop_dist_m: float = 0.25,
    ):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.replan_dist = replan_dist
        self.min_frontier_size = min_frontier_size
        self.fmm_step_size = fmm_step_size
        self.fmm_stop_dist_m = fmm_stop_dist_m

        self.current_goal_grid = None  # (gx, gy)

    def needs_replan(self, agent_gx, agent_gy):
        """Return True if the agent should pick a new goal.

        Returns True when:
        - No current goal exists
        - Agent is close to the current goal (reached it, need next goal)

        Returns False when:
        - Agent is far from current goal (should keep moving toward it)
        """
        if self.current_goal_grid is None:
            return True
        dx = agent_gx - self.current_goal_grid[0]
        dy = agent_gy - self.current_goal_grid[1]
        dist_cells = np.sqrt(dx**2 + dy**2)
        dist_metres = dist_cells * self.cell_size
        return dist_metres < self.replan_dist

    def select_goal(self, semantic_map, agent_gx, agent_gy,
                    person_category_id=0):
        """Pick a long-term goal.

        Strategy (mirrors SemExp's "explore until detection" logic):
        1. If there are un-visited person detections on the semantic map,
           navigate to the nearest one.
        2. Otherwise, pick the largest frontier cluster.
        3. If no frontiers remain, return None (exploration complete).

        Parameters
        ----------
        semantic_map : SemanticMap
        agent_gx, agent_gy : int
            Current agent position in grid coords.
        person_category_id : int
            Index of the "person" category in the vocabulary.

        Returns
        -------
        goal : (int, int) or None
            Grid coordinates of the selected goal, or None if done.
        """
        # --- 1. Semantic-guided: go to detected person -----------------
        person_ch = semantic_map.get_semantic_channel(person_category_id)
        visited = semantic_map.map[semantic_map.CH_VISITED]

        # Person cells that haven't been closely visited
        unvisited_person = (person_ch > 0) & (visited == 0)
        if unvisited_person.any():
            # Find the nearest unvisited person detection
            pts = np.argwhere(unvisited_person)  # (N, 2)
            dists = np.linalg.norm(
                pts - np.array([agent_gx, agent_gy]), axis=1)
            nearest_idx = np.argmin(dists)
            goal = (int(pts[nearest_idx, 0]), int(pts[nearest_idx, 1]))
            self.current_goal_grid = goal
            dist_m = dists[nearest_idx] * self.cell_size
            print(f"[Frontier] -> Person goal {goal} (dist={dist_m:.0f}m)")
            return goal

        # --- 2. Frontier-based exploration -----------------------------
        goal = self._find_frontier_goal(semantic_map, agent_gx, agent_gy)
        if goal is not None:
            self.current_goal_grid = goal
            print(f"[Frontier] -> Frontier goal {goal}")
        else:
            print("[Frontier] No frontier found")
        return goal

    def _find_frontier_goal(self, semantic_map, agent_gx, agent_gy):
        """Find the best frontier to explore.

        Returns
        -------
        (gx, gy) or None
        """
        explored = semantic_map.get_explored_mask().astype(np.uint8)
        unknown = (explored == 0).astype(np.uint8)

        # Dilate explored region and subtract to get boundary
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(explored, kernel, iterations=1)
        edge = dilated - explored

        # Frontier = edge pixels that border unknown space
        frontier_mask = cv2.bitwise_and(edge, unknown)

        # Find connected components (frontier clusters)
        contours, _ = cv2.findContours(
            frontier_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        # Filter small clusters
        contours = [c for c in contours
                    if cv2.contourArea(c) >= self.min_frontier_size
                    or len(c) >= self.min_frontier_size]
        if not contours:
            # Relax: accept any cluster
            contours_all, _ = cv2.findContours(
                frontier_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours_all:
                return None
            contours = contours_all

        # Strategy: pick the LARGEST frontier cluster
        best_cnt = max(contours, key=lambda c: max(cv2.contourArea(c), len(c)))

        # Use the midpoint of the contour (avoids centroid-inside-circle issue)
        mid_idx = len(best_cnt) // 2
        # OpenCV contour points are (col, row) = (gy, gx)
        cy, cx = best_cnt[mid_idx][0]
        goal = (int(cx), int(cy))

        # Ensure goal is in-bounds
        goal = (
            np.clip(goal[0], 0, self.grid_size - 1),
            np.clip(goal[1], 0, self.grid_size - 1),
        )

        return goal

    def plan_short_term_goal(self, semantic_map, agent_gx, agent_gy,
                             global_goal_grid):
        """Use FMM planner to find a short-term waypoint toward the global goal.

        Parameters
        ----------
        semantic_map : SemanticMap
        agent_gx, agent_gy : int
        global_goal_grid : (int, int)

        Returns
        -------
        stg : (float, float)
            Grid coordinates of the short-term goal.
        stop : bool
            True if the agent has reached the global goal.
        replan : bool
            True if the planner recommends replanning.
        """
        traversible = semantic_map.get_traversible()

        # Ensure agent position is traversible
        r = 2
        gx_lo = max(0, agent_gx - r)
        gx_hi = min(self.grid_size, agent_gx + r + 1)
        gy_lo = max(0, agent_gy - r)
        gy_hi = min(self.grid_size, agent_gy + r + 1)
        traversible[gx_lo:gx_hi, gy_lo:gy_hi] = 1.0

        # Add boundary to avoid edge effects
        def add_boundary(mat, value=1.0):
            h, w = mat.shape
            new_mat = np.full((h + 2, w + 2), value)
            new_mat[1:h+1, 1:w+1] = mat
            return new_mat

        traversible_padded = add_boundary(traversible)

        planner = FMMPlanner(
            traversible_padded,
            step_size=self.fmm_step_size,
            cell_size=self.cell_size,
            stop_dist_m=self.fmm_stop_dist_m,
        )

        # Set goal (offset by +1 for boundary padding)
        goal_map = np.zeros_like(traversible_padded)
        gg = (np.clip(global_goal_grid[0] + 1, 0, goal_map.shape[0] - 1),
              np.clip(global_goal_grid[1] + 1, 0, goal_map.shape[1] - 1))

        # Dilate goal for robustness
        goal_map[gg[0], gg[1]] = 1
        selem = skimage.morphology.disk(3)
        goal_dilated = skimage.morphology.binary_dilation(
            goal_map.astype(bool), selem).astype(np.float64)
        planner.set_multi_goal(goal_dilated)

        # Get short-term goal (offset by +1 for boundary padding)
        state = [agent_gx + 1, agent_gy + 1]
        stg_x, stg_y, replan, stop = planner.get_short_term_goal(state)

        # Remove boundary offset
        stg_x -= 1
        stg_y -= 1

        # Clip to valid grid range to prevent out-of-bounds flight commands
        stg_x = np.clip(stg_x, 0, self.grid_size - 1)
        stg_y = np.clip(stg_y, 0, self.grid_size - 1)

        # Log when FMM signals trouble
        if replan or stop:
            dx = stg_x - agent_gx
            dy = stg_y - agent_gy
            stg_dist = np.sqrt(dx**2 + dy**2) * self.cell_size
            print(f"[FMM] replan={replan} stop={stop} "
                  f"stg=({stg_x:.0f},{stg_y:.0f}) "
                  f"stg_dist={stg_dist:.1f}m")

        return (stg_x, stg_y), stop, replan
