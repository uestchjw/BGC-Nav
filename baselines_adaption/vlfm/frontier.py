"""
Frontier-based goal selection with value-map scoring for VLFM ASAR adaptation.

Combines semexp-style frontier extraction (explored/obstacle boundary) with
VLFM-style value-based ranking (BLIP2-ITM scores from the ValueMap).
Includes an acyclic enforcer to prevent oscillation between frontiers.
"""

import numpy as np
import cv2
import skimage.morphology

from .utils.fmm_planner import FMMPlanner


class VLFMFrontierSelector:
    """Select navigation goals using value-scored frontiers.

    Parameters
    ----------
    grid_size, cell_size : int, float
        Must match SemanticMap / ValueMap.
    replan_dist : float
        Re-plan when within this many metres of the current goal.
    min_frontier_size : int
        Ignore frontier clusters smaller than this (grid cells).
    fmm_step_size : int
        ``step_size`` for FMMPlanner.
    fmm_stop_dist_m : float
        Stop threshold (metres) for FMMPlanner.
    acyclic_radius : float
        Grid-cell distance within which a frontier is considered "same as"
        a previously selected one (for suppression).
    acyclic_history : int
        Number of recent frontier selections to remember.
    value_score_radius : int
        Neighbourhood radius (cells) for querying ValueMap around a frontier.
    """

    def __init__(
        self,
        grid_size=400,
        cell_size=5.0,
        replan_dist=10.0,
        min_frontier_size=3,
        fmm_step_size=5,
        fmm_stop_dist_m=0.25,
        acyclic_radius=3.0,
        acyclic_history=8,
        value_score_radius=2,
    ):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.replan_dist = replan_dist
        self.min_frontier_size = min_frontier_size
        self.fmm_step_size = fmm_step_size
        self.fmm_stop_dist_m = fmm_stop_dist_m
        self.acyclic_radius = acyclic_radius
        self.acyclic_history = acyclic_history
        self.value_score_radius = value_score_radius

        self.current_goal_grid = None
        self._history = []  # list of (gx, gy)

    # ------------------------------------------------------------------
    # Replan check
    # ------------------------------------------------------------------

    def needs_replan(self, agent_gx, agent_gy):
        if self.current_goal_grid is None:
            return True
        dx = agent_gx - self.current_goal_grid[0]
        dy = agent_gy - self.current_goal_grid[1]
        dist_m = np.sqrt(dx**2 + dy**2) * self.cell_size
        return dist_m < self.replan_dist

    # ------------------------------------------------------------------
    # Goal selection
    # ------------------------------------------------------------------

    def select_global_goal(self, semantic_map, value_map, agent_gx, agent_gy,
                           person_category_id=0):
        """Pick a long-term navigation goal (grid coordinates only).

        Priority:
        1. Un-visited person detection on semantic map → go there.
        2. Otherwise: extract frontiers → score by ValueMap → pick best
           (with acyclic suppression).

        Returns
        -------
        goal : (gx, gy) or None
        """
        # --- Priority 1: semantic target ---
        sem_goal = self._find_semantic_target(
            semantic_map, agent_gx, agent_gy, person_category_id)
        if sem_goal is not None:
            print(f"[VLFM-Frontier] Navigating to semantic target at "
                  f"grid ({sem_goal[0]}, {sem_goal[1]})")
            self.current_goal_grid = sem_goal
            return sem_goal

        # --- Priority 2: value-scored frontiers ---
        frontiers = self._extract_frontiers(
            semantic_map.get_explored_mask(),
            semantic_map.get_obstacle_mask(),
        )
        if len(frontiers) == 0:
            print("[VLFM-Frontier] No frontiers found.")
            return None

        scores = value_map.score_frontiers(
            frontiers, radius_cells=self.value_score_radius)

        # Sort descending by score
        order = np.argsort(-scores)
        frontiers_sorted = [frontiers[i] for i in order]

        # Pick best non-suppressed frontier
        chosen = None
        for f in frontiers_sorted:
            if not self._is_suppressed(f):
                chosen = f
                break
        if chosen is None:
            # All suppressed — clear history and pick the top one
            self._history.clear()
            chosen = frontiers_sorted[0]

        self.current_goal_grid = chosen
        self._record_history(chosen)
        best_score = scores[order[0]]
        print(f"[VLFM-Frontier] Selected frontier ({chosen[0]}, {chosen[1]}) "
              f"value={best_score:.4f}  candidates={len(frontiers)}")

        return chosen

    # ------------------------------------------------------------------
    # Frontier extraction (from semexp)
    # ------------------------------------------------------------------

    def _extract_frontiers(self, explored_mask, obstacle_mask):
        """Extract frontier centroids from explored/obstacle masks.

        A frontier cell is on the boundary of explored free-space and
        unexplored space.
        """
        explored = explored_mask.astype(np.uint8)
        obstacle = obstacle_mask.astype(np.uint8)
        free = ((explored > 0) & (obstacle == 0)).astype(np.uint8)

        kernel = np.ones((3, 3), dtype=np.uint8)
        dilated = cv2.dilate(free, kernel, iterations=1)
        frontier_mask = ((dilated - free) > 0) & (explored == 0) & (obstacle == 0)
        frontier_mask = frontier_mask.astype(np.uint8)

        num_labels, labels = cv2.connectedComponents(frontier_mask)
        centroids = []
        for label_id in range(1, num_labels):
            ys, xs = np.where(labels == label_id)
            if len(ys) < self.min_frontier_size:
                continue
            cx = int(np.mean(xs))
            cy = int(np.mean(ys))
            centroids.append((cy, cx))  # (gx, gy) = (row, col)
        return centroids

    # ------------------------------------------------------------------
    # Semantic target (same as semexp)
    # ------------------------------------------------------------------

    def _find_semantic_target(self, semantic_map, agent_gx, agent_gy,
                              person_category_id):
        if not semantic_map.has_semantic_detection(person_category_id):
            return None
        sem_ch = semantic_map.get_semantic_channel(person_category_id)
        visited = semantic_map.map[semantic_map.CH_VISITED]
        unvisited_sem = (sem_ch > 0) & (visited == 0)
        if not unvisited_sem.any():
            return None
        ys, xs = np.where(unvisited_sem)
        dists = (ys - agent_gx)**2 + (xs - agent_gy)**2
        nearest = np.argmin(dists)
        return (int(ys[nearest]), int(xs[nearest]))

    # ------------------------------------------------------------------
    # Acyclic enforcer
    # ------------------------------------------------------------------

    def _is_suppressed(self, frontier):
        for hf in self._history:
            d = np.sqrt((frontier[0] - hf[0])**2 + (frontier[1] - hf[1])**2)
            if d < self.acyclic_radius:
                return True
        return False

    def _record_history(self, frontier):
        self._history.append(frontier)
        if len(self._history) > self.acyclic_history:
            self._history.pop(0)

    # ------------------------------------------------------------------
    # FMM short-term planning (from semexp)
    # ------------------------------------------------------------------

    def plan_short_term(self, semantic_map, agent_gx, agent_gy,
                        global_goal_grid):
        """Run FMM local planner to get a short-term waypoint.

        Called every step (not just on replan).
        """
        traversible = semantic_map.get_traversible()

        # Ensure agent position is traversible (unlock neighbourhood)
        r = 2
        gx_lo = max(0, agent_gx - r)
        gx_hi = min(self.grid_size, agent_gx + r + 1)
        gy_lo = max(0, agent_gy - r)
        gy_hi = min(self.grid_size, agent_gy + r + 1)
        traversible[gx_lo:gx_hi, gy_lo:gy_hi] = 1.0

        # Add traversible boundary to avoid edge effects (same as semexp)
        def add_boundary(mat, value=1.0):
            h, w = mat.shape
            new_mat = np.full((h + 2, w + 2), value)
            new_mat[1:h+1, 1:w+1] = mat
            return new_mat

        traversible_padded = add_boundary(traversible)

        planner = FMMPlanner(
            traversible_padded,
            scale=1,
            step_size=self.fmm_step_size,
            cell_size=self.cell_size,
            stop_dist_m=self.fmm_stop_dist_m,
        )

        goal_map = np.zeros_like(traversible_padded)
        gg = (np.clip(global_goal_grid[0] + 1, 0, goal_map.shape[0] - 1),
              np.clip(global_goal_grid[1] + 1, 0, goal_map.shape[1] - 1))
        goal_map[gg[0], gg[1]] = 1
        selem = skimage.morphology.disk(3)
        goal_dilated = skimage.morphology.binary_dilation(
            goal_map.astype(bool), selem).astype(np.float64)
        planner.set_multi_goal(goal_dilated)

        state = [agent_gx + 1, agent_gy + 1]
        stg_x, stg_y, replan, stop = planner.get_short_term_goal(state)
        stg_x -= 1
        stg_y -= 1
        stg_x = np.clip(stg_x, 0, self.grid_size - 1)
        stg_y = np.clip(stg_y, 0, self.grid_size - 1)

        if replan or stop:
            dx = stg_x - agent_gx
            dy = stg_y - agent_gy
            stg_dist = np.sqrt(dx**2 + dy**2) * self.cell_size
            print(f"[VLFM-FMM] replan={replan} stop={stop} "
                  f"stg=({stg_x:.0f},{stg_y:.0f}) stg_dist={stg_dist:.1f}m")

        return (stg_x, stg_y), stop, replan
