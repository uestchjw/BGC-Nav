"""Graph-guided frontier selection for UniGoal ASAR adaptation.

Extends SemExp's frontier-based exploration with graph-informed scoring:
detected goal-relevant objects bias frontier selection toward areas where
more targets are likely to be found (SAR object co-occurrence).
"""

import numpy as np
import cv2
import skimage.morphology
from typing import List, Optional, Tuple

from .utils.fmm_planner import FMMPlanner
from .graph_matcher import OverlapResult


# --- Scoring weights (tunable) ---
SIZE_WEIGHT = 1.0        # Larger frontier clusters score higher
DISTANCE_WEIGHT = 0.3    # Mild preference for closer frontiers
CLUSTER_WEIGHT = 3.0     # Strong pull toward areas near found goal objects
DIVERSITY_WEIGHT = 2.0   # When coverage is low, prefer truly unexplored areas
HINT_WEIGHT = 2.5        # Bonus for frontiers near VLM-matched spatial hints

# Cluster radius: detected SAR objects may cluster within ~200m
CLUSTER_RADIUS_M = 200.0


class GraphGuidedFrontierSelector:
    """Frontier selector that uses graph overlap to bias exploration.

    The person-first priority from SemExp is preserved. On top of the
    basic frontier selection, graph overlap information adds scoring
    bonuses to frontiers near detected goal-relevant objects.

    Parameters
    ----------
    grid_size : int
    cell_size : float
    replan_dist : float
    min_frontier_size : int
    fmm_step_size : int
    fmm_stop_dist_m : float
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

        self.current_goal_grid = None
        self.cluster_radius_cells = CLUSTER_RADIUS_M / cell_size

        # VLM spatial hint matches: list of (grid_x, grid_y) positions
        # where the VLM confirmed a spatial hint from the goal graph
        self.hint_positions: List[Tuple[int, int]] = []

    def add_hint_position(self, grid_x: int, grid_y: int):
        """Record a position where a VLM spatial hint was matched."""
        self.hint_positions.append((grid_x, grid_y))

    def needs_replan(self, agent_gx: int, agent_gy: int) -> bool:
        """Return True if the agent should pick a new goal."""
        if self.current_goal_grid is None:
            return True
        dx = agent_gx - self.current_goal_grid[0]
        dy = agent_gy - self.current_goal_grid[1]
        dist_cells = np.sqrt(dx ** 2 + dy ** 2)
        dist_metres = dist_cells * self.cell_size
        return dist_metres < self.replan_dist

    def select_goal(
        self,
        semantic_map,
        agent_gx: int,
        agent_gy: int,
        person_category_id: int = 0,
        overlap_result: Optional[OverlapResult] = None,
    ):
        """Pick a long-term goal using graph-guided frontier scoring.

        Strategy:
        1. Person priority: if unvisited person detected, go there.
        2. Graph-guided frontier: score frontiers using size, distance,
           proximity to found goal objects, and spatial hint matches.
        3. Fallback: largest frontier (same as SemExp).

        Returns
        -------
        goal : (int, int) or None
        """
        # --- 1. Person priority (identical to SemExp) ---
        person_ch = semantic_map.get_semantic_channel(person_category_id)
        visited = semantic_map.map[semantic_map.CH_VISITED]

        unvisited_person = (person_ch > 0) & (visited == 0)
        if unvisited_person.any():
            pts = np.argwhere(unvisited_person)
            dists = np.linalg.norm(
                pts - np.array([agent_gx, agent_gy]), axis=1)
            nearest_idx = np.argmin(dists)
            goal = (int(pts[nearest_idx, 0]), int(pts[nearest_idx, 1]))
            self.current_goal_grid = goal
            dist_m = dists[nearest_idx] * self.cell_size
            print(f"[UniGoal-Frontier] -> Person goal {goal} "
                  f"(dist={dist_m:.0f}m)")
            return goal

        # --- 2. Find frontier clusters ---
        frontiers = self._find_frontier_clusters(semantic_map)
        if not frontiers:
            print("[UniGoal-Frontier] No frontiers found")
            return None

        # --- 3. Score frontiers with graph guidance ---
        if overlap_result is not None and overlap_result.coverage > 0:
            goal = self._graph_guided_select(
                frontiers, semantic_map, agent_gx, agent_gy, overlap_result
            )
        else:
            # No overlap info yet — fall back to largest frontier
            goal = self._largest_frontier_select(frontiers)

        if goal is not None:
            self.current_goal_grid = goal
            print(f"[UniGoal-Frontier] -> Graph-guided goal {goal}")
        return goal

    def _graph_guided_select(
        self,
        frontiers: List[Tuple[Tuple[int, int], int]],
        semantic_map,
        agent_gx: int,
        agent_gy: int,
        overlap_result: OverlapResult,
    ) -> Optional[Tuple[int, int]]:
        """Score frontiers using graph overlap information.

        Parameters
        ----------
        frontiers : list of ((gx, gy), size)
            Frontier centroids and their sizes.
        """
        scores = np.zeros(len(frontiers))
        max_size = max(f[1] for f in frontiers) or 1

        for i, (centroid, size) in enumerate(frontiers):
            fx, fy = centroid

            # Base: normalized frontier size
            scores[i] += (size / max_size) * SIZE_WEIGHT

            # Distance penalty (Euclidean in grid, rough but fast)
            dist = np.sqrt((fx - agent_gx) ** 2 + (fy - agent_gy) ** 2)
            dist_m = dist * self.cell_size
            # Normalize: 0 at 0m, 1.0 at 1000m
            scores[i] -= min(dist_m / 1000.0, 1.0) * DISTANCE_WEIGHT

            # Cluster bonus: proximity to found goal objects
            for label, pos in overlap_result.found_positions.items():
                if pos is None:
                    continue
                obj_gx, obj_gy = semantic_map.world_to_grid(pos[0], pos[1])
                obj_dist = np.sqrt((fx - obj_gx) ** 2 + (fy - obj_gy) ** 2)
                if obj_dist < self.cluster_radius_cells:
                    proximity = 1.0 - obj_dist / self.cluster_radius_cells
                    scores[i] += proximity * CLUSTER_WEIGHT

            # Spatial hint bonus: proximity to VLM-matched hint positions
            for hx, hy in self.hint_positions:
                hint_dist = np.sqrt((fx - hx) ** 2 + (fy - hy) ** 2)
                if hint_dist < self.cluster_radius_cells:
                    proximity = 1.0 - hint_dist / self.cluster_radius_cells
                    scores[i] += proximity * HINT_WEIGHT

            # Diversity bonus: when coverage is low, prefer unexplored dirs
            if overlap_result.coverage < 0.3:
                visited = semantic_map.map[semantic_map.CH_VISITED]
                x_lo = max(0, fx - 20)
                x_hi = min(self.grid_size, fx + 20)
                y_lo = max(0, fy - 20)
                y_hi = min(self.grid_size, fy + 20)
                local_visited = visited[x_lo:x_hi, y_lo:y_hi].sum()
                if local_visited < 5:
                    scores[i] += DIVERSITY_WEIGHT

        best_idx = int(np.argmax(scores))
        return frontiers[best_idx][0]

    def _largest_frontier_select(
        self,
        frontiers: List[Tuple[Tuple[int, int], int]],
    ) -> Optional[Tuple[int, int]]:
        """Fallback: pick the largest frontier cluster."""
        if not frontiers:
            return None
        best = max(frontiers, key=lambda f: f[1])
        return best[0]

    def _find_frontier_clusters(
        self, semantic_map
    ) -> List[Tuple[Tuple[int, int], int]]:
        """Find frontier clusters and their centroids.

        Returns
        -------
        list of ((gx, gy), size)
            Each element is (centroid_grid_coords, cluster_size_cells).
        """
        explored = semantic_map.get_explored_mask().astype(np.uint8)
        unknown = (explored == 0).astype(np.uint8)

        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(explored, kernel, iterations=1)
        edge = dilated - explored
        frontier_mask = cv2.bitwise_and(edge, unknown)

        contours, _ = cv2.findContours(
            frontier_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return []

        frontiers = []
        for cnt in contours:
            size = max(cv2.contourArea(cnt), len(cnt))
            if size < self.min_frontier_size:
                continue
            # Centroid via midpoint of contour
            mid_idx = len(cnt) // 2
            cy, cx = cnt[mid_idx][0]  # OpenCV: (col, row)
            gx = int(np.clip(cx, 0, self.grid_size - 1))
            gy = int(np.clip(cy, 0, self.grid_size - 1))
            frontiers.append(((gx, gy), int(size)))

        # If all clusters were too small, accept any
        if not frontiers and contours:
            for cnt in contours:
                size = max(cv2.contourArea(cnt), len(cnt))
                mid_idx = len(cnt) // 2
                cy, cx = cnt[mid_idx][0]
                gx = int(np.clip(cx, 0, self.grid_size - 1))
                gy = int(np.clip(cy, 0, self.grid_size - 1))
                frontiers.append(((gx, gy), max(int(size), 1)))

        return frontiers

    def plan_short_term_goal(
        self, semantic_map, agent_gx, agent_gy, global_goal_grid
    ):
        """Use FMM planner to find a short-term waypoint.

        Identical to SemExp's FrontierGoalSelector.plan_short_term_goal.
        """
        traversible = semantic_map.get_traversible()

        r = 2
        gx_lo = max(0, agent_gx - r)
        gx_hi = min(self.grid_size, agent_gx + r + 1)
        gy_lo = max(0, agent_gy - r)
        gy_hi = min(self.grid_size, agent_gy + r + 1)
        traversible[gx_lo:gx_hi, gy_lo:gy_hi] = 1.0

        def add_boundary(mat, value=1.0):
            h, w = mat.shape
            new_mat = np.full((h + 2, w + 2), value)
            new_mat[1:h + 1, 1:w + 1] = mat
            return new_mat

        traversible_padded = add_boundary(traversible)

        planner = FMMPlanner(
            traversible_padded,
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
            goal_map.astype(bool), selem
        ).astype(np.float64)
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
            stg_dist = np.sqrt(dx ** 2 + dy ** 2) * self.cell_size
            print(f"[UniGoal-FMM] replan={replan} stop={stop} "
                  f"stg=({stg_x:.0f},{stg_y:.0f}) "
                  f"stg_dist={stg_dist:.1f}m")

        return (stg_x, stg_y), stop, replan
