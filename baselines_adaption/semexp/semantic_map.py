"""
Multi-channel 2-D semantic map for the SemExp ASAR adaptation.

Inspired by the ``Semantic_Mapping`` module in the original SemExp paper, but
re-implemented with pure NumPy for the AirSim drone setting.

Channel layout
--------------
  0 : obstacle
  1 : explored
  2 : visited (agent trajectory)
  3 .. 3+N-1 : per-category semantic detections  (N = len(vocab))

Coordinate convention
---------------------
  Grid index (gx, gy) maps to world NED (x, y) via:
      gx = int((x - center_x) / cell_size + grid_size / 2)
      gy = int((y - center_y) / cell_size + grid_size / 2)
"""

import numpy as np
import cv2

from .utils.projection import batch_pixel_to_world


class SemanticMap:
    """Maintains a BEV (bird-eye-view) semantic occupancy grid.

    Parameters
    ----------
    grid_size : int
        Number of cells per side (default 400 -> 400x400).
    cell_size : float
        Metres per cell (default 5.0 -> covers 2 km with grid_size=400).
    num_categories : int
        Number of semantic object categories tracked in dedicated channels.
    center_x, center_y : float
        World NED coordinates of the grid centre (usually agent start position).
    camera_fov : float
        Horizontal field-of-view of the front camera in degrees.
    max_depth : float
        Maximum depth (metres) to trust from the depth sensor.
    sample_step : int
        Pixel stride when sampling the depth image for projection.
    obstacle_z_min : float
        DEPRECATED — kept for backward compat but no longer used.
    obstacle_z_margin : float
        Vertical band (metres) around the drone's z for obstacle detection.
        A depth point is an obstacle if ``|point_z - drone_z| < margin``.
        Default 10 m means objects within ±10 m of flight altitude are threats.
    """

    # Channel indices
    CH_OBSTACLE = 0
    CH_EXPLORED = 1
    CH_VISITED  = 2
    CH_SEM_START = 3  # semantic channels start here

    def __init__(
        self,
        grid_size: int = 400,
        cell_size: float = 5.0,
        num_categories: int = 11,
        center_x: float = 0.0,
        center_y: float = 0.0,
        camera_fov: float = 90.0,
        max_depth: float = 200.0,
        sample_step: int = 8,
        obstacle_z_min: float = -5.0,
        obstacle_z_margin: float = 10.0,
    ):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.num_categories = num_categories
        self.center_x = center_x
        self.center_y = center_y
        self.camera_fov = camera_fov
        self.max_depth = max_depth
        self.sample_step = sample_step
        self.obstacle_z_margin = obstacle_z_margin

        num_channels = self.CH_SEM_START + num_categories
        self.map = np.zeros((num_channels, grid_size, grid_size),
                            dtype=np.float32)

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def world_to_grid(self, x, y):
        """World NED (x, y) -> grid index (gx, gy)."""
        gx = int((x - self.center_x) / self.cell_size + self.grid_size / 2)
        gy = int((y - self.center_y) / self.cell_size + self.grid_size / 2)
        return gx, gy

    def grid_to_world(self, gx, gy):
        """Grid index (gx, gy) -> world NED (x, y)."""
        wx = (gx - self.grid_size / 2) * self.cell_size + self.center_x
        wy = (gy - self.grid_size / 2) * self.cell_size + self.center_y
        return wx, wy

    def _in_bounds(self, gx, gy):
        return 0 <= gx < self.grid_size and 0 <= gy < self.grid_size

    # ------------------------------------------------------------------
    # Map update
    # ------------------------------------------------------------------

    def update_agent_position(self, x, y, radius_cells: int = 2):
        """Mark the agent's current location as explored + visited."""
        gx, gy = self.world_to_grid(x, y)
        if not self._in_bounds(gx, gy):
            return
        # Mark a small disk around the agent
        cv2.circle(self.map[self.CH_EXPLORED], (gy, gx), radius_cells, 1, -1)
        cv2.circle(self.map[self.CH_VISITED],  (gy, gx), 1, 1, -1)

    def mark_visited(self, gx, gy, radius_cells: int = 3):
        """Mark a grid cell as visited (e.g. a reached goal).

        Used to prevent the goal selector from repeatedly choosing a target
        that has already been approached.
        """
        if not self._in_bounds(gx, gy):
            return
        cv2.circle(self.map[self.CH_VISITED], (gy, gx), radius_cells, 1, -1)

    def update_from_depth(self, depth_img, camera_position, camera_orientation,
                          camera_yaw_offset=0.0):
        """Project sampled depth pixels into the BEV grid.

        Marks projected cells as *explored*.  Points whose world-z is within
        a vertical band around the drone's flight altitude are marked as
        *obstacle* (i.e. objects at the same height that could collide).
        Ground-level points far below the drone are NOT obstacles.

        Parameters
        ----------
        depth_img : ndarray (H, W), float, metres
        camera_position : airsim.Vector3r
        camera_orientation : airsim.Quaternionr
        camera_yaw_offset : float
            Additional yaw of this camera relative to vehicle body (degrees).
        """
        h, w = depth_img.shape
        step = self.sample_step

        # Build sampling grid
        vs, us = np.mgrid[0:h:step, 0:w:step]
        vs = vs.ravel().astype(np.float64)
        us = us.ravel().astype(np.float64)
        ds = depth_img[vs.astype(int), us.astype(int)].astype(np.float64)

        # Filter invalid depth
        valid = (ds > 0.5) & (ds < self.max_depth)
        if valid.sum() == 0:
            return
        vs, us, ds = vs[valid], us[valid], ds[valid]

        world_xyz = batch_pixel_to_world(
            vs, us, ds, (h, w), self.camera_fov,
            camera_position, camera_orientation,
            camera_yaw_offset=camera_yaw_offset,
        )
        wx = world_xyz[:, 0]
        wy = world_xyz[:, 1]
        wz = world_xyz[:, 2]

        gxs = ((wx - self.center_x) / self.cell_size
               + self.grid_size / 2).astype(int)
        gys = ((wy - self.center_y) / self.cell_size
               + self.grid_size / 2).astype(int)

        # Keep only in-bounds points
        in_bounds = ((gxs >= 0) & (gxs < self.grid_size) &
                     (gys >= 0) & (gys < self.grid_size))
        gxs, gys, wz = gxs[in_bounds], gys[in_bounds], wz[in_bounds]

        # Mark explored
        self.map[self.CH_EXPLORED, gxs, gys] = 1.0

        # Mark obstacle: only points near the drone's flight altitude.
        # In NED, z is negative-upward. A point is a collision threat if
        # its z is within OBSTACLE_Z_MARGIN of the drone's current z.
        # Ground (z≈0) when drone is at z=-30 is 30m below → NOT obstacle.
        drone_z = camera_position.z_val
        obstacle_mask = (np.abs(wz - drone_z) < self.obstacle_z_margin)
        if obstacle_mask.any():
            self.map[self.CH_OBSTACLE, gxs[obstacle_mask],
                     gys[obstacle_mask]] = 1.0

    def update_semantic(self, category_id, world_x, world_y,
                        radius_cells: int = 1):
        """Mark a detected object on the corresponding semantic channel.

        Parameters
        ----------
        category_id : int
            Index into the vocabulary (0-based).
        world_x, world_y : float
            Object position in NED metres.
        radius_cells : int
            Size of the mark on the grid.
        """
        ch = self.CH_SEM_START + category_id
        if ch >= self.map.shape[0]:
            return
        gx, gy = self.world_to_grid(world_x, world_y)
        if not self._in_bounds(gx, gy):
            return
        cv2.circle(self.map[ch], (gy, gx), radius_cells, 1, -1)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_explored_mask(self):
        """Return boolean mask (grid_size, grid_size) of explored cells."""
        return self.map[self.CH_EXPLORED] > 0

    def get_obstacle_mask(self):
        return self.map[self.CH_OBSTACLE] > 0

    def get_traversible(self):
        """Return a float traversibility map suitable for FMMPlanner.

        1.0 = free / traversible, 0.0 = obstacle or unknown.
        Currently: explored AND not obstacle.
        For the drone case we are more permissive: unexplored cells are
        also treated as traversible (we assume open sky) unless they are
        explicitly marked as obstacles.
        """
        obstacle = self.map[self.CH_OBSTACLE] > 0
        traversible = (~obstacle).astype(np.float64)
        return traversible

    def get_semantic_channel(self, category_id):
        ch = self.CH_SEM_START + category_id
        if ch >= self.map.shape[0]:
            return np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        return self.map[ch]

    def has_semantic_detection(self, category_id):
        return self.get_semantic_channel(category_id).sum() > 0
