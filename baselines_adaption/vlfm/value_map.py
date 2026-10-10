"""
Simplified ValueMap for VLFM ASAR adaptation.

Maintains a BEV grid of BLIP2-ITM confidence scores.  Each ``update()``
projects the front-camera depth footprint onto the grid and writes the
current ITM score into those cells (max-fusion).

Frontier scoring queries the accumulated value around each frontier point.
"""

import numpy as np

from .utils.projection import batch_pixel_to_world


class ValueMap:
    """Bird-eye-view map storing ITM confidence values.

    Parameters
    ----------
    grid_size : int
        Cells per side (must match SemanticMap).
    cell_size : float
        Metres per cell (must match SemanticMap).
    center_x, center_y : float
        World NED coordinates of the grid centre.
    camera_fov : float
        Horizontal FOV of the front camera (degrees).
    max_depth : float
        Maximum trusted depth (metres).
    sample_step : int
        Pixel stride when sampling the depth image for projection.
    """

    def __init__(
        self,
        grid_size=400,
        cell_size=5.0,
        center_x=0.0,
        center_y=0.0,
        camera_fov=90.0,
        max_depth=500.0,
        sample_step=8,
    ):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.center_x = center_x
        self.center_y = center_y
        self.camera_fov = camera_fov
        self.max_depth = max_depth
        self.sample_step = sample_step

        self._map = np.zeros((grid_size, grid_size), dtype=np.float32)

    # ------------------------------------------------------------------
    # Coordinate helpers (same convention as SemanticMap)
    # ------------------------------------------------------------------

    def world_to_grid(self, wx, wy):
        gx = int((wx - self.center_x) / self.cell_size + self.grid_size / 2)
        gy = int((wy - self.center_y) / self.cell_size + self.grid_size / 2)
        return gx, gy

    def _in_bounds(self, gx, gy):
        return 0 <= gx < self.grid_size and 0 <= gy < self.grid_size

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, itm_score, depth_image, camera_position, camera_orientation,
               camera_yaw_offset=0.0):
        """Project an ITM score onto the BEV grid via the front-camera depth.

        Parameters
        ----------
        itm_score : float
            BLIP2-ITM cosine similarity for the current front-camera RGB.
        depth_image : ndarray (H, W), float
            Front-camera depth in metres (0 = invalid / sky).
        camera_position : airsim.Vector3r
        camera_orientation : airsim.Quaternionr
        camera_yaw_offset : float
            Yaw offset of this camera relative to body (degrees).
        """
        if itm_score <= 0:
            return

        h, w = depth_image.shape
        vs, us = np.mgrid[0:h:self.sample_step, 0:w:self.sample_step]
        vs = vs.ravel()
        us = us.ravel()
        depths = depth_image[vs, us]

        valid = (depths > 0.5) & (depths < self.max_depth)
        if valid.sum() == 0:
            return

        vs = vs[valid].astype(np.float64)
        us = us[valid].astype(np.float64)
        depths = depths[valid].astype(np.float64)

        world_xyz = batch_pixel_to_world(
            vs, us, depths,
            image_shape=(h, w),
            camera_fov=self.camera_fov,
            camera_position=camera_position,
            camera_orientation=camera_orientation,
            camera_yaw_offset=camera_yaw_offset,
        )

        for i in range(world_xyz.shape[0]):
            gx, gy = self.world_to_grid(world_xyz[i, 0], world_xyz[i, 1])
            if self._in_bounds(gx, gy):
                self._map[gx, gy] = max(self._map[gx, gy], itm_score)

    # ------------------------------------------------------------------
    # Frontier scoring
    # ------------------------------------------------------------------

    def score_frontiers(self, frontier_points, radius_cells=2):
        """Score a list of frontier grid points by nearby accumulated value.

        Parameters
        ----------
        frontier_points : list[(gx, gy)] or ndarray (N, 2)
        radius_cells : int
            Neighbourhood radius for averaging.

        Returns
        -------
        scores : ndarray (N,)
        """
        scores = np.zeros(len(frontier_points), dtype=np.float32)
        for i, (gx, gy) in enumerate(frontier_points):
            gx, gy = int(gx), int(gy)
            r = radius_cells
            x0 = max(0, gx - r)
            x1 = min(self.grid_size, gx + r + 1)
            y0 = max(0, gy - r)
            y1 = min(self.grid_size, gy + r + 1)
            patch = self._map[x0:x1, y0:y1]
            if patch.size > 0:
                scores[i] = patch.mean()
        return scores

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def reset(self):
        self._map[:] = 0.0

    def get_map(self):
        return self._map.copy()
