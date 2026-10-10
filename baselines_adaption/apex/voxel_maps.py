"""
3D voxel map structures for the APEX ASAR adaptation.

Manages three ego-centric 3D grids:
  - attraction_map  (40, 40, 10, 2): VLM-scored object relevance + depth weight
  - exploration_map (40, 40, 10):    exploration coverage
  - obstacle_map    (40, 40, 10):    obstacle occupancy

The maps slide with the drone so it stays near grid center (20, 20, 5).
"""

import numpy as np

from .utils.projection import batch_pixel_to_world

GRID_SCALE = 5.0
MAP_SHAPE = (40, 40, 10)
MAP_SIZE_WORLD = np.array([s * GRID_SCALE for s in MAP_SHAPE])  # 200, 200, 50
DRONE_CENTER_GRID = np.array([20, 20, 5])
SHIFT_THRESHOLD = 2  # shift when drone is >2 cells from center

# Exploration update parameters (from original map_update_simple)
VIEW_DEPTH = 100.0
VIEW_HEIGHT = 30.0
EXPLORATION_GAIN = 0.3


def create_maps():
    """Create fresh map arrays."""
    attraction = np.zeros((*MAP_SHAPE, 2), dtype=np.float32)
    attraction[..., 1] = -1.0
    exploration = np.zeros(MAP_SHAPE, dtype=np.float32)
    obstacle = np.zeros(MAP_SHAPE, dtype=np.float32)
    return attraction, exploration, obstacle


def world_to_grid(world_xyz, map_center_world):
    """Convert world NED coordinates to grid indices."""
    map_origin = map_center_world - MAP_SIZE_WORLD / 2.0
    grid = ((world_xyz - map_origin) / GRID_SCALE).astype(int)
    return grid


def grid_in_bounds(gx, gy, gz):
    return 0 <= gx < MAP_SHAPE[0] and 0 <= gy < MAP_SHAPE[1] and 0 <= gz < MAP_SHAPE[2]


def shift_maps(attraction, exploration, obstacle, old_center, new_drone_world):
    """Shift maps so the drone stays near grid center. Returns updated maps + new center."""
    delta = new_drone_world - old_center
    shift_cells = np.round(delta / GRID_SCALE).astype(int)

    if np.all(np.abs(shift_cells) <= SHIFT_THRESHOLD):
        return attraction, exploration, obstacle, old_center

    for axis in range(3):
        s = shift_cells[axis]
        if s == 0:
            continue
        attraction = np.roll(attraction, -s, axis=axis)
        exploration = np.roll(exploration, -s, axis=axis)
        obstacle = np.roll(obstacle, -s, axis=axis)

        # Zero-fill newly exposed region
        slc = [slice(None)] * attraction.ndim
        if s > 0:
            slc_3d = [slice(None)] * 3
            slc_3d[axis] = slice(MAP_SHAPE[axis] - s, None)
            exploration[tuple(slc_3d)] = 0.0
            obstacle[tuple(slc_3d)] = 0.0
            slc[axis] = slice(MAP_SHAPE[axis] - s, None)
            attraction[tuple(slc)] = 0.0
            # Reset depth weight channel
            slc_w = [slice(None)] * 3 + [1]
            slc_w[axis] = slice(MAP_SHAPE[axis] - s, None)
            attraction[tuple(slc_w)] = -1.0
        else:
            slc_3d = [slice(None)] * 3
            slc_3d[axis] = slice(None, -s)
            exploration[tuple(slc_3d)] = 0.0
            obstacle[tuple(slc_3d)] = 0.0
            slc[axis] = slice(None, -s)
            attraction[tuple(slc)] = 0.0
            slc_w = [slice(None)] * 3 + [1]
            slc_w[axis] = slice(None, -s)
            attraction[tuple(slc_w)] = -1.0

    new_center = old_center + shift_cells * GRID_SCALE
    return attraction, exploration, obstacle, new_center


def update_obstacle_from_depth(obstacle, depth_image, camera_position, camera_orientation,
                               camera_fov, map_center_world, camera_yaw_offset=0.0,
                               sample_step=4, max_depth=250.0):
    """Update obstacle map from a single depth image (vectorized)."""
    h, w = depth_image.shape
    vs, us = np.mgrid[0:h:sample_step, 0:w:sample_step]
    vs_flat = vs.ravel()
    us_flat = us.ravel()
    depths_flat = depth_image[vs_flat, us_flat]

    valid = (depths_flat > 0.5) & (depths_flat < max_depth)
    if not np.any(valid):
        return obstacle

    world_xyz = batch_pixel_to_world(
        vs_flat[valid], us_flat[valid], depths_flat[valid],
        (h, w), camera_fov, camera_position, camera_orientation,
        camera_yaw_offset=camera_yaw_offset,
    )

    grid_indices = world_to_grid(world_xyz, map_center_world)
    gx, gy, gz = grid_indices[:, 0], grid_indices[:, 1], grid_indices[:, 2]

    in_bounds = (
        (gx >= 0) & (gx < MAP_SHAPE[0])
        & (gy >= 0) & (gy < MAP_SHAPE[1])
        & (gz >= 0) & (gz < MAP_SHAPE[2])
    )
    gx, gy, gz = gx[in_bounds], gy[in_bounds], gz[in_bounds]
    obstacle[gx, gy, gz] = 1.0
    return obstacle


def update_exploration(exploration, drone_world_pos, drone_ori, map_center_world):
    """Update exploration map using view-frustum sweep (from original map_update_simple)."""
    map_origin = map_center_world - MAP_SIZE_WORLD / 2.0
    grid_size_arr = np.array([GRID_SCALE, GRID_SCALE, GRID_SCALE])

    ix, iy, iz = np.indices(MAP_SHAPE)
    all_indices = np.stack([ix.ravel(), iy.ravel(), iz.ravel()], axis=-1)
    all_centers_world = (all_indices + 0.5) * grid_size_arr + map_origin

    relative_coords = all_centers_world - drone_world_pos
    rel_x, rel_y, rel_z = relative_coords.T

    ori_conds = [drone_ori == i for i in range(4)]
    local_x = np.select(ori_conds, [-rel_y, -rel_x, rel_y, rel_x])
    local_y = np.select(ori_conds, [rel_x, -rel_y, -rel_x, rel_y])

    in_frustum = (
        (np.abs(rel_z) <= VIEW_HEIGHT / 2)
        & (local_y > 0)
        & (local_y <= VIEW_DEPTH)
        & (np.abs(local_x) <= local_y)
    )

    traversed = all_indices[in_frustum]
    if traversed.shape[0] > 0:
        gx, gy, gz = traversed.T
        distances = np.linalg.norm(relative_coords[in_frustum], axis=1)
        distances = np.clip(distances, 0, VIEW_DEPTH)
        increase = (1 - distances / VIEW_DEPTH) * EXPLORATION_GAIN
        exploration[gx, gy, gz] += increase

    return exploration


def update_attraction(attraction, detections, map_center_world):
    """Update attraction map from YOLO-World detections with VLM scores.

    Parameters
    ----------
    attraction : ndarray (40, 40, 10, 2)
    detections : list of dict, each with keys:
        "world_pos": np.array([x, y, z])
        "score": float (VLM relevance score)
    map_center_world : np.array([x, y, z])
    """
    for det in detections:
        grid = world_to_grid(det["world_pos"], map_center_world)
        gx, gy, gz = int(grid[0]), int(grid[1]), int(grid[2])
        if not grid_in_bounds(gx, gy, gz):
            continue
        score = det["score"]
        if score > attraction[gx, gy, gz, 0]:
            attraction[gx, gy, gz, 0] = score
            attraction[gx, gy, gz, 1] = 1.0
    return attraction


def crop_rotate_maps(attraction, exploration, obstacle, drone_grid_pos, drone_ori):
    """Crop and rotate maps to ego-centric local view for action_predict.

    Since maps are already ego-centric (drone near center), we just pass them
    through with rotation applied for the current orientation.
    """
    attr_scores = attraction[:, :, :, 0]

    def _rotate(m, ori):
        return np.rot90(m, k=ori, axes=(0, 1))

    return (
        _rotate(attr_scores, drone_ori),
        _rotate(exploration, drone_ori),
        _rotate(obstacle, drone_ori),
    )
