"""
Heuristic action selection for the APEX ASAR adaptation.

Adapted from baselines/uav_search/action_predict.py.
Evaluates 6 discrete actions by simulating each one and computing
expected attraction + exploration reward in the predicted view frustum.
"""

import numpy as np

ACTIONS = {
    0: "forward",
    1: "rotate_left",
    2: "rotate_right",
    3: "rotate_backward",
    4: "ascend",
    5: "descend",
}

MAP_SHAPE = np.array([40, 40, 10])
DRONE_POS_LOCAL = np.array([20, 20, 5])

VIEW_DEPTH = 100.0
VIEW_HEIGHT = 30.0
GRID_SIZE = 5.0

REWARD_DISTANCE_THRESHOLD = 50.0
KEY_THRESHOLD = 0.9
KEY_REWARD = 5.0
RATE_CENTER = 0.4

W_ATTRACTION = 1.0
W_EXPLORATION = 0.4

# Pre-compute grid indices (constant across calls)
_ix, _iy, _iz = np.indices(MAP_SHAPE)
_ALL_INDICES = np.stack([_ix.ravel(), _iy.ravel(), _iz.ravel()], axis=-1)
_ALL_CENTERS = _ALL_INDICES.astype(float)


def action_predict(attraction_map, exploration_map, obstacle_map):
    """Select the best discrete action based on map state.

    Parameters
    ----------
    attraction_map : ndarray (40, 40, 10) — attraction scores (channel 0 of the full map)
    exploration_map : ndarray (40, 40, 10)
    obstacle_map : ndarray (40, 40, 10)

    Returns
    -------
    int — action index (0-5)
    """
    predicted_rewards = {}
    current_pos_int = DRONE_POS_LOCAL.astype(int)

    for action_idx, action_name in ACTIONS.items():
        hypothetical_pos = DRONE_POS_LOCAL.copy().astype(float)
        hypothetical_ori = 0

        is_movement = False
        if action_name == "forward":
            hypothetical_pos[0] += 2
            is_movement = True
        elif action_name == "ascend":
            hypothetical_pos[2] -= 1
            is_movement = True
        elif action_name == "descend":
            hypothetical_pos[2] += 1
            is_movement = True
        elif action_name == "rotate_left":
            hypothetical_ori = 1
        elif action_name == "rotate_right":
            hypothetical_ori = 2
        elif action_name == "rotate_backward":
            hypothetical_ori = 3

        px, py, pz = int(hypothetical_pos[0]), int(hypothetical_pos[1]), int(hypothetical_pos[2])

        if not (1 < px < MAP_SHAPE[0] - 1 and 1 < py < MAP_SHAPE[1] - 1 and 1 < pz < MAP_SHAPE[2] - 1):
            predicted_rewards[action_idx] = -np.inf
            continue

        reward = 0.0
        if action_name in ("ascend", "descend"):
            reward -= 10.0

        # Obstacle safety check
        is_safe = True
        if is_movement:
            if action_name == "forward":
                sx, ex = current_pos_int[0], px + 2
                ex = min(ex, MAP_SHAPE[0])
                if np.any(obstacle_map[sx:ex, current_pos_int[1], current_pos_int[2]] >= 1.0):
                    is_safe = False
            elif action_name == "ascend":
                sz, ez = pz + 2, current_pos_int[2]
                sz = max(sz, 0)
                if sz < ez and np.any(obstacle_map[px, py, sz:ez] >= 1.0):
                    is_safe = False
            elif action_name == "descend":
                sz, ez = current_pos_int[2], pz + 2
                ez = min(ez, MAP_SHAPE[2])
                if sz < ez and np.any(obstacle_map[px, py, sz:ez] >= 1.0):
                    is_safe = False

        if obstacle_map[px, py, pz] >= 1.0:
            is_safe = False

        if not is_safe:
            predicted_rewards[action_idx] = -np.inf
            continue

        # Compute view frustum
        relative_coords = (_ALL_CENTERS - hypothetical_pos) * GRID_SIZE
        rel_x, rel_y, rel_z = relative_coords.T

        ori_conds = [hypothetical_ori == i for i in range(4)]
        local_y = np.select(ori_conds, [rel_x, -rel_y, rel_y, -rel_x])
        local_x = np.select(ori_conds, [rel_y, rel_x, -rel_x, -rel_y])

        in_frustum = (
            (np.abs(rel_z) <= VIEW_HEIGHT / 2)
            & (local_y > 0)
            & (local_y <= VIEW_DEPTH)
            & (np.abs(local_x) <= local_y)
        )

        observed = _ALL_INDICES[in_frustum]
        if observed.shape[0] == 0:
            predicted_rewards[action_idx] = reward
            continue

        distances = np.linalg.norm(relative_coords[in_frustum], axis=1)
        nearby = distances < REWARD_DISTANCE_THRESHOLD
        if not np.any(nearby):
            predicted_rewards[action_idx] = reward
            continue

        nearby_idx = observed[nearby]
        nearby_dist = distances[nearby]
        gx, gy, gz = nearby_idx.T

        valid = attraction_map[gx, gy, gz] != -1.0
        if not np.any(valid):
            predicted_rewards[action_idx] = reward
            continue

        gx, gy, gz = gx[valid], gy[valid], gz[valid]
        nearby_dist = nearby_dist[valid]
        dist_w = (REWARD_DISTANCE_THRESHOLD - nearby_dist) / REWARD_DISTANCE_THRESHOLD

        attr_vals = attraction_map[gx, gy, gz]
        attraction_reward = np.sum(attr_vals * dist_w)
        key_mask = attr_vals >= KEY_THRESHOLD
        attraction_reward += np.sum(KEY_REWARD * dist_w[key_mask])

        expl_vals = exploration_map[gx, gy, gz]
        exploration_reward = np.sum((RATE_CENTER - expl_vals) * dist_w)

        reward += W_ATTRACTION * attraction_reward + W_EXPLORATION * exploration_reward
        predicted_rewards[action_idx] = reward

    rewards_list = list(predicted_rewards.values())
    if max(rewards_list) - min(rewards_list) < 1e-2:
        sorted_actions = [1, 2, 0, 3, 4, 5]
    else:
        sorted_actions = sorted(predicted_rewards, key=lambda k: predicted_rewards[k], reverse=True)

    action = sorted_actions[0]
    print(f"[APEX] action_predict rewards={predicted_rewards} selected={action}")
    return int(action)
