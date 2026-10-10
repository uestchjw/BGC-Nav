"""
Frontier-Based Exploration (FBE) Baseline Agent for the ASAR benchmark.

Decision module: classic frontier-based exploration at fixed altitude.
  - Builds a BEV occupancy map from 4-camera depth projection.
  - Selects the largest unexplored frontier cluster as the next goal.
  - Uses FMM planner for local waypoint generation.
  - No semantic guidance — pure geometric exploration.
Detection module: shared 4-camera YOLO-World (same as SemExp).
"""

import sys
import os
import numpy as np
import airsim

_SOURCE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "source"))
if _SOURCE_DIR not in sys.path:
    sys.path.insert(0, _SOURCE_DIR)

from agent_base import BaseAgent
from .detection import DetectionModule, CAMERAS, MAX_DEPTH

from baselines_adaption.semexp.semantic_map import SemanticMap
from baselines_adaption.semexp.frontier import FrontierGoalSelector

# ── Configuration ──────────────────────────────────────────────────────
GRID_SIZE    = 400
CELL_SIZE    = 5.0
FLIGHT_SPEED = 5.0
FOV_W        = 90
SAMPLE_STEP  = 4
OBS_Z_MARGIN = 10.0
REPLAN_DIST  = 10.0
STEP_INTERVAL = 2.0


class FBEAgent(BaseAgent):

    def __init__(self, client, task):
        super().__init__(client, task)
        self.set_trajectory_interval(10)

        # Detection (4-camera YOLO)
        self.detector = DetectionModule(self, agent_tag="FBE")

        # Record start position & fixed flight altitude
        state = client.getMultirotorState()
        start_pos = state.kinematics_estimated.position
        self.flight_z = start_pos.z_val

        # Semantic map (only obstacle/explored channels used; no semantic guidance)
        self.semantic_map = SemanticMap(
            grid_size=GRID_SIZE,
            cell_size=CELL_SIZE,
            num_categories=len(self.detector.vocab),
            center_x=start_pos.x_val,
            center_y=start_pos.y_val,
            camera_fov=FOV_W,
            max_depth=MAX_DEPTH,
            sample_step=SAMPLE_STEP,
            obstacle_z_margin=OBS_Z_MARGIN,
        )

        # Frontier goal selector (pure frontier — we never update semantic
        # channels, so the person-priority path in select_goal is never taken)
        self.goal_selector = FrontierGoalSelector(
            grid_size=GRID_SIZE,
            cell_size=CELL_SIZE,
            replan_dist=REPLAN_DIST,
        )

        self.current_goal_world = None
        self.current_global_goal = None
        self.step_count = 0
        self._last_step_time = 0.0
        self._fmm_needs_replan = False
        self._missing_pose_cameras = set()
        self._task_id = str(self.get_task_field("task_id", "unknown"))

        print(f"[FBE] Initialised at ({start_pos.x_val:.0f}, "
              f"{start_pos.y_val:.0f}, z={start_pos.z_val:.0f})")

    def run_step(self, current_time):
        if current_time - self._last_step_time < STEP_INTERVAL:
            return
        self._last_step_time = current_time
        self.step_count += 1

        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        orientation = state.kinematics_estimated.orientation
        agent_gx, agent_gy = self.semantic_map.world_to_grid(
            pos.x_val, pos.y_val)

        # ── A. Perception: YOLO detection (clue/target reporting) ──────
        # Reuse the same 4-camera RGB+Depth batch for mapping (avoid a second
        # simGetImages call each step).
        responses = self.detector.capture_and_detect(self.step_count)

        # ── B. Depth map update (4-camera depth → BEV occupancy) ───────
        self.semantic_map.update_agent_position(pos.x_val, pos.y_val)

        for i, (cam_name, yaw_offset) in enumerate(CAMERAS):
            rgb_idx = i * 2
            depth_idx = rgb_idx + 1
            if depth_idx >= len(responses):
                break
            rgb_resp = responses[rgb_idx] if rgb_idx < len(responses) else None
            depth_resp = responses[depth_idx]
            if (depth_resp.width <= 0 or depth_resp.height <= 0
                    or len(depth_resp.image_data_float) == 0):
                continue

            depth_img = airsim.list_to_2d_float_array(
                depth_resp.image_data_float,
                depth_resp.width, depth_resp.height)
            depth_img[depth_img > MAX_DEPTH] = 0.0
            depth_img[depth_img < 0] = 0.0

            cam_pos = getattr(depth_resp, "camera_position", None)
            cam_ori = getattr(depth_resp, "camera_orientation", None)
            if (cam_pos is None or cam_ori is None) and rgb_resp is not None:
                cam_pos = getattr(rgb_resp, "camera_position", None)
                cam_ori = getattr(rgb_resp, "camera_orientation", None)
            use_response_pose = (cam_pos is not None and cam_ori is not None)
            proj_pos = cam_pos if use_response_pose else pos
            proj_ori = cam_ori if use_response_pose else orientation
            proj_yaw = 0.0 if use_response_pose else yaw_offset

            if (not use_response_pose
                    and cam_name not in self._missing_pose_cameras):
                self._missing_pose_cameras.add(cam_name)
                print(f"[FBE] Warning: {cam_name} missing camera pose; "
                      "using vehicle pose fallback")

            self.semantic_map.update_from_depth(
                depth_img, proj_pos, proj_ori,
                camera_yaw_offset=proj_yaw)

        # ── C. Global goal selection (pure frontier) ───────────────────
        force_replan = self.goal_selector.needs_replan(agent_gx, agent_gy)
        if force_replan or self._fmm_needs_replan:
            self._fmm_needs_replan = False
            global_goal = self.goal_selector.select_goal(
                self.semantic_map, agent_gx, agent_gy,
                person_category_id=self.detector.person_category_id,
            )
            if global_goal is None:
                print("[FBE] No frontiers left. Finishing mission.")
                self.finish_mission()
                return
            self.current_global_goal = global_goal

        # ── D. FMM local planning ─────────────────────────────────────
        if self.current_global_goal is not None:
            stg, stop, replan = self.goal_selector.plan_short_term_goal(
                self.semantic_map, agent_gx, agent_gy,
                self.current_global_goal)

            if stop:
                gg = self.current_global_goal
                self.semantic_map.mark_visited(gg[0], gg[1], radius_cells=3)
                self._fmm_needs_replan = True
            elif replan:
                self._fmm_needs_replan = True

            wx, wy = self.semantic_map.grid_to_world(
                int(stg[0]), int(stg[1]))
            self.current_goal_world = (wx, wy)

            # ── E. Execute flight (fixed altitude) ─────────────────────
            self.client.moveToPositionAsync(
                wx, wy, self.flight_z, FLIGHT_SPEED)

        # ── F. Periodic status log ─────────────────────────────────────
        if self.step_count % 10 == 0:
            explored_pct = (self.semantic_map.get_explored_mask().sum()
                            / self.semantic_map.grid_size ** 2 * 100)
            goal_str = (f"grid={self.current_global_goal}"
                        if self.current_global_goal else "None")
            stg_str = (f"({self.current_goal_world[0]:.0f},"
                       f"{self.current_goal_world[1]:.0f})"
                       if self.current_goal_world else "None")
            print(f"[FBE] Step {self.step_count} | "
                  f"pos=({pos.x_val:.0f},{pos.y_val:.0f},z={pos.z_val:.0f}) | "
                  f"goal={goal_str} stg={stg_str} | "
                  f"explored={explored_pct:.1f}% | "
                  f"clues={len(self._found_clues)} "
                  f"targets={len(self._reported_positions)}")
            self.record_position(self.step_count, {
                "x": pos.x_val, "y": pos.y_val, "z": pos.z_val,
            }, extra={
                "explored_pct": round(explored_pct, 2),
                "goal": goal_str,
            })
