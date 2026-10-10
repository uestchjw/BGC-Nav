"""
SPF (See-Point-Fly) Agent adapted for the ASAR benchmark.

Combines SPF's VLM-based spatial grounding navigation with a separate
YOLO-World detection module for clue/target reporting.

Navigation
----------
Uses a single forward RGB-D camera.  Each step, the VLM points at the most
promising location in the image; real AirSim depth at that pixel is used to
back-project to a 3-D displacement; the drone rotates to face the target
then moves forward (SPF's rotate-then-move pattern).

Detection
---------
Uses four cameras (front/right/back/left) with YOLO-World for 360-degree
object detection.  Detections are back-projected to world coordinates and
reported via ``report_clue`` / ``report_target``.

Usage
-----
::

    from baselines_adaption.SPF.spf_agent import SPFAgent
    agent = SPFAgent(client, task)
"""

import os
import sys

import airsim
import numpy as np

_SOURCE_DIR = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "source")
_SOURCE_DIR = os.path.abspath(_SOURCE_DIR)
_PROJECT_DIR = os.path.abspath(os.path.join(_SOURCE_DIR, os.pardir))
if _SOURCE_DIR not in sys.path:
    sys.path.insert(0, _SOURCE_DIR)

from agent_base import BaseAgent  # noqa: E402

from .detection import Detector, MAX_DEPTH
from .vlm_navigator import VLMNavigator

# ==================== Configuration ====================

STEP_INTERVAL     = 2.0
FLIGHT_SPEED      = 5.0
YAW_RATE          = 30.0
MIN_MOVE_DURATION = 1.5
MAX_MOVE_DURATION = 5.0
FOV_W             = 90

OBSTACLE_MIN_DEPTH    = 4.0

CAMERAS = [
    ("front_camera",   0.0),
    ("right_camera",  90.0),
    ("back_camera",  180.0),
    ("left_camera",  -90.0),
]


class SPFAgent(BaseAgent):
    """SPF-style agent for the ASAR drone benchmark.

    Parameters
    ----------
    client : airsim.MultirotorClient
    task : dict
    vocab : list[str] or None
    """

    def __init__(self, client, task, vocab=None):
        super().__init__(client, task)
        self.set_trajectory_interval(10)

        self._task_id = str(self.get_task_field("task_id", "unknown"))
        self._task_prompt = self.get_task_prompt(default="Search for victims.")

        self.detector = Detector(
            vocab=vocab,
            project_dir=_PROJECT_DIR,
            task_id=self._task_id,
            task_prompt=self._task_prompt,
        )

        self.vlm_nav = VLMNavigator(
            model="qwen3.5-plus",
            fov_h=FOV_W,
            project_dir=_PROJECT_DIR,
        )

        state = client.getMultirotorState()
        self._start_pos = state.kinematics_estimated.position
        self.flight_z = self._start_pos.z_val

        self._step_count = 0
        self._last_step_time = 0.0
        self._missing_pose_cameras = set()

        self._image_requests = []
        for cam_name, _ in CAMERAS:
            self._image_requests.append(
                airsim.ImageRequest(cam_name, airsim.ImageType.Scene, False, False))
            self._image_requests.append(
                airsim.ImageRequest(cam_name, airsim.ImageType.DepthPlanar, True))

        print(f"[SPF] Initialised. 4-camera detection + VLM navigation.")

    # ==================================================================
    # BaseAgent interface
    # ==================================================================

    def run_step(self, current_time):
        if current_time - self._last_step_time < STEP_INTERVAL:
            return
        self._last_step_time = current_time
        self._step_count += 1

        # A. Drone state
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        orientation = state.kinematics_estimated.orientation

        # B. Batch capture 8 images
        responses = self.client.simGetImages(self._image_requests)
        if len(responses) < len(self._image_requests):
            print(f"[SPF] Image capture incomplete: {len(responses)}/{len(self._image_requests)}")
            return

        camera_data = self._parse_camera_responses(responses)

        # C. YOLO detection on all 4 cameras
        front_rgb = None
        front_depth = None
        front_rgb_shape = (0, 0)

        for (cam_name, yaw_offset, img_rgb, rgb_shape, depth_img,
             cam_pos, cam_ori) in camera_data:
            use_response_pose = (cam_pos is not None and cam_ori is not None)
            proj_pos = cam_pos if use_response_pose else pos
            proj_ori = cam_ori if use_response_pose else orientation
            proj_yaw_offset = 0.0 if use_response_pose else yaw_offset

            if (not use_response_pose
                    and cam_name not in self._missing_pose_cameras):
                self._missing_pose_cameras.add(cam_name)
                print(f"[SPF] Warning: {cam_name} missing camera pose; "
                      "using vehicle pose fallback")

            if img_rgb is not None and depth_img is not None:
                self.detector.run_detection(
                    img_rgb, depth_img, rgb_shape,
                    proj_pos, proj_ori,
                    camera_yaw_offset=proj_yaw_offset,
                    camera_name=cam_name,
                    step_count=self._step_count,
                    agent=self,
                )

            if cam_name == "front_camera":
                front_rgb = img_rgb
                front_depth = depth_img
                front_rgb_shape = rgb_shape

        # D. VLM navigation on front camera only
        if front_rgb is not None and front_depth is not None:
            cmd = self.vlm_nav.get_navigation_command(
                front_rgb, front_depth, front_rgb_shape,
                task_prompt=self._task_prompt,
                step_count=self._step_count,
            )
            self._execute_nav_command(cmd, front_depth)
        else:
            self.client.moveByVelocityBodyFrameAsync(0, 0, 0, 1.0).join()

        # E. Periodic logging
        if self._step_count % 10 == 0:
            print(f"[SPF] Step {self._step_count} | "
                  f"pos=({pos.x_val:.0f},{pos.y_val:.0f},z={pos.z_val:.0f}) | "
                  f"clues={len(self._found_clues)} "
                  f"targets={len(self._reported_positions)}")

            self.record_position(self._step_count, {
                "x": pos.x_val,
                "y": pos.y_val,
                "z": pos.z_val,
            })

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _parse_camera_responses(self, responses):
        result = []
        for i, (cam_name, yaw_offset) in enumerate(CAMERAS):
            rgb_resp = responses[i * 2]
            depth_resp = responses[i * 2 + 1]

            img_rgb = None
            rgb_shape = (0, 0)
            if (rgb_resp.width > 0 and rgb_resp.height > 0
                    and len(rgb_resp.image_data_uint8) > 0):
                img_rgb = np.frombuffer(
                    rgb_resp.image_data_uint8, dtype=np.uint8
                ).reshape(rgb_resp.height, rgb_resp.width, 3)
                rgb_shape = (rgb_resp.height, rgb_resp.width)

            depth_img = None
            if (depth_resp.width > 0 and depth_resp.height > 0
                    and len(depth_resp.image_data_float) > 0):
                depth_img = airsim.list_to_2d_float_array(
                    depth_resp.image_data_float,
                    depth_resp.width, depth_resp.height)
                depth_img[depth_img > MAX_DEPTH] = 0.0
                depth_img[depth_img < 0] = 0.0

            cam_pos = getattr(depth_resp, "camera_position", None)
            cam_ori = getattr(depth_resp, "camera_orientation", None)
            if cam_pos is None or cam_ori is None:
                cam_pos = getattr(rgb_resp, "camera_position", None)
                cam_ori = getattr(rgb_resp, "camera_orientation", None)

            result.append((
                cam_name, yaw_offset, img_rgb, rgb_shape,
                depth_img, cam_pos, cam_ori,
            ))

        return result

    def _execute_nav_command(self, cmd, front_depth=None):
        """Execute SPF rotate-then-move-forward command with safety checks."""
        if not cmd.get("valid", False):
            self.client.moveByVelocityBodyFrameAsync(
                2.0, 0, 0, 2.0).join()
            return

        # --- Fix 3: obstacle check on front depth center region ---
        if front_depth is not None:
            dh, dw = front_depth.shape
            cy, cx = dh // 2, dw // 2
            r = max(dh // 6, 4)
            center_patch = front_depth[
                max(0, cy - r):min(dh, cy + r),
                max(0, cx - r):min(dw, cx + r),
            ]
            valid_depths = center_patch[(center_patch > 0.5) & (center_patch < MAX_DEPTH)]
            if len(valid_depths) > 0:
                min_front_depth = float(np.min(valid_depths))
                if min_front_depth < OBSTACLE_MIN_DEPTH:
                    print(f"[SPF] Obstacle ahead at {min_front_depth:.1f}m, "
                          f"avoid advancing")
                    self.client.rotateByYawRateAsync(YAW_RATE, 3.0).join()
                    # Go up
                    self.client.moveByVelocityBodyFrameAsync(0, 0, -2.0, 2.0).join()
                    self.client.hoverAsync().join()
                    return

        vz = cmd["vertical_speed"]

        yaw = cmd["yaw_angle"]
        if abs(yaw) > 5.0:
            rate = YAW_RATE if yaw > 0 else -YAW_RATE
            duration = abs(yaw) / YAW_RATE
            self.client.rotateByYawRateAsync(rate, duration).join()
            self.client.hoverAsync().join()

        self.client.moveByVelocityBodyFrameAsync(
            cmd["forward_speed"], 0, vz,
            cmd["duration"],
        ).join()
