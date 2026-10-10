"""
SemExp Agent adapted for the ASAR (Autonomous Search and Rescue) benchmark.

This agent combines the core ideas from Goal-Oriented Semantic Exploration
(SemExp) with the AirSim drone interface required by the ASAR platform:

  - **Semantic Map**: A multi-channel 2-D BEV grid built from depth projection
    and YOLO-World detections (replaces SemExp's Habitat Semantic_Mapping).
  - **Frontier-based Goal Selection**: Heuristic frontier exploration with
    semantic guidance (replaces SemExp's learned RL Global Policy).
  - **FMM Local Planner**: The original Fast-Marching-Method planner from
    SemExp, producing short-term waypoints toward the selected goal.

4-Camera Setup
--------------
Uses four cameras (front/right/back/left) for 360-degree perception.
All 8 image requests (4 RGB 512x512 + 4 Depth 128x128) are batched in a
single ``simGetImages`` call.  YOLO runs on all 4 RGB images; depth from
all 4 cameras updates the semantic map with per-camera yaw offsets.

Usage
-----
Plug into the ASAR platform by importing this agent in
``source/main_platform_linux.py``::

    from baselines_adaption.semexp.semexp_agent import SemExpAgent
    # then in run_agent_logic:  agent = SemExpAgent(client, task)
"""

import sys
import os
import time
from datetime import datetime
import cv2
import numpy as np
import airsim

# ---------------------------------------------------------------------------
# Make sure the ``source/`` directory is on sys.path so that ``agent_base``
# can be imported regardless of the working directory.
# ---------------------------------------------------------------------------
_SOURCE_DIR = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "source")
_SOURCE_DIR = os.path.abspath(_SOURCE_DIR)
_PROJECT_DIR = os.path.abspath(os.path.join(_SOURCE_DIR, os.pardir))
if _SOURCE_DIR not in sys.path:
    sys.path.insert(0, _SOURCE_DIR)

from agent_base import BaseAgent  # noqa: E402

from .semantic_map import SemanticMap
from .frontier import FrontierGoalSelector
from .utils.projection import pixel_to_world

# ==================== Configuration ====================

GRID_SIZE      = 400       # cells per side (400 x 400)
CELL_SIZE      = 5.0       # metres per cell  ->  2 km x 2 km coverage
FLIGHT_SPEED   = 5.0       # m/s cruise speed
FOV_W          = 90        # horizontal field-of-view (degrees)
MAX_DEPTH      = 500.0     # max trusted depth (metres); sky returns huge values
SAMPLE_STEP    = 4         # pixel stride for depth projection
OBS_Z_MARGIN   = 10.0      # obstacle band: ±10m around flight altitude
CONF_THRESHOLD = 0.25      # YOLO confidence threshold
REPLAN_DIST    = 10.0      # re-plan when within this distance to goal (m)
STEP_INTERVAL  = 2.0       # minimum seconds between run_step executions

# 4-camera definitions: (camera_name_in_settings, yaw_offset_degrees)
CAMERAS = [
    ("front_camera",   0.0),
    ("right_camera",  90.0),
    ("back_camera",  180.0),
    ("left_camera",  -90.0),
]

# Default vocabulary (SAR-relevant categories)
DEFAULT_VOCAB = [
    "person", "fire", "backpack", "bottle", "flashlight",
    "jacket", "bag", "shoe", "sleepingbag", "tent", "flare",
]


class SemExpAgent(BaseAgent):
    """SemExp-style agent for the ASAR drone benchmark (4-camera version).

    Parameters
    ----------
    client : airsim.MultirotorClient
        Connected AirSim client.
    task : dict
        Task metadata from ``task_data*.json``.
    vocab : list[str] or None
        Object vocabulary for YOLO-World.  Defaults to :data:`DEFAULT_VOCAB`.
    """

    def __init__(self, client, task, vocab=None):
        super().__init__(client, task)

        # SemExp 每10步记录一次轨迹
        self.set_trajectory_interval(10)

        # --- 1. Vocabulary & YOLO-World --------------------------------
        self.vocab = vocab if vocab is not None else list(DEFAULT_VOCAB)
        self.person_category_id = (
            self.vocab.index("person") if "person" in self.vocab else 0
        )

        print("[SemExp] Loading YOLO-World ...")
        from ultralytics import YOLOWorld
        weights_path = os.path.join(_SOURCE_DIR, "yolov8x-worldv2.pt")
        if not os.path.isfile(weights_path):
            weights_path = "yolov8x-worldv2.pt"
        self.yolo = YOLOWorld(weights_path)
        self.yolo_device = "cuda:0"
        self.yolo.to(self.yolo_device)
        self.yolo.set_classes(self.vocab)
        print(f"[SemExp] YOLO-World ready. Vocab({len(self.vocab)}): {self.vocab}")

        # --- 2. Record start position as map centre --------------------
        state = client.getMultirotorState()
        start_pos = state.kinematics_estimated.position
        self.flight_z = start_pos.z_val

        # --- 3. Semantic Map -------------------------------------------
        self.semantic_map = SemanticMap(
            grid_size=GRID_SIZE,
            cell_size=CELL_SIZE,
            num_categories=len(self.vocab),
            center_x=start_pos.x_val,
            center_y=start_pos.y_val,
            camera_fov=FOV_W,
            max_depth=MAX_DEPTH,
            sample_step=SAMPLE_STEP,
            obstacle_z_margin=OBS_Z_MARGIN,
        )

        # --- 4. Frontier Goal Selector ---------------------------------
        self.goal_selector = FrontierGoalSelector(
            grid_size=GRID_SIZE,
            cell_size=CELL_SIZE,
            replan_dist=REPLAN_DIST,
        )

        # --- 5. Internal state -----------------------------------------
        self.current_goal_world = None
        self.current_global_goal = None
        self.is_moving = False
        self.step_count = 0
        self._reported_target_positions = set()
        self._fmm_needs_replan = False
        self._last_step_time = 0.0
        self._missing_pose_cameras = set()

        # YOLO debug outputs
        self._detection_dir = os.path.join(_PROJECT_DIR, "images_detection")
        os.makedirs(self._detection_dir, exist_ok=True)
        self._debug_log_path = os.path.join(self._detection_dir, "debug.log")
        self._task_id = str(self.get_task_field("task_id", "unknown"))
        self._task_prompt = self.get_task_prompt(default="")
        safe_task_prompt = self._task_prompt.replace("\n", " ").strip()
        with open(self._debug_log_path, "a", encoding="utf-8") as f:
            f.write(
                f"# task_id={self._task_id} | SemExp YOLO debug session start: "
                f"{datetime.now().isoformat()}\n"
            )
            if safe_task_prompt:
                f.write(f"# task_prompt={safe_task_prompt}\n")

        # --- 6. Build the image request list once ----------------------
        # Order: front_rgb, front_depth, right_rgb, right_depth, ...
        self._image_requests = []
        for cam_name, _ in CAMERAS:
            self._image_requests.append(
                airsim.ImageRequest(cam_name, airsim.ImageType.Scene, False, False))
            self._image_requests.append(
                airsim.ImageRequest(cam_name, airsim.ImageType.DepthPlanar, True))

        print(f"[SemExp] 4-camera setup: {[c[0] for c in CAMERAS]}")

    # ==================================================================
    # BaseAgent interface
    # ==================================================================

    def run_step(self, current_time):
        """Execute one perception -> mapping -> planning -> action cycle."""
        if current_time - self._last_step_time < STEP_INTERVAL:
            return
        self._last_step_time = current_time
        self.step_count += 1

        # --- A. State ---------------------------------------------------
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        orientation = state.kinematics_estimated.orientation

        agent_gx, agent_gy = self.semantic_map.world_to_grid(
            pos.x_val, pos.y_val)

        # --- B. Perception: acquire all 8 images in one call ------------
        responses = self.client.simGetImages(self._image_requests)
        # We expect 8 responses: [front_rgb, front_depth, right_rgb, ...]
        if len(responses) < len(self._image_requests):
            print(f"[SemExp] Image capture incomplete: got {len(responses)}/"
                  f"{len(self._image_requests)}")
            return

        # Parse into per-camera tuples. Prefer per-frame camera pose from
        # response metadata; fall back to vehicle pose + static yaw offset.
        camera_data = self._parse_camera_responses(responses)

        # --- C. Per-camera: YOLO detection + depth map update -----------
        self.semantic_map.update_agent_position(pos.x_val, pos.y_val)

        for (cam_name, yaw_offset, img_rgb, rgb_shape, depth_img,
             cam_pos, cam_ori) in camera_data:
            use_response_pose = (cam_pos is not None and cam_ori is not None)
            proj_pos = cam_pos if use_response_pose else pos
            proj_ori = cam_ori if use_response_pose else orientation
            proj_yaw_offset = 0.0 if use_response_pose else yaw_offset

            if (not use_response_pose
                    and cam_name not in self._missing_pose_cameras):
                self._missing_pose_cameras.add(cam_name)
                print(f"[SemExp] Warning: {cam_name} missing camera pose; "
                      "using vehicle pose fallback")

            # C1. YOLO detection on this camera's RGB
            if img_rgb is not None and depth_img is not None:
                self._run_detection(
                    img_rgb, depth_img, rgb_shape, proj_pos, proj_ori,
                    camera_yaw_offset=proj_yaw_offset,
                    camera_name=cam_name,
                )

            # C2. Update semantic map from this camera's depth
            if depth_img is not None:
                self.semantic_map.update_from_depth(
                    depth_img, proj_pos, proj_ori,
                    camera_yaw_offset=proj_yaw_offset,
                )

        # --- D. Global goal selection ---------------------------------
        force_replan = self.goal_selector.needs_replan(agent_gx, agent_gy)
        if force_replan or self._fmm_needs_replan:
            if self._fmm_needs_replan and not force_replan:
                print(f"[SemExp] Step {self.step_count}: FMM triggered replan")
            self._fmm_needs_replan = False

            global_goal = self.goal_selector.select_goal(
                self.semantic_map, agent_gx, agent_gy,
                person_category_id=self.person_category_id,
            )
            if global_goal is None:
                print("[SemExp] No frontiers left. Finishing mission.")
                self.finish_mission()
                return

            self.current_global_goal = global_goal

        # --- E. FMM local planning (every step) ------------------------
        if self.current_global_goal is not None:
            stg, stop, replan = self.goal_selector.plan_short_term_goal(
                self.semantic_map, agent_gx, agent_gy, self.current_global_goal
            )

            if stop:
                # FMM says we've reached this goal — mark the goal area as
                # visited so select_goal won't pick the same person target
                # again in an infinite loop.
                gg = self.current_global_goal
                self.semantic_map.mark_visited(gg[0], gg[1], radius_cells=3)
                self._fmm_needs_replan = True
            elif replan:
                self._fmm_needs_replan = True

            wx, wy = self.semantic_map.grid_to_world(
                int(stg[0]), int(stg[1]))
            self.current_goal_world = (wx, wy)

            # --- F. Execute flight --------------------------------------
            self.client.moveToPositionAsync(
                wx, wy, self.flight_z, FLIGHT_SPEED)
            self.is_moving = True

        # --- G. Periodic status log (every 10 steps) -------------------
        if self.step_count % 10 == 0:
            explored_pct = (self.semantic_map.get_explored_mask().sum()
                            / self.semantic_map.grid_size**2 * 100)
            goal_str = (f"grid={self.current_global_goal}"
                        if self.current_global_goal else "None")
            stg_str = (f"({self.current_goal_world[0]:.0f},"
                       f"{self.current_goal_world[1]:.0f})"
                       if self.current_goal_world else "None")
            print(f"[SemExp] Step {self.step_count} | "
                  f"pos=({pos.x_val:.0f},{pos.y_val:.0f},z={pos.z_val:.0f}) | "
                  f"goal={goal_str} stg={stg_str} | "
                  f"explored={explored_pct:.1f}% | "
                  f"clues={len(self._found_clues)} "
                  f"targets={len(self._reported_positions)}")

            # 记录轨迹到日志
            self.record_position(self.step_count, {
                "x": pos.x_val,
                "y": pos.y_val,
                "z": pos.z_val,
            }, extra={
                "explored_pct": round(explored_pct, 2),
                "goal": goal_str,
            })

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _parse_camera_responses(self, responses):
        """Parse the flat response list into per-camera data.

        Returns
        -------
        list of (cam_name, yaw_offset, img_rgb_or_None, rgb_shape,
                 depth_or_None, cam_pos_or_None, cam_ori_or_None)
        """
        result = []
        for i, (cam_name, yaw_offset) in enumerate(CAMERAS):
            rgb_resp = responses[i * 2]
            depth_resp = responses[i * 2 + 1]

            # Parse RGB (512x512)
            img_rgb = None
            rgb_shape = (0, 0)
            if (rgb_resp.width > 0 and rgb_resp.height > 0
                    and len(rgb_resp.image_data_uint8) > 0):
                img_rgb = np.frombuffer(
                    rgb_resp.image_data_uint8, dtype=np.uint8
                ).reshape(rgb_resp.height, rgb_resp.width, 3)
                rgb_shape = (rgb_resp.height, rgb_resp.width)

            # Parse Depth (128x128)
            depth_img = None
            if (depth_resp.width > 0 and depth_resp.height > 0
                    and len(depth_resp.image_data_float) > 0):
                depth_img = airsim.list_to_2d_float_array(
                    depth_resp.image_data_float,
                    depth_resp.width, depth_resp.height)
                # Filter sky / invalid: clamp huge values to 0
                # (will be filtered by >0.5 check in update_from_depth)
                depth_img[depth_img > MAX_DEPTH] = 0.0
                depth_img[depth_img < 0] = 0.0

            # Prefer depth response pose (same optical path used for projection)
            # and fall back to RGB response pose if needed.
            cam_pos = getattr(depth_resp, "camera_position", None)
            cam_ori = getattr(depth_resp, "camera_orientation", None)
            if cam_pos is None or cam_ori is None:
                cam_pos = getattr(rgb_resp, "camera_position", None)
                cam_ori = getattr(rgb_resp, "camera_orientation", None)

            result.append((
                cam_name,
                yaw_offset,
                img_rgb,
                rgb_shape,
                depth_img,
                cam_pos,
                cam_ori,
            ))

        return result

    def _run_detection(self, img_rgb, depth_img, rgb_shape,
                       camera_position, camera_orientation,
                       camera_yaw_offset=0.0, camera_name="unknown_camera"):
        """Run YOLO-World on one camera's RGB and report clues / targets.

        Parameters
        ----------
        img_rgb : ndarray (H, W, 3)
        depth_img : ndarray (Hd, Wd), float — already filtered (sky = 0)
        rgb_shape : (H, W) of the RGB image
        camera_position : airsim.Vector3r
        camera_orientation : airsim.Quaternionr
            Camera orientation in world frame (response pose if available,
            otherwise vehicle orientation fallback).
        camera_yaw_offset : float
            Extra yaw offset relative to body (degrees). Only used in fallback
            mode when response camera pose is unavailable.
        camera_name : str
            Name of the camera stream (for debug outputs).
        """
        results = self.yolo.predict(
            img_rgb, conf=CONF_THRESHOLD, verbose=False,
            device=self.yolo_device,
        )

        annotated_bgr = None
        detection_count = 0
        det_details = []

        for r in results:
            if annotated_bgr is None and r.boxes is not None and len(r.boxes) > 0:
                # Use Ultralytics native visualization.
                annotated_bgr = r.plot()

            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id >= len(self.vocab):
                    continue
                cls_name = self.vocab[cls_id]
                conf = float(box.conf[0]) if box.conf is not None else 0.0

                # Bounding-box corners in RGB pixel coords
                x1f, y1f, x2f, y2f = box.xyxy[0].cpu().numpy()
                x1 = int(np.clip(round(x1f), 0, rgb_shape[1] - 1))
                y1 = int(np.clip(round(y1f), 0, rgb_shape[0] - 1))
                x2 = int(np.clip(round(x2f), 0, rgb_shape[1] - 1))
                y2 = int(np.clip(round(y2f), 0, rgb_shape[0] - 1))
                if x2 <= x1:
                    x2 = min(rgb_shape[1] - 1, x1 + 1)
                if y2 <= y1:
                    y2 = min(rgb_shape[0] - 1, y1 + 1)

                detection_count += 1

                # Bounding-box centre in RGB pixel coords
                u = int((x1 + x2) / 2)
                v = int((y1 + y2) / 2)
                u = int(np.clip(u, 0, rgb_shape[1] - 1))
                v = int(np.clip(v, 0, rgb_shape[0] - 1))

                # Map RGB pixel coords to depth image coords
                # (RGB is 512x512, Depth is 128x128)
                dh, dw = depth_img.shape
                du = int(u * dw / rgb_shape[1])
                dv = int(v * dh / rgb_shape[0])
                du = int(np.clip(du, 0, dw - 1))
                dv = int(np.clip(dv, 0, dh - 1))
                depth_val = float(depth_img[dv, du])

                # Skip invalid depth (0 = filtered sky/invalid, too close)
                if depth_val < 0.5:
                    det_details.append(
                        f"{cls_name}@{conf:.2f} bbox=({x1},{y1},{x2},{y2}) "
                        f"depth=invalid"
                    )
                    continue

                # Back-project to world coordinates
                world_pos = pixel_to_world(
                    v, u, depth_val, rgb_shape, FOV_W,
                    camera_position, camera_orientation,
                    camera_yaw_offset=camera_yaw_offset,
                )

                det_details.append(
                    f"{cls_name}@{conf:.2f} bbox=({x1},{y1},{x2},{y2}) "
                    f"depth={depth_val:.2f} "
                    f"world=({world_pos['x']:.1f},{world_pos['y']:.1f},{world_pos['z']:.1f})"
                )

                # Report clue with position
                self.report_clue(cls_name, world_pos)

                # Update semantic map
                self.semantic_map.update_semantic(
                    cls_id, world_pos["x"], world_pos["y"])

                # Report person targets
                if cls_name in ("person", "survivor", "body"):
                    pos_key = (
                        round(world_pos["x"], 0),
                        round(world_pos["y"], 0),
                    )
                    if pos_key not in self._reported_target_positions:
                        self._reported_target_positions.add(pos_key)
                        self.report_target(world_pos)

        if detection_count > 0:
            image_path = self._save_detection_image(annotated_bgr, camera_name)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            detail_str = " ; ".join(det_details) if det_details else "none"
            log_line = (
                f"task_id={self._task_id} | {ts} | step={self.step_count} | "
                f"camera={camera_name} | detections={detection_count} | "
                f"image={image_path} | detail={detail_str}"
            )
            self._append_detection_log(log_line)

    def _save_detection_image(self, annotated_bgr, camera_name):
        """Save one annotated detection image under images_detection/."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_camera = str(camera_name).replace("/", "_")
        file_name = (
            f"task_id_{self._task_id}_step{self.step_count:06d}_{safe_camera}_{ts}.jpg"
        )
        file_path = os.path.join(self._detection_dir, file_name)
        ok = cv2.imwrite(file_path, annotated_bgr)
        return file_path if ok else "-"

    def _append_detection_log(self, line):
        """Append one debug line to images_detection/debug.log."""
        try:
            with open(self._debug_log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as exc:
            print(f"[SemExp] Failed to write detection debug log: {exc}")
