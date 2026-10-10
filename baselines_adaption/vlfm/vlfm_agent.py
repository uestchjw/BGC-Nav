"""
VLFM Agent adapted for the ASAR (Autonomous Search and Rescue) benchmark.

This agent combines the core ideas from VLFM (Vision-Language Frontier Maps)
with the AirSim drone interface required by the ASAR platform:

  - **BLIP2-ITM Value Map**: Front-camera RGB is scored by BLIP2 Image-Text
    Matching against the task prompt.  The score is projected onto a BEV grid
    via depth, forming a "value map" that guides frontier selection.
  - **YOLO-World Detection**: All 4 cameras run YOLO-World for open-vocabulary
    object detection.  Detections are reported as clues/targets.
  - **Semantic Map + FMM Planner**: Reuses SemExp's obstacle/explored map and
    Fast-Marching-Method local planner (zero-training replacement for PointNav).

4-Camera Setup
--------------
Uses four cameras (front/right/back/left) for 360-degree perception.
All 8 image requests (4 RGB 512x512 + 4 Depth 128x128) are batched in a
single ``simGetImages`` call.  YOLO runs on all 4 RGB images; BLIP2-ITM
and ValueMap update use only the front camera.

Usage
-----
Plug into the ASAR platform by importing this agent in
``source/main_platform_linux.py``::

    from baselines_adaption.vlfm.vlfm_agent import VLFMAgent
    # then in run_agent_logic:  agent = VLFMAgent(client, task)
"""

import sys
import os
import time
from datetime import datetime
import cv2
import numpy as np
import airsim

_SOURCE_DIR = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "source")
_SOURCE_DIR = os.path.abspath(_SOURCE_DIR)
_PROJECT_DIR = os.path.abspath(os.path.join(_SOURCE_DIR, os.pardir))
if _SOURCE_DIR not in sys.path:
    sys.path.insert(0, _SOURCE_DIR)

from agent_base import BaseAgent  # noqa: E402

from .semantic_map import SemanticMap
from .value_map import ValueMap
from .frontier import VLFMFrontierSelector
from .detection import Detector
from .utils.projection import pixel_to_world

# ==================== Configuration ====================

GRID_SIZE      = 400
CELL_SIZE      = 5.0
FLIGHT_SPEED   = 5.0
FOV_W          = 90
MAX_DEPTH      = 500.0
SAMPLE_STEP    = 4
OBS_Z_MARGIN   = 10.0
CONF_THRESHOLD = 0.25
REPLAN_DIST    = 10.0
STEP_INTERVAL  = 2.0
VALUE_SCORE_RADIUS = 2

CAMERAS = [
    ("front_camera",   0.0),
    ("right_camera",  90.0),
    ("back_camera",  180.0),
    ("left_camera",  -90.0),
]

DEFAULT_VOCAB = [
    "person", "fire", "backpack", "bottle", "flashlight",
    "jacket", "bag", "shoe", "sleepingbag", "tent", "flare",
]


class VLFMAgent(BaseAgent):
    """VLFM-based drone agent for the ASAR benchmark.

    Parameters
    ----------
    client : airsim.MultirotorClient
    task : dict
    vocab : list[str], optional
    """

    def __init__(self, client, task, vocab=None):
        super().__init__(client, task)
        self.set_trajectory_interval(10)
        self.vocab = list(vocab or DEFAULT_VOCAB)
        self.person_category_id = (
            self.vocab.index("person") if "person" in self.vocab else 0
        )
        self._task_id = task.get("task_id", "unknown")

        # --- Flight altitude ---
        state = client.getMultirotorState()
        self.flight_z = state.kinematics_estimated.position.z_val

        # --- Semantic Map ---
        start_pos = state.kinematics_estimated.position
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

        # --- Value Map ---
        self.value_map = ValueMap(
            grid_size=GRID_SIZE,
            cell_size=CELL_SIZE,
            center_x=start_pos.x_val,
            center_y=start_pos.y_val,
            camera_fov=FOV_W,
            max_depth=MAX_DEPTH,
            sample_step=SAMPLE_STEP,
        )

        # --- YOLO-World Detector ---
        weights_path = os.path.join(_SOURCE_DIR, "yolov8x-worldv2.pt")
        if not os.path.isfile(weights_path):
            weights_path = "yolov8x-worldv2.pt"
        self.detector = Detector(
            model_path=weights_path,
            vocab=self.vocab,
            conf_threshold=CONF_THRESHOLD,
        )

        # --- BLIP2-ITM ---
        from .utils.blip2itm import BLIP2ITM
        self.itm = BLIP2ITM()

        # --- Text prompt for ITM ---
        raw_prompt = self.get_task_prompt(default="Find the missing person")
        self.itm_prompt = f"Seems like there is a {raw_prompt} ahead."
        print(f"[VLFM] ITM prompt: {self.itm_prompt}")

        # --- Frontier Selector ---
        self.goal_selector = VLFMFrontierSelector(
            grid_size=GRID_SIZE,
            cell_size=CELL_SIZE,
            replan_dist=REPLAN_DIST,
            value_score_radius=VALUE_SCORE_RADIUS,
        )

        # --- Step bookkeeping ---
        self.step_count = 0
        self._last_step_time = 0.0
        self._fmm_needs_replan = False
        self._reported_target_positions = set()
        self._missing_pose_cameras = set()
        self.current_global_goal = None

        # --- 4-camera image requests ---
        self._image_requests = []
        for cam_name, _ in CAMERAS:
            self._image_requests.append(
                airsim.ImageRequest(cam_name, airsim.ImageType.Scene, False, False))
            self._image_requests.append(
                airsim.ImageRequest(cam_name, airsim.ImageType.DepthPlanar, True))

        # --- Detection debug outputs ---
        self._detection_dir = os.path.join(
            _PROJECT_DIR, "images_detection", "vlfm_images")
        os.makedirs(self._detection_dir, exist_ok=True)
        self._debug_log_path = os.path.join(
            _PROJECT_DIR, "images_detection", "debug.log")
        try:
            with open(self._debug_log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"# task_id={self._task_id} | VLFM YOLO debug session start: "
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
        except Exception:
            pass

        print(f"[VLFM] Agent initialised. task_id={self._task_id} "
              f"flight_z={self.flight_z:.1f}")

    # ==================================================================
    # BaseAgent interface
    # ==================================================================

    def run_step(self, current_time):
        if current_time - self._last_step_time < STEP_INTERVAL:
            return
        self._last_step_time = current_time
        self.step_count += 1

        # --- A. State ---
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        orientation = state.kinematics_estimated.orientation

        agent_gx, agent_gy = self.semantic_map.world_to_grid(
            pos.x_val, pos.y_val)

        # --- B. Perception: acquire all 8 images ---
        responses = self.client.simGetImages(self._image_requests)
        if len(responses) < len(self._image_requests):
            print(f"[VLFM] Image capture incomplete: got {len(responses)}/"
                  f"{len(self._image_requests)}")
            return

        camera_data = self._parse_camera_responses(responses)

        # --- C. Per-camera: detection + depth map update ---
        self.semantic_map.update_agent_position(pos.x_val, pos.y_val)

        front_rgb = None
        front_depth = None
        front_cam_pos = None
        front_cam_ori = None
        front_yaw_offset = 0.0

        for (cam_name, yaw_offset, img_rgb, rgb_shape, depth_img,
             cam_pos, cam_ori) in camera_data:
            use_response_pose = (cam_pos is not None and cam_ori is not None)
            proj_pos = cam_pos if use_response_pose else pos
            proj_ori = cam_ori if use_response_pose else orientation
            proj_yaw_offset = 0.0 if use_response_pose else yaw_offset

            if (not use_response_pose
                    and cam_name not in self._missing_pose_cameras):
                self._missing_pose_cameras.add(cam_name)
                print(f"[VLFM] Warning: {cam_name} missing camera pose; "
                      "using vehicle pose fallback")

            # C1. YOLO detection
            if img_rgb is not None and depth_img is not None:
                self._run_detection(
                    img_rgb, depth_img, rgb_shape, proj_pos, proj_ori,
                    camera_yaw_offset=proj_yaw_offset,
                    camera_name=cam_name,
                )

            # C2. Update obstacle/explored map
            if depth_img is not None:
                self.semantic_map.update_from_depth(
                    depth_img, proj_pos, proj_ori,
                    camera_yaw_offset=proj_yaw_offset,
                )

            # C3. Capture front camera data for ITM
            if cam_name == "front_camera":
                front_rgb = img_rgb
                front_depth = depth_img
                front_cam_pos = proj_pos
                front_cam_ori = proj_ori
                front_yaw_offset = proj_yaw_offset

        # --- D. Value map update (front camera only) ---
        if front_rgb is not None and front_depth is not None:
            itm_score = self.itm.cosine(front_rgb, self.itm_prompt)
            self.value_map.update(
                itm_score, front_depth,
                front_cam_pos, front_cam_ori,
                camera_yaw_offset=front_yaw_offset,
            )

        # --- E. Global goal selection (only when replan needed) ---
        force_replan = self.goal_selector.needs_replan(agent_gx, agent_gy)
        if force_replan or self._fmm_needs_replan:
            if self._fmm_needs_replan and not force_replan:
                print(f"[VLFM] Step {self.step_count}: FMM triggered replan")
            self._fmm_needs_replan = False

            global_goal = self.goal_selector.select_global_goal(
                self.semantic_map, self.value_map,
                agent_gx, agent_gy,
                person_category_id=self.person_category_id,
            )
            if global_goal is None:
                print("[VLFM] No frontiers left. Finishing mission.")
                self.finish_mission()
                return

            self.current_global_goal = global_goal

        # --- F. FMM local planning (every step) ---
        if self.current_global_goal is not None:
            stg_grid, stop, replan = self.goal_selector.plan_short_term(
                self.semantic_map, agent_gx, agent_gy,
                self.current_global_goal,
            )

            if stop:
                gg = self.current_global_goal
                self.semantic_map.mark_visited(gg[0], gg[1], radius_cells=3)
                self._fmm_needs_replan = True
            elif replan:
                self._fmm_needs_replan = True

            stg_x, stg_y = self.semantic_map.grid_to_world(
                int(stg_grid[0]), int(stg_grid[1]))

            # --- G. Execute flight ---
            self.client.moveToPositionAsync(
                stg_x, stg_y, self.flight_z, FLIGHT_SPEED)

        # --- H. Trajectory recording ---
        self.auto_record_position(self.step_count)

        # --- I. Periodic status log (every 10 steps) ---
        if self.step_count % 10 == 0:
            explored_pct = (self.semantic_map.get_explored_mask().sum()
                            / self.semantic_map.grid_size**2 * 100)
            goal_str = (f"grid={self.current_global_goal}"
                        if self.current_global_goal else "None")
            print(f"[VLFM] Step {self.step_count} | "
                  f"pos=({pos.x_val:.0f},{pos.y_val:.0f},z={pos.z_val:.0f}) | "
                  f"goal={goal_str} | explored={explored_pct:.1f}% | "
                  f"clues={len(self._found_clues)} "
                  f"targets={len(self._reported_positions)}")

    # ==================================================================
    # Image parsing (same pattern as semexp)
    # ==================================================================

    def _parse_camera_responses(self, responses):
        result = []
        for idx, (cam_name, yaw_offset) in enumerate(CAMERAS):
            rgb_resp = responses[idx * 2]
            depth_resp = responses[idx * 2 + 1]

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
                cam_name, yaw_offset, img_rgb, rgb_shape, depth_img,
                cam_pos, cam_ori,
            ))
        return result

    # ==================================================================
    # Detection (YOLO-World)
    # ==================================================================

    def _run_detection(self, img_rgb, depth_img, rgb_shape,
                       camera_position, camera_orientation,
                       camera_yaw_offset=0.0, camera_name="unknown"):
        detections = self.detector.detect(img_rgb)
        if not detections:
            return

        det_details = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cls_name = det["class_name"]
            cls_id = det["class_id"]
            conf = det["conf"]

            x1 = int(np.clip(round(x1), 0, rgb_shape[1] - 1))
            y1 = int(np.clip(round(y1), 0, rgb_shape[0] - 1))
            x2 = int(np.clip(round(x2), 0, rgb_shape[1] - 1))
            y2 = int(np.clip(round(y2), 0, rgb_shape[0] - 1))
            if x2 <= x1:
                x2 = min(rgb_shape[1] - 1, x1 + 1)
            if y2 <= y1:
                y2 = min(rgb_shape[0] - 1, y1 + 1)

            u = int((x1 + x2) / 2)
            v = int((y1 + y2) / 2)

            dh, dw = depth_img.shape
            du = int(u * dw / rgb_shape[1])
            dv = int(v * dh / rgb_shape[0])
            du = int(np.clip(du, 0, dw - 1))
            dv = int(np.clip(dv, 0, dh - 1))
            depth_val = float(depth_img[dv, du])

            if depth_val < 0.5:
                det_details.append(
                    f"{cls_name}@{conf:.2f} bbox=({x1},{y1},{x2},{y2}) "
                    f"depth=invalid"
                )
                continue

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

            self.report_clue(cls_name, world_pos)
            self.semantic_map.update_semantic(
                cls_id, world_pos["x"], world_pos["y"])

            if cls_name in ("person", "survivor", "body"):
                pos_key = (round(world_pos["x"], 0), round(world_pos["y"], 0))
                if pos_key not in self._reported_target_positions:
                    self._reported_target_positions.add(pos_key)
                    self.report_target(world_pos)

        if det_details:
            annotated_bgr = self.detector.annotate(img_rgb, detections)
            image_path = self._save_detection_image(annotated_bgr, camera_name)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            detail_str = " ; ".join(det_details)
            log_line = (
                f"task_id={self._task_id} | {ts} | step={self.step_count} | "
                f"camera={camera_name} | detections={len(det_details)} | "
                f"image={image_path} | detail={detail_str}"
            )
            self._append_detection_log(log_line)

    def _save_detection_image(self, annotated_bgr, camera_name):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_camera = str(camera_name).replace("/", "_")
        file_name = (
            f"task_id_{self._task_id}_step{self.step_count:06d}_{safe_camera}_{ts}.jpg"
        )
        file_path = os.path.join(self._detection_dir, file_name)
        ok = cv2.imwrite(file_path, annotated_bgr)
        return file_path if ok else "-"

    def _append_detection_log(self, line):
        try:
            with open(self._debug_log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as exc:
            print(f"[VLFM] Failed to write detection debug log: {exc}")
