"""
APEX Agent adapted for the ASAR (Autonomous Search and Rescue) benchmark.

This agent combines the core ideas from the APEX UAV search method with the
AirSim drone interface required by the ASAR platform:

  - **3D Voxel Maps**: Ego-centric attraction / exploration / obstacle grids
    (40x40x10, 5m/cell) that slide with the drone.
  - **VLM-Guided Attraction**: DashScope VLM identifies scene objects and
    scores their relevance to the SAR task; scores are projected into the
    attraction map via YOLO-World detections.
  - **Heuristic Action Selection**: Evaluates 6 discrete actions (forward,
    rotate_left/right/backward, ascend, descend) by simulating view frustums
    and computing expected attraction + exploration reward.

4-Camera Setup
--------------
Uses four cameras (front/right/back/left) for 360-degree YOLO-World detection.
Map update (attraction, exploration, obstacle) uses only the front camera.

Usage
-----
    from baselines_adaption.apex.apex_agent import ApexAgent
    agent = ApexAgent(client, task)
"""

import sys
import os
import time
import base64
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

from .voxel_maps import (
    create_maps, shift_maps, update_obstacle_from_depth,
    update_exploration, update_attraction, crop_rotate_maps,
    GRID_SCALE, MAP_SHAPE, DRONE_CENTER_GRID,
)
from .action_predict import action_predict, ACTIONS
from .vlm_analyzer import analyze_scene
from .utils.projection import pixel_to_world

# ==================== Configuration ====================

FLIGHT_SPEED = 5.0
MOVE_DISTANCE = 10.0       # forward movement per step (metres)
VERTICAL_STEP = 5.0        # ascend/descend per step (metres)
FOV_W = 90                 # horizontal FOV (degrees)
MAX_DEPTH = 250.0          # max trusted depth (metres)
CONF_THRESHOLD = 0.25      # YOLO confidence threshold
STEP_INTERVAL = 2.0        # minimum seconds between run_step executions
VLM_INTERVAL = 3           # call VLM every N steps

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


class ApexAgent(BaseAgent):
    """APEX-style agent for the ASAR drone benchmark."""

    def __init__(self, client, task):
        super().__init__(client, task)
        self.set_trajectory_interval(10)

        # --- 1. Task prompt -----------------------------------------------
        self._task_prompt = self.get_task_prompt(default="")

        print("[APEX] Loading YOLO-World ...")
        from ultralytics import YOLOWorld
        weights_path = os.path.join(_SOURCE_DIR, "yolov8x-worldv2.pt")
        if not os.path.isfile(weights_path):
            weights_path = "yolov8x-worldv2.pt"
        self.yolo = YOLOWorld(weights_path)
        self.yolo_device = "cuda:0"
        self.yolo.to(self.yolo_device)
        self.vocab = list(DEFAULT_VOCAB)
        self.yolo.set_classes(self.vocab)
        self._yolo_default_names = list(self.vocab)
        print(f"[APEX] YOLO-World ready. Vocab({len(self.vocab)}): {self.vocab}")

        # --- 2. Drone state -----------------------------------------------
        state = client.getMultirotorState()
        start_pos = state.kinematics_estimated.position
        self.flight_z = start_pos.z_val
        self.yaw_deg = self._quaternion_to_yaw(state.kinematics_estimated.orientation)

        # --- 3. 3D voxel maps (ego-centric) -------------------------------
        self.map_center_world = np.array([
            start_pos.x_val, start_pos.y_val, start_pos.z_val
        ])
        self.attraction_map, self.exploration_map, self.obstacle_map = create_maps()

        # --- 4. Internal state --------------------------------------------
        self.step_count = 0
        self._last_step_time = 0.0
        self._reported_target_positions = set()
        self._missing_pose_cameras = set()

        # --- 5. Logging (semexp pattern) ----------------------------------
        self._detection_dir = os.path.join(_PROJECT_DIR, "images_detection")
        os.makedirs(self._detection_dir, exist_ok=True)
        self._debug_log_path = os.path.join(self._detection_dir, "debug.log")
        self._task_id = str(self.get_task_field("task_id", "unknown"))
        safe_prompt = self._task_prompt.replace("\n", " ").strip()
        with open(self._debug_log_path, "a", encoding="utf-8") as f:
            f.write(
                f"# task_id={self._task_id} | APEX debug session start: "
                f"{datetime.now().isoformat()}\n"
            )
            if safe_prompt:
                f.write(f"# task_prompt={safe_prompt}\n")

        # --- 6. Image request list ----------------------------------------
        self._image_requests = []
        for cam_name, _ in CAMERAS:
            self._image_requests.append(
                airsim.ImageRequest(cam_name, airsim.ImageType.Scene, False, False))
            self._image_requests.append(
                airsim.ImageRequest(cam_name, airsim.ImageType.DepthPlanar, True))

        print(f"[APEX] 4-camera setup: {[c[0] for c in CAMERAS]}")
        print(f"[APEX] Task prompt: {safe_prompt}")

    # ==================================================================
    # BaseAgent interface
    # ==================================================================

    def run_step(self, current_time):
        if current_time - self._last_step_time < STEP_INTERVAL:
            return
        self._last_step_time = current_time
        self.step_count += 1

        # --- A. Drone state -----------------------------------------------
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        ori = state.kinematics_estimated.orientation
        drone_world = np.array([pos.x_val, pos.y_val, pos.z_val])
        self.yaw_deg = self._quaternion_to_yaw(ori)
        drone_ori = self._yaw_to_discrete_ori(self.yaw_deg)

        # --- B. Ego-centric map shift -------------------------------------
        self.attraction_map, self.exploration_map, self.obstacle_map, \
            self.map_center_world = shift_maps(
                self.attraction_map, self.exploration_map, self.obstacle_map,
                self.map_center_world, drone_world,
            )

        # --- C. Capture images (4 cameras) --------------------------------
        responses = self.client.simGetImages(self._image_requests)
        if len(responses) < len(self._image_requests):
            print(f"[APEX] Image capture incomplete: {len(responses)}/{len(self._image_requests)}")
            return

        camera_data = self._parse_camera_responses(responses)

        # --- D. Object detection (YOLO-World on all 4 cameras) ------------
        for (cam_name, yaw_offset, img_rgb, rgb_shape, depth_img,
             cam_pos, cam_ori) in camera_data:
            use_response_pose = cam_pos is not None and cam_ori is not None
            proj_pos = cam_pos if use_response_pose else pos
            proj_ori = cam_ori if use_response_pose else ori
            proj_yaw = 0.0 if use_response_pose else yaw_offset

            if not use_response_pose and cam_name not in self._missing_pose_cameras:
                self._missing_pose_cameras.add(cam_name)
                print(f"[APEX] Warning: {cam_name} missing camera pose; using vehicle pose fallback")

            if img_rgb is not None and depth_img is not None:
                self._run_detection(
                    img_rgb, depth_img, rgb_shape, proj_pos, proj_ori,
                    camera_yaw_offset=proj_yaw, camera_name=cam_name,
                )

        # --- E. Map update (front camera only) ----------------------------
        front_data = camera_data[0]  # front_camera is first
        _, _, front_rgb, front_rgb_shape, front_depth, front_cam_pos, front_cam_ori = front_data
        use_front_pose = front_cam_pos is not None and front_cam_ori is not None
        front_proj_pos = front_cam_pos if use_front_pose else pos
        front_proj_ori = front_cam_ori if use_front_pose else ori
        front_yaw_offset = 0.0 if use_front_pose else 0.0  # front camera has 0 offset

        # E1. Obstacle update from front depth
        if front_depth is not None:
            self.obstacle_map = update_obstacle_from_depth(
                self.obstacle_map, front_depth,
                front_proj_pos, front_proj_ori, FOV_W,
                self.map_center_world,
                camera_yaw_offset=front_yaw_offset,
            )

        # E1b. Hook for subclasses to add obstacle evidence from the other
        # cameras (omnidirectional obstacle map). No-op by default -> plain
        # ApexAgent behaviour (front-only) is unchanged.
        self._extra_obstacle_update(camera_data, pos, ori)

        # E2. Exploration update (every step)
        self.exploration_map = update_exploration(
            self.exploration_map, drone_world, drone_ori, self.map_center_world,
        )

        # E3. VLM + attraction update (every N steps)
        if self.step_count % VLM_INTERVAL == 0 and front_rgb is not None:
            self._update_attraction_with_vlm(
                front_rgb, front_depth, front_rgb_shape,
                front_proj_pos, front_proj_ori, front_yaw_offset,
            )

        # --- F. Action selection ------------------------------------------
        self._augment_attraction_map(drone_world)
        attr_local, expl_local, obs_local = crop_rotate_maps(
            self.attraction_map, self.exploration_map, self.obstacle_map,
            DRONE_CENTER_GRID, drone_ori,
        )
        action = action_predict(attr_local, expl_local, obs_local)

        # --- G. Execute action --------------------------------------------
        self._execute_action(action, pos)

        # --- H. Trajectory recording & logging ----------------------------
        self.auto_record_position(self.step_count)

        if self.step_count % 10 == 0:
            expl_pct = (self.exploration_map > 0).sum() / self.exploration_map.size * 100
            print(
                f"[APEX] Step {self.step_count} | "
                f"pos=({pos.x_val:.0f},{pos.y_val:.0f},z={pos.z_val:.0f}) | "
                f"yaw={self.yaw_deg:.0f} ori={drone_ori} | "
                f"action={ACTIONS.get(action, '?')} | "
                f"explored={expl_pct:.1f}% | "
                f"clues={len(self._found_clues)} "
                f"targets={len(self._reported_positions)}"
            )

    def _extra_obstacle_update(self, camera_data, pos, ori):
        """Hook for subclasses to fold the non-front cameras' depth into the
        obstacle map. No-op by default -> plain ApexAgent stays front-only."""
        pass

    def _on_vlm_detections(self, detections):
        """Hook: called with the VLM-vocabulary detections (each has 'name',
        'score', 'world_pos') just before they are written into the attraction
        map. No-op by default -> plain ApexAgent behaviour is unchanged."""
        pass

    def _augment_attraction_map(self, drone_world):
        """Hook for subclasses to inject an external attraction prior
        (e.g. a high-altitude terrain-relevance field) into
        ``self.attraction_map`` right before action selection.

        No-op by default — plain ``ApexAgent`` behaviour is unchanged.
        """
        pass

    # ==================================================================
    # VLM-guided attraction update
    # ==================================================================

    def _update_attraction_with_vlm(self, front_rgb, front_depth, rgb_shape,
                                     cam_pos, cam_ori, yaw_offset):
        """Call VLM to get object relevance, then detect + project into attraction map."""
        # Encode front RGB as base64
        _, buf = cv2.imencode(".jpg", front_rgb)
        b64 = base64.b64encode(buf).decode("utf-8")

        vlm_result = analyze_scene(b64, self._task_prompt)
        if not vlm_result["success"] or not vlm_result["objects"]:
            return

        objects = vlm_result["objects"]
        scores = vlm_result["scores"]
        print(f"[APEX] VLM objects: {list(zip(objects, scores))}")

        # Set YOLO-World to VLM vocabulary for this detection
        self.yolo.set_classes(objects)
        results = self.yolo.predict(
            front_rgb, conf=CONF_THRESHOLD, verbose=False,
            device=self.yolo_device,
        )

        # Reset to default COCO80 vocabulary for general detection
        self.yolo.set_classes(self._yolo_default_names)

        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id >= len(objects):
                    continue

                x1f, y1f, x2f, y2f = box.xyxy[0].cpu().numpy()
                u = int((x1f + x2f) / 2)
                v = int((y1f + y2f) / 2)
                u = int(np.clip(u, 0, rgb_shape[1] - 1))
                v = int(np.clip(v, 0, rgb_shape[0] - 1))

                if front_depth is None:
                    continue
                dh, dw = front_depth.shape
                du = int(u * dw / rgb_shape[1])
                dv = int(v * dh / rgb_shape[0])
                du = int(np.clip(du, 0, dw - 1))
                dv = int(np.clip(dv, 0, dh - 1))
                depth_val = float(front_depth[dv, du])

                if depth_val < 0.5:
                    continue

                world_pos = pixel_to_world(
                    v, u, depth_val, rgb_shape, FOV_W,
                    cam_pos, cam_ori, camera_yaw_offset=yaw_offset,
                )

                detections.append({
                    "world_pos": np.array([world_pos["x"], world_pos["y"], world_pos["z"]]),
                    "score": scores[cls_id],
                    "name": objects[cls_id],
                })

        if detections:
            self._on_vlm_detections(detections)
            self.attraction_map = update_attraction(
                self.attraction_map, detections, self.map_center_world,
            )
            print(f"[APEX] Attraction updated with {len(detections)} detections")

    # ==================================================================
    # Action execution
    # ==================================================================

    def _execute_action(self, action, current_pos):
        """Convert discrete action to AirSim movement command."""
        action_name = ACTIONS.get(action, "forward")
        yaw_rad = np.deg2rad(self.yaw_deg)

        if action_name == "forward":
            # Move forward in current heading direction
            dx = MOVE_DISTANCE * np.cos(yaw_rad)
            dy = MOVE_DISTANCE * np.sin(yaw_rad)
            target_x = current_pos.x_val + dx
            target_y = current_pos.y_val + dy
            self.client.moveToPositionAsync(
                target_x, target_y, self.flight_z, FLIGHT_SPEED,
            ).join()

        elif action_name == "rotate_left":
            new_yaw = self.yaw_deg - 90
            self.client.rotateToYawAsync(new_yaw).join()
            self.yaw_deg = new_yaw

        elif action_name == "rotate_right":
            new_yaw = self.yaw_deg + 90
            self.client.rotateToYawAsync(new_yaw).join()
            self.yaw_deg = new_yaw

        elif action_name == "rotate_backward":
            new_yaw = self.yaw_deg + 180
            self.client.rotateToYawAsync(new_yaw).join()
            self.yaw_deg = new_yaw

        elif action_name == "ascend":
            new_z = self.flight_z - VERTICAL_STEP
            self.client.moveToPositionAsync(
                current_pos.x_val, current_pos.y_val, new_z, FLIGHT_SPEED,
            ).join()
            self.flight_z = new_z

        elif action_name == "descend":
            new_z = self.flight_z + VERTICAL_STEP
            self.client.moveToPositionAsync(
                current_pos.x_val, current_pos.y_val, new_z, FLIGHT_SPEED,
            ).join()
            self.flight_z = new_z

    # ==================================================================
    # Image parsing (same pattern as semexp)
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

            result.append((cam_name, yaw_offset, img_rgb, rgb_shape,
                           depth_img, cam_pos, cam_ori))
        return result

    # ==================================================================
    # Detection (same pattern as semexp)
    # ==================================================================

    def _run_detection(self, img_rgb, depth_img, rgb_shape,
                       camera_position, camera_orientation,
                       camera_yaw_offset=0.0, camera_name="unknown_camera"):
        results = self.yolo.predict(
            img_rgb, conf=CONF_THRESHOLD, verbose=False,
            device=self.yolo_device,
        )

        annotated_bgr = None
        detection_count = 0
        det_details = []

        for r in results:
            if annotated_bgr is None and r.boxes is not None and len(r.boxes) > 0:
                annotated_bgr = r.plot()

            for box in r.boxes:
                cls_id = int(box.cls[0])
                names = r.names
                if isinstance(names, dict):
                    cls_name = names.get(cls_id, f"class_{cls_id}")
                elif isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
                    cls_name = names[cls_id]
                else:
                    cls_name = f"class_{cls_id}"
                conf = float(box.conf[0]) if box.conf is not None else 0.0

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
                u = int((x1 + x2) / 2)
                v = int((y1 + y2) / 2)
                u = int(np.clip(u, 0, rgb_shape[1] - 1))
                v = int(np.clip(v, 0, rgb_shape[0] - 1))

                dh, dw = depth_img.shape
                du = int(u * dw / rgb_shape[1])
                dv = int(v * dh / rgb_shape[0])
                du = int(np.clip(du, 0, dw - 1))
                dv = int(np.clip(dv, 0, dh - 1))
                depth_val = float(depth_img[dv, du])

                if depth_val < 0.5:
                    det_details.append(
                        f"{cls_name}@{conf:.2f} bbox=({x1},{y1},{x2},{y2}) depth=invalid"
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

                if cls_name in ("person", "survivor", "body"):
                    pos_key = (round(world_pos["x"], 0), round(world_pos["y"], 0))
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

    # ==================================================================
    # Utilities
    # ==================================================================

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
            print(f"[APEX] Failed to write detection debug log: {exc}")

    @staticmethod
    def _quaternion_to_yaw(q):
        """Extract yaw (degrees) from AirSim quaternion."""
        siny_cosp = 2.0 * (q.w_val * q.z_val + q.x_val * q.y_val)
        cosy_cosp = 1.0 - 2.0 * (q.y_val * q.y_val + q.z_val * q.z_val)
        yaw_rad = np.arctan2(siny_cosp, cosy_cosp)
        return np.degrees(yaw_rad)

    @staticmethod
    def _yaw_to_discrete_ori(yaw_deg):
        """Convert yaw (degrees) to discrete orientation 0-3.

        0=North (yaw~0), 1=West (yaw~-90), 2=South (yaw~180), 3=East (yaw~90)
        Matches the original APEX convention.
        """
        normalized = (-yaw_deg + 45) % 360
        return int(normalized // 90)
