"""
Shared 4-camera YOLO-World detection module for base baselines.

Extracted from semexp_agent.py. Provides 360-degree perception via four
cameras (front/right/back/left), YOLO-World object detection, and depth
back-projection for clue/target localisation.
"""

import sys
import os
import cv2
import numpy as np
import airsim
from datetime import datetime

_SOURCE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "source"))
_PROJECT_DIR = os.path.abspath(os.path.join(_SOURCE_DIR, os.pardir))
if _SOURCE_DIR not in sys.path:
    sys.path.insert(0, _SOURCE_DIR)

from baselines_adaption.semexp.utils.projection import pixel_to_world

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

FOV_W = 90
MAX_DEPTH = 500.0
CONF_THRESHOLD = 0.25


class DetectionModule:
    """4-camera YOLO-World detection with depth back-projection.

    Parameters
    ----------
    agent : BaseAgent
        Owner agent -- used for report_clue / report_target calls.
    agent_tag : str
        Short tag for log prefixes (e.g. "Random", "FBE", "MLLM").
    vocab : list[str] or None
        YOLO-World vocabulary. Defaults to SAR categories.
    """

    def __init__(self, agent, agent_tag="Base", vocab=None):
        self.agent = agent
        self.tag = agent_tag
        self.vocab = vocab if vocab is not None else list(DEFAULT_VOCAB)
        self.person_category_id = (
            self.vocab.index("person") if "person" in self.vocab else 0
        )

        print(f"[{self.tag}] Loading YOLO-World ...")
        from ultralytics import YOLOWorld
        weights_path = os.path.join(_SOURCE_DIR, "yolov8x-worldv2.pt")
        if not os.path.isfile(weights_path):
            weights_path = "yolov8x-worldv2.pt"
        self.yolo = YOLOWorld(weights_path)
        self.yolo_device = "cuda:0"
        self.yolo.to(self.yolo_device)
        self.yolo.set_classes(self.vocab)
        print(f"[{self.tag}] YOLO-World ready. Vocab({len(self.vocab)}): {self.vocab}")

        self._image_requests = []
        for cam_name, _ in CAMERAS:
            self._image_requests.append(
                airsim.ImageRequest(cam_name, airsim.ImageType.Scene, False, False))
            self._image_requests.append(
                airsim.ImageRequest(cam_name, airsim.ImageType.DepthPlanar, True))

        self._reported_target_positions = set()
        self._missing_pose_cameras = set()

        self._detection_dir = os.path.join(_PROJECT_DIR, "images_detection")
        os.makedirs(self._detection_dir, exist_ok=True)
        self._debug_log_path = os.path.join(self._detection_dir, "debug.log")
        self._task_id = str(agent.get_task_field("task_id", "unknown"))
        task_prompt = agent.get_task_prompt(default="").replace("\n", " ").strip()
        with open(self._debug_log_path, "a", encoding="utf-8") as f:
            f.write(
                f"# task_id={self._task_id} | {self.tag} YOLO debug session start: "
                f"{datetime.now().isoformat()}\n")
            if task_prompt:
                f.write(f"# task_prompt={task_prompt}\n")

    # ── Public API ─────────────────────────────────────────────────────

    def capture_and_detect(self, step_count):
        """Capture 4-camera images, run YOLO, report clues/targets.

        Returns the raw AirSim image responses for downstream use.
        """
        client = self.agent.client
        responses = client.simGetImages(self._image_requests)
        if len(responses) < len(self._image_requests):
            print(f"[{self.tag}] Image capture incomplete: "
                  f"got {len(responses)}/{len(self._image_requests)}")
            return responses

        state = client.getMultirotorState()
        pos = state.kinematics_estimated.position
        orientation = state.kinematics_estimated.orientation

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

            use_response_pose = (cam_pos is not None and cam_ori is not None)
            proj_pos = cam_pos if use_response_pose else pos
            proj_ori = cam_ori if use_response_pose else orientation
            proj_yaw_offset = 0.0 if use_response_pose else yaw_offset

            if (not use_response_pose
                    and cam_name not in self._missing_pose_cameras):
                self._missing_pose_cameras.add(cam_name)
                print(f"[{self.tag}] Warning: {cam_name} missing camera pose; "
                      "using vehicle pose fallback")

            if img_rgb is not None and depth_img is not None:
                self._run_detection(
                    img_rgb, depth_img, rgb_shape,
                    proj_pos, proj_ori,
                    camera_yaw_offset=proj_yaw_offset,
                    camera_name=cam_name,
                    step_count=step_count,
                )

        return responses

    # ── Internal ───────────────────────────────────────────────────────

    def _run_detection(self, img_rgb, depth_img, rgb_shape,
                       camera_position, camera_orientation,
                       camera_yaw_offset=0.0, camera_name="unknown",
                       step_count=0):
        results = self.yolo.predict(
            img_rgb, conf=CONF_THRESHOLD, device=self.yolo_device, verbose=False)
        if not results or len(results[0].boxes) == 0:
            return

        boxes = results[0].boxes
        annotated_bgr = img_rgb[:, :, ::-1].copy()
        detection_count = 0
        det_details = []

        depth_h, depth_w = depth_img.shape[:2]
        rgb_h, rgb_w = rgb_shape

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = self.vocab[cls_id] if cls_id < len(self.vocab) else f"cls{cls_id}"
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            cx_rgb = (x1 + x2) // 2
            cy_rgb = (y1 + y2) // 2
            cx_depth = int(cx_rgb * depth_w / rgb_w)
            cy_depth = int(cy_rgb * depth_h / rgb_h)
            cx_depth = np.clip(cx_depth, 0, depth_w - 1)
            cy_depth = np.clip(cy_depth, 0, depth_h - 1)

            region = depth_img[
                max(0, cy_depth - 2):cy_depth + 3,
                max(0, cx_depth - 2):cx_depth + 3]
            valid = region[(region > 0.5) & (region < MAX_DEPTH)]
            if len(valid) == 0:
                continue

            median_depth = float(np.median(valid))
            world_pos = pixel_to_world(
                cy_depth, cx_depth, median_depth,
                (depth_h, depth_w), FOV_W,
                camera_position, camera_orientation,
                camera_yaw_offset=camera_yaw_offset,
            )

            detection_count += 1
            det_details.append(
                f"{cls_name}({conf:.2f}) depth={median_depth:.1f}m "
                f"world=({world_pos['x']:.0f},{world_pos['y']:.0f},{world_pos['z']:.0f})")

            cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_bgr, f"{cls_name} {conf:.2f}",
                        (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            self.agent.report_clue(cls_name, position=world_pos)

            if cls_name in ("person", "survivor", "body"):
                pos_key = (round(world_pos["x"], 0), round(world_pos["y"], 0))
                if pos_key not in self._reported_target_positions:
                    self._reported_target_positions.add(pos_key)
                    self.agent.report_target(world_pos)

        if detection_count > 0:
            image_path = self._save_detection_image(
                annotated_bgr, camera_name, step_count)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            detail_str = " ; ".join(det_details) if det_details else "none"
            log_line = (
                f"task_id={self._task_id} | {ts} | step={step_count} | "
                f"camera={camera_name} | detections={detection_count} | "
                f"image={image_path} | detail={detail_str}")
            self._append_detection_log(log_line)

    def _save_detection_image(self, annotated_bgr, camera_name, step_count):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_camera = str(camera_name).replace("/", "_")
        file_name = (
            f"task_id_{self._task_id}_step{step_count:06d}_{safe_camera}_{ts}.jpg")
        file_path = os.path.join(self._detection_dir, file_name)
        ok = cv2.imwrite(file_path, annotated_bgr)
        return file_path if ok else "-"

    def _append_detection_log(self, line):
        try:
            with open(self._debug_log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as exc:
            print(f"[{self.tag}] Failed to write detection debug log: {exc}")
