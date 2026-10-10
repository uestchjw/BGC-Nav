"""YOLO-World based object detection module for NavGPT ASAR adaptation."""

import os
from datetime import datetime

import cv2
import numpy as np

from .utils.projection import pixel_to_world

FOV_W = 90
MAX_DEPTH = 500.0
CONF_THRESHOLD = 0.25

DEFAULT_VOCAB = [
    "person", "fire", "backpack", "bottle", "flashlight",
    "jacket", "bag", "shoe", "sleepingbag", "tent", "flare",
]


class Detector:
    """Independent 4-camera detector for NavGPT adaptation."""

    def __init__(
        self,
        vocab=None,
        weights_path=None,
        project_dir="",
        task_id="unknown",
        task_prompt="",
        device="cuda:0",
    ):
        self.vocab = vocab if vocab is not None else list(DEFAULT_VOCAB)
        self.device = device

        print("[NavGPT-Det] Loading YOLO-World ...")
        from ultralytics import YOLOWorld

        if weights_path is None:
            source_dir = os.path.join(
                os.path.dirname(__file__), os.pardir, os.pardir, "source")
            source_dir = os.path.abspath(source_dir)
            candidate = os.path.join(source_dir, "yolov8x-worldv2.pt")
            weights_path = candidate if os.path.isfile(candidate) else "yolov8x-worldv2.pt"

        self.yolo = YOLOWorld(weights_path)
        self.yolo.to(self.device)
        self.yolo.set_classes(self.vocab)
        print(f"[NavGPT-Det] YOLO-World ready. Vocab({len(self.vocab)}): {self.vocab}")

        self._detection_dir = os.path.join(project_dir, "images_detection")
        os.makedirs(self._detection_dir, exist_ok=True)
        self._debug_log_path = os.path.join(self._detection_dir, "debug.log")
        self._task_id = task_id

        safe_prompt = task_prompt.replace("\n", " ").strip()
        with open(self._debug_log_path, "a", encoding="utf-8") as f:
            f.write(
                f"# task_id={self._task_id} | NavGPT YOLO debug session start: "
                f"{datetime.now().isoformat()}\n"
            )
            if safe_prompt:
                f.write(f"# task_prompt={safe_prompt}\n")

        self._reported_target_positions = set()

    def run_detection(
        self,
        img_rgb,
        depth_img,
        rgb_shape,
        camera_position,
        camera_orientation,
        camera_yaw_offset=0.0,
        camera_name="unknown_camera",
        step_count=0,
        agent=None,
    ):
        results = self.yolo.predict(
            img_rgb,
            conf=CONF_THRESHOLD,
            verbose=False,
            device=self.device,
        )

        annotated_bgr = None
        detection_count = 0
        det_details = []

        for r in results:
            if annotated_bgr is None and r.boxes is not None and len(r.boxes) > 0:
                annotated_bgr = r.plot()

            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id >= len(self.vocab):
                    continue
                cls_name = self.vocab[cls_id]
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

                u = int(np.clip((x1 + x2) / 2, 0, rgb_shape[1] - 1))
                v = int(np.clip((y1 + y2) / 2, 0, rgb_shape[0] - 1))

                dh, dw = depth_img.shape
                du = int(np.clip(u * dw / rgb_shape[1], 0, dw - 1))
                dv = int(np.clip(v * dh / rgb_shape[0], 0, dh - 1))
                depth_val = float(depth_img[dv, du])

                if depth_val < 0.5:
                    det_details.append(
                        f"{cls_name}@{conf:.2f} bbox=({x1},{y1},{x2},{y2}) depth=invalid"
                    )
                    continue

                world_pos = pixel_to_world(
                    v,
                    u,
                    depth_val,
                    rgb_shape,
                    FOV_W,
                    camera_position,
                    camera_orientation,
                    camera_yaw_offset=camera_yaw_offset,
                )

                det_details.append(
                    f"{cls_name}@{conf:.2f} bbox=({x1},{y1},{x2},{y2}) "
                    f"depth={depth_val:.2f} "
                    f"world=({world_pos['x']:.1f},{world_pos['y']:.1f},{world_pos['z']:.1f})"
                )

                if agent is not None:
                    agent.report_clue(cls_name, world_pos)

                    if cls_name in ("person", "survivor", "body"):
                        pos_key = (round(world_pos["x"], 0), round(world_pos["y"], 0))
                        if pos_key not in self._reported_target_positions:
                            self._reported_target_positions.add(pos_key)
                            agent.report_target(world_pos)

        if detection_count > 0:
            image_path = "-"
            if annotated_bgr is not None:
                image_path = self._save_detection_image(annotated_bgr, camera_name, step_count)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            detail_str = " ; ".join(det_details) if det_details else "none"
            line = (
                f"task_id={self._task_id} | {ts} | step={step_count} | "
                f"camera={camera_name} | detections={detection_count} | "
                f"image={image_path} | detail={detail_str}"
            )
            self._append_detection_log(line)

    def _save_detection_image(self, annotated_bgr, camera_name, step_count):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_camera = str(camera_name).replace("/", "_")
        file_name = f"task_id_{self._task_id}_step{step_count:06d}_{safe_camera}_{ts}.jpg"
        file_path = os.path.join(self._detection_dir, file_name)
        ok = cv2.imwrite(file_path, annotated_bgr)
        return file_path if ok else "-"

    def _append_detection_log(self, line):
        try:
            with open(self._debug_log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as exc:
            print(f"[NavGPT-Det] Failed to write detection debug log: {exc}")
