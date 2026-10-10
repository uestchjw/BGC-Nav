"""
VLM-based navigation module for SPF (See-Point-Fly) ASAR adaptation.

Core SPF logic: VLM points at a target location in the image, real AirSim
depth provides the distance, perspective back-projection yields a 3D
displacement, which is converted to a rotate-then-move-forward command.
"""

import base64
import json
import math
import os
from datetime import datetime
from http import HTTPStatus

import cv2
import dashscope
import numpy as np
from dashscope import MultiModalConversation

# dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY", "")
#! H95
dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY", "")



DEPTH_FALLBACK = 10.0
DEPTH_PATCH_RADIUS = 3
MIN_VALID_DEPTH = 0.5
STEP_DISTANCE = 6.0
SAFETY_MARGIN = 3.0
FLIGHT_SPEED = 5.0
MIN_MOVE_DURATION = 1.5
MAX_MOVE_DURATION = 4.0
MAX_VZ = 2.0
VZ_UPWARD_BIAS = -0.8
YAW_THRESHOLD = 5.0
SKY_STEP_DISTANCE = 10.0


class VLMNavigator:
    """SPF-style VLM spatial grounding navigator using DashScope API."""

    def __init__(self, model="qwen3.5-plus", fov_h=90, project_dir=""):
        self.model = model
        self.fov_h = fov_h

        self._viz_dir = os.path.join(project_dir, "images", "spf_decisions")
        os.makedirs(self._viz_dir, exist_ok=True)
        self._log_path = os.path.join(self._viz_dir, "decisions.log")

        print(f"[SPF-Nav] VLMNavigator ready. model={model}, fov={fov_h}°")

    def get_navigation_command(self, front_rgb, front_depth, rgb_shape,
                               task_prompt, step_count):
        """Run one SPF cycle: VLM point → depth lookup → back-project → command.

        Returns dict with keys: yaw_angle, forward_speed, vertical_speed,
        duration, label, valid, pixel_x, pixel_y, real_depth.
        """
        try:
            prompt = self._build_prompt(task_prompt)

            _, buf = cv2.imencode(".jpg", front_rgb)
            b64_img = base64.b64encode(buf).decode("utf-8")

            messages = [{
                "role": "user",
                "content": [
                    {"image": f"data:image/jpeg;base64,{b64_img}"},
                    {"text": prompt},
                ],
            }]

            resp = MultiModalConversation.call(
                model=self.model,
                messages=messages,
                enable_thinking=False,
            )

            if resp.status_code != HTTPStatus.OK:
                print(f"[SPF-Nav] API error: {resp.code} - {resp.message}")
                return self._fallback_command()

            raw_text = resp.output.choices[0].message.content[0]["text"].strip()
            clean = self._clean_json(raw_text)
            points_data = json.loads(clean)

            if not points_data:
                print("[SPF-Nav] VLM returned empty points")
                return self._fallback_command()

            point_info = points_data[0]
            y_norm, x_norm = point_info["point"]
            label = point_info.get("label", "")

            pixel_x = int(np.clip((x_norm / 1000.0) * rgb_shape[1], 0, rgb_shape[1] - 1))
            pixel_y = int(np.clip((y_norm / 1000.0) * rgb_shape[0], 0, rgb_shape[0] - 1))

            real_depth, depth_valid = self._lookup_depth(
                pixel_x, pixel_y, front_depth, rgb_shape)

            if depth_valid:
                dx_body, dy_body, dz_body = self._backproject(
                    pixel_x, pixel_y, real_depth, rgb_shape)
            else:
                dx_body, dy_body, dz_body = self._sky_backproject(
                    pixel_x, pixel_y, rgb_shape)

            cmd = self._to_movement_command(dx_body, dy_body, dz_body)
            cmd["label"] = label
            cmd["valid"] = True
            cmd["pixel_x"] = pixel_x
            cmd["pixel_y"] = pixel_y
            cmd["real_depth"] = real_depth

            self._save_debug(front_rgb, cmd, step_count, raw_text)

            print(f"[SPF-Nav] Step {step_count} | point=({pixel_x},{pixel_y}) "
                  f"depth={real_depth:.1f}m yaw={cmd['yaw_angle']:.1f}° "
                  f"fwd={cmd['forward_speed']:.1f}m/s vz={cmd['vertical_speed']:.2f} "
                  f"dur={cmd['duration']:.1f}s | {label}")

            return cmd

        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            print(f"[SPF-Nav] Parse error: {exc}")
            if "raw_text" in dir():
                print(f"[SPF-Nav] Raw response: {raw_text[:200]}")
            return self._fallback_command()
        except Exception as exc:
            print(f"[SPF-Nav] Unexpected error: {exc}")
            return self._fallback_command()

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(task_prompt):
        return f"""You are a drone navigation expert analyzing a drone camera view for search and rescue.

Task: {task_prompt}

Analyze the image and identify the most promising direction to fly toward for the search mission.
Consider paths, terrain features, signs of human activity, and unexplored areas.

Place a single point DIRECTLY ON the location in the image where the drone should fly next.

Return in this exact JSON format:
[{{"point": [y, x], "label": "brief description of target"}}]

Coordinate system:
- x: 0-1000 scale (0=left edge, 500=center, 1000=right edge)
- y: 0-1000 scale (0=top edge, 500=center, 1000=bottom edge)

IMPORTANT:
- Place the point PRECISELY on the target location
- Choose the most promising direction for finding victims or clues
- If nothing specific is visible, point toward open traversable terrain"""

    # ------------------------------------------------------------------
    # Depth lookup
    # ------------------------------------------------------------------

    def _lookup_depth(self, pixel_x, pixel_y, depth_img, rgb_shape):
        """Sample real depth at VLM-pointed pixel with patch-based robustness.

        Returns (depth, is_valid). is_valid=False when pointing at sky/invalid.
        """
        dh, dw = depth_img.shape
        du = int(np.clip(pixel_x * dw / rgb_shape[1], 0, dw - 1))
        dv = int(np.clip(pixel_y * dh / rgb_shape[0], 0, dh - 1))

        r = DEPTH_PATCH_RADIUS
        v_lo = max(0, dv - r)
        v_hi = min(dh, dv + r + 1)
        u_lo = max(0, du - r)
        u_hi = min(dw, du + r + 1)
        patch = depth_img[v_lo:v_hi, u_lo:u_hi]

        valid = patch[(patch > MIN_VALID_DEPTH) & (patch < 500.0)]
        if len(valid) > 0:
            return float(np.median(valid)), True

        print(f"[SPF-Nav] No valid depth at ({pixel_x},{pixel_y}), sky/invalid")
        return DEPTH_FALLBACK, False

    # ------------------------------------------------------------------
    # Back-projection (SPF reverse_project_point adapted for NED)
    # ------------------------------------------------------------------

    def _backproject(self, pixel_x, pixel_y, depth, rgb_shape):
        """Back-project VLM point to body-frame displacement (NED).

        Computes a direction vector from the pixel, then scales to a safe
        step distance instead of flying the full depth to the target surface.

        Returns (dx_body, dy_body, dz_body).
        """
        h, w = rgb_shape
        cx, cy = w / 2.0, h / 2.0
        fov_factor = math.tan(math.radians(self.fov_h / 2.0))

        x_offset = pixel_x - cx
        y_offset = cy - pixel_y

        dx_dir = (x_offset / (w / 2.0)) * fov_factor
        dy_dir = 1.0
        dz_dir = -(y_offset / (h / 2.0)) * fov_factor

        norm = math.sqrt(dx_dir ** 2 + dy_dir ** 2 + dz_dir ** 2)
        if norm < 1e-6:
            return 0.0, STEP_DISTANCE, 0.0

        safe_dist = min(STEP_DISTANCE, max(0.5, depth - SAFETY_MARGIN))

        return (dx_dir / norm * safe_dist,
                dy_dir / norm * safe_dist,
                dz_dir / norm * safe_dist)

    def _sky_backproject(self, pixel_x, pixel_y, rgb_shape):
        """Compute direction from pixel position when depth is invalid (sky).

        Ignores vertical component — fly level in the horizontal direction
        indicated by the pixel's x position.
        """
        h, w = rgb_shape
        cx = w / 2.0
        fov_factor = math.tan(math.radians(self.fov_h / 2.0))

        dx_dir = (pixel_x - cx) / (w / 2.0) * fov_factor
        dy_dir = 1.0

        norm = math.sqrt(dx_dir ** 2 + dy_dir ** 2)
        if norm < 1e-6:
            return 0.0, SKY_STEP_DISTANCE, 0.0

        return (dx_dir / norm * SKY_STEP_DISTANCE,
                dy_dir / norm * SKY_STEP_DISTANCE,
                0.0)

    # ------------------------------------------------------------------
    # Movement command conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _to_movement_command(dx_body, dy_body, dz_body):
        """Convert body-frame displacement to yaw + forward + vertical command."""
        dist_xy = math.sqrt(dx_body ** 2 + dy_body ** 2)

        if dist_xy < 0.1:
            return {"yaw_angle": 0.0, "forward_speed": FLIGHT_SPEED,
                    "vertical_speed": float(VZ_UPWARD_BIAS),
                    "duration": MIN_MOVE_DURATION}

        yaw_angle = math.degrees(math.atan2(dx_body, dy_body))

        clamped_dist = min(dist_xy, MAX_MOVE_DURATION * FLIGHT_SPEED)
        duration = max(clamped_dist / FLIGHT_SPEED, MIN_MOVE_DURATION)
        duration = min(duration, MAX_MOVE_DURATION)

        vz = float(np.clip(dz_body / duration + VZ_UPWARD_BIAS, -MAX_VZ, MAX_VZ))

        return {"yaw_angle": yaw_angle, "forward_speed": FLIGHT_SPEED,
                "vertical_speed": vz, "duration": duration}

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_command():
        return {"yaw_angle": 0.0, "forward_speed": 2.0, "vertical_speed": 0.0,
                "duration": 2.0, "label": "fallback", "valid": False,
                "pixel_x": -1, "pixel_y": -1, "real_depth": -1.0}

    # ------------------------------------------------------------------
    # Debug / visualization
    # ------------------------------------------------------------------

    def _save_debug(self, image, cmd, step_count, raw_response):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        if cmd.get("pixel_x", -1) >= 0:
            viz = image.copy()
            px, py = cmd["pixel_x"], cmd["pixel_y"]
            cv2.circle(viz, (px, py), 12, (0, 255, 0), -1)
            cv2.circle(viz, (px, py), 14, (255, 255, 255), 2)
            info = (f"yaw={cmd['yaw_angle']:.0f} d={cmd['real_depth']:.0f}m "
                    f"spd={cmd['forward_speed']:.0f}")
            cv2.putText(viz, info, (px + 18, py), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)
            img_path = os.path.join(self._viz_dir, f"step{step_count:06d}_{ts}.jpg")
            cv2.imwrite(img_path, viz)

        log_line = (
            f"{ts} | step={step_count} | "
            f"point=({cmd.get('pixel_x', -1)},{cmd.get('pixel_y', -1)}) | "
            f"depth={cmd.get('real_depth', -1):.1f} | "
            f"yaw={cmd.get('yaw_angle', 0):.1f} | "
            f"label={cmd.get('label', '')}"
        )
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(log_line + "\n")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # JSON cleaning (from SPF VLMClient.clean_response_text)
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_json(text):
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            return text[start:end].strip()
        if "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            return text[start:end].strip()
        return text.strip()
