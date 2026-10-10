"""NavGPT-style baseline adapted to ASAR with caption->policy pipeline."""

import json
import os
import sys
from collections import deque

import airsim
import cv2
import numpy as np

_SOURCE_DIR = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "source")
_SOURCE_DIR = os.path.abspath(_SOURCE_DIR)
_PROJECT_DIR = os.path.abspath(os.path.join(_SOURCE_DIR, os.pardir))
if _SOURCE_DIR not in sys.path:
    sys.path.insert(0, _SOURCE_DIR)

from agent_base import BaseAgent  # noqa: E402

from .detection import Detector, MAX_DEPTH
from .policy_prompt import build_policy_prompt
from .safety_filter import filter_safe_actions
from .vlm_client import DashScopeVLMClient

STEP_INTERVAL = 2.0
FLIGHT_SPEED = 5.0
STEP_MOVE_DISTANCE = 15.0
STEP_VERTICAL_DISTANCE = 10.0
CAPTION_EVERY_N_STEPS = 2

# Movement executed as constant-velocity body-frame motion for a fixed distance.
# duration = distance / speed
STEP_DURATION = STEP_MOVE_DISTANCE / FLIGHT_SPEED
VERTICAL_SPEED = 3.0
VERTICAL_DURATION = STEP_VERTICAL_DISTANCE / VERTICAL_SPEED

CAMERAS = [
    ("front_camera", 0.0),
    ("right_camera", 90.0),
    ("back_camera", 180.0),
    ("left_camera", -90.0),
]


class NavGPTAgent(BaseAgent):
    """NavGPT adaptation for ASAR using DashScope VLM, no LangChain."""

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
        self.vlm = DashScopeVLMClient(model="qwen3.5-plus")

        self._caption_cache = {c[0]: "No caption yet." for c in CAMERAS}
        self._history = deque(maxlen=8)
        self._step_count = 0
        self._last_step_time = 0.0

        state = self.client.getMultirotorState()
        self._start_pos = state.kinematics_estimated.position

        self._temp_img_dir = os.path.join(_PROJECT_DIR, "images", "temp", "navgpt")
        os.makedirs(self._temp_img_dir, exist_ok=True)

        self._image_requests = []
        for cam_name, _ in CAMERAS:
            self._image_requests.append(
                airsim.ImageRequest(cam_name, airsim.ImageType.Scene, False, False)
            )
            self._image_requests.append(
                airsim.ImageRequest(cam_name, airsim.ImageType.DepthPlanar, True)
            )

        self._missing_pose_cameras = set()

        print("[NavGPT] Initialised. caption->policy enabled.")

    def run_step(self, current_time):
        if current_time - self._last_step_time < STEP_INTERVAL:
            return
        self._last_step_time = current_time
        self._step_count += 1

        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        orientation = state.kinematics_estimated.orientation

        responses = self.client.simGetImages(self._image_requests)
        if len(responses) < len(self._image_requests):
            print(f"[NavGPT] Image capture incomplete: got {len(responses)}/{len(self._image_requests)}")
            return

        camera_data = self._parse_camera_responses(responses)
        image_paths = {}
        depth_by_camera = {}

        for (cam_name, yaw_offset, img_rgb, rgb_shape, depth_img, cam_pos, cam_ori) in camera_data:
            use_response_pose = (cam_pos is not None and cam_ori is not None)
            proj_pos = cam_pos if use_response_pose else pos
            proj_ori = cam_ori if use_response_pose else orientation
            proj_yaw_offset = 0.0 if use_response_pose else yaw_offset

            if (not use_response_pose) and cam_name not in self._missing_pose_cameras:
                self._missing_pose_cameras.add(cam_name)
                print(f"[NavGPT] Warning: {cam_name} missing camera pose; using vehicle pose fallback")

            if img_rgb is not None:
                image_paths[cam_name] = self._save_rgb_temp(img_rgb, cam_name)

            if depth_img is not None:
                depth_by_camera[cam_name] = depth_img

            if img_rgb is not None and depth_img is not None:
                self.detector.run_detection(
                    img_rgb,
                    depth_img,
                    rgb_shape,
                    proj_pos,
                    proj_ori,
                    camera_yaw_offset=proj_yaw_offset,
                    camera_name=cam_name,
                    step_count=self._step_count,
                    agent=self,
                )

        if self._step_count % CAPTION_EVERY_N_STEPS == 0:
            self._update_captions(image_paths)

        rel_pos_txt, rel_alt = self._get_relative_position_text()
        safe_actions = filter_safe_actions(depth_by_camera, rel_alt)

        policy_prompt = build_policy_prompt(
            task_prompt=self._task_prompt,
            captions=self._caption_cache,
            history=list(self._history),
            safe_actions=safe_actions,
            current_pos_text=rel_pos_txt,
        )

        try:
            raw = self.vlm.decide_policy(image_paths, policy_prompt)
            action, args, thought = self._parse_policy_json(raw, safe_actions)
        except Exception as exc:
            print(f"[NavGPT] Policy failed: {exc}; fallback to first safe action")
            action = safe_actions[0] if safe_actions else "move_up"
            args, thought = "none", "policy_error_fallback"

        self._history.append(
            f"step {self._step_count}: action={action}, args={args}, thought={thought[:120]}"
        )

        print(f"[NavGPT] Step {self._step_count} | action={action} | args={args} | thought={thought}")
        self._execute_action(action, args)

        if self._step_count % 10 == 0:
            self.record_position(self._step_count, {
                "x": pos.x_val,
                "y": pos.y_val,
                "z": pos.z_val,
            }, extra={
                "safe_actions": safe_actions,
                "caption_front": self._caption_cache.get("front_camera", ""),
            })

    def _parse_camera_responses(self, responses):
        result = []
        for i, (cam_name, yaw_offset) in enumerate(CAMERAS):
            rgb_resp = responses[i * 2]
            depth_resp = responses[i * 2 + 1]

            img_rgb = None
            rgb_shape = (0, 0)
            if (
                rgb_resp.width > 0
                and rgb_resp.height > 0
                and len(rgb_resp.image_data_uint8) > 0
            ):
                img_rgb = np.frombuffer(
                    rgb_resp.image_data_uint8, dtype=np.uint8
                ).reshape(rgb_resp.height, rgb_resp.width, 3)
                rgb_shape = (rgb_resp.height, rgb_resp.width)

            depth_img = None
            if (
                depth_resp.width > 0
                and depth_resp.height > 0
                and len(depth_resp.image_data_float) > 0
            ):
                depth_img = airsim.list_to_2d_float_array(
                    depth_resp.image_data_float,
                    depth_resp.width,
                    depth_resp.height,
                )
                depth_img[depth_img > MAX_DEPTH] = 0.0
                depth_img[depth_img < 0] = 0.0

            cam_pos = getattr(depth_resp, "camera_position", None)
            cam_ori = getattr(depth_resp, "camera_orientation", None)
            if cam_pos is None or cam_ori is None:
                cam_pos = getattr(rgb_resp, "camera_position", None)
                cam_ori = getattr(rgb_resp, "camera_orientation", None)

            result.append(
                (cam_name, yaw_offset, img_rgb, rgb_shape, depth_img, cam_pos, cam_ori)
            )

        return result

    def _save_rgb_temp(self, img_rgb, cam_name):
        path = os.path.join(self._temp_img_dir, f"{self._task_id}_{cam_name}.jpg")
        cv2.imwrite(path, img_rgb)
        return path

    def _update_captions(self, image_paths):
        for cam_name in ["front_camera", "right_camera", "back_camera", "left_camera"]:
            img_path = image_paths.get(cam_name)
            if not img_path:
                continue
            try:
                self._caption_cache[cam_name] = self.vlm.caption_image(img_path, cam_name)
            except Exception as exc:
                print(f"[NavGPT] Caption failed on {cam_name}: {exc}")

    def _get_relative_position_text(self):
        state = self.client.getMultirotorState().kinematics_estimated.position
        rel_x = state.x_val - self._start_pos.x_val
        rel_y = state.y_val - self._start_pos.y_val
        rel_alt = -(state.z_val - self._start_pos.z_val)
        txt = f"Forward_X={rel_x:.1f}m, Right_Y={rel_y:.1f}m, Altitude={rel_alt:.1f}m"
        return txt, rel_alt

    def _parse_policy_json(self, raw_text, safe_actions):
        clean = raw_text.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean)
        action = str(data.get("action", "move_up")).strip()
        args = str(data.get("args", "none")).strip()
        thought = str(data.get("thought", "")).strip()

        movement = {
            "move_forward", "move_backward", "move_left", "move_right", "move_up", "move_down"
        }

        fallback = safe_actions[0] if safe_actions else "move_up"

        # Navigation module only outputs movement actions.
        # Clue/target reporting is handled entirely by the detection module.
        if action not in movement or action not in safe_actions:
            action = fallback

        return action, "none", thought

    def _execute_action(self, action, args):
        if action == "move_forward":
            self.client.moveByVelocityBodyFrameAsync(FLIGHT_SPEED, 0, 0, STEP_DURATION).join()
        elif action == "move_backward":
            self.client.moveByVelocityBodyFrameAsync(-FLIGHT_SPEED, 0, 0, STEP_DURATION).join()
        elif action == "move_left":
            self.client.moveByVelocityBodyFrameAsync(0, -FLIGHT_SPEED, 0, STEP_DURATION).join()
        elif action == "move_right":
            self.client.moveByVelocityBodyFrameAsync(0, FLIGHT_SPEED, 0, STEP_DURATION).join()
        elif action == "move_up":
            self.client.moveByVelocityBodyFrameAsync(0, 0, -VERTICAL_SPEED, VERTICAL_DURATION).join()
        elif action == "move_down":
            self.client.moveByVelocityBodyFrameAsync(0, 0, VERTICAL_SPEED, VERTICAL_DURATION).join()
        else:
            # Unknown action — hover in place
            self.client.moveByVelocityBodyFrameAsync(0, 0, 0, 1.0).join()
