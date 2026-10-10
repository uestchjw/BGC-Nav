"""
Pure-MLLM Baseline Agent for the ASAR benchmark.

Decision module: forward RGB image + task prompt -> MLLM (qwen3.5-plus
via Dashscope API) decides one of 7 actions. No additional mechanisms
(no map, no planner, no memory beyond the current frame).
Detection module: shared 4-camera YOLO+Depth (same as SemExp).
"""

import sys
import os
import json
import base64
import cv2
import numpy as np
import airsim
import dashscope
from http import HTTPStatus
from dashscope import MultiModalConversation

_SOURCE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "source"))
if _SOURCE_DIR not in sys.path:
    sys.path.insert(0, _SOURCE_DIR)

from agent_base import BaseAgent
from .detection import DetectionModule

# dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY", "")
#! H95
dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY", "")


# ── Configuration ──────────────────────────────────────────────────────
MODEL_NAME    = "qwen3.5-plus"
XY_VELOCITY   = 6.0   # m/s horizontal
Z_VELOCITY    = 3.0   # m/s vertical
STEP_DURATION = 5.0   # seconds per action
STEP_INTERVAL = 2.0   # minimum seconds between run_step calls


class MLLMAgent(BaseAgent):

    def __init__(self, client, task):
        super().__init__(client, task)
        self.set_trajectory_interval(10)

        self.model = MODEL_NAME
        self.detector = DetectionModule(self, agent_tag="MLLM")

        self.step_count = 0
        self._last_step_time = 0.0
        self._task_id = str(self.get_task_field("task_id", "unknown"))

        state = client.getMultirotorState()
        self._start_pos = state.kinematics_estimated.position
        print(f"[MLLM] Initialised with model={self.model}")

    def run_step(self, current_time):
        if current_time - self._last_step_time < STEP_INTERVAL:
            return
        self._last_step_time = current_time
        self.step_count += 1

        # ── A. Perception: 4-camera YOLO detection ─────────────────────
        self.detector.capture_and_detect(self.step_count)

        # ── B. Capture front-camera RGB for MLLM ──────────────────────
        front_rgb = self._capture_front_rgb()
        if front_rgb is None:
            print("[MLLM] Front camera capture failed, hovering.")
            self.client.moveByVelocityBodyFrameAsync(0, 0, 0, 1.0).join()
            return

        # ── C. Build prompt & call MLLM ────────────────────────────────
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        rel_x = pos.x_val - self._start_pos.x_val
        rel_y = pos.y_val - self._start_pos.y_val
        rel_alt = -(pos.z_val - self._start_pos.z_val)

        task_desc = self.get_task_prompt(default="Search for victims and clues.")
        found_clues = ", ".join(self.get_clues()) if self.get_clues() else "None"

        prompt = (
            f"You are a Search and Rescue drone.\n"
            f"Mission: {task_desc}\n\n"
            f"Current position (relative to start): "
            f"Forward={rel_x:.1f}m, Right={rel_y:.1f}m, Altitude={rel_alt:.1f}m\n"
            f"Clues found so far: {found_clues}\n\n"
            f"The image shows your FRONT camera view.\n"
            f"Choose ONE action from: forward, backward, left, right, up, down, finish\n\n"
            f"Reply in strict JSON:\n"
            f'{{"thought": "brief reasoning", "action": "one_of_the_7_actions"}}'
        )

        b64_img = base64.b64encode(
            cv2.imencode(".jpg", front_rgb)[1]).decode("utf-8")

        messages = [{
            "role": "user",
            "content": [
                {"image": f"data:image/jpeg;base64,{b64_img}"},
                {"text": prompt},
            ]
        }]

        action = self._call_mllm(messages)

        # ── D. Execute action ──────────────────────────────────────────
        self._execute_action(action)

        # ── E. Periodic status log ─────────────────────────────────────
        if self.step_count % 10 == 0:
            print(f"[MLLM] Step {self.step_count} | "
                  f"pos=({pos.x_val:.0f},{pos.y_val:.0f},z={pos.z_val:.0f}) | "
                  f"action={action} | "
                  f"clues={len(self._found_clues)} "
                  f"targets={len(self._reported_positions)}")
            self.record_position(self.step_count, {
                "x": pos.x_val, "y": pos.y_val, "z": pos.z_val,
            }, extra={"action": action})

    # ── Helpers ────────────────────────────────────────────────────────

    def _capture_front_rgb(self):
        responses = self.client.simGetImages([
            airsim.ImageRequest("front_camera", airsim.ImageType.Scene, False, False)
        ])
        if (not responses or responses[0].width <= 0
                or len(responses[0].image_data_uint8) == 0):
            return None
        r = responses[0]
        return np.frombuffer(r.image_data_uint8, dtype=np.uint8).reshape(
            r.height, r.width, 3)

    def _call_mllm(self, messages):
        try:
            response = MultiModalConversation.call(
                model=self.model, messages=messages, enable_thinking=False)
            if response.status_code == HTTPStatus.OK:
                text = response.output.choices[0].message.content[0]["text"]
                return self._parse_action(text)
            else:
                print(f"[MLLM] API error: {response.code} - {response.message}")
                return "forward"
        except Exception as e:
            print(f"[MLLM] API exception: {e}")
            return "forward"

    def _parse_action(self, text):
        try:
            clean = text.replace("```json", "").replace("```", "").strip()
            decision = json.loads(clean)
            action = decision.get("action", "forward").strip().lower()
            thought = decision.get("thought", "")
            if thought:
                print(f"\033[94m[MLLM] Thought: {thought}\033[0m")
            valid = {"forward", "backward", "left", "right", "up", "down", "finish"}
            return action if action in valid else "forward"
        except json.JSONDecodeError:
            print(f"[MLLM] JSON parse failed: {text[:200]}")
            return "forward"

    def _execute_action(self, action):
        if action == "forward":
            self.client.moveByVelocityBodyFrameAsync(
                XY_VELOCITY, 0, 0, STEP_DURATION).join()
        elif action == "backward":
            self.client.moveByVelocityBodyFrameAsync(
                -XY_VELOCITY, 0, 0, STEP_DURATION).join()
        elif action == "left":
            self.client.moveByVelocityBodyFrameAsync(
                0, -XY_VELOCITY, 0, STEP_DURATION).join()
        elif action == "right":
            self.client.moveByVelocityBodyFrameAsync(
                0, XY_VELOCITY, 0, STEP_DURATION).join()
        elif action == "up":
            self.client.moveByVelocityBodyFrameAsync(
                0, 0, -Z_VELOCITY, STEP_DURATION).join()
        elif action == "down":
            self.client.moveByVelocityBodyFrameAsync(
                0, 0, Z_VELOCITY, STEP_DURATION).join()
        elif action == "finish":
            self.finish_mission()
