"""
Random Baseline Agent for the ASAR benchmark.

Decision module: uniformly random action selection from 7 actions
(forward / backward / left / right / up / down / finish).
Detection module: shared 4-camera YOLO+Depth (same as semexp).
"""

import sys
import os
import random

import airsim

_SOURCE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "source"))
if _SOURCE_DIR not in sys.path:
    sys.path.insert(0, _SOURCE_DIR)

from agent_base import BaseAgent
from .detection import DetectionModule

# ── Configuration ──────────────────────────────────────────────────────
XY_VELOCITY    = 6.0   # m/s horizontal
Z_VELOCITY     = 3.0   # m/s vertical
STEP_DURATION  = 5.0   # seconds per action
STEP_INTERVAL  = 2.0   # minimum seconds between run_step calls

ACTIONS = ["forward", "backward", "left", "right", "up", "down", "finish"]


class RandomAgent(BaseAgent):

    def __init__(self, client, task):
        super().__init__(client, task)
        self.set_trajectory_interval(10)

        self.detector = DetectionModule(self, agent_tag="Random")
        self.step_count = 0
        self._last_step_time = 0.0

        print("[Random] Agent initialised. Actions:", ACTIONS)

    def run_step(self, current_time):
        if current_time - self._last_step_time < STEP_INTERVAL:
            return
        self._last_step_time = current_time
        self.step_count += 1

        # A. Perception: 4-camera detection
        self.detector.capture_and_detect(self.step_count)

        # B. Random action selection
        action = random.choice(ACTIONS)

        # C. Execute action (body frame: X=forward, Y=right, Z=down)
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
            return

        # D. Periodic status log
        if self.step_count % 10 == 0:
            pos = self.client.getMultirotorState().kinematics_estimated.position
            print(f"[Random] Step {self.step_count} | "
                  f"pos=({pos.x_val:.0f},{pos.y_val:.0f},z={pos.z_val:.0f}) | "
                  f"action={action} | "
                  f"clues={len(self._found_clues)} "
                  f"targets={len(self._reported_positions)}")
            self.record_position(self.step_count, {
                "x": pos.x_val, "y": pos.y_val, "z": pos.z_val,
            }, extra={"action": action})
