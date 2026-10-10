"""UniGoal agent adapted for ASAR drone benchmark.

Design principles:
- Reuse SemExp-proven drone mapping + FMM local planning stack
- Keep UniGoal's text-goal advantage via GoalGraph parsing
- Use simplified graph overlap (node-only, no edges)
- Keep detection module independent (YOLO-World on 4 cameras)
"""

import os
import sys
import time
from typing import Dict, List

import airsim
import numpy as np

# Ensure source/ is importable for BaseAgent
_SOURCE_DIR = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "source")
_SOURCE_DIR = os.path.abspath(_SOURCE_DIR)
_PROJECT_DIR = os.path.abspath(os.path.join(_SOURCE_DIR, os.pardir))
if _SOURCE_DIR not in sys.path:
    sys.path.insert(0, _SOURCE_DIR)

from agent_base import BaseAgent  # noqa: E402

from baselines_adaption.semexp.semantic_map import SemanticMap
from .detection import Detector, DEFAULT_VOCAB, MAX_DEPTH
from .frontier import GraphGuidedFrontierSelector
from .goal_graph import parse_task_prompt
from .graph_matcher import compute_overlap, OverlapResult
from .llm_client import DashScopeLLMClient, DashScopeVLMClient
from .scene_graph import SceneGraph

# ==================== Configuration ====================
GRID_SIZE = 400
CELL_SIZE = 5.0
FLIGHT_SPEED = 5.0
FOV_W = 90
SAMPLE_STEP = 4
OBS_Z_MARGIN = 10.0
REPLAN_DIST = 10.0
STEP_INTERVAL = 2.0

GRAPH_UPDATE_EVERY = 20         # overlap recompute interval
VLM_HINT_EVERY = 40             # low-frequency VLM caption interval

CAMERAS = [
    ("front_camera", 0.0),
    ("right_camera", 90.0),
    ("back_camera", 180.0),
    ("left_camera", -90.0),
]


class UniGoalAgent(BaseAgent):
    """UniGoal-style ASAR agent with simplified graph-guided exploration."""

    def __init__(self, client, task, vocab=None):
        super().__init__(client, task)

        self.set_trajectory_interval(10)

        self.vocab = vocab if vocab is not None else list(DEFAULT_VOCAB)
        self.person_category_id = (
            self.vocab.index("person") if "person" in self.vocab else 0
        )

        self._task_id = str(self.get_task_field("task_id", "unknown"))
        self._task_prompt = self.get_task_prompt(default="")

        # 1) Detection module (independent)
        self.detector = Detector(
            vocab=self.vocab,
            project_dir=_PROJECT_DIR,
            task_id=self._task_id,
            task_prompt=self._task_prompt,
        )

        # 2) LLM/VLM clients
        self.llm = DashScopeLLMClient(model="qwen3-max")
        self.vlm = DashScopeVLMClient(model="qwen3.5-plus")

        # 3) Goal graph from text prompt
        self.goal_graph = parse_task_prompt(self.llm, self._task_prompt, add_related=True)

        # 4) Scene graph from detections
        self.scene_graph = SceneGraph()

        # 5) Map + planner
        state = client.getMultirotorState()
        start_pos = state.kinematics_estimated.position
        self.flight_z = start_pos.z_val

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

        self.goal_selector = GraphGuidedFrontierSelector(
            grid_size=GRID_SIZE,
            cell_size=CELL_SIZE,
            replan_dist=REPLAN_DIST,
        )

        # Runtime state
        self.step_count = 0
        self._last_step_time = 0.0
        self._missing_pose_cameras = set()
        self._reported_target_positions = set()
        self._known_labels = set()
        self._new_object_type_detected = False
        self._fmm_needs_replan = False

        self.current_goal_world = None
        self.current_global_goal = None

        self._overlap_result = OverlapResult()

        # Build image request list once
        self._image_requests = []
        for cam_name, _ in CAMERAS:
            self._image_requests.append(
                airsim.ImageRequest(cam_name, airsim.ImageType.Scene, False, False)
            )
            self._image_requests.append(
                airsim.ImageRequest(cam_name, airsim.ImageType.DepthPlanar, True)
            )

        print("[UniGoal] Init done.")
        print(f"[UniGoal] Task prompt: {self._task_prompt}")
        print(f"[UniGoal] Goal graph targets={self.goal_graph.target_objects} "
              f"hints={self.goal_graph.spatial_hints} related={len(self.goal_graph.related_objects)}")

    def run_step(self, current_time):
        if current_time - self._last_step_time < STEP_INTERVAL:
            return
        self._last_step_time = current_time
        self.step_count += 1

        # A. state
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        orientation = state.kinematics_estimated.orientation
        agent_gx, agent_gy = self.semantic_map.world_to_grid(pos.x_val, pos.y_val)

        # B. batched 4-cam RGBD
        responses = self.client.simGetImages(self._image_requests)
        if len(responses) < len(self._image_requests):
            print(f"[UniGoal] Image capture incomplete: got {len(responses)}/{len(self._image_requests)}")
            return

        camera_data = self._parse_camera_responses(responses)

        # C. per-camera detection + depth mapping
        self.semantic_map.update_agent_position(pos.x_val, pos.y_val)

        front_rgb_for_vlm = None
        for (cam_name, yaw_offset, img_rgb, rgb_shape, depth_img, cam_pos, cam_ori) in camera_data:
            use_response_pose = (cam_pos is not None and cam_ori is not None)
            proj_pos = cam_pos if use_response_pose else pos
            proj_ori = cam_ori if use_response_pose else orientation
            proj_yaw_offset = 0.0 if use_response_pose else yaw_offset

            if (not use_response_pose) and cam_name not in self._missing_pose_cameras:
                self._missing_pose_cameras.add(cam_name)
                print(f"[UniGoal] Warning: {cam_name} missing camera pose; using vehicle pose fallback")

            if cam_name == "front_camera" and img_rgb is not None:
                front_rgb_for_vlm = img_rgb

            if img_rgb is not None and depth_img is not None:
                detections = self.detector.run_detection(
                    img_rgb,
                    depth_img,
                    rgb_shape,
                    proj_pos,
                    proj_ori,
                    camera_yaw_offset=proj_yaw_offset,
                    camera_name=cam_name,
                    step_count=self.step_count,
                )
                self._consume_detections(detections)

            if depth_img is not None:
                self.semantic_map.update_from_depth(
                    depth_img,
                    proj_pos,
                    proj_ori,
                    camera_yaw_offset=proj_yaw_offset,
                )

        # D. periodic graph overlap update
        if (self.step_count % GRAPH_UPDATE_EVERY == 0) or self._new_object_type_detected:
            self._overlap_result = compute_overlap(self.goal_graph, self.scene_graph)
            self._new_object_type_detected = False
            print("[UniGoal-Graph] coverage={:.2f} found={} missing={}".format(
                self._overlap_result.coverage,
                sorted(list(self._overlap_result.found_objects)),
                sorted(list(self._overlap_result.missing_objects))[:6],
            ))

        # E. low-frequency VLM hint matching (front camera)
        if (
            front_rgb_for_vlm is not None
            and self.goal_graph.spatial_hints
            and self.step_count % VLM_HINT_EVERY == 0
        ):
            self._update_spatial_hint_bonus(front_rgb_for_vlm, agent_gx, agent_gy)

        # F. global goal selection
        force_replan = self.goal_selector.needs_replan(agent_gx, agent_gy)
        if force_replan or self._fmm_needs_replan:
            self._fmm_needs_replan = False
            global_goal = self.goal_selector.select_goal(
                self.semantic_map,
                agent_gx,
                agent_gy,
                person_category_id=self.person_category_id,
                overlap_result=self._overlap_result,
            )
            if global_goal is None:
                print("[UniGoal] No frontiers left. Finishing mission.")
                self.finish_mission()
                return
            self.current_global_goal = global_goal

        # G. local FMM planning + fly
        if self.current_global_goal is not None:
            stg, stop, replan = self.goal_selector.plan_short_term_goal(
                self.semantic_map, agent_gx, agent_gy, self.current_global_goal
            )
            if stop:
                gg = self.current_global_goal
                self.semantic_map.mark_visited(gg[0], gg[1], radius_cells=3)
                self._fmm_needs_replan = True
            elif replan:
                self._fmm_needs_replan = True

            wx, wy = self.semantic_map.grid_to_world(int(stg[0]), int(stg[1]))
            self.current_goal_world = (wx, wy)
            self.client.moveToPositionAsync(wx, wy, self.flight_z, FLIGHT_SPEED)

        # H. periodic logs
        if self.step_count % 10 == 0:
            explored_pct = (self.semantic_map.get_explored_mask().sum() / self.semantic_map.grid_size ** 2 * 100)
            goal_str = f"grid={self.current_global_goal}" if self.current_global_goal else "None"
            stg_str = (
                f"({self.current_goal_world[0]:.0f},{self.current_goal_world[1]:.0f})"
                if self.current_goal_world else "None"
            )
            print(f"[UniGoal] Step {self.step_count} | "
                  f"pos=({pos.x_val:.0f},{pos.y_val:.0f},z={pos.z_val:.0f}) | "
                  f"goal={goal_str} stg={stg_str} | "
                  f"explored={explored_pct:.1f}% | "
                  f"coverage={self._overlap_result.coverage:.2f} | "
                  f"scene_labels={sorted(list(self.scene_graph.get_labels()))[:8]} | "
                  f"clues={len(self._found_clues)} targets={len(self._reported_positions)}")

            self.record_position(
                self.step_count,
                {"x": pos.x_val, "y": pos.y_val, "z": pos.z_val},
                extra={
                    "explored_pct": round(explored_pct, 2),
                    "goal": goal_str,
                    "graph_coverage": round(self._overlap_result.coverage, 3),
                },
            )

    def _parse_camera_responses(self, responses):
        """Parse flat [rgb,depth,rgb,depth,...] list into per-camera tuples."""
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
                img_rgb = np.frombuffer(rgb_resp.image_data_uint8, dtype=np.uint8).reshape(
                    rgb_resp.height, rgb_resp.width, 3
                )
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

            result.append((cam_name, yaw_offset, img_rgb, rgb_shape, depth_img, cam_pos, cam_ori))

        return result

    def _consume_detections(self, detections: List[Dict]):
        """Apply detection outputs to reporting + semantic map + scene graph."""
        for det in detections:
            cls_id = int(det["cls_id"])
            cls_name = str(det["cls_name"])
            world_pos = det["world_pos"]
            conf = float(det["confidence"])

            # 1) Report clue
            self.report_clue(cls_name, world_pos)

            # 2) Semantic map update
            self.semantic_map.update_semantic(cls_id, world_pos["x"], world_pos["y"])

            # 3) Scene graph update
            gx, gy = self.semantic_map.world_to_grid(world_pos["x"], world_pos["y"])
            self.scene_graph.add_detection(
                cls_name,
                world_pos["x"],
                world_pos["y"],
                confidence=conf,
                step=self.step_count,
                grid_x=gx,
                grid_y=gy,
            )
            if cls_name.lower() not in self._known_labels:
                self._known_labels.add(cls_name.lower())
                self._new_object_type_detected = True

            # 4) Report person targets
            if cls_name in ("person", "survivor", "body"):
                pos_key = (round(world_pos["x"], 0), round(world_pos["y"], 0))
                if pos_key not in self._reported_target_positions:
                    self._reported_target_positions.add(pos_key)
                    self.report_target(world_pos)

    def _update_spatial_hint_bonus(self, front_rgb, agent_gx: int, agent_gy: int):
        """Low-frequency VLM caption; if hint matched, add frontier hint anchor."""
        hints = self.goal_graph.spatial_hints
        if not hints:
            return
        hint_text = ", ".join(hints)
        prompt = (
            "You are analyzing a drone front-view image in outdoor SAR. "
            "Does this view indicate terrain/location hints from this list: "
            f"[{hint_text}]? "
            "Answer in JSON only: {\"matched\": true/false, \"hints\": [..], \"reason\": \"...\"}"
        )
        try:
            out = self.vlm.caption_from_bgr(front_rgb, prompt)
            txt = out.lower()
            matched = any(h.lower() in txt for h in hints)
            if matched:
                self.goal_selector.add_hint_position(agent_gx, agent_gy)
                print(f"[UniGoal-VLM] hint matched, anchor added at ({agent_gx},{agent_gy})")
            else:
                print("[UniGoal-VLM] no spatial hint matched")
        except Exception as e:
            print(f"[UniGoal-VLM] caption failed: {e}")
