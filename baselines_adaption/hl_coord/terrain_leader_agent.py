"""High-altitude "leader" agent: terrain-relevance grounding (idea doc §4.2).

Flies a fixed boustrophedon (lawnmower) sweep over the task area at a much
higher altitude than the low-altitude follower, and periodically:
  1. captures a straight-down (BEV) image from ``bev_camera``,
  2. asks a VLM whether/how strongly the view matches the terrain keywords
     extracted from the task prompt (``spatial_hints``, reusing UniGoal's
     prompt parser — see ``baselines_adaption/unigoal/goal_graph.py``),
  3. writes the resulting score into the shared ``TerrainField``.

This agent does no object detection and reports no clues/targets itself —
in this MVP its only job is to inform the follower's attraction map via the
shared field; the follower does all the actual searching.
"""
import json
import math
import os
import re
import sys
import time

import threading

import airsim
import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - cv2 always present in esar_env
    cv2 = None

_SOURCE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "source"))
if _SOURCE_DIR not in sys.path:
    sys.path.insert(0, _SOURCE_DIR)

from agent_base import BaseAgent  # noqa: E402

from baselines_adaption.unigoal.goal_graph import parse_task_prompt
from baselines_adaption.unigoal.llm_client import DashScopeLLMClient, DashScopeVLMClient
from baselines_adaption.hl_coord import hl_config as C

STEP_INTERVAL = 2.0          # seconds between run_step executions
VLM_SCORE_INTERVAL = C.LEADER_VLM_INTERVAL  # call the VLM every N steps
WAYPOINT_ARRIVAL_M = 15.0    # advance to next sweep waypoint within this radius
LEADER_SPEED = C.LEADER_SPEED  # m/s — flies faster than the follower, less obstacle risk up high


class TerrainLeaderAgent(BaseAgent):

    def __init__(self, client, task, terrain_field,
                 sweep_half_size_m: float = None,
                 row_spacing_m: float = None):
        super().__init__(client, task)
        if sweep_half_size_m is None:
            sweep_half_size_m = C.LEADER_SWEEP_HALF_M
        if row_spacing_m is None:
            row_spacing_m = C.LEADER_ROW_SPACING_M
        self.set_trajectory_interval(10)
        self.terrain_field = terrain_field

        # The platform already teleports this vehicle to its survey altitude
        # before constructing the agent (see main_platform_high_low.py's
        # z_offsets[LEADER]) — same convention as FBEAgent/ApexAgent, which
        # also just adopt whatever altitude they were placed at. Applying an
        # extra offset here on top of that was a real bug: it silently
        # doubled the leader's height above the follower's start point.
        state = client.getMultirotorState()
        start_pos = state.kinematics_estimated.position
        self.flight_z = start_pos.z_val

        # bev_camera's static Pitch in settings_high_low.json turned out to be
        # unreliable — empirically verified (see docs/ notes) that the same
        # Pitch value behaves differently there than through this runtime
        # API, and the static config pointed the camera at the SKY instead
        # of straight down. Fix: set the camera orientation explicitly at
        # runtime, which was verified (by visual inspection of captured
        # frames) to reliably look straight down at Pitch=-90 here.
        client.simSetCameraPose(
            "bev_camera",
            airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(np.deg2rad(-90), 0, 0)),
        )

        self.waypoints = self._build_sweep(
            start_pos.x_val, start_pos.y_val, self.flight_z,
            sweep_half_size_m, row_spacing_m, mode=C.LEADER_SWEEP_MODE,
        )
        self._wp_idx = 0

        self._task_prompt = self.get_task_prompt(default="")
        self.llm = DashScopeLLMClient(model="qwen3-max")
        self.vlm = DashScopeVLMClient(model="qwen3.5-plus")
        self.goal_graph = parse_task_prompt(self.llm, self._task_prompt, add_related=False)
        self.hints = self.goal_graph.spatial_hints

        self._image_requests = [
            airsim.ImageRequest("bev_camera", airsim.ImageType.Scene, False, False),
        ]

        self.step_count = 0
        self._last_step_time = 0.0

        # Option A: grid the VLM output + project the frame footprint.
        self._grid = max(1, int(C.LEADER_VLM_GRID))
        fov_deg = 90.0
        self._footprint_m = (C.LEADER_FOOTPRINT_M if C.LEADER_FOOTPRINT_M > 0.0
                             else 2.0 * C.LEADER_ALT_OFFSET_M * math.tan(math.radians(fov_deg / 2.0)))
        if self._grid > 1:
            print(f"[TerrainLeader] Grid mode {self._grid}x{self._grid}, "
                  f"footprint~{self._footprint_m:.0f}m/frame")

        # The VLM call (~1s, network, occasionally rate-limited) used to run
        # synchronously inside run_step, stalling the shared control loop and
        # costing the *follower* ~1 step every VLM_SCORE_INTERVAL — measured
        # as a 12-15% drop in follower steps/path vs solo APEX. Run it on a
        # background thread instead: run_step just grabs the BEV frame (cheap
        # local RPC) and hands it off; the field is updated whenever the call
        # returns. At most one call in flight — if the VLM is slow, later
        # score opportunities are skipped rather than queued.
        self._vlm_inflight = False
        self._vlm_start = 0.0
        self._vlm_gen = 0
        self._vlm_calls = 0
        self._vlm_ok = 0

        print(f"[TerrainLeader] Init at z={self.flight_z:.0f} "
              f"({len(self.waypoints)} sweep waypoints), hints={self.hints}")
        if not self.hints:
            print("[TerrainLeader] No spatial hints in prompt — will sweep "
                  "without VLM terrain scoring (nothing to match against).")

    @staticmethod
    def _build_sweep(cx, cy, z, half_size, row_spacing, mode="boustrophedon"):
        """Waypoints covering a square centred on (cx, cy).

        boustrophedon - lawnmower rows, alternating direction so consecutive
                        legs are adjacent (no long transit flights).
        spiral        - outward square spiral starting AT the centre, so the
                        follower's immediate vicinity is scored first and the
                        terrain field is useful while the follower can still
                        act on it (the lawnmower spends its first minutes at
                        one map edge, far from the follower).
        """
        if mode == "spiral":
            waypoints = [(cx, cy, z)]
            step = row_spacing
            x, y = cx, cy
            leg = 1
            # E, N, W, S repeating; leg length grows every two turns
            dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            di = 0
            while step * (leg // 2 + 1) <= half_size + 1e-6:
                dx, dy = dirs[di % 4]
                for _ in range(leg):
                    x += dx * step
                    y += dy * step
                    if abs(x - cx) <= half_size + 1e-6 and abs(y - cy) <= half_size + 1e-6:
                        waypoints.append((x, y, z))
                di += 1
                if di % 2 == 0:
                    leg += 1
            return waypoints

        rows = np.arange(-half_size, half_size + 1e-6, row_spacing)
        waypoints = []
        for i, y in enumerate(rows):
            x_start, x_end = (-half_size, half_size) if i % 2 == 0 else (half_size, -half_size)
            waypoints.append((cx + x_start, cy + y, z))
            waypoints.append((cx + x_end, cy + y, z))
        return waypoints

    def run_step(self, current_time):
        if current_time - self._last_step_time < STEP_INTERVAL:
            return
        self._last_step_time = current_time
        self.step_count += 1

        if self._wp_idx >= len(self.waypoints):
            self.finish_mission()
            return

        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        q = state.kinematics_estimated.orientation
        yaw = math.atan2(2.0 * (q.w_val * q.z_val + q.x_val * q.y_val),
                         1.0 - 2.0 * (q.y_val * q.y_val + q.z_val * q.z_val))

        target = self.waypoints[self._wp_idx]
        dist = float(np.linalg.norm([pos.x_val - target[0], pos.y_val - target[1]]))
        if dist < WAYPOINT_ARRIVAL_M:
            self._wp_idx += 1
            if self._wp_idx >= len(self.waypoints):
                print("[TerrainLeader] Sweep complete. Finishing mission.")
                self.finish_mission()
                return
            target = self.waypoints[self._wp_idx]

        self.client.moveToPositionAsync(target[0], target[1], target[2], LEADER_SPEED)

        if self.hints and self.step_count % VLM_SCORE_INTERVAL == 0:
            self._score_terrain(pos, yaw)

        if self.step_count % 10 == 0:
            print(f"[TerrainLeader] Step {self.step_count} | "
                  f"pos=({pos.x_val:.0f},{pos.y_val:.0f},z={pos.z_val:.0f}) | "
                  f"wp={self._wp_idx}/{len(self.waypoints)} | "
                  f"field_size={len(self.terrain_field)}")
            self.record_position(self.step_count, {
                "x": pos.x_val, "y": pos.y_val, "z": pos.z_val,
            }, extra={"wp_idx": self._wp_idx})

    def _score_terrain(self, pos, yaw):
        """Grab the BEV frame now (cheap) and score it on a background
        thread so the shared control loop never blocks on the VLM."""
        if self._vlm_inflight:
            if time.time() - self._vlm_start < C.LEADER_VLM_TIMEOUT_S:
                return  # a call is genuinely still running — skip this chance
            # the in-flight call has hung past the timeout — abandon it (the
            # zombie thread's finally won't clobber state thanks to _vlm_gen)
            print(f"[TerrainLeader] VLM call timed out "
                  f"({time.time() - self._vlm_start:.0f}s) — abandoning, continuing")
        responses = self.client.simGetImages(self._image_requests)
        if not responses or responses[0].width <= 0 or len(responses[0].image_data_uint8) == 0:
            return
        resp = responses[0]
        img = np.frombuffer(resp.image_data_uint8, dtype=np.uint8).reshape(
            resp.height, resp.width, 3).copy()
        px, py = float(pos.x_val), float(pos.y_val)

        self._vlm_gen += 1
        self._vlm_inflight = True
        self._vlm_start = time.time()
        t = threading.Thread(target=self._vlm_worker,
                             args=(img, px, py, float(yaw), self._vlm_gen), daemon=True)
        t.start()

    # ------------------------------------------------------------------
    def _project_grid_cell(self, px, py, yaw, fi, fj):
        """World (x, y) of a point at fractional grid coords (fi, fj) — cell
        (row i, col j) centre is (i+0.5, j+0.5), its corners are integer coords.
        Nadir BEV frame at (px, py), heading ``yaw`` (NED, +x=North toward
        +y=East). Image convention: 'up' (row decreasing) = vehicle forward,
        'right' (col increasing) = vehicle right — for the runtime
        ``simSetCameraPose(Pitch=-90)`` mounting."""
        G = self._grid
        nx = fj / G - 0.5                 # image right, in [-0.5, 0.5]
        ny = fi / G - 0.5                 # image down,  in [-0.5, 0.5]
        fwd = (-ny) * self._footprint_m   # metres forward of the drone
        rgt = (nx) * self._footprint_m    # metres to the drone's right
        c, s = math.cos(yaw), math.sin(yaw)
        wx = fwd * c + rgt * (-s)
        wy = fwd * s + rgt * (c)
        return px + wx, py + wy

    def _paint_cell(self, px, py, yaw, i, j, score):
        """Write ``score`` into every terrain-field 50m cell whose centre falls
        inside grid cell (i, j)'s world footprint (axis-aligned bbox of its 4
        rotated corners — slight corner over-paint is fine for a coarse prior)."""
        corners = [self._project_grid_cell(px, py, yaw, i + di, j + dj)
                   for di in (0, 1) for dj in (0, 1)]
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        cs = self.terrain_field.cell_size
        x0 = math.floor(min(xs) / cs) * cs
        y0 = math.floor(min(ys) / cs) * cs
        x = x0
        while x <= max(xs):
            y = y0
            while y <= max(ys):
                self.terrain_field.update(x + cs / 2.0, y + cs / 2.0, score)
                y += cs
            x += cs

    def _draw_grid(self, img):
        if cv2 is None or self._grid <= 1:
            return img
        h, w = img.shape[:2]
        out = img.copy()
        for k in range(1, self._grid):
            x = int(round(w * k / self._grid))
            y = int(round(h * k / self._grid))
            cv2.line(out, (x, 0), (x, h - 1), (0, 255, 0), 2)
            cv2.line(out, (0, y), (w - 1, y), (0, 255, 0), 2)
        return out

    def _vlm_worker(self, img, px, py, yaw, gen):
        try:
            self._vlm_calls += 1
            hint_text = ", ".join(self.hints)
            if self._grid <= 1:
                self._score_point(img, px, py, hint_text)
            else:
                self._score_grid(img, px, py, yaw, hint_text)
        finally:
            # only clear the flag if we're still the current call — a call that
            # was abandoned as timed-out must not reopen the slot it lost.
            if gen == self._vlm_gen:
                self._vlm_inflight = False

    def _score_point(self, img, px, py, hint_text):
        """Original behaviour: one score for the whole frame -> the cell under
        the drone."""
        prompt = (
            "You are analyzing a straight-down (bird's-eye) aerial photo taken by a "
            "high-altitude reconnaissance drone during a search-and-rescue mission. "
            f"Rate how strongly this view matches ANY of these terrain/location "
            f"keywords: [{hint_text}]. "
            "Answer in JSON only, no other text: "
            "{\"score\": <float 0.0-1.0>, \"best_match\": \"<keyword or none>\", "
            "\"reason\": \"...\"}"
        )
        try:
            parsed = _extract_json(self.vlm.caption_from_bgr(img, prompt))
        except Exception as e:
            print(f"[TerrainLeader] VLM scoring failed: {e}")
            return
        score = max(0.0, min(1.0, float(parsed.get("score", 0.0)))) if parsed else 0.0
        self._vlm_ok += 1
        self.terrain_field.update(px, py, score)
        if score > 0.0:
            best = parsed.get("best_match", "?") if parsed else "?"
            print(f"[TerrainLeader] Terrain score={score:.2f} best_match={best} "
                  f"at ({px:.0f},{py:.0f})")

    def _score_grid(self, img, px, py, yaw, hint_text):
        """Option A: score a GxG grid over the frame, project each cell to its
        world (x, y), write each into the terrain field."""
        G = self._grid
        prompt = (
            "You are analyzing a straight-down (bird's-eye) aerial photo from a "
            "high-altitude search-and-rescue drone. The image is divided into a "
            f"{G}x{G} grid (row 0 = top, col 0 = left; green lines mark the cells). "
            f"For EACH of the {G*G} cells, rate 0.0-1.0 how strongly the GROUND in "
            f"that cell matches ANY of these terrain/location keywords: [{hint_text}]. "
            "Use 0.0 for cells that clearly do not match. "
            "Answer in JSON only, no other text: "
            f"{{\"grid\": [{G} rows, each a list of {G} floats], "
            "\"best_match\": \"<keyword or none>\"}}"
        )
        img_send = self._draw_grid(img) if C.LEADER_GRID_DRAW else img
        try:
            parsed = _extract_json(self.vlm.caption_from_bgr(img_send, prompt))
        except Exception as e:
            print(f"[TerrainLeader] VLM grid scoring failed: {e}")
            return
        grid = _coerce_grid(parsed.get("grid") if parsed else None, G)
        if grid is None:
            # fall back to a single score for the frame so the call isn't wasted
            s = 0.0
            if parsed:
                try:
                    s = max(0.0, min(1.0, float(parsed.get("score", 0.0))))
                except (TypeError, ValueError):
                    s = 0.0
            if s > 0.0:
                self.terrain_field.update(px, py, s)
            print(f"[TerrainLeader] grid parse failed, wrote centre score={s:.2f}")
            return

        self._vlm_ok += 1

        flat = [grid[i][j] for i in range(G) for j in range(G)]
        contrast = max(flat) - min(flat)
        if C.LEADER_GRID_MIN_CONTRAST > 0.0 and contrast < C.LEADER_GRID_MIN_CONTRAST:
            # uniform frame -> no directional information; don't flood the field
            print(f"[TerrainLeader] grid@({px:.0f},{py:.0f}) skipped "
                  f"(contrast {contrast:.2f} < {C.LEADER_GRID_MIN_CONTRAST:.2f}, "
                  f"mean {sum(flat)/len(flat):.2f})")
            return

        nonzero = 0
        gmax = 0.0
        for i in range(G):
            for j in range(G):
                sc = grid[i][j]
                if sc <= 0.0:
                    continue
                self._paint_cell(px, py, yaw, i, j, sc)
                nonzero += 1
                gmax = max(gmax, sc)
        if C.LEADER_GRID_KEEP_CENTER:
            # centre-of-frame score straight under the drone (rotation-proof)
            ci = cj = (G - 1) / 2.0
            c0, c1 = int(math.floor(ci)), int(math.ceil(ci))
            centre = max(grid[c0][c0], grid[c0][c1], grid[c1][c0], grid[c1][c1])
            if centre > 0.0:
                self.terrain_field.update(px, py, centre)
        if nonzero:
            best = parsed.get("best_match", "?") if parsed else "?"
            print(f"[TerrainLeader] grid@({px:.0f},{py:.0f}) yaw={math.degrees(yaw):.0f} "
                  f"wrote {nonzero}/{G*G} cells, max={gmax:.2f} best_match={best} "
                  f"field_size={len(self.terrain_field)}")


def _coerce_grid(g, G):
    """Best-effort turn whatever the VLM returned into a GxG list of floats in
    [0,1], or None if it's unusable."""
    if g is None:
        return None
    # flat list of G*G -> reshape
    if isinstance(g, list) and g and not isinstance(g[0], list):
        if len(g) == G * G:
            g = [g[r * G:(r + 1) * G] for r in range(G)]
        else:
            return None
    if not isinstance(g, list) or len(g) != G:
        return None
    out = []
    for row in g:
        if not isinstance(row, list) or len(row) != G:
            return None
        cr = []
        for v in row:
            try:
                cr.append(max(0.0, min(1.0, float(v))))
            except (TypeError, ValueError):
                cr.append(0.0)
        out.append(cr)
    return out


def _extract_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None
