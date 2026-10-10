"""Low-altitude "follower" agent: injects the shared high-altitude terrain
field into APEX's decision maps (idea doc §4.3).

Fusion modes (``HL_FUSION_MODE``, see ``hl_config.py``):

  off      - no fusion. Plain ApexAgent. Isolates the cost of merely having
             the leader drone in the sim.

  floor    - the original mechanism: raise the attraction-map *floor* to
             ``min(cap, gamma * A_high(region(v)))`` over every region the
             leader flagged. Now gated by ``HL_FUSION_SCORE_GATE`` (drop the
             VLM's "0.1 / none" noise) and hard-capped by ``HL_FUSION_CAP``
             (stay below action_predict's KEY_THRESHOLD=0.9 so the follower
             never treats terrain-match as "target certain").

  gradient - directional. Instead of a broad uniform plateau (which beelines
             the follower into whatever obstacle-dense terrain matched the
             keyword), lay down a *ramp* in the attraction map that rises
             toward the bearing of the leader's best region and is strongest
             at the far edge of the follower's local window — a carrot ahead,
             not a destination on top of danger.

  explore  - safest. Touch the *exploration* map, not attraction: pull the
             exploration value down (make it look under-explored) in the
             wedge toward the leader's best region. This just breaks
             action_predict's tie — when all six actions score within 1e-2
             and plain APEX would fall back to spinning left forever, this
             tips the choice toward the leader's heading. Cannot ever trigger
             KEY_REWARD, so it can't cause the fusion-induced crashes seen
             with ``floor``.

  hybrid   - explore bias + a small capped gradient attraction nudge.

Cross-cutting safety / behaviour knobs:
  - never raises attraction on a cell already marked as an obstacle;
  - ``HL_STUCK_GATE``: only fuse while the follower is actually stalled, so
    tasks where plain APEX already makes progress aren't perturbed;
  - ``HL_DECAY_ETA``: Q-value-style decay of A_high for regions the follower
    has already covered, so it stops fixating on a wrong region.

Reuses ``ApexAgent`` unmodified via its ``_augment_attraction_map`` hook.
"""
import math

import numpy as np
import airsim

from baselines_adaption.apex.apex_agent import (
    ApexAgent, FOV_W, FLIGHT_SPEED, MOVE_DISTANCE, MAX_DEPTH,
)
from baselines_adaption.apex.voxel_maps import (
    MAP_SHAPE, GRID_SCALE, MAP_SIZE_WORLD, update_obstacle_from_depth,
    world_to_grid, grid_in_bounds,
)
from baselines_adaption.apex.action_predict import RATE_CENTER, DRONE_POS_LOCAL
from baselines_adaption.hl_coord import hl_config as C
from baselines_adaption.hl_coord.aerial_bank import UNOBSERVED_PRIOR as AB_UNOBSERVED_PRIOR

ATTRACTION_MAX = 1.0  # matches update_attraction's [0, 1] detection-score scale
# Half-window in metres for the directional falloff (local map is 200m => 100m).
_LOCAL_HALF_M = MAP_SIZE_WORLD[0] / 2.0


class FusedApexAgent(ApexAgent):
    """ApexAgent whose decision maps are biased by a shared TerrainField."""

    def __init__(self, client, task, terrain_field, fusion_gamma=None,
                 dense_grid=None, fusion_mode=None, queue=None, decomp=None):
        super().__init__(client, task)

        # Role-specialized instruction decomposition (low-altitude half): refine
        # the detector vocabulary and the scene-analysis prompt with the
        # target classes / clue types / target state meant for the searcher.
        if decomp:
            tgts = [str(x).strip() for x in decomp.get("low_targets", []) if str(x).strip()]
            if tgts:
                self.vocab = list(dict.fromkeys(tgts + list(self.vocab)))
                self.yolo.set_classes(self.vocab)
                self._yolo_default_names = list(self.vocab)
            aug = []
            if decomp.get("low_clues"):
                aug.append("Environmental clues to log even without a target: "
                           + ", ".join(decomp["low_clues"]) + ".")
            if decomp.get("target_state"):
                aug.append("Target state: " + decomp["target_state"] + ".")
            if aug:
                self._task_prompt = (self._task_prompt + " " + " ".join(aug)).strip()
            print(f"[FusedApex] decomp: vocab+={tgts} state='{decomp.get('target_state','')}'")

        self.terrain_field = terrain_field
        # densemap / aeroduo mode: a pre-built full-coverage grid (RelevanceGrid
        # for densemap = terrain relevance; probability grid M for aeroduo).
        self.dense_grid = dense_grid
        # conav mode: a priority-ordered list of [x, y, score] candidate target
        # world-coords from the (amortised) leader survey.
        self._conav_queue = [list(map(float, q)) for q in (queue or [])]
        self._conav_done = set()          # indices of exhausted queue entries
        self._conav_target = None         # current [x, y] being worked
        self._conav_state = "transit"
        self._conav_enter_step = None
        self._conav_transit_steps = 0
        # field-mode give-up state: if the lean's mean-bearing target sits near
        # the same place for too long (no detection happening), mask that
        # region out for the rest of the episode -- otherwise a stateless
        # per-step lean re-attracts to the same unproductive spot forever.
        self._field_exhausted = []        # [(x, y)] centres given up on, FIFO-capped
        self._field_dwell_center = None
        self._field_dwell_steps = 0
        self._field_giveup_total = 0      # cumulative give-up triggers this episode, uncapped
        # Lever 1 commit / anti-wander: anchored search disc once the drone is
        # inside the aerial hot zone (see hl_config FIELD_COMMIT_*)
        self._field_commit_center = None
        self._field_commit_steps = 0
        # Lever 2b in-zone nadir detection pass (see hl_config FIELD_NADIR_DET)
        self._nadir_requests = None
        if C.FIELD_NADIR_DET:
            self._nadir_requests = [
                airsim.ImageRequest("nadir_camera", airsim.ImageType.Scene,
                                    False, False),
                airsim.ImageRequest("nadir_camera", airsim.ImageType.DepthPlanar,
                                    True),
            ]

        # aeroduo state
        self._aeroduo_visited = []        # [(x, y)] centres of masked-out regions
        self._aeroduo_subgoal = None
        self._aeroduo_enter_step = None
        self._aeroduo_goto_target = None
        self._aeroduo_goto_yaw = 0.0
        self.fusion_gamma = C.FUSION_GAMMA if fusion_gamma is None else fusion_gamma
        self.fusion_mode = fusion_mode or C.FUSION_MODE

        gx = np.arange(MAP_SHAPE[0])
        gy = np.arange(MAP_SHAPE[1])
        self._gx_idx, self._gy_idx = np.meshgrid(gx, gy, indexing="ij")

        self._pos_hist = []          # recent (x, y) for stuck detection
        self._stuck_streak = 0       # consecutive stuck checks, for FIELD_UNWEDGE
        self._dbg = {}               # per-step lean diagnostics (HL_FIELD_DEBUG)
        self._field_restore = None   # pre-bias attraction ch0, restored after action selection
        self._terrain_attr = np.zeros(MAP_SHAPE, np.float32)  # attraction written by VLM terrain-class detections
        self._obs_cnt = None   # per belief-cell count of follower observations (negative evidence)
        self._climb_active = False
        self._climb_wedge = None
        self._goto_last = None        # (x, y) at the start of the last executed go-to step
        self._goto_fail = 0
        self._goto_suspend = 0
        self._desc_active = False
        self._desc_wedge = None
        self._home_z = None   # flight altitude at the first guided step (target of the descend-back bias)
        self._terrain_center = None  # map_center_world the shadow array is aligned to
        self._gate_terms = {w.strip().lower() for w in C.FIELD_GATE_TARGET_TERMS.split(',') if w.strip()}
        self._gate_stats = {"steps": 0, "stood_down": 0, "old_gate_only": 0, "brake_blocked": 0, "arrive_steps": 0}
        self._fusion_active_steps = 0
        self._last_best_key = None

        # ROI-mode state machine
        self._roi_state = "transit"     # transit | search
        self._roi_key = None
        self._roi_center = None
        self._roi_conf = 0.0
        self._roi_enter_step = None
        self._roi_reached = False
        self._roi_done_count = 0
        self._roi_transit_steps = 0

    # ------------------------------------------------------------------
    def _mark_field_exhausted(self, xy):
        """Give up on ``xy`` (see hl_config FIELD_GIVEUP*): append to the
        exhausted-centre blacklist, evicting the oldest entry first once
        FIELD_GIVEUP_MAX_EXHAUSTED is reached. Without a cap this list only
        grows for the rest of the episode, so a longer run (extra time
        budget, a slow/far task) keeps masking out more of the map even
        after most of it has nothing left worth un-masking."""
        self._field_exhausted.append(xy)
        self._field_giveup_total += 1
        overflow = len(self._field_exhausted) - C.FIELD_GIVEUP_MAX_EXHAUSTED
        if overflow > 0:
            del self._field_exhausted[:overflow]

    # ---- stand-down evidence that ignores VLM terrain-class detections ----
    def _is_target_like(self, name):
        words = [w.rstrip("s") if len(w) > 3 else w
                 for w in str(name).lower().replace("-", " ").split()]
        terms = self._gate_terms
        return any((w in terms) or (w + "s" in terms) for w in words)

    def _sync_terrain_shadow(self):
        """Keep the terrain-attraction shadow array aligned with the attraction
        map, replicating voxel_maps.shift_maps (np.roll + zero-fill on the
        newly exposed slab) whenever the map centre has moved."""
        if self._terrain_center is None:
            self._terrain_center = self.map_center_world.copy()
            return
        cells = np.round((self.map_center_world - self._terrain_center) / GRID_SCALE).astype(int)
        for axis in range(3):
            sft = int(cells[axis])
            if sft == 0:
                continue
            self._terrain_attr = np.roll(self._terrain_attr, -sft, axis=axis)
            slc = [slice(None)] * 3
            slc[axis] = slice(MAP_SHAPE[axis] - sft, None) if sft > 0 else slice(None, -sft)
            self._terrain_attr[tuple(slc)] = 0.0
        self._terrain_center = self._terrain_center + cells * GRID_SCALE

    def _on_vlm_detections(self, detections):
        """Remember which attraction cells a VLM *terrain* word (trail, ridge,
        ...) is about to write, so the stand-down gate can ignore them."""
        if not C.FIELD_GATE_TARGET_ONLY:
            return
        self._sync_terrain_shadow()
        cur = self.attraction_map[:, :, :, 0]
        for det in detections:
            if self._is_target_like(det.get("name", "")):
                continue
            g = world_to_grid(det["world_pos"], self.map_center_world)
            gx, gy, gz = int(g[0]), int(g[1]), int(g[2])
            if grid_in_bounds(gx, gy, gz) and det["score"] > cur[gx, gy, gz]:
                self._terrain_attr[gx, gy, gz] = det["score"]

    def _gate_evidence(self):
        """(max attraction incl. terrain detections, max attraction excluding them)."""
        ch0 = self.attraction_map[:, :, :, 0]
        full = float(ch0.max())
        if not C.FIELD_GATE_TARGET_ONLY:
            return full, full
        self._sync_terrain_shadow()
        real = float(np.where(ch0 > self._terrain_attr + 1e-6, ch0, 0.0).max())
        return full, real

    def get_field_giveup_stats(self):
        """Diagnostics for the LEM give-up mechanism (see hl_config FIELD_GIVEUP*):
        ``exhausted_count`` is the current (FIFO-capped) blacklist size,
        ``giveup_total`` is how many times it ever fired this episode
        (uncapped -- the number that matters for "does more time trigger more
        give-ups"), ``dwell_steps``/``dwell_center`` are the in-progress dwell
        counter at whatever point the episode ended."""
        return dict(
            exhausted_count=len(self._field_exhausted),
            giveup_total=self._field_giveup_total,
            dwell_steps=self._field_dwell_steps,
            dwell_center=self._field_dwell_center,
            gate=dict(self._gate_stats),
        )

    # ------------------------------------------------------------------
    def _extra_obstacle_update(self, camera_data, pos, ori):
        """Omnidirectional obstacle map: fold the right/back/left cameras'
        depth into ``obstacle_map`` (plain APEX uses the front camera only).
        The leader steers the follower into steep, cluttered terrain where a
        front-only map lags badly and the drone flies into things off to the
        side. HL_OBSTACLE_360=0 turns this off for the ablation."""
        if not C.OBSTACLE_360:
            return
        for cam in camera_data[1:]:  # skip front (already done)
            _, yaw_offset, _, _, depth_img, cam_pos, cam_ori = cam
            if depth_img is None:
                continue
            use_pose = cam_pos is not None and cam_ori is not None
            self.obstacle_map = update_obstacle_from_depth(
                self.obstacle_map, depth_img,
                cam_pos if use_pose else pos,
                cam_ori if use_pose else ori,
                FOV_W, self.map_center_world,
                camera_yaw_offset=(0.0 if use_pose else yaw_offset),
            )

    def _is_stuck(self, drone_world):
        self._pos_hist.append((float(drone_world[0]), float(drone_world[1])))
        if len(self._pos_hist) > C.STUCK_WINDOW + 1:
            self._pos_hist.pop(0)
        if len(self._pos_hist) <= C.STUCK_WINDOW:
            return False  # not enough history yet — treat as "not stuck"
        x0, y0 = self._pos_hist[0]
        x1, y1 = self._pos_hist[-1]
        net = float(np.hypot(x1 - x0, y1 - y0))
        return net < C.STUCK_DIST_M

    def _maybe_unwedge_forward(self, stuck):
        """See hl_config FIELD_UNWEDGE: clear ONLY the exact obstacle cells
        action_predict.py's "forward" safety check reads (a short strip
        straight ahead at the drone's own local y, z), once the follower has
        been physically stalled for FIELD_UNWEDGE_AFTER consecutive checks.
        Narrowly targeted -- not a general-area clear -- so a genuinely
        obstructed direction elsewhere is left alone; this only rescues the
        specific "forward is a stale, never-decaying mark and rotating can
        never clear it or change what forward means" deadlock."""
        if not C.FIELD_UNWEDGE:
            return
        if not stuck:
            self._stuck_streak = 0
            return
        self._stuck_streak += 1
        if self._stuck_streak < C.FIELD_UNWEDGE_AFTER:
            return
        self._stuck_streak = 0  # reset so this fires at most once per streak
        gx, gy, gz = (int(v) for v in DRONE_POS_LOCAL)
        sx, ex = gx, min(gx + 4, MAP_SHAPE[0])
        strip = self.obstacle_map[sx:ex, gy, gz]
        if np.any(strip >= 1.0):
            self.obstacle_map[sx:ex, gy, gz] = 0.0
            print(f"[FusedApex] unwedge: cleared forward obstacle strip "
                  f"after {C.FIELD_UNWEDGE_AFTER} stuck steps")

    # ------------------------------------------------------------------
    def _local_world_xy(self):
        map_origin = self.map_center_world - MAP_SIZE_WORLD / 2.0
        wx = (self._gx_idx + 0.5) * GRID_SCALE + map_origin[0]
        wy = (self._gy_idx + 0.5) * GRID_SCALE + map_origin[1]
        return wx, wy

    def _wedge_weight(self, wx, wy, drone_world, target_xy):
        """(40, 40) weight in [0, 1]: 1 on the ray drone->target, tapering to
        0 at +/- HL_FUSION_WEDGE_DEG, times a mild near-field damp so the
        drone's own column isn't touched."""
        dx = wx - drone_world[0]
        dy = wy - drone_world[1]
        r = np.hypot(dx, dy)
        cell_bearing = np.arctan2(dy, dx)
        tgt_bearing = np.arctan2(target_xy[1] - drone_world[1],
                                 target_xy[0] - drone_world[0])
        ang = np.abs(np.arctan2(np.sin(cell_bearing - tgt_bearing),
                                np.cos(cell_bearing - tgt_bearing)))
        half = np.deg2rad(max(1.0, C.FUSION_WEDGE_DEG))
        wedge = np.clip((half - ang) / half, 0.0, 1.0)
        near_damp = np.clip(r / 20.0, 0.15, 1.0)
        return wedge * near_damp, r

    # ------------------------------------------------------------------
    def _spawn_escape_hatch(self):
        """Very conservative anti-entombment guard. A bad initial depth frame
        can mark the follower's own voxel column as an obstacle, so
        action_predict returns -inf for every action and the drone "crashes"
        on the spot (reproducible on task 558).

        Earlier this cleared a 3x3x6 block whenever the drone's own cell read
        blocked in the first 5 steps — but on these obstructed tasks the
        follower legitimately spawns next to a slope/tree line, so it was
        deleting *real* obstacles and letting the drone fly straight into
        terrain (3 extra dapeng crashes in B4/B5). Now it only fires when the
        follower is genuinely *entombed* — every cell in the 5x5 patch at its
        own altitude is solid — which cannot happen for a real spawn, only a
        depth glitch. And it clears only a single 1-cell vertical shaft so
        the drone can ascend out; it never opens a horizontal path."""
        if self.step_count > 3:
            return
        cx, cy, cz = 20, 20, 5
        patch = self.obstacle_map[cx - 2:cx + 3, cy - 2:cy + 3, cz]
        if patch.size and np.all(patch >= 1.0):
            self.obstacle_map[cx, cy, :cz + 1] = 0.0
            print(f"[FusedApex] anti-entombment: cleared 1-cell ascent shaft "
                  f"(step {self.step_count})")

    # ------------------------------------------------------------------
    def _fuse_frontier(self, drone_world):
        """Self-extinguishing directional bias — see the ``frontier`` entry in
        hl_config.FUSION_MODE. Keeps the follower moving toward prompt-relevant
        terrain the leader has spotted, without the fixation / crash failure
        modes of the earlier modes."""
        tf = self.terrain_field
        dx0, dy0 = float(drone_world[0]), float(drone_world[1])

        # keep a short position history for the progress / heading gates
        self._pos_hist.append((dx0, dy0))
        if len(self._pos_hist) > 9:
            self._pos_hist.pop(0)

        if C.FRONTIER_MAX_STEP and self.step_count > C.FRONTIER_MAX_STEP:
            return  # early-guidance-only: hand back to plain APEX after this step

        # 1. record coverage and hard-decay the leader's prior where we've been
        tf.note_visit(dx0, dy0)
        tf.decay_at(dx0, dy0, C.FRONTIER_DECAY_ETA, radius_cells=C.FRONTIER_DECAY_RADIUS)

        # GATE A — APEX already has a target candidate: get out of its way so the
        # terrain nudge can't pull it off a detection it's homing in on (k2 losses).
        if float(self.attraction_map[:, :, :, 0].max()) >= C.FRONTIER_DETECTION_GATE:
            return

        # GATE B — the follower is making progress on its own: only nudge when it's
        # actually stalled (spinning), which is when action_predict ties and APEX
        # would otherwise just rotate left forever.
        if C.FRONTIER_STUCK_DIST_M > 0.0 and len(self._pos_hist) >= 6:
            x0, y0 = self._pos_hist[0]
            if float(np.hypot(dx0 - x0, dy0 - y0)) >= C.FRONTIER_STUCK_DIST_M:
                return

        # 2. pick the best *unvisited* high-terrain region
        best = tf.best_region(min_score=max(C.FUSION_SCORE_GATE, 1e-6),
                              visit_penalty=C.VISIT_PENALTY)
        if best is None:
            return
        best_key, best_eff, best_center = best
        rng = float(np.hypot(best_center[0] - dx0, best_center[1] - dy0))
        if rng > C.FRONTIER_MAX_RANGE_M:
            return  # leader hasn't found anything actionable near the follower yet

        # GATE C — don't yank the heading. If the leader's region is way off the
        # follower's recent travel direction, skip (a big redirect is what cost
        # targets on k2).
        if C.FRONTIER_MAX_TURN_DEG > 0.0 and len(self._pos_hist) >= 4:
            xh, yh = self._pos_hist[0]
            move = np.hypot(dx0 - xh, dy0 - yh)
            if move > 5.0:
                travel_bearing = np.arctan2(dy0 - yh, dx0 - xh)
                tgt_bearing = np.arctan2(best_center[1] - dy0, best_center[0] - dx0)
                dang = abs(np.degrees(np.arctan2(np.sin(tgt_bearing - travel_bearing),
                                                 np.cos(tgt_bearing - travel_bearing))))
                if dang > C.FRONTIER_MAX_TURN_DEG:
                    return

        # GATE D — obstacle-aware corridor veto. The leader's region may sit
        # behind a ridge / tree line the follower's front camera hasn't cleared;
        # ramping attraction that way is what produced the crash tax on the
        # failure subset. Sample the straight line (in grid space, from the
        # follower's own cell at the map centre toward the region) and bail if
        # too much of it is already a blocked column.
        if C.FRONTIER_CORRIDOR_VETO > 0.0:
            obstacle_col_pre = np.any(self.obstacle_map >= 1.0, axis=2)
            ci, cj = MAP_SHAPE[0] // 2, MAP_SHAPE[1] // 2
            di = (best_center[0] - dx0) / GRID_SCALE
            dj = (best_center[1] - dy0) / GRID_SCALE
            norm = float(np.hypot(di, dj))
            if norm > 1e-6:
                di, dj = di / norm, dj / norm
                reach = min(norm, MAP_SHAPE[0] / 2.0 - 1.0)
                steps = max(1, int(reach))
                blocked = seen = 0
                for s in range(2, steps + 1):  # skip the first 2 cells (own column)
                    gi = int(round(ci + di * s))
                    gj = int(round(cj + dj * s))
                    if not (0 <= gi < MAP_SHAPE[0] and 0 <= gj < MAP_SHAPE[1]):
                        break
                    seen += 1
                    if obstacle_col_pre[gi, gj]:
                        blocked += 1
                if seen and blocked / seen > C.FRONTIER_CORRIDOR_VETO:
                    return

        self._last_best_key = best_key
        self._fusion_active_steps += 1

        wx, wy = self._local_world_xy()
        weight, r = self._wedge_weight(wx, wy, drone_world, best_center)
        if not np.any(weight > 0.0):
            return
        obstacle_col = np.any(self.obstacle_map >= 1.0, axis=2)

        # 3. low-capped attraction ramp — a carrot ahead (strongest at the far
        #    edge of the local window), never a sticky blob on a point. This
        #    alone breaks action_predict's rotate-left tie (any positive
        #    attraction in the frustum beats the ~0 baseline) and it
        #    self-extinguishes via the visit penalty + decay above.
        raw = tf.get(best_key)                       # current (possibly decayed) leader score
        amp = self.fusion_gamma * max(raw, best_eff)
        if C.FUSION_FLOOD_DAMP > 0.0:
            frac_hi = tf.frac_above(max(C.FUSION_SCORE_GATE, 1e-6))
            amp *= max(0.0, 1.0 - C.FUSION_FLOOD_DAMP * frac_hi)
        range_w = np.clip(r / _LOCAL_HALF_M, 0.30, 1.0)
        ramp = np.minimum(C.FRONTIER_CAP, amp * weight * range_w).astype(np.float32)
        ramp[obstacle_col] = 0.0
        if np.any(ramp):
            cur = self.attraction_map[:, :, :, 0]
            self.attraction_map[:, :, :, 0] = np.maximum(cur, ramp[:, :, None])

        # 4. optional exploration nudge — only for cells that are STILL
        #    genuinely unexplored (expl < RATE_CENTER). Earlier this used an
        #    unconditional min()-cap, which re-flagged covered ground as
        #    unexplored every step and made the follower commit to one
        #    heading forever (F1: path length halved). Guarding on
        #    "still unexplored" keeps it self-extinguishing. Disabled when
        #    FRONTIER_EXPL_DELTA == 0.
        if C.FRONTIER_EXPL_DELTA > 0.0:
            in_wedge = weight > 0.0
            pull = C.FRONTIER_EXPL_DELTA * amp * weight
            target_expl = np.maximum(0.0, RATE_CENTER - pull).astype(np.float32)
            expl_col_min = self.exploration_map.min(axis=2)  # (40,40)
            do_pull = in_wedge & (expl_col_min < RATE_CENTER)
            col_target = np.where(do_pull, target_expl, np.inf).astype(np.float32)
            self.exploration_map = np.minimum(self.exploration_map, col_target[:, :, None])

    # ------------------------------------------------------------------
    def _roi_confidence(self, key):
        raw = float(self.terrain_field.get(key))
        mode = C.ROI_CONF_MODE
        if mode == "const":
            return 1.0
        if mode == "contrast":
            frac_hi = self.terrain_field.frac_above(max(C.FUSION_SCORE_GATE, 1e-6))
            # stand-out bonus: low when the whole field is 'high' (uninformative)
            return float(np.clip(raw * (1.0 - 0.7 * frac_hi), 0.0, 1.0))
        return float(np.clip(raw, 0.0, 1.0))

    def _acquire_roi(self, dx0, dy0):
        if self._roi_done_count >= C.ROI_MAX:
            return False
        best = self.terrain_field.best_region(
            min_score=max(C.FUSION_SCORE_GATE, 1e-6), visit_penalty=C.VISIT_PENALTY)
        if best is None:
            return False
        key, _, center = best
        self._roi_key = key
        self._roi_center = center
        self._roi_conf = self._roi_confidence(key)
        self._roi_state = "transit"
        self._roi_enter_step = None
        self._roi_reached = False
        self._roi_transit_steps = 0
        return True

    def _release_roi(self, reason):
        if self._roi_key is not None:
            self.terrain_field.mark_exhausted(self._roi_key)
            self._roi_done_count += 1
            print(f"[FusedApex] ROI released ({reason}) after "
                  f"{self.step_count - (self._roi_enter_step or self.step_count)} "
                  f"search steps; done={self._roi_done_count}/{C.ROI_MAX}")
        self._roi_key = None
        self._roi_center = None
        self._roi_state = "transit"

    def _fuse_roi(self, drone_world):
        """TRANSIT/SEARCH state machine driven by confidence-weighted ROIs.
        Runs every step for the whole mission (continuous leader influence);
        each ROI self-releases on coverage / dwell / detection so no fixation."""
        tf = self.terrain_field
        dx0, dy0 = float(drone_world[0]), float(drone_world[1])
        tf.note_visit(dx0, dy0)

        # GATE A — APEX has a real target candidate: suspend ROI guidance, let
        # the follower's own detector home in.
        if float(self.attraction_map[:, :, :, 0].max()) >= C.FRONTIER_DETECTION_GATE:
            return

        if self._roi_key is None or tf.get(self._roi_key) < max(C.FUSION_SCORE_GATE, 1e-6):
            if not self._acquire_roi(dx0, dy0):
                return  # no ROI to work -> plain APEX

        cx, cy = self._roi_center
        dist = float(np.hypot(cx - dx0, cy - dy0))
        wx, wy = self._local_world_xy()
        obstacle_col = np.any(self.obstacle_map >= 1.0, axis=2)
        conf = max(0.05, self._roi_conf)

        if dist > C.ROI_BOX_M and not self._roi_reached:
            # -------- TRANSIT: head to the ROI centre --------
            self._roi_state = "transit"
            self._roi_transit_steps += 1
            # GATE C — don't yank the heading
            if C.FRONTIER_MAX_TURN_DEG > 0.0 and len(self._pos_hist) >= 4:
                xh, yh = self._pos_hist[0]
                if np.hypot(dx0 - xh, dy0 - yh) > 5.0:
                    tb = np.arctan2(cy - dy0, cx - dx0)
                    vb = np.arctan2(dy0 - yh, dx0 - xh)
                    dang = abs(np.degrees(np.arctan2(np.sin(tb - vb), np.cos(tb - vb))))
                    if dang > C.FRONTIER_MAX_TURN_DEG:
                        self._pos_hist.append((dx0, dy0))
                        if len(self._pos_hist) > 9:
                            self._pos_hist.pop(0)
                        return
            weight, r = self._wedge_weight(wx, wy, drone_world, self._roi_center)
            if np.any(weight > 0.0):
                range_w = np.clip(r / _LOCAL_HALF_M, 0.30, 1.0)
                ramp = np.minimum(C.ROI_ATTRACT_CAP,
                                  self.fusion_gamma * conf * weight * range_w).astype(np.float32)
                ramp[obstacle_col] = 0.0
                cur = self.attraction_map[:, :, :, 0]
                self.attraction_map[:, :, :, 0] = np.maximum(cur, ramp[:, :, None])
            # give up transit if it drags on (ROI unreachable)
            if self._roi_transit_steps > C.ROI_DWELL_MAX:
                self._release_roi("transit-timeout")
        else:
            # -------- SEARCH: comb the box, fenced by confidence --------
            self._roi_reached = True
            self._roi_state = "search"
            if self._roi_enter_step is None:
                self._roi_enter_step = self.step_count
            in_box = (np.hypot(wx - cx, wy - cy) <= C.ROI_BOX_M)
            expl_min = self.exploration_map.min(axis=2)
            # (1) pull down exploration for UNCOVERED in-box cells -> systematic sweep
            uncov = in_box & (expl_min < RATE_CENTER) & (~obstacle_col)
            if np.any(uncov):
                tgt = np.where(uncov, RATE_CENTER - C.ROI_SWEEP_DELTA * conf, np.inf).astype(np.float32)
                self.exploration_map = np.minimum(self.exploration_map, tgt[:, :, None])
            # (2) soft fence: raise exploration OUTSIDE the box so APEX's frontier
            #     pull doesn't drag the follower out before the box is covered
            out_box = ~in_box
            if np.any(out_box) and C.ROI_FENCE > 0.0:
                fence = np.where(out_box, RATE_CENTER + C.ROI_FENCE * conf, -np.inf).astype(np.float32)
                self.exploration_map = np.maximum(self.exploration_map, fence[:, :, None])
            # (3) release conditions
            n_box = int(in_box.sum())
            cover = float((in_box & (expl_min >= RATE_CENTER)).sum()) / max(1, n_box)
            dwell = self.step_count - self._roi_enter_step
            dwell_cap = C.ROI_DWELL_MIN + (C.ROI_DWELL_MAX - C.ROI_DWELL_MIN) * conf
            if cover >= C.ROI_COVER_FRAC and dwell >= C.ROI_DWELL_MIN:
                self._release_roi(f"covered {cover:.0%}")
            elif dwell >= dwell_cap:
                self._release_roi(f"dwell {dwell}>{dwell_cap:.0f} conf={conf:.2f}")

        self._pos_hist.append((dx0, dy0))
        if len(self._pos_hist) > 9:
            self._pos_hist.pop(0)
        self._fusion_active_steps += 1

    # ------------------------------------------------------------------
    def _fuse_densemap(self, drone_world):
        """Continuous, whole-mission bias from a pre-built full-coverage
        RelevanceGrid (survey-then-search).  Tie-breaker, not force:

          GATE A  - stand down entirely once APEX has a real detection.
          (1) guided exploration pull: only for cells that are BOTH still
              genuinely unexplored AND above the relevance gate AND not an
              obstacle column -> APEX's action scorer then prefers headings
              toward relevant unsearched ground.  Self-extinguishing (a cell
              stops being pulled once APEX marks it explored).
          (2) a small, hard-capped attraction bump for the very-highest
              relevance cells only -> breaks action_predict's rotate-left tie
              toward them without ever faking a target (cap << KEY_THRESHOLD).
        No transit, no fence, no dwell — the whole map is already known, so the
        follower just gets a standing gradient toward the good terrain.
        """
        dx0, dy0 = float(drone_world[0]), float(drone_world[1])
        self._pos_hist.append((dx0, dy0))
        if len(self._pos_hist) > 9:
            self._pos_hist.pop(0)

        # GATE A
        if float(self.attraction_map[:, :, :, 0].max()) >= C.FRONTIER_DETECTION_GATE:
            return

        wx, wy = self._local_world_xy()                 # (40,40) world coords
        rel = self.dense_grid.sample_at(wx, wy)         # (40,40) in [0,1]
        if not np.any(rel > C.DENSE_REL_GATE):
            return
        self._fusion_active_steps += 1

        obstacle_col = np.any(self.obstacle_map >= 1.0, axis=2)

        # (1) guided exploration pull
        pull = C.DENSE_EXPL_GAIN * np.clip(
            (rel - C.DENSE_REL_GATE) / max(1e-6, 1.0 - C.DENSE_REL_GATE), 0.0, 1.0)
        expl_col_min = self.exploration_map.min(axis=2)
        do_pull = (pull > 0.0) & (~obstacle_col) & (expl_col_min < RATE_CENTER)
        if np.any(do_pull):
            target_expl = np.maximum(0.0, RATE_CENTER - pull).astype(np.float32)
            col_target = np.where(do_pull, target_expl, np.inf).astype(np.float32)
            self.exploration_map = np.minimum(self.exploration_map, col_target[:, :, None])

        # (2) capped attraction for the very-highest relevance cells
        amp = C.DENSE_ATTRACT_GAIN * np.clip(
            (rel - C.DENSE_REL_HI) / max(1e-6, 1.0 - C.DENSE_REL_HI), 0.0, 1.0)
        amp = np.minimum(C.DENSE_ATTRACT_CAP, amp).astype(np.float32)
        amp[obstacle_col] = 0.0
        if np.any(amp):
            cur = self.attraction_map[:, :, :, 0]
            self.attraction_map[:, :, :, 0] = np.maximum(cur, amp[:, :, None])

    def _fuse_field(self, drone_world):
        """Stateless, per-cell relevance field. NO subgoal / centroid / visited-
        dwell-arrival state machine -- nothing is ever stored between steps.

          (a) direct bump on the follower's own local map, proportional to
              each cell's own sampled relevance (smooth, capped).
          (b) a small directional lean toward the relevance-mass-weighted mean
              bearing within FIELD_WINDOW_M (default 500 m) of the drone,
              recomputed fresh every step -- extends awareness a bit past the
              follower's own 200 m local map without ever committing to an
              (x, y) point to travel to.
        GATE A stand-down unchanged (softer threshold under FIELD_COMMIT so the
        commit FLOOR does not self-trip it)."""
        # Updated unconditionally (before any early return) so the stuck
        # detector's position history stays continuous across steps
        # regardless of which branch below fires this step.
        stuck = (self._is_stuck(drone_world)
                if (C.FIELD_STUCK_BOOST or C.FIELD_UNWEDGE) else False)
        self._maybe_unwedge_forward(stuck)
        if C.FIELD_DEBUG:
            self._dbg = {"stuck": bool(stuck), "stood_down": False, "lean_xy": None,
                         "blocked": None, "local_max": 0.0, "lean_max": 0.0, "bump_max": 0.0}
        self._climb_active = False
        self._desc_active = False
        if self._home_z is None:
            self._home_z = float(drone_world[2])
        gate_a = (C.FIELD_COMMIT_STANDDOWN_ATTRACT if C.FIELD_COMMIT
                  else C.FRONTIER_DETECTION_GATE)
        full_max, real_max = self._gate_evidence()
        self._gate_stats["steps"] += 1
        if full_max >= gate_a and real_max < gate_a:
            self._gate_stats["old_gate_only"] += 1
        if real_max >= gate_a:
            self._gate_stats["stood_down"] += 1
            if C.FIELD_DEBUG:
                self._dbg["stood_down"] = True
            self._field_commit_center = None
            self._field_commit_steps = 0
            return
        g = self.dense_grid
        dx0, dy0 = float(drone_world[0]), float(drone_world[1])

        # follower's own negative evidence (see hl_config FIELD_NEG_EVIDENCE)
        rel_grid = g.relevance
        decay_grid = None
        if C.FIELD_NEG_EVIDENCE:
            if self._obs_cnt is None:
                self._obs_cnt = np.zeros((g.H, g.W), np.float32)
            gy0, gx0 = np.mgrid[0:g.H, 0:g.W]
            near = np.hypot(g.origin[0] + (gx0 + 0.5) * g.cell - dx0,
                            g.origin[1] + (gy0 + 0.5) * g.cell - dy0) <= C.FIELD_OBS_RADIUS_M
            self._obs_cnt[near] += 1.0
            decay_grid = np.exp(-self._obs_cnt / max(C.FIELD_NEG_TAU_STEPS, 1e-6)).astype(np.float32)
            rel_grid = (AB_UNOBSERVED_PRIOR + (g.relevance - AB_UNOBSERVED_PRIOR) * decay_grid).astype(np.float32)

        # --- Lever 1: commit / anti-wander (v2) ---------------------------
        # When the drone's own cell is in the aerial hot zone and APEX has no
        # strong candidate yet, anchor a search disc at the windowed relevance
        # PEAK and drop a *peaked* attraction FLOOR over it (highest at the
        # peak, linear falloff to the disc edge) so APEX has a gradient to
        # orbit -- NO hard exploration fence (v1's fence boxed APEX into a 70 m
        # disc near terrain -> 33 % crash). Release on a strong detection or on
        # leaving the zone; exhaust the peak after MAX_STEPS with nothing found.
        if C.FIELD_COMMIT:
            rel_here = float(g.sample_at(np.array([dx0], float),
                                        np.array([dy0], float))[0])
            hot = rel_here >= C.FIELD_COMMIT_HOT_REL
            near_exhausted = any(
                math.hypot(dx0 - ex, dy0 - ey) <= C.FIELD_GIVEUP_BOX_M
                for (ex, ey) in self._field_exhausted)
            if hot and not near_exhausted:
                wx0, wy0 = self._local_world_xy()
                if self._field_commit_center is None:
                    # anchor at the windowed relevance peak, not wherever the
                    # drone happened to cross the HOT_REL threshold
                    gyy, gxx = np.mgrid[0:g.H, 0:g.W]
                    pwx = g.origin[0] + (gxx + 0.5) * g.cell
                    pwy = g.origin[1] + (gyy + 0.5) * g.cell
                    pm = g.relevance.copy()
                    pm[np.hypot(pwx - dx0, pwy - dy0) > C.FIELD_WINDOW_M] = 0.0
                    for (ex, ey) in self._field_exhausted:
                        pm[np.hypot(pwx - ex, pwy - ey)
                           <= C.FIELD_GIVEUP_BOX_M] = 0.0
                    if float(pm.max()) <= 0.0:
                        self._field_commit_center = (dx0, dy0)
                    else:
                        pr, pc = np.unravel_index(int(pm.argmax()), pm.shape)
                        self._field_commit_center = (float(pwx[pr, pc]),
                                                     float(pwy[pr, pc]))
                    self._field_commit_steps = 0
                ccx, ccy = self._field_commit_center
                self._field_commit_steps += 1
                d_disc = np.hypot(wx0 - ccx, wy0 - ccy)
                ramp = np.clip(1.0 - d_disc / max(C.FIELD_COMMIT_BOX_M, 1.0),
                               0.0, 1.0)
                floor = (C.FIELD_COMMIT_FLOOR * ramp).astype(np.float32)
                self.attraction_map[:, :, :, 0] = np.maximum(
                    self.attraction_map[:, :, :, 0], floor[:, :, None])
                if self._field_commit_steps >= C.FIELD_COMMIT_MAX_STEPS:
                    self._mark_field_exhausted(self._field_commit_center)
                    self._field_commit_center = None
                    self._field_commit_steps = 0
                return
            # left the hot zone -> drop the anchor
            self._field_commit_center = None
            self._field_commit_steps = 0

        # (a) direct per-cell bump over the follower's own local map
        wx, wy = self._local_world_xy()
        rel_local = g.sample_at(wx, wy)
        if decay_grid is not None:
            ci = np.clip(np.floor((wx - g.origin[0]) / g.cell).astype(int), 0, g.W - 1)
            ri = np.clip(np.floor((wy - g.origin[1]) / g.cell).astype(int), 0, g.H - 1)
            rel_local = AB_UNOBSERVED_PRIOR + (rel_local - AB_UNOBSERVED_PRIOR) * decay_grid[ri, ci]
        if C.FIELD_GIVEUP:
            for (vx, vy) in self._field_exhausted:
                rel_local = np.where(np.hypot(wx - vx, wy - vy) <= C.FIELD_GIVEUP_BOX_M,
                                     AB_UNOBSERVED_PRIOR, rel_local)
        local_bias = np.clip(C.FIELD_GAMMA_LOCAL * (rel_local - AB_UNOBSERVED_PRIOR),
                              0.0, C.FIELD_CAP_LOCAL).astype(np.float32)

        # (b) directional lean toward the mean bearing of relevant mass within
        #     a wider window -- NOT stored as a subgoal, just this step's
        #     gradient (but see the give-up bookkeeping below)
        gy, gx = np.mgrid[0:g.H, 0:g.W]
        cwx = g.origin[0] + (gx + 0.5) * g.cell
        cwy = g.origin[1] + (gy + 0.5) * g.cell
        M = rel_grid.copy()
        M[M < C.AERODUO_PROB_GATE] = 0.0
        d_all = np.hypot(cwx - dx0, cwy - dy0)
        M[d_all > C.FIELD_WINDOW_M] = 0.0
        if C.FIELD_GIVEUP:
            for (vx, vy) in self._field_exhausted:
                M[np.hypot(cwx - vx, cwy - vy) <= C.FIELD_GIVEUP_BOX_M] = 0.0
        obstacle_col = self._obstacle_cols(drone_world[2])
        lean = np.zeros_like(local_bias)

        # --- pick this step's lean-target point (the wedge uses only its
        #     bearing from the drone). Two estimators: ---
        lean_xy = None
        if C.FIELD_LEAN_MODE == "vecsum":
            # vector sum of per-cell pull vectors, nonlinear relevance weight
            # g(rel)=clip(rel-GATE,0,1)^P and distance falloff h=exp(-d/D).
            rel_w = rel_grid.copy()
            rel_w[d_all > C.FIELD_WINDOW_M] = 0.0
            if C.FIELD_GIVEUP:
                for (vx, vy) in self._field_exhausted:
                    rel_w[np.hypot(cwx - vx, cwy - vy) <= C.FIELD_GIVEUP_BOX_M] = AB_UNOBSERVED_PRIOR
            w = np.clip(rel_w - C.FIELD_VS_GATE, 0.0, 1.0) ** C.FIELD_VS_POWER
            w *= np.exp(-d_all / max(C.FIELD_VS_DIST_SCALE_M, 1e-6))
            w[d_all < C.FIELD_VS_NEAR_CUT_M] = 0.0      # leave the near field to APEX
            tot_w = float(w.sum())
            if tot_w > 1e-9:
                inv = 1.0 / np.maximum(d_all, 1.0)
                Vx = float((w * (cwx - dx0) * inv).sum())
                Vy = float((w * (cwy - dy0) * inv).sum())
                Vmag = math.hypot(Vx, Vy)
                if Vmag > 1e-9 and (Vmag / tot_w) >= C.FIELD_VS_MIN_CONC:
                    L = max(C.FIELD_WINDOW_M, 1.0)      # point far along the net direction
                    lean_xy = (dx0 + Vx / Vmag * L, dy0 + Vy / Vmag * L)
        elif C.FIELD_LEAN_MODE == "cluster":
            # connected-component the WINDOWED mass; lean toward the best
            # cluster's OWN centroid (mass / (1 + dist/D)). Never a global
            # centroid -- separated blobs are never blended.
            D = max(getattr(C, "FIELD_LEAN_DIST_SCALE_M", 250.0), 1e-6)
            xy, _m, _tot = self._best_cluster(M, cwx, cwy, dx0, dy0, D)
            if xy is not None:
                lean_xy = (float(xy[0]), float(xy[1]))
        else:                                          # legacy: windowed centroid
            Mlean = M
            if C.FIELD_LEAN_DISTWEIGHT:                 # B: discount by distance
                Mlean = M * np.exp(-d_all / max(C.FIELD_LEAN_DIST_SCALE_M, 1e-6))
            tot = float(Mlean.sum())
            if tot > 1e-9:
                lean_xy = (float((Mlean * cwx).sum() / tot),
                           float((Mlean * cwy).sum() / tot))

        if lean_xy is not None:
            mean_xy = lean_xy

            # --- lean brake: don't push toward a bearing that runs into
            #     observed terrain within FIELD_LEAN_BRAKE_M. The continuous
            #     per-step lean otherwise commits the follower to a heading that,
            #     in mountainous maps, ends in a terrain clip before the obstacle
            #     voxels populate. Skip only the directional push this step; the
            #     obstacle-masked local bump (a) still applies. Give-up
            #     bookkeeping below still runs so a chronically-blocked target
            #     is eventually exhausted.
            lean_blocked = False
            if C.FIELD_LEAN_BRAKE:
                bx, by = mean_xy[0] - dx0, mean_xy[1] - dy0
                dd = math.hypot(bx, by)
                if dd > 1.0:
                    reach = min(C.FIELD_LEAN_BRAKE_M, dd)
                    lean_blocked = self._seg_blocked(
                        dx0, dy0, dx0 + bx / dd * reach, dy0 + by / dd * reach,
                        drone_z=drone_world[2])
                    ex, ey = dx0 + bx / dd * reach, dy0 + by / dd * reach
                    self._gate_stats["brake_checks"] = self._gate_stats.get("brake_checks", 0) + 1
                    if self._seg_blocked(dx0, dy0, ex, ey):
                        self._gate_stats["brake_any_layer"] = self._gate_stats.get("brake_any_layer", 0) + 1

            climb_ok = False
            if lean_blocked and C.FIELD_CLIMB:
                climb_ok = self._climb_possible(
                    dx0, dy0, dx0 + bx / dd * reach, dy0 + by / dd * reach, drone_world[2])
            desc_ok = False
            if (not lean_blocked) and C.FIELD_DESCEND and dd > 1.0 and \
                    float(drone_world[2]) < self._home_z - C.FIELD_DESC_MIN_M:
                nl_lo = min(self._band_layers(drone_world[2]) + 1, MAP_SHAPE[2])
                desc_ok = not self._seg_blocked(
                    dx0, dy0, dx0 + bx / dd * reach, dy0 + by / dd * reach, layers=nl_lo)
            if lean_blocked:
                self._gate_stats["brake_blocked"] += 1
                if climb_ok:
                    self._gate_stats["climb_steps"] = self._gate_stats.get("climb_steps", 0) + 1
            if C.FIELD_DEBUG:
                self._dbg.update(lean_xy=(round(float(mean_xy[0])), round(float(mean_xy[1]))),
                                 blocked=bool(lean_blocked))
            weight, r = self._wedge_weight(wx, wy, drone_world, mean_xy)
            range_w = np.clip(r / _LOCAL_HALF_M, 0.30, 1.0)
            wedge = weight * range_w
            if C.FIELD_ARRIVE:
                d_t = math.hypot(dx0 - mean_xy[0], dy0 - mean_xy[1])
                ramp_a = float(np.clip((d_t - C.FIELD_ARRIVE_R_IN_M)
                                       / max(C.FIELD_ARRIVE_R_OUT_M - C.FIELD_ARRIVE_R_IN_M, 1.0),
                                       C.FIELD_ARRIVE_FLOOR, 1.0))
                if ramp_a < 1.0:
                    self._gate_stats["arrive_steps"] += 1
                wedge = wedge * ramp_a
            if desc_ok:
                self._desc_wedge = wedge.copy()
                self._gate_stats["descend_steps"] = self._gate_stats.get("descend_steps", 0) + 1
            if lean_blocked:
                if climb_ok:
                    self._climb_wedge = wedge.copy()
                wedge = np.zeros_like(wedge)

            if C.FIELD_LEAN_TARGET == "exploration":     # D: bias search ORDER
                pull = C.FIELD_EXPL_GAIN * wedge
                expl_col_min = self.exploration_map.min(axis=2)
                do_pull = (pull > 0.0) & (~obstacle_col) & (expl_col_min < RATE_CENTER)
                if np.any(do_pull):
                    target_expl = np.maximum(0.0, RATE_CENTER - pull).astype(np.float32)
                    col_target = np.where(do_pull, target_expl, np.inf).astype(np.float32)
                    self.exploration_map = np.minimum(self.exploration_map, col_target[:, :, None])
            else:                                        # default: bias ATTRACTION
                cap_lean = C.FIELD_CAP_LEAN_STUCK if stuck else C.FIELD_CAP_LEAN
                lean = np.minimum(cap_lean,
                                   self.fusion_gamma * wedge).astype(np.float32)

            # give-up bookkeeping: has the follower actually BEEN at the lean
            # target (not just had the belief point there) for too long with
            # nothing found? Dwell only accrues once the follower is within
            # FIELD_GIVEUP_BOX_M of mean_xy -- otherwise a distant, still-being
            # travelled-to candidate that the belief has stably favoured (e.g.
            # blocked by terrain, or briefly out-competed by a closer one)
            # would get blacklisted before the follower ever verified it.
            if C.FIELD_GIVEUP:
                same_target = (self._field_dwell_center is not None
                                and math.hypot(mean_xy[0] - self._field_dwell_center[0],
                                               mean_xy[1] - self._field_dwell_center[1])
                                <= C.FIELD_DWELL_TOL_M)
                if not same_target:
                    self._field_dwell_center = mean_xy
                    self._field_dwell_steps = 0
                near_target = math.hypot(dx0 - mean_xy[0], dy0 - mean_xy[1]) <= C.FIELD_GIVEUP_BOX_M
                if near_target:
                    self._field_dwell_steps += 1
                if self._field_dwell_steps >= C.FIELD_DWELL_MAX and not C.FIELD_NEG_EVIDENCE:
                    self._mark_field_exhausted(mean_xy)
                    self._field_dwell_center = None
                    self._field_dwell_steps = 0
        elif C.FIELD_GIVEUP:
            self._field_dwell_center = None
            self._field_dwell_steps = 0

        cap_total = C.FIELD_CAP_TOTAL_STUCK if stuck else C.FIELD_CAP_TOTAL
        bump = np.clip(local_bias + lean, 0.0, cap_total)
        bump[obstacle_col] = 0.0
        if C.FIELD_DEBUG:
            self._dbg.update(local_max=round(float(local_bias.max()), 3),
                             lean_max=round(float(lean.max()), 3), bump_max=round(float(bump.max()), 3))
        cur = self.attraction_map[:, :, :, 0]
        self.attraction_map[:, :, :, 0] = np.maximum(cur, bump[:, :, None])
        if C.FIELD_CLIMB and self._climb_wedge is not None:
            # obstacle ahead is passable by climbing: attract the layers ABOVE the drone
            # inside the lean wedge so APEX's own action_predict favours "ascend"
            climb = np.minimum(C.FIELD_CLIMB_ATTRACT, self.fusion_gamma * self._climb_wedge).astype(np.float32)
            up = self.attraction_map[:, :, :int(DRONE_POS_LOCAL[2]), 0]
            self.attraction_map[:, :, :int(DRONE_POS_LOCAL[2]), 0] = np.maximum(up, climb[:, :, None])
            self._climb_active = True
        self._climb_wedge = None
        if C.FIELD_DESCEND and self._desc_wedge is not None:
            # flying above the home altitude with a clear path one layer lower: attract the
            # layers BELOW the drone inside the wedge so action_predict favours "descend"
            lo = int(DRONE_POS_LOCAL[2]) + 1
            desc = np.minimum(C.FIELD_DESC_ATTRACT, self.fusion_gamma * self._desc_wedge).astype(np.float32)
            dn = self.attraction_map[:, :, lo:, 0]
            self.attraction_map[:, :, lo:, 0] = np.maximum(dn, desc[:, :, None])
            self._desc_active = True
        self._desc_wedge = None

    # ------------------------------------------------------------------
    def _fuse_conav(self, drone_world):
        """CoNav-UAV (training-free): navigate to the head of a priority queue of
        candidate target coords, search a box on arrival, advance on
        coverage/dwell/detection.  Queue re-ranked each step by
        score / (1 + dist_from_follower / K) (CoNav's path-efficiency re-rank)."""
        dx0, dy0 = float(drone_world[0]), float(drone_world[1])
        self._pos_hist.append((dx0, dy0))
        if len(self._pos_hist) > 9:
            self._pos_hist.pop(0)

        # GATE A — follower has its own detection: let APEX home in.
        if float(self.attraction_map[:, :, :, 0].max()) >= C.FRONTIER_DETECTION_GATE:
            if self._conav_target is not None:                 # credit the current target
                self._conav_done.add(self._conav_idx)
                self._conav_target = None
            return

        # pick / re-rank the queue head
        if self._conav_target is None:
            best_i, best_s = None, -1.0
            for i, (qx, qy, qs) in enumerate(self._conav_queue):
                if i in self._conav_done:
                    continue
                d = math.hypot(qx - dx0, qy - dy0)
                eff = qs / (1.0 + d / C.CONAV_PATHEFF_K)
                if eff > best_s:
                    best_s, best_i = eff, i
            if best_i is None:
                return  # queue exhausted -> plain APEX
            self._conav_idx = best_i
            self._conav_target = self._conav_queue[best_i][:2]
            self._conav_state = "transit"
            self._conav_enter_step = None
            self._conav_transit_steps = 0

        cx, cy = self._conav_target
        dist = math.hypot(cx - dx0, cy - dy0)
        wx, wy = self._local_world_xy()
        obstacle_col = np.any(self.obstacle_map >= 1.0, axis=2)

        if dist > C.CONAV_ARRIVE_M and self._conav_state == "transit":
            self._conav_transit_steps += 1
            if C.CONAV_MAX_TURN_DEG > 0 and len(self._pos_hist) >= 4:
                xh, yh = self._pos_hist[0]
                if math.hypot(dx0 - xh, dy0 - yh) > 5.0:
                    tb = math.atan2(cy - dy0, cx - dx0); vb = math.atan2(dy0 - yh, dx0 - xh)
                    if abs(math.degrees(math.atan2(math.sin(tb - vb), math.cos(tb - vb)))) > C.CONAV_MAX_TURN_DEG:
                        return
            weight, r = self._wedge_weight(wx, wy, drone_world, (cx, cy))
            if np.any(weight > 0.0):
                range_w = np.clip(r / _LOCAL_HALF_M, 0.30, 1.0)
                ramp = np.minimum(C.CONAV_ATTRACT_CAP,
                                  self.fusion_gamma * weight * range_w).astype(np.float32)
                ramp[obstacle_col] = 0.0
                cur = self.attraction_map[:, :, :, 0]
                self.attraction_map[:, :, :, 0] = np.maximum(cur, ramp[:, :, None])
            if self._conav_transit_steps > C.CONAV_TRANSIT_MAX:
                self._conav_done.add(self._conav_idx); self._conav_target = None
        else:
            # SEARCH: comb the box around the dispatched coord
            self._conav_state = "search"
            if self._conav_enter_step is None:
                self._conav_enter_step = self.step_count
            in_box = np.hypot(wx - cx, wy - cy) <= C.CONAV_BOX_M
            expl_min = self.exploration_map.min(axis=2)
            uncov = in_box & (expl_min < RATE_CENTER) & (~obstacle_col)
            if np.any(uncov):
                tgt = np.where(uncov, RATE_CENTER - C.CONAV_SWEEP_DELTA, np.inf).astype(np.float32)
                self.exploration_map = np.minimum(self.exploration_map, tgt[:, :, None])
            n_box = max(1, int(in_box.sum()))
            cover = float((in_box & (expl_min >= RATE_CENTER)).sum()) / n_box
            dwell = self.step_count - self._conav_enter_step
            if cover >= 0.6 or dwell >= C.CONAV_DWELL_MAX:
                self._conav_done.add(self._conav_idx); self._conav_target = None

    # ------------------------------------------------------------------
    def _select_subgoal(self, M, cwx, cwy, dx0, dy0):
        """Pick the next AeroDuo subgoal (world x, y) from the remaining
        relevance mass ``M``, per ``C.AERODUO_CENTROID_MODE``. Everything
        downstream (wedge, capped attraction floor, GATE A) is unchanged —
        this only changes which point becomes the wedge direction. Returns
        ``None`` if there is no usable mass."""
        mode = getattr(C, "AERODUO_CENTROID_MODE", "global")
        D = max(getattr(C, "AERODUO_DIST_SCALE_M", 250.0), 1e-6)

        if mode == "distweight":
            d = np.hypot(cwx - dx0, cwy - dy0)
            Mw = M * np.exp(-d / D)
            tot = float(Mw.sum())
            if tot < 1e-9:
                return None
            return [float((Mw * cwx).sum() / tot), float((Mw * cwy).sum() / tot)]

        if mode == "cluster":
            xy, _mass, _tot = self._best_cluster(M, cwx, cwy, dx0, dy0, D)
            return xy

        if mode == "adaptive":
            # Explore/exploit gate: how concentrated is the remaining mass?
            xy, top_mass, tot = self._best_cluster(M, cwx, cwy, dx0, dy0, D)
            if xy is None or tot < 1e-9:
                return None
            concentration = top_mass / tot
            if concentration >= getattr(C, "AERODUO_CONCENTRATION_THRESH", 0.6):
                return xy                              # confident -> precise commit
            tot_all = float(M.sum())                   # diffuse -> broad, low-commitment sweep
            return [float((M * cwx).sum() / tot_all), float((M * cwy).sum() / tot_all)]

        # "global" (default) — mass-weighted centroid of ALL remaining mass;
        # averages across separated regions into the space between them.
        tot = float(M.sum())
        if tot < 1e-9:
            return None
        return [float((M * cwx).sum() / tot), float((M * cwy).sum() / tot)]

    def _best_cluster(self, M, cwx, cwy, dx0, dy0, D):
        """Connected-component the remaining mass; return the (own centroid of
        the) best component by gain/cost = mass/(1+dist/D), that component's
        mass, and the total remaining mass. (xy, mass, None) if empty."""
        from scipy import ndimage
        labels, n = ndimage.label(M > 0)
        tot = float(M.sum())
        if n == 0:
            return None, 0.0, tot
        best_score, best_xy, best_mass = -1.0, None, 0.0
        for lbl in range(1, n + 1):
            mask = labels == lbl
            m = float(M[mask].sum())
            if m <= 0:
                continue
            cx = float((M[mask] * cwx[mask]).sum() / m)
            cy = float((M[mask] * cwy[mask]).sum() / m)
            dist = math.hypot(cx - dx0, cy - dy0)
            score = m / (1.0 + dist / D)
            if score > best_score:
                best_score, best_xy, best_mass = score, (cx, cy), m
        return (list(best_xy) if best_xy is not None else None), best_mass, tot

    def _fuse_aeroduo(self, drone_world):
        """AeroDuo (training-free): subgoal = centroid of the target-probability
        grid M over the not-yet-searched region; head there; on coverage/dwell,
        mask that region out of M and recompute the centroid ('request new
        guidance').  Stop when the remaining probability mass is negligible."""
        if self.dense_grid is None:
            return
        dx0, dy0 = float(drone_world[0]), float(drone_world[1])
        self._pos_hist.append((dx0, dy0))
        if len(self._pos_hist) > 9:
            self._pos_hist.pop(0)
        if self._goto_last is not None:
            moved = math.hypot(dx0 - self._goto_last[0], dy0 - self._goto_last[1])
            self._goto_last = None
            if moved < C.AERODUO_GOTO_MIN_PROGRESS_M:
                self._goto_fail += 1
                self._gate_stats["goto_noprogress"] = self._gate_stats.get("goto_noprogress", 0) + 1
                if self._goto_fail >= C.AERODUO_GOTO_STUCK_STEPS:
                    self._goto_suspend = C.AERODUO_GOTO_SUSPEND_STEPS
                    self._goto_fail = 0
                    self._gate_stats["goto_suspends"] = self._gate_stats.get("goto_suspends", 0) + 1
            else:
                self._goto_fail = 0
        if C.AERODUO_VISIT_ACTUAL:
            lv = self._aeroduo_visited[-1] if self._aeroduo_visited else None
            if lv is None or math.hypot(dx0 - lv[0], dy0 - lv[1]) >= C.AERODUO_VISIT_SPACING_M:
                self._aeroduo_visited.append((dx0, dy0))

        # same target-only stand-down evidence as Ours, so the baseline is not
        # handicapped by generic-terrain VLM detections silencing its guidance
        if self._gate_evidence()[1] >= C.FRONTIER_DETECTION_GATE:
            return

        g = self.dense_grid
        # world coords of every grid cell centre
        gy, gx = np.mgrid[0:g.H, 0:g.W]
        cwx = g.origin[0] + (gx + 0.5) * g.cell
        cwy = g.origin[1] + (gy + 0.5) * g.cell
        M = g.relevance.copy()
        M[M < C.AERODUO_PROB_GATE] = 0.0
        for (vx, vy) in self._aeroduo_visited:                 # mask searched regions
            M[np.hypot(cwx - vx, cwy - vy) <= C.AERODUO_BOX_M] = 0.0
        tot = float(M.sum())
        if tot < C.AERODUO_MIN_MASS * max(1e-6, float(g.relevance.sum())):
            return  # nothing relevant left -> plain APEX

        need_new = (self._aeroduo_subgoal is None
                    or math.hypot(self._aeroduo_subgoal[0] - dx0,
                                  self._aeroduo_subgoal[1] - dy0) <= C.AERODUO_ARRIVE_M
                    or (self._aeroduo_enter_step is not None
                        and self.step_count - self._aeroduo_enter_step >= C.AERODUO_DWELL_MAX))
        if need_new:
            if self._aeroduo_subgoal is not None:
                if not C.AERODUO_VISIT_ACTUAL:
                    self._aeroduo_visited.append(tuple(self._aeroduo_subgoal))
                    M[np.hypot(cwx - self._aeroduo_subgoal[0], cwy - self._aeroduo_subgoal[1]) <= C.AERODUO_BOX_M] = 0.0
                tot = float(M.sum())
                if tot < C.AERODUO_MIN_MASS * max(1e-6, float(g.relevance.sum())):
                    self._aeroduo_subgoal = None
                    return
            win = getattr(C, "AERODUO_WINDOW_M", 0.0)
            M_win = M
            if win and win > 0:
                # local-horizon: only mass within `win` of the follower's
                # CURRENT position is eligible for a new subgoal — the aerial
                # prior extends the follower's own limited sensing range, it
                # does not dispatch it across the map.
                M_win = M.copy()
                M_win[np.hypot(cwx - dx0, cwy - dy0) > win] = 0.0
            sg = self._select_subgoal(M_win, cwx, cwy, dx0, dy0)
            if sg is None:
                # nothing relevant within the local horizon right now -> no
                # bias this step (not a permanent give-up: re-tried next step,
                # picks up once the follower's own exploration wanders closer
                # to a relevant patch).
                self._aeroduo_subgoal = None
                return
            self._aeroduo_subgoal = sg
            self._aeroduo_enter_step = self.step_count

        cx, cy = self._aeroduo_subgoal
        wx, wy = self._local_world_xy()
        obstacle_col = self._obstacle_cols(drone_world[2])

        # --- (b) committed go-to: rotate + step toward the centroid, bypassing
        #     APEX's discrete action, unless the straight step is blocked ---
        self._aeroduo_goto_target = None
        if C.AERODUO_GOTO and self._goto_suspend > 0:
            # go-to made no progress repeatedly: hand control back to APEX's own actions
            self._goto_suspend -= 1
            self._gate_stats["goto_suspended_steps"] = self._gate_stats.get("goto_suspended_steps", 0) + 1
        elif C.AERODUO_GOTO:
            bx, by = cx - dx0, cy - dy0
            d = math.hypot(bx, by)
            if d > 1.0:
                step = min(C.AERODUO_GOTO_STEP_M, d)
                tx, ty = dx0 + bx / d * step, dy0 + by / d * step
                if not self._seg_blocked(dx0, dy0, tx, ty, drone_z=drone_world[2]):
                    self._aeroduo_goto_target = (tx, ty)
                    self._goto_last = (dx0, dy0)
                    self._gate_stats["goto_exec"] = self._gate_stats.get("goto_exec", 0) + 1
                    self._aeroduo_goto_yaw = math.degrees(math.atan2(by, bx))

        weight, r = self._wedge_weight(wx, wy, drone_world, (cx, cy))
        if np.any(weight > 0.0):
            range_w = np.clip(r / _LOCAL_HALF_M, 0.30, 1.0)
            ramp = np.minimum(C.AERODUO_ATTRACT_CAP,
                              self.fusion_gamma * weight * range_w).astype(np.float32)
            ramp[obstacle_col] = 0.0
            cur = self.attraction_map[:, :, :, 0]
            self.attraction_map[:, :, :, 0] = np.maximum(cur, ramp[:, :, None])

    def _obstacle_layers(self, drone_z):
        """Number of z-layers (from the top, index 0 = highest) that count as an
        obstacle for the follower at altitude ``drone_z``: the drone's own layer
        plus FIELD_OBS_BELOW_LAYERS below it. Ground voxels further below are
        ignored (NED: z index grows downward). None -> all layers (old behaviour)."""
        if drone_z is None or not C.FIELD_ALT_AWARE_OBSTACLE:
            return MAP_SHAPE[2]
        oz = float(self.map_center_world[2] - MAP_SIZE_WORLD[2] / 2.0)
        gz = int((float(drone_z) - oz) / GRID_SCALE)
        return int(np.clip(gz + 1 + C.FIELD_OBS_BELOW_LAYERS, 0, MAP_SHAPE[2]))

    def _band_layers(self, drone_z):
        oz = float(self.map_center_world[2] - MAP_SIZE_WORLD[2] / 2.0)
        gz = int((float(drone_z) - oz) / GRID_SCALE)
        return int(np.clip(gz + 1 + C.FIELD_OBS_BELOW_LAYERS, 0, MAP_SHAPE[2]))

    def _climb_possible(self, x0, y0, x1, y1, drone_z):
        """True if the segment is blocked at the drone's altitude band but clear
        once the band is raised by 1..FIELD_CLIMB_MAX_LAYERS layers (5 m each),
        i.e. the obstacle ahead can be climbed over."""
        nl0 = self._band_layers(drone_z)
        for k in range(1, C.FIELD_CLIMB_MAX_LAYERS + 1):
            if nl0 - k < 1:
                break
            if not self._seg_blocked(x0, y0, x1, y1, layers=nl0 - k):
                return True
        return False

    def _obstacle_cols(self, drone_z=None):
        """(40,40) bool: any obstacle voxel in the altitude band relevant to the drone."""
        return np.any(self.obstacle_map[:, :, :self._obstacle_layers(drone_z)] >= 1.0, axis=2)

    def _seg_blocked(self, x0, y0, x1, y1, n=10, drone_z=None, layers=None):
        """True if the segment (x0,y0)->(x1,y1) crosses a blocked obstacle column
        in the follower's local voxel map (only within the drone's altitude band
        when HL_FIELD_ALT_AWARE_OBSTACLE and ``drone_z`` are given)."""
        nl = layers if layers is not None else self._obstacle_layers(drone_z)
        origin = self.map_center_world - MAP_SIZE_WORLD / 2.0
        for k in range(2, n + 1):
            fx = x0 + (x1 - x0) * k / n
            fy = y0 + (y1 - y0) * k / n
            gi = int((fx - origin[0]) / GRID_SCALE)
            gj = int((fy - origin[1]) / GRID_SCALE)
            if 0 <= gi < MAP_SHAPE[0] and 0 <= gj < MAP_SHAPE[1]:
                if np.any(self.obstacle_map[gi, gj, :nl] >= 1.0):
                    return True
        return False

    def _execute_action(self, action, current_pos):
        """AeroDuo go-to override: when a committed centroid step is queued,
        rotate + moveToPosition toward it instead of APEX's discrete action."""
        if self._field_restore is not None:
            # HL_FIELD_TRANSIENT: the field bias only steers THIS step's action
            # selection; take it back out so it never accumulates in the map
            # and never trips the stand-down gate that reads that same map.
            self.attraction_map[:, :, :, 0] = self._field_restore
            self._field_restore = None
        if self._climb_active:
            self._gate_stats["climb_ascend_chosen"] = self._gate_stats.get("climb_ascend_chosen", 0) + int(action == 4)
            self._climb_active = False
        if self._desc_active:
            self._gate_stats["descend_chosen"] = self._gate_stats.get("descend_chosen", 0) + int(action == 5)
            self._desc_active = False
        tgt = getattr(self, "_aeroduo_goto_target", None)
        if self.fusion_mode == "aeroduo" and C.AERODUO_GOTO and tgt is not None:
            try:
                self.client.rotateToYawAsync(self._aeroduo_goto_yaw).join()
                self.yaw_deg = self._aeroduo_goto_yaw
            except Exception:
                pass
            self.client.moveToPositionAsync(tgt[0], tgt[1], self.flight_z,
                                            FLIGHT_SPEED).join()
            self._aeroduo_goto_target = None
            return
        return super()._execute_action(action, current_pos)

    def _nadir_detect_pass(self, drone_world):
        """Lever 2b: grab the downward-looking camera and run APEX's own
        _run_detection on it (report_clue / report_target unchanged). Called
        only when the drone is in the aerial hot zone with no target reported
        yet, so cost is zero on tasks that never reach a relevant patch."""
        if self._nadir_requests is None or self.dense_grid is None:
            return
        if self._reported_target_positions:
            return
        rel_here = float(self.dense_grid.sample_at(
            np.array([float(drone_world[0])], float),
            np.array([float(drone_world[1])], float))[0])
        if rel_here < C.FIELD_NADIR_DET_HOT_REL:
            return
        try:
            resp = self.client.simGetImages(self._nadir_requests)
        except Exception:
            return
        if not resp or len(resp) < 2:
            return
        rgb_r, dep_r = resp[0], resp[1]
        if not (rgb_r.width > 0 and rgb_r.height > 0
                and len(rgb_r.image_data_uint8) > 0):
            return
        img_rgb = np.frombuffer(rgb_r.image_data_uint8, dtype=np.uint8).reshape(
            rgb_r.height, rgb_r.width, 3)
        rgb_shape = (rgb_r.height, rgb_r.width)
        if not (dep_r.width > 0 and dep_r.height > 0
                and len(dep_r.image_data_float) > 0):
            return
        depth_img = airsim.list_to_2d_float_array(
            dep_r.image_data_float, dep_r.width, dep_r.height)
        depth_img[depth_img > MAX_DEPTH] = 0.0
        depth_img[depth_img < 0] = 0.0
        cam_pos = getattr(dep_r, "camera_position", None) \
            or getattr(rgb_r, "camera_position", None)
        cam_ori = getattr(dep_r, "camera_orientation", None) \
            or getattr(rgb_r, "camera_orientation", None)
        if cam_pos is None or cam_ori is None:
            return
        try:
            self._run_detection(img_rgb, depth_img, rgb_shape, cam_pos, cam_ori,
                                camera_yaw_offset=0.0, camera_name="nadir_camera")
        except Exception:
            pass

    def _debug_report(self, A0, drone_world):
        """HL_FIELD_DEBUG: rerun action_predict on the attraction map WITHOUT
        this step's field bias (A0) and WITH it, so the log shows whether the
        bias changed the chosen action and by how much each action's reward moved."""
        import io
        import re
        import contextlib
        from baselines_adaption.apex.voxel_maps import crop_rotate_maps, DRONE_CENTER_GRID
        from baselines_adaption.apex import action_predict as _ap

        ori = self._yaw_to_discrete_ori(self.yaw_deg)

        def run(A):
            al, el, ol = crop_rotate_maps(A, self.exploration_map, self.obstacle_map,
                                          DRONE_CENTER_GRID, ori)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                act = _ap.action_predict(al, el, ol)
            m = re.search(r"rewards=(\{.*\})", buf.getvalue())
            r = eval(m.group(1), {"inf": float("inf"), "np": np, "float64": np.float64}) if m else {}
            return int(act), [round(float(r.get(k, float("nan"))), 1) for k in range(6)]

        a0, r0 = run(A0)
        a1, r1 = run(self.attraction_map)
        print(f"[DBG] step={self.step_count} pos=({drone_world[0]:.0f},{drone_world[1]:.0f}) "
              f"yaw={self.yaw_deg:.0f} ori={ori} act_nobias={a0} act_bias={a1} "
              f"changed={a0 != a1} rew_nobias={r0} rew_bias={r1} dbg={self._dbg}")

    def _augment_attraction_map(self, drone_world):
        self._spawn_escape_hatch()
        if C.FIELD_NADIR_DET and self.fusion_mode == "field":
            self._nadir_detect_pass(drone_world)

        if self.fusion_mode == "densemap":
            if self.dense_grid is not None and self.fusion_gamma != 0.0:
                self._fuse_densemap(drone_world)
            return
        if self.fusion_mode == "conav":
            self._fuse_conav(drone_world)
            return
        if self.fusion_mode == "aeroduo":
            self._fuse_aeroduo(drone_world)
            return
        if self.fusion_mode == "field":
            if self.dense_grid is not None:
                A0 = self.attraction_map.copy() if (C.FIELD_DEBUG or C.FIELD_TRANSIENT) else None
                self._fuse_field(drone_world)
                if C.FIELD_TRANSIENT:
                    self._field_restore = A0[:, :, :, 0]
                if C.FIELD_DEBUG:
                    self._debug_report(A0, drone_world)
            return

        tf = self.terrain_field
        if self.fusion_mode == "off" or tf is None or len(tf) == 0 or self.fusion_gamma == 0.0:
            return

        if self.fusion_mode == "frontier":
            self._fuse_frontier(drone_world)
            return

        if self.fusion_mode == "roi":
            self._fuse_roi(drone_world)
            return

        stuck = self._is_stuck(drone_world)

        if C.DECAY_ETA > 0.0:
            tf.decay_at(drone_world[0], drone_world[1], C.DECAY_ETA)

        if C.STUCK_GATE and not stuck:
            return

        best = tf.best_region(min_score=max(C.FUSION_SCORE_GATE, 1e-6),
                              recency_halflife=0.0)
        if best is None:
            return
        best_key, best_score, best_center = best
        self._last_best_key = best_key
        self._fusion_active_steps += 1

        wx, wy = self._local_world_xy()
        obstacle0 = self.obstacle_map  # (40,40,10)
        obstacle_col = np.any(obstacle0 >= 1.0, axis=2)  # (40,40) any obstacle in column

        mode = self.fusion_mode

        if mode == "floor":
            # Per-cell region score, capped + gated. Broad, non-directional.
            cell_size = tf.cell_size
            rx = np.floor(wx / cell_size).astype(int)
            ry = np.floor(wy / cell_size).astype(int)
            bonus = np.zeros(MAP_SHAPE[:2], dtype=np.float32)
            for key in set(zip(rx.ravel().tolist(), ry.ravel().tolist())):
                val = tf.get(key)
                if val >= C.FUSION_SCORE_GATE and val > 0.0:
                    bonus[(rx == key[0]) & (ry == key[1])] = val
            floor = np.minimum(C.FUSION_CAP, self.fusion_gamma * bonus)
            floor[obstacle_col] = 0.0
            if np.any(floor):
                cur = self.attraction_map[:, :, :, 0]
                self.attraction_map[:, :, :, 0] = np.maximum(cur, floor[:, :, None])
            return

        # gradient / explore / hybrid all use the directional wedge toward best_center
        weight, r = self._wedge_weight(wx, wy, drone_world, best_center)
        if not np.any(weight > 0.0):
            return

        if mode in ("gradient", "hybrid"):
            range_w = np.clip(r / _LOCAL_HALF_M, 0.0, 1.0)  # carrot: stronger far ahead
            amp = self.fusion_gamma * best_score
            cap = C.FUSION_CAP if mode == "gradient" else min(C.FUSION_CAP, 0.3)
            floor = np.minimum(cap, amp * weight * range_w).astype(np.float32)
            floor[obstacle_col] = 0.0
            if np.any(floor):
                cur = self.attraction_map[:, :, :, 0]
                self.attraction_map[:, :, :, 0] = np.maximum(cur, floor[:, :, None])

        if mode in ("explore", "hybrid"):
            # Pull exploration down *inside the wedge only* => action_predict's
            # (RATE_CENTER - expl) term rewards heading that way. Cells outside
            # the wedge (weight == 0) are left completely untouched.
            in_wedge = weight > 0.0
            pull = C.EXPLORE_DELTA * self.fusion_gamma * best_score * weight
            target_expl = np.maximum(0.0, RATE_CENTER - pull).astype(np.float32)
            col_target = np.where(in_wedge, target_expl, np.inf).astype(np.float32)
            self.exploration_map = np.minimum(self.exploration_map, col_target[:, :, None])
