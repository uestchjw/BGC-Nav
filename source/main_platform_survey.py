"""Survey-then-search runner: NO live leader.  The follower is a FusedApexAgent
in ``densemap`` mode, reading a pre-built full-coverage RelevanceGrid
(cache/relevance/<map>_task<id>.npz, produced offline by
baselines_adaption/hl_coord/build_relevance.py from the aerial frame banks).

Reuses HighLowPlatform's hardened run_single_task (grace 6 s, hover-settle,
embedded-spawn climb-out, <40 s early-crash guard, 2-vehicle settings) so the
numbers are comparable to the frontier/roi runs — only run_hl_agent_logic is
overridden to drop the leader.

    cd source
    python main_platform_survey.py --gpus 0,1,2,3,4,5,6,7 \
        --task-ids "$(python -c 'import json;print(",".join(map(str,json.load(open("../data/task_ids_apex_fail_72.json"))["task_ids"])))')" \
        --relevance-dir ../cache/relevance --log-dir ../log_hl_densemap
"""
import json
import os
import pathlib
import time
import traceback

import airsim

from main_platform_linux import MAP_SETTINGS, CoordinateConverter
from main_platform_high_low import (
    HighLowPlatform, FOLLOWER, LEADER, VEHICLE_NAMES, FUSION_GAMMA,
    HL_SETTINGS_PATH, TERRAIN_FIELD_CELL_SIZE_M, LEADER_ALTITUDE_OFFSET_M, HLC,
)
from vehicle_client_proxy import VehicleClientProxy

# Single-vehicle settings template: only Drone0 (the follower) + its 4 cameras,
# no idle Drone1/bev_camera. Removes the "2nd drone spawned but never used"
# confound and halves the teleport-spawn-clip probability. Set
# HL_SURVEY_SINGLE_VEHICLE=0 to fall back to the 2-vehicle settings_high_low.json.
SINGLE_VEHICLE = str(os.environ.get("HL_SURVEY_SINGLE_VEHICLE", "1")).strip().lower() in ("1", "true", "yes", "on")
SURVEY_SINGLE_SETTINGS_PATH = HL_SETTINGS_PATH.parent / "settings_survey_single.json"
from baselines_adaption.hl_coord.fused_apex_agent import FusedApexAgent
from baselines_adaption.hl_coord.fused_fbe_agent import FusedFBEAgent
from baselines_adaption.hl_coord.fused_vlfm_agent import FusedVLFMAgent
from baselines_adaption.hl_coord.aerial_bank import RelevanceGrid

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_RELEVANCE_DIR = ROOT / "cache" / "relevance"


def _spawn_offset(task_id):
    """(dx, dy) AirSim-frame spawn shift for this task. Priority:
       HL_SPAWN_OFFSET_FILE (JSON {task_id: [dx, dy]}, per-task) >
       HL_SPAWN_OFFSET_X_M / HL_SPAWN_OFFSET_Y_M (constant) > (0, 0)."""
    f = os.environ.get("HL_SPAWN_OFFSET_FILE")
    if f and os.path.exists(f):
        try:
            m = json.load(open(f))
            v = m.get(str(task_id))
            if v:
                return float(v[0]), float(v[1])
        except Exception:
            pass
    return (float(os.environ.get("HL_SPAWN_OFFSET_X_M", "0")),
            float(os.environ.get("HL_SPAWN_OFFSET_Y_M", "0")))


class SurveyPlatform(HighLowPlatform):
    """Follower-only runner for the amortised-survey methods:
       fusion_mode densemap / aeroduo -> load a cached grid (.npz)
       fusion_mode conav              -> load a cached candidate queue (.json)"""

    def __init__(self, *a, relevance_dir=None, fusion_mode="densemap",
                 decomp_dir=None, agent_kind="apex", **kw):
        self.relevance_dir = pathlib.Path(relevance_dir) if relevance_dir else DEFAULT_RELEVANCE_DIR
        self.decomp_dir = pathlib.Path(decomp_dir) if decomp_dir else None
        self.fusion_mode = fusion_mode
        self.agent_kind = agent_kind          # "apex" (FusedApexAgent) | "fbe" (FusedFBEAgent)
        super().__init__(*a, **kw)

    def _prepare_settings_path(self):
        """Single-vehicle survey settings (Drone0 only). Falls back to the
        HighLowPlatform 2-vehicle template if HL_SURVEY_SINGLE_VEHICLE=0."""
        if not SINGLE_VEHICLE:
            return super()._prepare_settings_path()
        with open(SURVEY_SINGLE_SETTINGS_PATH, "r", encoding="utf-8") as f:
            settings = json.load(f)
        if self.num_workers <= 1:
            settings["ApiServerPort"] = self.port
            out = SURVEY_SINGLE_SETTINGS_PATH
        else:
            settings["ApiServerPort"] = self.port
            out = SURVEY_SINGLE_SETTINGS_PATH.parent / f"settings_survey_worker{self.worker_id}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
        return out

    def run_single_task(self, task):
        """Single-vehicle version of HighLowPlatform.run_single_task: only the
        follower (Drone0) is spawned / teleported / preflighted. No leader.
        A <40 s crash is still flagged (spawn glitch) but the log IS written
        (status Crashed, spawn_glitch=true) instead of being re-queued —
        stops the forever-retry livelock on embedded ESARBench start points."""
        if not SINGLE_VEHICLE:
            return super().run_single_task(task)

        print(f"\n=== Starting Task {task['task_id']} (Survey, single-vehicle) ===")
        ue_process = None
        client = None
        try:
            ue_process = self.launch_ue_env(task)
            time.sleep(5)
            client = self.wait_for_airsim_ready()
            if not client:
                raise TimeoutError("AirSim connection timed out.")

            client.enableApiControl(True, vehicle_name=FOLLOWER)
            client.armDisarm(True, vehicle_name=FOLLOWER)

            if not self._preflight_image_test(client, FOLLOWER, "front_camera"):
                self._save_untested_task(task)
                return

            start_pos_ue = task["start_location"]
            start_pos_airsim = CoordinateConverter.to_airsim(task["map_name"], start_pos_ue)
            _dx, _dy = _spawn_offset(task["task_id"])
            if _dx or _dy:
                start_pos_airsim = airsim.Vector3r(
                    start_pos_airsim.x_val + _dx, start_pos_airsim.y_val + _dy,
                    start_pos_airsim.z_val)
            pose_pos = airsim.Vector3r(start_pos_airsim.x_val,
                                       start_pos_airsim.y_val,
                                       start_pos_airsim.z_val)
            print(f"Teleporting {FOLLOWER} to: {pose_pos}")
            client.simSetVehiclePose(
                airsim.Pose(pose_pos, airsim.to_quaternion(0, 0, 0)),
                True, vehicle_name=FOLLOWER,
            )
            client.moveToZAsync(pose_pos.z_val - 5, 2, timeout_sec=2, vehicle_name=FOLLOWER).join()
            client.moveToZAsync(pose_pos.z_val, 2, timeout_sec=2, vehicle_name=FOLLOWER).join()

            try:
                client.hoverAsync(vehicle_name=FOLLOWER).join()
            except Exception:
                pass
            time.sleep(2.0)

            # Embedded-spawn climb-out (follower only) — same as the HL version.
            try:
                fz = start_pos_airsim.z_val
                for _ in range(6):
                    ci = client.simGetCollisionInfo(vehicle_name=FOLLOWER)
                    if not ci.has_collided:
                        break
                    fz -= 12.0
                    client.moveToZAsync(fz, 3, timeout_sec=5, vehicle_name=FOLLOWER).join()
                    time.sleep(0.5)
                else:
                    print(f"[Survey] Task {task['task_id']} follower still in contact "
                          f"after climb-out (now z={fz:.0f}).")
                if fz < start_pos_airsim.z_val - 1.0:
                    # Climb back toward the task's intended altitude in small,
                    # collision-checked steps rather than jumping straight to
                    # the midpoint between the escape depth and the intended
                    # height. A blind midpoint jump permanently strands the
                    # follower several metres closer to terrain than the task
                    # intended -- and APEX's obstacle_map is write-once/never
                    # decays and is world-frame-fixed (rotating never changes
                    # what "forward" means or clears a stale mark), so a dirty
                    # first depth read taken from that closer altitude can
                    # wedge the follower in an unrecoverable "every heading
                    # reads as blocked" state for the rest of the episode
                    # (root-caused on task 33/42, which spawn embedded here).
                    safe_z = fz
                    target_z = start_pos_airsim.z_val
                    while safe_z > target_z + 1.0:
                        step_z = max(target_z, safe_z - 4.0)
                        client.moveToZAsync(step_z, 2, timeout_sec=4,
                                            vehicle_name=FOLLOWER).join()
                        if client.simGetCollisionInfo(vehicle_name=FOLLOWER).has_collided:
                            client.moveToZAsync(safe_z, 2, timeout_sec=4,
                                                vehicle_name=FOLLOWER).join()
                            break
                        safe_z = step_z
                    client.hoverAsync(vehicle_name=FOLLOWER).join()
                    time.sleep(1.0)
            except Exception as e:
                print(f"[Survey] climb-out skipped: {e}")

            gt_targets_ue = task.get("targets", [])
            gt_targets_airsim = []
            for t in gt_targets_ue:
                as_pos = CoordinateConverter.to_airsim(task["map_name"], t["location"])
                gt_targets_airsim.append({
                    "target_id": t.get("target_id"),
                    "location": {"x": as_pos.x_val, "y": as_pos.y_val, "z": as_pos.z_val},
                })

            start_time = time.time()
            agent_result = self.run_hl_agent_logic(client, task)
            end_time = time.time()

            spawn_glitch = (str(agent_result.get("status", "")).lower() == "crashed"
                            and end_time - start_time < 40.0)
            if spawn_glitch:
                print(f"[Survey] Task {task['task_id']} crashed in "
                      f"{end_time - start_time:.0f}s (spawn glitch) — logging as Crashed "
                      f"(single-vehicle, no re-queue).")

            gt_cue_names = self.get_ground_truth_names(task["map_name"], task)
            gt_clues_with_pos = self.get_ground_truth_clues(task["map_name"], task)
            cds, eval_details = self.evaluator.evaluate(
                agent_result.get("found_clues", []), gt_clues_with_pos)
            cdr = eval_details.get("cdr_score", eval_details.get("c_exact", 0.0))

            log_entry = {
                "task_id": task["task_id"],
                "map": task["map_name"],
                "difficulty": task.get("difficulty_level", "Unknown"),
                "mode": "survey_single_vehicle",
                "spawn_glitch": bool(spawn_glitch),
                "leader_altitude_offset_m": LEADER_ALTITUDE_OFFSET_M,
                "fusion_gamma": FUSION_GAMMA,
                "terrain_field_cell_size_m": TERRAIN_FIELD_CELL_SIZE_M,
                "hl_config": HLC.snapshot(),
                "status": agent_result["status"],
                "time_cost": end_time - start_time,
                "cdr_score": cdr,
                "cds_score": cds,
                "error_info": agent_result.get("error_info"),
                "agent_output": agent_result,
                "ground_truth": {"targets": gt_targets_airsim},
                "ground_truth_cues": gt_cue_names,
                "ground_truth_clues_with_positions": gt_clues_with_pos,
            }
            log_path = self.log_dir / f"task_{task['task_id']}_hl_log.json"
            with open(log_path, "w") as f:
                json.dump(log_entry, f, indent=4)

            if str(agent_result.get("status", "")).upper() == "ERROR":
                self._save_untested_task(task)
            print(f"Task Finished. CDS: {cds:.2f}  (Survey single-vehicle)")

        except Exception as e:
            print(f"!!! Error in Task {task['task_id']}: {e}")
            traceback.print_exc()
        finally:
            if client:
                try:
                    client.enableApiControl(False, vehicle_name=FOLLOWER)
                except Exception:
                    pass
            self.kill_ue_process(process_obj=ue_process, map_name=task["map_name"])

    def _load_decomp(self, task):
        if not self.decomp_dir:
            return None
        p = self.decomp_dir / f"{task['map_name']}_task{task['task_id']}.json"
        if not p.exists():
            return None
        return json.load(open(p))

    def _load_repr(self, task):
        """Returns (dense_grid, queue) — one is None depending on the mode."""
        stem = f"{task['map_name']}_task{task['task_id']}"
        if self.fusion_mode == "conav":
            p = self.relevance_dir / f"{stem}.json"
            if not p.exists():
                print(f"[survey] NO conav queue {p.name} -> plain-APEX fallback")
                return None, []
            q = json.load(open(p)).get("queue", [])
            print(f"[survey] loaded {p.name}: {len(q)} queue entries, top={q[:2]}")
            return None, q
        p = self.relevance_dir / f"{stem}.npz"
        if not p.exists():
            print(f"[survey] NO grid {p.name} -> plain-APEX fallback ({self.fusion_mode} no-op)")
            return None, []
        g = RelevanceGrid.load(str(p))
        print(f"[survey] loaded {p.name}: {g.H}x{g.W}@{g.cell:.0f}m cov {g.coverage:.0%} "
              f"mean {g.relevance.mean():.2f} max {g.relevance.max():.2f}")
        return g, []

    def run_hl_agent_logic(self, client, task):
        grid, queue = self._load_repr(task)
        if self.agent_kind == "fbe":
            # Same aerial-relevance wrapper, second independent explorer.
            follower = FusedFBEAgent(
                VehicleClientProxy(client, FOLLOWER), task,
                relevance_grid=grid, fusion_gamma=FUSION_GAMMA,
            )
        elif self.agent_kind == "vlfm":
            # Same aerial-relevance wrapper, third independent explorer (VLFM).
            follower = FusedVLFMAgent(
                VehicleClientProxy(client, FOLLOWER), task, relevance_grid=grid,
            )
        else:
            follower = FusedApexAgent(
                VehicleClientProxy(client, FOLLOWER), task, terrain_field=None,
                dense_grid=grid, queue=queue, fusion_mode=self.fusion_mode,
                fusion_gamma=FUSION_GAMMA, decomp=self._load_decomp(task),
            )

        max_duration = (MAP_SETTINGS.get(task['map_name'], 600)
                        + int(os.environ.get("HL_EXTRA_DURATION_S", "0")))
        if os.environ.get("HL_MAX_DURATION_S"):
            # Hard override (not additive) — used for short probe runs (e.g.
            # spawn/config audits) that don't need a full-length episode.
            max_duration = min(max_duration, int(os.environ["HL_MAX_DURATION_S"]))
        start_loop = time.time()
        path_len = 0.0
        last_pos = None
        status = "Finished"
        error_info = None
        GRACE_PERIOD = 6.0

        while time.time() - start_loop < max_duration:
            current_time = time.time()
            elapsed = current_time - start_loop

            ci = client.simGetCollisionInfo(vehicle_name=FOLLOWER)
            if ci.has_collided and elapsed >= GRACE_PERIOD:
                print(f"[{FOLLOWER}] CRASH DETECTED!")
                status = "Crashed"
                break

            try:
                follower.run_step(current_time)
            except Exception as e:
                print(f"[{FOLLOWER}] Agent Logic Error: {e}")
                traceback.print_exc()
                error_info = {"error_type": type(e).__name__, "error_message": str(e),
                              "traceback": traceback.format_exc()}
                status = "Error"
                break

            st = client.getMultirotorState(vehicle_name=FOLLOWER)
            cp = st.kinematics_estimated.position
            if last_pos is not None:
                path_len += ((cp.x_val - last_pos.x_val) ** 2
                             + (cp.y_val - last_pos.y_val) ** 2
                             + (cp.z_val - last_pos.z_val) ** 2) ** 0.5
            last_pos = cp

            if follower.is_finished():
                print(f"[{FOLLOWER}] Agent requested mission finish.")
                break

        return {
            "status": status,
            "per_drone_status": {FOLLOWER: status, LEADER: "n/a"},
            "error_info": {FOLLOWER: error_info},
            "found_clues_text": follower.get_clues(),
            "found_clues": follower.get_clues_with_positions(),
            "reported_positions": follower.get_reported_positions(),
            "path_length": path_len,
            "per_drone_path_length": {FOLLOWER: path_len},
            "terrain_field_size": (0 if grid is None else int((grid.obs_cnt > 0).sum())),
            "dense_grid_coverage": (0.0 if grid is None else grid.coverage),
            "trajectory": {FOLLOWER: follower.get_trajectory()},
            "field_giveup_stats": (follower.get_field_giveup_stats()
                                    if hasattr(follower, "get_field_giveup_stats") else None),
        }


def _run_worker(worker_id, gpu_id, num_workers, base_port, task_ids, log_dir,
                relevance_dir, resume, fusion_mode, decomp_dir, agent_kind):
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    plat = SurveyPlatform(
        gpu_id=gpu_id, worker_id=worker_id, num_workers=num_workers,
        base_port=base_port, task_ids=task_ids, log_dir=log_dir,
        relevance_dir=relevance_dir, resume=resume, fusion_mode=fusion_mode,
        decomp_dir=decomp_dir, agent_kind=agent_kind,
    )
    plat.run_all()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=str, default=None)
    ap.add_argument("--base-port", type=int, default=41451)
    ap.add_argument("--task-ids", type=str, default=None)
    ap.add_argument("--log-dir", type=str, default=None)
    ap.add_argument("--relevance-dir", type=str, default=str(DEFAULT_RELEVANCE_DIR))
    ap.add_argument("--fusion-mode", choices=["densemap", "conav", "aeroduo", "field"],
                    default="densemap")
    ap.add_argument("--agent", choices=["apex", "fbe", "vlfm"], default="apex",
                    help="follower explorer: apex (default), fbe, or vlfm — 'fbe'/"
                         "'vlfm' apply the same relevance-bias wrapper to that explorer")
    ap.add_argument("--decomp-dir", type=str, default=None,
                    help="dir of role-decomposition JSONs for the follower half")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    tids = [int(x) for x in a.task_ids.split(",") if x.strip()] if a.task_ids else None
    if a.agent in ("fbe", "vlfm") and a.fusion_mode == "conav":
        ap.error(f"--agent {a.agent} does not support --fusion-mode conav (needs a grid)")

    if a.gpus:
        gids = [int(g) for g in a.gpus.split(",") if g.strip() != ""]
        import multiprocessing as mp
        ps = []
        for wid, gid in enumerate(gids):
            p = mp.Process(target=_run_worker,
                           args=(wid, gid, len(gids), a.base_port, tids,
                                 a.log_dir, a.relevance_dir, a.resume, a.fusion_mode,
                                 a.decomp_dir, a.agent))
            p.start(); ps.append(p)
        for p in ps:
            p.join()
    else:
        SurveyPlatform(task_ids=tids, log_dir=a.log_dir, relevance_dir=a.relevance_dir,
                       resume=a.resume, fusion_mode=a.fusion_mode,
                       decomp_dir=a.decomp_dir, agent_kind=a.agent).run_all()
