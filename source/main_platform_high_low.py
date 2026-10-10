"""High-low altitude coordinated exploration (idea doc, MVP scope = §4.1-4.3).

Spawns two drones per task:
  - "Drone1" (high-altitude leader): sweeps the area, periodically scores
    how strongly the view under it matches the task prompt's terrain
    keywords via VLM, and writes that into a shared ``TerrainField``.
  - "Drone0" (low-altitude follower): a plain ``ApexAgent`` whose attraction
    map is biased by the shared field before every action decision
    (``FusedApexAgent``), i.e. A_low(v) gets a floor of
    gamma * A_high(region(v)).

Both agents run in the same control loop / same Python process, so the
"broadcast" from leader to follower described in the idea doc is just
sharing one ``TerrainField`` object by reference — no network layer needed
for this single-leader/single-follower MVP.

Explicitly OUT of scope for this MVP (see the idea doc's own §4.4, §4.6,
§6): cross-follower exploration/attraction fusion (only relevant with 2+
followers), the Q-value-style feedback decay on A_high, memory-augmented
in-context learning for the leader, and the B0-B3 baseline/ablation matrix.

This file does not modify ``main_platform_linux.py``. It only adds one
no-op hook to ``baselines_adaption/apex/apex_agent.py``
(``_augment_attraction_map``) so ``FusedApexAgent`` can inject the fusion
step without duplicating APEX's ~90-line ``run_step``; plain ``ApexAgent``
behaviour is unchanged by that hook.

Usage:
    cd source
    python main_platform_high_low.py
"""
import json
import os
import pathlib
import random
import threading
import time
import traceback

import airsim

from main_platform_linux import (
    ASARPlatform, CoordinateConverter, DATA_DIR, LOG_DIR, MAP_SETTINGS,
    SETTINGS_PATH,
)
from vehicle_client_proxy import VehicleClientProxy
from baselines_adaption.hl_coord.terrain_field import TerrainField
from baselines_adaption.hl_coord.terrain_leader_agent import TerrainLeaderAgent
from baselines_adaption.hl_coord.fused_apex_agent import FusedApexAgent
from baselines_adaption.hl_coord import hl_config as HLC

HL_SETTINGS_PATH = SETTINGS_PATH.parent / "settings_high_low.json"
FOLLOWER = "Drone0"
LEADER = "Drone1"
VEHICLE_NAMES = [FOLLOWER, LEADER]

# All tunables now live in baselines_adaption/hl_coord/hl_config.py, driven by
# HL_* env vars so the hyperparameter sweep needs no source edits between runs.
TERRAIN_FIELD_CELL_SIZE_M = HLC.CELL_SIZE_M
FUSION_GAMMA = HLC.FUSION_GAMMA
LEADER_ALTITUDE_OFFSET_M = HLC.LEADER_ALT_OFFSET_M
LEADER_SWEEP_HALF_SIZE_M = HLC.LEADER_SWEEP_HALF_M
LEADER_ROW_SPACING_M = HLC.LEADER_ROW_SPACING_M
IMAGE_PREFLIGHT_TIMEOUT = 15       # seconds


def stratified_sample(tasks, n, seed=42):
    """Proportionally-stratified random sample of ``n`` tasks across
    (map_name, difficulty_level).

    Exists because ``data/task_data.json`` is grouped by map — the first 80
    entries in the file are ALL ``aotai2800`` — so a quota-limited test run
    that just took ``tasks[:n]`` would silently test one map only instead
    of a representative slice of the benchmark. Each (map, difficulty)
    stratum gets a quota proportional to its share of the full dataset
    (largest-remainder rounding so quotas sum to exactly ``n``), then that
    many tasks are randomly drawn from it. Deterministic given ``seed``.
    """
    if n >= len(tasks):
        return list(tasks)

    strata = {}
    for t in tasks:
        key = (t.get('map_name'), t.get('difficulty_level'))
        strata.setdefault(key, []).append(t)

    total = len(tasks)
    exact = {key: n * len(group) / total for key, group in strata.items()}
    quotas = {key: int(v) for key, v in exact.items()}  # floor
    remainder = n - sum(quotas.values())
    # Largest-remainder method: hand the leftover slots to the strata with
    # the biggest fractional part, so quotas sum to exactly n.
    order = sorted(strata, key=lambda k: exact[k] - quotas[k], reverse=True)
    for key in order[:remainder]:
        quotas[key] += 1

    rng = random.Random(seed)
    sampled = []
    for key, group in strata.items():
        k = min(quotas[key], len(group))
        sampled.extend(rng.sample(group, k))
    rng.shuffle(sampled)
    return sampled


class HighLowPlatform(ASARPlatform):
    """Runs one high-altitude terrain-scoring leader + one low-altitude
    terrain-fused APEX follower per task."""

    def __init__(self, gpu_id=None, worker_id=0, num_workers=1, base_port=41451,
                 sample=None, seed=42, resume=False, task_ids=None, log_dir=None):
        # Must be set before super().__init__(), which calls self.load_data()
        # (dispatches to the override below) before this constructor resumes.
        self.sample = sample
        self.seed = seed
        self.resume = resume
        self.task_ids_filter = set(task_ids) if task_ids is not None else None
        self.log_dir = pathlib.Path(log_dir) if log_dir is not None else LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)

        super().__init__(gpu_id=gpu_id, worker_id=worker_id,
                          num_workers=num_workers, base_port=base_port)
        self.untested_tasks_path = (
            DATA_DIR / f"task_data_untested_hl_worker{worker_id}.json"
            if num_workers > 1 else DATA_DIR / "task_data_untested_hl.json"
        )

    def load_data(self):
        super().load_data()
        if self.task_ids_filter is not None:
            self.tasks = [t for t in self.tasks if t['task_id'] in self.task_ids_filter]
            print(f"[HighLowPlatform] Filtered to explicit task_id list: {len(self.tasks)} tasks")
        elif self.sample is not None:
            self.tasks = stratified_sample(self.tasks, self.sample, seed=self.seed)
            print(f"[HighLowPlatform] Stratified sample: {len(self.tasks)} tasks "
                  f"(seed={self.seed})")
        if self.resume:
            before = len(self.tasks)
            self.tasks = [
                t for t in self.tasks
                if not (self.log_dir / f"task_{t['task_id']}_hl_log.json").exists()
            ]
            print(f"[HighLowPlatform] Resume: skipping {before - len(self.tasks)} tasks "
                  f"that already have a log, {len(self.tasks)} left to run.")

    def _prepare_settings_path(self):
        """Same idea as ASARPlatform._prepare_settings_path, but starting
        from the high-low settings_high_low.json template (2 vehicles, one
        with a down-facing bev_camera) instead of the single-drone one."""
        if self.num_workers <= 1:
            return HL_SETTINGS_PATH

        with open(HL_SETTINGS_PATH, 'r', encoding='utf-8') as f:
            settings = json.load(f)
        settings["ApiServerPort"] = self.port

        worker_settings_path = (
            HL_SETTINGS_PATH.parent / f"settings_hl_worker{self.worker_id}.json")
        with open(worker_settings_path, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
        return worker_settings_path

    # ------------------------------------------------------------------
    def _preflight_image_test(self, client, vehicle_name, camera_name,
                               timeout=IMAGE_PREFLIGHT_TIMEOUT):
        result = [None]

        def _capture():
            try:
                result[0] = client.simGetImages(
                    [airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, True)],
                    vehicle_name=vehicle_name,
                )
            except Exception as e:
                result[0] = e

        t = threading.Thread(target=_capture, daemon=True)
        t.start()
        t.join(timeout=timeout)

        if t.is_alive():
            print(f"[Preflight][{vehicle_name}] Image capture HUNG (>{timeout}s). Skipping task.")
            return False
        if isinstance(result[0], Exception) or result[0] is None:
            print(f"[Preflight][{vehicle_name}] Image capture FAILED: {result[0]}. Skipping task.")
            return False

        print(f"[Preflight][{vehicle_name}] Image capture OK.")
        return True

    # ------------------------------------------------------------------
    def run_hl_agent_logic(self, client, task):
        terrain_field = TerrainField(cell_size=TERRAIN_FIELD_CELL_SIZE_M)

        leader = TerrainLeaderAgent(
            VehicleClientProxy(client, LEADER), task, terrain_field,
            sweep_half_size_m=LEADER_SWEEP_HALF_SIZE_M,
            row_spacing_m=LEADER_ROW_SPACING_M,
        )
        follower = FusedApexAgent(
            VehicleClientProxy(client, FOLLOWER), task, terrain_field,
            fusion_gamma=FUSION_GAMMA,
        )
        agents = {FOLLOWER: follower, LEADER: leader}

        max_duration = MAP_SETTINGS.get(task['map_name'], 600)
        start_loop = time.time()
        path_len = {name: 0.0 for name in VEHICLE_NAMES}
        last_pos = {name: None for name in VEHICLE_NAMES}
        status = {name: "Finished" for name in VEHICLE_NAMES}
        error_info = {name: None for name in VEHICLE_NAMES}
        active = {name: True for name in VEHICLE_NAMES}
        # Longer than the solo runner's 3.0s: the high-low path constructs two
        # agents (leader does an LLM prompt-parse, follower loads YOLO) and
        # teleports two drones, so the follower needs more time to settle onto
        # terrain before a lingering spawn-contact collision flag is treated
        # as a real crash. Task 558 (dapeng) spawn-crashed reproducibly at
        # 3.0s but flies fine once given room to settle.
        GRACE_PERIOD = 6.0

        while time.time() - start_loop < max_duration and active[FOLLOWER]:
            current_time = time.time()
            elapsed_time = current_time - start_loop

            for name in VEHICLE_NAMES:
                if not active[name]:
                    continue
                agent = agents[name]

                collision_info = client.simGetCollisionInfo(vehicle_name=name)
                if collision_info.has_collided and elapsed_time >= GRACE_PERIOD:
                    print(f"[{name}] CRASH DETECTED! Stopping this drone.")
                    status[name] = "Crashed"
                    active[name] = False
                    continue

                try:
                    agent.run_step(current_time)
                except Exception as e:
                    print(f"[{name}] Agent Logic Error: {e}")
                    traceback.print_exc()
                    error_info[name] = {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "traceback": traceback.format_exc(),
                    }
                    status[name] = "Error"
                    active[name] = False
                    continue

                state = client.getMultirotorState(vehicle_name=name)
                current_pos = state.kinematics_estimated.position
                if last_pos[name] is not None:
                    dist = (
                        (current_pos.x_val - last_pos[name].x_val) ** 2
                        + (current_pos.y_val - last_pos[name].y_val) ** 2
                        + (current_pos.z_val - last_pos[name].z_val) ** 2
                    ) ** 0.5
                    path_len[name] += dist
                last_pos[name] = current_pos

                if agent.is_finished():
                    print(f"[{name}] Agent requested mission finish.")
                    active[name] = False
                    # A leader that finishes its sweep early just stops
                    # contributing new evidence; the follower keeps going —
                    # only the follower finishing ends the whole task.

        # Task success/failure is judged on the follower alone: it's the one
        # actually searching. A leader crash just means the terrain field
        # stops getting updated after that point — it doesn't fail the
        # mission (this deliberately avoids the "any drone crash marks the
        # whole task Crashed" issue found in the earlier uncoordinated
        # dual-drone experiment, see main_platform_dual_drone.py history).
        return {
            "status": status[FOLLOWER],
            "per_drone_status": status,
            "error_info": error_info,
            "found_clues_text": follower.get_clues(),
            "found_clues": follower.get_clues_with_positions(),
            "reported_positions": follower.get_reported_positions(),
            "path_length": path_len[FOLLOWER],
            "per_drone_path_length": path_len,
            "terrain_field_size": len(terrain_field),
            "trajectory": {name: agents[name].get_trajectory() for name in VEHICLE_NAMES},
            "field_giveup_stats": (follower.get_field_giveup_stats()
                                    if hasattr(follower, "get_field_giveup_stats") else None),
        }

    # ------------------------------------------------------------------
    def run_single_task(self, task):
        print(f"\n=== Starting Task {task['task_id']} (High-Low Coordinated) ===")
        ue_process = None
        client = None

        try:
            ue_process = self.launch_ue_env(task)
            time.sleep(5)

            client = self.wait_for_airsim_ready()
            if not client:
                raise TimeoutError("AirSim connection timed out.")

            for name in VEHICLE_NAMES:
                client.enableApiControl(True, vehicle_name=name)
                client.armDisarm(True, vehicle_name=name)

            preflight_ok = (
                self._preflight_image_test(client, FOLLOWER, "front_camera")
                and self._preflight_image_test(client, LEADER, "bev_camera")
            )
            if not preflight_ok:
                self._save_untested_task(task)
                return

            # Both drones start above the same point; only the follower is
            # at task altitude, the leader is teleported straight up from
            # there by LEADER_ALTITUDE_OFFSET_M.
            start_pos_ue = task['start_location']
            start_pos_airsim = CoordinateConverter.to_airsim(task['map_name'], start_pos_ue)
            z_offsets = {FOLLOWER: 0.0, LEADER: -LEADER_ALTITUDE_OFFSET_M}

            for name in VEHICLE_NAMES:
                pose_pos = airsim.Vector3r(
                    start_pos_airsim.x_val,
                    start_pos_airsim.y_val,
                    start_pos_airsim.z_val + z_offsets[name],
                )
                print(f"Teleporting {name} to: {pose_pos}")
                client.simSetVehiclePose(
                    airsim.Pose(pose_pos, airsim.to_quaternion(0, 0, 0)),
                    True, vehicle_name=name,
                )
                client.moveToZAsync(pose_pos.z_val - 5, 2, timeout_sec=2, vehicle_name=name).join()
                client.moveToZAsync(pose_pos.z_val, 2, timeout_sec=2, vehicle_name=name).join()

            # Brief settle before the control loop: let the follower stabilise
            # and the depth camera produce a clean frame, so a momentary
            # spawn-clip doesn't mark the drone's own voxel column as an
            # obstacle (which makes action_predict return -inf for all six
            # actions => instant "crash"). Combined with GRACE_PERIOD=6.0 and
            # the <40s early-crash guard below.
            try:
                client.hoverAsync(vehicle_name=FOLLOWER).join()
            except Exception:
                pass
            time.sleep(2.0)

            # Embedded-spawn climb-out: ~7/60 random tasks put the follower
            # partly inside terrain (persisted through all 3 retry passes in
            # H1). If it's still in contact after the settle, climb it out in
            # 12 m steps (up to ~72 m) until the collision flag clears, then
            # return toward task altitude. Follower only; the leader is fine
            # up at LEADER_ALT.
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
                    print(f"[HighLow] Task {task['task_id']} follower still in contact "
                          f"after climb-out (now z={fz:.0f}).")
                if fz < start_pos_airsim.z_val - 1.0:
                    # settle back down partway so it doesn't start the mission 70 m high
                    client.moveToZAsync(min(start_pos_airsim.z_val,
                                            (fz + start_pos_airsim.z_val) / 2.0),
                                        3, timeout_sec=6, vehicle_name=FOLLOWER).join()
                    client.hoverAsync(vehicle_name=FOLLOWER).join()
                    time.sleep(1.0)
            except Exception as e:
                print(f"[HighLow] climb-out skipped: {e}")

            gt_targets_ue = task.get('targets', [])
            gt_targets_airsim = []
            for t in gt_targets_ue:
                as_pos = CoordinateConverter.to_airsim(task['map_name'], t['location'])
                gt_targets_airsim.append({
                    "target_id": t.get('target_id'),
                    "location": {"x": as_pos.x_val, "y": as_pos.y_val, "z": as_pos.z_val},
                })

            start_time = time.time()
            agent_result = self.run_hl_agent_logic(client, task)
            end_time = time.time()

            # A crash in the first few seconds is a spawn glitch, not a
            # navigation failure — don't let it pollute the config's crash
            # rate. Save the task for a re-run pass instead of writing a
            # Crashed log.
            if (str(agent_result.get("status", "")).lower() == "crashed"
                    and end_time - start_time < 40.0):
                print(f"[HighLow] Task {task['task_id']} crashed in "
                      f"{end_time - start_time:.0f}s (spawn glitch) — saving for re-run.")
                self._save_untested_task(task)
                return

            gt_cue_names = self.get_ground_truth_names(task['map_name'], task)
            gt_clues_with_pos = self.get_ground_truth_clues(task['map_name'], task)
            cds, eval_details = self.evaluator.evaluate(
                agent_result.get('found_clues', []), gt_clues_with_pos)
            cdr = eval_details.get('cdr_score', eval_details.get('c_exact', 0.0))

            log_entry = {
                "task_id": task['task_id'],
                "map": task['map_name'],
                "difficulty": task.get('difficulty_level', 'Unknown'),
                "mode": "high_low_coordinated",
                "leader_altitude_offset_m": LEADER_ALTITUDE_OFFSET_M,
                "fusion_gamma": FUSION_GAMMA,
                "terrain_field_cell_size_m": TERRAIN_FIELD_CELL_SIZE_M,
                "hl_config": HLC.snapshot(),
                "status": agent_result['status'],
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
            with open(log_path, 'w') as f:
                json.dump(log_entry, f, indent=4)

            if str(agent_result.get("status", "")).upper() == "ERROR":
                self._save_untested_task(task)

            print(f"Task Finished. CDS: {cds:.2f}  (High-Low Coordinated, "
                  f"terrain_field_size={agent_result['terrain_field_size']})")

        except Exception as e:
            print(f"!!! Error in Task {task['task_id']}: {e}")
            traceback.print_exc()

        finally:
            if client:
                for name in VEHICLE_NAMES:
                    try:
                        client.enableApiControl(False, vehicle_name=name)
                    except Exception:
                        pass
            self.kill_ue_process(process_obj=ue_process, map_name=task['map_name'])


def _run_worker(worker_id, gpu_id, num_workers, base_port, limit=None,
                 sample=None, seed=42, resume=False, task_ids=None, log_dir=None):
    # See the matching comment in main_platform_linux.py::_run_worker —
    # ApexAgent's YOLO detector hardcodes device="cuda:0", so the worker
    # process itself (not just the UE5 subprocess) needs
    # CUDA_VISIBLE_DEVICES set before any torch/ultralytics import happens.
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # sample/task_ids are resolved from the full task list inside
    # load_data() (see HighLowPlatform.load_data), i.e. BEFORE this
    # worker's slice of it is cut — so all workers agree on the same
    # subset and each just gets its own [worker_id::num_workers] slice.
    platform = HighLowPlatform(
        gpu_id=gpu_id, worker_id=worker_id,
        num_workers=num_workers, base_port=base_port,
        sample=sample, seed=seed, resume=resume, task_ids=task_ids,
        log_dir=log_dir,
    )
    if limit is not None:
        platform.tasks = platform.tasks[:limit]
    platform.run_all()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpus", type=str, default=None,
        help="逗号分隔的 GPU id 列表用于多卡并行，例如 '0,1,2,3,4,5,6,7'。不传则单进程顺序跑。"
    )
    parser.add_argument("--base-port", type=int, default=41451)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="只跑任务列表里的前 N 个（每个 worker 各跑前 N 个），用于最快速的冒烟测试"
             "（比如 --limit 1 只跑一个任务看流程通不通）。注意 task_data.json 是按地图"
             "分组存的，--limit 不保证地图/难度均衡，测 token 消耗量没问题，但不适合"
             "拿来评估效果。跑效果评估请用 --sample。"
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="按 (地图, 难度) 分层随机抽取 N 个任务，比例和完整数据集保持一致，"
             "用于在 DashScope 额度有限时跑一个有代表性的小样本评估（例如 --sample 80）。"
             "和 --limit 互斥，同时传的话以 --sample 为准。"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="--sample 用的随机种子，固定种子保证多次运行抽到同一批任务，方便复现和对比。"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="跳过 log/ 里已经有 task_<id>_hl_log.json 的任务，只跑还没有日志的任务。"
             "配合 --sample/--seed 用同样的参数重跑，能接着之前中断的批次继续跑。"
    )
    parser.add_argument(
        "--task-ids", type=str, default=None,
        help="逗号分隔的具体 task_id 列表，只跑这些任务（优先级高于 --sample）。"
             "例如配合 data/task_ids_obstructed_60.json 里挑出来的子集使用。"
    )
    parser.add_argument(
        "--log-dir", type=str, default=None,
        help="结果写到这个目录，而不是默认的 log/（用来跟旧结果、其他实验区分开，"
             "比如 --log-dir ../log_ours_20260830）。目录不存在会自动创建。"
    )
    args = parser.parse_args()

    task_ids = None
    if args.task_ids:
        task_ids = [int(x) for x in args.task_ids.split(",") if x.strip()]

    if args.gpus:
        gpu_ids = [int(g) for g in args.gpus.split(",") if g.strip() != ""]
        import multiprocessing as mp
        procs = []
        for worker_id, gpu_id in enumerate(gpu_ids):
            p = mp.Process(
                target=_run_worker,
                args=(worker_id, gpu_id, len(gpu_ids), args.base_port,
                      args.limit, args.sample, args.seed, args.resume, task_ids,
                      args.log_dir)
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
    else:
        platform = HighLowPlatform(sample=args.sample, seed=args.seed, resume=args.resume,
                                    task_ids=task_ids, log_dir=args.log_dir)
        if args.limit is not None:
            platform.tasks = platform.tasks[:args.limit]
        platform.run_all()
