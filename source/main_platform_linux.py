import os
import json
import time
import signal
import subprocess
import threading
import airsim
import pathlib
import numpy as np
import sys
import traceback

from llm_evaluator import ClueEvaluator
import coordinate_utils as _coordinate_utils

# SemExp baseline
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.absolute()))
from baselines_adaption.semexp.semexp_agent import SemExpAgent

# NavGPT baseline
from baselines_adaption.navgpt.navgpt_agent import NavGPTAgent

# UniGoal baseline
from baselines_adaption.unigoal.unigoal_agent import UniGoalAgent

# SPF baseline
from baselines_adaption.SPF.spf_agent import SPFAgent

# APEX baseline
from baselines_adaption.apex.apex_agent import ApexAgent

# VLFM baseline
from baselines_adaption.vlfm.vlfm_agent import VLFMAgent

# Basic baseline
from baselines_adaption.base.random_agent import RandomAgent
from baselines_adaption.base.fbe_agent import FBEAgent
from baselines_adaption.base.mllm_agent import MLLMAgent

# ================= Settings =================
MAP_SETTINGS = {
    "aotai2800": 10 * 60,
    "dapeng":    10 * 60,
    "k2":        10 * 60,
    "lopnur":    15 * 60
}

SOURCE_DIR = pathlib.Path(__file__).parent.absolute()
ROOT_DIR = SOURCE_DIR.parent
DATA_DIR = ROOT_DIR / "data"
UE_PACKAGE_DIR = ROOT_DIR / "UEPackage"
# HL_LOG_DIR env override lets a run write its logs to a named directory
# (e.g. a time-budget ablation) without touching the default log/.
LOG_DIR = (pathlib.Path(os.environ["HL_LOG_DIR"]).expanduser()
           if os.environ.get("HL_LOG_DIR") else ROOT_DIR / "log")
SETTINGS_PATH = ROOT_DIR / "settings" / "settings.json"

# --- Bugfix: aotai2800 Vulkan rendering hang (A100 headless) ---
CLEAN_LAUNCHER = SOURCE_DIR / "launch_ue_clean.sh"
UNTESTED_TASKS_PATH = DATA_DIR / "task_data_untested.json"
IMAGE_PREFLIGHT_TIMEOUT = 15  # seconds to wait before declaring image capture hung
# --- End bugfix constants ---

LOG_DIR.mkdir(exist_ok=True)

# ================= Utils =================
class CoordinateConverter:
    """Wraps coordinate_utils.CoordinateConverter to return airsim.Vector3r."""

    PLAYER_STARTS = _coordinate_utils.CoordinateConverter.PLAYER_STARTS

    @staticmethod
    def to_airsim(map_name, ue_pos_dict):
        """
        输入: ue_pos_dict {'x': 123, 'y': 456, 'z': 789} (UE cm)
        输出: airsim.Vector3r (AirSim m, NED frame)
        """
        d = _coordinate_utils.CoordinateConverter.to_airsim(map_name, ue_pos_dict)
        return airsim.Vector3r(d["x"], d["y"], d["z"])

# ================= Core =================
class ASARPlatform:
    LOG_SUFFIX = "_log"  # log filename: task_<id>{LOG_SUFFIX}.json; subclasses override to avoid collisions
    SETTINGS_WORKER_PREFIX = "settings_worker"  # per-worker settings.json filename prefix; override to avoid collisions if two experiments' workers may run concurrently

    def __init__(self, gpu_id=None, worker_id=0, num_workers=1, base_port=41451):
        self.gpu_id = gpu_id
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.port = base_port + worker_id

        self.load_data()
        if num_workers > 1:
            # 按 worker 数量对任务分片，每个 worker 只跑属于自己的那一份
            self.tasks = self.tasks[worker_id::num_workers]

        self.settings_path = self._prepare_settings_path()
        self.untested_tasks_path = (
            DATA_DIR / f"task_data_untested_worker{worker_id}.json"
            if num_workers > 1 else UNTESTED_TASKS_PATH
        )

        self.evaluator = ClueEvaluator()

    def _prepare_settings_path(self):
        """为并行 worker 生成独立的 settings.json（不同 ApiServerPort），避免端口冲突。"""
        if self.num_workers <= 1:
            return SETTINGS_PATH

        with open(SETTINGS_PATH, 'r', encoding='utf-8') as f:
            settings = json.load(f)
        settings["ApiServerPort"] = self.port

        worker_settings_path = (
            SETTINGS_PATH.parent / f"{self.SETTINGS_WORKER_PREFIX}{self.worker_id}.json")
        with open(worker_settings_path, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
        return worker_settings_path

    def load_data(self):
        with open(DATA_DIR / "task_data.json", 'r', encoding='utf-8') as f:
            self.tasks = json.load(f)
        
        with open(DATA_DIR / "cue_list.json", 'r', encoding='utf-8') as f:
            self.cue_mapping_data = json.load(f)
            
        self.cue_dict = {}
        for map_data in self.cue_mapping_data:
            m_name = map_data['map_name']
            self.cue_dict[m_name] = {}
            all_items = map_data.get('cues', []) + map_data.get('fire', [])
            for item in all_items:
                self.cue_dict[m_name][item['id']] = item['name']

    def get_ground_truth_names(self, map_name, cue_list_in_task):
        """Backward-compat: return deduplicated clue names."""
        names = []
        all_cues = cue_list_in_task.get('cues', []) + cue_list_in_task.get('fire', [])
        for cue in all_cues:
            c_id = cue['id']
            if map_name in self.cue_dict and c_id in self.cue_dict[map_name]:
                names.append(self.cue_dict[map_name][c_id])
            else:
                names.append(f"Unknown_ID_{c_id}")
        return list(set(names))

    def get_ground_truth_clues(self, map_name, cue_list_in_task):
        """Return each GT clue instance with name AND AirSim position.

        Returns list of {"name": str, "position": {"x":…,"y":…,"z":…}}.
        """
        clues = []
        all_cues = cue_list_in_task.get('cues', []) + cue_list_in_task.get('fire', [])
        for cue in all_cues:
            c_id = cue['id']
            if map_name in self.cue_dict and c_id in self.cue_dict[map_name]:
                name = self.cue_dict[map_name][c_id]
            else:
                name = f"Unknown_ID_{c_id}"
            # Convert UE coordinates to AirSim NED
            ue_loc = cue.get('location')
            if ue_loc:
                as_pos = _coordinate_utils.CoordinateConverter.to_airsim(
                    map_name, ue_loc)
            else:
                as_pos = {"x": 0.0, "y": 0.0, "z": 0.0}
            clues.append({"name": name, "position": as_pos})
        return clues

    def resolve_ue_launch_cmd(self, map_name):
        map_dir = UE_PACKAGE_DIR / map_name
        candidates = [
            map_dir / "Linux" / f"{map_name}.sh",
            map_dir / "Linux" / map_name / "Binaries" / "Linux" / f"{map_name}-Linux-Shipping",
            map_dir / "Linux" / map_name / "Binaries" / "Linux" / map_name,
        ]

        for candidate in candidates:
            if candidate.exists():
                if candidate.suffix == ".sh":
                    return ["bash", str(candidate)]
                return [str(candidate)]

        raise FileNotFoundError(f"Cannot find launch file for map: {map_name} under {map_dir}")

    def launch_ue_env(self, task):
        map_name = task['map_name']

        # --- Bugfix: use clean launcher for aotai2800 (Vulkan hang workaround) ---
        if map_name == "aotai2800" and CLEAN_LAUNCHER.exists():
            map_sh = UE_PACKAGE_DIR / map_name / "Linux" / f"{map_name}.sh"
            cmd = [
                "bash", str(CLEAN_LAUNCHER), str(map_sh),
                f"-TaskNum={task['snapshot_id'] % 15}",
                f"-Weather={task['weather_id']}",
                f"-Time={task['time_of_day']}",
                f"-settings={str(self.settings_path)}",
                "-windowed", "-ResX=640", "-ResY=480"
            ]
        # --- End bugfix ---
        else:
            adapter_idx = self.gpu_id if self.gpu_id is not None else 0
            cmd = self.resolve_ue_launch_cmd(map_name) + [
                f"-TaskNum={task['snapshot_id'] % 15}",
                f"-Weather={task['weather_id']}",
                f"-Time={task['time_of_day']}",
                f"-settings={str(self.settings_path)}",
                "-RenderOffScreen",
                f"-GraphicsAdapter={adapter_idx}",
                "-windowed", "-ResX=640", "-ResY=480"
            ]

        # 多卡并行：实测这台机器上 CUDA_VISIBLE_DEVICES 对 Vulkan/UE5 渲染选卡不起作用，
        # 所以真正的选卡靠上面显式传入的 -GraphicsAdapter=<物理 GPU 编号>（aotai2800 分支
        # 则靠 launch_ue_clean.sh 里的 BEST_GPU，见该脚本注释）。这里仍设置
        # CUDA_VISIBLE_DEVICES，只是为了让子进程里任何 CUDA-aware 的组件也绑定到同一张卡。
        env = os.environ.copy()
        if self.gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        print(f"Launching UE5: {map_name} (Task {task['task_id']}) "
              f"[worker={self.worker_id} gpu={self.gpu_id} port={self.port}]")
        process = subprocess.Popen(cmd, start_new_session=True, env=env)
        return process

    def kill_ue_process(self, process_obj=None, map_name=None):
        print(f"Executing Force Kill for Map: {map_name}...")

        if process_obj:
            try:
                os.killpg(process_obj.pid, signal.SIGTERM)
                time.sleep(1)
                if process_obj.poll() is None:
                    os.killpg(process_obj.pid, signal.SIGKILL)
            except Exception as e:
                print(f"Warning: Failed to kill by PID: {e}")

        if map_name:
            if self.num_workers > 1:
                # 并行模式下多个 worker 可能同时跑同一张地图，pkill 必须按本 worker 的
                # settings 文件路径收窄匹配范围，否则会误杀其他 GPU 上的 UE5 进程。
                settings_marker = str(self.settings_path)
                subprocess.call(
                    ["pkill", "-f", f"{map_name}.*{settings_marker}"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            else:
                subprocess.call(
                    ["pkill", "-f", f"{map_name}-Linux-Shipping"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                subprocess.call(
                    ["pkill", "-f", f"/{map_name}.sh"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )

        time.sleep(3)

    def wait_for_airsim_ready(self, timeout_sec=60):
        print(f"Waiting for AirSim connection on port {self.port}...")
        start_time = time.time()
        while time.time() - start_time < timeout_sec:
            try:
                client = airsim.MultirotorClient(port=self.port, timeout_value=2)
                client.confirmConnection()
                print("AirSim connected successfully!")
                return client
            except Exception:
                print(f"Connection retry... ({int(time.time() - start_time)}s)")
                time.sleep(3)
        return None

    # --- Bugfix: preflight image capture test & untested task handling ---
    def _preflight_image_test(self, client, timeout=IMAGE_PREFLIGHT_TIMEOUT):
        """Try a small image capture with a timeout. Returns True if OK."""
        result = [None]

        def _capture():
            try:
                result[0] = client.simGetImages([
                    airsim.ImageRequest("0", airsim.ImageType.Scene, False, True),
                ])
            except Exception as e:
                result[0] = e

        t = threading.Thread(target=_capture, daemon=True)
        t.start()
        t.join(timeout=timeout)

        if t.is_alive():
            print(f"[Preflight] Image capture HUNG (>{timeout}s). Skipping task.")
            return False
        if isinstance(result[0], Exception) or result[0] is None:
            print(f"[Preflight] Image capture FAILED: {result[0]}. Skipping task.")
            return False

        print("[Preflight] Image capture OK.")
        return True

    def _save_untested_task(self, task):
        """Append task to this worker's untested-tasks file for later retry."""
        if self.untested_tasks_path.exists():
            with open(self.untested_tasks_path, 'r', encoding='utf-8') as f:
                untested = json.load(f)
        else:
            untested = []
        untested.append(task)
        with open(self.untested_tasks_path, 'w', encoding='utf-8') as f:
            json.dump(untested, f, indent=4, ensure_ascii=False)
        print(f"[Preflight] Task {task['task_id']} saved to {self.untested_tasks_path.name}")
    # --- End bugfix ---

    def _create_agent(self, client, task):
        """Hook for subclasses to swap in a different single-drone agent
        without duplicating the rest of run_agent_logic. Defaults to the
        FBE baseline (current behaviour, unchanged)."""
        return FBEAgent(client, task)

    def run_agent_logic(self, client, task):
        task_prompt = task.get("prompt", "")
        print(f"Agent executing task: {task_prompt if task_prompt else '[No prompt provided]'}")

        agent = self._create_agent(client, task)

        # --- Agent Loop ---
        max_duration = (MAP_SETTINGS.get(task['map_name'], 600)
                        + int(os.environ.get("HL_EXTRA_DURATION_S", "0")))
        start_loop = time.time()
        total_path_len = 0.0
        last_pos = None
        
        status = "Finished"
        error_info = None
        GRACE_PERIOD = 3.0

        while time.time() - start_loop < max_duration:
            current_time = time.time()
            elapsed_time = current_time - start_loop

            # A. 平台级安全检查 (Collision Check)
            collision_info = client.simGetCollisionInfo()
            if collision_info.has_collided:
                if elapsed_time >= GRACE_PERIOD:
                    print("CRASH DETECTED! Mission Failed.")
                    status = "Crashed"
                    break
            
            # B. 执行 Agent 的单步逻辑
            try:
                agent.run_step(current_time)
            except Exception as e:
                print(f"Agent Logic Error: {e}")
                traceback.print_exc()
                error_info = {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "traceback": traceback.format_exc(),
                }
                status = "Error"
                break

            # C. 累加距离
            state = client.getMultirotorState()
            current_pos = state.kinematics_estimated.position
            
            if last_pos:
                dist = np.linalg.norm([
                    current_pos.x_val - last_pos.x_val,
                    current_pos.y_val - last_pos.y_val,
                    current_pos.z_val - last_pos.z_val
                ])
                total_path_len += dist
            last_pos = current_pos

            # D. 检查 Agent 是否主动要求结束
            if agent.is_finished():
                print("Agent requested mission finish.")
                break

        return {
            "status": status,
            "error_info": error_info,
            "found_clues_text": agent.get_clues(),
            "found_clues": agent.get_clues_with_positions(),
            "path_length": total_path_len,
            "reported_positions": agent.get_reported_positions(),
            "trajectory": agent.get_trajectory()
        }

    def run_single_task(self, task):
        print(f"\n=== Starting Task {task['task_id']} ===")
        ue_process = None
        client = None
        
        try:
            # 1. 启动环境
            ue_process = self.launch_ue_env(task)
            time.sleep(5)

            # 2. 连接 AirSim
            client = self.wait_for_airsim_ready()
            if not client:
                raise TimeoutError("AirSim connection timed out.")

            client.enableApiControl(True)
            client.armDisarm(True)

            # --- Bugfix: preflight check before teleport ---
            if not self._preflight_image_test(client):
                self._save_untested_task(task)
                return
            # --- End bugfix ---

            # 3. 传送位置 (Agent 起始点)
            start_pos_ue = task['start_location']
            start_pos_airsim = CoordinateConverter.to_airsim(task['map_name'], start_pos_ue)
            # HL_SPAWN_OFFSET_Y_M: shift the spawn along AirSim-Y — used to run a
            # second, independent full-budget drone from an offset start point
            # (fair multi-drone: each drone gets its own 600 s / ~26 steps).
            _off = float(os.environ.get("HL_SPAWN_OFFSET_Y_M", "0"))
            if _off:
                start_pos_airsim = airsim.Vector3r(
                    start_pos_airsim.x_val, start_pos_airsim.y_val + _off,
                    start_pos_airsim.z_val)

            print(f"Teleporting Agent to: {start_pos_airsim}")
            client.simSetVehiclePose(
                airsim.Pose(start_pos_airsim, airsim.to_quaternion(0, 0, 0)),
                True
            )
            client.moveToZAsync(start_pos_airsim.z_val - 5, 2, timeout_sec=2).join()
            client.moveToZAsync(start_pos_airsim.z_val, 2, timeout_sec=2).join()

            # 4. 准备 Ground Truth 数据
            # 获取受困者真实位置 (UE坐标 -> AirSim坐标)
            gt_targets_ue = task.get('targets', [])
            gt_targets_airsim = []

            for t in gt_targets_ue:
                as_pos = CoordinateConverter.to_airsim(task['map_name'], t['location'])
                gt_targets_airsim.append({
                    "target_id": t.get('target_id'),
                    "location": {
                        "x": as_pos.x_val,
                        "y": as_pos.y_val,
                        "z": as_pos.z_val
                    }
                })
            
            # 5. 运行 Agent
            start_time = time.time()
            agent_result = self.run_agent_logic(client, task)
            end_time = time.time()
            
            # 6. 评估 CDS (线索发现分) — 语义匹配 + 空间距离验证
            gt_cue_names = self.get_ground_truth_names(task['map_name'], task)
            gt_clues_with_pos = self.get_ground_truth_clues(task['map_name'], task)
            cds, eval_details = self.evaluator.evaluate(
                agent_result.get('found_clues', []),
                gt_clues_with_pos
            )
            cdr = eval_details.get('cdr_score', eval_details.get('c_exact', 0.0))
            
            # 7. 保存日志 (整合了 GT 信息)
            log_entry = {
                "task_id": task['task_id'],
                "map": task['map_name'],
                "difficulty": task.get('difficulty_level', 'Unknown'),
                "status": agent_result['status'],
                "time_cost": end_time - start_time,
                "cdr_score": cdr,
                "cds_score": cds,
                "error_info": agent_result.get("error_info"),

                # Agent 输出 (包含 reported_target_position)
                "agent_output": agent_result,

                # Ground Truth
                "ground_truth": {
                    "targets": gt_targets_airsim
                },
                "ground_truth_cues": gt_cue_names,
                "ground_truth_clues_with_positions": gt_clues_with_pos
            }
            
            with open(LOG_DIR / f"task_{task['task_id']}{self.LOG_SUFFIX}.json", 'w') as f:
                json.dump(log_entry, f, indent=4)

            if str(agent_result.get("status", "")).upper() == "ERROR":
                self._save_untested_task(task)

            print(f"Task Finished. CDS: {cds:.2f}")

        except Exception as e:
            print(f"!!! Error in Task {task['task_id']}: {e}")
            traceback.print_exc()
            
        finally:
            # 8. 清理资源
            if client:
                try:
                    client.enableApiControl(False)
                except:
                    pass
            
            self.kill_ue_process(process_obj=ue_process, map_name=task['map_name'])

    def run_all(self):
        for task in self.tasks:
            self.run_single_task(task)


def _run_worker(worker_id, gpu_id, num_workers, base_port):
    # Must happen before any torch/ultralytics import (which YOLO-World
    # detectors do lazily inside agent __init__, so this is early enough):
    # every baseline's YOLO detector hardcodes device="cuda:0", so without
    # this the worker process would always see the *first* physical GPU
    # regardless of which one was assigned to it, and every worker's YOLO
    # inference would collide on GPU 0 while only UE5 rendering (pinned via
    # explicit -GraphicsAdapter=N) actually spreads across GPUs.
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    platform = ASARPlatform(
        gpu_id=gpu_id, worker_id=worker_id,
        num_workers=num_workers, base_port=base_port
    )
    platform.run_all()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpus", type=str, default=None,
        help="逗号分隔的 GPU id 列表用于多卡并行，例如 '0,1,2,3,4,5,6,7'。"
             "不传则按原来的方式单进程顺序跑（不做 GPU 绑定）。"
    )
    parser.add_argument(
        "--base-port", type=int, default=41451,
        help="worker 0 使用的 AirSim API 端口，之后每个 worker 依次 +1。"
    )
    args = parser.parse_args()

    if args.gpus:
        gpu_ids = [int(g) for g in args.gpus.split(",") if g.strip() != ""]
        import multiprocessing as mp
        procs = []
        for worker_id, gpu_id in enumerate(gpu_ids):
            p = mp.Process(
                target=_run_worker,
                args=(worker_id, gpu_id, len(gpu_ids), args.base_port)
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
    else:
        platform = ASARPlatform()
        platform.run_all()