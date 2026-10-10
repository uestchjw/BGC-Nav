import json
import pathlib
import pandas as pd
import glob
import numpy as np
from scipy.optimize import linear_sum_assignment

from coordinate_utils import calculate_distance
from llm_evaluator import ClueEvaluator

# ================= 配置与常量 =================
SOURCE_DIR = pathlib.Path(__file__).parent.absolute()
ROOT_DIR = SOURCE_DIR.parent
DATA_DIR = ROOT_DIR / "data"
LOG_DIR = ROOT_DIR / "log"

# 评分权重配置
W_COMP = 0.1   # 任务完成分 (Finished = 0.1, otherwise 0)
W_BASE = 0.3   # SR 分
W_TIME = 0.3   # 效率分 (SR * e_t)
W_CLUE = 0.3   # 线索发现分 (CDS)

THETA = 15.0   # 距离误差阈值

# 地图对应的最大任务时长 (秒)
MAP_SETTINGS = {
    "aotai2800": 10 * 60,
    "dapeng":    10 * 60,
    "k2":        10 * 60,
    "lopnur":    15 * 60
}


# ================= 辅助函数 =================


def _load_cue_dict() -> dict:
    """加载 cue_list.json，构建 {map_name: {id: name}} 映射。"""
    cue_list_path = DATA_DIR / "cue_list.json"
    if not cue_list_path.exists():
        print(f"Warning: {cue_list_path} not found, CDS recalculation disabled.")
        return {}

    with open(cue_list_path, 'r', encoding='utf-8') as f:
        cue_mapping_data = json.load(f)

    cue_dict = {}
    for map_data in cue_mapping_data:
        m_name = map_data['map_name']
        cue_dict[m_name] = {}
        all_items = map_data.get('cues', []) + map_data.get('fire', [])
        for item in all_items:
            cue_dict[m_name][item['id']] = item['name']
    return cue_dict


def _load_task_gt_clues(task_data_path: pathlib.Path) -> dict:
    """加载 task_data，构建 {task_id: task_dict} 方便通过 task_id 查 GT 线索。"""
    if not task_data_path.exists():
        return {}
    with open(task_data_path, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    return {t['task_id']: t for t in tasks}


# ================= 核心分析 =================

def analyze_results(recalc_cds: bool = True):
    """分析所有日志文件并输出评估报告。

    Args:
        recalc_cds: 若为 True，使用 llm_evaluator 独立重新计算 CDS，
                    而不是信任日志中的 cds_score。需要 DASHSCOPE_API_KEY。
    """
    log_files = glob.glob(str(LOG_DIR / "task_*_log.json"))

    if not log_files:
        print("No log files found in log directory.")
        return

    # --- 预加载数据 (用于 CDS 重算) ---
    cue_dict = _load_cue_dict()

    # 尝试加载多个可能的 task_data 文件
    task_gt_map = {}
    for fname in ["task_data_sample.json", "task_data.json"]:
        task_gt_map.update(_load_task_gt_clues(DATA_DIR / fname))

    evaluator = ClueEvaluator() if recalc_cds else None

    data_list = []

    for log_file in log_files:
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                log = json.load(f)

                # --- 1. 获取基础数据 ---
                task_id = log.get("task_id")
                map_name = log.get("map", "unknown")
                t_actual = log.get("time_cost", 0)
                t_max = MAP_SETTINGS.get(map_name, 1200)

                agent_out = log.get("agent_output", {})
                gt_data = log.get("ground_truth", {})

                # --- 2. 计算 SR (多目标匹配逻辑) ---
                # 获取 Agent 报告的坐标列表
                reported_positions = agent_out.get("reported_positions", [])

                # 获取 GT 列表
                gt_targets = gt_data.get("targets", [])

                n_total = len(gt_targets)
                n_found = 0

                # 匈牙利算法 (Hungarian Algorithm) 最优二分匹配
                n_reports = len(reported_positions)
                if n_total > 0 and n_reports > 0:
                    # 构建代价矩阵 (GT x Report)
                    cost_matrix = np.zeros((n_total, n_reports))
                    for gi, gt in enumerate(gt_targets):
                        for ri, report in enumerate(reported_positions):
                            cost_matrix[gi, ri] = calculate_distance(gt['location'], report)

                    # 求解最优匹配
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)

                    # 统计阈值内的成功匹配数
                    for gi, ri in zip(row_ind, col_ind):
                        if cost_matrix[gi, ri] <= THETA:
                            n_found += 1

                # 计算单次任务的 SR (0.0 ~ 1.0)
                task_sr = n_found / n_total if n_total > 0 else 0.0

                # --- 3. 计算 CDS ---
                # 判断该任务是否有 GT 线索
                gt_cue_names = log.get("ground_truth_cues") or []
                has_gt_clues = len(gt_cue_names) > 0

                if not has_gt_clues:
                    # 无线索任务：CDS/CDR 不参与均值统计，RS 中线索项统一视为 1.0
                    cds_score = None
                    cdr_score = None
                    cds_for_rs = 1.0
                elif recalc_cds and evaluator is not None:
                    # 独立重算 CDS（以及其中的 CDR 分量）
                    cds_score, cdr_score = _recalc_cds(
                        evaluator, cue_dict, task_gt_map,
                        task_id, map_name, agent_out, log
                    )
                    cds_for_rs = cds_score
                else:
                    # 使用日志分数：CDS 为主；CDR 只读旧字段或新日志显式字段
                    cds_score = log.get("cds_score", log.get("cdr_score", 0))
                    cdr_score = log.get("cdr_score")
                    cds_for_rs = cds_score

                # --- 4. 计算 RS (救援效能得分) ---
                # 任务完成度: Finished = 1, 否则 = 0
                task_status = agent_out.get("status", log.get("status", ""))
                task_completed = 1.0 if task_status == "Finished" else 0.0

                e_t = max(0, 1 - (t_actual / t_max))

                tsr_score = task_sr * e_t

                # RS = W_COMP * completed + SR * (W_BASE + W_TIME * e_t) + W_CLUE * CDS
                term_comp = W_COMP * task_completed
                term_sr   = task_sr * (W_BASE + W_TIME * e_t)
                term_clue = W_CLUE * cds_for_rs

                rs_score = term_comp + term_sr + term_clue

                path_len = agent_out.get("path_length")

                # --- 5. 存入列表 ---
                entry = {
                    "Task ID": task_id,
                    "Map": map_name,
                    "Difficulty": log.get("difficulty", "Unknown"),

                    "Status": task_status,
                    "Time Actual": t_actual,
                    "N_Found": n_found,
                    "N_Total": n_total,

                    "Task Completed": task_completed,
                    "Success Rate": task_sr,
                    "TSR": tsr_score,
                    "CDR": cdr_score,
                    "CDS": cds_score,        # None for no-clue tasks
                    "RS Score": rs_score,
                    "Safe Dist.": path_len
                }
                data_list.append(entry)

        except Exception as e:
            print(f"Error reading {log_file}: {e}")
            import traceback
            traceback.print_exc()

    # ================= 统计分析 =================
    if not data_list:
        print("No valid data parsed.")
        return

    df = pd.DataFrame(data_list)
    df.sort_values(by="Task ID", inplace=True)

    print("\n" + "="*60)
    print("ASAR Benchmark Evaluation Report")
    print("="*60)

    # 1. 总体指标
    # CDR/CDS 均值只计算有线索的任务 (排除 None)
    cdr_valid = df['CDR'].dropna()
    cds_valid = df['CDS'].dropna()
    cdr_mean_str = f"{cdr_valid.mean() * 100:.2f}%" if len(cdr_valid) > 0 else "N/A"
    cds_mean_str = f"{cds_valid.mean() * 100:.2f}%" if len(cds_valid) > 0 else "N/A"
    n_no_clue_tasks = df['CDS'].isna().sum()

    # 任务完成率
    completion_rate = df['Task Completed'].mean()

    print(f"Total Tasks: {len(df)}")
    print(f"Task Completion Rate:    {completion_rate * 100:.2f}%")
    print(f"Mean Success Rate:       {df['Success Rate'].mean() * 100:.2f}% (Found {df['N_Found'].sum()}/{df['N_Total'].sum()})")
    print(f"Mean Task Time:          {df['Time Actual'].mean():.2f} s")
    print(f"Mean TSR (Time-weighted):{df['TSR'].mean() * 100:.2f}%")
    print(f"Mean CDR:                {cdr_mean_str} ({len(cdr_valid)} tasks with clues, {n_no_clue_tasks} tasks without)")
    print(f"Mean CDS:                {cds_mean_str} ({len(cds_valid)} tasks with clues, {n_no_clue_tasks} tasks without)")
    print(f"Mean Rescue Score (RS):  {df['RS Score'].mean():.4f}")
    print(f"Mean Safe Dist. :  {df['Safe Dist.'].mean():.4f}")

    # 2. 按难度分组
    print("\n" + "-"*30)
    print("Metrics by Difficulty")
    print("-"*30)
    map_stats = df.groupby("Difficulty").agg({
        "N_Found": "sum",
        "N_Total": "sum",
        "Success Rate": "mean",
        "Time Actual": "mean",
        "TSR": "mean",
        "CDR": "mean",
        "CDS": "mean",     # NaN (no-clue) 自动被 pandas 排除
        "RS Score": "mean"
    })
    print(map_stats[["Success Rate", "TSR", "Time Actual", "CDR", "CDS", "RS Score"]])

    # 3. 导出
    export_path = LOG_DIR / "evaluation_report_full.csv"
    df.to_csv(export_path, index=False)
    print(f"\nSaved to: {export_path}")


def _recalc_cds(evaluator, cue_dict, task_gt_map,
                task_id, map_name, agent_out, log):
    """独立重新计算 CDS 与 CDR 分数.

    优先从日志 ground_truth_clues_with_positions 字段获取带坐标的 GT 线索;
    若不存在，则从原始 task_data + cue_list 重建。

    Args:
        evaluator: ClueEvaluator 实例。
        cue_dict: {map_name: {id: name}} 映射。
        task_gt_map: {task_id: task_dict} 原始任务数据。
        task_id: 当前任务 ID。
        map_name: 地图名。
        agent_out: 日志中的 agent_output 字典。
        log: 完整日志字典。

    Returns:
        tuple[float, float]: (CDS, CDR) 分数。
    """
    from coordinate_utils import CoordinateConverter

    # Agent 线索 (带坐标)
    agent_clues = agent_out.get("found_clues", [])
    if not agent_clues:
        return 0.0, 0.0

    # 尝试从日志获取带坐标的 GT 线索
    gt_clues_with_pos = log.get("ground_truth_clues_with_positions",
                                log.get("ground_truth_clues_with_pos"))

    if not gt_clues_with_pos:
        # 从原始数据重建
        task_data = task_gt_map.get(task_id)
        if not task_data or map_name not in cue_dict:
            print(f"  [CDS Recalc] Task {task_id}: cannot rebuild GT clues, "
                  f"falling back to logged cds_score/cdr_score.")
            fallback_cdr = log.get("cdr_score")
            fallback_cds = log.get("cds_score", fallback_cdr if fallback_cdr is not None else 0)
            return fallback_cds, fallback_cdr if fallback_cdr is not None else 0.0

        gt_clues_with_pos = []
        all_cues = task_data.get('cues', []) + task_data.get('fire', [])
        for cue in all_cues:
            c_id = cue['id']
            name = cue_dict[map_name].get(c_id, f"Unknown_ID_{c_id}")
            ue_loc = cue.get('location')
            if ue_loc:
                as_pos = CoordinateConverter.to_airsim(map_name, ue_loc)
            else:
                as_pos = {"x": 0.0, "y": 0.0, "z": 0.0}
            gt_clues_with_pos.append({"name": name, "position": as_pos})

    cds, details = evaluator.evaluate(agent_clues, gt_clues_with_pos)
    cdr = details.get("cdr_score", details.get("c_exact", 0.0))
    return cds, cdr


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ASAR Benchmark Result Analysis")
    parser.add_argument("--no-recalc-cds", action="store_false", dest="recalc_cds",
                        help="Disable independent CDS recalculation and use logged "
                             "cds_score instead.")
    parser.set_defaults(recalc_cds=True)
    args = parser.parse_args()
    analyze_results(recalc_cds=args.recalc_cds)
