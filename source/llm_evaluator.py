import json
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
import dashscope
from dashscope import Generation
from http import HTTPStatus

# Load API key from environment variable
# dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY", "")
#! H95
dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY", "")


# Default spatial distance threshold for CDS (metres)
CDS_DISTANCE_THRESHOLD = 15.0


class ClueEvaluator:
    def __init__(self, distance_threshold: float = CDS_DISTANCE_THRESHOLD):
        self.model = "qwen3-max"
        self.distance_threshold = distance_threshold

    def evaluate(self, agent_clues: list, ground_truth_clues: list):
        """
        计算 CDS (Clue Discovery Score).

        CDS 由两部分等权加权组成:
          - C_loc  (Spatial Localization):  只看空间距离是否匹配
          - C_exact (Exact Semantic Matching): 空间距离 + LLM 语义同时匹配

        CDS = 0.5 * C_loc + 0.5 * C_exact

        :param agent_clues: Agent 输出的线索列表, 每项为
               {"name": str, "position": {"x":…, "y":…, "z":…}}
        :param ground_truth_clues: 真值线索列表, 每项为
               {"name": str, "position": {"x":…, "y":…, "z":…}}
        :return: (cds_score, match_details_dict)
        """
        if not ground_truth_clues:
            return (1.0 if not agent_clues else 0.0), {}

        if not agent_clues:
            return 0.0, {}

        n_gt = len(ground_truth_clues)
        n_ag = len(agent_clues)
        INF = 1e9

        # ============================================================
        # Part 1: C_loc — Spatial Localization (distance only)
        # ============================================================
        spatial_cost = np.full((n_gt, n_ag), INF)
        for gi, gt_clue in enumerate(ground_truth_clues):
            gt_pos = gt_clue["position"]
            for ai, ag_clue in enumerate(agent_clues):
                ag_pos = ag_clue.get("position")
                if ag_pos is None:
                    continue
                dist = np.linalg.norm([
                    gt_pos["x"] - ag_pos["x"],
                    gt_pos["y"] - ag_pos["y"],
                    gt_pos["z"] - ag_pos["z"],
                ])
                spatial_cost[gi, ai] = dist

        loc_matched = 0
        loc_matches = []
        if n_gt > 0 and n_ag > 0:
            row_ind, col_ind = linear_sum_assignment(spatial_cost)
            for gi, ai in zip(row_ind, col_ind):
                if spatial_cost[gi, ai] < self.distance_threshold:
                    loc_matched += 1
                    loc_matches.append({
                        "gt_name": ground_truth_clues[gi]["name"],
                        "agent_name": agent_clues[ai]["name"],
                        "distance": float(spatial_cost[gi, ai]),
                    })

        c_loc = loc_matched / n_gt

        # ============================================================
        # Part 2: C_exact — Exact Semantic Matching (distance + LLM)
        # ============================================================
        agent_names = [c["name"] for c in agent_clues]
        gt_names = [c["name"] for c in ground_truth_clues]

        name_mapping = self._llm_semantic_match(agent_names, gt_names)
        if name_mapping is None:
            # LLM 失败时，C_exact 为 0，仅保留 C_loc
            c_exact = 0.0
            exact_matched = 0
            exact_matches = []
            name_mapping = []
        else:
            exact_cost = np.full((n_gt, n_ag), INF)
            for gi, gt_clue in enumerate(ground_truth_clues):
                gt_pos = gt_clue["position"]
                for ai, ag_clue in enumerate(agent_clues):
                    if not self._names_match(ag_clue["name"], gt_clue["name"],
                                             name_mapping):
                        continue
                    ag_pos = ag_clue.get("position")
                    if ag_pos is None:
                        continue
                    dist = np.linalg.norm([
                        gt_pos["x"] - ag_pos["x"],
                        gt_pos["y"] - ag_pos["y"],
                        gt_pos["z"] - ag_pos["z"],
                    ])
                    exact_cost[gi, ai] = dist

            exact_matched = 0
            exact_matches = []
            if n_gt > 0 and n_ag > 0:
                row_ind, col_ind = linear_sum_assignment(exact_cost)
                for gi, ai in zip(row_ind, col_ind):
                    if exact_cost[gi, ai] < self.distance_threshold:
                        exact_matched += 1
                        exact_matches.append({
                            "gt_name": ground_truth_clues[gi]["name"],
                            "agent_name": agent_clues[ai]["name"],
                            "distance": float(exact_cost[gi, ai]),
                        })

            c_exact = exact_matched / n_gt

        # ============================================================
        # CDS = 0.5 * C_loc + 0.5 * C_exact
        # Legacy CDR corresponds to C_exact.
        # ============================================================
        cds = 0.5 * c_loc + 0.5 * c_exact
        cdr = c_exact

        details = {
            "c_loc": c_loc,
            "c_exact": c_exact,
            "cds_score": cds,
            "cdr_score": cdr,
            "loc_matched_count": loc_matched,
            "exact_matched_count": exact_matched,
            "total_gt": n_gt,
            "distance_threshold": self.distance_threshold,
            "loc_matches": loc_matches,
            "exact_matches": exact_matches,
            "name_mapping": name_mapping,
        }
        return cds, details

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _llm_semantic_match(self, agent_names: list, gt_names: list):
        """Call LLM to determine which agent name *categories* map to which GT
        name *categories*.

        The LLM only sees the unique category names (not per-instance lists) and
        returns a category-level mapping.  Instance-level matching is handled by
        the Hungarian algorithm in :meth:`evaluate`.

        Returns a list of {"agent_name": str, "gt_name": str} pairs,
        or None on failure.
        """
        # LLM only needs unique category names to build the mapping table.
        # Instance-level (multi-to-multi) matching uses spatial distance later.
        unique_agent = sorted(set(agent_names))
        unique_gt = sorted(set(gt_names))

        prompt = f"""
        Role: You are an expert judge for a Search and Rescue (SAR) mission.
        Task: Determine the semantic name-level mapping between drone-reported
        object names and ground-truth clue category names.

        Input Data:
        - Ground Truth Category Names (unique): {json.dumps(unique_gt)}
        - Drone Reported Category Names (unique): {json.dumps(unique_agent)}

        Matching Logic:
        1. Perform semantic matching (e.g., "red bag" matches "Backpack",
           "fire" matches "Campfire", "bag" matches "PlasticBag" or "Backpack").
        2. Each agent name may match at most one GT category name.
        3. Not every name needs a match.
        4. This is a CATEGORY-level mapping only; the same category may have
           multiple physical instances at different locations.

        Output Format (Strict JSON only, no conversational filler):
        {{
            "name_pairs": [
                {{"agent_name": "reported_term", "gt_name": "ground_truth_term"}}
            ]
        }}
        """

        try:
            response = Generation.call(
                model=self.model,
                prompt=prompt,
                result_format='message'
            )
            if response.status_code == HTTPStatus.OK:
                content = response.output.choices[0].message.content
                content = content.replace("```json", "").replace("```", "").strip()
                # Remove <think>...</think> blocks (Qwen3 reasoning mode)
                import re
                content = re.sub(r'<think>.*?</think>', '', content,
                                 flags=re.DOTALL).strip()
                result = json.loads(content)
                return result.get("name_pairs", [])
            else:
                print(f"LLM API Error: {response.code} - {response.message}")
                return None
        except Exception as e:
            print(f"LLM Semantic Match Exception: {e}")
            return None

    @staticmethod
    def _names_match(agent_name: str, gt_name: str, name_mapping: list) -> bool:
        """Check if agent_name is mapped to gt_name by the LLM."""
        for pair in name_mapping:
            if (pair["agent_name"].lower() == agent_name.lower()
                    and pair["gt_name"].lower() == gt_name.lower()):
                return True
        return False


# 测试用
if __name__ == "__main__":
    evaluator = ClueEvaluator(distance_threshold=15.0)
    agent = [
        {"name": "fire", "position": {"x": 10, "y": 20, "z": -5}},
        {"name": "bag",  "position": {"x": 50, "y": 60, "z": -3}},
    ]
    gt = [
        {"name": "Campfire", "position": {"x": 12, "y": 21, "z": -5}},
        {"name": "Backpack", "position": {"x": 100, "y": 100, "z": -3}},
        {"name": "Tent",     "position": {"x": 200, "y": 200, "z": -3}},
    ]
    cds, details = evaluator.evaluate(agent, gt)
    print(f"CDS: {cds:.2f}, Details: {json.dumps(details, indent=2)}")
