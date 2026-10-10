"""Prompt builders for NavGPT-style caption->policy decision in ASAR."""

from typing import Dict, List


def build_policy_prompt(
    task_prompt: str,
    captions: Dict[str, str],
    history: List[str],
    safe_actions: List[str],
    current_pos_text: str,
) -> str:
    """Build strict JSON policy prompt for action decision."""

    hist = "\n".join(history) if history else "None"
    cap_front = captions.get("front_camera", "N/A")
    cap_right = captions.get("right_camera", "N/A")
    cap_back = captions.get("back_camera", "N/A")
    cap_left = captions.get("left_camera", "N/A")
    safe_actions_str = ", ".join(safe_actions) if safe_actions else "none"

    return f"""
Role: You are NavGPT adapted for aerial search and rescue (ASAR).
Mission: {task_prompt}

Current relative position:
{current_pos_text}

Perception captions from four views:
- Front: {cap_front}
- Right: {cap_right}
- Back: {cap_back}
- Left: {cap_left}

Recent action history:
{hist}

Safe candidate actions (you MUST choose from these):
{safe_actions_str}

Rules:
1) You MUST choose exactly one action from the safe candidate actions listed above.
2) Your job is navigation only. Object detection and target reporting are handled by a separate module.
3) Output STRICT JSON only, no markdown.

Output JSON format:
{{
  "thought": "...",
  "action": "move_forward"
}}
""".strip()
