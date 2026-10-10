"""
DashScope VLM wrapper for the APEX ASAR adaptation.

Calls Qwen VL to analyze aerial images and identify objects relevant
to the search-and-rescue task, returning object names and relevance scores.
"""

import os
import time
import base64
from http import HTTPStatus

import dashscope
from dashscope import MultiModalConversation

# dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY", "")

#! H95
dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY", "")


SAR_PROMPT_TEMPLATE = """You are an aerial search-and-rescue drone's perception module. You are given:
1. An aerial image from the drone's front camera
2. A search task description: "{task_prompt}"

Your job:
1. Identify all concrete, detectable objects in the image that could be relevant to finding survivors or clues (e.g., "tent", "campfire remains", "backpack", "person", "vehicle", "trail"). Exclude hard-to-segment objects like power lines or thin wires.
2. Score each object's relevance to the search task (0.1 = weak, 1.0 = strong). Survivors and direct evidence score highest. Terrain features suggesting human activity score medium. Generic landscape scores low.

Output format (single line, no explanation):
object1,object2,object3;score1,score2,score3

Example:
Input: "Target count: 2. Last seen near the mountain ridge."
Output: tent,hiking trail,person,backpack,rocky ridge;0.9,0.7,1.0,0.8,0.5"""


def analyze_scene(rgb_bytes_or_b64, task_prompt, model="qwen-vl-max"):
    """Call VLM to identify objects and their relevance scores.

    Parameters
    ----------
    rgb_bytes_or_b64 : str or bytes
        Base64-encoded image string, or raw image bytes.
    task_prompt : str
        The ASAR task prompt describing the search mission.
    model : str
        DashScope model name.

    Returns
    -------
    dict with keys:
        "success": bool
        "objects": list[str]  — detected object names
        "scores": list[float] — corresponding relevance scores
        "raw_response": str   — raw VLM output
    """
    if isinstance(rgb_bytes_or_b64, bytes):
        b64 = base64.b64encode(rgb_bytes_or_b64).decode("utf-8")
    else:
        b64 = rgb_bytes_or_b64

    prompt = SAR_PROMPT_TEMPLATE.format(task_prompt=task_prompt)

    messages = [
        {
            "role": "user",
            "content": [
                {"image": f"data:image/png;base64,{b64}"},
                {"text": prompt},
            ],
        }
    ]

    # dashscope.aliyuncs.com is directly reachable from this host; the
    # system-wide proxy is not needed for it and has been observed to fail
    # under sustained load (ConnectionRefusedError / ProxyError), silently
    # stalling every VLM call for the rest of the episode. Bypass it just
    # for this call regardless of what the caller's environment has set.
    _proxy_keys = ("http_proxy", "https_proxy", "all_proxy",
                   "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")
    _saved_proxies = {k: os.environ.pop(k, None) for k in _proxy_keys}
    os.environ["NO_PROXY"] = "dashscope.aliyuncs.com," + os.environ.get("NO_PROXY", "")
    os.environ["no_proxy"] = "dashscope.aliyuncs.com," + os.environ.get("no_proxy", "")

    MAX_ATTEMPTS = 3
    REQUEST_TIMEOUT_S = 20
    t0 = time.time()
    try:
        last_err = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = MultiModalConversation.call(
                    model=model,
                    messages=messages,
                    enable_thinking=False,
                    request_timeout=REQUEST_TIMEOUT_S,
                )
                elapsed = time.time() - t0

                if response.status_code != HTTPStatus.OK:
                    print(f"[APEX-VLM] API error attempt {attempt}/{MAX_ATTEMPTS} "
                          f"({elapsed:.1f}s): {response.code} - {response.message}")
                    last_err = response.message
                    continue

                text = response.output.choices[0].message.content[0]["text"]
                print(f"[APEX-VLM] response ({elapsed:.1f}s): {text}")

                objects, scores = _parse_response(text)
                return {
                    "success": True,
                    "objects": objects,
                    "scores": scores,
                    "raw_response": text,
                }

            except Exception as e:
                elapsed = time.time() - t0
                print(f"[APEX-VLM] error attempt {attempt}/{MAX_ATTEMPTS} ({elapsed:.1f}s): {e}")
                last_err = str(e)

        return {"success": False, "objects": [], "scores": [], "raw_response": last_err or ""}
    finally:
        for k, v in _saved_proxies.items():
            if v is not None:
                os.environ[k] = v


def _parse_response(text):
    """Parse 'obj1,obj2;0.5,0.8' format. Returns (objects, scores)."""
    text = text.strip()
    # Handle potential markdown formatting
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(l for l in lines if not l.startswith("```")).strip()

    if ";" not in text:
        return [], []

    parts = text.split(";", 1)
    objects = [o.strip() for o in parts[0].split(",") if o.strip()]
    try:
        scores = [float(s.strip()) for s in parts[1].split(",") if s.strip()]
    except ValueError:
        return objects, [0.5] * len(objects)

    # Ensure same length
    if len(scores) < len(objects):
        scores.extend([0.5] * (len(objects) - len(scores)))
    elif len(scores) > len(objects):
        scores = scores[: len(objects)]

    return objects, scores
