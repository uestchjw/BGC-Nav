"""Offline 'leader' for the two paper-baseline reproductions (training-free).

Both consume the SAME amortised BEV survey (cache/aerial_bank/<map>/) as densemap,
so the only thing that differs across the three methods at eval time is the
coordination representation and the follower's use of it.

  conav   : CoNav-UAV-TF -- open-vocab (YOLO-World) target detection on each BEV
            frame, project bbox centres to world, cluster, (optional VLM confirm),
            -> a priority-ordered queue of [x, y, score]  ->  cache/conav/<map>_task<id>.json
  aeroduo : AeroDuo-TF -- zero-shot referring segmentation (CLIPSeg) of the TARGET
            phrase over the ortho survey -> a target-probability grid M
            (RelevanceGrid format)                        ->  cache/aeroduo/<map>_task<id>.npz

Neither requires training / fine-tuning.  Faithful to each paper's COORDINATION
architecture; the trained brains are swapped for off-the-shelf zero-shot models.
"""
from __future__ import annotations
import json, math, os, re, sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, "source"))
for p in (_SRC, os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))):
    if p not in sys.path:
        sys.path.insert(0, p)

from baselines_adaption.hl_coord.aerial_bank import (
    AerialFrameBank, RelevanceGrid, project_norm, aoi_bbox, radius_for_prompt,
)
from baselines_adaption.hl_coord.build_relevance import build_relevance, ClipSegBackend

TARGET_VOCAB = ["person", "people", "hiker", "tent", "backpack", "campfire",
                "fire", "bag", "jacket", "sleeping bag"]


# --------------------------------------------------------------------------
def target_terms(task) -> list[str]:
    """The TARGET referent(s) for grounding — CoNav detects these, AeroDuo
    segments them."""
    try:
        from baselines_adaption.unigoal.goal_graph import parse_task_prompt
        from baselines_adaption.unigoal.llm_client import DashScopeLLMClient
        gg = parse_task_prompt(DashScopeLLMClient(model="qwen3-max"),
                               task.get("prompt", ""), add_related=False)
        objs = [o.lower().strip() for o in (gg.target_objects or []) if o.strip()]
        if objs:
            return objs
    except Exception as e:
        print(f"[build_baselines] target parse failed ({e}); regex fallback")
    low = (task.get("prompt", "") or "").lower()
    hits = [w for w in TARGET_VOCAB if w in low]
    return hits or ["person"]


def aeroduo_phrase(task) -> str:
    """AeroDuo grounds the target *location* — target noun + any spatial hint."""
    try:
        from baselines_adaption.unigoal.goal_graph import parse_task_prompt
        from baselines_adaption.unigoal.llm_client import DashScopeLLMClient
        gg = parse_task_prompt(DashScopeLLMClient(model="qwen3-max"),
                               task.get("prompt", ""), add_related=False)
        obj = (gg.target_objects or ["person"])[0]
        hints = " ".join(gg.spatial_hints or [])
        return f"{obj} {hints}".strip()
    except Exception:
        return (target_terms(task) or ["person"])[0]


# ======================= CoNav-UAV-TF =======================
def build_conav(bank_dir, task, yolo, out_path, conf=0.12, cluster_m=45.0,
                vlm=None, verbose=True):
    from coordinate_utils import CoordinateConverter as CC
    bank = AerialFrameBank.load(bank_dir)
    s = CC.to_airsim(task["map_name"], task["start_location"])
    r = radius_for_prompt(task.get("prompt", ""))
    box = aoi_bbox((s["x"], s["y"]), r)
    frames = bank.frames_overlapping(box)
    terms = target_terms(task)
    yolo.set_classes(terms)

    dets = []   # (wx, wy, conf)
    for fr in frames:
        bgr = bank.load_image(fr)
        if bgr is None:
            continue
        import cv2
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = yolo.predict(rgb, conf=conf, verbose=False, device="cuda:0")
        h, w = rgb.shape[:2]
        for rr in res:
            for b in rr.boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                cu, cv_ = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                nx, ny = cu / w - 0.5, cv_ / h - 0.5
                wx, wy = project_norm(fr.x, fr.y, fr.yaw, nx, ny, fr.footprint_m)
                dets.append((wx, wy, float(b.conf[0]) if b.conf is not None else conf))

    # greedy clustering
    clusters = []   # [ [sx, sy, wsum, n, maxc] ]
    for wx, wy, c in dets:
        for cl in clusters:
            if math.hypot(cl[0] / cl[2] - wx, cl[1] / cl[2] - wy) <= cluster_m:
                cl[0] += wx * c; cl[1] += wy * c; cl[2] += c; cl[3] += 1
                cl[4] = max(cl[4], c)
                break
        else:
            clusters.append([wx * c, wy * c, c, 1, c])

    queue = []
    for cl in clusters:
        cx, cy = cl[0] / cl[2], cl[1] / cl[2]
        score = cl[4] * (1.0 + 0.15 * min(cl[3] - 1, 5))   # conf * mild multi-hit bonus
        if vlm is not None:
            try:
                score *= _vlm_confirm(vlm, bank, frames, cx, cy, task)
            except Exception as e:
                print(f"[build_conav] vlm confirm failed ({e})")
        queue.append([round(cx, 1), round(cy, 1), round(float(score), 4)])
    queue.sort(key=lambda q: -q[2])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump({"map": task["map_name"], "task_id": task["task_id"],
               "terms": terms, "n_frames": len(frames), "n_det": len(dets),
               "queue": queue}, open(out_path, "w"))
    if verbose:
        print(f"[build_conav] task {task['task_id']} ({task['map_name']}): "
              f"{len(frames)} frames, {len(dets)} dets -> {len(queue)} queue entries "
              f"top={queue[:2]}")
    return queue


def _vlm_confirm(vlm, bank, frames, cx, cy, task):
    """One VLM call: pick the frame whose footprint centre is nearest (cx,cy),
    ask whether the target is really there. Returns a multiplier in [0.2, 1.0]."""
    fr = min(frames, key=lambda f: math.hypot(f.x - cx, f.y - cy))
    bgr = bank.load_image(fr)
    if bgr is None:
        return 1.0
    prompt = (f"Aerial top-down photo. The search-and-rescue instruction is: "
              f"\"{task.get('prompt','')}\". Is there a search target (a person / "
              f"tent / campfire / backpack) actually visible near the CENTRE of "
              f"this image? Answer JSON only: {{\"present\": true|false, "
              f"\"confidence\": <0..1>}}")
    txt = vlm.caption_from_bgr(bgr, prompt)
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m:
        return 1.0
    d = json.loads(m.group())
    if not d.get("present", True):
        return 0.2
    return float(np.clip(0.4 + 0.6 * float(d.get("confidence", 0.5)), 0.2, 1.0))


# ======================= AeroDuo-TF =======================
def build_aeroduo(bank_dir, task, clipseg, out_path, G=8, cell_m=25.0, verbose=True):
    from coordinate_utils import CoordinateConverter as CC
    bank = AerialFrameBank.load(bank_dir)
    s = CC.to_airsim(task["map_name"], task["start_location"])
    phrase = aeroduo_phrase(task)
    r = radius_for_prompt(task.get("prompt", ""))
    g = build_relevance(bank, (s["x"], s["y"]), r, [phrase], clipseg,
                        G=G, cell_m=cell_m, normalize=True, verbose=False)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    g.save(out_path)
    if verbose:
        print(f"[build_aeroduo] task {task['task_id']} ({task['map_name']}): "
              f"phrase='{phrase}' cov {g.coverage:.0%} Mmean {g.relevance.mean():.2f} "
              f"Mmax {g.relevance.max():.2f} -> {os.path.basename(out_path)}")
    return g
