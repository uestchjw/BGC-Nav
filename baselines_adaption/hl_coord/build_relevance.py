"""Offline: turn an aerial frame bank + a task prompt into a dense RelevanceGrid.

No simulator.  Runs the relevance model on the bank frames that overlap the
task's area of interest, projects every grid tile's score to world coordinates
via the stored pose, and accumulates into a per-cell mean (the full-coverage
sweep means each cell is seen by several frames -> the per-frame VLM noise
that used to flood the field averages out).

    python -m baselines_adaption.hl_coord.build_relevance \
        --bank cache/aerial_bank/k2 --task 368 \
        --out  cache/relevance/k2_task368.npz --backend vlm

The relevance model is pluggable (``RelevanceBackend``); ``vlm`` uses the same
DashScope VLM as the leader but with a RANK-WITHIN-FRAME prompt (forces the
model to spread its scores across the grid instead of "everything is 0.9"),
``clipseg`` is a stub for a transformers CLIPSeg drop-in.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, "source"))
for p in (_SRC, os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))):
    if p not in sys.path:
        sys.path.insert(0, p)

from baselines_adaption.hl_coord.aerial_bank import (
    AerialFrameBank, RelevanceGrid, project_tile_bbox, per_frame_normalize,
    aoi_bbox, radius_for_prompt,
)
from baselines_adaption.hl_coord import hl_config as C


# --------------------------------------------------------------------------
# prompt -> terrain terms
# --------------------------------------------------------------------------
def extract_terrain_terms(prompt: str, use_llm: bool = True) -> list[str]:
    """Prefer UniGoal's parser (same as the leader: goal_graph.spatial_hints).
    Falls back to a keyword regex if the LLM client is unavailable."""
    if use_llm:
        try:
            from baselines_adaption.unigoal.goal_graph import parse_task_prompt
            from baselines_adaption.unigoal.llm_client import DashScopeLLMClient
            gg = parse_task_prompt(DashScopeLLMClient(model="qwen3-max"),
                                   prompt or "", add_related=False)
            if gg.spatial_hints:
                return list(gg.spatial_hints)
        except Exception as e:
            print(f"[build_relevance] LLM term extraction failed ({e}); regex fallback")
    VOCAB = ("ridge summit peak slope hillside valley gully ravine cliff plateau "
             "river stream creek riverbed lakeside shore seashore beach pond "
             "forest woods treeline meadow grassland clearing scrub bushes "
             "road path trail track dirt road switchback bridge "
             "boulder rocks scree talus sand dune yardang desert "
             "campsite tent hut cabin ruin field farmland").split()
    low = (prompt or "").lower()
    hits = [w for w in VOCAB if w in low]
    # also 2-grams like "dirt road", "forest road"
    for bg in ("dirt road", "forest road", "dry riverbed", "boulder field"):
        if bg in low:
            hits.append(bg)
    return list(dict.fromkeys(hits)) or ["terrain matching the description"]


# --------------------------------------------------------------------------
# relevance backends
# --------------------------------------------------------------------------
class VLMGridBackend:
    """DashScope VLM, RANK-within-frame prompt."""

    def __init__(self, model="qwen3.5-plus", draw_grid=True):
        from baselines_adaption.unigoal.llm_client import DashScopeVLMClient
        self.vlm = DashScopeVLMClient(model=model)
        self.draw_grid = draw_grid

    def _prompt(self, G, terms):
        return (
            "You are analyzing a straight-down aerial photo from a high-altitude "
            f"search-and-rescue drone. It is divided into a {G}x{G} grid "
            "(row 0 = top, col 0 = left; green lines mark the cells). "
            f"RANK the {G*G} cells by how well the GROUND in each matches ANY of "
            f"these terrain/location descriptions: [{', '.join(terms)}]. "
            "Give the single best-matching cell 1.0, the worst 0.0, and spread the "
            "rest in between by relative match. Do NOT give many cells the same "
            "value. Answer JSON only: "
            f"{{\"grid\": [{G} rows of {G} floats], \"best_match\": \"<term or none>\"}}"
        )

    def score_grid(self, bgr, terms, G):
        import cv2
        img = bgr
        if self.draw_grid and G > 1:
            img = bgr.copy()
            h, w = img.shape[:2]
            for k in range(1, G):
                cv2.line(img, (int(w*k/G), 0), (int(w*k/G), h-1), (0, 255, 0), 2)
                cv2.line(img, (0, int(h*k/G)), (w-1, int(h*k/G)), (0, 255, 0), 2)
        try:
            txt = self.vlm.caption_from_bgr(img, self._prompt(G, terms))
        except Exception as e:
            print(f"[VLMGridBackend] call failed: {e}")
            return np.full((G, G), 0.5, np.float32)
        m = re.search(r"\{.*\}", txt.strip(), re.DOTALL)
        if not m:
            return np.full((G, G), 0.5, np.float32)
        try:
            g = json.loads(m.group()).get("grid")
            a = np.asarray(g, np.float32)
            if a.shape == (G, G):
                return np.clip(a, 0.0, 1.0)
        except Exception:
            pass
        return np.full((G, G), 0.5, np.float32)


class ClipSegBackend:
    """transformers CLIPSeg — per-pixel relevance per term, pooled to a GxG grid.
    Prompt-agnostic at survey time; here we just query the task terms."""

    def __init__(self, device="cuda"):
        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
        import torch
        self.torch = torch
        self.proc = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined").to(device).eval()
        self.device = device

    def score_grid(self, bgr, terms, G):
        import cv2
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        with self.torch.no_grad():
            inp = self.proc(text=list(terms), images=[rgb] * len(terms),
                            return_tensors="pt", padding=True).to(self.device)
            logits = self.model(**inp).logits            # (T, h, w)
            prob = self.torch.sigmoid(logits).max(0).values.cpu().numpy()  # (h,w)
        h, w = prob.shape
        out = np.zeros((G, G), np.float32)
        for i in range(G):
            for j in range(G):
                out[i, j] = prob[i*h//G:(i+1)*h//G, j*w//G:(j+1)*w//G].mean()
        return np.clip(out, 0.0, 1.0)


BACKENDS = {"vlm": VLMGridBackend, "clipseg": ClipSegBackend}


# --------------------------------------------------------------------------
# the builder
# --------------------------------------------------------------------------
def build_relevance(bank: AerialFrameBank, center_xy, radius_m, terms, backend,
                    G=8, cell_m=None, normalize=True, verbose=True,
                    neg_terms=None, neg_w=0.5):
    """``neg_terms`` (optional): terrain phrases to DOWN-weight — scored the same
    way and subtracted (weighted by ``neg_w``) from the positive score per tile."""
    cell_m = cell_m or C.CELL_SIZE_M / 2.0        # default: half the old 50 m field cell
    box = aoi_bbox(center_xy, radius_m)
    (x0, y0), (x1, y1) = box
    grid = RelevanceGrid((x0, y0), (x1 - x0, y1 - y0), cell_m)
    frames = bank.frames_overlapping(box)
    if verbose:
        print(f"[build_relevance] AOI r={radius_m:.0f} m at ({center_xy[0]:.0f},"
              f"{center_xy[1]:.0f}) -> {len(frames)}/{len(bank.frames)} frames, "
              f"grid {grid.H}x{grid.W} @ {cell_m:.0f} m, terms={terms}"
              f"{' neg='+str(neg_terms) if neg_terms else ''}")
    for n, fr in enumerate(frames):
        bgr = bank.load_image(fr)
        if bgr is None:
            continue
        s = backend.score_grid(bgr, terms, G)
        if neg_terms:
            sn = backend.score_grid(bgr, list(neg_terms), G)
            s = np.clip(s - neg_w * sn, 0.0, 1.0)
        if normalize:
            s = per_frame_normalize(s)
        for i in range(G):
            for j in range(G):
                bbox = project_tile_bbox(fr.x, fr.y, fr.yaw, i, j, G, fr.footprint_m)
                grid.add_tile(bbox, float(s[i, j]))
        if verbose and (n + 1) % 10 == 0:
            print(f"[build_relevance]   {n+1}/{len(frames)} frames scored")
    grid.finalize()
    if verbose:
        print(f"[build_relevance] coverage {grid.coverage:.0%}  "
              f"relevance mean {grid.relevance.mean():.2f} max {grid.relevance.max():.2f}")
    return grid


def build_for_task(bank_dir, task, backend, G=8, cell_m=None, terms=None,
                   neg_terms=None):
    """``task`` = a dict from data/task_data.json (needs map_name, start_location,
    prompt).  Uses AirSim-NED start as the AOI centre.  ``terms`` overrides the
    default terrain-word extraction; ``neg_terms`` are down-weighted."""
    from coordinate_utils import CoordinateConverter as CC
    bank = AerialFrameBank.load(bank_dir)
    s = CC.to_airsim(task["map_name"], task["start_location"])
    if terms is None:
        terms = extract_terrain_terms(task.get("prompt", ""))
    r = radius_for_prompt(task.get("prompt", ""))
    return build_relevance(bank, (s["x"], s["y"]), r, terms, backend, G, cell_m,
                           neg_terms=neg_terms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True, help="cache/aerial_bank/<map>")
    ap.add_argument("--task", type=int, help="task_id from data/task_data.json")
    ap.add_argument("--terms", help="comma-separated terrain terms (overrides prompt parse)")
    ap.add_argument("--center", help="x,y AirSim-NED AOI centre (with --terms, no --task)")
    ap.add_argument("--radius", type=float, default=900.0)
    ap.add_argument("--backend", choices=list(BACKENDS), default="vlm")
    ap.add_argument("--grid", type=int, default=8)
    ap.add_argument("--cell", type=float, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    backend = BACKENDS[a.backend]()
    if a.task is not None:
        tasks = {t["task_id"]: t for t in json.load(open(os.path.join(_SRC, os.pardir, "data", "task_data.json")))}
        grid = build_for_task(a.bank, tasks[a.task], backend, a.grid, a.cell)
    else:
        terms = [s.strip() for s in a.terms.split(",")]
        cx, cy = [float(v) for v in a.center.split(",")]
        bank = AerialFrameBank.load(a.bank)
        grid = build_relevance(bank, (cx, cy), a.radius, terms, backend, a.grid, a.cell)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    grid.save(a.out)
    print(f"[build_relevance] saved -> {a.out}")


if __name__ == "__main__":
    main()
