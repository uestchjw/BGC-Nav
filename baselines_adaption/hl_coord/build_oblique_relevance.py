"""Offline: oblique 4-camera bank (RGB + DepthPlanar + pose) -> dense RelevanceGrid.

Same output format as build_relevance.py (cache/relevance_combo), so the follower's
_fuse_* code reads it unchanged. Difference: oblique frames are unprojected
PER PIXEL via their real depth (no nadir footprint assumption). Each CLIPSeg
score tile is placed at the world (x,y) its median-depth pixels actually hit.

    python -m baselines_adaption.hl_coord.build_oblique_relevance \
        --bank cache/oblique_bank_agl300/k2 --task 368 \
        --out cache/relevance_oblique/k2_task368.npz

Env / knobs: OBQ_TILE_H, OBQ_TILE_W (tile grid on the image), OBQ_MAX_RANGE_M,
OBQ_HORIZON_SKY_ROWS_FRAC.
"""
from __future__ import annotations
import argparse, json, math, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, "source"))
for p in (_SRC, os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))):
    if p not in sys.path:
        sys.path.insert(0, p)

from baselines_adaption.hl_coord.aerial_bank import (
    RelevanceGrid, aoi_bbox, radius_for_prompt, MAP_GROUND_Z_TOP_M)
from baselines_adaption.hl_coord.build_relevance import extract_terrain_terms, ClipSegBackend
from baselines_adaption.hl_coord.build_baselines import target_terms
from baselines_adaption.hl_coord import hl_config as C

DEPTH_VOID = 6000.0
TILE_H = int(os.environ.get("OBQ_TILE_H", 16))      # rows: more (far tiles cover more ground)
TILE_W = int(os.environ.get("OBQ_TILE_W", 12))
MAX_RANGE_M = float(os.environ.get("OBQ_MAX_RANGE_M", 450.0))   # drop tiles beyond this slant range
NEAR_CUT_M = float(os.environ.get("OBQ_NEAR_CUT_M", 8.0))       # drop rotor / too-near pixels


def _cam_world_dirs(H, W, fov_deg, drone_yaw_deg, cam_pitch_deg, cam_yaw_off_deg):
    """(H,W,3) unit ray direction in NED world for every pixel, and the boresight
    (H,W)-broadcastable optical axis. Camera: optical X=right, Y=down, Z=forward.
    body: camera looks along +x(forward) after yaw_off, then pitched down, then
    the vehicle yaw takes body->world."""
    f = W / (2.0 * math.tan(math.radians(fov_deg / 2.0)))
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    xn = (us - W / 2.0) / f            # +right
    yn = (vs - H / 2.0) / f            # +down
    d_cam = np.stack([np.ones_like(xn), xn, yn], axis=-1)   # [forward, right, down], unnormalised
    d_cam /= np.linalg.norm(d_cam, axis=-1, keepdims=True)

    def Rz(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def Ry(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    # body frame: x fwd, y right, z down.  d_cam is [fwd,right,down] already.
    # cam_pitch_deg is negative for nose-down; Ry(-50deg) maps forward [1,0,0]
    # to [cos50, 0, +sin50] = forward+DOWN in NED.
    R_yawoff = Rz(math.radians(cam_yaw_off_deg))
    R_pitch = Ry(math.radians(cam_pitch_deg))
    R_veh = Rz(math.radians(drone_yaw_deg))
    R = R_veh @ R_pitch @ R_yawoff                    # body/cam -> world (NED)
    dirs = d_cam @ R.T                                # (H,W,3)
    bore = R @ np.array([1.0, 0.0, 0.0])              # optical axis in world
    return dirs, bore / np.linalg.norm(bore)


def build(bank_dir, center_xy, radius_m, terms, backend, cell_m=None, verbose=True,
          map_name=None):
    cell_m = cell_m or C.CELL_SIZE_M / 2.0
    (x0, y0), (x1, y1) = aoi_bbox(center_xy, radius_m)
    grid = RelevanceGrid((x0, y0), (x1 - x0, y1 - y0), cell_m)
    idx = json.load(open(os.path.join(bank_dir, "index.json")))
    frames = [f for f in idx["frames"] if f.get("depth")]
    fov = idx.get("fov_deg", 90.0)
    mn = map_name or idx.get("map") or os.path.basename(bank_dir.rstrip("/"))
    z_sky = MAP_GROUND_Z_TOP_M.get(mn, -200.0) - 250.0   # reject hits >250m above highest ground
    # keep frames whose drone position is within radius + a slop of the AOI centre
    cx, cy = center_xy
    frames = [f for f in frames if math.hypot(f["x"] - cx, f["y"] - cy) < radius_m + MAX_RANGE_M]
    if verbose:
        print(f"[oblique-rel] AOI r={radius_m:.0f} @({cx:.0f},{cy:.0f}) -> {len(frames)} frames, "
              f"grid {grid.H}x{grid.W}@{cell_m:.0f}m  terms={terms}")
    from PIL import Image
    n_used = 0
    for n, fr in enumerate(frames):
        dep = np.load(os.path.join(bank_dir, fr["depth"]))
        H, W = dep.shape
        dirs, bore = _cam_world_dirs(H, W, fov, fr["drone_yaw_deg"],
                                     fr["cam_pitch_deg"], fr["cam_yaw_offset_deg"])
        cosang = np.clip(dirs @ bore, 1e-3, 1.0)
        rng = dep / cosang
        org = np.array([fr["x"], fr["y"], fr["z"]], float)
        world = org[None, None, :] + dirs * rng[..., None]          # (H,W,3)
        valid = ((dep > NEAR_CUT_M) & (dep < DEPTH_VOID) & (rng < MAX_RANGE_M)
                 & (world[..., 2] > z_sky))                          # reject sky/horizon hits
        if valid.sum() < 200:
            continue

        rgb = np.asarray(Image.open(os.path.join(bank_dir, fr["rgb"])).convert("RGB"))
        s = backend.score_grid(rgb[:, :, ::-1], terms, max(TILE_H, TILE_W))  # backend wants BGR-ish; it cvts
        # backend returns (G,G); resample to (TILE_H,TILE_W)
        s = np.asarray(s, np.float32)
        if s.shape != (TILE_H, TILE_W):
            gi = np.linspace(0, s.shape[0] - 1, TILE_H).round().astype(int)
            gj = np.linspace(0, s.shape[1] - 1, TILE_W).round().astype(int)
            s = s[np.ix_(gi, gj)]
        # per-frame normalise (centre 0.5)
        m = np.median(s); iqr = np.percentile(s, 75) - np.percentile(s, 25)
        s = np.clip(0.5 + (s - m) / (2.5 * iqr + 1e-6), 0.0, 1.0) if iqr > 1e-6 else np.full_like(s, 0.5)

        # for each image tile, take the median world (x,y) of its VALID pixels,
        # place a cell-sized bbox there, weighted by range (near = trust more)
        ti = (np.arange(H)[:, None] * TILE_H // H)
        tj = (np.arange(W)[None, :] * TILE_W // W)
        for a in range(TILE_H):
            for b in range(TILE_W):
                m_tile = valid & (ti == a) & (tj == b)
                k = int(m_tile.sum())
                if k < 15:
                    continue
                wx = float(np.median(world[..., 0][m_tile]))
                wy = float(np.median(world[..., 1][m_tile]))
                r_med = float(np.median(rng[m_tile]))
                w_rng = float(np.clip(1.0 - r_med / MAX_RANGE_M, 0.15, 1.0))
                half = grid.cell * 0.75
                for _ in range(max(1, int(round(w_rng * 3)))):     # crude range weighting via repeats
                    grid.add_tile(((wx - half, wy - half), (wx + half, wy + half)), float(s[a, b]))
        n_used += 1
        if verbose and n_used % 20 == 0:
            print(f"[oblique-rel]   {n_used}/{len(frames)} frames placed")
    grid.finalize()
    if verbose:
        print(f"[oblique-rel] coverage {grid.coverage:.0%}  mean {grid.relevance.mean():.2f} "
              f"max {grid.relevance.max():.2f}  frames_used {n_used}")
    return grid


def build_for_task(bank_dir, task, backend, cell_m=None, terms=None):
    from coordinate_utils import CoordinateConverter as CC
    s = CC.to_airsim(task["map_name"], task["start_location"])
    if terms is None:
        tgt = target_terms(task)
        ter = extract_terrain_terms(task.get("prompt", ""))
        terms = list(dict.fromkeys(tgt + ter))
    r = radius_for_prompt(task.get("prompt", ""))
    return build(bank_dir, (s["x"], s["y"]), r, terms, backend, cell_m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tasks = {t["task_id"]: t for t in json.load(open(os.path.join(_SRC, os.pardir, "data", "task_data.json")))}
    g = build_for_task(a.bank, tasks[a.task], ClipSegBackend())
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    g.save(a.out)
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
