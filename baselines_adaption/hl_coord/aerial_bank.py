"""Offline aerial map-building: frame bank + relevance grid + projection.

Design (see docs/hl_coord_research_log.md, the "survey-then-search" redesign):

  * There are only 4 maps (aotai2800 / lopnur / k2 / dapeng), 150 tasks each.
    The terrain geometry is FIXED across a map's 150 tasks — only weather,
    time-of-day, spawn, targets and prompt change.

  * So the high-altitude survey is done ONCE PER MAP, in clear daylight, and
    stores RAW SENSOR DATA: the straight-down (BEV) frames plus the drone
    pose for each frame. No VLM / semantics at survey time.  ->  AerialFrameBank

  * At task time (offline, no simulator): take the bank frames that overlap the
    follower's area of interest, run a relevance model on each frame CONDITIONED
    ON THE TASK PROMPT's terrain words, project every grid tile's score to world
    coordinates via the stored pose, and accumulate into a dense grid.
    Because the full-coverage sweep observes every ground cell from several
    frames, the per-cell average cancels the per-frame VLM noise that made the
    old concurrent "grid VLM" flood ("everything is a ridge").      ->  RelevanceGrid

This module has NO airsim dependency — it is pure numpy + json + cv2 image IO,
so relevance can be rebuilt / a different perception model tried without ever
re-flying.  The flight driver that produces a bank is ``source/survey_flight.py``.

Coordinate frame throughout: AirSim NED metres (x North, y East, z Down /
negative = up), the same frame the agents and ``simGetVehiclePose`` use.
UE-cm -> NED-m conversion is ``coordinate_utils.CoordinateConverter.to_airsim``.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


# ---------------------------------------------------------------------------
# Per-map survey area (AirSim NED metres).
# Bounding box of every task's start + target + cue point on that map, padded
# by 250 m.  Regenerate with tools: see docs; do not hand-edit casually.
# The z entry is the NED-z of the *player-start reference* for that map
# (coordinate_utils.CoordinateConverter.PLAYER_STARTS[map][2] / 100, sign per
# the to_airsim formula z' = (c - z_ue)/100); "point-z" comment = the NED-z
# spread of the map's known ground points, i.e. how much terrain relief there is.
# ---------------------------------------------------------------------------
MAP_BOUNDS_M = {
    "aotai2800": dict(x=(-986.0, 548.0),  y=(-1166.0, 973.0)),   # 1.53 x 2.14 km ; ground-z NED  -135 .. 262
    "lopnur":    dict(x=(-2210.0, 1595.0), y=(-1239.0, 2101.0)),  # 3.80 x 3.34 km ; ground-z NED   -35 .. 6
    "k2":        dict(x=(-1450.0, 1265.0), y=(-1436.0, 821.0)),   # 2.71 x 2.26 km ; ground-z NED  -115 .. 726
    "dapeng":    dict(x=(-892.0, 971.0),  y=(-687.0, 1130.0)),    # 1.86 x 1.82 km ; ground-z NED  -179 .. 196
}

# Highest known ground on each map (most-negative NED-z of any known point).
# The survey drone flies SURVEY_ALT_M above THIS so it clears the terrain.
MAP_GROUND_Z_TOP_M = {
    "aotai2800": -135.0,
    "lopnur":    -35.0,
    "k2":        -115.0,
    "dapeng":    -179.0,
}

BEV_FOV_DEG = 90.0  # settings_high_low.json bev_camera FOV


def survey_z_ned(map_name: str, survey_alt_m: float) -> float:
    """NED-z (negative = up) the survey drone should hold: ``survey_alt_m``
    above the highest known ground on this map."""
    return MAP_GROUND_Z_TOP_M[map_name] - float(survey_alt_m)


def footprint_m(alt_above_ground_m: float, fov_deg: float = BEV_FOV_DEG) -> float:
    """Ground width (metres) one nadir BEV frame covers at this height."""
    return 2.0 * float(alt_above_ground_m) * math.tan(math.radians(fov_deg / 2.0))


# ---------------------------------------------------------------------------
# Full-coverage sweep planner
# ---------------------------------------------------------------------------
def plan_boustrophedon(bounds: dict, fp_m: float, overlap: float = 0.30,
                       margin_m: float = 0.0):
    """Lawnmower waypoints (x, y) covering the whole ``bounds`` rectangle.

    Row spacing = fp_m * (1 - overlap) so consecutive rows overlap; within a
    row the drone captures continuously, so along-track spacing is handled by
    the capture cadence, not extra waypoints — we only emit the two row ends.
    Alternating row direction keeps legs adjacent (no long transits).
    """
    x0, x1 = bounds["x"];  y0, y1 = bounds["y"]
    x0 -= margin_m; x1 += margin_m; y0 -= margin_m; y1 += margin_m
    row = fp_m * (1.0 - overlap)
    ys = np.arange(y0 + row / 2.0, y1 + row, row)
    wps = []
    for k, yy in enumerate(ys):
        xa, xb = (x0, x1) if k % 2 == 0 else (x1, x0)
        wps.append((float(xa), float(min(yy, y1))))
        wps.append((float(xb), float(min(yy, y1))))
    return wps


def capture_positions(bounds: dict, fp_m: float, overlap: float = 0.30,
                      margin_m: float = 0.0):
    """Grid of (x, y) points at which to actually TRIGGER a frame capture:
    both along-track and cross-track spacing = fp_m * (1 - overlap).  The
    flight driver flies the boustrophedon rows and fires a capture whenever it
    passes near one of these — guarantees full overlap coverage regardless of
    flight speed / loop timing."""
    x0, x1 = bounds["x"];  y0, y1 = bounds["y"]
    x0 -= margin_m; x1 += margin_m; y0 -= margin_m; y1 += margin_m
    step = fp_m * (1.0 - overlap)
    xs = np.arange(x0 + step / 2.0, x1 + step, step)
    ys = np.arange(y0 + step / 2.0, y1 + step, step)
    pts = []
    for k, yy in enumerate(ys):
        row = xs if k % 2 == 0 else xs[::-1]
        pts += [(float(x), float(yy)) for x in row]
    return pts


# ---------------------------------------------------------------------------
# Projection: nadir BEV pixel/tile -> world (x, y), flat-ground assumption.
# Mirrors TerrainLeaderAgent._project_grid_cell EXACTLY (image 'up' = vehicle
# forward, 'right' = vehicle right, for simSetCameraPose(Pitch=-90)).
# ---------------------------------------------------------------------------
def project_norm(px: float, py: float, yaw: float, nx: float, ny: float,
                 fp_m: float):
    """(nx, ny) are normalised image coords in [-0.5, 0.5]: nx = image-right,
    ny = image-down.  Returns world (x, y) of that image point on flat ground."""
    fwd = (-ny) * fp_m
    rgt = (nx) * fp_m
    c, s = math.cos(yaw), math.sin(yaw)
    wx = fwd * c + rgt * (-s)
    wy = fwd * s + rgt * (c)
    return px + wx, py + wy


def project_tile_bbox(px, py, yaw, i, j, G, fp_m):
    """Axis-aligned world bbox ((x0,y0),(x1,y1)) of grid tile (row i, col j) of
    a G x G grid over the frame."""
    corners = [project_norm(px, py, yaw, (j + dj) / G - 0.5, (i + di) / G - 0.5, fp_m)
               for di in (0, 1) for dj in (0, 1)]
    xs = [c[0] for c in corners];  ys = [c[1] for c in corners]
    return (min(xs), min(ys)), (max(xs), max(ys))


def project_frame_bbox(px, py, yaw, fp_m):
    """World bbox of the WHOLE frame footprint (for the spatial index / coverage)."""
    return project_tile_bbox(px, py, yaw, 0, 0, 1, fp_m)


# ---------------------------------------------------------------------------
# The frame bank
# ---------------------------------------------------------------------------
@dataclass
class Frame:
    id: str
    x: float; y: float; z: float          # drone NED position at capture
    yaw: float                            # drone yaw (rad, NED)
    alt_above_ground_m: float             # z below the drone at capture (for footprint)
    footprint_m: float
    fov_deg: float = BEV_FOV_DEG
    img_w: int = 512
    img_h: int = 512
    bbox: list = field(default_factory=list)   # [[x0,y0],[x1,y1]] world footprint
    t: float = 0.0

    def image_path(self, bank_dir: str) -> str:
        return os.path.join(bank_dir, "frames", f"{self.id}.jpg")


class AerialFrameBank:
    """A per-map set of BEV frames + poses.  Directory layout:

        cache/aerial_bank/<map>/
            frames/000000.jpg ...
            index.json          # map, survey params, list[Frame]
    """

    def __init__(self, map_name: str, survey_alt_m: float,
                 weather: str = "clear", time_of_day: str = "noon"):
        self.map_name = map_name
        self.survey_alt_m = float(survey_alt_m)
        self.weather = weather
        self.time_of_day = time_of_day
        self.frames: list[Frame] = []

    # ---- build ----
    def add_frame(self, fid, pos_xyz, yaw, alt_above_ground_m, bgr, t=0.0):
        fp = footprint_m(alt_above_ground_m)
        (x0, y0), (x1, y1) = project_frame_bbox(pos_xyz[0], pos_xyz[1], yaw, fp)
        h, w = (bgr.shape[:2] if bgr is not None else (512, 512))
        fr = Frame(id=str(fid), x=float(pos_xyz[0]), y=float(pos_xyz[1]),
                   z=float(pos_xyz[2]), yaw=float(yaw),
                   alt_above_ground_m=float(alt_above_ground_m), footprint_m=fp,
                   img_w=int(w), img_h=int(h),
                   bbox=[[x0, y0], [x1, y1]], t=float(t))
        self.frames.append(fr)
        self._pending = getattr(self, "_pending", [])
        self._pending.append((fr, bgr))
        return fr

    def save(self, bank_dir: str):
        os.makedirs(os.path.join(bank_dir, "frames"), exist_ok=True)
        for fr, bgr in getattr(self, "_pending", []):
            if bgr is not None and cv2 is not None:
                cv2.imwrite(fr.image_path(bank_dir), bgr,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        self._pending = []
        with open(os.path.join(bank_dir, "index.json"), "w") as f:
            json.dump(dict(map_name=self.map_name, survey_alt_m=self.survey_alt_m,
                           weather=self.weather, time_of_day=self.time_of_day,
                           fov_deg=BEV_FOV_DEG,
                           frames=[asdict(fr) for fr in self.frames]), f, indent=1)

    @classmethod
    def load(cls, bank_dir: str) -> "AerialFrameBank":
        with open(os.path.join(bank_dir, "index.json")) as f:
            d = json.load(f)
        b = cls(d["map_name"], d["survey_alt_m"], d.get("weather", "?"),
                d.get("time_of_day", "?"))
        b.frames = [Frame(**fr) for fr in d["frames"]]
        b.dir = bank_dir
        return b

    # ---- query ----
    def frames_overlapping(self, aoi_bbox):
        """All frames whose footprint bbox intersects the axis-aligned
        ``aoi_bbox`` = ((x0,y0),(x1,y1))."""
        (ax0, ay0), (ax1, ay1) = aoi_bbox
        out = []
        for fr in self.frames:
            (fx0, fy0), (fx1, fy1) = fr.bbox
            if fx1 >= ax0 and fx0 <= ax1 and fy1 >= ay0 and fy0 <= ay1:
                out.append(fr)
        return out

    def load_image(self, fr: Frame):
        if cv2 is None:
            raise RuntimeError("cv2 required to load bank images")
        return cv2.imread(fr.image_path(self.dir))


# ---------------------------------------------------------------------------
# The dense relevance grid (built at task time from bank frames + a prompt)
# ---------------------------------------------------------------------------
UNOBSERVED_PRIOR = 0.05


class RelevanceGrid:
    """Bounded dense grid over an area of interest.  Each cell accumulates the
    SUM of tile-scores painted onto it and a count of observations; ``finalize``
    turns that into a per-cell mean (noise-cancelling)."""

    def __init__(self, origin_xy, extent_m, cell_m=25.0):
        self.origin = np.asarray(origin_xy, float)     # SW corner (min x, min y)
        self.cell = float(cell_m)
        self.W = max(1, int(math.ceil(extent_m[0] / self.cell)))
        self.H = max(1, int(math.ceil(extent_m[1] / self.cell)))
        self.score_sum = np.zeros((self.H, self.W), np.float32)
        self.obs_cnt = np.zeros((self.H, self.W), np.float32)
        self.relevance = None

    def _cell_range(self, bbox):
        (x0, y0), (x1, y1) = bbox
        c0 = int(math.floor((x0 - self.origin[0]) / self.cell))
        c1 = int(math.floor((x1 - self.origin[0]) / self.cell))
        r0 = int(math.floor((y0 - self.origin[1]) / self.cell))
        r1 = int(math.floor((y1 - self.origin[1]) / self.cell))
        c0, c1 = max(0, min(c0, c1)), min(self.W - 1, max(c0, c1))
        r0, r1 = max(0, min(r0, r1)), min(self.H - 1, max(r0, r1))
        return r0, r1, c0, c1

    def add_tile(self, bbox, score):
        r0, r1, c0, c1 = self._cell_range(bbox)
        if r0 > r1 or c0 > c1:
            return
        self.score_sum[r0:r1 + 1, c0:c1 + 1] += float(score)
        self.obs_cnt[r0:r1 + 1, c0:c1 + 1] += 1.0

    def finalize(self, stretch=True):
        n = np.maximum(self.obs_cnt, 1e-6)
        rel = self.score_sum / n
        seen = self.obs_cnt > 0
        # Per-task contrast stretch: per_frame_normalize centres every frame at
        # 0.5, so after multi-frame averaging the grid clusters near 0.5-0.6 and
        # DENSE_REL_GATE stops meaning anything.  Rescale THIS task's observed
        # range (p5..p95) to [0,1] so "relative to the rest of this AOI" is what
        # the gate sees, regardless of how uniformly the terms match the map.
        if stretch and seen.sum() > 20:
            lo, hi = np.percentile(rel[seen], [5, 95])
            if hi - lo > 1e-3:
                rel = np.clip((rel - lo) / (hi - lo), 0.0, 1.0)
        rel[~seen] = UNOBSERVED_PRIOR
        self.relevance = rel.astype(np.float32)
        self.coverage = float(seen.mean())
        return self

    # ---- follower-side consumption ----
    def query_window(self, center_xy, half_m, out_hw):
        """Resample an ``out_hw`` = (H, W) patch of ``relevance`` centred on
        ``center_xy`` spanning +/- ``half_m``, aligned so row 0 = -y (South),
        col 0 = -x (West) — caller maps it onto its own voxel grid.  Nearest
        cell (bilinear is overkill at 25 m).  Out-of-grid -> UNOBSERVED_PRIOR."""
        if self.relevance is None:
            self.finalize()
        oh, ow = out_hw
        cx, cy = float(center_xy[0]), float(center_xy[1])
        wx = np.linspace(cx - half_m, cx + half_m, ow)
        wy = np.linspace(cy - half_m, cy + half_m, oh)
        gx = np.floor((wx - self.origin[0]) / self.cell).astype(int)
        gy = np.floor((wy - self.origin[1]) / self.cell).astype(int)
        out = np.full((oh, ow), UNOBSERVED_PRIOR, np.float32)
        for ri, r in enumerate(gy):
            if not (0 <= r < self.H):
                continue
            for ci, c in enumerate(gx):
                if 0 <= c < self.W:
                    out[ri, ci] = self.relevance[r, c]
        return out

    def sample_at(self, wx, wy):
        """Nearest-cell relevance at world coords ``wx, wy`` (any matching shape,
        e.g. the follower's (40,40) voxel-grid world coords).  Cells outside the
        grid -> UNOBSERVED_PRIOR."""
        if self.relevance is None:
            self.finalize()
        wx = np.asarray(wx); wy = np.asarray(wy)
        c = np.floor((wx - self.origin[0]) / self.cell).astype(int)
        r = np.floor((wy - self.origin[1]) / self.cell).astype(int)
        inb = (c >= 0) & (c < self.W) & (r >= 0) & (r < self.H)
        out = np.full(wx.shape, UNOBSERVED_PRIOR, np.float32)
        cc = np.clip(c, 0, self.W - 1); rr = np.clip(r, 0, self.H - 1)
        out[inb] = self.relevance[rr[inb], cc[inb]]
        return out

    def save(self, path):
        np.savez_compressed(path, score_sum=self.score_sum, obs_cnt=self.obs_cnt,
                            relevance=(self.relevance if self.relevance is not None
                                       else np.zeros_like(self.score_sum)),
                            origin=self.origin, cell=self.cell,
                            HW=np.array([self.H, self.W]))

    @classmethod
    def load(cls, path):
        d = np.load(path)
        H, W = map(int, d["HW"])
        g = cls(d["origin"], (W * float(d["cell"]), H * float(d["cell"])), float(d["cell"]))
        g.score_sum = d["score_sum"]; g.obs_cnt = d["obs_cnt"]
        g.relevance = d["relevance"]
        g.coverage = float((g.obs_cnt > 0).mean())
        return g


def per_frame_normalize(scores: np.ndarray) -> np.ndarray:
    """Kill the "whole frame scores 0.9" flood: subtract the frame median,
    divide by its IQR, squash back to [0, 1].  A uniform frame -> all ~0.5
    (no directional info); a real stand-out tile still pokes above."""
    s = np.asarray(scores, np.float32)
    med = np.median(s)
    spread = np.subtract(*np.percentile(s, [75, 25]))
    if spread < 1e-6:                       # IQR degenerate (e.g. one outlier)
        spread = float(s.std())
    if spread < 1e-6:                       # genuinely uniform -> no direction info
        return np.full_like(s, 0.5)
    z = (s - med) / (1.349 * spread)        # ~unit std for a normal
    return np.clip(0.5 + 0.5 * z, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Area-of-interest helper
# ---------------------------------------------------------------------------
def aoi_bbox(center_xy, radius_m):
    cx, cy = float(center_xy[0]), float(center_xy[1])
    return (cx - radius_m, cy - radius_m), (cx + radius_m, cy + radius_m)


def radius_for_prompt(prompt: str, default_m: float = 900.0,
                      nearby_m: float = 400.0) -> float:
    """The 'Location: Nearby.' prompts have the target a median 107 m from
    spawn (vs 145 m overall, long tail to 1.9 km).  Shrink the AOI for them."""
    return nearby_m if "nearby" in (prompt or "").lower() else default_m
