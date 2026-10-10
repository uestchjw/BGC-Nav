"""Relevance-biased VLFM.

The SAME aerial-prior wrapper that `combo_soft` applies to APEX, applied to a
third, independent explorer: VLFM (Vision-Language Frontier Maps).  VLFM already
scores frontier points by a language-grounded value map (BLIP2-ITM) and drives
to the highest — so the aerial relevance field enters as a **second, bounded
value source** fused into that same frontier score:

    combined(f) = itm_value(f)  +  clip(GAMMA * (rel(f) - prior), 0, CAP)

- Bounded (`CAP` small): the aerial prior can re-rank frontiers VLFM is
  indifferent about (many have itm_value ~= 0 because the drone never looked
  that way), but cannot override a confident ITM hit.
- Revocable / self-limiting: if the surveyed relevance has no structure over the
  candidate frontiers (std < FLAT_STD) the bias is dropped -> pure VLFM.
- Stand-down: VLFM's `select_global_goal` already bypasses frontier scoring
  entirely when it has an unvisited person detection (Priority 1), so the bias
  is automatically inert once the follower has its own strong signal.

VLFM's BLIP2-ITM value map, YOLO detection, FMM planner and flight control are
untouched.  The bias lives in a ValueMap subclass -> the frontier selector code
is byte-for-byte the stock VLFM selector.
"""

import math
import os

import numpy as np

from baselines_adaption.vlfm.vlfm_agent import (
    VLFMAgent, GRID_SIZE, CELL_SIZE, FOV_W, MAX_DEPTH, SAMPLE_STEP,
)
from baselines_adaption.vlfm.value_map import ValueMap
from baselines_adaption.hl_coord.aerial_bank import UNOBSERVED_PRIOR

# Knobs (env-overridable).  Defaults tuned to VLFM's ITM value scale (~0-0.4):
# a bias worth at most ~0.10 re-ranks weak/tied frontiers without swamping a
# strong ITM signal.
VLFM_REL_GAMMA = float(os.environ.get("HL_VLFM_REL_GAMMA", 0.30))
VLFM_REL_CAP = float(os.environ.get("HL_VLFM_REL_CAP", 0.10))
VLFM_FLAT_STD = float(os.environ.get("HL_VLFM_FLAT_STD", 0.03))

# Give-up / elimination (the field_giveup idea, ported to VLFM). Otherwise the
# per-frontier relevance bias keeps re-ranking frontiers toward a nearby
# terrain-matched-but-empty patch every planning cycle. If the bias-chosen
# frontier stays within DWELL_TOL_M of itself for DWELL_MAX cycles, mask a
# GIVEUP_BOX_M-radius disc around it (relevance -> prior) for the rest of the
# episode so the lean moves on.
VLFM_GIVEUP = str(os.environ.get("HL_VLFM_GIVEUP", "false")).strip().lower() in ("1", "true", "yes", "on")
VLFM_DWELL_MAX = int(os.environ.get("HL_VLFM_DWELL_MAX", 8))
VLFM_DWELL_TOL_M = float(os.environ.get("HL_VLFM_DWELL_TOL_M", 40.0))
VLFM_GIVEUP_BOX_M = float(os.environ.get("HL_VLFM_GIVEUP_BOX_M", 80.0))

# --- Uncertainty-gated adaptive fusion (Idea E, Stage 0) -------------------
# Instead of a fixed gamma/cap, scale the aerial bias by the follower's OWN
# navigation uncertainty: how confident VLFM's own (local, no external API)
# ITM value map is about which frontier to pick right now. Confident steps
# (large margin between the best and second-best frontier) suppress the
# external bias; uncertain steps (close/tied candidates) let it through.
# Below UNCERTAINTY_GATE the bias is skipped entirely (event-triggered
# communication: no "query" of the relevance field on confident steps).
VLFM_UNCERTAINTY_ADAPTIVE = str(os.environ.get("HL_VLFM_UNCERTAINTY_ADAPTIVE", "false")).strip().lower() in ("1", "true", "yes", "on")
VLFM_UNCERTAINTY_MARGIN_SCALE = float(os.environ.get("HL_VLFM_UNCERTAINTY_MARGIN_SCALE", 0.15))  # ITM units; margin >= this -> u=0 (fully confident)
VLFM_UNCERTAINTY_GATE = float(os.environ.get("HL_VLFM_UNCERTAINTY_GATE", 0.15))  # u below this -> skip the bias/query entirely
# "full" = event-triggered stand-down AND continuous strength scaling (both
# pieces of the mechanism); "scale_only" = never stand down, only scale
# gamma/cap by u (isolates the adaptive-strength piece); "gate_only" = binary
# stand-down but fixed-strength gamma/cap once above the gate (isolates the
# event-triggering piece). Ablation switch for Idea E Stage 0.
VLFM_UNCERTAINTY_MODE = os.environ.get("HL_VLFM_UNCERTAINTY_MODE", "full").strip().lower()


class RelevanceBiasedValueMap(ValueMap):
    """VLFM ValueMap whose per-frontier score gets a bounded additive nudge from
    a world-space aerial relevance field.  No sampler -> identical to ValueMap."""

    def __init__(self, *a, relevance_sampler=None, gamma=VLFM_REL_GAMMA,
                 cap=VLFM_REL_CAP, **kw):
        super().__init__(*a, **kw)
        self._rel = relevance_sampler          # fn(wx, wy) -> relevance in [0,1]
        self.gamma = float(gamma)
        self.cap = float(cap)
        self.last_debug = None
        # give-up state
        self._exhausted = []                   # [(wx, wy)] discs masked out
        self._dwell_center = None
        self._dwell_steps = 0

    def _grid_to_world(self, gx, gy):
        wx = (np.asarray(gx, float) - self.grid_size / 2) * self.cell_size + self.center_x
        wy = (np.asarray(gy, float) - self.grid_size / 2) * self.cell_size + self.center_y
        return wx, wy

    def _own_uncertainty(self, base):
        """Follower's own navigation uncertainty from its local ITM scores
        alone (no external signal, no API call): 0 = confident (large gap
        between best and second-best candidate), 1 = torn between >=2
        near-tied candidates."""
        if len(base) < 2:
            return 0.0
        top2 = np.partition(np.asarray(base, np.float64), -2)[-2:]
        margin = float(top2[1] - top2[0])
        return float(np.clip(1.0 - margin / max(VLFM_UNCERTAINTY_MARGIN_SCALE, 1e-9), 0.0, 1.0))

    def score_frontiers(self, frontier_points, radius_cells=2):
        base = super().score_frontiers(frontier_points, radius_cells)
        if self._rel is None or len(frontier_points) == 0:
            return base

        u = self._own_uncertainty(base) if VLFM_UNCERTAINTY_ADAPTIVE else None
        gate_active = VLFM_UNCERTAINTY_ADAPTIVE and VLFM_UNCERTAINTY_MODE in ("full", "gate_only")
        scale_active = VLFM_UNCERTAINTY_ADAPTIVE and VLFM_UNCERTAINTY_MODE in ("full", "scale_only")
        if gate_active and u < VLFM_UNCERTAINTY_GATE:
            # Confident enough on its own -> event-triggered stand-down,
            # skip querying/using the aerial bias this step entirely.
            self.last_debug = {"n": len(frontier_points), "uncertainty": round(u, 3),
                               "fallback": "confident_standdown"}
            return base

        pts = np.asarray(frontier_points, float)          # (N, 2) = (gx, gy)
        wx, wy = self._grid_to_world(pts[:, 0], pts[:, 1])
        wx = np.asarray(wx); wy = np.asarray(wy)
        rel = np.asarray(self._rel(wx, wy), np.float64)
        if VLFM_GIVEUP and self._exhausted:               # mask given-up discs
            rel = rel.copy()
            for (ex, ey) in self._exhausted:
                rel[np.hypot(wx - ex, wy - ey) <= VLFM_GIVEUP_BOX_M] = UNOBSERVED_PRIOR
        if rel.std() < VLFM_FLAT_STD:                     # no structure -> pure VLFM
            self.last_debug = {"n": len(pts), "rel_std": float(rel.std()),
                               "fallback": "flat_relevance"}
            return base
        gamma, cap = self.gamma, self.cap
        if scale_active:
            # Scale the coupling strength by how uncertain the follower is of
            # its OWN pick right now -- confident steps get a weaker nudge,
            # uncertain (tied) steps let the aerial prior matter more.
            gamma, cap = self.gamma * u, self.cap * u
        bias = np.clip(gamma * (rel - UNOBSERVED_PRIOR), 0.0, cap)
        out = np.asarray(base, np.float64) + bias
        j = int(np.argmax(out))
        if VLFM_GIVEUP:                                   # dwell bookkeeping
            cx, cy = float(wx[j]), float(wy[j])
            if (self._dwell_center is not None
                    and math.hypot(cx - self._dwell_center[0],
                                   cy - self._dwell_center[1]) <= VLFM_DWELL_TOL_M):
                self._dwell_steps += 1
            else:
                self._dwell_center = (cx, cy)
                self._dwell_steps = 1
            if self._dwell_steps >= VLFM_DWELL_MAX:
                self._exhausted.append((cx, cy))
                self._dwell_center = None
                self._dwell_steps = 0
        self.last_debug = {
            "n": len(pts), "rel_std": round(float(rel.std()), 3),
            "chosen_itm": round(float(base[j]), 3),
            "chosen_rel": round(float(rel[j]), 3),
            "chosen_bias": round(float(bias[j]), 3),
            "itm_would_pick": int(np.argmax(base)), "combined_picks": j,
        }
        if VLFM_UNCERTAINTY_ADAPTIVE:
            self.last_debug["uncertainty"] = round(u, 3)
            self.last_debug["gamma_used"] = round(gamma, 3)
            self.last_debug["cap_used"] = round(cap, 3)
        return out


class FusedVLFMAgent(VLFMAgent):
    """VLFM with the aerial-relevance bias fused into its value map.
    Drop-in for VLFMAgent; identical BaseAgent API for the runner."""

    def __init__(self, client, task, relevance_grid=None, vocab=None):
        super().__init__(client, task, vocab=vocab)
        self.relevance_grid = relevance_grid
        sm = self.semantic_map
        self.value_map = RelevanceBiasedValueMap(
            grid_size=GRID_SIZE, cell_size=CELL_SIZE,
            center_x=sm.center_x, center_y=sm.center_y,
            camera_fov=FOV_W, max_depth=MAX_DEPTH, sample_step=SAMPLE_STEP,
            relevance_sampler=(relevance_grid.sample_at if relevance_grid is not None
                               else None),
        )
        mode = ("relevance-biased" if relevance_grid is not None
                else "PLAIN VLFM (no grid)")
        if VLFM_UNCERTAINTY_ADAPTIVE:
            mode += " + uncertainty-adaptive/event-triggered"
        print(f"[FusedVLFM] {mode} | gamma={VLFM_REL_GAMMA} cap={VLFM_REL_CAP} "
              f"uncertainty_adaptive={VLFM_UNCERTAINTY_ADAPTIVE} "
              f"gate={VLFM_UNCERTAINTY_GATE} margin_scale={VLFM_UNCERTAINTY_MARGIN_SCALE}")

    def fusion_snapshot(self):
        return {
            "agent": "fused_vlfm",
            "biased": self.relevance_grid is not None,
            "gamma": VLFM_REL_GAMMA, "cap": VLFM_REL_CAP, "flat_std": VLFM_FLAT_STD,
            "uncertainty_adaptive": VLFM_UNCERTAINTY_ADAPTIVE,
            "uncertainty_gate": VLFM_UNCERTAINTY_GATE,
            "uncertainty_margin_scale": VLFM_UNCERTAINTY_MARGIN_SCALE,
            "last_frontier_debug": getattr(self.value_map, "last_debug", None),
        }
