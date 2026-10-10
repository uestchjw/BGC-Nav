"""Relevance-biased Frontier-Based Exploration.

The SAME aerial-prior wrapper that `combo_soft` applies to APEX, applied instead
to a second, fully independent explorer (classic FBE).  Purpose: show the
coordination interface is a general, planner-agnostic wrapper — not something
tailored to APEX's value maps.

Mechanism.  Plain FBE picks the *largest* frontier cluster and drives to its
geometric midpoint.  On a real map the frontier is one big connected boundary
plus fragments, so "largest cluster" is winner-take-all and a small additive
nudge on a cluster score can never move it.  Instead we keep FBE's notion of
*which* regions are worth visiting (the substantial frontiers) but let the
aerial relevance field choose *where along them* to go:

  * Take the top-K frontier clusters by size (the parts of the boundary big
    enough to be worth exploring at all).
  * Score every frontier point on them by the leader's relevance field,
    minus a gentle proximity regulariser:

        score(p) = relevance(p)  -  LAMBDA_PROX * dist_norm(p, agent)

  * Go to the highest-scoring frontier point.
  * If relevance has no structure over the candidates (std < FLAT_STD, e.g.
    the survey never covered this area), fall back to plain FBE.

This only re-orders which frontier point is visited first; the set of
reachable frontier goals is unchanged (the frontier-selection analogue of
potential-based value shaping).  FBE's mapping, FMM planning and flight
control are untouched.  Stand-down gate: FBE's own `select_goal` already
bypasses frontier ranking when it has an unvisited person detection, so the
bias is automatically inert once the follower has its own signal.
"""

import os

import cv2
import numpy as np

from baselines_adaption.base.fbe_agent import (
    FBEAgent, GRID_SIZE, CELL_SIZE, REPLAN_DIST,
)
from baselines_adaption.semexp.frontier import FrontierGoalSelector
from baselines_adaption.hl_coord.aerial_bank import UNOBSERVED_PRIOR

# Knobs (env-overridable, same convention as hl_config).
FBE_TOPK = int(os.environ.get("HL_FBE_TOPK", 4))              # frontier clusters kept
FBE_PROX_LAMBDA = float(os.environ.get("HL_FBE_PROX_LAMBDA", 0.25))
FBE_DIST_SCALE_M = float(os.environ.get("HL_FBE_DIST_SCALE_M", 400.0))
FBE_FLAT_STD = float(os.environ.get("HL_FBE_FLAT_STD", 0.03))  # "no structure" cutoff
FBE_MAX_CAND = int(os.environ.get("HL_FBE_MAX_CAND", 300))    # subsample cap


class RelevanceBiasedFrontierSelector(FrontierGoalSelector):
    """FrontierGoalSelector whose frontier goal is chosen by a world-space
    relevance field (scored per frontier point, not per cluster).  Falls back to
    the unbiased parent when no sampler is given, or when relevance is flat."""

    def __init__(self, *a, relevance_sampler=None, grid_to_world=None,
                 prox_lambda=FBE_PROX_LAMBDA, dist_scale_m=FBE_DIST_SCALE_M,
                 topk=FBE_TOPK, **kw):
        super().__init__(*a, **kw)
        self._rel = relevance_sampler          # fn(wx, wy) -> relevance in [0,1]
        self._g2w = grid_to_world              # fn(gx, gy) -> (wx, wy)
        self.prox_lambda = float(prox_lambda)
        self.dist_scale_m = float(dist_scale_m)
        self.topk = int(topk)
        self.last_debug = None

    def _find_frontier_goal(self, semantic_map, agent_gx, agent_gy):
        if self._rel is None or self._g2w is None:
            return super()._find_frontier_goal(semantic_map, agent_gx, agent_gy)

        explored = semantic_map.get_explored_mask().astype(np.uint8)
        unknown = (explored == 0).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        edge = cv2.dilate(explored, kernel, iterations=1) - explored
        frontier_mask = cv2.bitwise_and(edge, unknown)

        contours, _ = cv2.findContours(
            frontier_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        big = [c for c in contours
               if cv2.contourArea(c) >= self.min_frontier_size
               or len(c) >= self.min_frontier_size]
        contours = big or contours

        # top-K clusters by size -> candidate frontier points (grid gx, gy)
        contours.sort(key=lambda c: -max(cv2.contourArea(c), len(c)))
        pts = []
        for c in contours[:max(1, self.topk)]:
            p = c.reshape(-1, 2)                 # rows = (col, row) = (gy, gx)
            pts.append(np.stack([p[:, 1], p[:, 0]], axis=1))   # -> (gx, gy)
        cand = np.concatenate(pts, axis=0).astype(np.float64)
        if len(cand) > FBE_MAX_CAND:
            cand = cand[np.linspace(0, len(cand) - 1, FBE_MAX_CAND).astype(int)]

        # relevance per candidate (world coords)
        wx, wy = self._g2w(cand[:, 0], cand[:, 1])
        rel = np.asarray(self._rel(np.asarray(wx), np.asarray(wy)), np.float64)

        # flat relevance -> nothing to steer by -> plain FBE
        if rel.std() < FBE_FLAT_STD:
            self.last_debug = {"n_cand": int(len(cand)), "rel_std": float(rel.std()),
                               "fallback": "flat_relevance"}
            return super()._find_frontier_goal(semantic_map, agent_gx, agent_gy)

        d = np.hypot(cand[:, 0] - agent_gx, cand[:, 1] - agent_gy) * self.cell_size
        dist_norm = np.clip(d / max(self.dist_scale_m, 1e-6), 0.0, 1.0)
        score = rel - self.prox_lambda * dist_norm

        j = int(np.argmax(score))
        goal = (int(np.clip(cand[j, 0], 0, self.grid_size - 1)),
                int(np.clip(cand[j, 1], 0, self.grid_size - 1)))
        self.last_debug = {
            "n_cand": int(len(cand)), "rel_std": round(float(rel.std()), 3),
            "chosen_rel": round(float(rel[j]), 3),
            "chosen_dist_m": round(float(d[j]), 1),
            "rel_max": round(float(rel.max()), 3),
        }
        print(f"[FBE-relbias] {len(cand)} cand pts | chose rel={rel[j]:.2f} "
              f"(max {rel.max():.2f}, std {rel.std():.2f}) dist={d[j]:.0f}m -> {goal}")
        return goal


class FusedFBEAgent(FBEAgent):
    """FBE with the aerial-relevance bias spliced into frontier selection.
    Drop-in for FBEAgent; identical BaseAgent API for the runner."""

    def __init__(self, client, task, relevance_grid=None,
                 fusion_gamma=None, bias_cap=None):
        super().__init__(client, task)
        self.relevance_grid = relevance_grid
        # fusion_gamma / bias_cap kept for call-site compatibility (unused by the
        # point-scoring mechanism); real knobs are the module HL_FBE_* env vars.
        self.goal_selector = RelevanceBiasedFrontierSelector(
            grid_size=GRID_SIZE,
            cell_size=CELL_SIZE,
            replan_dist=REPLAN_DIST,
            relevance_sampler=(relevance_grid.sample_at if relevance_grid is not None
                               else None),
            grid_to_world=self.semantic_map.grid_to_world,
        )
        mode = ("relevance-biased" if relevance_grid is not None
                else "PLAIN FBE (no grid)")
        print(f"[FusedFBE] {mode} | topk={FBE_TOPK} prox_lambda={FBE_PROX_LAMBDA} "
              f"dist_scale={FBE_DIST_SCALE_M}m")

    def fusion_snapshot(self):
        return {
            "agent": "fused_fbe",
            "biased": self.relevance_grid is not None,
            "topk": FBE_TOPK, "prox_lambda": FBE_PROX_LAMBDA,
            "dist_scale_m": FBE_DIST_SCALE_M, "flat_std": FBE_FLAT_STD,
            "last_frontier_debug": getattr(self.goal_selector, "last_debug", None),
        }
