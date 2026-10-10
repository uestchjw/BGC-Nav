"""Central, env-var-driven knob set for the high-low coordination experiment.

Every tunable in the high-low pipeline is read from an environment variable
here, once, at import time. This lets the 4-hour hyperparameter sweep launch
config after config without editing source between batches — the driver just
sets ``HL_*`` env vars and calls ``main_platform_high_low.py``.

``snapshot()`` returns all resolved values so each run's log records exactly
which config produced it.
"""
import os


def _f(name, default):
    return float(os.environ.get(name, default))


def _i(name, default):
    return int(os.environ.get(name, default))


def _s(name, default):
    return os.environ.get(name, default)


def _b(name, default):
    return str(os.environ.get(name, default)).strip().lower() in ("1", "true", "yes", "on")


# ============================ Fusion (follower) ============================
# How the shared terrain field is injected into the follower's APEX maps.
#   off      - no fusion (plain ApexAgent behaviour; isolates leader-presence cost)
#   floor    - original: raise attraction floor to gamma*A_high over matching regions
#   gradient - directional ramp in attraction toward the best region's bearing
#   explore  - bias the *exploration* map toward the best region (safest; never
#              fakes a target, just breaks action_predict's tie toward that heading)
#   hybrid   - explore bias + a small capped attraction nudge
#   frontier - self-extinguishing directional bias (the one that actually works):
#              a LOW-capped attraction ramp + gentle exploration pull in a wedge
#              toward the leader's best *unvisited* high-terrain region. The
#              target region is down-weighted by how long the follower has spent
#              near it (visit penalty) and hard-decayed once reached, so the
#              carrot recedes and the follower keeps moving instead of camping.
#              No stuck-gate needed.
FUSION_MODE = _s("HL_FUSION_MODE", "floor")

FUSION_GAMMA = _f("HL_FUSION_GAMMA", 0.5)
# Hard cap on any attraction value the fusion may write. action_predict treats
# >=0.9 as "target certain" (KEY_REWARD). Keep well below that for a nudge.
FUSION_CAP = _f("HL_FUSION_CAP", 1.0)
# Ignore leader regions scored below this — filters the VLM's "0.1 / none" noise.
FUSION_SCORE_GATE = _f("HL_FUSION_SCORE_GATE", 0.0)
# gradient/explore: half-angle (deg) of the wedge toward the best region that
# gets biased. Cells outside the wedge are untouched.
FUSION_WEDGE_DEG = _f("HL_FUSION_WEDGE_DEG", 60.0)
# explore/hybrid: how far below action_predict's RATE_CENTER (0.4) to pull the
# exploration value in the wedge (larger => stronger pull toward leader heading).
EXPLORE_DELTA = _f("HL_EXPLORE_DELTA", 0.30)

# Only apply fusion when the follower is actually stalled (protects tasks where
# plain APEX already makes progress, e.g. 101/303).
STUCK_GATE = _b("HL_STUCK_GATE", "false")
STUCK_WINDOW = _i("HL_STUCK_WINDOW", 8)          # steps of history to check
STUCK_DIST_M = _f("HL_STUCK_DIST_M", 25.0)       # net displacement below this => stuck

# Q-value-style feedback decay: each step, decay A_high for the region the
# follower currently occupies (covered, presumed no new evidence). 0 => off.
DECAY_ETA = _f("HL_DECAY_ETA", 0.0)

# ---- frontier mode ----
FRONTIER_CAP = _f("HL_FRONTIER_CAP", 0.18)          # max attraction the ramp may add
FRONTIER_EXPL_DELTA = _f("HL_FRONTIER_EXPL_DELTA", 0.12)  # exploration pull in the wedge
FRONTIER_DECAY_ETA = _f("HL_FRONTIER_DECAY_ETA", 0.12)    # hard-decay of the reached region
FRONTIER_DECAY_RADIUS = _i("HL_FRONTIER_DECAY_RADIUS", 1)  # + neighbour cells
VISIT_PENALTY = _f("HL_VISIT_PENALTY", 0.8)         # exp(-p*visits) down-weight of best region
FRONTIER_MAX_RANGE_M = _f("HL_FRONTIER_MAX_RANGE_M", 500.0)  # ignore best region beyond this
# Disable fusion after this follower step (0 => always on). Idea: use the leader
# to point the follower the right way *early*, then let plain APEX explore freely
# so total path length isn't suppressed by the follower committing to one region.
FRONTIER_MAX_STEP = _i("HL_FRONTIER_MAX_STEP", 0)
# Flood damping: with grid VLM scoring the terrain field lights up almost
# everywhere on mountain maps ("summit"/"ridge" match all over), making the
# prior uninformative AND over-nudging the follower into rough terrain
# (aotai crash spike in GA1). Scale the fusion amplitude by
# (1 - FLOOD_DAMP * frac_of_field_cells_at_or_above_the_score_gate).
# 0 => off. 1.0 => amp goes to ~0 when the whole field is 'high'.
FUSION_FLOOD_DAMP = _f("HL_FUSION_FLOOD_DAMP", 0.0)

# Fusion must be a TIE-BREAKER, not a FORCE (random-60 H1: helped lopnur 7x, hurt k2 10x
# by yanking the follower off targets APEX would have found):
#  - skip fusion once APEX has a real detection to home in on (max attraction >= this)
FRONTIER_DETECTION_GATE = _f("HL_FRONTIER_DETECTION_GATE", 0.30)
#  - skip fusion while the follower is clearly making progress on its own (only nudge
#    when it hasn't moved much lately). 0 => no such gate.
FRONTIER_STUCK_DIST_M = _f("HL_FRONTIER_STUCK_DIST_M", 0.0)
#  - don't redirect the follower's heading by more than this many degrees from its
#    recent travel direction (0 => no limit).
FRONTIER_MAX_TURN_DEG = _f("HL_FRONTIER_MAX_TURN_DEG", 0.0)
#  - obstacle-aware corridor veto: before ramping attraction toward the leader's
#    region, trace the straight line from the follower to that region across the
#    CURRENT obstacle map. If more than this fraction of the sampled corridor
#    cells are blocked columns, suppress the nudge this step (don't steer the
#    follower into terrain the front camera hasn't cleared yet — the source of
#    the ~9/72 crash tax on the failure subset). 0 => off.
FRONTIER_CORRIDOR_VETO = _f("HL_FRONTIER_CORRIDOR_VETO", 0.0)

# ================================ Leader =================================
LEADER_ALT_OFFSET_M = _f("HL_LEADER_ALT", 150.0)
LEADER_SPEED = _f("HL_LEADER_SPEED", 8.0)
LEADER_SWEEP_HALF_M = _f("HL_LEADER_SWEEP_HALF", 400.0)
LEADER_ROW_SPACING_M = _f("HL_LEADER_ROW_SPACING", 100.0)
LEADER_VLM_INTERVAL = _i("HL_LEADER_VLM_INTERVAL", 5)
# The DashScope call has no client-side timeout; a hung socket read would block
# the leader's single in-flight VLM slot forever and stop all further scoring
# for that task. If a call has been running longer than this (seconds), abandon
# it (zombie thread + generation counter) and allow a new one.
LEADER_VLM_TIMEOUT_S = _f("HL_LEADER_VLM_TIMEOUT_S", 25.0)
# boustrophedon - lawnmower rows (original)
# spiral        - outward square spiral: covers the follower's vicinity FIRST,
#                 so the field is useful while the follower can still act on it
LEADER_SWEEP_MODE = _s("HL_LEADER_SWEEP_MODE", "boustrophedon")

# ---- Option A: grid the VLM output + project the BEV footprint ----
# The BEV camera at LEADER_ALT_OFFSET_M with a 90deg FOV sees a ~2*alt metre
# square of ground, but the original scorer collapsed that whole frame into ONE
# scalar written to the single 50m cell under the drone (~90% of the frame's
# spatial info discarded). With LEADER_VLM_GRID = G > 1 the VLM instead scores a
# GxG grid over the frame; each cell's score is projected to its world (x,y)
# via the leader's pose + altitude + FOV and written to the terrain field.
# G = 1 keeps the exact original point behaviour.
LEADER_VLM_GRID = _i("HL_LEADER_VLM_GRID", 1)
# Full ground footprint width (metres) of one BEV frame. 0 => auto:
# 2 * LEADER_ALT_OFFSET_M * tan(FOV/2), FOV = 90deg  =>  2 * alt.
LEADER_FOOTPRINT_M = _f("HL_LEADER_FOOTPRINT_M", 0.0)
# Draw the grid lines onto the image before sending to the VLM (helps it localise).
LEADER_GRID_DRAW = _b("HL_LEADER_GRID_DRAW", "true")
# Safety: also write the frame's overall/centre score to the cell directly under
# the drone (correct even if the grid->world rotation is off). 1 => on.
LEADER_GRID_KEEP_CENTER = _b("HL_LEADER_GRID_KEEP_CENTER", "true")
# Source-level flood fix: only record a grid frame if it has spatial CONTRAST
# (max cell score - min cell score >= this). A uniform frame ("the whole view
# is summit") carries no directional information for the follower and just
# floods the field on terrain-saturated maps. 0 => record every frame.
LEADER_GRID_MIN_CONTRAST = _f("HL_LEADER_GRID_MIN_CONTRAST", 0.0)

# ============================ ROI mode (continuous) ============================
# `HL_FUSION_MODE=roi` : the leader's field defines regions-of-interest with a
# CONFIDENCE; the follower runs a TRANSIT/SEARCH state machine that persists for
# the whole mission (no early-only cutoff). It heads to the highest-confidence
# unvisited ROI, then does a confidence-scaled coverage sweep inside it behind a
# confidence-scaled soft fence, and RELEASES the ROI on coverage-done OR a
# confidence-scaled dwell cap OR a real detection. Release is coverage-based
# (not "stuck"-based) so the fixation feedback loop of always-on fusion doesn't
# form.
ROI_BOX_M = _f("HL_ROI_BOX_M", 90.0)              # search box radius around an ROI centre
ROI_CONF_MODE = _s("HL_ROI_CONF_MODE", "raw")     # raw | contrast | const (=1, ablation)
ROI_COVER_FRAC = _f("HL_ROI_COVER_FRAC", 0.65)    # in-box coverage that counts as "searched"
ROI_DWELL_MIN = _i("HL_ROI_DWELL_MIN", 12)        # min search steps per ROI (low confidence)
ROI_DWELL_MAX = _i("HL_ROI_DWELL_MAX", 45)        # max search steps per ROI (high confidence)
ROI_FENCE = _f("HL_ROI_FENCE", 0.25)              # soft-fence strength at confidence 1
ROI_SWEEP_DELTA = _f("HL_ROI_SWEEP_DELTA", 0.15)  # in-box exploration pull at confidence 1
ROI_ATTRACT_CAP = _f("HL_ROI_ATTRACT_CAP", 0.15)  # transit attraction ramp cap
ROI_MAX = _i("HL_ROI_MAX", 4)                     # give up leader guidance after this many ROIs

# The leader steers the follower into steep/cluttered terrain (ridge/forest),
# where APEX's front-camera-only obstacle map lags and it flies into things it
# hasn't seen. Let the fused follower build an OMNIDIRECTIONAL obstacle map
# from all 4 cameras. HL-follower only; the APEX baseline stays front-only.
OBSTACLE_360 = _b("HL_OBSTACLE_360", "true")

# ===================== densemap mode (survey-then-search) =====================
# `HL_FUSION_MODE=densemap`: the follower reads a PRE-BUILT full-coverage
# RelevanceGrid (from an offline high-altitude survey + prompt-conditioned
# scoring, baselines_adaption/hl_coord/aerial_bank.py) instead of a live
# leader's sparse field. Continuous, whole-mission, tie-breaker semantics:
#   - guided exploration pull toward still-unexplored cells whose relevance is
#     above DENSE_REL_GATE (scaled up to DENSE_EXPL_GAIN at relevance 1.0)
#   - a hard-capped attraction bump (<= DENSE_ATTRACT_CAP) for cells above
#     DENSE_REL_HI only, to break action_predict's rotate-left tie toward them
#   - stands down entirely once APEX has its own detection (FRONTIER_DETECTION_GATE)
DENSE_REL_GATE = _f("HL_DENSE_REL_GATE", 0.55)
DENSE_EXPL_GAIN = _f("HL_DENSE_EXPL_GAIN", 0.25)
DENSE_REL_HI = _f("HL_DENSE_REL_HI", 0.80)
DENSE_ATTRACT_GAIN = _f("HL_DENSE_ATTRACT_GAIN", 0.50)
DENSE_ATTRACT_CAP = _f("HL_DENSE_ATTRACT_CAP", 0.12)
# offline builder: grid tile count per frame, and RelevanceGrid cell size (m)
DENSE_CELL_M = _f("HL_DENSE_CELL_M", 25.0)
LEADER_VLM_GRID_BUILD = _i("HL_LEADER_VLM_GRID_BUILD", 8)

# ================= Paper baselines (training-free reproductions) =================
# CoNav-UAV-TF (`HL_FUSION_MODE=conav`): leader = open-vocab target detection on
# the (amortised) BEV survey + optional VLM confirm -> a priority-ordered queue of
# candidate target world-coords, re-ranked each step by score / path-efficiency
# from the follower.  Follower runs a TRANSIT/SEARCH machine toward the queue head
# (the follower's "task" is defined by the dispatch), releasing on
# coverage / dwell / its own detection, then advancing the queue.
CONAV_ARRIVE_M = _f("HL_CONAV_ARRIVE_M", 20.0)      # counts as "at" the queue-head coord
CONAV_BOX_M = _f("HL_CONAV_BOX_M", 80.0)            # local search box radius on arrival
CONAV_DWELL_MAX = _i("HL_CONAV_DWELL_MAX", 22)      # search steps before giving up a coord
CONAV_TRANSIT_MAX = _i("HL_CONAV_TRANSIT_MAX", 40)  # transit steps before abandoning a coord
CONAV_ATTRACT_CAP = _f("HL_CONAV_ATTRACT_CAP", 0.15)
CONAV_SWEEP_DELTA = _f("HL_CONAV_SWEEP_DELTA", 0.20)
CONAV_MAX_TURN_DEG = _f("HL_CONAV_MAX_TURN_DEG", 90.0)  # follower "always executes toward head" -> loose
CONAV_PATHEFF_K = _f("HL_CONAV_PATHEFF_K", 150.0)   # re-rank: score / (1 + dist/K)

# AeroDuo-TF (`HL_FUSION_MODE=aeroduo`): leader = zero-shot referring segmentation
# (CLIPSeg) of the TARGET phrase over the (amortised) BEV ortho survey -> a target
# location probability grid M.  Follower's subgoal = centroid of M over the
# not-yet-searched region; A*/attraction toward it; on coverage/dwell it masks
# that region out of M and recomputes the centroid ("requests new guidance").
AERODUO_ARRIVE_M = _f("HL_AERODUO_ARRIVE_M", 25.0)
AERODUO_BOX_M = _f("HL_AERODUO_BOX_M", 80.0)        # region masked out on completion
AERODUO_DWELL_MAX = _i("HL_AERODUO_DWELL_MAX", 30)  # steps per subgoal before replanning
AERODUO_ATTRACT_CAP = _f("HL_AERODUO_ATTRACT_CAP", 0.15)
AERODUO_MIN_MASS = _f("HL_AERODUO_MIN_MASS", 0.15)  # stop when remaining M mass-fraction < this
AERODUO_PROB_GATE = _f("HL_AERODUO_PROB_GATE", 0.5) # cells below this don't count toward the centroid
# Follower faithfulness to AeroDuo's "A* + trained navigator drives to the
# centroid" (our attraction-bias version is much gentler than the paper):
#   HL_AERODUO_GOTO=true  -> the follower directly rotates toward the centroid and
#     moveToPositionAsync a step toward it each control step (APEX's discrete
#     action is bypassed); it defers to APEX only when the straight step is
#     obstacle-blocked or once APEX has its own detection.
AERODUO_GOTO = _b("HL_AERODUO_GOTO", "false")
AERODUO_GOTO_STEP_M = _f("HL_AERODUO_GOTO_STEP_M", 40.0)  # metres per committed step

# Subgoal selection: the "global" default mass-weights the centroid over the
# WHOLE remaining relevance mass, which averages across separated relevant
# regions (e.g. two summits) into empty space between them.  Two alternatives
# that keep the same soft/capped/gated consumption downstream, only changing
# which point (cx, cy) becomes the wedge direction:
#   global     - centroid of all remaining mass (baseline, unchanged behaviour)
#   distweight - centroid of mass reweighted by exp(-dist/D) from the drone,
#                so a nearby concentration wins over an equally-massive far one
#   cluster    - connected-component the remaining mass; score each component
#                by mass/(1+dist/D) (gain/cost); target the best component's
#                OWN centroid (never blends across separated components)
AERODUO_CENTROID_MODE = _s("HL_AERODUO_CENTROID_MODE", "global")
AERODUO_DIST_SCALE_M = _f("HL_AERODUO_DIST_SCALE_M", 250.0)  # D in the two formulas above

# "adaptive": explore/exploit gate on how CONCENTRATED the remaining relevance
# mass is (top-cluster mass / total remaining mass). Diffuse/multi-modal
# (concentration < thresh) -> broad, low-commitment mean-direction sweep
# (= "global"); concentrated / effectively single-candidate (>= thresh) ->
# precise commitment to that one cluster's own centroid (= "cluster"). Both
# sub-behaviours are the SAME code paths as centroid-mode global/cluster; this
# only chooses between them per-decision based on the field's own shape.
AERODUO_CONCENTRATION_THRESH = _f("HL_AERODUO_CONCENTRATION_THRESH", 0.6)

# Local-horizon window: the aerial prior only extends the follower's OWN
# limited sensing range a bit, it does not dispatch it across the map. When
# picking a NEW subgoal, mass beyond this radius of the follower's current
# position is ignored entirely (0 = unbounded, legacy behaviour). Orthogonal
# to AERODUO_CENTROID_MODE -- composes with global/distweight/cluster/adaptive.
# 250 m ~= a bit past APEX's own local voxel map (200 m span).
AERODUO_WINDOW_M = _f("HL_AERODUO_WINDOW_M", 0.0)

# `HL_FUSION_MODE=field`: stateless per-cell relevance field, no subgoal / no
# centroid / no visited-dwell-arrival state machine.
#   (a) direct per-cell bump on the follower's own local map, proportional to
#       that cell's own sampled relevance (smooth, not threshold-split).
#   (b) a small directional lean toward the relevance-mass-weighted mean
#       bearing within FIELD_WINDOW_M of the drone -- recomputed fresh every
#       step (never stored), so there is nothing to arrive at or give up on;
#       it just continuously leans wherever the NEARBY field currently favours.
FIELD_WINDOW_M = _f("HL_FIELD_WINDOW_M", 500.0)
FIELD_GAMMA_LOCAL = _f("HL_FIELD_GAMMA_LOCAL", 0.5)
FIELD_CAP_LOCAL = _f("HL_FIELD_CAP_LOCAL", 0.12)
FIELD_CAP_LEAN = _f("HL_FIELD_CAP_LEAN", 0.10)
FIELD_CAP_TOTAL = _f("HL_FIELD_CAP_TOTAL", 0.15)
# B: sharpen the lean's mean-bearing estimate by discounting mass by distance
# within the window (same idea as AERODUO's distweight centroid, applied here
# only to the directional component).
FIELD_LEAN_DISTWEIGHT = _b("HL_FIELD_LEAN_DISTWEIGHT", "false")
FIELD_LEAN_DIST_SCALE_M = _f("HL_FIELD_LEAN_DIST_SCALE_M", 250.0)
# D: route the directional component into the EXPLORATION map (bias search
# order, like densemap's tier-1) instead of the attraction map, so it can't be
# diluted/dominated by attraction's own weighting in action_predict.
FIELD_LEAN_TARGET = _s("HL_FIELD_LEAN_TARGET", "attraction")  # attraction|exploration
FIELD_EXPL_GAIN = _f("HL_FIELD_EXPL_GAIN", 0.25)

# Give-up mechanism: field is otherwise fully stateless, so a lean toward a
# nearby-but-unproductive patch (terrain matched, no real target) re-attracts
# EVERY step forever -- no way to disengage. If the lean's mean-bearing target
# stays within FIELD_DWELL_TOL_M of itself for FIELD_DWELL_MAX steps (with no
# detection, i.e. GATE A never firing), mask a FIELD_GIVEUP_BOX_M radius around
# it for the rest of the episode (same idea as AERODUO's visited-masking, but
# triggered by dwell time near a recurring lean target instead of arrival at a
# committed subgoal).
#
# The exhausted-centre list is otherwise unbounded: on a long episode (extra
# time budget, or a far/slow task) it keeps accumulating and never decays, so
# the masked-out area can grow to swallow most of the map, degrading the field
# toward no-guidance regardless of whether those regions actually held the
# target. FIELD_GIVEUP_MAX_EXHAUSTED caps it -- oldest entry is evicted (FIFO)
# once the cap is hit, so a long episode can still give up on new patches
# without permanently blacklisting an ever-growing fraction of the map.
FIELD_GIVEUP = _b("HL_FIELD_GIVEUP", "false")
FIELD_DWELL_MAX = _i("HL_FIELD_DWELL_MAX", 30)
FIELD_DWELL_TOL_M = _f("HL_FIELD_DWELL_TOL_M", 40.0)
FIELD_GIVEUP_BOX_M = _f("HL_FIELD_GIVEUP_BOX_M", 80.0)
FIELD_GIVEUP_MAX_EXHAUSTED = _i("HL_FIELD_GIVEUP_MAX_EXHAUSTED", 8)

# Stuck-aware lean escalation: the lean's cap is deliberately small (an order
# of magnitude below a genuine detection's reward) so it can only break ties
# among otherwise-similar candidates -- but that also means it can be too
# weak to pull the follower out of a local trap where APEX's own
# VISIT_PENALTY/FRONTIER_DECAY has pushed every nearby candidate negative
# (task 42 case study: the follower rotated in place for 20 steps while its
# own reward landscape decayed, and the lean toward the correct distant
# region never got a chance to dominate). Reusing the existing _is_stuck
# (STUCK_WINDOW/STUCK_DIST_M) signal, ONLY while the follower is physically
# stalled do we relax the lean/total caps -- still just a floor on A_t,
# still gated by stand-down and the obstacle veto, still far below a real
# detection's reward scale; the moment it moves again (own signal recovers,
# or the stronger lean itself gets it moving), the normal small caps return.
FIELD_STUCK_BOOST = _b("HL_FIELD_STUCK_BOOST", "false")
FIELD_CAP_LEAN_STUCK = _f("HL_FIELD_CAP_LEAN_STUCK", 0.22)
FIELD_CAP_TOTAL_STUCK = _f("HL_FIELD_CAP_TOTAL_STUCK", 0.30)

# Forward-unwedge: APEX's obstacle_map (baselines_adaption/apex/voxel_maps.py)
# is write-once and never decays, and is world-frame-fixed -- shift_maps only
# rolls it on translation past SHIFT_THRESHOLD, never on rotation. So once a
# stray depth read (e.g. a close-range frame taken while still near terrain
# right after an embedded-spawn climb-out) marks the "forward" strip as
# blocked, no amount of rotating can ever clear it or change what "forward"
# means -- the follower is wedged for the rest of the episode regardless of
# lean strength (root-caused on task 33/42, which spawn embedded in terrain).
# If _is_stuck holds for FIELD_UNWEDGE_AFTER consecutive checks, clear ONLY
# the exact obstacle cells action_predict.py's "forward" safety check reads
# (a short strip straight ahead at the drone's own local y,z) -- narrowly
# targeted, not a general clear, to avoid deleting real nearby obstacles.
FIELD_UNWEDGE = _b("HL_FIELD_UNWEDGE", "false")
FIELD_UNWEDGE_AFTER = _i("HL_FIELD_UNWEDGE_AFTER", 10)

# Per-step diagnostics for _fuse_field: prints "[DBG] ..." lines with action_predict
# rewards computed with and without the field bias. Default off; no behaviour change.
FIELD_DEBUG = _b("HL_FIELD_DEBUG", "false")

# The field bias is written into the persistent attraction map with np.maximum. If
# it stays there, a bias above FRONTIER_DETECTION_GATE (0.30) is read back next step
# by the stand-down gate as a real detection, and the field switches itself off for
# the rest of the episode (any FIELD_CAP_TOTAL >= 0.30; found on task 416). With
# FIELD_TRANSIENT the bias only steers the current step's action and is removed
# right after action selection, matching _fuse_field's "nothing is stored between
# steps" design. Set false to reproduce the old (persistent) behaviour.
FIELD_TRANSIENT = _b("HL_FIELD_TRANSIENT", "true")

# Stand-down gate evidence. APEX's own VLM prompt (unchanged, initial commit) asks for
# "terrain features suggesting human activity" (trail, ridge, ...), and every such
# detection is written into the attraction map with the VLM's score (0.5-0.7), which
# is above FRONTIER_DETECTION_GATE (0.30). One 'hiking trail' box therefore silenced
# the whole field for 77 of 183 steps on task 416. With FIELD_GATE_TARGET_ONLY the
# gate only counts attraction that did NOT come from a VLM terrain-class detection
# (person/clue classes, and the default cue-vocabulary detector, still count).
# APEX itself is untouched; this only changes what OUR gate reads.
FIELD_GATE_TARGET_ONLY = _b("HL_FIELD_GATE_TARGET_ONLY", "true")
FIELD_GATE_TARGET_TERMS = _s(
    "HL_FIELD_GATE_TARGET_TERMS",
    "person,people,human,survivor,man,woman,child,hiker,climber,tent,backpack,bag,sleepingbag,"
    "campfire,camp,campsite,fire,flare,smoke,vehicle,car,truck,bottle,flashlight,jacket,coat,"
    "clothing,shoe,boot,helmet,rope,blanket,footprint,footprints,gear")

# Lean direction estimator:
#   mean    - relevance-mass-weighted mean bearing (= a windowed centroid; the
#             legacy behaviour; averages multi-modal fields into the gap)
#   vecsum  - vector sum of per-cell pull vectors with a nonlinear relevance
#             weight g(rel)=clip(rel-GATE,0,1)^P and a distance falloff
#             h=exp(-d/D). Mode-selecting (nearest x strongest concentration
#             dominates), and self-silencing: if the summed vector is small
#             relative to the total weight (concentration < MIN_CONC) the field
#             is diffuse / symmetric -> drop the lean this step.
FIELD_LEAN_MODE = _s("HL_FIELD_LEAN_MODE", "mean")     # mean | vecsum | cluster
# vecsum: loosened defaults -- the first sweep (GATE .5/POW 2/CONC .3) silenced
# the lean on the diffuse nadir grids (concentration almost never cleared the
# gate). GATE .3 keeps more mass, POW 1 = linear, CONC .1 barely gates.
FIELD_VS_GATE = _f("HL_FIELD_VS_GATE", 0.3)
FIELD_VS_POWER = _f("HL_FIELD_VS_POWER", 1.0)
FIELD_VS_DIST_SCALE_M = _f("HL_FIELD_VS_DIST_SCALE_M", 150.0)
FIELD_VS_NEAR_CUT_M = _f("HL_FIELD_VS_NEAR_CUT_M", 25.0)
FIELD_VS_MIN_CONC = _f("HL_FIELD_VS_MIN_CONC", 0.10)

# --- Lever 1: commit / anti-wander --------------------------------------------
# Diagnostic on the midrange subset: the follower reaches its closest point to
# the target mid-episode (NEmin at step ~10-21) and then the exploration-map
# frontier drive pulls it straight back out -- final distance 2-4x the NEmin.
# GATE A stands the whole field bias down as soon as APEX's own attraction map
# hits 0.30 (a weak frontier signal, not a real detection), handing the follower
# to native APEX which then leaves the hot zone.
#
# COMMIT: when the drone's own cell is in the aerial hot zone (rel >= HOT_REL)
# and there is no *strong* detection, anchor a search disc at the current
# position: put an attraction FLOOR inside the disc and fence the exploration
# map OUTSIDE it, so APEX orbits the disc instead of wandering off. Release the
# anchor on a strong detection or on leaving the zone; exhaust it (mask like a
# give-up centre) after MAX_STEPS with nothing found. Requires the softer
# GATE A threshold STANDDOWN_ATTRACT so the FLOOR we write doesn't self-trip it.
FIELD_COMMIT = _b("HL_FIELD_COMMIT", "false")
FIELD_COMMIT_HOT_REL = _f("HL_FIELD_COMMIT_HOT_REL", 0.50)   # drone-cell relevance to trigger
FIELD_COMMIT_BOX_M = _f("HL_FIELD_COMMIT_BOX_M", 90.0)       # commit-disc radius (peaked FLOOR falls to 0 here)
FIELD_COMMIT_FLOOR = _f("HL_FIELD_COMMIT_FLOOR", 0.45)       # attraction FLOOR at the disc centre (peaked, linear falloff)
FIELD_COMMIT_MAX_STEPS = _i("HL_FIELD_COMMIT_MAX_STEPS", 20) # steps anchored w/o detection -> exhaust
FIELD_COMMIT_STANDDOWN_ATTRACT = _f("HL_FIELD_COMMIT_STANDDOWN_ATTRACT", 0.90)  # real-detection GATE A

# --- Lever 2b: in-zone nadir detection pass ----------------------------------
# APEX's 4 cameras are all Pitch 0 (horizontal). At 25-100 m AGL a victim near
# the drone's ground position is straight down and invisible to every camera --
# the follower can fly directly over the target and detect nothing (near-miss
# tasks: reached < 120 m, zero detections). This adds a downward-looking camera
# ("nadir_camera" in the settings) and runs APEX's own _run_detection on it
# (reusing report_clue / report_target unchanged) whenever the drone is in the
# aerial hot zone (drone-cell relevance >= HOT_REL) and has not reported a
# target yet. Off by default.
FIELD_NADIR_DET = _b("HL_FIELD_NADIR_DET", "false")
FIELD_NADIR_DET_HOT_REL = _f("HL_FIELD_NADIR_DET_HOT_REL", 0.50)  # gate; reuses the commit hot threshold by default

# --- lean brake: crash reduction for field_giveup -------------------------- #
# field_giveup crashes ~10% on the midrange subset (vs APEX 0%, combo_windowed
# 2%) -- the continuous per-step directional lean commits the follower to a
# heading that, on k2/aotai slopes, ends in a terrain clip before the local
# obstacle voxels populate. When on, cast a short ray from the drone toward the
# lean-target bearing; if it crosses an observed blocked obstacle column within
# FIELD_LEAN_BRAKE_M, drop the directional push for that step (the local
# per-cell bump still applies). Principled: keeps fusion a tie-breaker, never a
# steer-into-terrain force.
FIELD_LEAN_BRAKE = _b("HL_FIELD_LEAN_BRAKE", "false")
FIELD_LEAN_BRAKE_M = _f("HL_FIELD_LEAN_BRAKE_M", 30.0)   # was 60; APEX's own forward check only looks ~20 m

# ============================= Terrain field =============================
CELL_SIZE_M = _f("HL_CELL_SIZE", 50.0)


def snapshot():
    """All resolved knob values, for logging."""
    return {k: v for k, v in globals().items()
            if k.isupper() and not k.startswith("_")}


# --- Autonomy of the low-altitude UAV (Ours field mode) -----------------------
# The aerial belief is advisory. Two ways the follower's OWN evidence overrides it:
#  * NEG_EVIDENCE: every step the follower flies, belief cells within OBS_RADIUS_M of it
#    count as observed (a step with no detection = negative evidence); the belief there
#    decays as exp(-count / NEG_TAU_STEPS) toward the unobserved prior. This replaces the
#    disc-shaped dwell give-up (which masked an 80 m disc around the cluster centroid even
#    if the follower had not looked at it), so a cluster is only dropped as far as the
#    follower has actually verified it.
#  * ARRIVE: within ARRIVE_R_OUT_M of the lean target the directional lean ramps down to
#    ARRIVE_FLOOR (at <= ARRIVE_R_IN_M) so APEX's own exploration/verification takes over
#    near the target instead of being pulled through it.
FIELD_NEG_EVIDENCE = _b("HL_FIELD_NEG_EVIDENCE", "false")
FIELD_OBS_RADIUS_M = _f("HL_FIELD_OBS_RADIUS_M", 60.0)
FIELD_NEG_TAU_STEPS = _f("HL_FIELD_NEG_TAU_STEPS", 12.0)
FIELD_ARRIVE = _b("HL_FIELD_ARRIVE", "false")
FIELD_ARRIVE_R_IN_M = _f("HL_FIELD_ARRIVE_R_IN_M", 60.0)
FIELD_ARRIVE_R_OUT_M = _f("HL_FIELD_ARRIVE_R_OUT_M", 150.0)
FIELD_ARRIVE_FLOOR = _f("HL_FIELD_ARRIVE_FLOOR", 0.25)

# AeroDuo baseline, "visited" bookkeeping. Default (false) = original behaviour: the
# SUBGOAL coordinate is appended to the visited list on arrival/dwell timeout.
# VISIT_ACTUAL=true: only places the follower has actually FLOWN through count -- its own
# position is appended (spaced >= VISIT_SPACING_M apart) every step, the subgoal is never
# marked visited by itself, and each visited point masks AERODUO_BOX_M around it.
AERODUO_VISIT_ACTUAL = _b("HL_AERODUO_VISIT_ACTUAL", "false")
AERODUO_VISIT_SPACING_M = _f("HL_AERODUO_VISIT_SPACING_M", 10.0)

# Obstacle test used by the field lean brake, the field/aeroduo bump masking and the
# aeroduo go-to segment check. Old behaviour: a column counts as blocked if ANY z-layer
# holds an obstacle voxel -- including ground far below the drone, which on mountain maps
# made the lean brake fire on ~83% (median) of guided steps. ALT_AWARE: only the drone's
# own layer and FIELD_OBS_BELOW_LAYERS layers below it (5 m each) count.
FIELD_ALT_AWARE_OBSTACLE = _b("HL_FIELD_ALT_AWARE_OBSTACLE", "false")
FIELD_OBS_BELOW_LAYERS = _i("HL_FIELD_OBS_BELOW_LAYERS", 1)

# Climb-over instead of plain brake: when the lean segment is blocked at the drone's altitude
# band but clear at a band raised by up to CLIMB_MAX_LAYERS (5 m each), keep the lean's
# bearing and put attraction (<= CLIMB_ATTRACT, scaled by the wedge) on the layers ABOVE the
# drone inside the wedge so action_predict prefers "ascend". Horizontal lean stays braked.
FIELD_CLIMB = _b("HL_FIELD_CLIMB", "false")
FIELD_CLIMB_MAX_LAYERS = _i("HL_FIELD_CLIMB_MAX_LAYERS", 4)
FIELD_CLIMB_ATTRACT = _f("HL_FIELD_CLIMB_ATTRACT", 0.95)   # >= APEX KEY_THRESHOLD 0.9 so ascend clearly beats its -10 penalty

# Descend-back (companion of FIELD_CLIMB): once the drone is more than FIELD_DESC_MIN_M above
# the altitude it had at its first guided step, the lean path is clear at its current band and
# still clear one layer lower, put attraction (<= FIELD_DESC_ATTRACT, wedge-scaled) on the
# layers BELOW it so action_predict prefers "descend" and it returns to search height.
FIELD_DESCEND = _b("HL_FIELD_DESCEND", "false")
FIELD_DESC_MIN_M = _f("HL_FIELD_DESC_MIN_M", 8.0)
FIELD_DESC_ATTRACT = _f("HL_FIELD_DESC_ATTRACT", 0.95)

# AeroDuo go-to "no progress" fallback: if a committed go-to step (up to AERODUO_GOTO_STEP_M)
# moves the follower less than AERODUO_GOTO_MIN_PROGRESS_M for AERODUO_GOTO_STUCK_STEPS
# consecutive steps (blocked by terrain the voxel map does not show), suspend go-to for
# AERODUO_GOTO_SUSPEND_STEPS steps and let APEX's own discrete actions (ascend/rotate/...) recover.
AERODUO_GOTO_MIN_PROGRESS_M = _f("HL_AERODUO_GOTO_MIN_PROGRESS_M", 5.0)
AERODUO_GOTO_STUCK_STEPS = _i("HL_AERODUO_GOTO_STUCK_STEPS", 3)
AERODUO_GOTO_SUSPEND_STEPS = _i("HL_AERODUO_GOTO_SUSPEND_STEPS", 10)
