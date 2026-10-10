#!/bin/bash
# Launches the paper's best "Ours" config (FusedApexAgent, fusion-mode=field,
# Claude-scored belief map) via SurveyPlatform (single-vehicle follower).
#
# Prereqs: ESARBench installed (maps + task_data.json + cue_list.json under
# ../data and ../UEPackage, see https://github.com/<ESARBench-repo> for setup),
# and this repo's cache/relevance_h95test5_v2 (or your own belief-map cache)
# present under ../cache relative to this script.
cd "$(dirname "$0")/source"

export HL_FIELD_WINDOW_M=250 HL_FIELD_LEAN_MODE=cluster HL_FIELD_LEAN_DIST_SCALE_M=120 \
  HL_FIELD_LEAN_BRAKE=true HL_FIELD_GIVEUP=true HL_FIELD_STUCK_BOOST=true HL_FIELD_UNWEDGE=true \
  HL_FUSION_GAMMA=2.0 HL_FIELD_CAP_LEAN=0.50 HL_FIELD_CAP_TOTAL=0.65 \
  HL_FIELD_CAP_LEAN_STUCK=0.60 HL_FIELD_CAP_TOTAL_STUCK=0.75 \
  HL_FIELD_NEG_EVIDENCE=true HL_FIELD_ARRIVE=true HL_FIELD_ALT_AWARE_OBSTACLE=true

# Edit TASK_IDS to the task_id list you want to evaluate (comma-separated),
# e.g. all task ids in your task_data.json, or a specific subset file:
#   TASK_IDS=$(python3 -c "import json;print(','.join(str(x) for x in json.load(open('path/to/task_ids.json'))['task_ids']))")
TASK_IDS="${1:-}"

python3 main_platform_survey.py --agent apex --fusion-mode field \
  --relevance-dir ../cache/relevance_h95test5_v2 \
  --task-ids "$TASK_IDS" \
  --log-dir ../log_ours \
  --gpus "${2:-0}" --base-port "${3:-41451}"
