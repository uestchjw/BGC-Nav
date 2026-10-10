# BGC-Nav

<p align="center">
  <img src="docs/bgc-nav-violet-light.svg" alt="BGC-Nav" width="500">
</p>

**BGC-Nav: Cross-Altitude Multi-UAV Collaborative Navigation via Belief-Guided Coupling**

<p align="center">
<a href="mailto:haojiuwu2022@ia.ac.cn">Jiuwu Hao</a><sup>1,2</sup>,
Yueyang Wu<sup>1,2</sup>,
Liguo Sun<sup>1</sup>,
Haolin Song<sup>1,2</sup>,
Zao Zhang<sup>1</sup>,
Pin Lv<sup>1</sup>
</p>
<p align="center">
<sup>1</sup>CASIA &nbsp;
<sup>2</sup>School of Artificial Intelligence, University of Chinese Academy of Sciences
</p>

## 🚁 What Is BGC-Nav?

BGC-Nav is a **training-free** framework for embodied, language-guided
search-and-rescue with a pair of UAVs flying at different altitudes. A
high-altitude UAV surveys broadly and builds a **belief map**: a
task-conditioned estimate of where the mission's targets are likely to be,
scored against the search prompt. A low-altitude UAV explores the terrain up
close with its own frontier-exploration policy, and incorporates the
high-altitude belief map through **Belief-Guided Coupling** — a lightweight
bias on its own candidate-action scores, not a rigid waypoint command. The
low-altitude agent keeps full autonomy: the belief map only nudges which
nearby direction looks more promising, is strongest when the agent's own
exploration policy is uncertain, and is overridden the moment the agent
directly detects a target or clue itself. The result is a coupling mechanism
that lets a cheap, broad high-altitude prior improve a capable low-altitude
searcher without any extra training, and without ever overriding what the
searcher sees with its own eyes. BGC-Nav is evaluated on
[ESARBench](#environment-setup), and the method is designed to generalize to
other low-altitude navigation backbones beyond the APEX agent used here.

Code for the "Ours" method (training-free belief-guided coupling between a
high-altitude relevance/belief map and a low-altitude frontier-exploration
agent) evaluated on ESARBench.

This repository currently contains **only the Ours method's own code** —
the agent itself, the offline belief-map builders, the evaluation runner, and
the exact launch script/hyperparameters used for the paper's reported
numbers. Baselines, the task dataset, and the simulation environment
packages are **not** duplicated here; see "Environment setup" below.

## Method

- `baselines_adaption/hl_coord/fused_apex_agent.py` — `FusedApexAgent`: the
  core method. Wraps a frontier-exploration APEX agent and biases its
  per-frontier scores with a pre-built belief/relevance map.
- `baselines_adaption/hl_coord/hl_config.py` — all tunable hyperparameters
  (read from `HL_*` environment variables; defaults match `hl_config.py`,
  the paper's values are set explicitly in `run_ours.sh`).
- `baselines_adaption/hl_coord/aerial_bank.py` — `RelevanceGrid`, the belief
  map data structure shared by all builders and the agent.
- `baselines_adaption/hl_coord/build_relevance.py` — builds a relevance map
  from a nadir aerial frame bank using CLIPSeg (open-vocabulary segmentation
  scored against the task prompt's extracted terrain/target terms).
- `baselines_adaption/hl_coord/build_oblique_relevance.py` — same output
  format, but for oblique (non-nadir) camera banks, unprojecting each
  CLIPSeg score tile through its own per-pixel depth.
- `baselines_adaption/apex/` — the underlying APEX frontier-exploration
  agent that `FusedApexAgent` wraps (voxel map, action selection, VLM scene
  analysis via Qwen-VL, detection->world-point projection).

### Belief map sources used in the paper

Two belief-map sources are evaluated:

- **CLIPSeg** (`cache/relevance/`) — built automatically by
  `build_relevance.py` / `build_oblique_relevance.py` from the aerial frame
  bank + task prompt. Fully reproducible offline, no API calls.
- **Claude-scored** (`cache/relevance_h95test5_v2/`) — produced by having
  Claude read each task's aerial frame bank image(s) and prompt directly and
  score relevance, saved into the same `RelevanceGrid` `.npz` format. There
  is **no standalone script** for this path (it was done as a manual,
  per-task scoring pass, not an automated API pipeline) — the precomputed
  `.npz` files are shipped here so the exact maps used for the paper's
  numbers are reproducible even though the generation process itself isn't
  a single rerunnable script.

Both are drop-in: `main_platform_survey.py --relevance-dir <dir>` just reads
whichever `.npz` cache you point it at.

## Running the paper's best config

```bash
cd BGC-Nav
./run_ours.sh "<comma-separated task ids>" "<gpu ids, e.g. 0,1,2,3>" <base-port>
```

This runs `FusedApexAgent` (fusion-mode `field`) with the exact
hyperparameters used for the paper's main results, reading belief maps from
`cache/relevance_h95test5_v2/` (Claude-scored). See `run_ours.sh` for the
full `HL_*` environment variable block.

Logs are written to `log_ours/task_<id>_hl_log.json`. To compute SR / RS /
CDS from a log directory:

```bash
cd source
python3 -c "import result_analysis; result_analysis.analyze_results()"
```

(`result_analysis.py` expects `log_ours/` and the ESARBench dataset files
described below to be present.)

## Environment setup

This repo does **not** bundle the ESARBench simulation environment, the UE5
map packages, or the task dataset (`task_data.json`, `cue_list.json`). Set
those up exactly as described in ESARBench's own installation guide:

- ESARBench repository / tutorial: **[add the ESARBench repo link here]**

Once ESARBench is installed and its `data/` and UE5 map packages are in
place, point this repo's `source/*.py` at the same relative layout ESARBench
expects (`../data/task_data.json`, `../UEPackage/<map>/...`, `../settings/`)
— i.e. this repo's `source/` and `settings/` directories are meant to sit
inside an ESARBench checkout (or symlinked into one), not stand alone.

Python dependencies: see `requirements.txt`. Additionally requires:
- An AirSim-compatible UE5 build (from ESARBench) and the `airsim` Python
  client.
- A YOLO-World checkpoint for the APEX agent's object detector (not
  included; see `baselines_adaption/apex/apex_agent.py` for the expected
  path/format).
- A `DASHSCOPE_API_KEY` environment variable if you want the APEX agent's
  VLM scene-analysis call (`baselines_adaption/apex/vlm_analyzer.py`,
  Qwen-VL via DashScope) to run — it reads the key from the environment
  only, no key is bundled with this code.

## What's deliberately not in this repo

- Other baselines (SemExp, NavGPT, UniGoal, SPF, VLFM, random/FBE/MLLM
  agents) — `main_platform_linux.py` imports them because it is ESARBench's
  shared multi-baseline runner, but their own code is bundled here only as
  a dependency of that shared runner, not as part of "Ours".
- `baselines_adaption/hl_coord/prompt_decomp.py` (role-decomposition) — used
  by a separate ablation, not the paper's main config.
- The ESARBench dataset, cue lists, and UE5 map packages — get these from
  ESARBench directly.
