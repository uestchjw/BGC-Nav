<p align="center">
  <img src="docs/bgc-nav-violet-light.svg" alt="BGC-Nav" width="500">
</p>

<p align="center">
<b>BGC-Nav: Cross-Altitude Multi-UAV Collaborative Navigation via Belief-Guided Coupling</b>
</p>

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

**BGC-Nav** is a **training-free** coupling mechanism for cross-altitude
multi-UAV search-and-rescue: a high-altitude UAV builds a task-conditioned
**belief map**, and a low-altitude UAV's own frontier-exploration policy
incorporates it as an advisory bias — never a rigid waypoint, and overridden
the moment the low-altitude agent detects a target itself. BGC-Nav is
evaluated on ESARBench.

This repository contains **only the Ours method's own code** — the agent,
the offline belief-map builders, the evaluation runner, and the exact
launch script/hyperparameters used for the paper's reported numbers.

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

## 🚀 Quick Start

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

## 🛠️ Installation

This repo does **not** bundle the ESARBench simulation environment, the UE5
map packages, or the task dataset (`task_data.json`, `cue_list.json`) —
those come from [ESARBench](https://github.com/4amGodvzx/ESAR). Set up
ESARBench first, then drop this repo's `baselines_adaption/`, `source/`,
`settings/`, and `cache/` into (or symlinked onto) that same checkout, so
the relative paths each script expects (`../data/task_data.json`,
`../UEPackage/<map>/...`, `../settings/...`) resolve correctly.

### Linux

1. Create a Python 3.9 conda environment:
   ```bash
   conda create -n esar_env python=3.9
   conda activate esar_env
   ```
2. Install the required Python packages:
   ```bash
   pip install numpy msgpack-rpc-python pandas scipy dashscope opencv-python \
       scikit-image scikit-fmm ultralytics matplotlib transformers pillow
   ```
   Install PyTorch separately for your CUDA version (needed to reproduce the
   VLFM baseline). On headless servers, use `opencv-python-headless` instead
   of `opencv-python`.
3. Clone and install the AirSim-Colosseum client:
   ```bash
   git clone https://github.com/CodexLabsLLC/Colosseum.git
   cd Colosseum/PythonClient
   pip install -e . --no-build-isolation
   ```
4. Download the four UE5 map packages from
   [ESARBench's Hugging Face dataset](https://huggingface.co/datasets/4amGodvzx/ESAR)
   into `UEPackage/`.
5. Set `DASHSCOPE_API_KEY` as an environment variable if you want the APEX
   agent's VLM scene-analysis call (Qwen-VL via DashScope) to run — none of
   the code in this repo reads a hardcoded key.
6. If running on a headless Linux server (no display):
   ```bash
   sudo apt-get install xvfb x11-utils
   ```

7. Place a YOLO-World checkpoint at `source/yolov8x-worldv2.pt` for the APEX
   agent's object detector, or let `ultralytics` fetch it on first run.

See [ESARBench's own README](https://github.com/4amGodvzx/ESAR) for the
Windows installation steps, the required directory layout, map-launch
command-line flags, and the UE/AirSim coordinate conversion — all of that
applies unchanged here.
