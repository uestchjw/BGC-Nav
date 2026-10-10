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

**BGC-Nav** is a **training-free** cross-altitude multi-UAV framework for
search-and-rescue: a high-altitude UAV builds a Global Belief Memory of
task-conditioned priors, and the low-altitude agent folds it into its own
action scoring via **Belief-Guided Coupling** — an advisory bias, never a
rigid dispatched coordinate, that is overridden the moment the agent detects
a target itself. On ESARBench, this lets BGC-Nav consistently outperform
single-agent and existing collaborative baselines, even under unreliable
high-altitude guidance.

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
described above to be present.)

## ✒️ Citation

```bibtex
@inproceedings{hao2026bgcnav,
  title     = {BGC-Nav: Cross-Altitude Multi-UAV Collaborative Navigation via Belief-Guided Coupling},
  author    = {Hao, Jiuwu and Wu, Yueyang and Sun, Liguo and Song, Haolin and Zhang, Zao and Lv, Pin},
}
```

## ✨ Acknowledgements

This work is built and evaluated on [ESARBench](https://github.com/4amGodvzx/ESAR),
a benchmark for agentic UAV embodied search and rescue. We thank its authors
for the simulation environment, task dataset, and baseline implementations.
