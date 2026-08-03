# RadarHPE-Toolbox

**A unified, physics-guided toolbox for mmWave radar Human Pose Estimation (HPE).**

RadarHPE-Toolbox brings together three previously separate, single-paper
research codebases into one library with a consistent API, so that
beginners can learn mmWave heatmap / point-cloud basics, run a pretrained
model in a few lines of Python, and researchers can benchmark new ideas
against multiple physics-guided baselines without reimplementing data
loading or metrics from scratch. The project's structure and developer
experience are directly modeled on
[IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D1.13-ee4c2c)](https://pytorch.org/)
[![CI](https://github.com/DandongExpress/RadarHPE-Toolbox/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)

## Why this exists

mmWave radar enables privacy-preserving, illumination-robust human pose
estimation, but the research code that accompanies each new paper is
usually a standalone repository with its own data format, training loop,
and evaluation script — making it hard to compare methods or build on top
of more than one at a time. RadarHPE-Toolbox unifies:

- **[PULSE](https://github.com/DandongExpress/Doppler-Prompting-for-Stable-mmWave-based-Human-Pose-Estimation)** — Doppler Prompting for Stable mmWave-based Human Pose Estimation (ICML 2026)
- **[Agile-MmWave-Hpe](https://github.com/DandongExpress/Agile-MmWave-Hpe)** — physics-guided, deterministic SSP/MCP/HMSF preprocessing for edge deployment (ICME 2026)
- **[PPPR](https://github.com/DandongExpress/PPPR)** — Person Parametric Physics-informed Representation, modeling each joint as a Gaussian primitive (ACM IMWUT/UbiComp 2026)

under one registry-based package, `radarhpe`, with shared dataset loaders
(HuPR, XRF55, mmRadPose, MMVR), shared metrics (MPJPE, PA-MPJPE, MPJVE),
and a beginner track for mmWave fundamentals (`radarhpe.basics` +
[docs/MmWave_Fundamentals.md](docs/MmWave_Fundamentals.md)).

> **Integration status:** `radarhpe.basics` (heatmap I/O, CFAR, visualisation),
> the registry, CLI, dataset loaders, and metrics below are fully functional
> today. The three architectures are wired into the registry with their
> documented public APIs, but their tensor-level forward passes are still
> `TODO(integration)` stubs pending a verified port from each original
> repository — see [docs/ModelCard.md](docs/ModelCard.md#integration-status)
> for the per-model checklist and
> [docs/history_changelog.md](docs/history_changelog.md) for what's planned next.

## Installation

**Option 1 — editable install from source (recommended while the project is pre-release):**

```bash
git clone https://github.com/DandongExpress/RadarHPE-Toolbox.git
cd RadarHPE-Toolbox
pip install -e .
```

**Option 2 — plain requirements install (no packaging):**

```bash
git clone https://github.com/DandongExpress/RadarHPE-Toolbox.git
cd RadarHPE-Toolbox
pip install -r requirements.txt
```

**Option 3 — pip install once published to PyPI:**

```bash
pip install radarhpe
```

Requirements: Python ≥ 3.8, PyTorch ≥ 1.13. See `pyproject.toml` for the
full dependency list, and the `dev` / `onnx` extras for contributor tooling
and edge-export support respectively.

## Quick Start 
 
### A. mmWave fundamentals (heatmap → CFAR → point cloud)

New to radar? Start here — no trained model required. Theory lives in
[docs/MmWave_Fundamentals.md](docs/MmWave_Fundamentals.md); code lives in
`radarhpe.basics`:

```python
from radarhpe.basics import (
    synthesize_hupr_frame, heatmap_to_pointcloud, plot_hupr_overview,
)

cube = synthesize_hupr_frame()          # or load a HuPR .npy heatmap
cloud, cfar = heatmap_to_pointcloud(cube)
plot_hupr_overview(cube, cloud=cloud, show=True)
```

```bash
python examples/demo_heatmap.py --save outputs/ra_rd.png
python examples/demo_cfar_pointcloud.py --save outputs/overview.png
```

### B. Models / datasets / metrics

```python
import radarhpe

# See everything the toolbox currently ships.
print(radarhpe.list_models())    # ['agile_hpe', 'pppr', 'pulse_1f', 'pulse_kf']
print(radarhpe.list_datasets())  # ['hupr', 'mmradpose', 'mmvr', 'xrf55']
print(radarhpe.list_metrics())   # ['akv', 'mpjpe', 'mpjve', 'pa_mpjpe']

# Instantiate a model (pretrained checkpoints coming soon — see ModelCard).
model = radarhpe.create_model('pulse_1f', device='cpu')
```
 
Or from the command line: 
 
```bash
python -m radarhpe.inference -m pulse_1f -r frame.npy --ckpt path/or/hf-hub-id
```
 
See [docs/Tutorials.md](docs/Tutorials.md) for a full beginner-to-advanced 
walkthrough (radar basics, loading datasets, training, evaluation, and writing
a 5-minute plugin for your own model). 

## Model Zoo

| Model | Paper | Datasets | Key metrics |
|---|---|---|---|
| `pulse_1f` / `pulse_kf` | [Doppler Prompting for Stable mmWave-based HPE](https://arxiv.org/abs/2605.13233), ICML 2026 | HuPR, XRF55, mmRadPose | MPJPE, PA-MPJPE, MPJVE, AKV |
| `agile_hpe` | [Why Learn What Physics Already Knows?](https://arxiv.org/abs/2603.08236), ICME 2026 | HuPR | MAJPE, PA-MAJPE |
| `pppr` | [Person Parametric Physics-informed Representation](https://arxiv.org/abs/2512.23054), ACM IMWUT/UbiComp 2026 | MMVR, HuPR, XRF55 | MPJPE, PA-MPJPE |

Full details (parameter counts, runtime profiles, supported backbones, and
checkpoint links as they become available) are in
[docs/ModelCard.md](docs/ModelCard.md).

## Dataset Support

`radarhpe.create_dataset(...)` provides a single interface across:

- **HuPR** — TI IWR1843BOOST, single-person
- **XRF55** — TI IWR6843ISK, multi-person capable
- **mmRadPose** — mocap ground truth, single-person
- **MMVR** — TI AWR2243, used by PPPR

See [docs/Dataset_Preparation.md](docs/Dataset_Preparation.md) for download
links, licenses, and the exact on-disk layout each dataset expects.

## Repository Structure 
 
```
RadarHPE-Toolbox/
├── radarhpe/                     # the installable package
│   ├── basics/                   # mmWave fundamentals (I/O, CFAR, viz)
│   │   ├── physics.py            # FMCW range / Doppler / AoA equations
│   │   ├── io.py                 # HuPR heatmap loaders & RA/RD/RAD converters
│   │   ├── cfar.py               # CA-CFAR / OS-CFAR
│   │   ├── pointcloud.py         # heatmap → Cartesian points
│   │   ├── visualize.py          # matplotlib helpers
│   │   └── synthetic.py          # demo heatmaps (no dataset download)
│   ├── archs/                    # model architectures (registry-based)
│   ├── data/                     # dataset loaders (registry-based)
│   ├── metrics/                  # MPJPE / PA-MPJPE / MPJVE / AKV
│   ├── utils/                    # registry.py, download_util.py
│   └── inference.py
├── examples/                     # beginner scripts for heatmaps / CFAR
├── options/
│   ├── train/
│   └── test/
├── docs/
│   ├── MmWave_Fundamentals.md    # TI academy + HuPR primer
│   ├── ModelCard.md
│   ├── Dataset_Preparation.md
│   ├── Instruction.md
│   ├── Tutorials.md
│   └── history_changelog.md
├── tests/
├── train.py
├── eval.py
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── NOTICE
```

## Contributing

Contributions are very welcome, whether that's porting one of the three
original architectures' real forward pass, adding a new dataset, fixing a
bug, or adding a new baseline model entirely. See
[docs/Instruction.md](docs/Instruction.md) for the plugin-authoring guide
and pull request checklist.

## Updates

See [docs/history_changelog.md](docs/history_changelog.md) for the full,
dated changelog.

## Citation

If you use this toolbox, please cite the paper(s) corresponding to the
model(s) you use:

```bibtex
@article{zheng2026doppler,
  title={Doppler Prompting for Stable mmWave-based Human Pose Estimation},
  author={Zheng, Shuntian and Li, Jiaqi and Lu, Xiaoman and He, Shuai and Guan, Yu},
  journal={arXiv preprint arXiv:2605.13233},
  year={2026}
}

@article{zheng2026learn,
  title={Why Learn What Physics Already Knows? Realizing Agile mmWave-based Human Pose Estimation via Physics-Guided Preprocessing},
  author={Zheng, Shuntian and Li, Jiaqi and Ni, Minzhe and Lu, Xiaoman and Guan, Yu},
  journal={arXiv preprint arXiv:2603.08236},
  year={2026}
}

@article{zheng2025person,
  title={Person Parametric Physics-informed Representation for mmWave-based Human Pose Estimation},
  author={Zheng, Shuntian and Li, Jiaqi and Wang, Guangming and Ni, Minzhe and Palit, Arnad and Montana, Giovanni and Guan, Yu},
  journal={arXiv preprint arXiv:2512.23054},
  year={2025}
}
```

A `CITATION.cff` file is also provided for GitHub's "Cite this repository"
feature.

## License

RadarHPE-Toolbox is released under the [MIT License](LICENSE). It vendors
(or, per the integration roadmap, will vendor) code originally released
under the Apache License 2.0 and MIT License by the same authors — see
[NOTICE](NOTICE) for full attribution and per-file license terms.
