# Model Card

Every model registered with `radarhpe.MODEL_REGISTRY` is documented here. Use
`radarhpe.list_models()` to see the live list, and
`radarhpe.create_model(name).model_info()` for the same metadata from Python.

| Name | Paper | Datasets | Metrics | Notes |
|---|---|---|---|---|
| `pulse_1f` | Zheng et al., *Doppler Prompting for Stable mmWave-based Human Pose Estimation*, ICML 2026 ([arXiv:2605.13233](https://arxiv.org/abs/2605.13233)) | HuPR, XRF55, mmRadPose | MPJPE, PA-MPJPE, MPJVE, AKV | Single-frame. 12M params, ~5.1 ms/frame, 75 MFLOPs. Plug-in replacement for the front-end fusion of mmDiff / milliMamba. |
| `pulse_kf` | same as above | HuPR, XRF55, mmRadPose | MPJPE, PA-MPJPE, MPJVE, AKV | Multi-frame (K=9 default), confidence-weighted Doppler aggregation across the window. Best accuracy of the two PULSE variants. |
| `agile_hpe` | Zheng et al., *Why Learn What Physics Already Knows? Realizing Agile mmWave-based Human Pose Estimation via Physics-Guided Preprocessing*, ICME 2026 ([arXiv:2603.08236](https://arxiv.org/abs/2603.08236)) | HuPR | MAJPE, PA-MAJPE | Deterministic (parameter-free) SSP/MCP/HMSF front-end + lightweight PRN regressor. 5.1M trainable params, 55.7–88.9% smaller than baselines, 18.2 FPS / 7.3 MB peak memory on Raspberry Pi 5. Five runtime profiles: `ultra_light`, `light`, `balanced`, `high_precision`, `ultra_precision`. |
| `pppr` | Zheng et al., *Person Parametric Physics-informed Representation for mmWave-based Human Pose Estimation*, ACM IMWUT/UbiComp 2026 ([arXiv:2512.23054](https://arxiv.org/abs/2512.23054)) | MMVR, HuPR, XRF55 | MPJPE, PA-MPJPE | Models each joint as a Gaussian primitive (kinematic + electromagnetic parameters) optimised by the differentiable MHP pipeline. Supports 5 backbones (`RETR`, `HuprModel`, `mmDiff`, `PoseformerV2`, `MLP`) × 5 input representations (`heatmap`, `pc`, `pppr`, `pppr_heatmap`, `pppr_pc`). |

## Integration status

The architectures above are registered and importable today, but the actual
tensor-level forward passes are **structural stubs** — see the
`TODO(integration)` docstring at the top of each `radarhpe/archs/*_arch.py`
file for exactly which file(s) to copy over from the original per-paper
repositories. This keeps the public API (`radarhpe.create_model(...)`,
config schema, checkpoint naming) stable while the real implementations are
migrated in incrementally, one architecture at a time, each verified against
its original repo's reported numbers before being marked "integrated" below.

| Model | Integrated? |
|---|---|
| `pulse_1f` / `pulse_kf` | ☐ not yet — pending copy from PULSE repo |
| `agile_hpe` | ☐ not yet — pending copy from Agile-MmWave-Hpe repo |
| `pppr` | ☐ not yet — pending copy from PPPR repo |

## Adding a new model

See [Instruction.md](Instruction.md) for the plugin-authoring guide (new
`*_arch.py` files are auto-discovered — no other file needs to change).
