# Changelog

All notable changes to RadarHPE-Toolbox are documented in this file.

## [Unreleased]

### Added

- Initial project scaffold: registry-based architecture (`MODEL_REGISTRY`,
  `DATASET_REGISTRY`, `METRIC_REGISTRY`, `BACKBONE_REGISTRY`), unified
  dataset loaders for HuPR / XRF55 / mmRadPose / MMVR, pose metrics
  (MPJPE, PA-MPJPE, MPJVE), and CLI entry points (`train.py`, `eval.py`,
  `radarhpe.inference`).
- Structural (not-yet-runnable) arch wrappers for `pulse_1f` / `pulse_kf`
  (PULSE, ICML 2026), `agile_hpe` (Agile-MmWave-Hpe, ICME 2026), and `pppr`
  (PPPR, ACM IMWUT/UbiComp 2026), each with `TODO(integration)` markers
  pointing at the exact files to port from the original single-paper
  repositories.
- Example training/eval configs under `options/` for all three models.
- Docs: `ModelCard.md`, `Dataset_Preparation.md`, `Instruction.md` (this
  contribution guide).

### Planned

- Port the real forward passes for `pulse_1f` / `pulse_kf`, `agile_hpe`, and
  `pppr` from their original repositories (see `docs/ModelCard.md`
  "Integration status").
- Upload pretrained checkpoints to the Hugging Face Hub and fill in
  `radarhpe/default_model_configs.yml`.
- Verify and implement the `AKV` metric definition (see
  `radarhpe/metrics/pose_metrics.py`).
- Publish to PyPI as `radarhpe`.
