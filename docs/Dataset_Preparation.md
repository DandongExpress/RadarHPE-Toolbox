# Dataset Preparation

RadarHPE-Toolbox does not redistribute any of the datasets below — download
them from their original sources and follow each dataset's own license terms.
Once downloaded, place (or symlink) them under `data/` using the layouts
described here; `radarhpe.create_dataset(...)` auto-detects the layout.

```
data/
├── HuPR/
├── XRF55/
├── mmRadPose/
└── MMVR/
```

## HuPR

- **Used by:** `pulse_1f`, `pulse_kf`, `agile_hpe`, `pppr`
- **Radar hardware:** TI IWR1843BOOST
- **Native resolution:** 64×64×8 (range × angle × Doppler)
- **Download:** [HuPR GitHub](https://github.com/robert80203/HuPR-A-Benchmark-for-Human-Pose-Estimation-Using-Millimeter-Wave-Radar) /
  [Hugging Face](https://huggingface.co/datasets/nirajpkini/HuPR)
- **Raw per-frame heatmaps** (as released after HuPR preprocessing):

  ```
  data/HuPR/
  ├── single_1/
  │   ├── hori/000000000.npy   # complex (D, R, A, E) ≈ (16, 64, 64, 8)
  │   ├── vert/000000000.npy
  │   └── visualization/
  ├── single_2/
  └── hrnet_annot_{train,val,test}.json
  ```

  Explore these with `radarhpe.basics` — see
  [MmWave_Fundamentals.md](MmWave_Fundamentals.md) and
  `examples/demo_heatmap.py`.

- **Expected layout for model training** (packed npz, one file per split):

  ```
  data/HuPR/
  ├── train.npz    # rad [N, 64, 64, 16], joints [N, 17, 3]
  ├── val.npz
  └── test.npz
  ```

  Complex RAD cubes are accepted and converted to reflection magnitude on
  load. Axis order must remain `(range, angle, doppler)`. Use
  `radarhpe.basics.to_rad_magnitude` to convert a native HuPR frame into this
  layout.

## XRF55

- **Used by:** `pulse_1f`, `pulse_kf`, `pppr`
- **Radar hardware:** TI IWR6843ISK
- **Provides:** RA (range-angle) and RD (range-Doppler) maps rather than a
  native RAD cube; multi-person capable.
- **Preprocessing required:** reconstruct a unified RAD tensor via the
  weighted-distribution procedure before use:

  ```bash
  python tools/preprocess_xrf55.py --root data/XRF55
  ```

  (Port this script from the original PULSE repository's `tools/` directory —
  see `TODO(integration)` notes in `radarhpe/archs/pulse_arch.py`.)

- **Expected layout after preprocessing** (paired-sequence):

  ```
  data/XRF55/<split>/seq_0000/rad.npy   # [T, R, A, D]
  data/XRF55/<split>/seq_0000/pose.npy  # [T, J, 3]
  ```

## mmRadPose

- **Used by:** `pulse_1f`, `pulse_kf`
- **Radar hardware:** TI AWR2243 family (mocap ground truth), single-person
- **Download:** IEEE DataPort (link TBD — update once confirmed)
- **Expected layout:** paired-sequence, same shape convention as HuPR
  (`[T, 64, 64, 16]` RAD, `[T, 17, 3]` joints in mm, pelvis-centred).

## MMVR

- **Used by:** `pppr`
- **Radar hardware:** TI AWR2243
- **Native resolution:** 256×128 heatmaps
- **Expected layout:** per-frame `.npz`, each with keys:
  - `heatmap`: `[R, A]` or `[R, A, E]`
  - `joints` (optional): `[J, 3]`
  - `doppler` (optional)

  ```
  data/MMVR/
  ├── MMVR/
  ├── HuPR/     # PPPR also evaluates cross-dataset on HuPR / XRF55
  └── XRF55/
  ```

  No dataset yet? All `prepare_pppr.py`-equivalent entry points should
  support a `--synthetic` flag to generate demo heatmaps and exercise the
  full pipeline end-to-end (port from the original PPPR repository).

## Generic paired / per-frame layouts

For any dataset not covered above, `BaseRadarPoseDataset` also accepts:

```
<root>/<split>/seq_0000/rad.npy   # [T, R, A, D]
<root>/<split>/seq_0000/pose.npy  # [T, J, 3]  (mm, pelvis-centred preferred)
```

or

```
<root>/<split>/rad/*.npy   # [R, A, D]
<root>/<split>/pose/*.npy  # [J, 3]  (matching filenames)
```

Pass `layout='paired'` or `layout='per_frame'` explicitly to
`radarhpe.create_dataset(...)` if auto-detection picks the wrong one.
