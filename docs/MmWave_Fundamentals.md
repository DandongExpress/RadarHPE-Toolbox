# mmWave Radar Fundamentals

A beginner-friendly guide to the radar signal path that produces the heatmaps
consumed by RadarHPE models, and the classical CFAR step that turns those
heatmaps into point clouds. Theory follows the TI Radar Academy module
[Fundamentals of mmWave Radar Sensors](https://dev.ti.com/tirex/explore/node?node=A__ABXnTSsO03lbOuTTMnyueg__RADAR-ACADEMY__GwxShWe__LATEST);
data conventions follow the
[HuPR](https://github.com/robert80203/HuPR-A-Benchmark-for-Human-Pose-Estimation-Using-Millimeter-Wave-Radar)
benchmark (TI IWR1843BOOST).

Runnable code lives in `radarhpe.basics` — you do **not** need a trained HPE
model to explore this section.

```python
from radarhpe.basics import (
    RadarConfig, synthesize_hupr_frame, heatmap_to_pointcloud, plot_hupr_overview,
)

cube = synthesize_hupr_frame()          # or load_heatmap('.../hori/000000000.npy')
cloud, cfar = heatmap_to_pointcloud(cube)
plot_hupr_overview(cube, cloud=cloud, show=True)
```

---

## 1. Why mmWave for human sensing?

Millimetre-wave (mmWave) FMCW radars transmit continuously modulated chirps
in the 60 GHz or 76–81 GHz bands. Short wavelengths give:

- centimetre-scale range resolution,
- sensitivity to sub-millimetre displacements (vital signs, micro-Doppler),
- small antennas that fit on a PCB,
- operation that is illumination-invariant and privacy-preserving compared
  with RGB cameras.

HuPR and the models in this toolbox all start from **FFT heatmaps** derived
from such chirps — not from raw ADC samples at training time.

---

## 2. FMCW chirps in one page

A chirp is a tone whose frequency rises linearly from start frequency \(f_c\)
over bandwidth \(B\) during duration \(T_c\). The chirp slope is

\[
S = \frac{B}{T_c}.
\]

The radar mixes the transmitted chirp with the delayed echo. For a target at
distance \(d\), the round-trip delay is \(\tau = 2d / c\), and the mixer
output (IF signal) is a tone

\[
f_0 = S\,\tau = \frac{S\,2d}{c}, \qquad
\Phi_0 = \frac{4\pi d}{\lambda}.
\]

A **Range-FFT** across ADC samples therefore peaks at the bin corresponding
to \(d\). Key closed-form results from the TI academy:

| Quantity | Formula | Typical HuPR / IWR1843 value |
|---|---|---|
| Range resolution | \(d_\mathrm{res} = c / (2B)\) | ≈ 4.8 cm |
| Max unambiguous velocity (2 chirps) | \(v_\mathrm{max} = \lambda / (4T_c)\) | depends on chirp period |
| Velocity resolution (frame of \(N\) chirps) | \(v_\mathrm{res} = \lambda / (2T_f),\ T_f=N T_c\) | centimetres per second |
| Angle of arrival (2 RX, spacing \(l\)) | \(\theta = \arcsin(\lambda\Delta\Phi / (2\pi l))\) | improved via virtual array |

`radarhpe.basics.RadarConfig` and `summarize_resolutions()` expose these
numbers in Python.

---

## 3. From ADC cube to heatmap

A single radar frame is a 3-D (or 4-D with elevation) complex cube of ADC
samples. The classical DSP chain is:

```
ADC samples
   │  Range-FFT   →  separate distances
   │  Doppler-FFT →  separate radial velocities (chirp axis)
   │  Angle-FFT   →  separate azimuth / elevation (antenna axis)
   ▼
Heatmap  (range × angle × Doppler [× elevation])
```

HuPR additionally:

1. removes static clutter (subtract chirp-axis mean),
2. uses two physically rotated IWR1843 boards (`hori` / `vert`) so both
   azimuth and elevation get a usable aperture,
3. keeps only a velocity band around zero (VRDAEMap) because human motion
   is slow relative to automotive radar scales.

The released HuPR `.npy` files are therefore **already FFT heatmaps**,
typically complex arrays of shape `(D, R, A, E) ≈ (16, 64, 64, 8)`.

### Layouts used in this toolbox

| Layout | Shape | Where it appears |
|---|---|---|
| Native HuPR frame | `(D, R, A, E)` complex | `single_*/hori/*.npy`, `vert/*.npy` |
| Paper VRDAEMap | `(2, K, H, W, E)` real/imag | HuPR paper notation |
| RadarHPE RAD | `(R, A, D)` magnitude | `create_dataset('hupr')`, model inputs |
| Collapsed RA / RD | `(R, A)` / `(R, D)` | CFAR + visualisation |

Converters:

```python
from radarhpe.basics import load_heatmap, to_ra_map, to_rd_map, to_rad_magnitude

hm = load_heatmap('data/HuPR/single_1/hori/000000000.npy')
ra  = to_ra_map(hm)            # (64, 64)
rd  = to_rd_map(hm)            # (64, 16)
rad = to_rad_magnitude(hm)     # (64, 64, 16) — model-ready
```

---

## 4. CFAR → point cloud

Deep HPE models can consume the dense heatmap directly. Classical pipelines
(and many fusion / preprocessing papers) first sparsify it with **CFAR**
(Constant False Alarm Rate):

1. estimate a local noise floor around each cell under test (CUT),
2. declare a detection if \(\mathrm{CUT} > \alpha \cdot \mathrm{noise}\),
3. map surviving `(range_bin, angle_bin)` indices to metres / radians,
4. convert polar \((r, \theta, \phi)\) to Cartesian \((x, y, z)\).

This toolbox ships:

- **CA-CFAR** (`ca_cfar_2d`) — cell averaging; fast, educational default,
- **OS-CFAR** (`os_cfar_2d`) — order statistic; more robust with multiple
  nearby targets,
- `heatmap_to_pointcloud(...)` — RA CFAR + NMS + polar→Cartesian,
- optional Doppler attachment (argmax over the Doppler axis at each peak).

```python
from radarhpe.basics import heatmap_to_pointcloud, RadarConfig

cloud, cfar = heatmap_to_pointcloud(hm, cfg=RadarConfig(), cfar_mode='ca', top_k=64)
print(cloud.xyz.shape)        # (N, 3) metres
print(cloud.dopplers[:5])     # m/s when Doppler is available
```

Coordinate convention (radar at the origin):

- \(+x\) forward (range),
- \(+y\) left (azimuth),
- \(+z\) up (elevation; 0 when collapsed).

---

## 5. Visualisation checklist

```bash
# Synthetic demo (no dataset download required)
python examples/demo_heatmap.py --save outputs/ra_rd.png
python examples/demo_cfar_pointcloud.py --save outputs/overview.png

# Real HuPR frame
python examples/demo_heatmap.py --input data/HuPR/single_1/hori/000000000.npy
python examples/demo_cfar_pointcloud.py \
    --hori data/HuPR/single_1/hori/000000000.npy \
    --vert data/HuPR/single_1/vert/000000000.npy \
    --save outputs/hupr_frame0.png
```

Suggested reading order for newcomers:

1. Plot RA / RD heatmaps and confirm a person appears as a bright blob.
2. Sweep CFAR `pfa` / `threshold_scale` and watch the point count change.
3. Feed the same frame through `to_rad_magnitude` into a RadarHPE model
   (`docs/Tutorials.md`) and compare heatmap → pose.

---

## 6. Where this connects to HPE models

```
HuPR .npy heatmap
        │
        ├─ radarhpe.basics.*     →  RA/RD plots, CFAR point clouds  (this doc)
        │
        └─ to_rad_magnitude()
                │
                ▼
        radarhpe.create_dataset / create_model
                │
                ▼
        3D joints (MPJPE / PA-MPJPE / MPJVE)
```

Physics-guided models in the zoo (PULSE Doppler prompting, Agile SSP/MCP
preprocessing, PPPR Gaussian primitives) all assume you already understand
the heatmap axes above. Spending an hour with `radarhpe.basics` usually
makes those papers much easier to read.

---

## References

1. TI Radar Academy — *Fundamentals of mmWave Radar Sensors*
   ([TIREX node](https://dev.ti.com/tirex/explore/node?node=A__ABXnTSsO03lbOuTTMnyueg__RADAR-ACADEMY__GwxShWe__LATEST)).
2. Lee et al., *HuPR: A Benchmark for Human Pose Estimation Using Millimeter
   Wave Radar*, WACV 2023
   ([paper](https://arxiv.org/abs/2210.12564),
   [code](https://github.com/robert80203/HuPR-A-Benchmark-for-Human-Pose-Estimation-Using-Millimeter-Wave-Radar)).
3. Richards, *Fundamentals of Radar Signal Processing* — CFAR chapter
   (CA-CFAR / OS-CFAR derivations).
