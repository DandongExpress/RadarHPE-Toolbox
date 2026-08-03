"""Convert CFAR detections on heatmaps into Cartesian point clouds."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from radarhpe.basics.cfar import CFARResult, cfar_2d, nms_peaks
from radarhpe.basics.io import to_ra_map, to_rd_map
from radarhpe.basics.physics import RadarConfig


@dataclass
class PointCloud:
    """Sparse radar detections in both index and Cartesian coordinates."""

    xyz: np.ndarray          # (N, 3) metres, radar frame: +x forward, +y left, +z up
    ranges: np.ndarray       # (N,) m
    azimuths: np.ndarray     # (N,) rad
    elevations: np.ndarray   # (N,) rad
    dopplers: np.ndarray     # (N,) m/s  (NaN if unknown)
    intensities: np.ndarray  # (N,) linear magnitude / power
    indices: np.ndarray      # (N, 2) or (N, 3) heatmap indices


def _ra_indices_to_polar(
    rows: np.ndarray,
    cols: np.ndarray,
    cfg: RadarConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    range_axis = cfg.range_axis()
    az_axis = cfg.azimuth_axis()
    r = range_axis[np.clip(rows, 0, len(range_axis) - 1)]
    az = az_axis[np.clip(cols, 0, len(az_axis) - 1)]
    return r, az


def polar_to_cartesian(
    ranges: np.ndarray,
    azimuths: np.ndarray,
    elevations: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Radar polar → Cartesian.

    Convention (right-handed, radar at origin):
      x = r cos(el) cos(az)   # forward
      y = r cos(el) sin(az)   # left
      z = r sin(el)           # up
    """
    if elevations is None:
        elevations = np.zeros_like(ranges)
    x = ranges * np.cos(elevations) * np.cos(azimuths)
    y = ranges * np.cos(elevations) * np.sin(azimuths)
    z = ranges * np.sin(elevations)
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def heatmap_to_pointcloud(
    heatmap: np.ndarray,
    cfg: Optional[RadarConfig] = None,
    cfar_mode: str = 'ca',
    guard: Tuple[int, int] = (2, 2),
    train: Tuple[int, int] = (6, 6),
    pfa: float = 1e-3,
    threshold_scale: Optional[float] = None,
    nms_radius: int = 2,
    top_k: Optional[int] = 128,
    reduce: str = 'max',
) -> Tuple[PointCloud, CFARResult]:
    """Run 2-D CFAR on the range-azimuth projection and emit a point cloud.

    Elevation is set to 0 (unknown) when the input has been collapsed; Doppler
    is taken from the strongest Doppler bin at each (R, A) detection when a
    3-D/4-D cube is available.
    """
    cfg = cfg or RadarConfig()
    ra = to_ra_map(heatmap, reduce=reduce)
    kwargs = dict(guard=guard, train=train)
    if threshold_scale is not None:
        kwargs['threshold_scale'] = threshold_scale
    else:
        kwargs['pfa'] = pfa
    result = cfar_2d(ra, mode=cfar_mode, **kwargs)  # type: ignore[arg-type]

    peaks = result.peaks
    if len(peaks) == 0:
        empty = PointCloud(
            xyz=np.zeros((0, 3), np.float32),
            ranges=np.zeros(0, np.float32),
            azimuths=np.zeros(0, np.float32),
            elevations=np.zeros(0, np.float32),
            dopplers=np.zeros(0, np.float32),
            intensities=np.zeros(0, np.float32),
            indices=np.zeros((0, 2), int),
        )
        return empty, result

    scores = ra[peaks[:, 0], peaks[:, 1]]
    peaks = nms_peaks(peaks, scores, radius=nms_radius, top_k=top_k)
    scores = ra[peaks[:, 0], peaks[:, 1]]

    ranges, azimuths = _ra_indices_to_polar(peaks[:, 0], peaks[:, 1], cfg)
    elevations = np.zeros_like(ranges)
    dopplers = np.full(len(ranges), np.nan, dtype=np.float32)

    # If Doppler is present, attach the argmax velocity at each (R, A).
    from radarhpe.basics.io import canonicalize_hupr
    cube = canonicalize_hupr(heatmap)  # (D, R, A, E)
    if cube.shape[0] > 1:
        d_cube = cube.max(axis=-1)  # (D, R, A)
        d_axis = cfg.doppler_axis()
        for i, (rr, aa) in enumerate(peaks):
            rr = int(np.clip(rr, 0, d_cube.shape[1] - 1))
            aa = int(np.clip(aa, 0, d_cube.shape[2] - 1))
            d_idx = int(np.clip(np.argmax(d_cube[:, rr, aa]), 0, len(d_axis) - 1))
            dopplers[i] = d_axis[d_idx]

    xyz = polar_to_cartesian(ranges, azimuths, elevations)
    cloud = PointCloud(
        xyz=xyz,
        ranges=ranges.astype(np.float32),
        azimuths=azimuths.astype(np.float32),
        elevations=elevations.astype(np.float32),
        dopplers=dopplers,
        intensities=scores.astype(np.float32),
        indices=peaks.astype(int),
    )
    return cloud, result


def rd_heatmap_to_points(
    heatmap: np.ndarray,
    cfg: Optional[RadarConfig] = None,
    **cfar_kwargs,
) -> Tuple[np.ndarray, CFARResult]:
    """CFAR on a range-Doppler map; returns ``(N, 2)`` of ``(range_m, velocity_mps)``."""
    cfg = cfg or RadarConfig()
    rd = to_rd_map(heatmap, reduce=cfar_kwargs.pop('reduce', 'max'))
    result = cfar_2d(rd, **cfar_kwargs)
    if len(result.peaks) == 0:
        return np.zeros((0, 2), np.float32), result
    r_axis = cfg.range_axis()
    d_axis = cfg.doppler_axis()
    rows = np.clip(result.peaks[:, 0], 0, len(r_axis) - 1)
    cols = np.clip(result.peaks[:, 1], 0, len(d_axis) - 1)
    pts = np.stack([r_axis[rows], d_axis[cols]], axis=-1).astype(np.float32)
    return pts, result
