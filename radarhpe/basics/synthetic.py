"""Synthetic mmWave heatmaps for demos and unit tests (no real data needed)."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from radarhpe.basics.physics import RadarConfig


def _gaussian_blob(shape: Tuple[int, ...], center: Sequence[float], sigma: Sequence[float], amp: float) -> np.ndarray:
    grids = np.meshgrid(*[np.arange(s, dtype=np.float64) for s in shape], indexing='ij')
    exponent = 0.0
    for g, c, s in zip(grids, center, sigma):
        exponent = exponent + ((g - c) / max(s, 1e-6)) ** 2
    return amp * np.exp(-0.5 * exponent)


def synthesize_rad_cube(
    targets: Optional[List[dict]] = None,
    cfg: Optional[RadarConfig] = None,
    noise_floor: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Synthesize a magnitude RAD cube ``(R, A, D)`` with a few point targets.

    Each target dict may contain:
      ``range_m``, ``azimuth_deg``, ``velocity_mps``, ``rcs`` (amplitude).
    """
    cfg = cfg or RadarConfig()
    rng = rng or np.random.default_rng(0)
    if targets is None:
        targets = [
            dict(range_m=3.5, azimuth_deg=-10.0, velocity_mps=0.4, rcs=1.0),
            dict(range_m=4.2, azimuth_deg=15.0, velocity_mps=-0.2, rcs=0.7),
            dict(range_m=5.0, azimuth_deg=0.0, velocity_mps=0.0, rcs=0.5),
        ]

    r_axis = cfg.range_axis()
    a_axis = np.rad2deg(cfg.azimuth_axis())
    d_axis = cfg.doppler_axis()
    cube = rng.rayleigh(noise_floor, size=(cfg.num_range_bins, cfg.num_azimuth_bins, cfg.num_doppler_bins))

    for t in targets:
        ri = float(np.argmin(np.abs(r_axis - t['range_m'])))
        ai = float(np.argmin(np.abs(a_axis - t['azimuth_deg'])))
        di = float(np.argmin(np.abs(d_axis - t.get('velocity_mps', 0.0))))
        amp = float(t.get('rcs', 1.0))
        blob = _gaussian_blob(
            cube.shape,
            center=(ri, ai, di),
            sigma=(1.2, 1.5, 1.0),
            amp=amp,
        )
        cube = cube + blob
    return cube.astype(np.float32)


def synthesize_hupr_frame(
    targets: Optional[List[dict]] = None,
    cfg: Optional[RadarConfig] = None,
    complex_output: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Synthesize a HuPR-like ``(D, R, A, E)`` cube (complex or magnitude)."""
    cfg = cfg or RadarConfig()
    rad = synthesize_rad_cube(targets=targets, cfg=cfg, rng=rng)  # (R, A, D)
    # (D, R, A) then expand elevation with a mild taper.
    dra = np.transpose(rad, (2, 0, 1))
    e = cfg.num_elevation_bins
    elev_profile = np.exp(-0.5 * ((np.arange(e) - (e - 1) / 2.0) / max(e / 4.0, 1.0)) ** 2)
    cube = dra[..., None] * elev_profile.reshape(1, 1, 1, -1)
    if not complex_output:
        return cube.astype(np.float32)
    rng = rng or np.random.default_rng(1)
    phase = rng.uniform(-np.pi, np.pi, size=cube.shape)
    return (cube * np.exp(1j * phase)).astype(np.complex64)
