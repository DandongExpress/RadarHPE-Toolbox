"""FMCW mmWave radar physics helpers.

Equations follow the TI Radar Academy module
`Fundamentals of mmWave Radar Sensors` and the HuPR IWR1843BOOST setup.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# Speed of light (m/s).
C = 2.99792458e8


@dataclass(frozen=True)
class RadarConfig:
    """Chirp / array configuration used to map FFT bins to physical units.

    Defaults approximate the HuPR IWR1843BOOST setup described in the paper:
    max range ≈ 11 m, range resolution ≈ 4.8 cm, azimuth FOV ≈ ±60°.
    """

    # Carrier / chirp
    start_freq_hz: float = 77e9
    bandwidth_hz: float = 3.125e9  # → d_res ≈ c / (2B) ≈ 4.8 cm
    chirp_duration_s: float = 40e-6
    chirp_period_s: float = 100e-6  # Tc between consecutive chirps
    num_chirps: int = 64

    # Sampling / FFT sizes (after HuPR-style cropping / padding)
    num_range_bins: int = 64
    num_doppler_bins: int = 16
    num_azimuth_bins: int = 64
    num_elevation_bins: int = 8

    # Antenna / FOV (wavelength / spacing default to λ and λ/2 in __post_init__)
    wavelength_m: Optional[float] = None
    rx_spacing_m: Optional[float] = None
    azimuth_fov_deg: float = 120.0
    elevation_fov_deg: float = 30.0
    max_range_m: float = 11.0

    def __post_init__(self):
        if self.wavelength_m is None:
            object.__setattr__(self, 'wavelength_m', C / self.start_freq_hz)
        if self.rx_spacing_m is None:
            object.__setattr__(self, 'rx_spacing_m', self.wavelength_m / 2.0)

    @property
    def slope_hz_per_s(self) -> float:
        """Chirp slope S = B / Tc (Hz/s)."""
        return self.bandwidth_hz / self.chirp_duration_s

    @property
    def range_resolution_m(self) -> float:
        """d_res = c / (2B)."""
        return C / (2.0 * self.bandwidth_hz)

    @property
    def max_unambiguous_velocity_mps(self) -> float:
        """v_max = λ / (4 Tc)."""
        return self.wavelength_m / (4.0 * self.chirp_period_s)

    @property
    def velocity_resolution_mps(self) -> float:
        """v_res = λ / (2 Tf), Tf = N * Tc."""
        frame_time = self.num_chirps * self.chirp_period_s
        return self.wavelength_m / (2.0 * frame_time)

    def range_axis(self) -> np.ndarray:
        """Physical range (m) for each range bin (near → far)."""
        dr = self.max_range_m / self.num_range_bins
        return (np.arange(self.num_range_bins, dtype=np.float64) + 0.5) * dr

    def doppler_axis(self) -> np.ndarray:
        """Signed radial velocity (m/s) for each Doppler bin (neg → pos)."""
        vmax = self.max_unambiguous_velocity_mps
        n = self.num_doppler_bins
        return np.linspace(-vmax, vmax, n, endpoint=False, dtype=np.float64)

    def azimuth_axis(self) -> np.ndarray:
        """Azimuth angle (radians) for each angle bin (left → right)."""
        half = np.deg2rad(self.azimuth_fov_deg / 2.0)
        return np.linspace(-half, half, self.num_azimuth_bins, dtype=np.float64)

    def elevation_axis(self) -> np.ndarray:
        """Elevation angle (radians) for each elevation bin (down → up)."""
        half = np.deg2rad(self.elevation_fov_deg / 2.0)
        return np.linspace(-half, half, self.num_elevation_bins, dtype=np.float64)


def if_frequency_for_range(range_m: float, slope_hz_per_s: float) -> float:
    """IF tone frequency f0 = S * 2d / c for a static target at distance d."""
    return slope_hz_per_s * 2.0 * range_m / C


def phase_for_range(range_m: float, wavelength_m: float) -> float:
    """IF initial phase Φ0 = 4π d / λ."""
    return 4.0 * np.pi * range_m / wavelength_m


def velocity_from_phase_diff(delta_phi: float, wavelength_m: float, chirp_period_s: float) -> float:
    """v = λ ΔΦ / (4π Tc) from two consecutive chirps."""
    return wavelength_m * delta_phi / (4.0 * np.pi * chirp_period_s)


def aoa_from_phase_diff(delta_phi: float, wavelength_m: float, spacing_m: float) -> float:
    """θ = arcsin(λ ΔΦ / (2π l)) for a two-antenna array."""
    arg = np.clip(wavelength_m * delta_phi / (2.0 * np.pi * spacing_m), -1.0, 1.0)
    return float(np.arcsin(arg))


def summarize_resolutions(cfg: Optional[RadarConfig] = None) -> Tuple[str, ...]:
    """Human-readable resolution summary for tutorials / logging."""
    cfg = cfg or RadarConfig()
    return (
        f'range_res={cfg.range_resolution_m * 100:.2f} cm',
        f'v_max=±{cfg.max_unambiguous_velocity_mps:.2f} m/s',
        f'v_res={cfg.velocity_resolution_mps:.3f} m/s',
        f'azimuth_fov=±{cfg.azimuth_fov_deg / 2:.0f}°',
        f'elevation_fov=±{cfg.elevation_fov_deg / 2:.0f}°',
    )
