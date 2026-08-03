"""I/O helpers for HuPR-style mmWave heatmaps.

HuPR stores per-frame complex FFT cubes as ``.npy`` under::

    data/HuPR/single_<id>/hori/000000000.npy
    data/HuPR/single_<id>/vert/000000000.npy

Native HuPR shape after preprocessing is ``(D, R, A, E)`` complex, typically
``(16, 64, 64, 8)``. RadarHPE model loaders instead expect magnitude RAD cubes
``(R, A, D)`` — see :func:`to_rad_magnitude`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, Path]


def load_heatmap(path: PathLike) -> np.ndarray:
    """Load a single HuPR (or compatible) heatmap ``.npy`` file.

    Returns the array as stored on disk (complex or real). Does not change
    axis order — use :func:`canonicalize_hupr` / :func:`to_rad_magnitude`.
    """
    arr = np.load(path, allow_pickle=False)
    if arr.ndim < 2:
        raise ValueError(f'Expected ≥2-D heatmap, got shape {arr.shape} from {path}')
    return arr


def load_hupr_frame(hori_path: PathLike, vert_path: Optional[PathLike] = None) -> Dict[str, np.ndarray]:
    """Load one HuPR frame (horizontal ± optional vertical radar).

    Returns a dict with keys ``hori`` and optionally ``vert``.
    """
    out = {'hori': load_heatmap(hori_path)}
    if vert_path is not None:
        out['vert'] = load_heatmap(vert_path)
    return out


def list_hupr_frames(sequence_dir: PathLike) -> List[Tuple[Path, Optional[Path]]]:
    """Pair ``hori/*.npy`` with matching ``vert/*.npy`` under a HuPR sequence.

    ``sequence_dir`` should be e.g. ``data/HuPR/single_1``.
    """
    root = Path(sequence_dir)
    hori_dir = root / 'hori'
    vert_dir = root / 'vert'
    if not hori_dir.is_dir():
        raise FileNotFoundError(f'Missing hori/ under {root}')

    frames: List[Tuple[Path, Optional[Path]]] = []
    for hori in sorted(hori_dir.glob('*.npy')):
        vert = vert_dir / hori.name if vert_dir.is_dir() else None
        if vert is not None and not vert.exists():
            vert = None
        frames.append((hori, vert))
    return frames


def magnitude(heatmap: np.ndarray) -> np.ndarray:
    """Complex → magnitude; real arrays are returned as float64/float32 copy."""
    if np.iscomplexobj(heatmap):
        return np.abs(heatmap).astype(np.float32)
    return np.asarray(heatmap, dtype=np.float32)


def canonicalize_hupr(heatmap: np.ndarray) -> np.ndarray:
    """Normalise HuPR-like arrays to ``(D, R, A, E)`` magnitude.

    Accepted inputs:
      * ``(D, R, A, E)`` complex / real  — native HuPR save format
      * ``(R, A, D)`` magnitude RAD      — RadarHPE packed layout
      * ``(R, A)`` range-azimuth map
      * ``(2, D, R, A, E)`` real/imag stacked (paper VRDAEMap layout)
    """
    x = np.asarray(heatmap)
    if x.ndim == 5 and x.shape[0] == 2 and not np.iscomplexobj(x):
        x = x[0] + 1j * x[1]
    mag = magnitude(x)

    if mag.ndim == 4:
        # Assume (D, R, A, E)
        return mag
    if mag.ndim == 3:
        # (R, A, D) → (D, R, A, 1)
        r, a, d = mag.shape
        return np.transpose(mag, (2, 0, 1))[..., None]
    if mag.ndim == 2:
        # (R, A) → (1, R, A, 1)
        return mag[None, ..., None]
    raise ValueError(f'Unsupported heatmap shape {mag.shape}')


def to_rad_magnitude(heatmap: np.ndarray, elevation_reduce: str = 'mean') -> np.ndarray:
    """Convert a HuPR-like cube to RadarHPE ``(R, A, D)`` magnitude RAD.

    Elevation is collapsed with ``mean`` or ``max``.
    """
    cube = canonicalize_hupr(heatmap)  # (D, R, A, E)
    if elevation_reduce == 'mean':
        cube = cube.mean(axis=-1)
    elif elevation_reduce == 'max':
        cube = cube.max(axis=-1)
    else:
        raise ValueError("elevation_reduce must be 'mean' or 'max'")
    # (D, R, A) → (R, A, D)
    return np.transpose(cube, (1, 2, 0)).astype(np.float32)


def to_ra_map(heatmap: np.ndarray, reduce: str = 'mean') -> np.ndarray:
    """Collapse a HuPR-like cube to a 2-D range-azimuth power map ``(R, A)``."""
    cube = canonicalize_hupr(heatmap)  # (D, R, A, E)
    axes = (0, 3)  # Doppler + elevation
    if reduce == 'mean':
        return cube.mean(axis=axes).astype(np.float32)
    if reduce == 'max':
        return cube.max(axis=axes).astype(np.float32)
    raise ValueError("reduce must be 'mean' or 'max'")


def to_rd_map(heatmap: np.ndarray, reduce: str = 'mean') -> np.ndarray:
    """Collapse to a range-Doppler map ``(R, D)``."""
    cube = canonicalize_hupr(heatmap)  # (D, R, A, E)
    if reduce == 'mean':
        ra_collapsed = cube.mean(axis=(2, 3))  # (D, R)
    elif reduce == 'max':
        ra_collapsed = cube.max(axis=(2, 3))
    else:
        raise ValueError("reduce must be 'mean' or 'max'")
    return np.transpose(ra_collapsed, (1, 0)).astype(np.float32)  # (R, D)


def power_db(power: np.ndarray, floor_db: float = -60.0) -> np.ndarray:
    """Convert linear magnitude to normalised dB, clipped at ``floor_db``.

    Uses ``20 * log10(|x|)`` (magnitude convention) and subtracts the peak so
    the colour scale is relative SNR rather than absolute ADC units.
    """
    p = np.maximum(np.asarray(power, dtype=np.float64), 1e-12)
    db = 20.0 * np.log10(p)
    peak = float(np.max(db))
    return np.maximum(db - peak, floor_db).astype(np.float32)


def stack_views(maps: Sequence[np.ndarray], axis: int = 0) -> np.ndarray:
    """Stack multiple 2-D maps (e.g. hori + vert RA) into one array."""
    return np.stack([np.asarray(m) for m in maps], axis=axis)
