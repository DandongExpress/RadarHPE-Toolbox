"""CFAR detectors for mmWave heatmaps.

CFAR (Constant False Alarm Rate) estimates a local noise floor around each
cell under test (CUT) and declares a detection when the CUT exceeds
``threshold_factor * noise``. This is the standard step that turns a dense
range-azimuth (or range-Doppler) heatmap into a sparse point set.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
from scipy.ndimage import uniform_filter

CFARMode = Literal['ca', 'os', 'go', 'so']


@dataclass
class CFARResult:
    """Output of a 2-D CFAR pass."""

    detections: np.ndarray  # bool mask, same shape as input
    threshold: np.ndarray   # float threshold map
    noise: np.ndarray       # estimated noise floor
    peaks: np.ndarray       # (N, 2) int indices of True cells as (row, col)


def _guard_mask(guard: Tuple[int, int], train: Tuple[int, int]) -> np.ndarray:
    """Boolean kernel: True = training cells, False = guard + CUT."""
    gr, gc = guard
    tr, tc = train
    kr, kc = 2 * (gr + tr) + 1, 2 * (gc + tc) + 1
    kernel = np.ones((kr, kc), dtype=bool)
    # Zero out guard + CUT window.
    r0, r1 = tr, tr + 2 * gr + 1
    c0, c1 = tc, tc + 2 * gc + 1
    kernel[r0:r1, c0:c1] = False
    return kernel


def ca_cfar_2d(
    power: np.ndarray,
    guard: Tuple[int, int] = (2, 2),
    train: Tuple[int, int] = (8, 8),
    pfa: float = 1e-4,
    threshold_scale: Optional[float] = None,
) -> CFARResult:
    """Cell-Averaging CFAR on a 2-D power / magnitude map.

    Args:
        power: ``(H, W)`` non-negative array (linear magnitude or power).
        guard: half-size of the guard window ``(row, col)`` excluding CUT.
        train: half-size of the training window beyond the guard cells.
        pfa: design false-alarm probability used when ``threshold_scale`` is
            None. For CA-CFAR with ``N`` training cells,
            ``α = N * (pfa^(-1/N) - 1)``.
        threshold_scale: optional explicit multiplier overriding ``pfa``.
    """
    x = np.asarray(power, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f'ca_cfar_2d expects a 2-D map, got {x.shape}')

    kernel = _guard_mask(guard, train)
    n_train = int(kernel.sum())
    if n_train < 1:
        raise ValueError('Training window is empty — increase `train`.')

    # Sum of training cells via two box filters (full window − guard/CUT).
    gr, gc = guard
    tr, tc = train
    full = 2 * (gr + tr) + 1, 2 * (gc + tc) + 1
    guard_win = 2 * gr + 1, 2 * gc + 1

    sum_full = uniform_filter(x, size=full, mode='nearest') * (full[0] * full[1])
    sum_guard = uniform_filter(x, size=guard_win, mode='nearest') * (guard_win[0] * guard_win[1])
    noise = (sum_full - sum_guard) / n_train
    noise = np.maximum(noise, 1e-12)

    if threshold_scale is None:
        alpha = n_train * (pfa ** (-1.0 / n_train) - 1.0)
    else:
        alpha = float(threshold_scale)

    threshold = alpha * noise
    detections = x > threshold
    peaks = np.argwhere(detections)
    return CFARResult(detections=detections, threshold=threshold, noise=noise, peaks=peaks)


def os_cfar_2d(
    power: np.ndarray,
    guard: Tuple[int, int] = (2, 2),
    train: Tuple[int, int] = (6, 6),
    rank: Optional[int] = None,
    threshold_scale: float = 6.0,
) -> CFARResult:
    """Order-Statistic CFAR — more robust when multiple targets occupy the window.

    Uses the ``rank``-th ordered training cell as the noise estimate
    (default: 75th percentile of the training cells).
    """
    x = np.asarray(power, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f'os_cfar_2d expects a 2-D map, got {x.shape}')

    kernel = _guard_mask(guard, train)
    kr, kc = kernel.shape
    pr, pc = kr // 2, kc // 2
    n_train = int(kernel.sum())
    if rank is None:
        rank = max(1, int(0.75 * n_train))
    rank = int(np.clip(rank, 1, n_train))

    # Pad so every CUT has a full neighbourhood.
    padded = np.pad(x, ((pr, pr), (pc, pc)), mode='edge')
    h, w = x.shape
    noise = np.empty_like(x)
    train_idx = np.argwhere(kernel)

    for i in range(h):
        for j in range(w):
            window = padded[i:i + kr, j:j + kc]
            samples = window[train_idx[:, 0], train_idx[:, 1]]
            noise[i, j] = np.partition(samples, rank - 1)[rank - 1]

    noise = np.maximum(noise, 1e-12)
    threshold = threshold_scale * noise
    detections = x > threshold
    peaks = np.argwhere(detections)
    return CFARResult(detections=detections, threshold=threshold, noise=noise, peaks=peaks)


def cfar_2d(
    power: np.ndarray,
    mode: CFARMode = 'ca',
    **kwargs,
) -> CFARResult:
    """Dispatch to CA-CFAR or OS-CFAR.

    ``mode='go'`` / ``'so'`` are accepted as aliases that currently fall back
    to CA-CFAR (greatest/smallest-of variants can be added later).
    """
    if mode in ('ca', 'go', 'so'):
        return ca_cfar_2d(power, **kwargs)
    if mode == 'os':
        return os_cfar_2d(power, **kwargs)
    raise ValueError(f'Unknown CFAR mode: {mode!r}')


def nms_peaks(peaks: np.ndarray, scores: np.ndarray, radius: int = 2, top_k: Optional[int] = None) -> np.ndarray:
    """Greedy non-maximum suppression on integer ``(row, col)`` detections."""
    if len(peaks) == 0:
        return peaks
    order = np.argsort(scores)[::-1]
    selected = []
    suppressed = np.zeros(len(peaks), dtype=bool)
    for idx in order:
        if suppressed[idx]:
            continue
        selected.append(peaks[idx])
        if top_k is not None and len(selected) >= top_k:
            break
        r0, c0 = peaks[idx]
        dist = np.abs(peaks[:, 0] - r0) + np.abs(peaks[:, 1] - c0)  # L1
        suppressed |= dist <= radius
    return np.asarray(selected, dtype=int)
