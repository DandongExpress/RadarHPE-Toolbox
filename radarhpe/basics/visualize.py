"""Matplotlib visualisers for heatmaps and radar point clouds."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from radarhpe.basics.io import power_db, to_ra_map, to_rd_map
from radarhpe.basics.physics import RadarConfig
from radarhpe.basics.pointcloud import PointCloud

PathLike = Union[str, Path]


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            'matplotlib is required for radarhpe.basics.visualize — '
            'install it with `pip install matplotlib`.'
        ) from exc
    return plt


def plot_ra_heatmap(
    heatmap: np.ndarray,
    cfg: Optional[RadarConfig] = None,
    ax=None,
    title: str = 'Range–Azimuth heatmap',
    db: bool = True,
    reduce: str = 'max',
    cmap: str = 'viridis',
    detections: Optional[np.ndarray] = None,
    save_path: Optional[PathLike] = None,
    show: bool = False,
):
    """Plot a range-azimuth map; optionally overlay CFAR peak indices."""
    plt = _require_matplotlib()
    cfg = cfg or RadarConfig()
    ra = to_ra_map(heatmap, reduce=reduce)
    img = power_db(ra) if db else ra

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    extent = [
        np.rad2deg(cfg.azimuth_axis()[0]),
        np.rad2deg(cfg.azimuth_axis()[-1]),
        cfg.range_axis()[0],
        cfg.range_axis()[-1],
    ]
    im = ax.imshow(img, origin='lower', aspect='auto', extent=extent, cmap=cmap)
    if detections is not None and len(detections):
        az = np.rad2deg(cfg.azimuth_axis()[np.clip(detections[:, 1], 0, cfg.num_azimuth_bins - 1)])
        rng = cfg.range_axis()[np.clip(detections[:, 0], 0, cfg.num_range_bins - 1)]
        ax.scatter(az, rng, s=18, c='cyan', marker='x', linewidths=1.2, label='CFAR')
        ax.legend(loc='upper right', fontsize=8)
    ax.set_xlabel('Azimuth (deg)')
    ax.set_ylabel('Range (m)')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='dB' if db else 'mag')
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return fig, ax


def plot_rd_heatmap(
    heatmap: np.ndarray,
    cfg: Optional[RadarConfig] = None,
    ax=None,
    title: str = 'Range–Doppler heatmap',
    db: bool = True,
    reduce: str = 'max',
    cmap: str = 'magma',
    save_path: Optional[PathLike] = None,
    show: bool = False,
):
    """Plot a range-Doppler map."""
    plt = _require_matplotlib()
    cfg = cfg or RadarConfig()
    rd = to_rd_map(heatmap, reduce=reduce)
    img = power_db(rd) if db else rd

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    extent = [
        cfg.doppler_axis()[0],
        cfg.doppler_axis()[-1],
        cfg.range_axis()[0],
        cfg.range_axis()[-1],
    ]
    im = ax.imshow(img, origin='lower', aspect='auto', extent=extent, cmap=cmap)
    ax.set_xlabel('Radial velocity (m/s)')
    ax.set_ylabel('Range (m)')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='dB' if db else 'mag')
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return fig, ax


def plot_pointcloud(
    cloud: PointCloud,
    ax=None,
    title: str = 'Radar point cloud',
    color_by: str = 'intensity',
    elev: float = 20.0,
    azim: float = -60.0,
    save_path: Optional[PathLike] = None,
    show: bool = False,
):
    """3-D scatter of a :class:`PointCloud`."""
    plt = _require_matplotlib()
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    created = ax is None
    if created:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure

    if len(cloud.xyz) == 0:
        ax.set_title(title + ' (empty)')
        return fig, ax

    if color_by == 'doppler' and np.isfinite(cloud.dopplers).any():
        c = cloud.dopplers
        cbar_label = 'Doppler (m/s)'
    else:
        c = cloud.intensities
        cbar_label = 'Intensity'

    sc = ax.scatter(
        cloud.xyz[:, 0], cloud.xyz[:, 1], cloud.xyz[:, 2],
        c=c, s=12, cmap='coolwarm', depthshade=True,
    )
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.08, label=cbar_label)
    ax.set_xlabel('X forward (m)')
    ax.set_ylabel('Y left (m)')
    ax.set_zlabel('Z up (m)')
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return fig, ax


def plot_hupr_overview(
    hori: np.ndarray,
    vert: Optional[np.ndarray] = None,
    cloud: Optional[PointCloud] = None,
    cfg: Optional[RadarConfig] = None,
    save_path: Optional[PathLike] = None,
    show: bool = False,
):
    """One-page overview: hori RA, vert RA (optional), RD, and point cloud."""
    plt = _require_matplotlib()
    cfg = cfg or RadarConfig()
    ncols = 3 if cloud is not None else 2
    fig = plt.figure(figsize=(4.2 * ncols, 4))

    ax0 = fig.add_subplot(1, ncols, 1)
    plot_ra_heatmap(hori, cfg=cfg, ax=ax0, title='Horizontal RA',
                    detections=None if cloud is None else cloud.indices)

    ax1 = fig.add_subplot(1, ncols, 2)
    if vert is not None:
        plot_ra_heatmap(vert, cfg=cfg, ax=ax1, title='Vertical RA')
    else:
        plot_rd_heatmap(hori, cfg=cfg, ax=ax1, title='Range–Doppler')

    if cloud is not None:
        ax2 = fig.add_subplot(1, ncols, 3, projection='3d')
        plot_pointcloud(cloud, ax=ax2, title='CFAR point cloud')

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return fig
