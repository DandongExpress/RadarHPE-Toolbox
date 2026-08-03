#!/usr/bin/env python3
"""Run CFAR on a heatmap and visualise the resulting point cloud.

Examples
--------
python examples/demo_cfar_pointcloud.py --save outputs/overview.png

python examples/demo_cfar_pointcloud.py \\
    --hori data/HuPR/single_1/hori/000000000.npy \\
    --vert data/HuPR/single_1/vert/000000000.npy \\
    --mode os --top-k 64 --save outputs/hupr_cfar.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

from radarhpe.basics import (
    RadarConfig,
    heatmap_to_pointcloud,
    load_heatmap,
    plot_hupr_overview,
    synthesize_hupr_frame,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--hori', type=str, default=None, help='Horizontal HuPR .npy frame.')
    p.add_argument('--vert', type=str, default=None, help='Optional vertical HuPR .npy frame.')
    p.add_argument('--mode', choices=('ca', 'os'), default='ca', help='CFAR variant.')
    p.add_argument('--pfa', type=float, default=1e-3, help='Design P_fa for CA-CFAR.')
    p.add_argument('--top-k', type=int, default=64, help='Keep at most this many peaks after NMS.')
    p.add_argument('--save', type=str, default=None, help='Save overview figure to this path.')
    p.add_argument('--show', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = RadarConfig()

    if args.hori is None:
        print('No --hori given; synthesising a demo frame with 3 targets.')
        hori = synthesize_hupr_frame(cfg=cfg)
        vert = None
    else:
        hori = load_heatmap(args.hori)
        vert = load_heatmap(args.vert) if args.vert else None
        print(f'Loaded hori {hori.shape}', end='')
        print(f', vert {vert.shape}' if vert is not None else '')

    cloud, cfar = heatmap_to_pointcloud(
        hori,
        cfg=cfg,
        cfar_mode=args.mode,
        pfa=args.pfa,
        top_k=args.top_k,
    )
    print(f'CFAR ({args.mode}): {int(cfar.detections.sum())} raw cells → '
          f'{len(cloud.xyz)} points after NMS (top_k={args.top_k})')
    if len(cloud.xyz):
        print(f'  range  [{cloud.ranges.min():.2f}, {cloud.ranges.max():.2f}] m')
        print(f'  xyz    mean=({cloud.xyz.mean(0).round(2)})')

    save_path = Path(args.save) if args.save else None
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plot_hupr_overview(
        hori, vert=vert, cloud=cloud, cfg=cfg,
        save_path=save_path,
        show=args.show or save_path is None,
    )
    if save_path is not None and not args.show:
        import matplotlib.pyplot as plt
        plt.close(fig)
        print(f'Saved {save_path}')


if __name__ == '__main__':
    main()
