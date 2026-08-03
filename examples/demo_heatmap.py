#!/usr/bin/env python3
"""Visualise a HuPR-style heatmap (synthetic by default).

Examples
--------
# No dataset required:
python examples/demo_heatmap.py --save outputs/ra_rd.png

# Real HuPR frame:
python examples/demo_heatmap.py --input data/HuPR/single_1/hori/000000000.npy
"""
from __future__ import annotations

import argparse
from pathlib import Path

from radarhpe.basics import (
    RadarConfig,
    load_heatmap,
    plot_ra_heatmap,
    plot_rd_heatmap,
    synthesize_hupr_frame,
    summarize_resolutions,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', type=str, default=None, help='Path to a .npy heatmap (HuPR hori/vert frame).')
    p.add_argument('--save', type=str, default=None, help='If set, save a side-by-side RA|RD figure here.')
    p.add_argument('--show', action='store_true', help='Display the figure interactively.')
    p.add_argument('--reduce', choices=('mean', 'max'), default='max')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = RadarConfig()
    print('RadarConfig resolutions:', ', '.join(summarize_resolutions(cfg)))

    if args.input is None:
        print('No --input given; synthesising a demo HuPR-like frame.')
        heatmap = synthesize_hupr_frame(cfg=cfg)
    else:
        heatmap = load_heatmap(args.input)
        print(f'Loaded {args.input} with shape {heatmap.shape} dtype={heatmap.dtype}')

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    plot_ra_heatmap(heatmap, cfg=cfg, ax=axes[0], reduce=args.reduce, title='Range–Azimuth')
    plot_rd_heatmap(heatmap, cfg=cfg, ax=axes[1], reduce=args.reduce, title='Range–Doppler')
    fig.tight_layout()

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f'Saved {out}')
    if args.show or args.save is None:
        plt.show()
    else:
        plt.close(fig)


if __name__ == '__main__':
    main()
