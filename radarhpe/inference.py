"""Command-line single-sample inference, mirrors IQA-PyTorch's inference_iqa.py.

Usage::

    python -m radarhpe.inference -m pulse_1f -r path/to/frame.npy \\
        --ckpt path/or/hf-hub-id --device cuda

    # list available models
    python -m radarhpe.inference --list
"""
import argparse

import numpy as np
import torch

import radarhpe


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='radarhpe-infer',
        description='RadarHPE-Toolbox single-sample inference CLI.',
    )
    parser.add_argument('-m', '--model', help='registered model name (see --list)')
    parser.add_argument('-r', '--rad', help='path to a .npy RAD / heatmap tensor')
    parser.add_argument('--ckpt', default=None, help='local checkpoint path or HF Hub repo id')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--list', action='store_true', help='list registered models and exit')
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list:
        print('Registered models:')
        for name in radarhpe.list_models():
            print(f'  - {name}')
        return

    if not args.model or not args.rad:
        parser.error('--model and --rad are required unless --list is given.')

    model = radarhpe.create_model(args.model, pretrained=args.ckpt, device=args.device)
    rad = torch.from_numpy(np.load(args.rad)).float().unsqueeze(0).to(args.device)

    joints = model.predict(rad)
    joints_np = joints.squeeze(0).cpu().numpy()

    print(f'Predicted {joints_np.shape[-2]} joints (mm, pelvis-centred):')
    print(joints_np)


if __name__ == '__main__':
    main()
