#!/usr/bin/env python
"""Config-driven evaluation entry point::

    python eval.py --config options/test/pulse_1f_hupr.yml --ckpt checkpoints/pulse_1f_hupr_best.pth

Reports every metric declared in the config's ``metrics`` list (default:
MPJPE + PA-MPJPE) without any post-hoc smoothing, matching the evaluation
protocol described in all three source papers.
"""
import argparse

import torch
from torch.utils.data import DataLoader

import radarhpe
from train import load_config  # reuse the same YAML loader


def evaluate(cfg: dict, ckpt: str, device: str):
    dataset_cfg = cfg['dataset']
    model_cfg = cfg['model']
    metric_names = cfg.get('metrics', ['mpjpe', 'pa_mpjpe'])

    test_set = radarhpe.create_dataset(
        dataset_cfg['name'],
        root=dataset_cfg['root'],
        split=dataset_cfg.get('test_split', 'test'),
    )
    test_loader = DataLoader(test_set, batch_size=cfg.get('eval', {}).get('batch_size', 8))

    model = radarhpe.create_model(model_cfg['name'], pretrained=ckpt, device=device, **model_cfg.get('kwargs', {}))
    model.eval()

    metrics = {name: radarhpe.create_metric(name) for name in metric_names}
    totals = {name: 0.0 for name in metric_names}
    n_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            rad = batch['rad'].to(device)
            gt_pose = batch['pose'].to(device)
            pred_pose = model(rad)

            for name, fn in metrics.items():
                totals[name] += fn(pred_pose, gt_pose).item()
            n_batches += 1

    print('Evaluation results:')
    for name in metric_names:
        avg = totals[name] / max(n_batches, 1)
        print(f'  {name}: {avg:.4f}')


def main():
    parser = argparse.ArgumentParser(description='RadarHPE-Toolbox evaluation entry point.')
    parser.add_argument('--config', required=True, help='path to a YAML config under options/test/')
    parser.add_argument('--ckpt', required=True, help='local checkpoint path or HF Hub repo id')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluate(cfg, args.ckpt, args.device)


if __name__ == '__main__':
    main()
