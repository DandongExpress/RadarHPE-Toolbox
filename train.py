#!/usr/bin/env python
"""Config-driven training entry point, callable the same way as train.py in
each of the three original single-paper repos::

    python train.py --config options/train/pulse_1f_hupr.yml

The YAML schema unifies the three repos' training configs (dataset, model,
optimizer, schedule) under the shared ``radarhpe`` registries so the same
script works for PULSE / Agile-HPE / PPPR configs alike. See
options/train/*.yml for worked examples and docs/Instruction.md for the full
schema reference.
"""
import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

import radarhpe


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def build_from_config(cfg: dict):
    dataset_cfg = cfg['dataset']
    model_cfg = cfg['model']

    train_set = radarhpe.create_dataset(
        dataset_cfg['name'],
        root=dataset_cfg['root'],
        split=dataset_cfg.get('train_split', 'train'),
    )
    val_set = radarhpe.create_dataset(
        dataset_cfg['name'],
        root=dataset_cfg['root'],
        split=dataset_cfg.get('val_split', 'val'),
    )

    device = cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model = radarhpe.create_model(model_cfg['name'], device=device, **model_cfg.get('kwargs', {}))

    return model, train_set, val_set, device


def train(cfg: dict):
    model, train_set, val_set, device = build_from_config(cfg)
    train_cfg = cfg.get('train', {})

    train_loader = DataLoader(
        train_set,
        batch_size=train_cfg.get('batch_size', 8),
        shuffle=True,
        num_workers=train_cfg.get('num_workers', 4),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.get('lr', 1e-4),
        weight_decay=train_cfg.get('weight_decay', 0.01),
    )
    mpjpe = radarhpe.create_metric('mpjpe')

    epochs = train_cfg.get('epochs', 100)
    ckpt_dir = Path(train_cfg.get('checkpoint_dir', 'checkpoints'))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            rad = batch['rad'].to(device)
            gt_pose = batch['pose'].to(device)

            optimizer.zero_grad()
            pred_pose = model(rad)
            loss = mpjpe(pred_pose, gt_pose)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / max(len(train_loader), 1)
        print(f'[epoch {epoch + 1}/{epochs}] train MPJPE loss: {avg_loss:.4f}')

        if (epoch + 1) % train_cfg.get('save_every', 10) == 0:
            ckpt_path = ckpt_dir / f"{cfg.get('name', 'model')}_epoch{epoch + 1}.pth"
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch + 1}, ckpt_path)
            print(f'  saved checkpoint -> {ckpt_path}')


def main():
    parser = argparse.ArgumentParser(description='RadarHPE-Toolbox training entry point.')
    parser.add_argument('--config', required=True, help='path to a YAML config under options/train/')
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)


if __name__ == '__main__':
    main()
