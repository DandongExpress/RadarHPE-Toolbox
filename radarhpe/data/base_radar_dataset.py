"""Common dataset interface unifying the on-disk layouts used across the
three source papers (HuPR, XRF55, mmRadPose, MMVR).

See docs/Dataset_Preparation.md for full download instructions and dataset
licenses. This module intentionally contains real, runnable code (unlike the
``archs`` stubs) since the on-disk layouts are fully documented in the
original repositories and do not depend on any proprietary model internals.
"""
from pathlib import Path
from typing import Callable, ClassVar, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class BaseRadarPoseDataset(Dataset):
    """Loads paired (RAD or heatmap, 3D joints) samples from one of three
    on-disk layouts, auto-detected unless ``layout`` is given explicitly:

    1. **Packed npz** (Agile-MmWave-Hpe / PPPR style)::

        <root>/<split>.npz   # keys: 'rad' or 'heatmap' [N, ...], 'joints' [N, J, 3], optional 'doppler'

    2. **Paired-sequence** (PULSE style)::

        <root>/<split>/seq_0000/rad.npy    # [T, R, A, D]
        <root>/<split>/seq_0000/pose.npy   # [T, J, 3]  (mm, pelvis-centred)

    3. **Per-frame** (PULSE style, alternative)::

        <root>/<split>/rad/*.npy    # [R, A, D]
        <root>/<split>/pose/*.npy   # [J, 3]  (matching filenames)

    Args:
        root: dataset root directory (see docs/Dataset_Preparation.md).
        split: one of 'train', 'val', 'test' (dataset-dependent).
        layout: 'auto' (default), 'npz', 'paired', or 'per_frame'.
        transform: optional callable applied to the ``{'rad': ..., 'pose': ...}`` sample dict.
    """

    dataset_name: ClassVar[str] = 'base'

    def __init__(self, root: str, split: str = 'train', layout: str = 'auto',
                 transform: Optional[Callable] = None):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.samples = self._index_samples(layout)

    def _index_samples(self, layout: str):
        split_dir = self.root / self.split
        npz_file = self.root / f'{self.split}.npz'

        if layout in ('auto', 'npz') and npz_file.exists():
            return self._index_npz(npz_file)
        if layout in ('auto', 'paired') and split_dir.exists() and any(split_dir.glob('seq_*')):
            return self._index_paired(split_dir)
        if layout in ('auto', 'per_frame') and (split_dir / 'rad').exists():
            return self._index_per_frame(split_dir)

        raise FileNotFoundError(
            f"Could not find a recognised layout for dataset '{self.dataset_name}' "
            f"under '{self.root}' (split='{self.split}'). "
            'See docs/Dataset_Preparation.md for the expected directory structure.'
        )

    @staticmethod
    def _index_npz(npz_file: Path):
        with np.load(npz_file, allow_pickle=True) as data:
            n = len(data['joints'])
        return [{'source': 'npz', 'file': npz_file, 'index': i} for i in range(n)]

    @staticmethod
    def _index_paired(split_dir: Path):
        samples = []
        for seq_dir in sorted(split_dir.glob('seq_*')):
            rad = np.load(seq_dir / 'rad.npy', mmap_mode='r')
            samples.extend({'source': 'paired', 'seq_dir': seq_dir, 'frame': t} for t in range(rad.shape[0]))
        return samples

    @staticmethod
    def _index_per_frame(split_dir: Path):
        rad_files = sorted((split_dir / 'rad').glob('*.npy'))
        pose_files = sorted((split_dir / 'pose').glob('*.npy'))
        if len(rad_files) != len(pose_files):
            raise ValueError(
                f'rad/ ({len(rad_files)}) and pose/ ({len(pose_files)}) file counts '
                f'differ under {split_dir}.'
            )
        return [{'source': 'per_frame', 'rad': r, 'pose': p} for r, p in zip(rad_files, pose_files)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        item = self.samples[idx]

        if item['source'] == 'npz':
            with np.load(item['file'], allow_pickle=True) as data:
                key = 'rad' if 'rad' in data else 'heatmap'
                rad = data[key][item['index']]
                pose = data['joints'][item['index']]
        elif item['source'] == 'paired':
            rad = np.load(item['seq_dir'] / 'rad.npy', mmap_mode='r')[item['frame']]
            pose = np.load(item['seq_dir'] / 'pose.npy', mmap_mode='r')[item['frame']]
        else:
            rad = np.load(item['rad'])
            pose = np.load(item['pose'])

        sample = {
            'rad': torch.from_numpy(np.array(rad, copy=True)).float(),
            'pose': torch.from_numpy(np.array(pose, copy=True)).float(),
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
