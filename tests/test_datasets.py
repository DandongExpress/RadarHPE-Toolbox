"""Dataset loader tests using synthetic on-disk fixtures (no real radar data
required) — validates all three supported layouts (npz / paired / per-frame).
"""
import numpy as np

from radarhpe.data import HuPRDataset


def test_npz_layout(tmp_path):
    root = tmp_path / 'HuPR'
    root.mkdir()
    n = 5
    np.savez(
        root / 'train.npz',
        rad=np.random.randn(n, 64, 64, 16).astype(np.float32),
        joints=np.random.randn(n, 17, 3).astype(np.float32),
    )

    ds = HuPRDataset(root=str(root), split='train')
    assert len(ds) == n
    sample = ds[0]
    assert sample['rad'].shape == (64, 64, 16)
    assert sample['pose'].shape == (17, 3)


def test_paired_layout(tmp_path):
    root = tmp_path / 'HuPR'
    split_dir = root / 'train'
    seq_dir = split_dir / 'seq_0000'
    seq_dir.mkdir(parents=True)
    t = 6
    np.save(seq_dir / 'rad.npy', np.random.randn(t, 64, 64, 16).astype(np.float32))
    np.save(seq_dir / 'pose.npy', np.random.randn(t, 17, 3).astype(np.float32))

    ds = HuPRDataset(root=str(root), split='train')
    assert len(ds) == t
    sample = ds[0]
    assert sample['rad'].shape == (64, 64, 16)
    assert sample['pose'].shape == (17, 3)


def test_per_frame_layout(tmp_path):
    root = tmp_path / 'HuPR'
    split_dir = root / 'train'
    (split_dir / 'rad').mkdir(parents=True)
    (split_dir / 'pose').mkdir(parents=True)
    n = 4
    for i in range(n):
        np.save(split_dir / 'rad' / f'{i:04d}.npy', np.random.randn(64, 64, 16).astype(np.float32))
        np.save(split_dir / 'pose' / f'{i:04d}.npy', np.random.randn(17, 3).astype(np.float32))

    ds = HuPRDataset(root=str(root), split='train')
    assert len(ds) == n
    sample = ds[0]
    assert sample['rad'].shape == (64, 64, 16)
    assert sample['pose'].shape == (17, 3)
