"""radarhpe.data — dataset registry.

Importing this package auto-discovers every ``*_dataset.py`` module in this
folder, triggering each dataset's ``@DATASET_REGISTRY.register(...)``
decorator. Add a new dataset by dropping a new ``my_dataset.py`` file here.
"""
import importlib
from pathlib import Path

from radarhpe.data.base_radar_dataset import BaseRadarPoseDataset  # noqa: F401
from radarhpe.utils.registry import DATASET_REGISTRY  # noqa: F401

_dataset_folder = Path(__file__).parent
_dataset_filenames = [f.stem for f in _dataset_folder.glob('*_dataset.py')]
_dataset_modules = [importlib.import_module(f'radarhpe.data.{name}') for name in _dataset_filenames]

# Re-export the concrete dataset classes for convenient direct imports, e.g.
# `from radarhpe.data import HuPRDataset`.
from radarhpe.data.hupr_dataset import HuPRDataset  # noqa: F401,E402
from radarhpe.data.xrf55_dataset import XRF55Dataset  # noqa: F401,E402
from radarhpe.data.mmradpose_dataset import MMRadPoseDataset  # noqa: F401,E402
from radarhpe.data.mmvr_dataset import MMVRDataset  # noqa: F401,E402

__all__ = [
    'BaseRadarPoseDataset', 'DATASET_REGISTRY',
    'HuPRDataset', 'XRF55Dataset', 'MMRadPoseDataset', 'MMVRDataset',
]
