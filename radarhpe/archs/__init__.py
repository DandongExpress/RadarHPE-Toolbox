"""radarhpe.archs — model architecture registry.

Importing this package auto-discovers and imports every ``*_arch.py`` module
in this folder (mirroring BasicSR / IQA-PyTorch's plugin discovery), which
triggers each model's ``@MODEL_REGISTRY.register(...)`` decorator. Add a new
model by dropping a new ``my_model_arch.py`` file here — no other file needs
to change.
"""
import importlib
from pathlib import Path

from radarhpe.archs.base_model import BaseRadarHPEModel  # noqa: F401
from radarhpe.utils.registry import MODEL_REGISTRY, BACKBONE_REGISTRY  # noqa: F401

_arch_folder = Path(__file__).parent
_arch_filenames = [f.stem for f in _arch_folder.glob('*_arch.py')]
_arch_modules = [importlib.import_module(f'radarhpe.archs.{name}') for name in _arch_filenames]

__all__ = ['BaseRadarHPEModel', 'MODEL_REGISTRY', 'BACKBONE_REGISTRY']
