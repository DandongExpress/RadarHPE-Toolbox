"""RadarHPE-Toolbox — a unified, physics-guided toolbox for mmWave radar
Human Pose Estimation (HPE).

Quick start::

    import radarhpe

    print(radarhpe.list_models())
    # ['agile_hpe', 'pppr', 'pulse_1f', 'pulse_kf']

    model = radarhpe.create_model('pulse_1f', pretrained='path/or/hf-hub-id')
    joints = model.predict(rad_tensor)   # [B, 17, 3], mm, pelvis-centred

See README.md for installation options and docs/ for tutorials.

Note:
    This file **must** be present at ``radarhpe/__init__.py``. Without it,
    ``import radarhpe`` resolves to a PEP 420 namespace package and the public
    API below (``list_models``, ``create_model``, …) will be missing — which
    is exactly what breaks ``tests/test_registry.py`` on CI.
"""
from __future__ import annotations

from typing import Optional

from radarhpe.version import __version__
from radarhpe.utils.registry import MODEL_REGISTRY, DATASET_REGISTRY, METRIC_REGISTRY


def list_models():
    """Return the names of all registered HPE models."""
    return MODEL_REGISTRY.keys()


def list_datasets():
    """Return the names of all registered datasets."""
    return DATASET_REGISTRY.keys()


def list_metrics():
    """Return the names of all registered evaluation metrics."""
    return METRIC_REGISTRY.keys()


def create_model(name: str, pretrained: Optional[str] = None, device: str = 'cpu', **kwargs):
    """Instantiate a registered model by name, optionally loading pretrained weights.

    Args:
        name: a name returned by :func:`list_models`, e.g. ``'pulse_1f'``.
        pretrained: local checkpoint path or Hugging Face Hub repo id.
        device: torch device string.
        **kwargs: forwarded to the model's constructor.
    """
    model_cls = MODEL_REGISTRY.get(name)
    model = model_cls(**kwargs)
    if pretrained is not None:
        model.load_pretrained(pretrained)
    return model.to(device)


def create_dataset(name: str, **kwargs):
    """Instantiate a registered dataset by name (see :func:`list_datasets`)."""
    dataset_cls = DATASET_REGISTRY.get(name)
    return dataset_cls(**kwargs)


def create_metric(name: str):
    """Return a registered metric function by name (see :func:`list_metrics`)."""
    return METRIC_REGISTRY.get(name)


# Side-effect imports that populate the registries. Kept *after* the public
# API definitions so a circular import still sees ``list_models`` etc.
from radarhpe import archs as _archs  # noqa: E402,F401
from radarhpe import data as _data  # noqa: E402,F401
from radarhpe import metrics as _metrics  # noqa: E402,F401


__all__ = [
    '__version__',
    'MODEL_REGISTRY', 'DATASET_REGISTRY', 'METRIC_REGISTRY',
    'list_models', 'list_datasets', 'list_metrics',
    'create_model', 'create_dataset', 'create_metric',
]
