"""PULSE — Doppler Prompting for Stable mmWave-based Human Pose Estimation.

Original repository:
    https://github.com/DandongExpress/Doppler-Prompting-for-Stable-mmWave-based-Human-Pose-Estimation
Paper:
    Zheng, S., Li, J., Lu, X., He, S., & Guan, Y. (2026). Doppler Prompting for
    Stable mmWave-based Human Pose Estimation. ICML 2026. arXiv:2605.13233.
License of the original repository: MIT.

TODO(integration): this file only reproduces the *public constructor
signature and module boundaries* documented in the original README so that
``radarhpe.create_model('pulse_1f')`` / ``radarhpe.create_model('pulse_kf')``
resolve to the right place. To make the models actually runnable, copy the
real implementations over:

    original repo file      -> this package
    ------------------------   ----------------------------------------
    pulse/tokenizer.py       -> radarhpe/archs/pulse/tokenizer.py
    pulse/prompting.py       -> radarhpe/archs/pulse/prompting.py
    pulse/model.py            -> radarhpe/archs/pulse/model.py
    pulse/multiframe.py       -> radarhpe/archs/pulse/multiframe.py

and replace the ``NotImplementedError`` bodies below with calls into those
modules. Keep the original MIT license header in each copied file.
"""
from typing import Tuple

import torch
import torch.nn as nn

from radarhpe.archs.base_model import BaseRadarHPEModel
from radarhpe.utils.registry import MODEL_REGISTRY

_SUPPORTED_DATASETS = ('HuPR', 'XRF55', 'mmRadPose')


class PULSEPrompting(nn.Module):
    """Confidence-gated, locality-restricted Doppler-to-spatial cross-attention.

    Reconceives the Doppler signature as a screened *motion prompt* rather
    than a symmetric feature channel: a learnable gate scores each Doppler
    token for motion relevance, and spatial tokens only attend to Doppler
    tokens within a local neighbourhood window.

    Args:
        range_bins: number of range bins in the input RAD tensor.
        angle_bins: number of angle bins in the input RAD tensor.
        doppler_bins: number of Doppler bins in the input RAD tensor.
        patch_size: spatial tokenization patch size ``(P_r, P_a)``.
        embed_dim: shared token embedding dimension.
        neighborhood: side length of the local cross-attention window
            (default ``3`` -> a 3x3 patch neighbourhood).
        beta: gate strength / temperature.
    """

    def __init__(
        self,
        range_bins: int = 64,
        angle_bins: int = 64,
        doppler_bins: int = 16,
        patch_size: Tuple[int, int] = (4, 4),
        embed_dim: int = 32,
        neighborhood: int = 3,
        beta: float = 1.0,
    ):
        super().__init__()
        self.range_bins = range_bins
        self.angle_bins = angle_bins
        self.doppler_bins = doppler_bins
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.neighborhood = neighborhood
        self.beta = beta
        raise NotImplementedError(
            'PULSEPrompting is a structural stub. Paste the real tokenizer.py + '
            'prompting.py implementation from the original PULSE repository here '
            '(see the TODO(integration) note at the top of this file).'
        )

    def forward(self, rad_tensor: torch.Tensor) -> torch.Tensor:
        """Args: rad_tensor ``[B, R, A, D]`` (single frame) or ``[B, K, R, A, D]``
        (multi-frame window). Returns confidence-gated spatial tokens ready
        for a transformer pose backbone."""
        raise NotImplementedError


@MODEL_REGISTRY.register(name='pulse_1f')
class PULSESingleFrame(BaseRadarHPEModel):
    """Single-frame PULSE: dual-domain tokenization + confidence-gated
    cross-attention + transformer pose head. 12M parameters, ~5.1 ms/frame."""

    paper = 'Zheng et al., ICML 2026 (arXiv:2605.13233)'
    datasets = _SUPPORTED_DATASETS
    input_type = 'rad_single_frame'
    metrics = ('MPJPE', 'PA-MPJPE', 'MPJVE', 'AKV')

    def __init__(
        self,
        range_bins: int = 64,
        angle_bins: int = 64,
        doppler_bins: int = 16,
        patch_size: Tuple[int, int] = (4, 4),
        embed_dim: int = 32,
        neighborhood: int = 3,
        beta: float = 1.0,
        transformer_depth: int = 4,
        attention_heads: int = 4,
        num_joints: int = 17,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.prompting = PULSEPrompting(
            range_bins=range_bins,
            angle_bins=angle_bins,
            doppler_bins=doppler_bins,
            patch_size=patch_size,
            embed_dim=embed_dim,
            neighborhood=neighborhood,
            beta=beta,
        )
        # TODO(integration): attach the stacked transformer encoder + MLP pose
        # head from pulse/model.py (PULSE class) here.

    def forward(self, rad_tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            'Paste the pulse/model.py forward pass (spatial tokens -> '
            'transformer -> MLP -> [B, J, 3] joints) here.'
        )


@MODEL_REGISTRY.register(name='pulse_kf')
class PULSEMultiFrame(PULSESingleFrame):
    """Multi-frame PULSE (K=9 by default): replaces single-frame Doppler
    tokens with confidence-weighted aggregates over a short window,
    see pulse/multiframe.py."""

    input_type = 'rad_multi_frame'

    def __init__(self, window_size: int = 9, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
        # TODO(integration): paste pulse/multiframe.py's aggregation module.
