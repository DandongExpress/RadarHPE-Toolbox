"""Agile-HPE — Physics-guided preprocessing for agile mmWave Human Pose Estimation.

Original repository:
    https://github.com/DandongExpress/Agile-MmWave-Hpe
Paper:
    Zheng, S., Li, J., Ni, M., Lu, X., & Guan, Y. (2026). Why Learn What
    Physics Already Knows? Realizing Agile mmWave-based Human Pose Estimation
    via Physics-Guided Preprocessing. ICME 2026. arXiv:2603.08236.
License of the original repository: Apache License 2.0 (see NOTICE).

TODO(integration): copy the deterministic front-end and regressor over:

    original repo file        -> this package
    --------------------------   -------------------------------------------
    models/physics.py           -> radarhpe/archs/agile/physics.py
      (SSP: Eq. 1-2, MCP: Eq. 3-7, HMSF: Eq. 8-9)
    models/pose_regressor.py    -> radarhpe/archs/agile/pose_regressor.py
      (PRN: Eq. 10-12)

and replace the ``NotImplementedError`` bodies below. Preserve the original
Apache-2.0 file headers when copying (see NOTICE at the repo root).
"""
from typing import Dict

import torch
import torch.nn as nn

from radarhpe.archs.base_model import BaseRadarHPEModel
from radarhpe.utils.registry import MODEL_REGISTRY

# The five deployment-time runtime profiles shipped in the original repo
# (see options/*.yaml equivalents under options/agile/ in this toolbox).
RUNTIME_PROFILES: Dict[str, str] = {
    'ultra_light': 'Maximum throughput',
    'light': 'Low-resource inference',
    'balanced': 'Accuracy-efficiency trade-off',
    'high_precision': 'Greater feature detail',
    'ultra_precision': 'Highest retained spatial detail',
}


class SSPMCPHMSFFrontend(nn.Module):
    """Deterministic, parameter-free physics-guided preprocessing front-end.

    Three stages applied directly to the range-angle-Doppler (RAD) cube:

    1. Spatial Structure Preservation (SSP) — anthropometrically plausible
       range-angle ROI mask, suppresses background clutter (paper Eq. 1-2).
    2. Motion Continuity Preservation (MCP) — dominant Doppler velocity per
       spatial cell + local consistency filtering (paper Eq. 3-7).
    3. Hierarchical Multi-Scale Fusion (HMSF) — three-scale 3D average
       pooling aligned with torso/limb/joint scales, trilinear restoration,
       and concatenation (paper Eq. 8-9).

    This front-end has **no learnable parameters**: all thresholds and
    pooling scales are configured via the runtime profile YAML (see
    ``options/agile/*.yml``), giving deployment-time adaptability without
    retraining the downstream regressor.
    """

    def __init__(self, roi_threshold: float = 0.5, doppler_threshold: float = 0.1,
                 variance_threshold: float = 0.05, local_window: int = 3,
                 pooling_scales=((4, 4, 2), (2, 2, 1))):
        super().__init__()
        self.roi_threshold = roi_threshold
        self.doppler_threshold = doppler_threshold
        self.variance_threshold = variance_threshold
        self.local_window = local_window
        self.pooling_scales = pooling_scales
        raise NotImplementedError(
            'SSPMCPHMSFFrontend is a structural stub. Paste the real SSP/MCP/HMSF '
            'implementation from models/physics.py in the original repository here.'
        )

    def forward(self, rad_tensor: torch.Tensor) -> torch.Tensor:
        """Args: rad_tensor ``[B, R, A, D]``. Returns pooled physics-guided
        features + global motion descriptors (paper Sec. 3)."""
        raise NotImplementedError


class PoseRegressionNetwork(nn.Module):
    """PRN: maps pooled physics-guided features + global motion descriptors
    to 17 three-dimensional joints via a two-ReLU-layer MLP (paper Eq. 10-12)."""

    def __init__(self, in_features: int = 512, hidden_dim: int = 256, num_joints: int = 17):
        super().__init__()
        self.num_joints = num_joints
        raise NotImplementedError(
            'PoseRegressionNetwork is a structural stub. Paste models/pose_regressor.py here.'
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


@MODEL_REGISTRY.register(name='agile_hpe')
class AgileHPE(BaseRadarHPEModel):
    """Agile mmWave HPE: deterministic SSP/MCP/HMSF front-end + lightweight PRN.

    5.1M trainable parameters, 55.7-88.9% smaller than representative mmWave
    HPE baselines, 18.2 FPS on a Raspberry Pi 5 (7.3 MB peak runtime memory).
    """

    paper = 'Zheng et al., ICME 2026 (arXiv:2603.08236)'
    datasets = ('HuPR',)
    input_type = 'rad_single_frame'
    metrics = ('MAJPE', 'PA-MAJPE')

    def __init__(self, runtime_profile: str = 'balanced', num_joints: int = 17, **frontend_kwargs):
        super().__init__()
        if runtime_profile not in RUNTIME_PROFILES:
            raise ValueError(
                f"Unknown runtime_profile '{runtime_profile}'. "
                f"Choose one of {list(RUNTIME_PROFILES)}."
            )
        self.runtime_profile = runtime_profile
        self.num_joints = num_joints
        self.frontend = SSPMCPHMSFFrontend(**frontend_kwargs)
        self.regressor = PoseRegressionNetwork(num_joints=num_joints)

    def forward(self, rad_tensor: torch.Tensor) -> torch.Tensor:
        features = self.frontend(rad_tensor)
        return self.regressor(features)

    def export_onnx(self, output_path: str, example_input: torch.Tensor) -> None:
        """Edge-deployment export, mirrors export_onnx.py in the original repo."""
        self.eval()
        torch.onnx.export(
            self, example_input, output_path,
            input_names=['rad'], output_names=['joints'],
            dynamic_axes={'rad': {0: 'batch'}, 'joints': {0: 'batch'}},
            opset_version=17,
        )
