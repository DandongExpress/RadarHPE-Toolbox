"""PPPR — Person Parametric Physics-informed Representation for mmWave HPE.

Original repository:
    https://github.com/DandongExpress/PPPR
Paper:
    Zheng, S., Li, J., Wang, G., Ni, M., Palit, A., Montana, G., & Guan, Y.
    (2025). Person Parametric Physics-informed Representation for
    mmWave-based Human Pose Estimation. ACM IMWUT/UbiComp 2026.
    arXiv:2512.23054.
License of the original repository: MIT.

TODO(integration): copy the MHP pipeline and reconstruction utilities over:

    original repo file       -> this package
    -------------------------   ------------------------------------------
    pppr/representation.py    -> radarhpe/archs/pppr/representation.py  (Theta_j = {p, s, q, v, beta, omega})
    pppr/initialization.py    -> radarhpe/archs/pppr/initialization.py  (Sec. 4.1)
    pppr/radar_simulation.py  -> radarhpe/archs/pppr/radar_simulation.py (Sec. 4.2)
    pppr/losses.py             -> radarhpe/archs/pppr/losses.py          (Sec. 4.3-4.4)
    pppr/mhp.py                -> radarhpe/archs/pppr/mhp.py             (full optimisation loop)
    pppr/multi_person.py       -> radarhpe/archs/pppr/multi_person.py    (ETCM-CFAR + DBSCAN + MLP)
    pppr/reconstruction.py     -> radarhpe/archs/pppr/reconstruction.py  (PPPR -> Heatmap/PC)
    pppr/skeleton.py           -> radarhpe/archs/pppr/skeleton.py
    models/*                   -> radarhpe/archs/backbones/  (RETR, HuprModel, mmDiff, PoseformerV2, MLP)

and replace the ``NotImplementedError`` bodies below. Preserve the original
MIT license header when copying.
"""
import torch
import torch.nn as nn

from radarhpe.archs.base_model import BaseRadarHPEModel
from radarhpe.utils.registry import MODEL_REGISTRY, BACKBONE_REGISTRY

SUPPORTED_DATASETS = ('MMVR', 'HuPR', 'XRF55')
SUPPORTED_BACKBONES = ('RETR', 'HuprModel', 'mmDiff', 'PoseformerV2', 'MLP')
SUPPORTED_INPUT_TYPES = ('heatmap', 'pc', 'pppr', 'pppr_heatmap', 'pppr_pc')


class MHPPipeline(nn.Module):
    """MmWave Human Parameterization (MHP): differentiable Heatmap -> PPPR
    optimisation pipeline (paper Sec. 4).

    Each joint ``j`` is modelled as a Gaussian primitive
    ``Theta_j = {p_j, s_j, q_j, v_j, beta_j, omega_j}`` encoding kinematic
    (position/scale/orientation/velocity) and electromagnetic (scattering
    intensity, Doppler signature) properties. The pipeline:

    1. Initialization — peak detection + Doppler/gradient velocity gives
       skeletal seeds (Sec. 4.1).
    2. Radar Simulation — re-renders a synthetic heatmap from the current
       primitives: ``H_sim = sum_j M_atten * M_range * M_Dopp * M_angle * R_j``
       (Sec. 4.2).
    3. Dual-Constraint Optimization — jointly minimises a kinematic loss
       (bone length + rigidity + joint-limit priors) and an electromagnetic
       IoU loss between simulated and original top-tau% energy masks
       (Sec. 4.3-4.4).
    """

    def __init__(self, w_em: float = 0.5, w_kine: float = 0.5, n_iter: int = 100,
                 tau_pct: float = 10.0, lr: float = 1e-3):
        super().__init__()
        self.w_em = w_em
        self.w_kine = w_kine
        self.n_iter = n_iter
        self.tau_pct = tau_pct
        self.lr = lr
        raise NotImplementedError(
            'MHPPipeline is a structural stub. Paste representation.py, '
            'initialization.py, radar_simulation.py, losses.py, and mhp.py '
            'from the original PPPR repository here.'
        )

    def fit(self, heatmap: torch.Tensor, doppler: torch.Tensor = None) -> dict:
        """Optimise PPPR primitives for a single heatmap (+ optional Doppler).
        Returns a dict of per-joint parameters ``{p, s, q, v, beta, omega}``."""
        raise NotImplementedError

    def reconstruct(self, pppr_params: dict, target: str = 'heatmap') -> torch.Tensor:
        """Reconstruct back to 'heatmap' or 'pc' (point cloud), see
        pppr/reconstruction.py."""
        raise NotImplementedError


@BACKBONE_REGISTRY.register(name='retr')
@BACKBONE_REGISTRY.register(name='hupr_model')
@BACKBONE_REGISTRY.register(name='mmdiff')
@BACKBONE_REGISTRY.register(name='poseformer_v2')
@BACKBONE_REGISTRY.register(name='mlp')
class _PlaceholderBackbone(nn.Module):
    """Structural stand-in for the pose-regression backbones evaluated in the
    PPPR paper (RETR / HuprModel / mmDiff / PoseformerV2 / MLP).

    TODO(integration): each of these should become its own class copied from
    ``models/`` in the original PPPR repository (or re-implemented from the
    respective baseline's own open-source release, with attribution). The
    single-registry-name-per-class pattern above is a placeholder; split into
    5 real classes before publishing.
    """

    def __init__(self, num_joints: int = 17, **kwargs):
        super().__init__()
        self.num_joints = num_joints
        raise NotImplementedError('Paste the real backbone implementation here.')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


@MODEL_REGISTRY.register(name='pppr')
class PPPRModel(BaseRadarHPEModel):
    """End-to-end PPPR pipeline: MHP preprocessing + a selectable pose
    backbone, trained on one of ``{heatmap, pc, pppr, pppr_heatmap, pppr_pc}``
    input representations (see paper Table 3-5)."""

    paper = 'Zheng et al., ACM IMWUT/UbiComp 2026 (arXiv:2512.23054)'
    datasets = SUPPORTED_DATASETS
    input_type = 'pppr'
    metrics = ('MPJPE', 'PA-MPJPE')

    def __init__(self, backbone: str = 'retr', input_repr: str = 'pppr',
                 num_joints: int = 17, mhp_kwargs: dict = None):
        super().__init__()
        valid_backbones = {'retr', 'hupr_model', 'mmdiff', 'poseformer_v2', 'mlp'}
        if backbone.lower().replace('-', '_') not in valid_backbones:
            raise ValueError(f"Unknown backbone '{backbone}'. Choose one of {SUPPORTED_BACKBONES}.")
        if input_repr not in SUPPORTED_INPUT_TYPES:
            raise ValueError(f"Unknown input_repr '{input_repr}'. Choose one of {SUPPORTED_INPUT_TYPES}.")

        self.backbone_name = backbone
        self.input_repr = input_repr
        self.input_type = input_repr
        self.num_joints = num_joints
        self.mhp = MHPPipeline(**(mhp_kwargs or {}))
        backbone_key = backbone.lower().replace('-', '_')
        self.backbone = BACKBONE_REGISTRY.get(backbone_key)(num_joints=num_joints)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` is a raw heatmap/point-cloud tensor; PPPR-based ``input_repr``
        values first pass through ``self.mhp`` before reaching the backbone."""
        if self.input_repr.startswith('pppr'):
            pppr_params = self.mhp.fit(x)
            if self.input_repr == 'pppr':
                backbone_input = pppr_params
            else:
                target = 'heatmap' if self.input_repr == 'pppr_heatmap' else 'pc'
                backbone_input = self.mhp.reconstruct(pppr_params, target=target)
        else:
            backbone_input = x
        return self.backbone(backbone_input)
