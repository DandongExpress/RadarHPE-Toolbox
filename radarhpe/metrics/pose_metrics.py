"""Standard 3D human pose evaluation metrics, shared across all models in the
toolbox so that PULSE / Agile-HPE / PPPR results are always computed the same
way and are directly comparable.

All functions accept / return plain ``torch.Tensor``s in millimetres, with
joints stored as ``[..., J, 3]`` (pelvis-centred is recommended but not
required for MPJPE/PA-MPJPE; MPJVE/AKV require a temporal ``T`` axis).
"""
import torch

from radarhpe.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register(name='mpjpe')
def mpjpe(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Mean Per-Joint Position Error.

    Args:
        pred, gt: ``[..., J, 3]`` predicted / ground-truth joint coordinates.

    Returns:
        Scalar tensor: mean Euclidean joint error (same units as input, mm by convention).
    """
    return torch.norm(pred - gt, dim=-1).mean()


def _procrustes_align(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Umeyama similarity alignment: finds the optimal rotation, scale, and
    translation that maps ``pred`` onto ``gt`` in a least-squares sense, per
    sample. Operates on a single pose ``[J, 3]``.
    """
    mu_pred = pred.mean(dim=0, keepdim=True)
    mu_gt = gt.mean(dim=0, keepdim=True)
    pred_c = pred - mu_pred
    gt_c = gt - mu_gt

    norm_pred = torch.norm(pred_c)
    norm_gt = torch.norm(gt_c)
    pred_c = pred_c / (norm_pred + 1e-8)
    gt_c = gt_c / (norm_gt + 1e-8)

    # Optimal rotation via SVD of the cross-covariance matrix.
    cov = gt_c.T @ pred_c
    u, s, vt = torch.linalg.svd(cov)
    d = torch.sign(torch.det(u @ vt))
    correction = torch.eye(3, device=pred.device, dtype=pred.dtype)
    correction[-1, -1] = d
    rotation = u @ correction @ vt

    scale = (s * torch.diag(correction)).sum() * (norm_gt / (norm_pred + 1e-8))
    aligned = scale * (pred_c * norm_pred) @ rotation.T + mu_gt
    return aligned


@METRIC_REGISTRY.register(name='pa_mpjpe')
def pa_mpjpe(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Procrustes-Aligned MPJPE: MPJPE after per-sample optimal similarity
    (rotation + uniform scale + translation) alignment, removing global pose
    ambiguities that a rigid backbone cannot be expected to resolve.

    Args:
        pred, gt: ``[N, J, 3]`` (a leading batch dim is required so each
            sample can be aligned independently; for a single pose, add a
            batch dim of 1).
    """
    if pred.dim() == 2:
        pred, gt = pred.unsqueeze(0), gt.unsqueeze(0)
    aligned = torch.stack([_procrustes_align(p, g) for p, g in zip(pred, gt)], dim=0)
    return torch.norm(aligned - gt, dim=-1).mean()


@METRIC_REGISTRY.register(name='mpjve')
def mpjve(pred_seq: torch.Tensor, gt_seq: torch.Tensor) -> torch.Tensor:
    """Mean Per-Joint Velocity Error: MPJPE computed on first-order finite
    differences (frame-to-frame joint displacement) rather than raw
    positions, penalising jittery / temporally inconsistent predictions.

    Args:
        pred_seq, gt_seq: ``[T, J, 3]`` (or ``[N, T, J, 3]``) joint sequences.
    """
    pred_vel = pred_seq[..., 1:, :, :] - pred_seq[..., :-1, :, :]
    gt_vel = gt_seq[..., 1:, :, :] - gt_seq[..., :-1, :, :]
    return torch.norm(pred_vel - gt_vel, dim=-1).mean()


@METRIC_REGISTRY.register(name='akv')
def akv(pred_seq: torch.Tensor, gt_seq: torch.Tensor = None) -> torch.Tensor:
    """Placeholder for the "AKV" metric referenced in the PULSE paper's
    results tables.

    TODO(verify): the exact definition of AKV was not confirmed from the
    original repository's README alone (the results table's column headers
    were not recoverable from the fetched page). Before relying on this
    metric, check the PULSE paper appendix / pulse/eval.py `eval.py` in the
    original repo for the precise formula, then replace this implementation.
    Do not report numbers produced by this stub as the paper's AKV metric.
    """
    raise NotImplementedError(
        'AKV is a placeholder metric — verify its definition against the PULSE '
        'paper appendix or eval.py before implementing (see TODO(verify) above).'
    )
