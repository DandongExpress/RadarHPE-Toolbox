"""Numerical correctness tests for radarhpe.metrics using small synthetic
tensors — no dataset or pretrained weights required.
"""
import torch

from radarhpe.metrics import mpjpe, pa_mpjpe, mpjve


def test_mpjpe_zero_for_identical_poses():
    pose = torch.randn(4, 17, 3)
    assert mpjpe(pose, pose).item() == 0.0


def test_mpjpe_known_offset():
    gt = torch.zeros(1, 17, 3)
    pred = torch.zeros(1, 17, 3)
    pred[..., 0] = 3.0
    pred[..., 1] = 4.0  # each joint offset by (3, 4, 0) -> distance 5
    assert torch.isclose(mpjpe(pred, gt), torch.tensor(5.0), atol=1e-4)


def test_pa_mpjpe_invariant_to_rotation_scale_translation():
    torch.manual_seed(0)
    gt = torch.randn(2, 17, 3)

    # Apply a known rotation (about z-axis), uniform scale, and translation.
    theta = 0.7
    rot = torch.tensor([
        [torch.cos(torch.tensor(theta)), -torch.sin(torch.tensor(theta)), 0.0],
        [torch.sin(torch.tensor(theta)), torch.cos(torch.tensor(theta)), 0.0],
        [0.0, 0.0, 1.0],
    ])
    scale = 2.5
    translation = torch.tensor([10.0, -3.0, 4.0])

    pred = torch.einsum('njk,kl->njl', gt, rot.T) * scale + translation

    # Raw MPJPE should be large (poses are in totally different frames)...
    raw_error = torch.norm(pred - gt, dim=-1).mean()
    assert raw_error > 1.0

    # ...but PA-MPJPE should recover ~0 after optimal alignment.
    aligned_error = pa_mpjpe(pred, gt)
    assert aligned_error.item() < 1e-3


def test_mpjve_zero_for_constant_velocity_match():
    gt_seq = torch.arange(10).float().view(10, 1, 1).repeat(1, 17, 3)
    pred_seq = gt_seq.clone()
    assert mpjve(pred_seq, gt_seq).item() == 0.0
