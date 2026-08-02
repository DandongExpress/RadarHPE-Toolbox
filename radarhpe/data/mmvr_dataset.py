"""MMVR dataset loader.

Captured with a TI AWR2243, [256, 128] heatmaps. Used by PPPR; each frame is
stored as an ``.npz`` with keys ``heatmap`` ``[R, A]`` (or ``[R, A, E]``),
optional ``joints`` ``[J, 3]``, and optional ``doppler``.
Download page: see docs/Dataset_Preparation.md for the current link.
"""
from radarhpe.data.base_radar_dataset import BaseRadarPoseDataset
from radarhpe.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register(name='mmvr')
class MMVRDataset(BaseRadarPoseDataset):
    dataset_name = 'MMVR'
    default_shape = (256, 128)  # (range, angle)
    num_joints = 17
