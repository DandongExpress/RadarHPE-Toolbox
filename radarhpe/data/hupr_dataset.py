"""HuPR dataset loader.

Used by all three source papers. RAD tensors, single-person.
Download page: https://github.com/HuPR-project/HuPR (see docs/Dataset_Preparation.md).
"""
from radarhpe.data.base_radar_dataset import BaseRadarPoseDataset
from radarhpe.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register(name='hupr')
class HuPRDataset(BaseRadarPoseDataset):
    dataset_name = 'HuPR'
    #: native RAD resolution used across the source papers.
    default_shape = (64, 64, 16)  # (range, angle, doppler)
    num_joints = 17
