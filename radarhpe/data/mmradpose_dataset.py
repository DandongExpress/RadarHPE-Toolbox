"""mmRadPose dataset loader.

RAD tensors with motion-capture ground truth, single-person.
Download page: IEEE DataPort (see docs/Dataset_Preparation.md).
"""
from radarhpe.data.base_radar_dataset import BaseRadarPoseDataset
from radarhpe.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register(name='mmradpose')
class MMRadPoseDataset(BaseRadarPoseDataset):
    dataset_name = 'mmRadPose'
    default_shape = (64, 64, 16)  # (range, angle, doppler)
    num_joints = 17
