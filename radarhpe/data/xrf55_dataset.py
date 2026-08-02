"""XRF55 dataset loader.

Provides RA (range-angle) and RD (range-Doppler) maps rather than a native
RAD cube; PULSE's ``tools/preprocess_xrf55.py`` reconstructs a unified RAD
tensor via a weighted-distribution procedure (paper Appendix C). Run that
preprocessing script (see docs/Dataset_Preparation.md) before loading with
this class, which then reads the resulting standard layout.
Project page: see docs/Dataset_Preparation.md for the current link.
"""
from radarhpe.data.base_radar_dataset import BaseRadarPoseDataset
from radarhpe.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register(name='xrf55')
class XRF55Dataset(BaseRadarPoseDataset):
    dataset_name = 'XRF55'
    default_shape = (256, 128)  # (range, angle) / (range, doppler) before RAD reconstruction
    num_joints = 17
    #: multi-person capable dataset.
    supports_multi_person = True
