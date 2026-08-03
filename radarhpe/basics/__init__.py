"""mmWave radar fundamentals: physics, heatmap I/O, CFAR, point clouds.

This subpackage is intentionally independent of the deep-learning model zoo
so beginners can explore radar data before training HPE models.

Quick start::

    from radarhpe.basics import (
        RadarConfig, synthesize_hupr_frame, to_ra_map,
        heatmap_to_pointcloud, plot_hupr_overview,
    )

    cube = synthesize_hupr_frame()
    cloud, cfar = heatmap_to_pointcloud(cube)
    plot_hupr_overview(cube, cloud=cloud, show=True)
"""
from radarhpe.basics.physics import (
    C,
    RadarConfig,
    aoa_from_phase_diff,
    if_frequency_for_range,
    phase_for_range,
    summarize_resolutions,
    velocity_from_phase_diff,
)
from radarhpe.basics.io import (
    canonicalize_hupr,
    list_hupr_frames,
    load_heatmap,
    load_hupr_frame,
    magnitude,
    power_db,
    to_ra_map,
    to_rad_magnitude,
    to_rd_map,
)
from radarhpe.basics.cfar import CFARResult, ca_cfar_2d, cfar_2d, nms_peaks, os_cfar_2d
from radarhpe.basics.pointcloud import PointCloud, heatmap_to_pointcloud, polar_to_cartesian, rd_heatmap_to_points
from radarhpe.basics.synthetic import synthesize_hupr_frame, synthesize_rad_cube
from radarhpe.basics.visualize import (
    plot_hupr_overview,
    plot_pointcloud,
    plot_ra_heatmap,
    plot_rd_heatmap,
)

__all__ = [
    'C',
    'RadarConfig',
    'if_frequency_for_range',
    'phase_for_range',
    'velocity_from_phase_diff',
    'aoa_from_phase_diff',
    'summarize_resolutions',
    'load_heatmap',
    'load_hupr_frame',
    'list_hupr_frames',
    'magnitude',
    'canonicalize_hupr',
    'to_rad_magnitude',
    'to_ra_map',
    'to_rd_map',
    'power_db',
    'CFARResult',
    'ca_cfar_2d',
    'os_cfar_2d',
    'cfar_2d',
    'nms_peaks',
    'PointCloud',
    'polar_to_cartesian',
    'heatmap_to_pointcloud',
    'rd_heatmap_to_points',
    'synthesize_rad_cube',
    'synthesize_hupr_frame',
    'plot_ra_heatmap',
    'plot_rd_heatmap',
    'plot_pointcloud',
    'plot_hupr_overview',
]
