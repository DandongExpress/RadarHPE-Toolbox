"""Unit tests for radarhpe.basics (physics, I/O, CFAR, point clouds)."""
import numpy as np
import pytest

from radarhpe.basics import (
    RadarConfig,
    ca_cfar_2d,
    canonicalize_hupr,
    heatmap_to_pointcloud,
    os_cfar_2d,
    polar_to_cartesian,
    synthesize_hupr_frame,
    synthesize_rad_cube,
    to_ra_map,
    to_rad_magnitude,
    to_rd_map,
)


def test_range_resolution_matches_bandwidth():
    cfg = RadarConfig(bandwidth_hz=3.125e9)
    assert cfg.range_resolution_m == pytest.approx(0.04795, rel=1e-3)


def test_synthesize_and_convert_layouts():
    cube = synthesize_hupr_frame(complex_output=True)
    assert cube.ndim == 4
    assert np.iscomplexobj(cube)

    canon = canonicalize_hupr(cube)
    assert canon.shape == (cube.shape[0], 64, 64, 8)
    assert not np.iscomplexobj(canon)
    ra = to_ra_map(cube)
    rd = to_rd_map(cube)
    rad = to_rad_magnitude(cube)
    assert ra.shape == (64, 64)
    assert rd.shape[0] == 64
    assert rad.shape[0] == 64 and rad.shape[1] == 64
    assert rad.ndim == 3


def test_rad_cube_roundtrip_axes():
    rad = synthesize_rad_cube()
    assert rad.shape == (64, 64, 16)
    # Feed (R,A,D) into canonicalize → (D,R,A,1)
    canon = canonicalize_hupr(rad)
    assert canon.shape[0] == 16
    assert canon.shape[1] == 64
    assert canon.shape[2] == 64


def test_ca_cfar_finds_bright_blob():
    img = np.ones((32, 32), dtype=np.float64) * 0.1
    img[10:13, 10:13] = 5.0
    result = ca_cfar_2d(img, guard=(1, 1), train=(4, 4), pfa=1e-3)
    assert result.detections[11, 11]
    assert len(result.peaks) >= 1


def test_os_cfar_runs():
    img = np.random.default_rng(0).rayleigh(0.1, size=(24, 24))
    img[8, 8] = 3.0
    result = os_cfar_2d(img, guard=(1, 1), train=(3, 3), threshold_scale=4.0)
    assert result.detections.shape == img.shape


def test_heatmap_to_pointcloud_synthetic():
    cube = synthesize_hupr_frame()
    cloud, cfar = heatmap_to_pointcloud(cube, top_k=32, pfa=1e-2)
    assert cloud.xyz.shape[1] == 3
    assert len(cloud.xyz) == len(cloud.ranges) == len(cloud.intensities)
    assert len(cloud.xyz) > 0
    assert cfar.detections.any()


def test_polar_to_cartesian_forward():
    xyz = polar_to_cartesian(
        ranges=np.array([2.0]),
        azimuths=np.array([0.0]),
        elevations=np.array([0.0]),
    )
    assert xyz.shape == (1, 3)
    np.testing.assert_allclose(xyz[0], [2.0, 0.0, 0.0], atol=1e-6)
