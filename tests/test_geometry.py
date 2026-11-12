from __future__ import annotations

import json

import numpy as np
import pytest

from sitewatch.geometry import (
    calibration_homography,
    calibration_scale,
    footpoint_xy,
    footprint_radius_m,
    ground_to_image,
    image_to_ground,
    load_calibration,
    save_calibration,
)


def test_homography_roundtrip():
    points = [[0, 0], [10, 0], [10, 6], [0, 6]]
    calibration = calibration_homography(points, 10, 6)
    assert np.allclose(image_to_ground(calibration, points), [[0, 0], [10, 0], [10, 6], [0, 6]], atol=1e-3)
    assert np.allclose(ground_to_image(calibration, [[5, 3]]), [[5, 3]], atol=1e-2)


def test_perspective_edge_length():
    source = [[0, 0], [100, 10], [90, 80], [10, 70]]
    calibration = calibration_homography(source, 20, 10)
    edge = image_to_ground(calibration, [source[0], source[1]])
    assert np.linalg.norm(edge[1] - edge[0]) == pytest.approx(20, abs=1e-2)
    back = ground_to_image(calibration, image_to_ground(calibration, [[40, 40]]))
    assert np.allclose(back, [[40, 40]], atol=0.3)


def test_scale_distance_and_reject_identical_points():
    calibration = calibration_scale([0, 0], [100, 0], 5)
    assert np.allclose(image_to_ground(calibration, [[100, 0]]), [[5, 0]])
    assert np.allclose(ground_to_image(calibration, [[5, 0]]), [[100, 0]])
    with pytest.raises(ValueError):
        calibration_scale([0, 0], [0, 0], 5)


def test_collinear_points_rejected():
    with pytest.raises(ValueError):
        calibration_homography([[0, 0], [1, 0], [2, 0], [3, 0]], 10, 5)


def test_footpoint_and_footprint():
    assert footpoint_xy([0, 0, 10, 20]).tolist() == [5.0, 20.0]
    mask = [[0, 1], [4, 9], [2, 3]]
    assert np.allclose(footpoint_xy([0, 0, 10, 20], mask), [4, 9])
    calibration = calibration_scale([0, 0], [100, 0], 10)
    assert footprint_radius_m(calibration, [0, 0, 40, 80]) == pytest.approx(2.0)


def test_calibration_json_roundtrip(tmp_path):
    calibration = calibration_homography([[10, 10], [110, 20], [100, 80], [5, 70]], 8, 4)
    path = tmp_path / "cal.json"
    save_calibration(calibration, path)
    loaded = load_calibration(path)
    assert loaded.mode == "homography"
    assert loaded.width_m == pytest.approx(8)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["image_points"][1][0] == pytest.approx(110)
