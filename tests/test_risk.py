from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from sitewatch.config import load_config
from sitewatch.risk import assess_frame, assess_pair, kind_of, max_level, stopping_parts, zone_radii

CONFIG = load_config()


def actor(track_id, class_name, xy, velocity=None, footprint=1.0, forecast=None):
    point = np.asarray(xy, dtype=float)
    speed = np.zeros(2) if velocity is None else np.asarray(velocity, dtype=float)
    return SimpleNamespace(
        track_id=track_id,
        class_name=class_name,
        xy=point,
        near_xy=point.copy(),
        velocity=speed,
        footprint_radius_m=footprint,
        forecast=forecast or [],
    )


def forecast(xy, velocity, horizon=3.0, step=0.5):
    origin = np.asarray(xy, dtype=float)
    speed = np.asarray(velocity, dtype=float)
    points = []
    t = 0.0
    while t + 1e-9 < horizon:
        t = min(horizon, round(t + step, 10))
        points.append((t, origin + speed * t, np.eye(2)))
    return points


def test_stopping_distance_at_10_kmh():
    reaction, braking, _total = stopping_parts(10 / 3.6, 2.5, 2.0)
    assert reaction == pytest.approx(6.94, abs=0.02)
    assert braking == pytest.approx(1.93, abs=0.02)


def test_excavator_zones_are_swing_radius():
    red, yellow, awareness = zone_radii("excavator", 3.0, 0.0, CONFIG)
    assert red == pytest.approx(5)
    assert yellow == pytest.approx(5)
    assert awareness == pytest.approx(8)


def test_worker_inside_swing_radius_is_critical():
    pair = assess_pair(actor(1, "person", [4, 0]), actor(2, "excavator", [0, 0], footprint=2), CONFIG)
    assert pair.level == "critical"
    assert "위험반경" in pair.reason


def test_worker_in_awareness_only_is_caution():
    pair = assess_pair(actor(1, "person", [6.5, 0]), actor(2, "excavator", [0, 0]), CONFIG)
    assert pair.level == "caution"
    assert pair.reason == "인지 반경 안"


def test_fast_truck_stopping_band_is_warning():
    truck = actor(2, "dump truck", [0, 0], velocity=[10 / 3.6, 0], footprint=1.5)
    pair = assess_pair(actor(1, "person", [10, 0]), truck, CONFIG)
    assert pair.level == "warning"
    assert pair.reason == "정지거리 범위 안"
    assert pair.red_m == pytest.approx(3.0)
    assert pair.yellow_m > pair.red_m


def test_short_ttc_is_critical_before_contact():
    worker = actor(1, "person", [14, 0], velocity=[-10, 0])
    machine = actor(2, "excavator", [0, 0])
    pair = assess_pair(worker, machine, CONFIG)
    assert pair.level == "critical"
    assert "충돌예상시간" in pair.reason
    assert pair.ttc_s == pytest.approx(1.4)


def test_predicted_entry_within_1_5s_is_critical():
    worker = actor(1, "person", [11, 0], velocity=[-4, 0], forecast=forecast([11, 0], [-4, 0]))
    machine = actor(2, "excavator", [0, 0], forecast=forecast([0, 0], [0, 0]))
    pair = assess_pair(worker, machine, CONFIG)
    assert pair.enter_s == pytest.approx(1.5)
    assert pair.level == "critical"
    assert "진입 예측" in pair.reason


def test_far_worker_is_safe_and_people_are_not_paired():
    pair = assess_pair(actor(1, "person", [30, 0]), actor(2, "excavator", [0, 0]), CONFIG)
    assert pair.level == "safe"
    pairs = assess_frame(
        [actor(1, "person", [0, 0]), actor(2, "person", [1, 0])],
        [],
        CONFIG,
    )
    assert pairs == []
    assert max_level([]) == "safe"
    assert kind_of("person", CONFIG) == "worker"
    assert kind_of("dump truck", CONFIG) == "mobile"
