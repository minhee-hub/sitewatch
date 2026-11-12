from __future__ import annotations

import numpy as np
import pytest

from sitewatch.config import load_config
from sitewatch.tracking import Observation, TrackStore, cap_speed


def _observation(index: int) -> Observation:
    point = np.array([float(index), 0.0])
    return Observation(
        track_id=1,
        class_name="person",
        ground_xy=point,
        near_xy=point.copy(),
        image_xy=np.array([0.0, 0.0]),
        xyxy=np.array([0, 0, 10, 10], dtype=float),
        conf=1.0,
        footprint_radius_m=0.4,
    )


def test_cap_speed():
    assert cap_speed([30, 0], 15)[0] == pytest.approx(15)
    assert cap_speed([3, 4], 15)[0] == pytest.approx(3)


def test_constant_velocity_forecast():
    store = TrackStore(load_config(), metric=True)
    actor = None
    for index in range(10):
        actors = store.sync([_observation(index)], index * 0.1)
        actor = actors[0]
    assert actor is not None
    assert actor.velocity[0] == pytest.approx(10, abs=0.8)
    future = min(actor.forecast, key=lambda item: abs(item[0] - 1.0))
    assert future[1][0] == pytest.approx(actor.xy[0] + 10, abs=1.5)
