from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sitewatch.risk import kind_of


def cap_speed(velocity, max_speed: float | None) -> np.ndarray:
    vector = np.asarray(velocity, dtype=float).reshape(2)
    if max_speed is None:
        return vector
    speed = float(np.linalg.norm(vector))
    if speed > float(max_speed) > 0:
        return vector * (float(max_speed) / speed)
    return vector


def step_cv(state: np.ndarray, covariance: np.ndarray, dt: float, process_noise: float):
    dt = float(dt)
    if dt <= 0:
        return state, covariance
    transition = np.array(
        [
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt2 * dt2
    noise = float(process_noise) * np.array(
        [
            [dt4 / 4, 0, dt3 / 2, 0],
            [0, dt4 / 4, 0, dt3 / 2],
            [dt3 / 2, 0, dt2, 0],
            [0, dt3 / 2, 0, dt2],
        ],
        dtype=float,
    )
    return transition @ state, transition @ covariance @ transition.T + noise


class KalmanCV:
    """지면 좌표의 등속도 칼만 필터."""

    def __init__(self, xy, process_noise: float, measurement_noise: float):
        self.x = np.zeros(4, dtype=float)
        self.x[:2] = np.asarray(xy, dtype=float).reshape(2)
        self.P = np.diag([0.25, 0.25, 25.0, 25.0])
        self.q = float(process_noise)
        self.r = float(measurement_noise) ** 2

    @property
    def xy(self) -> np.ndarray:
        return self.x[:2]

    @property
    def velocity(self) -> np.ndarray:
        return self.x[2:4]

    def predict(self, dt: float) -> None:
        self.x, self.P = step_cv(self.x, self.P, dt, self.q)

    def update(self, measurement) -> None:
        z = np.asarray(measurement, dtype=float).reshape(2)
        observation = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        innovation = z - observation @ self.x
        innovation_cov = observation @ self.P @ observation.T + np.eye(2) * self.r
        gain = self.P @ observation.T @ np.linalg.inv(innovation_cov)
        self.x = self.x + gain @ innovation
        identity = np.eye(4)
        self.P = (identity - gain @ observation) @ self.P
        self.P = 0.5 * (self.P + self.P.T)

    def forecast(self, horizon: float, step: float) -> list[tuple[float, np.ndarray, np.ndarray]]:
        state = self.x.copy()
        covariance = self.P.copy()
        t = 0.0
        horizon = float(horizon)
        step = float(step)
        points = []
        if step <= 0 or horizon <= 0:
            return points
        while t + 1e-9 < horizon:
            dt = min(step, horizon - t)
            state, covariance = step_cv(state, covariance, dt, self.q)
            t += dt
            points.append((t, state[:2].copy(), covariance[:2, :2].copy()))
        return points


@dataclass
class Observation:
    track_id: int
    class_name: str
    ground_xy: np.ndarray
    near_xy: np.ndarray
    image_xy: np.ndarray
    xyxy: np.ndarray
    conf: float
    footprint_radius_m: float


@dataclass
class Actor:
    track_id: int
    class_name: str
    xy: np.ndarray
    near_xy: np.ndarray
    velocity: np.ndarray
    footprint_radius_m: float
    forecast: list
    image_xy: np.ndarray
    xyxy: np.ndarray
    conf: float
    trail: list = field(default_factory=list)


class _Track:
    def __init__(self, obs: Observation, t: float, process_noise: float, measurement_noise: float, max_speed):
        self.track_id = int(obs.track_id)
        self.kf = KalmanCV(obs.ground_xy, process_noise, measurement_noise)
        self.last_t = float(t)
        self.last_meas = float(t)
        self.born_t = float(t)
        self.n = 1
        self.first_z = np.asarray(obs.ground_xy, dtype=float).copy()
        self.max_speed = max_speed
        self.trail = [self.kf.xy.copy()]
        self._remember(obs)

    def _remember(self, obs: Observation) -> None:
        self.class_name = obs.class_name
        self.image_xy = np.asarray(obs.image_xy, dtype=float).copy()
        self.xyxy = np.asarray(obs.xyxy, dtype=float).copy()
        self.conf = float(obs.conf)
        self.footprint_radius_m = float(obs.footprint_radius_m)
        self.near_offset = np.asarray(obs.near_xy, dtype=float) - np.asarray(obs.ground_xy, dtype=float)

    def _cap(self) -> None:
        self.kf.x[2:4] = cap_speed(self.kf.x[2:4], self.max_speed)

    def _push_trail(self, limit: int) -> None:
        self.trail.append(self.kf.xy.copy())
        if len(self.trail) > limit:
            self.trail = self.trail[-limit:]

    def update(self, obs: Observation, t: float, trail_limit: int) -> None:
        dt = max(0.0, float(t) - self.last_t)
        if dt > 0:
            self.kf.predict(dt)
        self.kf.update(obs.ground_xy)
        self.n += 1
        if self.n == 2:
            elapsed = max(1e-3, float(t) - self.born_t)
            self.kf.x[2:4] = (np.asarray(obs.ground_xy, dtype=float) - self.first_z) / elapsed
        self._cap()
        self.last_t = float(t)
        self.last_meas = float(t)
        self._remember(obs)
        self._push_trail(trail_limit)

    def coast(self, t: float, trail_limit: int) -> None:
        dt = float(t) - self.last_t
        if dt > 0:
            self.kf.predict(dt)
            self._cap()
            self.last_t = float(t)
            self._push_trail(trail_limit)

    def actor(self, horizon: float, step: float) -> Actor:
        xy = self.kf.xy.copy()
        return Actor(
            track_id=self.track_id,
            class_name=self.class_name,
            xy=xy,
            near_xy=xy + self.near_offset,
            velocity=self.kf.velocity.copy(),
            footprint_radius_m=self.footprint_radius_m,
            forecast=self.kf.forecast(horizon, step),
            image_xy=self.image_xy.copy(),
            xyxy=self.xyxy.copy(),
            conf=self.conf,
            trail=list(self.trail),
        )


class TrackStore:
    def __init__(self, config: dict, metric: bool = True):
        tracking = config["tracking"]
        risk = config["risk"]
        self.process_noise = float(tracking["process_noise"])
        self.measurement_noise = float(tracking["measurement_noise"])
        self.max_age = float(tracking["max_age_s"])
        self.trail_length = int(tracking["trail_length"])
        self.horizon = float(risk["prediction_horizon_s"])
        self.step = float(risk["prediction_step_s"])
        self.max_speed = float(risk["max_speed_m_s"]) if metric else None
        self.tracks: dict[int, _Track] = {}

    def sync(self, observations: list[Observation], t: float) -> list[Actor]:
        seen = set()
        for obs in observations:
            seen.add(int(obs.track_id))
            current = self.tracks.get(int(obs.track_id))
            if current is None:
                self.tracks[int(obs.track_id)] = _Track(
                    obs,
                    t,
                    self.process_noise,
                    self.measurement_noise,
                    self.max_speed,
                )
            else:
                current.update(obs, t, self.trail_length)
        for track_id, track in list(self.tracks.items()):
            if track_id in seen:
                continue
            if t - track.last_meas > self.max_age:
                del self.tracks[track_id]
            else:
                track.coast(t, self.trail_length)
        return [track.actor(self.horizon, self.step) for track in self.tracks.values()]


def split_actors(actors: list[Actor], config: dict) -> tuple[list[Actor], list[Actor]]:
    workers = []
    equipment = []
    for actor in actors:
        if kind_of(actor.class_name, config) == "worker":
            workers.append(actor)
        else:
            equipment.append(actor)
    return workers, equipment
