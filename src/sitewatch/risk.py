from __future__ import annotations

import math

import numpy as np

LEVELS = ("safe", "caution", "warning", "critical")
RANK = {name: index for index, name in enumerate(LEVELS)}


def stopping_parts(speed_m_s: float, reaction_s: float, decel_m_s2: float) -> tuple[float, float, float]:
    """반응거리, 제동거리, 합. 속도는 m/s."""
    speed = max(0.0, float(speed_m_s))
    reaction = speed * float(reaction_s)
    if decel_m_s2 <= 1e-6:
        braking = 0.0
    else:
        braking = (speed * speed) / (2.0 * float(decel_m_s2))
    return reaction, braking, reaction + braking


def kind_of(class_name: str, config: dict) -> str:
    key = class_name.strip().lower()
    profile = config.get("profiles", {}).get(key)
    if profile:
        return str(profile.get("kind", "mobile"))
    if key == "person":
        return "worker"
    return "mobile"


def zone_radii(class_name: str, footprint_radius_m: float, speed_m_s: float, config: dict) -> tuple[float, float, float]:
    """적색, 황색, 인지 반경(m). 반경은 서로 겹치지 않게 바깥으로만 커진다."""
    risk = config["risk"]
    key = class_name.strip().lower()
    profile = config.get("profiles", {}).get(key, {"kind": "mobile", "awareness_radius_m": 8.0})
    kind = profile.get("kind", "mobile")
    if kind == "swing":
        red = float(profile["swing_radius_m"])
    else:
        red = float(footprint_radius_m) + float(risk["static_buffer_m"])
    speed = min(max(0.0, float(speed_m_s)), float(risk["max_speed_m_s"]))
    _reaction, _braking, stopping = stopping_parts(
        speed,
        float(risk["reaction_time_s"]),
        float(risk["braking_decel_m_s2"]),
    )
    yellow = red + stopping
    awareness = max(float(profile.get("awareness_radius_m", yellow)), yellow)
    return red, yellow, awareness


def time_to_collision(worker_xy, worker_vel, equip_xy, equip_vel, min_closing: float) -> float:
    delta = np.asarray(worker_xy, dtype=float) - np.asarray(equip_xy, dtype=float)
    distance = float(np.linalg.norm(delta))
    if distance < 1e-6:
        return 0.0
    relative = np.asarray(worker_vel, dtype=float) - np.asarray(equip_vel, dtype=float)
    rate = float(np.dot(delta, relative) / distance)
    closing = -rate
    if closing <= float(min_closing):
        return math.inf
    return distance / closing


def predicted_entry_time(worker_forecast, equipment_forecast, radius: float) -> float | None:
    if not worker_forecast or not equipment_forecast:
        return None
    for worker_item, equip_item in zip(worker_forecast, equipment_forecast):
        distance = float(np.linalg.norm(np.asarray(worker_item[1], float) - np.asarray(equip_item[1], float)))
        if distance <= radius:
            return float(worker_item[0])
    return None


def _limit_speed(velocity, max_speed: float) -> np.ndarray:
    vector = np.asarray(velocity, dtype=float).reshape(2)
    speed = float(np.linalg.norm(vector))
    if max_speed > 0 and speed > max_speed:
        return vector * (max_speed / speed)
    return vector


class PairAssessment:
    def __init__(
        self,
        worker_id: int,
        equipment_id: int,
        equipment_class: str,
        distance_m: float,
        ttc_s: float,
        enter_s: float | None,
        level: str,
        reason: str,
        red_m: float,
        yellow_m: float,
        awareness_m: float,
        worker_xy: np.ndarray,
        equipment_xy: np.ndarray,
    ):
        self.worker_id = worker_id
        self.equipment_id = equipment_id
        self.equipment_class = equipment_class
        self.distance_m = distance_m
        self.ttc_s = ttc_s
        self.enter_s = enter_s
        self.level = level
        self.reason = reason
        self.red_m = red_m
        self.yellow_m = yellow_m
        self.awareness_m = awareness_m
        self.worker_xy = worker_xy
        self.equipment_xy = equipment_xy


def assess_pair(worker, equipment, config: dict) -> PairAssessment:
    risk = config["risk"]
    max_speed = float(risk["max_speed_m_s"])
    worker_xy = np.asarray(worker.xy, dtype=float)
    equip_xy = np.asarray(equipment.xy, dtype=float)
    near_xy = np.asarray(getattr(equipment, "near_xy", equip_xy), dtype=float)
    distance = min(
        float(np.linalg.norm(worker_xy - equip_xy)),
        float(np.linalg.norm(worker_xy - near_xy)),
    )
    worker_vel = _limit_speed(worker.velocity, max_speed)
    equip_vel = _limit_speed(equipment.velocity, max_speed)
    speed = float(np.linalg.norm(equip_vel))
    red, yellow, awareness = zone_radii(equipment.class_name, equipment.footprint_radius_m, speed, config)
    ttc = time_to_collision(
        worker_xy,
        worker_vel,
        equip_xy,
        equip_vel,
        float(risk["min_closing_speed_m_s"]),
    )
    horizon = float(risk["prediction_horizon_s"])
    critical_ttc = float(risk["critical_ttc_s"])
    warning_ttc = float(risk["warning_ttc_s"])
    enter = predicted_entry_time(getattr(worker, "forecast", []), getattr(equipment, "forecast", []), red)

    candidates: list[tuple[int, str]] = []
    if distance <= awareness:
        candidates.append((1, "인지 반경 안"))
    if red < distance <= yellow:
        candidates.append((2, "정지거리 범위 안"))
    if math.isfinite(ttc) and ttc < warning_ttc:
        candidates.append((2, f"충돌예상시간 {ttc:.1f}초"))
    if enter is not None and enter <= horizon:
        candidates.append((2, f"{enter:.1f}초 내 위험반경 진입 예측"))
    if distance <= red:
        candidates.append((3, "위험반경 안"))
    if math.isfinite(ttc) and ttc < critical_ttc:
        candidates.append((3, f"충돌예상시간 {ttc:.1f}초"))
    if enter is not None and enter <= critical_ttc:
        candidates.append((3, f"{enter:.1f}초 내 위험반경 진입 예측"))

    if not candidates:
        level, reason = "safe", ""
    else:
        rank = max(item[0] for item in candidates)
        reason = next(item[1] for item in candidates if item[0] == rank)
        level = LEVELS[rank]

    return PairAssessment(
        worker_id=int(worker.track_id),
        equipment_id=int(equipment.track_id),
        equipment_class=str(equipment.class_name),
        distance_m=distance,
        ttc_s=ttc,
        enter_s=enter,
        level=level,
        reason=reason,
        red_m=red,
        yellow_m=yellow,
        awareness_m=awareness,
        worker_xy=worker_xy.copy(),
        equipment_xy=equip_xy.copy(),
    )


def assess_frame(workers, equipment, config: dict) -> list[PairAssessment]:
    pairs = []
    for worker in workers:
        for machine in equipment:
            pairs.append(assess_pair(worker, machine, config))
    return pairs


def max_level(pairs: list[PairAssessment]) -> str:
    if not pairs:
        return "safe"
    return max(pairs, key=lambda pair: RANK[pair.level]).level
