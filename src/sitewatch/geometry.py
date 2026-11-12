from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Calibration:
    """이미지 좌표를 바닥 평면의 미터 좌표로 바꾼다."""

    mode: str
    image_points: np.ndarray
    width_m: float = 0.0
    height_m: float = 0.0
    distance_m: float = 0.0
    homography: np.ndarray | None = None
    inverse: np.ndarray | None = None
    meters_per_pixel: float | None = None

    def to_dict(self) -> dict:
        data = {
            "mode": self.mode,
            "image_points": np.asarray(self.image_points, dtype=float).reshape(-1, 2).tolist(),
        }
        if self.mode == "homography":
            data["width_m"] = self.width_m
            data["height_m"] = self.height_m
        else:
            data["distance_m"] = self.distance_m
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Calibration:
        mode = data["mode"]
        points = data["image_points"]
        if mode == "homography":
            return calibration_homography(points, float(data["width_m"]), float(data["height_m"]))
        if mode == "scale":
            if len(points) != 2:
                raise ValueError("두 점 보정에는 점이 두 개 필요합니다.")
            return calibration_scale(points[0], points[1], float(data["distance_m"]))
        raise ValueError(f"알 수 없는 보정 방식입니다: {mode}")


def save_calibration(calibration: Calibration, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(calibration.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_calibration(path: str | Path) -> Calibration:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Calibration.from_dict(data)


def calibration_homography(image_points, width_m: float, height_m: float) -> Calibration:
    """바닥 직사각형의 좌상, 우상, 우하, 좌하 네 점으로 호모그래피를 만든다."""
    if width_m <= 0 or height_m <= 0:
        raise ValueError("가로와 세로는 0보다 커야 합니다.")
    src = np.asarray(image_points, dtype=np.float32).reshape(4, 2)
    if _polygon_area(src) < 1.0:
        raise ValueError("네 점이 너무 가깝거나 한 직선 위에 있습니다.")
    dst = np.array(
        [[0, 0], [width_m, 0], [width_m, height_m], [0, height_m]],
        dtype=np.float32,
    )
    try:
        homography = cv2.getPerspectiveTransform(src, dst)
    except cv2.error as exc:
        raise ValueError("호모그래피를 만들지 못했습니다. 점 순서를 확인해 주세요.") from exc
    if not np.isfinite(homography).all() or abs(np.linalg.det(homography)) < 1e-12:
        raise ValueError("네 점이 한 직선에 가깝습니다.")
    try:
        inverse = np.linalg.inv(homography)
    except np.linalg.LinAlgError as exc:
        raise ValueError("호모그래피의 역변환을 만들지 못했습니다.") from exc
    return Calibration(
        mode="homography",
        image_points=src.astype(float),
        width_m=float(width_m),
        height_m=float(height_m),
        homography=homography,
        inverse=inverse,
    )


def calibration_scale(point_a, point_b, distance_m: float) -> Calibration:
    """두 점 사이 실거리로 균일 배율만 만든다. 원근 오차는 남는다."""
    if distance_m <= 0:
        raise ValueError("실거리는 0보다 커야 합니다.")
    start = np.asarray(point_a, dtype=float).reshape(2)
    end = np.asarray(point_b, dtype=float).reshape(2)
    pixel_distance = float(np.linalg.norm(end - start))
    if pixel_distance < 1.0:
        raise ValueError("두 점이 너무 가깝습니다.")
    return Calibration(
        mode="scale",
        image_points=np.stack([start, end]),
        distance_m=float(distance_m),
        meters_per_pixel=float(distance_m) / pixel_distance,
    )


def image_to_ground(calibration: Calibration, points) -> np.ndarray:
    pts = _as_points(points)
    if calibration.mode == "homography":
        transformed = cv2.perspectiveTransform(pts.astype(np.float32).reshape(-1, 1, 2), calibration.homography)
        return transformed.reshape(-1, 2).astype(float)
    return pts * float(calibration.meters_per_pixel)


def ground_to_image(calibration: Calibration, points) -> np.ndarray:
    pts = _as_points(points)
    if calibration.mode == "homography":
        transformed = cv2.perspectiveTransform(pts.astype(np.float32).reshape(-1, 1, 2), calibration.inverse)
        return transformed.reshape(-1, 2).astype(float)
    return pts / float(calibration.meters_per_pixel)


def meters_per_pixel_at(calibration: Calibration, image_xy) -> float:
    origin = _as_points(image_xy)[0]
    right = image_to_ground(calibration, origin + np.array([10.0, 0.0]))[0]
    base = image_to_ground(calibration, origin)[0]
    span = float(np.linalg.norm(right - base))
    if span < 1e-6:
        down = image_to_ground(calibration, origin + np.array([0.0, 10.0]))[0]
        span = float(np.linalg.norm(down - base))
    if span < 1e-6:
        return 1.0
    return span / 10.0


def footpoint_xy(xyxy, mask_xy=None) -> np.ndarray:
    """마스크의 가장 아래 점, 없으면 박스 하단 중앙."""
    if mask_xy is not None:
        pts = np.asarray(mask_xy, dtype=float).reshape(-1, 2)
        if len(pts) > 0:
            return pts[int(np.argmax(pts[:, 1]))].copy()
    x1, _y1, x2, y2 = [float(v) for v in xyxy]
    return np.array([(x1 + x2) * 0.5, y2], dtype=float)


def bottom_center_xy(xyxy) -> np.ndarray:
    return footpoint_xy(xyxy, None)


def footprint_radius_m(calibration: Calibration, xyxy) -> float:
    """바닥 접지 폭과 박스 높이의 국소 배율로 기체 반지름을 추정한다."""
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    bottom = image_to_ground(calibration, [[x1, y2], [x2, y2]])
    width_m = float(np.linalg.norm(bottom[1] - bottom[0]))
    foot = np.array([(x1 + x2) * 0.5, y2], dtype=float)
    depth_m = abs(y2 - y1) * meters_per_pixel_at(calibration, foot) * 0.45
    return float(0.5 * max(width_m, depth_m, 0.8))


def _as_points(points) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return np.zeros((0, 2), dtype=float)
    return pts.reshape(-1, 2)


def _polygon_area(points) -> float:
    pts = _as_points(points)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
