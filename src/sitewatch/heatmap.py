from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from sitewatch.media import write_image

TURBO = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)


def _splat(grid: np.ndarray, cx: int, cy: int, sigma: float, weight: float) -> None:
    if weight <= 0 or grid.size == 0:
        return
    sigma = max(float(sigma), 1.0)
    radius = int(np.ceil(sigma * 3))
    height, width = grid.shape
    x0 = max(0, cx - radius)
    x1 = min(width, cx + radius + 1)
    y0 = max(0, cy - radius)
    y1 = min(height, cy + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    patch = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma))
    grid[y0:y1, x0:x1] += np.float32(weight) * patch.astype(np.float32)


def _closest_on_box(box, point) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in np.asarray(box, dtype=float).reshape(4)]
    px, py = float(point[0]), float(point[1])
    return np.array([min(max(px, x1), x2), min(max(py, y1), y2)], dtype=float)


def _intersection(a, b):
    ax1, ay1, ax2, ay2 = [float(v) for v in np.asarray(a, dtype=float).reshape(4)]
    bx1, by1, bx2, by2 = [float(v) for v in np.asarray(b, dtype=float).reshape(4)]
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2, y2


def colorize(grid: np.ndarray, colormap: int = TURBO) -> tuple[np.ndarray, np.ndarray]:
    if grid.size == 0 or float(grid.max()) <= 1e-8:
        blank = np.zeros((*grid.shape, 3), np.uint8)
        return blank, np.zeros(grid.shape, np.uint8)
    positive = grid[grid > 1e-6]
    scale = float(np.percentile(positive, 96)) if positive.size else float(grid.max())
    scale = max(scale, 1e-6)
    norm = np.clip(grid / scale * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(norm, colormap), norm


class Heatmaps:
    """영상과 같은 크기의 겹침 히트맵. 시간이 지나면 옅어져 재생을 따라 움직인다."""

    def __init__(self, config: dict, image_shape, calibration=None, realtime: bool = False):
        self.config = config
        self.calibration = calibration
        self.realtime = realtime
        heat = config["heatmap"]
        self.trail_seconds = max(0.4, float(heat.get("trail_seconds", 2.5)))
        self.sigma_px = float(heat.get("overlap_sigma_px", 22))
        height, width = image_shape[:2]
        self.image_risk = np.zeros((height, width), np.float32)
        self.image_occ = self.image_risk

    def decay(self, dt: float = 0.0) -> None:
        if dt <= 0:
            return
        self.image_risk *= math.exp(-float(dt) / self.trail_seconds)

    def add_overlap(self, worker_xy, equip_xy, worker_box, equip_box, weight: float) -> None:
        if weight <= 0:
            return
        rect = _intersection(worker_box, equip_box)
        if rect is not None:
            self._paint_rect(rect, weight)
            return
        person = np.asarray(worker_xy, dtype=float).reshape(2)
        contact = _closest_on_box(equip_box, person)
        gap = float(np.linalg.norm(contact - person))
        reach = self.sigma_px * 5
        if gap > reach:
            _splat(self.image_risk, int(round(person[0])), int(round(person[1])), self.sigma_px * 0.7, weight * 0.8)
            return
        steps = max(3, int(gap / 8))
        for t in np.linspace(0.0, 1.0, steps):
            point = (1.0 - t) * person + t * contact
            near_person = 1.0 - float(t)
            _splat(
                self.image_risk,
                int(round(point[0])),
                int(round(point[1])),
                self.sigma_px * (0.35 + 0.25 * near_person),
                weight * (0.35 + 0.65 * near_person),
            )

    def _paint_rect(self, rect, weight: float) -> None:
        x1, y1, x2, y2 = rect
        span = max(x2 - x1, y2 - y1, 8.0)
        sigma = float(np.clip(span * 0.28, 6.0, self.sigma_px))
        nx = int(np.clip(round((x2 - x1) / max(sigma, 8.0)), 2, 6))
        ny = int(np.clip(round((y2 - y1) / max(sigma, 8.0)), 2, 6))
        for sx in np.linspace(x1, x2, nx):
            for sy in np.linspace(y1, y2, ny):
                _splat(self.image_risk, int(round(float(sx))), int(round(float(sy))), sigma, weight * 0.8)

    def blend(self, frame: np.ndarray) -> np.ndarray:
        color, norm = colorize(self.image_risk)
        if color.shape[:2] != frame.shape[:2]:
            size = (frame.shape[1], frame.shape[0])
            color = cv2.resize(color, size)
            norm = cv2.resize(norm, size)
        alpha = (norm.astype(np.float32) / 255.0) * 0.62
        alpha = alpha[..., None]
        mixed = frame.astype(np.float32) * (1.0 - alpha) + color.astype(np.float32) * alpha
        return np.clip(mixed, 0, 255).astype(np.uint8)

    def save(self, output_dir) -> dict[str, str]:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        color, _norm = colorize(self.image_risk)
        path = target / "heatmap_overlap.png"
        write_image(path, color)
        return {"heatmap_overlap": str(path)}
