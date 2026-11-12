from __future__ import annotations

import math

import cv2
import numpy as np

from sitewatch.config import label_of, level_label
from sitewatch.geometry import ground_to_image
from sitewatch.risk import RANK, kind_of
from sitewatch.text import draw_texts, text_size

LEVEL_BGR = {
    "safe": (90, 160, 90),
    "caution": (50, 180, 50),
    "warning": (0, 196, 255),
    "critical": (48, 48, 230),
}


def draw_calibration_preview(frame: np.ndarray, points, mode: str) -> np.ndarray:
    canvas = frame.copy()
    pixels = [(int(round(point[0])), int(round(point[1]))) for point in points]
    if mode == "homography" and len(pixels) == 4:
        cv2.polylines(canvas, [np.array(pixels, np.int32)], True, (0, 220, 255), 2, cv2.LINE_AA)
    elif len(pixels) >= 2:
        cv2.line(canvas, pixels[0], pixels[1], (0, 220, 255), 2, cv2.LINE_AA)
    for index, point in enumerate(pixels, start=1):
        cv2.circle(canvas, point, 7, (40, 40, 230), -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            str(index),
            (point[0] + 8, point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return canvas


def draw_scene(frame, actors, pairs, level: str, calibration, config: dict, notices: list[str]) -> np.ndarray:
    canvas = frame.copy()
    _draw_trails(canvas, actors, calibration, config)
    texts: list[tuple] = []
    _draw_pairs(canvas, pairs, calibration, texts)
    _draw_boxes(canvas, actors, pairs, config, texts)
    _draw_banner(canvas, pairs, level, config, notices, texts)
    return draw_texts(canvas, texts)


def _project(calibration, points) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if calibration is None or len(pts) == 0:
        return pts
    return ground_to_image(calibration, pts)


def _finite(points, limit: float = 20000) -> list[tuple[int, int]]:
    clean = []
    for point in np.asarray(points, dtype=float).reshape(-1, 2):
        if not np.all(np.isfinite(point)):
            continue
        x, y = float(point[0]), float(point[1])
        if abs(x) > limit or abs(y) > limit:
            continue
        clean.append((int(round(x)), int(round(y))))
    return clean


def _polyline(image, points, color, thickness: int, closed: bool = False) -> None:
    poly = _finite(points)
    if len(poly) < 2:
        return
    cv2.polylines(image, [np.array(poly, np.int32)], closed, color, thickness, cv2.LINE_AA)


def _dashed(image, points, color, thickness: int = 2) -> None:
    poly = _finite(points)
    for start, end in zip(poly, poly[1:]):
        length = math.hypot(end[0] - start[0], end[1] - start[1])
        if length < 1:
            continue
        cursor = 0.0
        drawing = True
        while cursor < length:
            piece = 10 if drawing else 7
            nxt = min(length, cursor + piece)
            if drawing:
                p0 = _lerp(start, end, cursor / length)
                p1 = _lerp(start, end, nxt / length)
                cv2.line(image, p0, p1, color, thickness, cv2.LINE_AA)
            cursor = nxt
            drawing = not drawing


def _lerp(start, end, t: float) -> tuple[int, int]:
    return (
        int(round(start[0] + (end[0] - start[0]) * t)),
        int(round(start[1] + (end[1] - start[1]) * t)),
    )


def _ellipse(mean, covariance, nsig: float = 2.0, count: int = 36) -> np.ndarray:
    mean = np.asarray(mean, dtype=float).reshape(2)
    cov = np.asarray(covariance, dtype=float).reshape(2, 2)
    cov = 0.5 * (cov + cov.T)
    try:
        values, vectors = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return np.zeros((0, 2))
    values = np.clip(values, 1e-6, None)
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    circle = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    transform = vectors @ np.diag(nsig * np.sqrt(values))
    return mean + circle @ transform.T


def _worst_level(pairs, equipment_id: int) -> str:
    levels = [pair.level for pair in pairs if pair.equipment_id == equipment_id]
    if not levels:
        return "safe"
    return max(levels, key=lambda level: RANK[level])


def _worker_level(pairs, worker_id: int) -> str:
    levels = [pair.level for pair in pairs if pair.worker_id == worker_id]
    if not levels:
        return "safe"
    return max(levels, key=lambda level: RANK[level])


def _actor_color(actor, config) -> tuple[int, int, int]:
    if kind_of(actor.class_name, config) == "worker":
        return (230, 170, 40)
    return (40, 140, 255)


def _draw_trails(image, actors, calibration, config) -> None:
    for actor in actors:
        color = _actor_color(actor, config)
        if len(actor.trail) >= 2:
            _polyline(image, _project(calibration, actor.trail), color, 2)
        if actor.forecast:
            future = [actor.xy] + [item[1] for item in actor.forecast]
            _dashed(image, _project(calibration, future), color, 2)
            last = actor.forecast[-1]
            if len(last) >= 3:
                ellipse = _ellipse(last[1], last[2])
                _polyline(image, _project(calibration, ellipse), color, 1, closed=True)


def _draw_pairs(image, pairs, calibration, texts: list) -> None:
    for pair in pairs:
        if pair.level == "safe":
            continue
        ends = _finite(_project(calibration, [pair.worker_xy, pair.equipment_xy]))
        if len(ends) < 2:
            continue
        color = LEVEL_BGR[pair.level]
        cv2.line(image, ends[0], ends[1], color, 2, cv2.LINE_AA)
        if pair.level in ("warning", "critical"):
            mid = ((ends[0][0] + ends[1][0]) // 2, (ends[0][1] + ends[1][1]) // 2)
            texts.append((mid[0] + 4, mid[1] - 18, f"{pair.distance_m:.1f}m", (255, 255, 255), 16))


def _draw_boxes(image, actors, pairs, config, texts: list) -> None:
    for actor in actors:
        worker = kind_of(actor.class_name, config) == "worker"
        level = _worker_level(pairs, actor.track_id) if worker else _worst_level(pairs, actor.track_id)
        color = LEVEL_BGR[level] if level != "safe" else _actor_color(actor, config)
        x1, y1, x2, y2 = [int(round(float(v))) for v in actor.xyxy]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        caption = f"{label_of(config, actor.class_name)} #{actor.track_id}"
        width, height = text_size(caption, 16)
        top = y1 - height - 8
        if top < 40:
            top = y2 + 4
            if top + height > image.shape[0] - 4:
                top = max(40, y1 + 4)
        cv2.rectangle(image, (x1, top), (x1 + width + 8, top + height + 6), color, -1)
        texts.append((x1 + 4, top + 2, caption, (255, 255, 255), 16))


def _headline(level: str, pairs, config: dict) -> str:
    label = level_label(config, level)
    if level in ("critical", "warning", "caution"):
        chosen = [pair for pair in pairs if pair.level == level]
        if chosen:
            pair = min(chosen, key=lambda item: item.distance_m)
            machine = label_of(config, pair.equipment_class)
            return (
                f"{label}  사람 #{pair.worker_id} · {machine} #{pair.equipment_id}"
                f"  {pair.distance_m:.1f}m  {pair.reason}"
            )
    return label


def _draw_banner(image, pairs, level: str, config: dict, notices: list[str], texts: list) -> None:
    lines = [(_headline(level, pairs, config), LEVEL_BGR.get(level, (255, 255, 255)), 20)]
    for notice in notices:
        lines.append((notice, (230, 230, 230), 16))
    height = 10
    for text, _color, size in lines:
        _width, text_height = text_size(text, size)
        height += text_height + 8
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], height), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.72, image, 0.28, 0, dst=image)
    y = 8
    for text, color, size in lines:
        texts.append((12, y, text, color, size))
        _width, text_height = text_size(text, size)
        y += text_height + 8
