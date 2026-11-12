from __future__ import annotations

import csv
import json
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from sitewatch.config import label_of, level_label
from sitewatch.geometry import (
    bottom_center_xy,
    footprint_radius_m,
    footpoint_xy,
    image_to_ground,
    save_calibration,
)
from sitewatch.heatmap import Heatmaps
from sitewatch.media import Capture, VideoSink, ensure_bgr, parse_source
from sitewatch.risk import RANK, assess_frame, kind_of, max_level
from sitewatch.tracking import Observation, TrackStore, split_actors
from sitewatch.visualize import draw_scene

EVENT_FIELDS = [
    "시각_초",
    "등급",
    "근로자ID",
    "장비ID",
    "장비",
    "거리_m",
    "충돌예상_초",
    "진입예상_초",
    "이유",
]


@dataclass
class FrameResult:
    frame: np.ndarray
    level: str
    pairs: list
    new_events: list
    new_alerts: list
    calibrated: bool
    notices: list[str] = field(default_factory=list)


class Pipeline:
    def __init__(
        self,
        config: dict,
        calibration=None,
        detector=None,
        realtime: bool = False,
        sound: bool = False,
        overlay_heatmap: bool = True,
    ):
        self.config = config
        self.calibration = calibration
        self.detector = detector
        self.realtime = realtime
        self.sound = sound
        self.overlay_heatmap = overlay_heatmap
        self.tracks = TrackStore(config, metric=calibration is not None)
        self.heatmaps: Heatmaps | None = None
        self.prev_level: dict[tuple[int, int], str] = {}
        self.last_seen: dict[tuple[int, int], float] = {}
        self._last_t: float | None = None

    def ensure_detector(self):
        if self.detector is None:
            from sitewatch.detector import YoloeDetector

            self.detector = YoloeDetector(self.config)
        return self.detector

    def process(self, frame: np.ndarray, timestamp: float) -> FrameResult:
        frame = ensure_bgr(frame)
        detections = self.ensure_detector().track(frame)
        observations = [self._observation(detection) for detection in detections]
        actors = self.tracks.sync(observations, timestamp)
        calibrated = self.calibration is not None
        pairs = []
        level = "safe"
        if calibrated:
            workers, equipment = split_actors(actors, self.config)
            pairs = assess_frame(workers, equipment, self.config)
            level = max_level(pairs)
        else:
            workers, equipment = split_actors(actors, self.config)
        events, alerts = self._edges(pairs, timestamp) if calibrated else ([], [])
        if self.heatmaps is None or self.heatmaps.image_risk.shape != frame.shape[:2]:
            self.heatmaps = Heatmaps(self.config, frame.shape, self.calibration, realtime=self.realtime)
            self._last_t = None
        dt = 0.0 if self._last_t is None else max(0.0, float(timestamp) - self._last_t)
        self._last_t = float(timestamp)
        self.heatmaps.decay(dt)
        self._paint_overlap(workers, equipment, pairs, calibrated)
        notices = []
        if not calibrated:
            notices.append(self.config["risk"]["uncalibrated_notice"])
        drawn = draw_scene(frame.copy(), actors, pairs, level, self.calibration, self.config, notices)
        if self.overlay_heatmap:
            drawn = self.heatmaps.blend(drawn)
        if alerts and self.sound:
            beep()
        return FrameResult(
            frame=drawn,
            level=level,
            pairs=pairs,
            new_events=events,
            new_alerts=alerts,
            calibrated=calibrated,
            notices=notices,
        )

    def _observation(self, detection) -> Observation:
        kind = kind_of(detection.class_name, self.config)
        name = detection.class_name.strip().lower()
        if kind == "worker":
            image_ref = footpoint_xy(detection.xyxy, detection.mask_xy)
            image_near = image_ref
        elif kind == "swing":
            image_ref = bottom_center_xy(detection.xyxy)
            image_near = footpoint_xy(detection.xyxy, detection.mask_xy)
        else:
            image_ref = bottom_center_xy(detection.xyxy)
            image_near = image_ref
        if self.calibration is not None:
            ground = image_to_ground(self.calibration, image_ref)[0]
            near = image_to_ground(self.calibration, image_near)[0]
            radius = footprint_radius_m(self.calibration, detection.xyxy)
        else:
            ground = np.asarray(image_ref, dtype=float)
            near = np.asarray(image_near, dtype=float)
            radius = 0.0
        return Observation(
            track_id=int(detection.track_id),
            class_name=name,
            ground_xy=np.asarray(ground, dtype=float),
            near_xy=np.asarray(near, dtype=float),
            image_xy=np.asarray(image_ref, dtype=float),
            xyxy=np.asarray(detection.xyxy, dtype=float),
            conf=float(detection.conf),
            footprint_radius_m=radius,
        )

    def _paint_overlap(self, workers, equipment, pairs, calibrated: bool) -> None:
        weights = {"caution": 0.4, "warning": 0.75, "critical": 1.0}
        if calibrated:
            workers_by_id = {actor.track_id: actor for actor in workers}
            equipment_by_id = {actor.track_id: actor for actor in equipment}
            for pair in pairs:
                weight = weights.get(pair.level)
                if weight is None:
                    continue
                worker = workers_by_id.get(pair.worker_id)
                machine = equipment_by_id.get(pair.equipment_id)
                if worker is None or machine is None:
                    continue
                self.heatmaps.add_overlap(worker.image_xy, machine.image_xy, worker.xyxy, machine.xyxy, weight)
            return
        for worker in workers:
            for machine in equipment:
                if not _boxes_near(worker.xyxy, machine.xyxy, 48):
                    continue
                self.heatmaps.add_overlap(worker.image_xy, machine.image_xy, worker.xyxy, machine.xyxy, 0.7)

    def _edges(self, pairs, timestamp: float) -> tuple[list[dict], list]:
        events = []
        alerts = []
        seen = set()
        for pair in pairs:
            key = (pair.worker_id, pair.equipment_id)
            seen.add(key)
            previous = self.prev_level.get(key, "safe")
            if RANK[pair.level] >= RANK["warning"] and RANK[pair.level] > RANK[previous]:
                events.append(event_row(pair, timestamp, self.config))
            if pair.level == "critical" and previous != "critical":
                alerts.append(pair)
            self.prev_level[key] = pair.level
            self.last_seen[key] = timestamp
        for key in list(self.prev_level):
            if key not in seen and timestamp - self.last_seen.get(key, timestamp) > 2.0:
                self.prev_level[key] = "safe"
        return events, alerts


def _boxes_near(a, b, margin: float) -> bool:
    ax1, ay1, ax2, ay2 = [float(v) for v in np.asarray(a, dtype=float).reshape(4)]
    bx1, by1, bx2, by2 = [float(v) for v in np.asarray(b, dtype=float).reshape(4)]
    return ax1 - margin < bx2 and bx1 - margin < ax2 and ay1 - margin < by2 and by1 - margin < ay2


def event_row(pair, timestamp: float, config: dict) -> dict:
    ttc = "" if not math.isfinite(pair.ttc_s) else round(float(pair.ttc_s), 2)
    enter = "" if pair.enter_s is None else round(float(pair.enter_s), 2)
    return {
        "시각_초": round(float(timestamp), 3),
        "등급": level_label(config, pair.level),
        "근로자ID": pair.worker_id,
        "장비ID": pair.equipment_id,
        "장비": label_of(config, pair.equipment_class),
        "거리_m": round(float(pair.distance_m), 2),
        "충돌예상_초": ttc,
        "진입예상_초": enter,
        "이유": pair.reason,
    }


def beep() -> None:
    def _play() -> None:
        try:
            import winsound

            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        except Exception:
            return

    threading.Thread(target=_play, daemon=True).start()


def analyze_source(
    source,
    output_dir,
    config: dict,
    calibration=None,
    realtime: bool = False,
    sound: bool = False,
    detector=None,
    overlay_heatmap: bool = True,
    progress=None,
    on_frame=None,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pipeline = Pipeline(
        config,
        calibration=calibration,
        detector=detector,
        realtime=realtime,
        sound=sound,
        overlay_heatmap=overlay_heatmap,
    )
    with Capture(parse_source(source)) as capture:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        if not math.isfinite(fps) or fps < 1 or fps > 240:
            fps = 30.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frames = _iter_capture(capture)
        return run_frames(
            frames,
            fps,
            pipeline,
            output,
            progress=progress,
            total=total,
            calibration=calibration,
            on_frame=on_frame,
        )


def run_frames(
    frames,
    fps: float,
    pipeline: Pipeline,
    output_dir,
    progress=None,
    total: int = 0,
    calibration=None,
    on_frame=None,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sink = None
    events: list[dict] = []
    levels: list[str] = []
    index = 0
    try:
        for frame in frames:
            result = pipeline.process(frame, index / float(fps))
            if sink is None:
                sink = VideoSink(output / "annotated.mp4", fps, (result.frame.shape[1], result.frame.shape[0]))
            sink.write(result.frame)
            events.extend(result.new_events)
            levels.append(result.level)
            index += 1
            if progress is not None:
                progress(index, total)
            if on_frame is not None and (index == 1 or index % 3 == 0):
                on_frame(result.frame, index, total)
    finally:
        if sink is not None:
            sink.close()
        from sitewatch.media import transcode_h264

        video_path = output / "annotated.mp4"
        if video_path.exists():
            transcode_h264(video_path)
    if calibration is not None:
        save_calibration(calibration, output / "calibration.json")
    return _write_report(output, pipeline, events, levels, calibration, index)


def _iter_capture(capture: Capture):
    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        yield frame


def _write_report(output: Path, pipeline: Pipeline, events: list[dict], levels: list[str], calibration, frames: int) -> dict:
    event_path = output / "events.csv"
    with event_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDS)
        writer.writeheader()
        writer.writerows(events)
    heat_paths = {}
    if pipeline.heatmaps is not None:
        heat_paths = pipeline.heatmaps.save(output)
    highest = max(levels, key=lambda level: RANK[level]) if levels else "safe"
    summary = {
        "frames": frames,
        "calibrated": calibration is not None,
        "calibration_mode": None if calibration is None else calibration.mode,
        "max_level": highest,
        "max_level_ko": level_label(pipeline.config, highest),
        "critical_frames": sum(level == "critical" for level in levels),
        "warning_frames": sum(level == "warning" for level in levels),
        "caution_frames": sum(level == "caution" for level in levels),
        "event_count": len(events),
        "events": str(event_path),
        "video": str(output / "annotated.mp4") if frames else None,
        **heat_paths,
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary"] = str(summary_path)
    summary["event_rows"] = events
    return summary
