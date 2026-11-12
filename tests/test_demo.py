from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from sitewatch.config import load_config
from sitewatch.detector import Detection, ScriptedDetector
from sitewatch.media import VideoSink, write_image
from sitewatch.pipeline import Pipeline, run_frames


def _imread(path):
    raw = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def test_unicode_image_and_video(tmp_path):
    image_path = tmp_path / "현장" / "heat.png"
    image = np.zeros((20, 30, 3), np.uint8)
    image[5, 5] = (0, 0, 255)
    write_image(image_path, image)
    loaded = _imread(image_path)
    assert loaded[5, 5, 2] == 255

    video_path = tmp_path / "영상" / "out.mp4"
    frame = np.zeros((48, 64, 3), np.uint8)
    with VideoSink(video_path, 10, (64, 48)) as sink:
        sink.write(frame)
        sink.write(frame)
    assert video_path.stat().st_size > 0


def test_uncalibrated_pipeline_skips_meter_risk():
    detector = ScriptedDetector()
    detector.set_frame(
        [
            Detection(1, "person", 1.0, np.array([10, 10, 40, 80], dtype=float)),
            Detection(2, "excavator", 1.0, np.array([20, 20, 120, 100], dtype=float)),
        ]
    )
    pipeline = Pipeline(load_config(), calibration=None, detector=detector, sound=False)
    result = pipeline.process(np.zeros((180, 240, 3), np.uint8), 0.0)
    assert result.calibrated is False
    assert result.level == "safe"
    assert result.pairs == []
    assert any("보정 필요" in notice for notice in result.notices)
    assert pipeline.heatmaps.image_risk.max() > 0


def test_overlap_heatmap_follows_the_frame(tmp_path):
    config = load_config()
    detector = ScriptedDetector()
    pipeline = Pipeline(config, calibration=None, detector=detector, sound=False)

    def frames():
        for index in range(6):
            shift = index * 12
            detector.set_frame(
                [
                    Detection(1, "person", 0.95, np.array([20 + shift, 30, 48 + shift, 110], dtype=float)),
                    Detection(2, "excavator", 0.95, np.array([70 + shift, 24, 150 + shift, 120], dtype=float)),
                ]
            )
            yield np.zeros((140, 240, 3), np.uint8)

    summary = run_frames(frames(), 8, pipeline, tmp_path / "영상", total=6)
    assert summary["frames"] == 6
    assert "disclaimer" not in summary
    assert "last_frame" not in summary
    video = Path(summary["video"])
    assert video.stat().st_size > 500
    heat = _imread(summary["heatmap_overlap"])
    assert heat is not None and heat.max() > 0
    assert heat.shape[1] == 240 and heat.shape[0] == 140
    before = pipeline.heatmaps.image_risk.max()
    detector.set_frame([])
    pipeline.process(np.zeros((140, 240, 3), np.uint8), 8.0)
    assert pipeline.heatmaps.image_risk.max() < before
