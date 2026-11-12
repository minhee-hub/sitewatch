from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Detection:
    track_id: int
    class_name: str
    conf: float
    xyxy: np.ndarray
    mask_xy: np.ndarray | None = None


class ScriptedDetector:
    """테스트와 합성 데모용. 모델 없이 박스 목록을 그대로 돌려준다."""

    def __init__(self):
        self.current: list[Detection] = []

    def set_frame(self, detections: list[Detection]) -> None:
        self.current = detections

    def track(self, frame) -> list[Detection]:
        return self.current


class YoloeDetector:
    """YOLOE 텍스트 프롬프트로 사람과 중장비를 찾고 ByteTrack으로 ID를 유지한다."""

    def __init__(self, config: dict):
        import torch
        from ultralytics import YOLOE

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model_key = "gpu" if self.device.startswith("cuda") else "cpu"
        self.names = [str(name) for name in config["classes"]]
        self.conf = float(config["model"]["confidence"])
        self.iou = float(config["model"]["iou"])
        self.imgsz = int(config["model"]["imgsz"])
        self._next_id = 1
        models_dir = Path(config["_root"]) / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        weight_name = Path(str(config["model"][model_key])).name
        self.weights = weight_name
        weight_path = models_dir / weight_name
        root_weight = Path(config["_root"]) / weight_name
        if not weight_path.exists() and root_weight.exists():
            weight_path = root_weight
        previous = Path.cwd()
        os.chdir(models_dir)
        try:
            # 가중치와 텍스트 인코더는 models/ 에 받는다.
            self.model = YOLOE(str(weight_path if weight_path.exists() else weight_name))
            text_embeddings = self.model.get_text_pe(self.names)
            self.model.set_classes(self.names, text_embeddings)
        finally:
            os.chdir(previous)

    def track(self, frame) -> list[Detection]:
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
            device=self.device,
        )
        if not results:
            return []
        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = _to_numpy(boxes.xyxy)
        conf = _to_numpy(boxes.conf)
        classes = _to_numpy(boxes.cls).astype(int)
        ids = None if boxes.id is None else _to_numpy(boxes.id).astype(int)
        masks = []
        if result.masks is not None and getattr(result.masks, "xy", None) is not None:
            masks = result.masks.xy
        names = result.names
        detections = []
        for index in range(len(xyxy)):
            class_id = int(classes[index])
            if isinstance(names, dict):
                class_name = str(names.get(class_id, class_id))
            else:
                class_name = str(names[class_id])
            if ids is None:
                track_id = self._next_id
                self._next_id += 1
            else:
                track_id = int(ids[index])
            mask_xy = None
            if index < len(masks):
                mask_xy = np.asarray(masks[index], dtype=float)
                if mask_xy.size == 0:
                    mask_xy = None
            detections.append(
                Detection(
                    track_id=track_id,
                    class_name=class_name.strip().lower(),
                    conf=float(conf[index]),
                    xyxy=np.asarray(xyxy[index], dtype=float),
                    mask_xy=mask_xy,
                )
            )
        return detections


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)
