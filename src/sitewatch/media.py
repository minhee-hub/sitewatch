from __future__ import annotations

import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import cv2
import numpy as np


def _ascii_path(path: Path) -> bool:
    try:
        str(path).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def write_image(path: str | Path, image: np.ndarray) -> None:
    """한글 경로에서도 PNG/JPG를 저장한다."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    extension = target.suffix if target.suffix else ".png"
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise RuntimeError(f"이미지를 저장할 수 없습니다: {target}")
    target.write_bytes(encoded.tobytes())


def read_image(path: str | Path) -> np.ndarray:
    raw = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"이미지를 읽지 못했습니다: {path}")
    return image


def ensure_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return np.ascontiguousarray(frame)


class VideoSink:
    def __init__(self, path: str | Path, fps: float, size: tuple[int, int]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.tmp: Path | None = None
        target = self.path
        if not _ascii_path(self.path):
            self.tmp = Path(tempfile.gettempdir()) / f"sitewatch_{uuid.uuid4().hex}.mp4"
            target = self.tmp
        width, height = int(size[0]), int(size[1])
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(target), fourcc, float(fps), (width, height))
        self._closed = False
        if not self.writer.isOpened():
            raise RuntimeError(f"영상을 저장할 수 없습니다: {self.path}")

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(ensure_bgr(frame))

    def close(self) -> None:
        if self._closed:
            return
        self.writer.release()
        self._closed = True
        if self.tmp is not None:
            if self.path.exists():
                self.path.unlink()
            shutil.move(str(self.tmp), str(self.path))

    def __enter__(self) -> VideoSink:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def transcode_h264(path: str | Path) -> bool:
    """브라우저에서 재생되도록 mp4v 영상을 H.264로 다시 담는다. ffmpeg가 없으면 그대로 둔다."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    target = Path(path)
    if not target.exists():
        return False
    source = target
    copied = None
    if not _ascii_path(target):
        copied = Path(tempfile.gettempdir()) / f"sitewatch_{uuid.uuid4().hex}.mp4"
        shutil.copy(target, copied)
        source = copied
    temporary = Path(tempfile.gettempdir()) / f"sitewatch_{uuid.uuid4().hex}.h264.mp4"
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    try:
        result = subprocess.run(command, capture_output=True, check=False)
    except OSError:
        result = None
    if copied is not None and copied.exists():
        copied.unlink()
    if result is None or result.returncode != 0 or not temporary.exists() or temporary.stat().st_size == 0:
        if temporary.exists():
            temporary.unlink()
        return False
    if target.exists():
        target.unlink()
    shutil.move(str(temporary), str(target))
    return True


class Capture:
    def __init__(self, source):
        self.source = source
        self.tmp: Path | None = None
        self.cap = self._open(source)

    def _open(self, source):
        if isinstance(source, int):
            capture = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if not capture.isOpened():
                capture.release()
                capture = cv2.VideoCapture(source)
            if not capture.isOpened():
                raise RuntimeError(f"카메라를 열 수 없습니다: {source}")
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return capture

        path = Path(str(source))
        capture = cv2.VideoCapture(str(source))
        if capture.isOpened():
            if str(source).startswith(("http://", "https://")):
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return capture
        capture.release()
        if path.exists() and not _ascii_path(path):
            self.tmp = Path(tempfile.gettempdir()) / f"sitewatch_in_{uuid.uuid4().hex}{path.suffix or '.mp4'}"
            shutil.copy(path, self.tmp)
            capture = cv2.VideoCapture(str(self.tmp))
            if capture.isOpened():
                return capture
            capture.release()
        raise RuntimeError(f"영상을 열 수 없습니다: {source}")

    def read(self):
        return self.cap.read()

    def get(self, prop):
        return self.cap.get(prop)

    def release(self) -> None:
        self.cap.release()
        if self.tmp is not None and self.tmp.exists():
            self.tmp.unlink()
            self.tmp = None

    def __enter__(self) -> Capture:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def parse_source(value):
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.lstrip("-").isdigit():
        return int(text)
    return text


def first_frame(source) -> np.ndarray:
    with Capture(parse_source(source)) as capture:
        ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError("첫 프레임을 읽지 못했습니다.")
    return ensure_bgr(frame)
