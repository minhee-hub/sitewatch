from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font_path() -> Path | None:
    candidates = [
        Path(r"C:\Windows\Fonts\malgun.ttf"),
        Path(r"C:\Windows\Fonts\malgunbd.ttf"),
        Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


@lru_cache(maxsize=16)
def load_font(size: int):
    path = _font_path()
    if path is not None:
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def text_size(text: str, size: int) -> tuple[int, int]:
    bbox = load_font(size).getbbox(text)
    return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])


def draw_texts(image: np.ndarray, items: list[tuple]) -> np.ndarray:
    """items: (x, y, text, bgr, size). y는 글자 상단."""
    if not items:
        return image
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    pen = ImageDraw.Draw(pil)
    for x, y, text, bgr, size in items:
        color = (int(bgr[2]), int(bgr[1]), int(bgr[0]))
        pen.text((int(x), int(y)), str(text), font=load_font(int(size)), fill=color)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
