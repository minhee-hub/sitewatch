from __future__ import annotations

import json
import sys
import time
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from sitewatch.config import load_config, level_label
from sitewatch.geometry import Calibration, calibration_homography, calibration_scale
from sitewatch.media import Capture, first_frame
from sitewatch.pipeline import Pipeline, analyze_source
from sitewatch.visualize import draw_calibration_preview
from sitewatch.youtube import is_youtube, prepare_source

CONFIG = load_config()
POINT_NAMES = ["좌상", "우상", "우하", "좌하"]


def main() -> None:
    st.set_page_config(page_title="현장 히트맵·위험판단", layout="wide")
    _style()
    st.title("건설현장 객체 인식 히트맵·위험판단")
    st.write(
        "검출이 끝난 영상을 같은 화면 비율로 재생합니다. "
        "사람과 중장비가 겹치거나 맞닿은 자리만 히트맵으로 올리고, 재생 위치에 따라 최근 겹침은 진해지고 지나간 자리는 옅어집니다."
    )
    _init_state()

    with st.sidebar:
        st.header("입력")
        source_mode = st.radio("방식", ["영상 파일", "유튜브"], label_visibility="collapsed")
        overlay = st.checkbox("겹침 히트맵 표시", value=True)
        sound = st.checkbox("위험으로 바뀌는 순간 알림음", value=False)
        with st.expander("위험거리 기준", expanded=False):
            st.markdown(_criteria_markdown())
        st.header("바닥 보정")
        calibration_mode = st.selectbox(
            "방식",
            ["네 점 호모그래피", "두 점 배율", "보정 파일", "보정 안 함"],
        )
        _render_calibration_inputs(calibration_mode)

    if source_mode == "영상 파일":
        _video_page(calibration_mode, overlay, sound)
    else:
        _youtube_page(calibration_mode, overlay, sound)

    if st.session_state.get("last_result"):
        st.divider()
        _show_result(st.session_state.last_result)


def _style() -> None:
    st.markdown(
        """
        <style>
        html, body, [class*="css"] { font-family: "Malgun Gothic", "Apple SD Gothic Neo", sans-serif; }
        .block-container { padding-top: 1.4rem; max-width: 1180px; }
        h1 { font-size: 1.85rem; letter-spacing: -0.03em; }
        h2, h3 { letter-spacing: -0.02em; }
        div[data-testid="stMetric"] {
            background: #f3f6fb;
            border: 1px solid #d5deea;
            border-radius: 14px;
            padding: 0.7rem 0.9rem;
        }
        div[data-testid="stCaptionContainer"] { color: #d5deea; }
        [data-testid="stFileUploaderDropzoneInstructions"] span { font-size: 0 !important; line-height: 0 !important; }
        [data-testid="stFileUploaderDropzoneInstructions"] div::before {
            content: "영상 파일을 여기에 놓으세요";
            display: block;
            font-size: 1rem;
            line-height: 1.4;
            color: #e8eef6;
        }
        [data-testid="stFileUploaderDropzoneInstructions"] div::after {
            content: "mp4, avi, mov, mkv";
            display: block;
            font-size: 0.85rem;
            line-height: 1.4;
            color: #b7c3d4;
            margin-top: 0.2rem;
        }
        [data-testid="stFileUploader"] button[data-testid="stBaseButton-secondary"] { font-size: 0 !important; }
        [data-testid="stFileUploader"] button[data-testid="stBaseButton-secondary"]::after {
            content: "파일 선택";
            font-size: 0.95rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_state() -> None:
    if "px0" not in st.session_state:
        for index, (x, y) in enumerate(_default_points(960, 540)):
            st.session_state[f"px{index}"] = float(x)
            st.session_state[f"py{index}"] = float(y)
    st.session_state.setdefault("live_on", False)
    st.session_state.setdefault("width_m", 20.0)
    st.session_state.setdefault("height_m", 12.0)
    st.session_state.setdefault("distance_m", 5.0)
    if float(st.session_state.width_m) < 1:
        st.session_state.width_m = 20.0
    if float(st.session_state.height_m) < 1:
        st.session_state.height_m = 12.0


def _default_points(width: float, height: float) -> list[list[float]]:
    return [
        [0.23 * width, 0.22 * height],
        [0.77 * width, 0.22 * height],
        [0.94 * width, 0.90 * height],
        [0.06 * width, 0.90 * height],
    ]


def _criteria_markdown() -> str:
    risk = CONFIG["risk"]
    profiles = CONFIG["profiles"]
    speed = 10 / 3.6
    reaction = speed * float(risk["reaction_time_s"])
    braking = (speed * speed) / (2 * float(risk["braking_decel_m_s2"]))
    lines = [
        "산업안전보건기준에 관한 규칙은 차량계 건설기계에 부딪힐 장소와 굴착기 선회 위험 장소의 출입을 금지합니다. 고정된 미터 값은 없어서, 아래 숫자는 현장 설정값입니다.",
        f"이동 장비의 적색 반경은 지면 윤곽 반지름 + {risk['static_buffer_m']:g}m 입니다.",
        f"황색 반경은 적색에 정지거리를 더한 값입니다. 정지거리 = 속력 × {risk['reaction_time_s']}초 + 속력² / (2 × {risk['braking_decel_m_s2']}). 시속 10km이면 반응 약 {reaction:.1f}m, 제동 약 {braking:.1f}m입니다.",
        f"충돌예상시간이 {risk['critical_ttc_s']}초 미만이거나 그 안에 예측 동선이 적색으로 들어오면 위험, {risk['warning_ttc_s']}초 기준이면 경고입니다.",
        f"굴착기 작업반경 {profiles['excavator']['swing_radius_m']:.0f}m (인지 {profiles['excavator']['awareness_radius_m']:.0f}m), 크레인 {profiles['crane']['swing_radius_m']:.0f}m (인지 {profiles['crane']['awareness_radius_m']:.0f}m).",
        "보정 전에는 미터 거리를 계산하지 않습니다. 바닥 가로는 네 점이 둘러싼 실제 길이입니다. 값은 `config/defaults.yaml`에서 바꿉니다.",
    ]
    return "\n\n".join(lines)


def _render_calibration_inputs(mode: str) -> None:
    frame = st.session_state.get("snap")
    if frame is not None:
        token = f"{frame.shape[1]}x{frame.shape[0]}"
        if st.session_state.get("snap_token") != token:
            st.session_state.snap_token = token
            for index, (x, y) in enumerate(_default_points(frame.shape[1], frame.shape[0])):
                st.session_state[f"px{index}"] = float(round(x))
                st.session_state[f"py{index}"] = float(round(y))
    if mode == "네 점 호모그래피":
        st.caption("순서는 바닥 직사각형의 좌상, 우상, 우하, 좌하입니다.")
        count = 4
    elif mode == "두 점 배율":
        st.caption("원근은 그대로라 화면 가장자리 거리는 어긋날 수 있습니다.")
        count = 2
    elif mode == "보정 파일":
        uploaded = st.file_uploader("보정 JSON", type=["json"])
        if uploaded is not None:
            try:
                data = json.loads(uploaded.getvalue().decode("utf-8"))
                st.session_state.file_calibration = Calibration.from_dict(data)
                st.success(f"보정 파일을 읽었습니다. 방식: {data.get('mode')}")
            except (ValueError, json.JSONDecodeError, KeyError) as exc:
                st.error(f"보정 파일을 읽지 못했습니다: {exc}")
        return
    else:
        st.caption("보정 없이는 상자와 화면상 궤적만 보여주고, 미터 위험도는 계산하지 않습니다.")
        return

    for index in range(count):
        left, right = st.columns(2)
        with left:
            st.number_input(f"{POINT_NAMES[index]} X", key=f"px{index}", step=1.0)
        with right:
            st.number_input(f"{POINT_NAMES[index]} Y", key=f"py{index}", step=1.0)
    if mode == "네 점 호모그래피":
        st.number_input("바닥 가로 (m)", key="width_m", min_value=1.0, step=0.5, help="네 점이 둘러싼 실제 가로입니다.")
        st.number_input("바닥 세로 (m)", key="height_m", min_value=1.0, step=0.5, help="네 점이 둘러싼 실제 세로입니다.")
    else:
        st.number_input("두 점 사이 거리 (m)", key="distance_m", min_value=1.0, step=0.5)


def _points(count: int) -> list[list[float]]:
    return [[float(st.session_state[f"px{index}"]), float(st.session_state[f"py{index}"])] for index in range(count)]


def _build_calibration(mode: str):
    if mode == "보정 안 함":
        return None
    if mode == "보정 파일":
        calibration = st.session_state.get("file_calibration")
        if calibration is None:
            raise ValueError("보정 JSON 파일을 먼저 넣어 주세요.")
        return calibration
    if mode == "네 점 호모그래피":
        return calibration_homography(_points(4), float(st.session_state.width_m), float(st.session_state.height_m))
    if mode == "두 점 배율":
        points = _points(2)
        return calibration_scale(points[0], points[1], float(st.session_state.distance_m))
    return None


def _video_page(calibration_mode: str, overlay: bool, sound: bool) -> None:
    st.subheader("영상 파일")
    st.caption("분석을 시작하면 인식이 끝난 프레임 위에 겹침 히트맵이 프레임마다 바뀌며 재생됩니다.")
    uploaded = st.file_uploader("영상", type=["mp4", "avi", "mov", "mkv"], label_visibility="collapsed")
    if uploaded is not None:
        token = f"{uploaded.name}:{uploaded.size}"
        if st.session_state.get("upload_token") != token:
            dest = ROOT / "runs" / "uploads" / uploaded.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(uploaded.getvalue())
            st.session_state.upload_token = token
            st.session_state.video_path = str(dest)
            st.session_state.snap_source = "file"
            try:
                st.session_state.snap = first_frame(dest)
            except RuntimeError as exc:
                st.error(str(exc))
            else:
                st.rerun()
    snap = st.session_state.get("snap")
    if snap is not None and st.session_state.get("snap_source") == "file" and calibration_mode in ("네 점 호모그래피", "두 점 배율"):
        count = 4 if calibration_mode == "네 점 호모그래피" else 2
        preview = draw_calibration_preview(snap, _points(count), "homography" if count == 4 else "scale")
        _show_bgr(preview, "보정 미리보기")
    if st.button("분석 시작", type="primary", disabled=not st.session_state.get("video_path")):
        try:
            calibration = _build_calibration(calibration_mode)
        except ValueError as exc:
            st.error(str(exc))
            return
        bar = st.progress(0.0, text="분석을 준비하는 중")
        screen = st.empty()

        def progress(index, total):
            if total:
                bar.progress(min(1.0, index / total), text=f"{index} / {total} 프레임")
            else:
                bar.progress(0.0, text=f"{index} 프레임")

        def on_frame(image, _index, _total):
            _put_bgr(screen, image, "인식 영상 위 겹침 히트맵")

        try:
            summary = analyze_source(
                st.session_state.video_path,
                ROOT / "runs" / "ui-video",
                CONFIG,
                calibration=calibration,
                sound=sound,
                overlay_heatmap=overlay,
                progress=progress,
                on_frame=on_frame,
            )
        except Exception as exc:
            st.error(f"분석을 끝내지 못했습니다: {exc}")
            return
        st.session_state.last_result = summary
        st.session_state.snap_source = "file"


def _youtube_page(calibration_mode: str, overlay: bool, sound: bool) -> None:
    st.subheader("유튜브")
    st.caption("실시간 방송 주소와 일반 영상 주소를 넣을 수 있습니다.")
    url = st.text_input("유튜브 주소", placeholder="https://www.youtube.com/watch?v= 또는 /live/")
    left, right = st.columns(2)
    with left:
        if st.button("첫 화면으로 보정"):
            _grab_youtube_frame(url)
    with right:
        if st.button("실시간 분석 시작", type="primary"):
            _start_youtube(url, calibration_mode, overlay, sound)
    snap = st.session_state.get("snap")
    if snap is not None and st.session_state.get("snap_source") == "youtube" and calibration_mode in ("네 점 호모그래피", "두 점 배율"):
        count = 4 if calibration_mode == "네 점 호모그래피" else 2
        _show_bgr(draw_calibration_preview(snap, _points(count), "homography" if count == 4 else "scale"), "보정 미리보기")
    if st.session_state.get("live_on"):
        _live_view()


def _grab_youtube_frame(url: str) -> None:
    text = url.strip()
    if not is_youtube(text):
        st.error("유튜브 주소를 넣어 주세요.")
        return
    try:
        with st.spinner("유튜브 영상을 확인하는 중"):
            source, info = prepare_source(text)
            st.session_state.snap = first_frame(source)
            st.session_state.snap_source = "youtube"
            st.session_state.youtube_title = info["title"] if info else ""
    except RuntimeError as exc:
        st.error(str(exc))
        return
    st.rerun()


def _start_youtube(url: str, calibration_mode: str, overlay: bool, sound: bool) -> None:
    text = url.strip()
    if not is_youtube(text):
        st.error("유튜브 주소를 넣어 주세요.")
        return
    try:
        calibration = _build_calibration(calibration_mode)
    except ValueError as exc:
        st.error(str(exc))
        return
    _stop_live()
    try:
        with st.spinner("유튜브 재생 주소를 받는 중"):
            source, info = prepare_source(text)
        capture = Capture(source)
    except RuntimeError as exc:
        st.error(str(exc))
        return
    pipeline = Pipeline(
        CONFIG,
        calibration=calibration,
        realtime=True,
        sound=sound,
        overlay_heatmap=overlay,
    )
    try:
        with st.spinner("검출 모델을 준비합니다. 처음에는 가중치를 받습니다."):
            detector = pipeline.ensure_detector()
    except Exception as exc:
        capture.release()
        st.error(f"모델을 불러오지 못했습니다: {exc}")
        return
    st.session_state.live_cap = capture
    st.session_state.live_pipeline = pipeline
    st.session_state.live_t0 = time.perf_counter()
    st.session_state.live_on = True
    title = info["title"] if info else "유튜브"
    live = "실시간 방송" if info and info.get("is_live") else "영상"
    st.session_state.live_device = f"{title} · {live} · {detector.device} · {detector.weights}"


def _stop_live() -> None:
    capture = st.session_state.get("live_cap")
    if capture is not None:
        capture.release()
    st.session_state.live_cap = None
    st.session_state.live_pipeline = None
    st.session_state.live_on = False


@st.fragment(run_every=timedelta(milliseconds=250))
def _live_view() -> None:
    if not st.session_state.get("live_on"):
        return
    capture = st.session_state.get("live_cap")
    pipeline = st.session_state.get("live_pipeline")
    if capture is None or pipeline is None:
        st.warning("유튜브 영상이 연결되어 있지 않습니다.")
        return
    st.caption(st.session_state.get("live_device", ""))
    ok, frame = capture.read()
    if not ok or frame is None:
        st.error("프레임을 읽지 못했습니다. 방송이 끝났거나 주소가 만료되었을 수 있습니다.")
        _stop_live()
        return
    result = pipeline.process(frame, time.perf_counter() - st.session_state.live_t0)
    _show_bgr(result.frame, "인식 영상 위 겹침 히트맵")
    st.write(f"현재 등급: {level_label(CONFIG, result.level)}")
    for alert in result.new_alerts:
        st.toast(f"위험 {alert.distance_m:.1f}m · {alert.reason}")
    if st.button("분석 중지"):
        _stop_live()
        st.rerun()


def _show_result(summary: dict) -> None:
    st.subheader("결과")
    c1, c2, c3 = st.columns(3)
    c1.metric("최고 등급", summary.get("max_level_ko", "-"))
    c2.metric("위험 프레임", summary.get("critical_frames", 0))
    c3.metric("경고 이상 사건", summary.get("event_count", 0))
    video = summary.get("video")
    if video and Path(video).exists():
        st.video(video)
        st.caption("인식이 끝난 영상과 같은 비율입니다. 겹침 히트맵은 재생 위치를 따라 변합니다.")
    _show_path(summary.get("heatmap_overlap"), "겹침 히트맵")
    rows = summary.get("event_rows") or []
    if rows:
        st.dataframe(rows, width="stretch")
    else:
        st.write("경고 이상으로 올라간 사건이 없습니다.")
    files = [
        ("annotated.mp4", summary.get("video"), "인식 영상 받기"),
        ("events.csv", summary.get("events"), "사건 목록 받기"),
        ("heatmap_overlap.png", summary.get("heatmap_overlap"), "겹침 히트맵 받기"),
    ]
    columns = st.columns(len(files))
    for column, (name, path, label) in zip(columns, files):
        if path and Path(path).exists():
            column.download_button(label, Path(path).read_bytes(), file_name=name, key=f"dl-{name}-{path}")


def _show_path(path, caption: str) -> None:
    if path and Path(path).exists():
        _show_bgr(_imread(path), caption)


def _imread(path):
    raw = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def _show_bgr(image, caption: str | None = None) -> None:
    _put_bgr(st, image, caption)


def _put_bgr(slot, image, caption: str | None = None) -> None:
    if image is None:
        return
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    try:
        slot.image(rgb, caption=caption, width="stretch")
    except TypeError:
        slot.image(rgb, caption=caption, use_container_width=True)


if __name__ == "__main__":
    main()
else:
    main()
