from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import cv2

from sitewatch.config import load_config
from sitewatch.geometry import calibration_homography, calibration_scale, load_calibration, save_calibration
from sitewatch.media import Capture, first_frame
from sitewatch.pipeline import Pipeline, analyze_source
from sitewatch.text import draw_texts
from sitewatch.visualize import draw_calibration_preview
from sitewatch.youtube import prepare_source


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="건설현장 히트맵·위험판단")
    sub = parser.add_subparsers(dest="command", required=True)

    video = sub.add_parser("video", help="영상 파일을 분석합니다.")
    video.add_argument("--source", required=True)
    video.add_argument("--calibration", default=None)
    video.add_argument("--output", default="runs/video")
    video.add_argument("--config", default=None)
    video.add_argument("--sound", action="store_true")

    live = sub.add_parser("live", help="유튜브 주소를 실시간으로 분석합니다.")
    live.add_argument("--source", required=True, help="유튜브 주소")
    live.add_argument("--calibration", default=None)
    live.add_argument("--config", default=None)
    live.add_argument("--no-sound", action="store_true")

    calibrate = sub.add_parser("calibrate", help="바닥 점을 찍어 보정 파일을 저장합니다.")
    calibrate.add_argument("--source", required=True)
    calibrate.add_argument("--output", default="runs/calibration.json")

    args = parser.parse_args(argv)
    if args.command == "video":
        return _video(args)
    if args.command == "live":
        return _live(args)
    if args.command == "calibrate":
        return _calibrate(args)
    parser.error("명령을 알 수 없습니다.")
    return 2


def _config(args):
    return load_config(args.config)


def _calibration(path):
    if not path:
        return None
    return load_calibration(path)


def _print_summary(summary: dict) -> None:
    print(
        f"프레임 {summary['frames']}  최고 {summary['max_level_ko']}  "
        f"위험 프레임 {summary['critical_frames']}  사건 {summary['event_count']}"
    )
    print(f"결과 폴더: {Path(summary['summary']).parent}")


def _video(args) -> int:
    config = _config(args)
    calibration = _calibration(args.calibration)
    if calibration is None:
        print(config["risk"]["uncalibrated_notice"])

    def progress(index, total):
        if total:
            print(f"\r{index}/{total}", end="", flush=True)
        elif index % 30 == 0:
            print(f"\r{index}", end="", flush=True)

    summary = analyze_source(
        args.source,
        args.output,
        config,
        calibration=calibration,
        realtime=False,
        sound=args.sound,
        progress=progress,
    )
    print()
    _print_summary(summary)
    return 0


def _live(args) -> int:
    import time

    config = _config(args)
    calibration = _calibration(args.calibration)
    try:
        source, info = prepare_source(args.source)
    except RuntimeError as exc:
        print(exc)
        return 1
    if info:
        kind = "실시간 방송" if info.get("is_live") else "영상"
        print(f"{info['title']} ({kind})")
    pipeline = Pipeline(
        config,
        calibration=calibration,
        realtime=True,
        sound=not args.no_sound,
        overlay_heatmap=True,
    )
    if calibration is None:
        print(config["risk"]["uncalibrated_notice"])
    print("모델을 준비합니다. 처음 실행이면 가중치를 받습니다.")
    detector = pipeline.ensure_detector()
    print(f"장치 {detector.device}  모델 {detector.weights}")
    print("q 종료, h 히트맵 끄기/켜기")
    started = time.perf_counter()
    try:
        capture_cm = Capture(source)
    except RuntimeError as exc:
        print(exc)
        return 1
    with capture_cm as capture:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                print("프레임을 읽지 못했습니다.")
                break
            result = pipeline.process(frame, time.perf_counter() - started)
            view = draw_texts(result.frame, [(12, result.frame.shape[0] - 28, "q 종료   h 히트맵", (240, 240, 240), 18)])
            cv2.imshow("현장 분석", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("h"):
                pipeline.overlay_heatmap = not pipeline.overlay_heatmap
    cv2.destroyAllWindows()
    return 0


def _calibrate(args) -> int:
    try:
        source, _info = prepare_source(args.source)
        frame = first_frame(source)
    except RuntimeError as exc:
        print(exc)
        return 1
    points: list[list[int]] = []
    state = {"mode": "homography"}

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            limit = 4 if state["mode"] == "homography" else 2
            if len(points) < limit:
                points.append([x, y])

    window = "바닥 보정"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        canvas = draw_calibration_preview(frame, points, state["mode"])
        if state["mode"] == "homography":
            hint = "좌상, 우상, 우하, 좌하 순서로 클릭. s 저장  r 다시  t 두 점  q 취소"
        else:
            hint = "바닥의 두 점을 클릭. s 저장  r 다시  t 네 점  q 취소"
        canvas = draw_texts(canvas, [(12, 12, hint, (255, 255, 255), 20)])
        cv2.imshow(window, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            print("보정을 취소했습니다.")
            return 1
        if key == ord("r"):
            points.clear()
        if key == ord("t"):
            state["mode"] = "scale" if state["mode"] == "homography" else "homography"
            points.clear()
        if key == ord("s"):
            needed = 4 if state["mode"] == "homography" else 2
            if len(points) >= needed:
                break
    cv2.destroyAllWindows()
    if state["mode"] == "homography":
        width = float(input("바닥 가로(m): ").strip())
        height = float(input("바닥 세로(m): ").strip())
        calibration = calibration_homography(points, width, height)
    else:
        distance = float(input("두 점 사이 실거리(m): ").strip())
        calibration = calibration_scale(points[0], points[1], distance)
        print("두 점 배율은 화면 전체의 원근을 없애지 못합니다. 가장자리 거리가 어긋날 수 있습니다.")
    save_calibration(calibration, args.output)
    print(f"저장했습니다: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
